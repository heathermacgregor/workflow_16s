# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
from biom import load_table
from biom.table import Table

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level, _ProcessingMixin
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence, table_to_df, 
    update_table_and_meta, to_biom
)
from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.utils.io import (
    export_h5py, import_merged_metadata_tsv, import_merged_table_biom
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.utils.nfc_facilities import load_nfc_facilities, match_facilities_to_samples

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

def import_metadata_tsv(
    tsv_path: Union[str, Path],
    column_renames: Optional[List[Tuple[str, str]]] = []
) -> pd.DataFrame:
    """Load and standardize a sample metadata TSV file.
    
    Args:
        tsv_path:       Path to metadata TSV file.
        column_renames: List of (old_name, new_name) tuples for column renaming.
    
    Returns:
        Standardized metadata DataFrame.
    
    Raises:
        FileNotFoundError: If specified path doesn't exist.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {tsv_path}")

    df = pd.read_csv(tsv_path, sep='\t')
  
    df.columns = df.columns.str.lower()

    sample_id_col = next((col for col in ['run_accession', '#sampleid', 'sample-id'] if col in df.columns), None)
    df['SAMPLE ID'] = (df[sample_id_col] if sample_id_col else [f"{tsv_path.parents[5].name}_x{i}" for i in range(1, len(df)+1)])

    dataset_id_col = next((col for col in ['project_accession', 'dataset_id', 'dataset_name'] if col in df.columns), None)
    df['DATASET ID'] = (df[dataset_id_col] if dataset_id_col else tsv_path.parents[5].name)
  
    for col in constants.DEFAULT_GROUP_COLUMNS:
        if col['type'] == 'bool' and col not in df.columns:
            df[col] = False
          
    for old, new in column_renames:
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df

def import_merged_metadata_tsv(
    tsv_paths: List[Union[str, Path]],
    column_renames: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """Merge multiple metadata files into a single DataFrame.
    
    Args:
        tsv_paths:      List of paths to metadata TSV files.
        column_renames: List of (old_name, new_name) tuples for column renaming.
        verbose:        Verbosity flag.
    
    Returns:
        Concatenated metadata DataFrame.
    
    Raises:
        FileNotFoundError: If no valid metadata files could be loaded.
    """
    dfs: List[pd.DataFrame] = []
    with get_progress_bar() as progress:
        task_desc = "Loading metadata files..."
        task = progress.add_task(_format_task_desc(task_desc), total=len(tsv_paths))
        for tsv_path in tsv_paths:
            try:
                dfs.append(import_metadata_tsv(tsv_path, column_renames))
            except Exception as e:
                logger.error(f"Loading metadata failed for {tsv_path}: {e!r}")
            finally:
                progress.update(task, advance=1)

    if not dfs:
        raise FileNotFoundError("No valid metadata files loaded. Check paths and file formats.")

    return pd.concat(dfs, ignore_index=True)
  
########################################################################################

def import_table_biom(
    biom_path: Union[str, Path], 
    as_type: str = 'table'
) -> Union[Table, pd.DataFrame]:
    """Load a BIOM table from file.
    
    Args:
        biom_path: Path to .biom file.
        as_type:   Output format ('table' or 'dataframe').
    
    Returns:
        BIOM Table object or pandas DataFrame.
    
    Raises:
        ValueError: For invalid 'as_type' values.
    """
    try:
        with h5py.File(biom_path) as f:
            table = Table.from_hdf5(f)
    except:
        table = load_table(biom_path)
        
    if as_type == 'table':
        return table
    elif as_type == 'dataframe':
        return table_to_df(table)
    else:
        raise ValueError(
            f"Invalid output type: {as_type}. Use 'table' or 'dataframe'"
        )


def import_merged_table_biom(
    biom_paths: List[Union[str, Path]], 
    as_type: str = 'table',
    verbose: bool = False
) -> Union[Table, pd.DataFrame]:
    """Merge multiple BIOM tables into a single unified table.
    
    Args:
        biom_paths: List of paths to .biom files.
        as_type:    Output format ('table' or 'dataframe').
        verbose:    Verbosity flag.
    
    Returns:
        Merged BIOM Table or DataFrame.
    
    Raises:
        ValueError: If no valid tables are loaded.
    """
    tables: List[Table] = []
    with get_progress_bar() as progress:
        task_desc = "Loading feature tables"
        task = progress.add_task(_format_task_desc(task_desc), total=len(biom_paths))
        for path in biom_paths:
            try:
                tables.append(import_table_biom(path, 'table'))
            except Exception as e:
                logger.error(f"BIOM load failed for {path}: {e}")
            finally:
                progress.update(task, advance=1)

    if not tables:
        raise ValueError("No valid BIOM tables loaded")

    # ACTUALLY MERGE THE TABLES USING REDUCE
    merged_table = reduce(lambda t1, t2: t1.merge(t2), tables)
    
    return merged_table if as_type == 'table' else table_to_df(merged_table)

########################################################################################

def update_table_and_metadata(
    table: Table,
    metadata: pd.DataFrame,
    sample_col: str = constants.DEFAULT_META_ID_COLUMN
) -> Tuple[Table, pd.DataFrame]:
    """Align BIOM table with metadata using sample IDs.
    
    Args:
        table:         BIOM feature table.
        metadata:      Sample metadata DataFrame.
        sample_column: Metadata column containing sample IDs.
    
    Returns:
        Tuple of (filtered BIOM table, filtered metadata DataFrame)
    
    Raises:
        ValueError: For duplicate lowercase sample IDs in BIOM table.
    """
    norm_metadata = _normalize_metadata(metadata, sample_col)
    biom_mapping = _create_biom_id_mapping(table)
    
    shared_ids = [sid for sid in norm_metadata[sample_col] if sid in biom_mapping]
    filtered_metadata = norm_meta[norm_meta[sample_col].isin(shared_ids)]
    original_ids = [biom_mapping[sid] for sid in filtered_metadata[sample_col]]
    filtered_table = table.filter(original_ids, axis='sample', inplace=False)
    
    return filtered_table, filtered_meta


def _normalize_metadata(metadata: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    """Normalize sample IDs and remove duplicates."""
    metadata[sample_col] = metadata[sample_col].astype(str).str.lower()
    return metadata.drop_duplicates(subset=[sample_col])


def _create_biom_id_mapping(table: Table) -> Dict[str, str]:
    """Create lowercase to original-case ID mapping for BIOM table samples."""
    mapping: Dict[str, str] = {}
    for orig_id in table.ids(axis='sample'):
        lower_id = orig_id.lower()
        if lower_id in mapping:
            raise ValueError(
                f"Duplicate lowercase sample ID: '{lower_id}' "
                f"(from '{orig_id}' and '{mapping[lower_id]}')"
            )
        mapping[lower_id] = orig_id
    return mapping
  
########################################################################################

class DownstreamDataLoader:
    ModeConfig = {
        "asv": ("asv", "table", "asv"), 
        "genus": ("genus", "table_6", "l6")
    }
    def __init__(
        self,
        config: Dict,
        mode: str,
        project_dir: SubDirs,
        existing_subsets: Any = None,
        verbose: bool = False
    ):
        self.config = config
        self.project_dir = project_dir
        self.existing_subsets = existing_subsets
        self.verbose = verbose
        
        self.tables: Dict = {}
        self.metadata: Dict = {}

        self._load_nfc_facilities()

        # Load ASV feature table if appropriate
        if not self.config["target_subfragment_mode"] == 'any':
            table_level, table_dir, _ = self.ModeConfig['asv']
            self.tables[table_level], self.metadata[table_level] = {}, {}
            self.tables[table_level]['raw'], self.metadata[table_level]['raw'] = self._filter_and_align(
                self._load_table_biom(table_level, table_dir), 
                self._match_facilities_to_samples(self._load_metadata_df(table_level, table_dir))
            )
        
        # Load taxonomically-assigned feature table at genus level     
        table_level, table_dir, _ = self.ModeConfig['genus']
        self.tables[table_level], self.metadata[table_level] = {}, {}
        self.tables[table_level]['raw'], self.metadata[table_level]['raw'] = self._filter_and_align(
            self._load_table_biom(table_level, table_dir), 
            self._match_facilities_to_samples(self._load_metadata_df(table_level, table_dir))
        )
        #self._log_results()

    def _load_table_biom(self, table_level, table_dir) -> None:
        table_biom_paths = self._find_table_biom_paths(table_level, table_dir)  
        if not table_biom_paths:
            raise FileNotFoundError("No BIOM files found")
        self.tables[table_level]['raw'] = import_merged_table_biom(table_biom_paths, "table", self.verbose)

    def _find_table_biom_paths(self, table_level, table_dir) -> List[Path]:
        if self.existing_subsets == None:
            # Use wildcard for subfragment when in 'any' mode or not in 'genus' mode
            subfragment_part = (
              "*" if self.config["target_subfragment_mode"] == 'any' or table_level == 'asv'
              else self.config["target_subfragment_mode"]
            )
            pattern = "/".join([
                "*", "*", "*", subfragment_part, 
                "FWD_*_REV_*", table_dir, "feature-table.biom"
            ])
            globbed = glob.glob(
                str(Path(self.project_dir.qiime_data_per_dataset) / pattern), 
                recursive=True
            )
            if self.verbose:
                logger.info(f"Found {len(globbed)} feature tables")
            return [Path(p) for p in globbed]
        else:
            table_biom_paths = [paths[table_dir] for subset_id, paths in self.existing_subsets.items()]
            if self.verbose:
                logger.info(f"Found {len(biom_paths)} feature tables")
            return table_biom_paths

    def _load_metadata_df(self, table_level, table_dir) -> None:
        tsv_paths = self._find_metadata_paths(table_level, table_dir)
        metadata = import_merged_metadata_tsv(tsv_paths, None, self.verbose)

        # Remove duplicated columns
        if metadata.columns.duplicated().any():
            duplicated_columns = metadata.columns[metadata.columns.duplicated()].tolist()
            logger.debug(
                f"Found duplicate columns in metadata: {duplicated_columns}. "
                "Removing duplicates."
            )
            metadata = metadata.loc[:, ~metadata.columns.duplicated()]
          
        self.metadata = metadata

    def _find_metadata_paths(self, table_level, table_dir) -> List[Path]:
        tsv_paths: List[Path] = []
        if self.existing_subsets == None:
            for biom_path in self._find_table_biom_paths(table_level, table_dir):
                dataset_dir = biom_path.parent if biom_path.is_file() else biom_path
                tail = dataset_dir.parts[-6:-1]
                tsv_path = Path(self.project_dir.metadata_per_dataset).joinpath(
                    *tail, "sample-metadata.tsv"
                )
                if tsv_path.exists():
                    paths.append(tsv_path)
            if self.verbose:
                logger.info(f"Found {len(paths)} metadata files")
            return paths
        else:
            metadata_paths = [paths["metadata"] 
                              for subset_id, paths in self.existing_subsets.items()]
            if self.verbose:
                (f"Found {len(metadata_paths)} metadata files")
            return metadata_paths
          
    def _load_nfc_facilities(self) -> None:
        # If enabled, find samples within a threshold distance from NFC facilities
        if self.config.get("nfc_facilities", {}).get("enabled", False):
            self.nfc_facilities = load_nfc_facilities(
                cfg=self.config,
                output_dir=self.project_dir.final
            )
        else:
            self.nfc_facilities = None

    def _match_facilities_to_samples(self, metadata) -> None:
        if not self.nfc_facilities:
            return
        else:
            return match_facilities_to_samples(self.config, metadata, self.nfc_facilities)

    def _filter_and_align(self, table, metadata) -> None:
        table, metadata = update_table_and_meta(
            table, metadata, 
            self.config.get("metadata_id_column", constants.DEFAULT_META_ID_COLUMN)
        )
      
    def _log_results(self) -> None:   
        logger.info(
            f"{'Loaded metadata:':<30}{self.meta.shape[0]:>6} samples "
            f"× {self.meta.shape[1]:>5} cols"
        )
      
        feature_type = "genera" if self.mode == "genus" else "ASVs"
        logger.info(
            f"{'Loaded features:':<30}{self.table.shape[1]:>6} samples "
            f"× {self.table.shape[0]:>5} {feature_type}"
        )

    
