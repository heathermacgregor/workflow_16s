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
    column_renames: Optional[List[Tuple[str, str]]] = None
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
        col_name = col.get('name')
        if col.get('type') == 'bool' and col_name and col_name not in df.columns:
            df[col_name] = False

    if column_renames is None:
        column_renames = []      
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
        task_desc = "Loading metadata files"
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
        existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None,
        verbose: bool = False
    ):
        self.config = config
        self.mode = mode
        self.project_dir = project_dir
        self.existing_subsets = existing_subsets
        self.verbose = verbose

        self.tables: Dict[str, Dict[str, Table]] = {}
        self.metadata: Dict[str, Dict[str, pd.DataFrame]] = {}

        self.nfc_facilities = self._load_nfc_facilities()

        for level in self.ModeConfig:
            self._load_data_level(*self.ModeConfig[level])

    def _load_data_level(self, level: str, table_dir: str, _tax_level: str):
        self.tables[level], self.metadata[level] = {}, {}

        table = self._load_table_biom(level, table_dir)
        metadata = self._load_metadata_df(level, table_dir)
        metadata = self._match_facilities_to_samples(metadata)

        table, metadata = self._filter_and_align(table, metadata)

        self.tables[level]["raw"] = table
        self.metadata[level]["raw"] = metadata

    def _load_table_biom(self, table_level: str, table_dir: str) -> Table:
        biom_paths = self._find_table_biom_paths(table_level, table_dir)
        if not biom_paths:
            raise FileNotFoundError(f"No BIOM files found for {table_level} in {table_dir}")
        return import_merged_table_biom(biom_paths, as_type="table", verbose=self.verbose)

    def _load_metadata_df(self, table_level: str, table_dir: str) -> pd.DataFrame:
        tsv_paths = self._find_metadata_paths(table_level, table_dir)
        metadata = import_merged_metadata_tsv(tsv_paths)

        if metadata.columns.duplicated().any():
            dupes = metadata.columns[metadata.columns.duplicated()].tolist()
            logger.warning(f"Duplicate columns found in metadata: {dupes}. Removing duplicates.")
            metadata = metadata.loc[:, ~metadata.columns.duplicated()]

        return metadata

    def _find_table_biom_paths(self, table_level: str, table_dir: str) -> List[Path]:
        if self.existing_subsets:
            return [paths[table_dir] for paths in self.existing_subsets.values()]

        subfragment = (
            "*" if self.config["target_subfragment_mode"] == "any" or table_level == "asv"
            else self.config["target_subfragment_mode"]
        )

        pattern = Path(self.project_dir.qiime_data_per_dataset) / "*/*/*" / subfragment / "FWD_*_REV_*" / table_dir / "feature-table.biom"
        paths = glob.glob(str(pattern), recursive=True)

        if self.verbose:
            logger.info(f"[{table_level}] Found {len(paths)} BIOM files")

        return [Path(p) for p in paths]

    def _find_metadata_paths(self, table_level: str, table_dir: str) -> List[Path]:
        if self.existing_subsets:
            return [paths["metadata"] for paths in self.existing_subsets.values()]

        tsv_paths: List[Path] = []
        for biom_path in self._find_table_biom_paths(table_level, table_dir):
            dataset_parts = biom_path.parts[-6:-1]
            tsv_path = Path(self.project_dir.metadata_per_dataset).joinpath(*dataset_parts, "sample-metadata.tsv")
            if tsv_path.exists():
                tsv_paths.append(tsv_path)

        if self.verbose:
            logger.info(f"[{table_level}] Found {len(tsv_paths)} metadata files")

        return tsv_paths

    def _load_nfc_facilities(self) -> Optional[pd.DataFrame]:
        if self.config.get("nfc_facilities", {}).get("enabled", False):
            return load_nfc_facilities(self.config, output_dir=self.project_dir.final)
        return None

    def _match_facilities_to_samples(self, metadata: pd.DataFrame) -> pd.DataFrame:
        if self.nfc_facilities is not None:
            return match_facilities_to_samples(self.config, metadata, self.nfc_facilities)
        return metadata

    def _filter_and_align(self, table: Table, metadata: pd.DataFrame) -> Tuple[Table, pd.DataFrame]:
        return update_table_and_meta(
            table,
            metadata,
            sample_col=self.config.get("metadata_id_column", constants.DEFAULT_META_ID_COLUMN)
        )

    def log_summary(self) -> None:
        """Optional logging method to summarize data loaded"""
        for level in self.tables:
            table = self.tables[level]["raw"]
            meta = self.metadata[level]["raw"]
            logger.info(
                f"[{level}] {table.shape[0]} features × {table.shape[1]} samples — "
                f"{meta.shape[0]} metadata samples × {meta.shape[1]} columns"
            )

    
