# Standard Imports
import glob
import logging
from typing import Tuple

# Third Party Imports
import pandas as pd
from biom.table import Table

# Local Imports
from workflow_16s.constants import MODE, SAMPLE_ID_COLUMN
from workflow_16s.nuclear_fuel_cycle.nuclear_fuel_cycle import update_nfc_facilities_data
from workflow_16s.utils.biom import import_merged_biom_table, export_h5py, sample_id_map
from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.utils.metadata import MetadataCleaner, import_merged_metadata_tsv


def align_table_and_metadata(
    table: Table,
    metadata: pd.DataFrame,
    sample_id_col: str = SAMPLE_ID_COLUMN
) -> Tuple[Table, pd.DataFrame]:
    """Align BIOM table with metadata using sample IDs.
    
    Args:
        table :         
            BIOM feature table.
        metadata :      
            Sample metadata DataFrame.
        sample_id_col : 
            Metadata column containing sample IDs.
    
    Returns:
        Tuple of (filtered BIOM table, filtered metadata DataFrame)
    
    Raises:
        ValueError: For duplicate lowercase sample IDs in BIOM table.
    """
    # Handle empty metadata
    if metadata.empty:
        return Table(np.array([]), [], []), pd.DataFrame(columns=[sample_id_col])
    
    biom_mapping = sample_id_map(table)
    shared_ids = [id for id in metadata[sample_id_col] if id in biom_mapping]
    
    # Handle no shared IDs
    if not shared_ids:
        return Table(np.array([]), [], []), pd.DataFrame(columns=[sample_id_col])
    
    filtered_metadata = metadata[metadata[sample_id_col].isin(shared_ids)]
    original_ids = [biom_mapping[id] for id in filtered_metadata[sample_id_col]]
    filtered_table = table.filter(original_ids, axis='sample', inplace=False)
    return filtered_table, filtered_metadata
  

class DownstreamDataLoader:
    ModeConfig = {
        "asv": ("asv", "table", "asv"), 
        "genus": ("genus", "table_6", "l6")
    }
    def __init__(
        self,
        config: Dict,
        project_dir: SubDirs,
        existing_subsets: Any = None
    ):
        self.config = config
        self.target_subfragment_mode = self.config.get("target_subfragment_mode", MODE)
        self.metadata_id_column = self.config.get("metadata_id_column", SAMPLE_ID_COLUMN)
      
        self.verbose = self.config("verbose", False)
      
        self.project_dir = project_dir
        self.existing_subsets = existing_subsets
        self._load_nfc_facilities()

        # Initialize storage for feature tables and metadata
        self.tables: Dict = {'raw': {}}
        self.metadata: Dict = {'raw': {}}
      
        self.table_paths = None

        self.nfc_facilities = None

    def run(self):
        # Load the ASV feature table if the target subfragment is specified (so, not 'any')
        self._load_table_and_metadata('asv') if not self.target_subfragment_mode == 'any'
        # Load the taxonomically-assigned feature table at the genus level     
        self._load_table_and_metadata('genus')
      
    def _load_table_and_metadata(self, mode: str = 'genus') -> None:
        level, subdir, _ = self.ModeConfig[mode]
        table = self._load_biom_table(level, subdir)
        metadata = self._load_metadata(level, subdir)
        metadata = self._clean_metadata(metadata)
      
        # If enabled, find samples within a threshold distance from NFC facilities
        if self.config.get("nfc_facilities", {}).get("enabled", False):
            self.nfc_facilities, metadata = self._load_nfc_facilities(metadata)
          
        table, metadata = self._filter_and_align(table, metadata)
        self._log_results(level, table, metadata)
        self.tables['raw'][level], self.metadata['raw'][level] = table, metadata

    def _filter_and_align(self, table, metadata) -> Tuple:
        table, metadata = align_table_and_metadata(
            table, metadata, self.metadata_id_column
        )
        if table.is_empty() or metadata.empty:
            logger.warning(f"Alignment resulted in empty table for level '{level}'")
        return table, metadata
        
    # BIOM FEATURE TABLE    
    def _load_biom_table(self, level, subdir) -> Table:
        table_paths = self._get_table_paths(level, subdir)  
        if not table_paths:
            raise FileNotFoundError("No BIOM table filepaths found")
        return import_merged_biom_table(biom_paths=table_paths)

    def _load_metadata(self, level, table_dir) -> pd.DataFrame:
        metadata_paths = self._find_metadata_paths(level, subdir)
        if not metadata_paths:
            raise FileNotFoundError("No metadata TSV filepaths found")
        metadata = import_merged_metadata_tsv(
            tsv_paths=metadata_paths, 
            columns_to_rename=self.config.get("columns_to_rename", None)
        )
        
        return metadata

    def _clean_metadata(self, metadata: pd.DataFrame):
        cleaner = MetadataCleaner(
            config=self.config, 
            metadata=metadata
        )
        cleaner.run_all()
        return cleaner.df

    def _get_table_paths(self, level: str, subdir: str) -> List[Path]:
        # If there are existing subsets of datasets from upstream processing loaded
        if self.existing_subsets is not None:
            table_paths = [paths[dir] for subset_id, paths in self.existing_subsets.items()]
            
        # If there are NOT existing subsets, search in the directory files matching a pattern
        else:
            if self.config["target_subfragment_mode"] == 'any':
                subfragment = "*"
            else:
                subfragment = self.config["target_subfragment_mode"]
            qiime_data_dir = Path(self.project_dir.qiime_data_per_dataset)   
            pattern = "/".join([
                str(qiime_data_dir), "*", "*", "*", subfragment, 
                "FWD_*_REV_*", subdir, "feature-table.biom"
            ])
            table_paths = glob.glob(pattern, recursive=True)
        table_paths = [Path(p) for p in table_paths]
        if self.verbose:
            n = len(table_paths)
            logger.info(f"Found {n} feature tables")
        self.table_paths = table_paths 
        return table_paths 

    def _get_metadata_paths(self, level, subdir) -> List[Path]:
        tsv_paths: List[Path] = []
        # If there are existing subsets of datasets from upstream processing loaded
        if self.existing_subsets is not None:
            tsv_paths = [paths["metadata"] for subset_id, paths in self.existing_subsets.items()]
        # If there are NOT existing subsets, find metadata files corresponding to each biom table file
        else:
            table_paths = self.table_paths if self.table_paths is not None else self._get_table_paths(level, subdir)
            for table_path in table_paths:
                dataset_dir = table_path.parent if table_path.is_file() else table_path
                tail = dataset_dir.parts[-6:-1]
                metadata_dir = Path(self.project_dir.metadata_per_dataset)
                tsv_path = metadata_dir.joinpath(*tail, "sample-metadata.tsv")
                if metadata_path.exists():
                    tsv_paths.append(tsv_path)
                  
        if self.verbose:
            (f"Found {len(tsv_paths)} metadata files")
        return tsv_paths

    # SPECIAL CASE: LOAD NFC FACILITIES DATA
    def _load_nfc_facilities(self, metadata: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return update_nfc_facilities_data(
            config=self.config,
            metadata=metadata
        )

    def _log_results(self, level, table, metadata) -> None:
        table_size = "Empty" if table.is_empty() else f"{table.shape[0]} features × {table.shape[1]} samples"
        metadata_size = "Empty" if metadata.empty else f"{metadata.shape[0]} samples × {metadata.shape[1]} cols"
        feature_type = "genera" if level == "genus" else "ASVs"
        logger.info(f"{'Loaded metadata:':<30}{metadata_size}")
        logger.info(f"{'Loaded features:':<30}{table_size} {feature_type}")


# API
def load_data(config: Dict, project_dir: SubDirs, existing_subsets: Any = None):
    loader = DownstreamDataLoader(
        config=config,
        project_dir=project_dir,
        existing_subsets=existing_subsets
    )
    loader.run()
    return loader
