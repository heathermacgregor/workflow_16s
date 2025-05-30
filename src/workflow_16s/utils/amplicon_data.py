# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import reduce

# Third-Party Imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from biom import load_table
from biom.table import Table
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
    TaskID
)

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.biom import (
    collapse_taxa, 
    convert_to_biom, 
    export_h5py, 
    presence_absence, 
    filter_presence_absence
)
from workflow_16s.utils.progress import create_progress
from workflow_16s.utils.file_utils import (
    import_merged_table_biom,
    import_merged_meta_tsv
)
from workflow_16s.utils import df_utils
from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.stats import beta_diversity 
from workflow_16s.stats.utils import (
    clr_transform_table,
    filter_table, 
    merge_table_with_metadata,
    normalize_table,
    table_to_dataframe
)
from workflow_16s.stats.tests import kruskal_bonferroni, mwu_bonferroni, ttest 
from workflow_16s.figures.html_report import HTMLReport
from workflow_16s.figures.merged.merged import (
    mds, pca, pcoa, sample_map_categorical
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ================================= DEFAULT VALUES =================================== #

DEFAULT_PROGRESS_TEXT_N = 50

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# ==================================== FUNCTIONS ===================================== #    

def filter_and_reorder_biom_and_metadata(
    table: Table,
    metadata_df: pd.DataFrame,
    sample_column: str = '#sampleid'
) -> tuple[Table, pd.DataFrame]:
    """
    Filters and reorders a BIOM table and metadata DataFrame so that their sample IDs match exactly and are in the same order.

    Args:
        table (biom.Table): The BIOM table (features x samples).
        metadata_df (pd.DataFrame): Metadata DataFrame with a sample ID column.
        sample_column (str): Name of the column in metadata_df containing sample IDs.

    Returns:
        tuple: (filtered and reordered biom.Table, filtered and reordered metadata DataFrame)
    """
    # Normalize sample IDs in metadata
    metadata_df = metadata_df.copy()
    metadata_df[sample_column] = metadata_df[sample_column].str.lower()

    # Get mapping from lowercase sample ID to original BIOM sample ID
    biom_sample_ids = table.ids(axis='sample')
    lowercase_to_original = {sid.lower(): sid for sid in biom_sample_ids}

    # Deduplicate metadata sample IDs (preserve order)
    seen = set()
    ordered_lowercase_ids = []
    for sid in metadata_df[sample_column]:
        if sid in lowercase_to_original and sid not in seen:
            seen.add(sid)
            ordered_lowercase_ids.append(sid)

    # Map back to original-case sample IDs for BIOM
    ordered_biom_sample_ids = [lowercase_to_original[sid] for sid in ordered_lowercase_ids]

    # Filter BIOM table
    table_filtered = table.filter(ordered_biom_sample_ids, axis='sample', inplace=False)

    # Reorder BIOM data matrix
    data = table_filtered.matrix_data.toarray()
    obs_ids = table_filtered.ids(axis='observation')
    current_sample_ids = table_filtered.ids(axis='sample')
    sample_index = {sid: i for i, sid in enumerate(current_sample_ids)}
    ordered_indices = [sample_index[sid] for sid in ordered_biom_sample_ids]
    reordered_data = data[:, ordered_indices]
    table_reordered = Table(reordered_data, observation_ids=obs_ids, sample_ids=ordered_biom_sample_ids)

    # Filter and reorder metadata
    metadata_filtered = metadata_df.set_index(sample_column).loc[ordered_lowercase_ids].reset_index()

    return table_reordered, metadata_filtered


class AmpliconData:
    """
    A class for processing and analyzing amplicon sequencing data.
    
    This class handles data loading, processing at various taxonomic levels,
    statistical analysis, and visualization based on configuration settings.
    
    Attributes:
        cfg:         Configuration settings
        project_dir: Project directory structure
        mode:        Processing mode ('asv' or 'genus')
        verbose:     Verbosity flag
        meta:        Sample metadata
        table:       Feature table
        color_maps:  Color mappings for visualization
        figures:     Generated figures
        tables:      Processed tables at different taxonomic levels
        stats:       Statistical analysis results
    """
    
    def __init__(
        self, 
        cfg: Dict,
        project_dir: Union[str, Path],
        mode: str = 'genus',
        verbose: bool = False
    ):
        """
        Initialize AmpliconData instance.
        
        Args:
            cfg:         Configuration dictionary
            project_dir: Project directory structure
            mode:        Processing mode ('asv' or 'genus')
            verbose:     Verbosity flag
        """
        self.cfg = cfg
        self.project_dir = project_dir
        self.mode = mode
        self.verbose = verbose

        # Initialize data storage attributes
        self.meta = None
        self.table = None
        self.taxa = None
        self.color_maps = {}
        self.figures = {}
        self.tables = {}
        self.stats = {}
        
        # Mode configuration mapping
        self._mode_config = {
            'asv': ('table', 'asv'),
            'genus': ('table_6', 'l6')
        }
        
        # Set output paths
        self._set_output_paths()
        
        # Load data
        self._load_data()
        
        # Execute processing pipeline
        self._execute_processing_pipeline()
        
        # Run statistical analyses
        #self._run_all_statistical_analyses()

    def _set_output_paths(self):
        """Set output paths for tables and metadata based on processing mode."""
        table_dir, output_dir = self._mode_config.get(self.mode, (None, None))
        if table_dir is None:
            raise ValueError(f"Invalid processing mode: {self.mode}")
        
        self.table_output_path = (
            Path(self.project_dir.data) / 'merged' / 'table' / output_dir / 
            'feature-table.biom'
        )
        self.meta_output_path = (
            Path(self.project_dir.data) / 'merged' / 'metadata' / 
            'sample-metadata.tsv'
        )

    def _load_data(self):
        """Load metadata and BIOM table data."""
        self._load_metadata()
        self._load_biom_table()
        original_n_samples = self.table.shape[1]
        self.table, self.meta = filter_and_reorder_biom_and_metadata(table=self.table, metadata_df=self.meta, sample_column='#sampleid')
        
        logger.info(
            f"Loaded (samples x features) metadata table with "
            f"{RED}{self.meta.shape[0]}{RESET} samples "
            f"and {RED}{self.meta.shape[1]}{RESET} columns"
        )
        
        feature_type = 'genera' if self.mode == 'genus' else 'ASVs'
        logger.info(
            f"Loaded (features x samples) feature table with "
            f"{RED}{self.table.shape[1]} ({original_n_samples}){RESET} samples "
            f"and {RED}{self.table.shape[0]}{RESET} {feature_type}"
        )
    

    def _load_metadata(self):
        """Load and merge metadata from multiple sources."""
        meta_paths = self._get_metadata_paths()
        self.meta = import_merged_meta_tsv(
            meta_paths, 
            None,
            self.verbose
        )

    def _load_biom_table(self):
        """Load and merge BIOM feature tables."""
        biom_paths = self._get_biom_paths()
        if not biom_paths:
            error_text = "No BIOM files found matching pattern"
            logger.error(error_text)
            raise FileNotFoundError(error_text)   
            
        self.table = import_merged_table_biom(
            biom_paths, 
            'table',
            self.verbose
        )

    def _get_biom_paths(self) -> List[Path]:
        """
        Get paths to BIOM feature tables using a glob pattern.
        
        Returns:
            List of Path objects to BIOM files
        """
        pattern = '/'.join(
          ['*', '*', '*', '*', 'FWD_*_REV_*', self._mode_config[self.mode][0], 
           'feature-table.biom']
        )
        biom_paths = glob.glob(
            str(Path(self.project_dir.qiime_data_per_dataset) / pattern), 
            recursive=True
        )
        if self.verbose:
            logger.info(f"Found {RED}{len(biom_paths)}{RESET} feature tables")
        return [Path(p) for p in biom_paths]

    def _get_metadata_paths(self) -> List[Path]:
        """Get paths to metadata files corresponding to BIOM tables."""
        meta_paths = []
        for biom_path in self._get_biom_paths():
            # Handle both file paths and directory paths
            dataset_dir = biom_path.parent if biom_path.is_file() else biom_path
            
            # Extract relevant path components
            tail_parts = dataset_dir.parts[-6:-1]
            
            # Construct metadata path
            meta_path = Path(self.project_dir.metadata_per_dataset).joinpath(
                *tail_parts, "sample-metadata.tsv"
            )
            meta_paths.append(meta_path)
            
        if self.verbose:
            logger.info(f"Found {RED}{len(meta_paths)}{RESET} metadata files")
        return meta_paths

    

    def _execute_processing_pipeline(self):
        """Execute the appropriate processing pipeline based on mode."""
        processor = {
            'asv': self._process_asv_mode,
            'genus': self._process_genus_mode
        }.get(self.mode, self._process_asv_mode)
        
        processor()

    def _process_asv_mode(self):
        """Process data in ASV mode (not yet implemented)."""
        logger.info("ASV mode is not yet supported!")

    def _process_genus_mode(self):
        """Process data in genus mode through multiple processing steps."""
        tax_levels = ['phylum', 'class', 'order', 'family', 'genus']
        table_dir = Path(self.project_dir.data) / 'merged' / 'table'
        
        # Apply filtering, normalization, and CLR before collapsing
        self._apply_preprocessing_steps()
        
        # Execute processing steps
        for table_type in self.tables.keys():
            self._collapse_taxa(table_type, tax_levels)
        self._presence_absence(tax_levels)
        print(self.tables)
        
        # Save all generated tables
        #self._save_all_tables(table_dir)

    def _apply_preprocessing_steps(self):
        """Apply filtering, normalization, and CLR transformation to the table before collapsing."""
        # Start with the original table
        table = self.table
        self.tables["raw"] = {}
        self.tables["raw"][self.mode] = table
        
        filtering_enabled = self.cfg['features']['filter']
        normalization_enabled = (
            self.cfg['features']['filter'] and 
            self.cfg['features']['normalize']
        )
        clr_transformation_enabled = (
            self.cfg['features']['filter'] and 
            self.cfg['features']['normalize'] and 
            self.cfg['features']['clr_transform']
        )
        enabled_steps = [filtering_enabled, normalization_enabled, clr_transformation_enabled]
        n_enabled_steps = sum(enabled_steps)

        with create_progress() as progress:
            main_task = progress.add_task(
                f"[white]Preprocessing {self.mode} tables".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=n_enabled_steps
            )
            # Apply filtering if enabled
            if filtering_enabled:
                if self.verbose:
                    logger.info("Applying filtering to table...")
                filtered_table = filter_table(table)
                self.tables["filtered"] = {}
                self.tables["filtered"][self.mode] = filtered_table
                progress.update(main_task, advance=1)
            
            # Apply normalization if enabled (requires prior filtering)
            if normalization_enabled:
                if self.verbose:
                    logger.info("Applying normalization to table...")
                normalized_table = normalize_table(filtered_table, axis=1)
                self.tables["normalized"] = {}
                self.tables["normalized"][self.mode] = normalized_table
                progress.update(main_task, advance=1)
            
            # Apply CLR transformation if enabled (requires prior normalization)
            if clr_transformation_enabled:
                if self.verbose:
                    logger.info("Applying CLR transformation to table...")
                clr_transformed_table = clr_transform_table(normalized_table)
                self.tables["clr_transformed"] = {}
                self.tables["clr_transformed"][self.mode] = clr_transformed_table
                progress.update(main_task, advance=1)

    def _collapse_taxa(self, table_type: str, levels: List[str]):
        """Generate raw tables by collapsing taxa at different levels."""
        process_name = f"Collapsing {table_type} taxonomy"
        log_template_0 = f"Collapsed {table_type} to"
        self.tables[table_type] = self._run_processing_step(
            process_name=process_name,
            process_func=collapse_taxa,
            levels=levels,
            func_args=(),
            get_source=lambda _: self.table,
            log_template=log_template_0 + " {level} level"
        )

    def _presence_absence(self, levels: List[str]):
        """Generate presence/absence tables if enabled in config."""
        if not self.cfg['features']['presence_absence']:
            return
            
        self.tables["presence_absence"] = self._run_processing_step(
            process_name="Converting to presence/absence",
            process_func=presence_absence,
            levels=levels,
            func_args=(),
            get_source=lambda level: self.tables["raw"][level]
        )

    def _run_processing_step(
        self,
        process_name: str,
        process_func: Callable,
        levels: List[str],
        func_args: tuple,
        get_source: Callable,
        log_template: Optional[str] = None,
        log_action: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute a processing step across multiple taxonomic levels.
        
        Args:
            process_name: Name of the processing step for logging
            process_func: Function to execute for each level
            levels:       Taxonomic levels to process
            func_args:    Additional arguments for process_func
            get_source:   Function to get input table for a level
            log_template: Template for logging messages
            log_action:   Action name for simple logging
            
        Returns:
            Dictionary of processed tables keyed by taxonomic level
        """
        processed_tables = {}
        
        if self.verbose:
            # Verbose mode: Use logging
            logger.info(f"{process_name}...")
            for level in levels:
                source_table = get_source(level)
                try:
                    processed = process_func(source_table, *func_args)
                except:
                    processed = process_func(source_table, level, *func_args)
                processed_tables[level] = processed
                self._log_level_action(level, log_template, log_action)
        else:
            # Non-verbose mode: Use progress bars
            with create_progress() as progress:
                task = progress.add_task(
                    f"[white]{process_name}...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                    total=len(levels)
                )
                for level in levels:
                    source_table = get_source(level)
                    try:
                        processed = process_func(source_table, *func_args)
                    except:
                        processed = process_func(source_table, level, *func_args)
                    processed_tables[level] = processed
                    progress.update(task, advance=1)
                    
        return processed_tables

    def _log_level_action(self, level: str, template: Optional[str] = None, 
                         action: Optional[str] = None):
        """
        Log action for a specific taxonomic level.
        
        Args:
            level:    Taxonomic level being processed
            template: String template for logging (uses {level} placeholder)
            action:   Simple action description
        """
        if template:
            logger.info(template.format(level=level))
        elif action:
            logger.info(f"{level} {action}")
            
