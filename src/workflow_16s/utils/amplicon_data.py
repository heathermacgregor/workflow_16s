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
from workflow_16s.stats.tests import fisher_exact_bonferroni, kruskal_bonferroni, mwu_bonferroni, ttest 
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
    Filters and reorders a BIOM table and metadata DataFrame so that sample IDs 
    match exactly and are aligned.

    Args:
        table:         The BIOM table (features x samples).
        metadata_df:   Metadata DataFrame with a sample ID column.
        sample_column: Column name in metadata_df with sample IDs.

    Returns:
        Tuple containing the filtered table and metadata_df.
    """
    # Lowercase and deduplicate sample IDs in metadata
    metadata_df = metadata_df.copy()
    metadata_df[sample_column] = metadata_df[sample_column].astype(str).str.lower()
    metadata_df = metadata_df.drop_duplicates(subset=[sample_column])

    # Lowercase biom sample IDs, mapping to original IDs
    original_sample_ids = table.ids(axis='sample')
    lowercase_to_original = {}
    for sid in original_sample_ids:
        key = sid.lower()
        if key in lowercase_to_original:
            raise ValueError(
                f"Duplicate lowercase sample ID found in BIOM table: "
                f"'{key}' from '{sid}' and '{lowercase_to_original[key]}'"
            )
        lowercase_to_original[key] = sid

    # Get shared sample IDs (in lowercase)
    metadata_ids = metadata_df[sample_column].tolist()
    shared_ids = [sid for sid in metadata_ids if sid in lowercase_to_original]

    # Filter metadata by shared IDs
    metadata_df = metadata_df.set_index(sample_column).loc[shared_ids].reset_index()

    # Map back to original-case sample IDs for biom
    ordered_biom_sample_ids = [lowercase_to_original[sid] 
                               for sid in metadata_df[sample_column]]

    # Filter and reorder BIOM table
    table_filtered = table.filter(
        ordered_biom_sample_ids, axis='sample', inplace=False
    )
    sample_index = {sid: i 
                    for i, sid in enumerate(table_filtered.ids(axis='sample'))}
    reordered_indices = [sample_index[sid] 
                         for sid in ordered_biom_sample_ids]
    data = table_filtered.matrix_data.toarray()
    reordered_data = data[:, reordered_indices]

    table_reordered = Table(
        reordered_data,
        observation_ids=table_filtered.ids(axis='observation'),
        sample_ids=ordered_biom_sample_ids
    )

    return table_reordered, metadata_df



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
        self._run_all_statistical_analyses()

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
        self._save_all_tables(table_dir)

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
            
    def _save_all_tables(self, base_dir: Path):
        """
        Save all generated tables to appropriate directories.
        
        Args:
            base_dir: Base directory for table storage
        """
        # Calculate total tables to save
        total_tables = sum(len(level_tables) for level_tables in self.tables.values())
        
        with create_progress() as progress:
            task = progress.add_task(
                "[white]Saving tables...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=total_tables
            )
            base_dir.mkdir(parents=True, exist_ok=True)
            for table_type, level_tables in self.tables.items():
                type_dir = base_dir / table_type
                type_dir.mkdir(parents=True, exist_ok=True)
                for level, table in level_tables.items():
                    level_dir = type_dir / level
                    level_dir.mkdir(parents=True, exist_ok=True)
                    output_path = level_dir / f"feature-table.biom"
                    # Save table
                    export_h5py(table, output_path)
                    if self.verbose:
                        n_features, n_samples = table.shape
                        shape_str = f"[{n_features}, {n_samples}]"
                        logger.info(
                            f"Wrote {table_type} {level} table {shape_str} to '{output_path}'"
                        )
                    # Update progress after each table save
                    progress.update(task, advance=1)

    def _run_all_statistical_analyses(self):
        """Run statistical analyses for all generated table types."""
        for table_type in self.tables:
            self._run_statistical_analyses(table_type)
        self._save_statistical_results(Path(self.project_dir.tables) / 'stats' / 'tests')
        self._top_features()

    def _run_statistical_analyses(self, table_type: str):
        """
        Run configured statistical analyses for a specific table type.
        
        Args:
            table_type: Type of table to analyze ('raw', 'presence_absence', etc.)
        """
        self.stats[table_type] = {}
        tables = self.tables[table_type]
        
        # Get enabled tests from configuration
        enabled_tests = [
            test for test in [
                'ttest', 'mwu_bonferroni', 'kruskal_bonferroni', 'pca', 'tsne'
            ] if self.cfg['stats'][table_type].get(test, False)
        ]
        logger.info(f"Enabled tests for {table_type}: {enabled_tests}")
        
        # Test execution configuration
        test_config = {
            'fisher': {
                'key': 'fisher',
                'func': fisher_exact_bonferroni,
                'name': 'Fisher test (w/ Bonferroni)'
            },
            'ttest': {
                'key': 'ttest',
                'func': ttest,
                'name': 't-test'
            },
            'mwu_bonferroni': {
                'key': 'mwub',
                'func': mwu_bonferroni,
                'name': 'Mann-Whitney U test (w/ Bonferroni)'
            },
            'kruskal_bonferroni': {
                'key': 'kwb',
                'func': kruskal_bonferroni,
                'name': 'Kruskal-Wallis test (w/ Bonferroni)'
            }
        }
        
        # Execute configured tests
        with create_progress() as progress:
            main_task = progress.add_task(
                f"[white]Analyzing {table_type} tables".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=len(enabled_tests)
            )
            
            for test in enabled_tests:
                if test not in test_config:
                    # Handle visualization tests separately
                    if test in ['pca', 'tsne']:
                        self._run_visual_analyses(
                            table_type=table_type,
                            test_type=test,
                            progress=progress,
                            parent_task_id=main_task
                        )
                    continue
                    
                config = test_config[test]
                test_key = config['key']
                test_name = config['name']
                
                # Create progress task for this test
                test_task = progress.add_task(
                    f"[white]{test_name}".ljust(DEFAULT_PROGRESS_TEXT_N), 
                    total=len(tables)
                )
                
                # Run test for all levels
                self.stats[table_type][test_key] = {}
                for level in tables:
                    self.stats[table_type][test_key][level] = config['func'](
                        table=tables[level],
                        metadata=self.meta,
                        group_column='nuclear_contamination_status',
                        group_column_values=[True, False],
                    )
                    progress.update(test_task, advance=1)
                
                # Update main progress
                progress.update(main_task, advance=1)
                progress.remove_task(test_task)

    def _save_statistical_results(self, base_dir: Path):
        """
        Save statistical analysis results to CSV files.
            
        Results are organized in the directory structure:
            {output_dir}/{table_type}/{test_key}/{taxonomic_level}.csv
                
        Args:
            output_dir: Base directory for saving results (defaults to project_dir/stats)
        """            
        total_files = 0
        # Count total files to save for progress tracking
        for table_type, tests in self.stats.items():
            for test_key, levels in tests.items():
                total_files += len(levels)
            
        with create_progress() as progress:
            task = progress.add_task(
                "[white]Saving statistical results...".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=total_files
            )
            base_dir.mkdir(parents=True, exist_ok=True) 
            for table_type, tests in self.stats.items():
                type_dir = base_dir / table_type
                type_dir.mkdir(parents=True, exist_ok=True)
                for test_key, levels in tests.items():
                    for level, result_df in levels.items():
                        level_dir = type_dir / level
                        level_dir.mkdir(parents=True, exist_ok=True)
                        output_path = level_dir / f"{test_key}.csv"
                        # Sort by p-value before saving
                        if 'p_value' in result_df.columns:
                            result_df = result_df.sort_values(by='p_value', ascending=True)
                        # Save DataFrame to CSV
                        result_df.set_index('feature').to_csv(output_path, index=True)
                            
                        if self.verbose:
                            logger.info(
                                f"Saved {table_type}/{test_key}/{level} stats "
                                f"to {output_path}"
                            )
                                
                        # Update progress
                        progress.update(task, advance=1)

    def _top_features(self):
        """
        Identify top features associated with contamination status by analyzing
        statistical results from ALL table types (raw, normalized, etc.) for each
        taxonomic level. Features are ranked by effect size and significance, with
        separate lists for contaminated and pristine associations.
        
        Steps:
        1. For each taxonomic level, collect results from all table types and tests
        2. For each feature, find its most significant association across all table types
        3. Classify features as contaminated-associated (positive effect) or pristine-associated (negative effect)
        4. Sort features by effect size magnitude and then by significance (p-value)
        5. Save the top features to class attributes
        
        Note: Each feature is represented only once (by its most significant association)
        """
        # Initialize lists to store classified features
        contaminated_features = []
        pristine_features = []
        
        # Define column mappings for different statistical tests
        TEST_COLUMN_MAP = {
            'ttest': {
                'effect': 'mean_difference',  # Primary effect size measure
                'alt_effect': 'cohens_d',     # Alternative effect size measure
                'p_value': 'p_value'          # Significance value
            },
            'mwub': {
                'effect': 'effect_size_r',    # Rank-biserial correlation effect size
                'alt_effect': 'median_difference',  # Alternative effect measure
                'p_value': 'p_value'
            },
            'kwb': {
                'effect': 'epsilon_squared',  # Variance explained effect size
                'p_value': 'p_value'
            },
            'fisher': {
                'effect': 'proportion_diff',  # Difference in proportion
                'alt_effect': 'odds_ratio',   # Alternative effect measure
                'p_value': 'p_value'
            }
        }
        
        # Process each taxonomic level (from phylum to genus)
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            # Dictionary to store best result per feature at this level
            # Format: {feature: {'p_value': float, 'effect': float, 'table_type': str, 'test': str}}
            level_features = {}
            
            # Collect results from all table types for this level
            for table_type in self.stats:
                # Skip if no results for this table type
                if table_type not in self.stats:
                    continue
                    
                for test_name, test_results in self.stats[table_type].items():
                    # Skip visualization results (PCA/t-SNE)
                    if test_name in ['pca', 'tsne']:
                        continue
                        
                    # Skip if no results for this level
                    if level not in test_results:
                        continue
                        
                    # Get results DataFrame for this test and level
                    df = test_results[level]
                    
                    # Process each feature in the results
                    for _, row in df.iterrows():
                        feature = row['feature']
                        p_value = row['p_value']
                        
                        # Skip non-significant results (p > 0.05)
                        if pd.isna(p_value) or p_value > 0.05:
                            continue
                            
                        # Get test configuration
                        if test_name not in TEST_COLUMN_MAP:
                            continue
                        config = TEST_COLUMN_MAP[test_name]
                        
                        # Get effect size - try primary then alternative
                        effect = None
                        if config['effect'] in row:
                            effect = row[config['effect']]
                        elif 'alt_effect' in config and config['alt_effect'] in row:
                            effect = row[config['alt_effect']]
                        
                        # Skip if no effect size found
                        if effect is None:
                            continue
                        
                        # Check if this is the best result for this feature
                        current_best = level_features.get(feature, None)
                        if current_best is None or p_value < current_best['p_value']:
                            level_features[feature] = {
                                'p_value': p_value,
                                'effect': effect,
                                'table_type': table_type,
                                'test': test_name
                            }
            
            # Classify features for this level
            for feature, result in level_features.items():
                feature_data = {
                    'feature': feature,
                    'level': level,
                    'table_type': result['table_type'],
                    'test': result['test'],
                    'effect': result['effect'],
                    'p_value': result['p_value']
                }
                
                if result['effect'] > 0:
                    contaminated_features.append(feature_data)
                else:
                    pristine_features.append(feature_data)
        
        # Sort features by effect size magnitude (absolute value) then by significance
        # Priority: 1. Strongest effects (large |effect|) 2. Most significant (low p-value)
        key_func = lambda x: (-abs(x['effect']), x['p_value'])
        
        contaminated_features = sorted(contaminated_features, key=key_func)
        pristine_features = sorted(pristine_features, key=key_func)
        
        # Store sorted results as class attributes
        self.top_contaminated_features = contaminated_features
        self.top_pristine_features = pristine_features
        
        # Log summary of findings
        logger.info(f"Identified {len(contaminated_features)} contaminated-associated features across all table types")
        logger.info(f"Identified {len(pristine_features)} pristine-associated features across all table types")
        
        # Optionally save to files
        self._save_top_features(contaminated_features, pristine_features, Path(self.project_dir.tables) / 'stats' / 'top_features')

    def _save_top_features(self, contaminated_features: List[dict], pristine_features: List[dict], base_dir: Path):
        """
        Save top feature associations to CSV files.
        
        Creates three files:
        1. Contaminated-associated features
        2. Pristine-associated features
        3. Combined report with all significant associations
        
        Files are saved in: {project_dir}/tables/top_features/
        
        Args:
            contaminated_features: List of dictionaries for contamination-associated features
            pristine_features: List of dictionaries for pristine-associated features
        """
        # Create output directory
        output_dir = Path(self.project_dir.tables) / "top_features"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current timestamp for filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create DataFrames from feature lists
        contam_df = pd.DataFrame(contaminated_features)
        pristine_df = pd.DataFrame(pristine_features)
        
        # Add direction column
        if not contam_df.empty:
            contam_df['direction'] = 'contaminated'
        if not pristine_df.empty:
            pristine_df['direction'] = 'pristine'
        
        # Save separate files
        if not contam_df.empty:
            contam_path = output_dir / f"contaminated_features_{timestamp}.csv"
            contam_df.to_csv(contam_path, index=False)
            logger.info(f"Saved {len(contam_df)} contaminated features to {contam_path}")
        
        if not pristine_df.empty:
            pristine_path = output_dir / f"pristine_features_{timestamp}.csv"
            pristine_df.to_csv(pristine_path, index=False)
            logger.info(f"Saved {len(pristine_df)} pristine features to {pristine_path}")
        
        # Create and save combined report
        if not contam_df.empty or not pristine_df.empty:
            combined_df = pd.concat([contam_df, pristine_df], ignore_index=True)
            combined_path = output_dir / f"all_significant_features_{timestamp}.csv"
            combined_df.to_csv(combined_path, index=False)
            logger.info(f"Saved {len(combined_df)} total features to {combined_path}")
            
            # Also save a version without timestamp for easy access
            latest_path = output_dir / "latest_significant_features.csv"
            combined_df.to_csv(latest_path, index=False)
            logger.info(f"Saved latest features to {latest_path}")
        else:
            logger.warning("No significant features found to save")
