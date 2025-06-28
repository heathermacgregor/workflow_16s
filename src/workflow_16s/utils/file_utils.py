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
from workflow_16s.utils.progress import get_progress_bar

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

DEFAULT_PROGRESS_TEXT_N = 65

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# ==================================== FUNCTIONS ===================================== #    

def load_datasets_list(path: Union[str, Path]) -> List[str]:
    """
    Load dataset IDs from configuration file.
    
    Args:
        path: Path to text file containing dataset IDs (one per line).
    
    Returns:
        List of dataset ID strings.
    """
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]
    
    
def load_datasets_info(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load dataset metadata from TSV file.
    
    Args:
        tsv_path: Path to TSV file containing dataset metadata.
    
    Returns:
        DataFrame with dataset information, cleaned of unnamed columns.
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype={'ena_project_accession': str})
    # Remove 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    return df


def fetch_first_match(dataset_info: pd.DataFrame, dataset: str) -> pd.Series:
    """
    Find matching dataset information from metadata DataFrame.
    
    Args:
        dataset_info: DataFrame containing dataset metadata.
        dataset:      Dataset identifier to search for.
    
    Returns:
        First matching row from dataset_info as a pandas Series.
    
    Raises:
        ValueError: If no matches found for the dataset.
    """
    # Case-insensitive masks
    mask_ena_type = dataset_info['dataset_type'].str.lower().eq('ena')
    mask_manual_type = dataset_info['dataset_type'].str.lower().eq('manual')

    # Check ENA metadata: either ena_project_accession OR dataset_id contains 
    # the dataset
    mask_ena = ((
        dataset_info['ena_project_accession'].str.contains(
            dataset, case=False, regex=False
        )
        | dataset_info['dataset_id'].str.contains(
            dataset, case=False, regex=False
        )
    ) & mask_ena_type)

    # Check Manual metadata: dataset_id contains the dataset
    mask_manual = (
        dataset_info['dataset_id'].str.contains(
            dataset, case=False, regex=False
        ) 
        & mask_manual_type
    )

    combined_mask = mask_ena | mask_manual
    matching_rows = dataset_info[combined_mask]

    # Handle no matches
    if matching_rows.empty:
        raise ValueError(f"No matches found for dataset: {dataset}")

    # Prioritize ENA matches over manual ones
    matching_rows = matching_rows.sort_values(
         by='dataset_type', 
         key=lambda x: x.str.lower().map({'ena': 0, 'manual': 1})  # ENA first
    )
    return matching_rows.iloc[0]


def processed_dataset_files(
    dirs: SubDirs, 
    dataset: str, 
    params: Any, 
    cfg: Any
) -> Dict[str, Path]:
    """
    Generate expected file paths for processed dataset outputs.
    
    Args:
        dirs:    Project directory structure.
        dataset: Dataset identifier.
        params:  Processing parameters dictionary.
        cfg:     Configuration dictionary.
    
    Returns:
        Dictionary mapping file types to their expected paths.
    """
    classifier = cfg["classifier"]
    base_dir = (
        Path(dirs.qiime_data_per_dataset) / dataset / 
        params['instrument_platform'].lower() / 
        params['library_layout'].lower() / 
        params['target_subfragment'].lower() / 
        f"FWD_{params['pcr_primer_fwd_seq']}_REV_{params['pcr_primer_rev_seq']}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)
    return {
        'metadata_tsv': Path(dirs.metadata_per_dataset) / dataset / 'metadata.tsv',
        'manifest_tsv': base_dir / 'manifest.tsv',
        # BIOM feature table
        'table_biom': base_dir / 'table' / 'feature-table.biom',  
        # Representative seqs
        'seqs_fasta': base_dir / 'rep-seqs' / 'dna-sequences.fasta',  
        # Taxonomy
        'taxonomy_tsv': base_dir / classifier / 'taxonomy' / 'taxonomy.tsv',  
    }


def missing_output_files(file_list: List[Union[str, Path]]) -> List[Path]:
    """
    Identify missing output files from a list of expected paths.
    
    Args:
        file_list: List of file paths to check.
    
    Returns:
        List of Path objects for files that don't exist.
    """
    return [Path(file) for file in file_list if not Path(file).exists()]

# ==================================== FUNCTIONS ===================================== #   

class AmpliconData:
    """
    A class for processing and analyzing amplicon sequencing data.
    
    This class handles data loading, processing at various taxonomic levels,
    statistical analysis, and visualization based on configuration settings.
    
    Attributes:
        cfg (Dict): Configuration settings
        project_dir (Union[str, Path]): Project directory structure
        mode (str): Processing mode ('asv' or 'genus')
        verbose (bool): Verbosity flag
        meta (pd.DataFrame): Sample metadata
        table (pd.DataFrame): Feature table
        color_maps (Dict): Color mappings for visualization
        figures (Dict): Generated figures
        tables (Dict): Processed tables at different taxonomic levels
        stats (Dict): Statistical analysis results
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
            cfg: Configuration dictionary
            project_dir: Project directory path
            mode: Processing mode ('asv' or 'genus')
            verbose: Enable verbose logging
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
        logger.info(
            f"Loaded metadata table with {RED}{self.meta.shape[0]}{RESET} samples "
            f"and {RED}{self.meta.shape[1]}{RESET} features"
        )
        self._load_biom_table()
        logger.info(
            f"Loaded feature table with {RED}{self.table.shape[1]}{RESET} samples "
            f"and {RED}{self.table.shape[0]}{RESET} features"
        )

    def _load_metadata(self):
        """Load and merge metadata from multiple sources."""
        meta_paths = self._get_metadata_paths()
        self.meta = import_merged_meta_tsv(
            meta_paths, 
            self.meta_output_path, 
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
            'dataframe',
            #self.table_output_path,
            self.verbose
        )

    def _get_biom_paths(self) -> List[Path]:
        """
        Get paths to BIOM feature tables using a glob pattern.
        
        Returns:
            List of Path objects to BIOM files
        """
        pattern = '/'.join(['*', '*', '*', '*', 'FWD_*_REV_*', 
                           self._mode_config[self.mode][0], 'feature-table.biom'])
        biom_paths = glob.glob(
            str(Path(self.project_dir.qiime_data_per_dataset) / pattern), 
            recursive=True
        )
        if self.verbose:
            logger.info(f"Found {len(biom_paths)} BIOM tables")
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
            logger.info(f"Found {len(meta_paths)} metadata files")
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
        self._process_raw_tables(tax_levels)
        self._process_presence_absence(tax_levels)
        
        # Save all generated tables
        self._save_all_tables(table_dir)

    def _apply_preprocessing_steps(self):
        """Apply filtering, normalization, and CLR transformation to the table before collapsing."""
        # Start with the original table
        processed_table = self.table
        
        # Apply filtering if enabled
        if self.cfg['features']['filter']:
            if self.verbose:
                logger.info("Applying filtering to table...")
            processed_table = filter_table(processed_table)
        
        # Apply normalization if enabled (requires prior filtering)
        if self.cfg['features']['filter'] and self.cfg['features']['normalize']:
            if self.verbose:
                logger.info("Applying normalization to table...")
            processed_table = normalize_table(processed_table, axis=1)
        
        # Apply CLR transformation if enabled (requires prior normalization)
        enabled = (self.cfg['features']['filter'] and 
                  self.cfg['features']['normalize'] and 
                  self.cfg['features']['clr_transform'])
        if enabled:
            if self.verbose:
                logger.info("Applying CLR transformation to table...")
            processed_table = clr_transform_table(processed_table)
        
        # Update the table with processed version
        self.table = processed_table

    def _process_raw_tables(self, levels: List[str]):
        """Generate raw tables by collapsing taxa at different levels."""
        self.tables["raw"] = self._run_processing_step(
            process_name="Collapsing taxonomy",
            process_func=collapse_taxa,
            levels=levels,
            func_args=(),
            get_source=lambda _: self.table,
            log_template="Collapsed to {level} level"
        )

    def _process_presence_absence(self, levels: List[str]):
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
            levels: Taxonomic levels to process
            func_args: Additional arguments for process_func
            get_source: Function to get input table for a level
            log_template: Template for logging messages
            log_action: Action name for simple logging
            
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
            level: Taxonomic level being processed
            template: String template for logging (uses {level} placeholder)
            action: Simple action description
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
        base_dir.mkdir(parents=True, exist_ok=True)
        for table_type, level_tables in self.tables.items():
            type_dir = base_dir / table_type
            type_dir.mkdir(exist_ok=True)
            for level, table in level_tables.items():
                output_path = type_dir / f"feature-table_{level}.biom"
                print(output_path)
                print(table.shape)
                export_h5py(table, output_path)
                if self.verbose:
                    n_features, n_samples = table.shape
                    shape_str = f"[{n_features}, {n_samples}]"
                    logger.info(
                        f"Wrote {table_type} {level} table {shape_str} to '{output_path}'"
                    )
                

    def _run_all_statistical_analyses(self):
        """Run statistical analyses for all generated table types."""
        for table_type in self.tables:
            self._run_statistical_analyses(table_type)

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
        with get_progress_bar() as progress:
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

    def _run_visual_analyses(
        self,
        table_type: str,
        test_type: str,
        progress: Progress,
        parent_task_id: int
    ):
        """
        Run visual analyses (PCA/t-SNE) for a table type.
        
        Args:
            table_type: Type of table to analyze
            test_type: Type of visual analysis ('pca' or 'tsne')
            progress: Rich progress instance
            parent_task_id: Parent task ID for progress tracking
        """
        self.stats[table_type][test_type] = {}
        tables = self.tables[table_type]
        
        # Create task for visual analyses
        vis_task = progress.add_task(
            f"[bold magenta]Visual Analyses: {test_type.upper()}",
            parent=parent_task_id,
            total=len(tables))
        
        for level in tables:
            progress.update(vis_task, description=f"[white]{test_type.upper()} {level}")
            
            # Match metadata and table indices
            meta, table, _ = df_utils.match_indices_or_transpose(
                self.meta, tables[level]
            )
            
            # Execute appropriate analysis
            if test_type == 'pca':
                result = beta_diversity.pca(table=table, n_components=3)
            elif test_type == 'tsne':
                result = beta_diversity.tsne(table=table, n_components=3)
            else:
                logger.warning(f"Unsupported visual analysis: {test_type}")
                continue
                
            self.stats[table_type][test_type][level] = result
            progress.advance(vis_task)
        
        # Update parent task
        progress.update(parent_task_id, advance=1)
        progress.remove_task(vis_task)

    def _top_features(self, table_type: str = 'presence_absence'):
        """
        Identify top features associated with contamination status by analyzing
        statistical results from multiple tests. Features are ranked by effect size
        and significance, with separate lists for contaminated and pristine associations.
        
        Args:
            table_type: Type of table to analyze (e.g., 'raw', 'presence_absence')
        """
        contaminated_features = []
        pristine_features = []
        
        # Check if statistical results exist for this table type
        if table_type not in self.stats:
            logger.warning(
                f"No statistical results found for table type: {table_type}"
            )
            return
            
        # Define column mappings for different statistical tests
        TEST_COLUMN_MAP = {
            'ttest': {
                'effect': 'mean_diff',
                'p_value': 'p_value',
                'q_value': 'q_value'
            },
            'mwub': {  # Mann-Whitney U with Bonferroni
                'effect': 'effect_size',
                'p_value': 'p_value',
                'q_value': 'q_value'
            },
            'kwb': {   # Kruskal-Wallis with Bonferroni
                'effect': 'effect_size',
                'p_value': 'p_value',
                'q_value': 'q_value'
            }
        }
        
        # Process each taxonomic level
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            # Collect all DataFrames for this level across tests
            result_dfs = []
            for test_name, level_results in self.stats[table_type].items():
                # Skip visualization results (PCA/t-SNE)
                if test_name in ['pca', 'tsne']:
                    continue
                    
                # Get results for this level if available
                if level in level_results:
                    df = level_results[level].copy()
                    
                    # Add test name identifier to columns
                    df.columns = [f"{test_name}_{col}" if col != 'feature' else col 
                                 for col in df.columns]
                    result_dfs.append(df)
            
            # Skip if no results for this level
            if not result_dfs:
                continue
                
            # Merge results from different tests
            merged_df = reduce(
                lambda left, right: pd.merge(left, right, on='feature', how='outer'),
                result_dfs
            )
            
            # Process each feature in the merged results
            for _, row in merged_df.iterrows():
                best_effect = 0
                best_q_value = 1.0
                best_test = None
                
                # Find the most significant result across tests
                for test_name in TEST_COLUMN_MAP:
                    # Construct column names for this test
                    effect_col = f"{test_name}_{TEST_COLUMN_MAP[test_name]['effect']}"
                    q_value_col = f"{test_name}_{TEST_COLUMN_MAP[test_name]['q_value']}"
                    
                    # Skip if test results not available for this feature
                    if effect_col not in row or q_value_col not in row:
                        continue
                        
                    effect = row[effect_col]
                    q_value = row[q_value_col]
                    
                    # Skip non-significant or missing results
                    if pd.isna(effect) or pd.isna(q_value) or q_value >= 0.05:
                        continue
                    
                    # Track the most significant result
                    if abs(effect) > abs(best_effect) or \
                       (abs(effect) == abs(best_effect) and q_value < best_q_value):
                        best_effect = effect
                        best_q_value = q_value
                        best_test = test_name
                
                # If no significant result found, skip feature
                if best_test is None:
                    continue
                    
                # Classify as contaminated or pristine association
                feature_data = {
                    'feature': row['feature'],
                    'level': level,
                    'test': best_test,
                    'effect': best_effect,
                    'q_value': best_q_value
                }
                
                if best_effect > 0:
                    contaminated_features.append(feature_data)
                else:
                    pristine_features.append(feature_data)
        
        # Save top features
        self._save_top_features(contaminated_features, pristine_features)
    
    def _save_top_features(
        self, 
        contaminated_features: list, 
        pristine_features: list,
        n_top: int = 20
    ):
        """
        Save top features to TSV files, sorted by effect size and significance.
        
        Args:
            contaminated_features: Features positively associated with contamination
            pristine_features: Features negatively associated with contamination
            n_top: Number of top features to save
        """
        top_dir = Path(self.project_dir.tables) / 'stats' / 'top_features'
        top_dir.mkdir(parents=True, exist_ok=True)
        
        # Process contaminated features (positive effect)
        if contaminated_features:
            contam_df = pd.DataFrame(contaminated_features)
            contam_df = contam_df.sort_values(
                by=['effect', 'q_value'], 
                ascending=[False, True]  # High effect first, low q-value first
            ).head(n_top)
            
            output_path = top_dir / 'top20_contaminated.tsv'
            contam_df.to_csv(output_path, sep='\t', index=False)
            logger.info(f"Saved top contaminated features to {output_path}")
        else:
            logger.info("No significant contaminated features found")
        
        # Process pristine features (negative effect)
        if pristine_features:
            pristine_df = pd.DataFrame(pristine_features)
            pristine_df = pristine_df.sort_values(
                by=['effect', 'q_value'], 
                ascending=[True, True]  # Most negative effect first
            ).head(n_top)
            
            # Convert effect to absolute value for easier interpretation
            pristine_df['effect'] = pristine_df['effect'].abs()
            
            output_path = top_dir / 'top20_pristine.tsv'
            pristine_df.to_csv(output_path, sep='\t', index=False)
            logger.info(f"Saved top pristine features to {output_path}")
        else:
            logger.info("No significant pristine features found")
    
            logger.info(f"Saved top pristine features to {top_dir / 'top20_pristine.tsv'}")

    def _plot_stuff(self, table_type: str = 'presence_absence', figure_type: str = 'pca'):
        """
        Generate visualizations for analysis results.
        
        Args:
            table_type:  Type of table used for visualization.
            figure_type: Type of visualization to create.
        """
        levels = {'phylum': 1, 'class': 2, 'order': 3, 'family': 4, 'genus': 5}
        color_cols = ['dataset_name']
        symbol_col = 'nuclear_contamination_status'
        
        tables = self.tables[table_type] 
        self.figures[table_type] = {}
        self.figures[table_type][figure_type] = []
        
        # Generate plots for each level
        for level in tables:
            if figure_type not in self.stats[table_type] or level not in self.stats[table_type][figure_type]:
                continue
                
            results = self.stats[table_type][figure_type][level]
            
            for color_col in color_cols:
                if figure_type == 'pca':
                    plot, _ = pca(
                        components=results['components'], 
                        proportion_explained=results['exp_var_ratio'], 
                        metadata=self.meta,
                        color_col=color_col, 
                        color_map=self.color_maps.get(color_col, None),
                        symbol_col=symbol_col,
                        show=False,
                        output_dir=Path(self.project_dir.figures) / 'pca' / f'l{levels[level]+1}', 
                        x=1, 
                        y=2
                    )
                elif figure_type == 'tsne':
                    plot, _ = mds(
                        df=results, 
                        metadata=self.meta,
                        group_col=color_col, 
                        symbol_col=symbol_col,
                        show=False,
                        output_dir=Path(self.project_dir.figures) / 'tsne' / f'l{levels[level]+1}',
                        mode='TSNE',
                        x=1, 
                        y=2
                    )
                
                # Store figure reference
                self.figures[table_type][figure_type].append({
                    'title': f'{figure_type.upper()} - {level}',
                    'level': level,
                    'color_col': color_col,
                    'symbol_col': symbol_col,
                    'figure': plot
                })

#######################################################

def import_meta_tsv(
    tsv_path: Union[str, Path],
    column_renames: Optional[List[Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Load a sample metadata TSV file, rename columns, and ensure required fields exist.
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {tsv_path}")

    # ensure we can iterate even if None was passed
    if column_renames is None:
        column_renames = []

    df = pd.read_csv(tsv_path, sep='\t')
    df.columns = df.columns.str.lower()

    # determine sample ID
    sample_id_col = next(
        (col for col in ['run_accession', '#sampleid', 'sample-id'] if col in df.columns),
        None
    )
    if sample_id_col:
        df['SAMPLE ID'] = df[sample_id_col]
    else:
        df['SAMPLE ID'] = [
            f"{Path(tsv_path).parents[5].name}_x{i}"
            for i in range(1, len(df) + 1)
        ]

    # determine dataset ID
    dataset_id_col = next(
        (col for col in ['project_accession', 'dataset_id', 'dataset_name'] if col in df.columns),
        None
    )
    if dataset_id_col:
        df['DATASET ID'] = df[dataset_id_col]
    else:
        df['DATASET ID'] = Path(tsv_path).parents[5].name

    # ensure nuclear_contamination_status exists
    if 'nuclear_contamination_status' not in df.columns:
        df['nuclear_contamination_status'] = False

    # apply any requested renames
    for old, new in column_renames:
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)

    return df


def import_merged_meta_tsv(
    meta_paths: List[Union[str, Path]],
    #output_path: Union[str, Path] = None,
    column_renames: Optional[List[Tuple[str, str]]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Load and merge multiple sample metadata TSV files.
    """
    dfs = []

    if verbose:
        for path in meta_paths:
            try:
                df = import_meta_tsv(path, column_renames)
                dfs.append(df)
                logger.info(f"Loaded {Path(path).name} with {df.shape[0]} samples")
            except Exception as e:
                logger.error(f"Failed to load {path}: {e!r}")
    else:
        with get_progress_bar() as progress:
            task = progress.add_task(
                "[white]Loading metadata files...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=len(meta_paths)
            )
            for path in meta_paths:
                try:
                    df = import_meta_tsv(path, column_renames)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e!r}")
                finally:
                    progress.update(task, advance=1)

    if not dfs:
        raise FileNotFoundError(
            "None of the metadata files could be loaded; "
            "check that your sample-metadata.tsv files exist and are readable."
        )

    df_merged = pd.concat(dfs, ignore_index=True)

    #if output_path:
    #    output_path = Path(output_path)
    #    output_path.parent.mkdir(parents=True, exist_ok=True)
    #    df_merged.to_csv(output_path, sep='\t', index=True)
    #    if verbose:
    #        n_samples, n_features = df_merged.shape
    #        logger.info(f"Wrote merged metadata (samples × features) [{n_samples}, {n_features}] to {output_path}")

    return df_merged

# ======================================= BIOM ======================================= #

def import_table_biom(
    biom_path: Union[str, Path], 
    as_type: str = 'table'
) -> Union[Table, pd.DataFrame]:
    """
    Import BIOM table from file.
    
    Args:
        biom_path: Path to .biom file.
        as_type:   Return type ('table' or 'dataframe').
    
    Returns:
        BIOM Table or DataFrame
    """
    try:
        with biom_open(biom_path) as f:
            table = Table.from_hdf5(f)
    except:
        table = load_table(biom_path)
        
    if as_type == 'table':
        return table
    elif as_type == 'dataframe':
        feature_ids = table.ids(axis='observation')
        sample_ids = table.ids(axis='sample')
        data = table.matrix_data.toarray()
        return pd.DataFrame(data, index=feature_ids, columns=sample_ids)
    else:
        raise ValueError(f"Invalid as_type: {as_type}. Use 'table' or 'dataframe'")


def import_merged_table_biom(
    biom_paths: List[Union[str, Path]], 
    as_type: str = 'table',
    #output_path: Union[str, Path] = None,
    verbose: bool = False
) -> Union[Table, pd.DataFrame]:
    """Merge multiple BIOM tables into one."""
    tables = []

    if verbose:
        for path in biom_paths:
            try:
                table = import_table_biom(path, 'table')
                tables.append(table)
                logger.info(f"Loaded {Path(path).name} with {len(table.ids('sample'))} samples")
            except Exception as e:
                logger.error(f"Failed to load {path}: {str(e)}")
    else:
        with get_progress_bar() as progress:
            task = progress.add_task(
                "[white]Loading BIOM files...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=len(biom_paths)
            )
            for path in biom_paths:
                try:
                    table = import_table_biom(path, 'table')
                    tables.append(table)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {str(e)}")
                finally:
                    progress.update(task, advance=1)

    if not tables:
        raise ValueError("No valid BIOM tables loaded")

    merged_table = tables[0]
    for table in tables[1:]:
        merged_table = merged_table.merge(table)

    #if output_path:
    #    output_path = Path(output_path)
    #    output_path.parent.mkdir(parents=True, exist_ok=True)
    #    with h5py.File(output_path, 'w') as f:
    #        merged_table.to_hdf5(f, generated_by="workflow_16s")
    #    if verbose:
    #        n_features, n_samples = merged_table.shape
    #        logger.info(f"Wrote table [{n_features}, {n_samples}] to {output_path}")

    return merged_table if as_type == 'table' else merged_table.to_dataframe()

# ====================================== FASTA ======================================= #

def import_seqs_fasta(fasta_path: Union[str, Path]) -> Dict[str, str]:
    """
    Import sequences from FASTA file.
    
    Args:
        fasta_path: Path to FASTA file
    
    Returns:
        Dictionary mapping sequence IDs to sequences
    """
    return {
        record.id: str(record.seq)
        for record in SeqIO.parse(fasta_path, "fasta")
    }

def import_faprotax_tsv(tsv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Import FAPROTAX results from TSV.
    
    Args:
        tsv_path: Path to FAPROTAX TSV output
    
    Returns:
        Transposed DataFrame with samples as rows and functions as columns
    """
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    return df.T

# ==================================== CLASSES ====================================== #

class Taxonomy:
    """
    Class for handling taxonomic classification data.
    
    Attributes:
        taxonomy: DataFrame containing taxonomy information
    """
    def __init__(self, tsv_path: Union[str, Path]):
        self.taxonomy = self.import_taxonomy_tsv(tsv_path)
        
    def import_taxonomy_tsv(self, tsv_path: Union[str, Path]) -> pd.DataFrame:
        """Process taxonomy TSV file into structured DataFrame.
        
        Args:
            tsv_path: Path to QIIME2 taxonomy TSV
        
        Returns:
            DataFrame with separate columns for each taxonomic level
        """
        def extract_level(taxonomy: str, level: str) -> str:
            prefix = level + '__'
            if not taxonomy or taxonomy in ['Unassigned', 'Unclassified']:
                return 'Unclassified'
            start = taxonomy.find(prefix)
            if start == -1:
                return None
            end = taxonomy.find(';', start)
            return taxonomy[start+len(prefix):end] if end != -1 else taxonomy[start+len(prefix):]
        
        df = pd.read_csv(tsv_path, sep='\t')
        df = df.rename(columns={
            'Feature ID': 'id', 
            'Taxon': 'taxonomy', 
            'Consensus': 'confidence'
        }).set_index('id')
        
        # Clean taxonomy strings
        df['taxstring'] = df['taxonomy'].str.replace(r' *[dpcofgs]__', '', regex=True)
        
        # Add taxonomic levels
        for level in ['d', 'p', 'c', 'o', 'f', 'g', 's']:
            df[level.upper()] = df['taxonomy'].apply(lambda x: extract_level(x, level))
            
        return df.rename(columns={
            'D': 'Domain', 'P': 'Phylum', 'C': 'Class',
            'O': 'Order', 'F': 'Family', 'G': 'Genus', 'S': 'Species'
        })
        
    def get_taxstring_by_id(self, feature_id: str) -> Optional[str]:
        """Get taxonomy string for a feature ID.
        
        Args:
            feature_id: Feature ID to look up
        
        Returns:
            Taxonomy string or None if not found
        """
        return self.taxonomy.loc[feature_id, 'taxstring'] if feature_id in self.taxonomy.index else None


def write_metadata_tsv(df: pd.DataFrame, tsv_path: str) -> None:
    """
    Write metadata DataFrame to TSV format.
    
    Args:
        df: Metadata DataFrame
        tsv_path: Output path for TSV file
    """
    df = df.copy()
    if '#SampleID' not in df.columns and 'run_accession' in df.columns:
        df['#SampleID'] = df['run_accession']
    df.set_index('#SampleID', inplace=True)
    df.to_csv(tsv_path, sep='\t', index=True)


def write_manifest_tsv(results: Dict, tsv_path: str) -> None:
    """
    Write manifest file for QIIME2 import.
    
    Args:
        results: Dictionary mapping run accessions to file paths
        tsv_path: Output path for manifest file
    """
    rows = []
    for run_id, paths in results.items():
        if len(paths) == 1:
            rows.append({'sample-id': run_id, 'absolute-filepath': paths[0]})
        elif len(paths) == 2:
            rows.append({
                'sample-id': run_id,
                'forward-absolute-filepath': paths[0],
                'reverse-absolute-filepath': paths[1]
            })
    pd.DataFrame(rows).set_index('sample-id').to_csv(tsv_path, sep='\t', index=True)
    

def manual_meta(dataset: str, metadata_dir: Union[str, Path]) -> pd.DataFrame:
    """
    Load manually curated metadata for a dataset.
    
    Args:
        dataset: Dataset identifier
        metadata_dir: Base directory containing metadata files
    
    Returns:
        DataFrame with manual metadata, empty if not found
    """
    path = Path(metadata_dir) / dataset / 'manual-metadata.tsv'
    return pd.read_csv(path, sep="\t") if path.exists() else pd.DataFrame()

def filter_and_reorder_biom_and_metadata(
    table: Table,
    metadata_df: pd.DataFrame,
    sample_column: str = '#sampleid'
) -> tuple[Table, pd.DataFrame]:
    """
    Filters and reorders a BIOM table and metadata DataFrame to align sample IDs.

    Args:
        table:         BIOM table (features x samples)
        metadata_df:   Metadata DataFrame containing sample IDs
        sample_column: Column name in metadata_df containing sample IDs

    Returns:
        Tuple of filtered/reordered BIOM table and metadata DataFrame
    """
    # Create normalized copies to avoid mutating inputs
    metadata_df = _normalize_metadata(metadata_df, sample_column)
    biom_mapping = _create_biom_id_mapping(table)
    
    # Find shared sample IDs (lowercase) in metadata order
    metadata_ids = metadata_df[sample_column].tolist()
    shared_lower_ids = [sid for sid in metadata_ids if sid in biom_mapping]
    
    # Filter metadata to shared samples (preserves order)
    metadata_df = metadata_df[metadata_df[sample_column].isin(shared_lower_ids)]
    
    # Get original-case IDs from biom mapping in metadata order
    original_case_ids = [biom_mapping[sid] for sid in metadata_df[sample_column]]
    
    # Filter and reorder biom table
    filtered_table = table.filter(original_case_ids, axis='sample', inplace=False)
    
    return filtered_table, metadata_df

def _normalize_metadata(
    metadata_df: pd.DataFrame, 
    sample_column: str
) -> pd.DataFrame:
    """Normalize metadata sample IDs and remove duplicates."""
    df = metadata_df.copy()
    df[sample_column] = df[sample_column].astype(str).str.lower()
    return df.drop_duplicates(subset=[sample_column])

def _create_biom_id_mapping(table: Table) -> dict:
    """Create lowercase to original ID mapping with duplicate checking."""
    mapping = {}
    for orig_id in table.ids(axis='sample'):
        lower_id = orig_id.lower()
        if lower_id in mapping:
            raise ValueError(
                f"Duplicate lowercase sample ID in BIOM table: '{lower_id}' "
                f"(from '{orig_id}' and '{mapping[lower_id]}')"
            )
        mapping[lower_id] = orig_id
    return mapping

