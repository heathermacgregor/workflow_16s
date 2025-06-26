# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from biom import load_table
from biom.table import Table
from rich.progress import Progress, TaskID
from skbio.stats.ordination import pcoa as PCoA

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.biom import collapse_taxa, convert_to_biom, export_h5py, presence_absence
from workflow_16s.utils.progress import create_progress
from workflow_16s.utils.file_utils import import_merged_table_biom, import_merged_meta_tsv
from workflow_16s.stats.utils import clr_transform_table, filter_table, normalize_table
from workflow_16s.stats.utils import merge_table_with_metadata, table_to_dataframe
from workflow_16s.stats.tests import fisher_exact_bonferroni, kruskal_bonferroni, mwu_bonferroni, ttest
from workflow_16s.stats.beta_diversity import pcoa as calculate_pcoa, pca as calculate_pca, tsne as calculate_tsne, umap as calculate_umap
from workflow_16s.figures.merged.merged import mds, pca, pcoa, sample_map_categorical
from workflow_16s.models.feature_selection import (
    filter_data, perform_feature_selection, grid_search,
    save_feature_importances
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
warnings.filterwarnings("ignore")  # Suppress warnings

# ================================= DEFAULT VALUES =================================== #

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

DEFAULT_PROGRESS_TEXT_N = 50
DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'

# ============================= STATISTICAL ANALYZER CLASS ============================ #
class StatisticalAnalyzer:
    """Handles all statistical analyses for amplicon data"""
    
    TEST_CONFIG = {
        'fisher': {
            'key': 'fisher',
            'func': fisher_exact_bonferroni,
            'name': 'Fisher test (w/ Bonferroni)',
            'effect_col': 'proportion_diff',
            'alt_effect_col': 'odds_ratio'
        },
        'ttest': {
            'key': 'ttest',
            'func': ttest,
            'name': 't-test',
            'effect_col': 'mean_difference',
            'alt_effect_col': 'cohens_d'
        },
        'mwu_bonferroni': {
            'key': 'mwub',
            'func': mwu_bonferroni,
            'name': 'Mann-Whitney U test (w/ Bonferroni)',
            'effect_col': 'effect_size_r',
            'alt_effect_col': 'median_difference'
        },
        'kruskal_bonferroni': {
            'key': 'kwb',
            'func': kruskal_bonferroni,
            'name': 'Kruskal-Wallis test (w/ Bonferroni)',
            'effect_col': 'epsilon_squared',
            'alt_effect_col': None
        }
    }
    
    def __init__(self, cfg: Dict, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
    
    def run_tests(
        self,
        table: Table,
        metadata: pd.DataFrame,
        group_column: str,
        group_values: List[Any],
        enabled_tests: List[str],
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None
    ) -> Dict[str, Any]:
        """
        Run statistical tests on a feature table
        
        Args:
            table:         BIOM feature table
            metadata:      Sample metadata
            group_column:  Column in metadata to group by
            group_values:  Values to compare in group_column
            enabled_tests: List of tests to run
            progress:      Rich progress object
            task_id:       Parent task ID
            
        Returns:
            Dictionary of test results keyed by test name
        """
        results = {}
        total_tests = len(enabled_tests)
        
        if progress and task_id:
            main_task = progress.add_task(
                f"[white]Running statistical tests", 
                total=total_tests,
                parent=task_id
            )
        
        for test_name in enabled_tests:
            if test_name not in self.TEST_CONFIG:
                continue
                
            config = self.TEST_CONFIG[test_name]
            test_key = config['key']
            
            if self.verbose:
                logger.info(f"Running {config['name']}...")
                
            results[test_key] = config['func'](
                table=table,
                metadata=metadata,
                group_column=group_column,
                group_column_values=group_values,
            )
            
            if progress and task_id:
                progress.update(main_task, advance=1)
                
        return results

    def get_effect_size(self, test_name: str, result_row: pd.Series) -> Optional[float]:
        """
        Extract effect size from statistical test results
        
        Args:
            test_name:  Name of statistical test
            result_row: Row from results DataFrame
            
        Returns:
            Effect size value or None if not found
        """
        if test_name not in self.TEST_CONFIG:
            return None
            
        config = self.TEST_CONFIG[test_name]
        effect_col = config['effect_col']
        alt_effect_col = config['alt_effect_col']
        
        if effect_col in result_row:
            return result_row[effect_col]
        elif alt_effect_col and alt_effect_col in result_row:
            return result_row[alt_effect_col]
        return None

# ============================= ORDINATION CLASS ============================ #
class Ordination:
    """Handles ordination analyses and plotting for all taxonomic levels"""
    
    TEST_CONFIG = {
        'pca': {
            'key': 'pca',
            'func': calculate_pca,
            'plot_func': pca,
            'name': 'Principal Components Analysis'
        },
        'pcoa': {
            'key': 'pcoa',
            'func': calculate_pcoa,
            'plot_func': pcoa,
            'name': 'Principal Coordinates Analysis',
        },
        'tsne': {
            'key': 'tsne',
            'func': calculate_tsne,
            'plot_func': mds,
            'name': 'TSNE',
            'plot_kwargs': {'mode': 'TSNE'}
        },
        'umap': {
            'key': 'umap',
            'func': calculate_umap,
            'plot_func': mds,
            'name': 'UMAP',
            'plot_kwargs': {'mode': 'UMAP'}
        }
    }
    
    def __init__(self, cfg: Dict, output_dir: Union[str, Path], verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.figure_output_dir = Path(output_dir)
        self.results = {}
        self.figures = {}
    
    def run_tests(
        self,
        table: Table,
        metadata: pd.DataFrame,
        color_col: str,
        symbol_col: str,
        transformation: str,
        enabled_tests: List[str],
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run ordination analyses and generate plots for a specific table
        
        Args:
            table:         BIOM feature table for current taxonomic level
            metadata:      Sample metadata
            color_col:     Column for coloring points
            symbol_col:    Column for point symbols
            transformation: Data transformation applied + taxonomic level
            enabled_tests: List of ordination methods to run
            progress:      Rich progress object
            task_id:       Parent task ID
            
        Returns:
            Dictionary of ordination results keyed by method name
        """
        results = {}
        figures = {}
        total_tests = len(enabled_tests)
        tests_to_run = [test for test in enabled_tests if test in self.TEST_CONFIG]
        
        if not tests_to_run:
            return results, figures
            
        if progress and task_id:
            main_task = progress.add_task(
                f"[white]Running ordination for {transformation}", 
                total=total_tests,
                parent=task_id
            )
        
        try:
            # Filter and align table with metadata
            logger.info(f"Aligning samples for {transformation}")
            table, metadata = filter_and_reorder_biom_and_metadata(table, metadata)
            logger.info(f"Aligned table: {table.shape[0]} features × {table.shape[1]} samples")
            
            # Run each enabled ordination method
            for test_name in tests_to_run:
                config = self.TEST_CONFIG[test_name]
                test_key = config['key']
                
                if self.verbose:
                    logger.info(f"Running {config['name']} for {transformation}...")
                    
                try:
                    # Compute ordination
                    ordination_result = config['func'](table=table)
                    results[test_key] = ordination_result
                    
                    # Prepare plot arguments
                    plot_kwargs = config.get('plot_kwargs', {})
                    plot_kwargs.update({
                        'metadata': metadata,
                        'color_col': color_col,
                        'symbol_col': symbol_col,
                        'transformation': transformation,
                        'output_dir': self.figure_output_dir,
                        **kwargs
                    })
                    
                    # Handle different result types
                    if test_key == 'pca':
                        plot_kwargs.update({
                            'components': ordination_result['components'],
                            'proportion_explained': ordination_result['exp_var_ratio']
                        })
                    elif test_key == 'pcoa':
                        plot_kwargs.update({
                            'components': ordination_result.samples,
                            'proportion_explained': ordination_result.proportion_explained
                        })
                    else:  # t-SNE or UMAP
                        plot_kwargs['df'] = ordination_result
                    
                    # Generate plot and capture figure
                    logger.info(f"Generating {test_key} plot for {transformation}")
                    fig, colordict = config['plot_func'](**plot_kwargs)
                    figures[test_key] = fig
                    
                except Exception as e:
                    logger.error(f"Failed {test_name} for {transformation}: {str(e)}")
                    logger.debug("Traceback:", exc_info=True)
                    figures[test_key] = None  # Store placeholder for failed plot
                    
                finally:
                    # Update progress after each test
                    if progress and task_id:
                        progress.update(main_task, advance=1)
                        
        except Exception as e:
            logger.error(f"Ordination failed for {transformation}: {str(e)}")
            logger.debug("Traceback:", exc_info=True)
            # Advance progress even if alignment fails
            if progress and task_id:
                progress.update(main_task, advance=total_tests)
                
        return results, figures

# ================================= PLOTTER CLASS ================================== #
class Plotter:
    """Handles sample map visualization"""
    
    def __init__(self, cfg: Dict, output_dir: Path, verbose: bool = False):
        self.cfg = cfg
        self.output_dir = output_dir
        self.verbose = verbose
        
    def generate_sample_map(
        self, 
        metadata: pd.DataFrame,
        color_columns: List[str] = ['dataset_name', 'nuclear_contamination_status'],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate sample location maps
        
        Args:
            metadata:      Sample metadata with location data
            color_columns: Columns to use for coloring points
            **kwargs:      Additional plot arguments
            
        Returns:
            Dictionary of generated figures keyed by color column
        """
        figures = {}
        for color_col in color_columns:
            fig, _ = sample_map_categorical(
                metadata=metadata,
                output_dir=self.output_dir,
                color_col=color_col,
                **kwargs
            )
            figures[color_col] = fig
        return figures

# ================================== TOP FEATURES ANALYZER ================================== #
class TopFeaturesAnalyzer:
    """Identifies top features associated with experimental groups"""
    
    def __init__(self, cfg: Dict, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
    
    def analyze(
        self, 
        stats_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        group_column: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Identify top features associated with group differences
        
        Args:
            stats_results: Nested dictionary of statistical results
            group_column:  Metadata column defining groups
            
        Returns:
            Tuple of (contaminated_features, pristine_features)
        """
        contaminated_features = []
        pristine_features = []
        analyzer = StatisticalAnalyzer(self.cfg, self.verbose)
        
        # Process each taxonomic level
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            level_features = {}
            
            # Collect results across all table types and tests
            for table_type, tests in stats_results.items():
                for test_name, test_results in tests.items():
                    if level not in test_results:
                        continue
                    
                    # Process each feature in results
                    for _, row in test_results[level].iterrows():
                        feature = row['feature']
                        p_value = row['p_value']
                        
                        # Skip non-significant results
                        if pd.isna(p_value) or p_value > 0.05:
                            continue
                            
                        # Get effect size
                        effect = analyzer.get_effect_size(test_name, row)
                        if effect is None:
                            continue
                            
                        # Track best result per feature
                        current = level_features.get(feature)
                        if not current or p_value < current['p_value']:
                            level_features[feature] = {
                                'p_value': p_value,
                                'effect': effect,
                                'table_type': table_type,
                                'test': test_name
                            }
            
            # Classify features by effect direction
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
        
        # Sort by effect size magnitude and significance
        key_func = lambda x: (-abs(x['effect']), x['p_value'])
        contaminated_features.sort(key=key_func)
        pristine_features.sort(key=key_func)
        
        return contaminated_features, pristine_features


# ================================== UTILITY FUNCTIONS ================================== #    

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


# ================================== AMPLICON DATA CLASS ================================== #
class AmpliconData:
    """Main class for processing amplicon sequencing data"""
    
    MODE_CONFIG = {
        'asv': ('table', 'asv'),
        'genus': ('table_6', 'l6')
    }
    
    def __init__(self, cfg, project_dir, mode='genus', verbose=False):
        self.cfg = cfg
        self.project_dir = project_dir
        self.mode = mode
        self.verbose = verbose
        self.meta = None
        self.table = None
        self.tables = {}
        self.stats = {}
        self.ordination = {}
        self.figures = {}
        self.top_contaminated_features = []
        self.top_pristine_features = []
        
        self._validate_mode()
        self._set_output_paths()
        self._load_data()
        self._process_data()
        self._run_analyses()

    def _biom_to_df(self, table: Table) -> pd.DataFrame:
        """
        Convert a BIOM table to a pandas DataFrame with samples as rows and features as columns.
        
        Args:
            table: BIOM table to convert
            
        Returns:
            DataFrame with samples as rows and features as columns
        """
        # Transpose to get samples as rows and features as columns
        data = table.matrix_data.toarray().T
        sample_ids = table.ids(axis='sample')
        feature_ids = table.ids(axis='observation')
        return pd.DataFrame(data, index=sample_ids, columns=feature_ids)
        
    def _validate_mode(self):
        """Validate processing mode"""
        if self.mode not in self.MODE_CONFIG:
            raise ValueError(f"Invalid processing mode: {self.mode}")
    
    def _set_output_paths(self):
        """Set output paths for processed data"""
        table_dir, output_dir = self.MODE_CONFIG[self.mode]
        self.figure_output_dir = Path(self.project_dir.figures)
        self.table_output_path = (
            Path(self.project_dir.data) / 'merged' / 'table' / 
            output_dir / 'feature-table.biom'
        )
        self.meta_output_path = (
            Path(self.project_dir.data) / 'merged' / 'metadata' / 
            'sample-metadata.tsv'
        )
    
    def _load_data(self):
        """Load metadata and feature tables"""
        self._load_metadata()
        self._load_biom_table()
        self._filter_and_align_data()
    
    def _load_metadata(self):
        """Load and merge metadata from multiple sources"""
        meta_paths = self._get_metadata_paths()
        self.meta = import_merged_meta_tsv(
            meta_paths, 
            None,
            self.verbose
        )
    
    def _get_metadata_paths(self) -> List[Path]:
        """Get paths to metadata files corresponding to BIOM tables"""
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
            if meta_path.exists():
                meta_paths.append(meta_path)
            
        if self.verbose:
            logger.info(f"Found {RED}{len(meta_paths)}{RESET} metadata files")
        return meta_paths
    
    def _load_biom_table(self):
        """Load and merge BIOM feature tables"""
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
          ['*', '*', '*', '*', 'FWD_*_REV_*', self.MODE_CONFIG[self.mode][0], 
           'feature-table.biom']
        )
        biom_paths = glob.glob(
            str(Path(self.project_dir.qiime_data_per_dataset) / pattern), 
            recursive=True
        )
        if self.verbose:
            logger.info(f"Found {RED}{len(biom_paths)}{RESET} feature tables")
        return [Path(p) for p in biom_paths]
    
    def _filter_and_align_data(self):
        """Filter and align BIOM table with metadata"""
        original_n_samples = self.table.shape[1]
        self.table, self.meta = filter_and_reorder_biom_and_metadata(
            table=self.table, metadata_df=self.meta, sample_column="#sampleid"
        )
        
        logger.info(
            f"Loaded (samples x features) metadata table with "
            f"{RED}{self.meta.shape[0]}{RESET} samples "
            f"and {RED}{self.meta.shape[1]}{RESET} columns"
        )

        if self.cfg["figures"]["map"]:
            self.plotter = Plotter(self.cfg, self.figure_output_dir, self.verbose)
            self.figures["map"] = self.plotter.generate_sample_map(self.meta)
            
        feature_type = 'genera' if self.mode == 'genus' else 'ASVs'
        logger.info(
            f"Loaded (features x samples) feature table with "
            f"{RED}{self.table.shape[1]} ({original_n_samples}){RESET} samples "
            f"and {RED}{self.table.shape[0]}{RESET} {feature_type}"
        )
    
    def _process_data(self):
        """Process data through filtering and transformation pipeline"""
        self._apply_preprocessing()
        self._collapse_taxa()
        self._create_presence_absence()
        self._save_tables()
    
    def _apply_preprocessing(self):
        """
        Apply filtering, normalization, and CLR transformation to the table before 
        collapsing.
        """
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
        enabled_steps = [
            filtering_enabled, normalization_enabled, clr_transformation_enabled
        ]
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
                table = filtered_table  # Use filtered table for next steps
            
            # Apply normalization if enabled (requires prior filtering)
            if normalization_enabled:
                if self.verbose:
                    logger.info("Applying normalization to table...")
                normalized_table = normalize_table(table, axis=1)
                self.tables["normalized"] = {}
                self.tables["normalized"][self.mode] = normalized_table
                progress.update(main_task, advance=1)
                table = normalized_table  # Use normalized table for next step
            
            # Apply CLR transformation if enabled (requires prior normalization)
            if clr_transformation_enabled:
                if self.verbose:
                    logger.info("Applying CLR transformation to table...")
                clr_transformed_table = clr_transform_table(table)
                self.tables["clr_transformed"] = {}
                self.tables["clr_transformed"][self.mode] = clr_transformed_table
                progress.update(main_task, advance=1)
    
    def _collapse_taxa(self):
        """Generate tables by collapsing taxa at different levels"""
        tax_levels = ['phylum', 'class', 'order', 'family', 'genus']
        
        # Process all table types
        for table_type in list(self.tables.keys()):
            if table_type not in self.tables:
                continue
                
            current_table = self.tables[table_type][self.mode]
            self.tables[table_type] = self._run_processing_step(
                process_name=f"Collapsing {table_type} taxonomy",
                process_func=collapse_taxa,
                levels=tax_levels,
                func_args=(),
                get_source=lambda level: current_table,
                log_template=f"Collapsed {table_type} to {{level}} level"
            )
    
    def _create_presence_absence(self):
        """Generate presence/absence tables if enabled in config"""
        if not self.cfg['features']['presence_absence']:
            return
            
        # Only create presence/absence for raw tables
        self.tables["presence_absence"] = self._run_processing_step(
            process_name="Converting to presence/absence",
            process_func=presence_absence,
            levels=['phylum', 'class', 'order', 'family', 'genus'],
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
    
    def _save_tables(self):
        """Save all generated tables to appropriate directories"""
        # Calculate total tables to save
        total_tables = sum(len(level_tables) for level_tables in self.tables.values())
        
        with create_progress() as progress:
            task = progress.add_task(
                "[white]Saving tables...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=total_tables
            )
            base_dir = Path(self.project_dir.data) / 'merged' / 'table'
            base_dir.mkdir(parents=True, exist_ok=True)
            for table_type, level_tables in self.tables.items():
                type_dir = base_dir / table_type
                type_dir.mkdir(parents=True, exist_ok=True)
                for level, table in level_tables.items():
                    level_dir = type_dir / level
                    level_dir.mkdir(parents=True, exist_ok=True)
                    output_path = level_dir / "feature-table.biom"
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
    
    def _run_analyses(self):
        """Run statistical analyses and visualizations"""
        self._identify_top_features()
        self._run_ml_feature_selection()
        self._run_ordination()
        
    
    def _run_ordination(self):
        """Run ordination analyses for all table types and levels"""
        # Define ordination methods to run
        ordination_methods = ['pca', 'pcoa', 'tsne', 'umap']
        
        # Calculate total plots: (table_types * levels * methods)
        total_plots = 0
        for table_type in self.tables:
            for level in self.tables[table_type]:
                total_plots += len(ordination_methods)
        
        logger.debug(f"Total ordination plots to generate: {total_plots}")
        
        with create_progress() as progress:
            main_task = progress.add_task(
                "[white]Running ordination analyses".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=total_plots
            ) if total_plots > 0 else None
            
            # Initialize storage structures
            self.ordination = {}
            self.figures = {}
            
            for table_type, level_tables in self.tables.items():
                self.ordination[table_type] = {}
                self.figures[table_type] = {}
                
                for level, table in level_tables.items():
                    # Create ordination instance with proper output directory
                    ordination_output_dir = self.figure_output_dir / level / table_type
                    ordination = Ordination(
                        cfg=self.cfg,
                        output_dir=ordination_output_dir,
                        verbose=self.verbose
                    )
                    
                    # Run ordination and capture results + figures
                    results, figures = ordination.run_tests(
                        table=table,
                        metadata=self.meta,
                        color_col='dataset_name',
                        symbol_col='nuclear_contamination_status',
                        transformation=f"{table_type} ({level})",
                        enabled_tests=ordination_methods,
                        progress=progress,
                        task_id=main_task
                    )
                    
                    # Store results
                    self.ordination[table_type][level] = results
                    self.figures[table_type][level] = figures
                    
                    # Additional debug logging
                    if self.verbose:
                        logger.info(f"Stored ordination results for {table_type}/{level}:")
                        logger.info(f"  Methods: {list(results.keys())}")
                        logger.info(f"  Figures: {list(figures.keys())}")
    
    def _identify_top_features(self):
        """Identify top features associated with contamination status"""
        analyzer = TopFeaturesAnalyzer(self.cfg, self.verbose)
        self.top_contaminated_features, self.top_pristine_features = analyzer.analyze(
            self.stats, DEFAULT_GROUP_COLUMN
        )

    def _run_ml_feature_selection(self):
        ml_cfg = self.cfg.get("ml", {})
        if not ml_cfg.get("enable", False):
            return
    
        tbl_type = ml_cfg["table_type"]
        level = ml_cfg["level"]
        
        # Get the table and convert to DataFrame
        table = self.tables[tbl_type][level]
        X = table_to_dataframe(table)
        X.index = X.index.str.lower()

        print(X.shape)
        print(X.index)
        # Get labels and align with feature matrix

        
        y = self.meta.set_index('#sampleid')[[DEFAULT_GROUP_COLUMN]]  # contamination label
        print(y.shape)
        print(y.index)
        # Align indices (crucial for correct sample-feature matching)
        common_samples = X.index.intersection(y.index)
        print(common_samples)
        X = X.loc[common_samples]
        y = y.loc[common_samples]
        
        # Stratified split preserving class balance
        X_tr, X_te, y_tr, y_te = filter_data(
            X, y, self.meta.loc[list(common_samples)], DEFAULT_GROUP_COLUMN,
            test_size=0.3, random_state=42
        )
        
        # Feature selection
        X_tr_sel, X_te_sel, selected = perform_feature_selection(
            X_tr, y_tr, X_te, y_te,
            feature_selection=ml_cfg["method"],
            num_features=ml_cfg["num_features"],
            step_size=ml_cfg.get("step_size", 100),
            thread_count=ml_cfg["catboost_threads"],
            random_state=42,
            use_permutation_importance=ml_cfg["permutation_importance"],
        )
        
        # Hyperparameter tuning
        params_grid = {
            "iterations": [1000],
            "learning_rate": [0.1],
            "depth": [4],
            "loss_function": ["Logloss"],
            "thread_count": [ml_cfg["catboost_threads"]],
        }
        best_model, best_params, best_mcc = grid_search(
            X_tr_sel, y_tr, X_te_sel, y_te,
            params_grid,
            output_dir=self.figure_output_dir / "ml"
        )
        
        # Save feature importances
        save_feature_importances(
            best_model,
            pd.DataFrame(X_tr_sel, columns=selected),
            self.figure_output_dir / "ml"
        )
        
        # Store results for later use
        self.ml_selected_features = selected
        self.ml_model = best_model
