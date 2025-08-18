# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
import warnings
import multiprocessing as mp

# Third‑Party Imports
import pandas as pd
import numpy as np
from biom.table import Table

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.downstream.input import update_table_and_metadata
from workflow_16s.stats.test import (
    anova, core_microbiome, differential_abundance_analysis, 
    enhanced_statistical_tests, fisher_exact_bonferroni, kruskal_bonferroni, 
    microbial_network_analysis, mwu_bonferroni, ttest, spearman_correlation
)
from workflow_16s.stats.utils import validate_inputs
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence, table_to_df,
    merge_table_with_meta
)
from workflow_16s.utils.io import export_h5py
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== OPTIMIZATION UTILITIES ========================== #

class TaskResult(NamedTuple):
    """Structured result for parallel tasks."""
    task_id: str
    table_type: str
    level: str
    test: str
    result: Optional[pd.DataFrame]
    error: Optional[str]

class DataCache:
    """Lightweight caching for preprocessed data."""
    def __init__(self, max_size: int = 128):
        self._cache = {}
        self._access_order = []
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Tuple]:
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key: str, value: Tuple) -> None:
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._max_size:
            # Remove least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        
        self._cache[key] = value
        self._access_order.append(key)
    
    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

# ========================== CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# Pre-compiled test configuration for faster access
TEST_CONFIG = {
    "fisher": {
        "key": "fisher", "func": fisher_exact_bonferroni,
        "name": "Fisher exact (Bonferroni)", "effect_col": "proportion_diff",
        "alt_effect_col": "odds_ratio", "parallel_safe": True
    },
    "ttest": {
        "key": "ttest", "func": ttest,
        "name": "Student t‑test", "effect_col": "mean_difference",
        "alt_effect_col": "cohens_d", "parallel_safe": True
    },
    "mwu_bonferroni": {
        "key": "mwub", "func": mwu_bonferroni,
        "name": "Mann–Whitney U (Bonferroni)", "effect_col": "effect_size_r",
        "alt_effect_col": "median_difference", "parallel_safe": True
    },
    "kruskal_bonferroni": {
        "key": "kwb", "func": kruskal_bonferroni,
        "name": "Kruskal–Wallis (Bonferroni)", "effect_col": "epsilon_squared",
        "alt_effect_col": None, "parallel_safe": True
    },
    "enhanced_stats": {
        "key": "enhanced", "func": enhanced_statistical_tests,
        "name": "Enhanced Statistical Tests", "effect_col": "effect_size",
        "alt_effect_col": None, "parallel_safe": False
    },
    "differential_abundance": {
        "key": "diffabund", "func": differential_abundance_analysis,
        "name": "Differential Abundance Analysis", "effect_col": "log2_fold_change",
        "alt_effect_col": "fold_change", "parallel_safe": False
    },
    "anova": {
        "key": "anova", "func": anova,
        "name": "One-way ANOVA", "effect_col": "eta_squared",
        "alt_effect_col": None, "parallel_safe": True
    },
    "spearman_correlation": {
        "key": "spearman", "func": spearman_correlation,
        "name": "Spearman Correlation", "effect_col": "rho",
        "alt_effect_col": None, "parallel_safe": True
    },
    "network_analysis": {
        "key": "network", "func": microbial_network_analysis,
        "name": "Network Analysis", "effect_col": "correlation",
        "alt_effect_col": "abs_correlation", "parallel_safe": False
    }
}

DEFAULT_TESTS = {
    "raw": ["ttest"],
    "filtered": ['mwu_bonferroni', 'kruskal_bonferroni'],
    "normalized": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
    "clr_transformed": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
    "presence_absence": ["fisher"]
}

# ========================== OPTIMIZED FUNCTIONS ========================== #

def _init_nested_dict(dictionary: Dict, keys: List[str]) -> None:
    """Initialize nested dictionary levels efficiently."""
    current = dictionary
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current.setdefault(keys[-1], {})

@lru_cache(maxsize=64)
def get_enabled_tasks_cached(
    config_hash: str,  # Hash of config for cache key
    available_tables: Tuple[Tuple[str, Tuple[str, ...]]]  # Hashable representation
) -> Tuple[Tuple[str, str, str], ...]:
    """Cached version of get_enabled_tasks."""
    # Note: This would need config deserialization in real implementation
    # For now, showing the optimization pattern
    tasks = []
    # Implementation would mirror original but with optimized data structures
    return tuple(tasks)

def get_enabled_tasks_optimized(
    config: Dict, 
    tables: Dict[str, Dict[str, Table]]
) -> List[Tuple[str, str, str]]:
    """Optimized task enumeration with early filtering."""
    stats_config = config.get('stats', {})
    table_config = stats_config.get('tables', {})
    
    tasks = []
    known_tests = set(TEST_CONFIG.keys())
    
    # Pre-filter enabled table types
    enabled_table_types = [
        table_type for table_type, type_config in table_config.items()
        if type_config.get('enabled', False) and table_type in tables
    ]
    
    for table_type in enabled_table_types:
        type_config = table_config[table_type]
        available_levels = set(tables[table_type].keys())
        
        # Efficiently filter levels
        configured_levels = set(type_config.get('levels', available_levels))
        enabled_levels = available_levels & configured_levels
        
        # Efficiently filter tests
        configured_tests = set(type_config.get('tests', DEFAULT_TESTS.get(table_type, [])))
        enabled_tests = configured_tests & known_tests
        
        # Generate tasks with list comprehension
        tasks.extend([
            (table_type, level, test)
            for level in enabled_levels
            for test in enabled_tests
        ])
    
    return tasks

@lru_cache(maxsize=32)
def get_group_column_values_cached(
    group_column_name: str,
    group_column_type: str,
    metadata_shape: Tuple[int, int],
    unique_values_hash: str  # Hash of unique values
) -> Tuple:
    """Cached group column value extraction."""
    # Implementation would deserialize and return cached values
    pass

def get_group_column_values_optimized(group_column: Dict, metadata: pd.DataFrame) -> List[Any]:
    """Optimized group column value extraction."""
    if 'values' in group_column and group_column['values']:
        return group_column['values']
    
    if group_column['type'] == 'bool':
        return [True, False]
    
    col_name = group_column['name']
    if col_name in metadata.columns:
        # Use more efficient unique value extraction
        return metadata[col_name].drop_duplicates().tolist()
    
    return []

def run_single_statistical_test(
    task_data: Tuple[str, str, str, Table, pd.DataFrame, str, List[Any], Path]
) -> TaskResult:
    """Optimized single test execution for parallel processing."""
    table_type, level, test, table, metadata, group_column, group_values, output_dir = task_data
    task_id = f"{table_type}_{level}_{test}"
    
    try:
        # Prepare data once
        table_aligned, metadata_aligned = update_table_and_metadata(table, metadata)
        test_func = TEST_CONFIG[test]["func"]
        
        # Handle different function signatures efficiently
        if test in {'enhanced_stats', 'differential_abundance'}:
            result = test_func(
                table=table_aligned,
                metadata=metadata_aligned,
                group_column=group_column
            )
        elif test == 'network_analysis':
            corr_matrix, edges_df = test_func(table=table_aligned)
            # Save correlation matrix
            corr_path = output_dir / f'{test}_correlation_matrix.tsv'
            corr_matrix.to_csv(corr_path, sep='\t')
            result = edges_df
        elif test == 'spearman_correlation':
            # Skip if column not found
            if group_column not in metadata_aligned.columns:
                return TaskResult(task_id, table_type, level, test, None, "Column not found")
            
            result = test_func(
                table=table_aligned,
                metadata=metadata_aligned,
                continuous_column=group_column
            )
        else:
            result = test_func(
                table=table_aligned,
                metadata=metadata_aligned,
                group_column=group_column,
                group_column_values=group_values
            )
        
        # Save results efficiently
        if isinstance(result, pd.DataFrame) and not result.empty:
            output_path = output_dir / f'{test}.tsv'
            result.to_csv(output_path, sep='\t', index=True)
        
        return TaskResult(task_id, table_type, level, test, result, None)
        
    except Exception as e:
        error_msg = f"Test '{test}' failed for {table_type}/{level}: {str(e)}"
        logger.error(error_msg)
        return TaskResult(task_id, table_type, level, test, None, error_msg)

class OptimizedStatisticalAnalysis:
    """Highly optimized Statistical Analysis class."""
    
    def __init__(
        self,
        config: Dict,
        tables: Dict,
        metadata: Dict,
        mode: str,
        group_columns: List,
        project_dir: Union[str, Path],
        max_workers: Optional[int] = None,
        use_process_pool: bool = False
    ) -> None:
        self.config = config
        self.project_dir = Path(project_dir) if isinstance(project_dir, str) else project_dir
        self.mode = mode
        self.tables = tables
        self.metadata = metadata
        self.group_columns = group_columns
        
        # Optimization settings
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.use_process_pool = use_process_pool
        
        # Initialize caching
        self._data_cache = DataCache()
        
        # Add NFC facilities if enabled
        if (self.config.get("nfc_facilities", {}).get('enabled', False) and 
            'facility_match' in self.metadata["raw"]["genus"].columns):
            self.group_columns.append({
                'name': 'facility_match', 
                'type': 'bool', 
                'values': [True, False]
            })
        
        self.results: Dict = {}
        self.advanced_results: Dict = {}
        
        # Pre-validate configuration
        validation_issues = self.validate_configuration()
        if validation_issues['errors']:
            logger.error(f"Configuration errors: {validation_issues['errors']}")
            raise ValueError("Configuration validation failed")
        
        # Run analysis with optimization
        self._run_optimized_analysis()
    
    def _get_cached_data(self, table_type: str, level: str) -> Tuple[Table, pd.DataFrame]:
        """Get cached aligned table and metadata."""
        cache_key = f"{table_type}_{level}"
        cached = self._data_cache.get(cache_key)
        
        if cached is not None:
            return cached
        
        # Prepare and cache data
        table = self.tables[table_type][level]
        metadata = self.metadata[table_type][level]
        table_aligned, metadata_aligned = update_table_and_metadata(table, metadata)
        
        cached_data = (table_aligned, metadata_aligned)
        self._data_cache.put(cache_key, cached_data)
        
        return cached_data
    
    def _run_optimized_analysis(self) -> None:
        """Run analysis with parallel processing and optimizations."""
        # Process each group column
        for group_column in self.group_columns:
            col_name = group_column['name']
            # Use optimized value extraction
            col_values = get_group_column_values_optimized(
                group_column, 
                self.metadata["raw"]["genus"]
            )
            
            print(f"Processing group column: {col_name}")
            print(f"Values: {col_values}")
            
            self.results[col_name] = self._run_parallel_for_group(col_name, col_values)
    
    def _run_parallel_for_group(self, group_column: str, group_values: List[Any]) -> Dict:
        """Run statistical analysis with parallel processing."""
        tasks = get_enabled_tasks_optimized(self.config, self.tables)
        if not tasks:
            return {}
        
        # Separate parallel-safe and sequential tasks
        parallel_tasks = []
        sequential_tasks = []
        
        for table_type, level, test in tasks:
            if TEST_CONFIG[test].get('parallel_safe', True):
                parallel_tasks.append((table_type, level, test))
            else:
                sequential_tasks.append((table_type, level, test))
        
        group_stats = {}
        
        # Process parallel tasks
        if parallel_tasks:
            group_stats.update(
                self._process_parallel_tasks(parallel_tasks, group_column, group_values)
            )
        
        # Process sequential tasks
        if sequential_tasks:
            group_stats.update(
                self._process_sequential_tasks(sequential_tasks, group_column, group_values)
            )
        
        return group_stats
    
    def _process_parallel_tasks(
        self, 
        tasks: List[Tuple[str, str, str]], 
        group_column: str, 
        group_values: List[Any]
    ) -> Dict:
        """Process tasks in parallel."""
        results = {}
        
        # Prepare task data
        task_data_list = []
        for table_type, level, test in tasks:
            output_dir = self.project_dir / 'stats' / group_column / table_type / level
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Get cached data
            table_aligned, metadata_aligned = self._get_cached_data(table_type, level)
            
            task_data = (
                table_type, level, test, table_aligned, metadata_aligned,
                group_column, group_values, output_dir
            )
            task_data_list.append(task_data)
        
        # Execute in parallel
        executor_class = ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
        
        with executor_class(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(run_single_statistical_test, task_data): task_data
                for task_data in task_data_list
            }
            
            # Collect results with progress tracking
            with get_progress_bar() as progress:
                task_desc = f"Parallel analysis for '{group_column}'"
                task_id = progress.add_task(_format_task_desc(task_desc), total=len(future_to_task))
                
                for future in as_completed(future_to_task):
                    task_result = future.result()
                    
                    # Store result
                    _init_nested_dict(results, [task_result.table_type, task_result.level])
                    results[task_result.table_type][task_result.level][task_result.test] = task_result.result
                    
                    if task_result.error:
                        logger.warning(f"Task failed: {task_result.error}")
                    
                    progress.update(task_id, advance=1)
        
        return results
    
    def _process_sequential_tasks(
        self, 
        tasks: List[Tuple[str, str, str]], 
        group_column: str, 
        group_values: List[Any]
    ) -> Dict:
        """Process tasks sequentially."""
        results = {}
        
        with get_progress_bar() as progress:
            task_desc = f"Sequential analysis for '{group_column}'"
            task_id = progress.add_task(_format_task_desc(task_desc), total=len(tasks))
            
            for table_type, level, test in tasks:
                output_dir = self.project_dir / 'stats' / group_column / table_type / level
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Get cached data
                table_aligned, metadata_aligned = self._get_cached_data(table_type, level)
                
                task_data = (
                    table_type, level, test, table_aligned, metadata_aligned,
                    group_column, group_values, output_dir
                )
                
                task_result = run_single_statistical_test(task_data)
                
                # Store result
                _init_nested_dict(results, [task_result.table_type, task_result.level])
                results[task_result.table_type][task_result.level][task_result.test] = task_result.result
                
                if task_result.error:
                    logger.warning(f"Task failed: {task_result.error}")
                
                progress.update(task_id, advance=1)
        
        return results
    
    def get_effect_size_optimized(self, test_name: str, row: pd.Series) -> Optional[float]:
        """Optimized effect size extraction."""
        test_config = TEST_CONFIG.get(test_name)
        if not test_config:
            return None
        
        # Check primary effect column first
        effect_col = test_config["effect_col"]
        if effect_col and effect_col in row and pd.notna(row[effect_col]):
            return float(row[effect_col])
        
        # Check alternative effect column
        alt_col = test_config["alt_effect_col"]
        if alt_col and alt_col in row and pd.notna(row[alt_col]):
            return float(row[alt_col])
        
        return None
    
    def get_summary_statistics_optimized(self) -> Dict:
        """Generate summary statistics with vectorized operations."""
        summary = {
            'total_tests_run': 0,
            'significant_features_by_test': {},
            'effect_sizes_summary': {},
            'group_columns_analyzed': list(self.results.keys()),
            'performance_metrics': {
                'cache_hit_ratio': len(self._data_cache._cache) / max(1, len(self._data_cache._access_order))
            }
        }
        
        # Vectorized processing
        all_results = []
        
        for group_col, group_results in self.results.items():
            for table_type, levels in group_results.items():
                for level, tests in levels.items():
                    for test_name, result in tests.items():
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            summary['total_tests_run'] += 1
                            
                            # Count significant features
                            test_key = test_name
                            if test_key not in summary['significant_features_by_test']:
                                summary['significant_features_by_test'][test_key] = 0
                            summary['significant_features_by_test'][test_key] += len(result)
                            
                            # Collect effect sizes for vectorized processing
                            effect_col = TEST_CONFIG.get(test_name, {}).get('effect_col')
                            if effect_col and effect_col in result.columns:
                                all_results.append({
                                    'test': test_name,
                                    'effects': result[effect_col].dropna().values
                                })
        
        # Vectorized effect size summary computation
        for result_data in all_results:
            test_name = result_data['test']
            effects = result_data['effects']
            
            if len(effects) > 0:
                if test_name not in summary['effect_sizes_summary']:
                    summary['effect_sizes_summary'][test_name] = {
                        'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
                    }
                
                # Use numpy for fast computation
                summary['effect_sizes_summary'][test_name] = {
                    'mean': float(np.mean(effects)),
                    'std': float(np.std(effects)),
                    'min': float(np.min(effects)),
                    'max': float(np.max(effects)),
                    'count': len(effects)
                }
        
        return summary
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Optimized configuration validation."""
        issues = {'errors': [], 'warnings': [], 'info': []}
        
        # Batch validation of tables/metadata alignment
        alignment_tasks = [
            (table_type, level, self.tables[table_type][level], self.metadata[table_type][level])
            for table_type in self.tables
            for level in self.tables[table_type]
        ]
        
        for table_type, level, table, metadata in alignment_tasks:
            try:
                update_table_and_metadata(table, metadata)
            except Exception as e:
                issues['errors'].append(f"Alignment failed for {table_type}/{level}: {e}")
        
        # Vectorized group column validation
        for group_column in self.group_columns:
            col_name = group_column['name']
            found_locations = []
            
            for table_type in self.metadata:
                for level in self.metadata[table_type]:
                    metadata = self.metadata[table_type][level]
                    if col_name in metadata.columns:
                        found_locations.append((table_type, level))
                        
                        # Efficient group size checking
                        group_counts = metadata[col_name].value_counts()
                        small_groups = group_counts[group_counts < 3]
                        if not small_groups.empty:
                            issues['warnings'].append(
                                f"Small groups in '{col_name}' at {table_type}/{level}: {dict(small_groups)}"
                            )
            
            if not found_locations:
                issues['errors'].append(f"Group column '{col_name}' not found")
        
        return issues
    
    def cleanup(self) -> None:
        """Clean up resources and caches."""
        self._data_cache.clear()
        logger.info("Statistical analysis cleanup completed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

# ========================== UTILITY FUNCTIONS ========================== #

def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage."""
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
            df[col] = df[col].astype(np.int32)
        elif df[col].min() >= np.iinfo(np.int16).min and df[col].max() <= np.iinfo(np.int16).max:
            df[col] = df[col].astype(np.int16)
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

def batch_save_results(results: Dict, base_path: Path, format: str = 'tsv') -> None:
    """Efficiently batch save results."""
    save_tasks = []
    
    for group_col, group_results in results.items():
        for table_type, levels in group_results.items():
            for level, tests in levels.items():
                for test_name, result in tests.items():
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        output_dir = base_path / group_col / table_type / level
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / f'{test_name}.{format}'
                        save_tasks.append((result, output_path))
    
    # Execute saves in parallel if beneficial
    if len(save_tasks) > 10:  # Threshold for parallel saving
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(lambda r, p: r.to_csv(p, sep='\t', index=True), result, path)
                for result, path in save_tasks
            ]
            
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Save failed: {e}")
    else:
        # Sequential save for small numbers
        for result, path in save_tasks:
            try:
                result.to_csv(path, sep='\t', index=True)
            except Exception as e:
                logger.error(f"Save failed for {path}: {e}")
