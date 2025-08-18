# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings
from dataclasses import dataclass
from functools import lru_cache, partial
import time

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

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ==================================== DATA CLASSES ===================================== #

@dataclass(frozen=True)
class TestConfig:
    """Immutable test configuration."""
    key: str
    func: Callable
    name: str
    effect_col: str
    alt_effect_col: Optional[str] = None
    requires_continuous: bool = False
    min_samples: int = 3
    supports_multigroup: bool = True

@dataclass
class AnalysisTask:
    """Task definition for statistical analysis."""
    table_type: str
    level: str
    test_name: str
    group_column: str
    priority: int = 0

@dataclass
class NetworkStats:
    """Network analysis statistics."""
    total_edges: int
    positive_edges: int
    negative_edges: int
    mean_correlation: float
    unique_nodes: int
    density: float

# ==================================== CONSTANTS ===================================== #

# Optimized test configurations with better organization
TEST_CONFIGS = {
    "fisher": TestConfig(
        key="fisher",
        func=fisher_exact_bonferroni,
        name="Fisher exact (Bonferroni)",
        effect_col="proportion_diff",
        alt_effect_col="odds_ratio",
        min_samples=5,
        supports_multigroup=False
    ),
    "ttest": TestConfig(
        key="ttest",
        func=ttest,
        name="Student t‑test",
        effect_col="mean_difference",
        alt_effect_col="cohens_d",
        min_samples=3,
        supports_multigroup=False
    ),
    "mwu_bonferroni": TestConfig(
        key="mwub",
        func=mwu_bonferroni,
        name="Mann–Whitney U (Bonferroni)",
        effect_col="effect_size_r",
        alt_effect_col="median_difference",
        min_samples=3,
        supports_multigroup=False
    ),
    "kruskal_bonferroni": TestConfig(
        key="kwb",
        func=kruskal_bonferroni,
        name="Kruskal–Wallis (Bonferroni)",
        effect_col="epsilon_squared",
        min_samples=5
    ),
    "enhanced_stats": TestConfig(
        key="enhanced",
        func=enhanced_statistical_tests,
        name="Enhanced Statistical Tests",
        effect_col="effect_size"
    ),
    "differential_abundance": TestConfig(
        key="diffabund",
        func=differential_abundance_analysis,
        name="Differential Abundance Analysis",
        effect_col="log2_fold_change",
        alt_effect_col="fold_change"
    ),
    "anova": TestConfig(
        key="anova",
        func=anova,
        name="One-way ANOVA",
        effect_col="eta_squared",
        min_samples=5
    ),
    "spearman_correlation": TestConfig(
        key="spearman",
        func=spearman_correlation,
        name="Spearman Correlation",
        effect_col="rho",
        requires_continuous=True,
        min_samples=10
    ),
    "network_analysis": TestConfig(
        key="network",
        func=microbial_network_analysis,
        name="Network Analysis",
        effect_col="correlation",
        alt_effect_col="abs_correlation",
        min_samples=20
    )
}

DEFAULT_TESTS = {
    "raw": ["ttest"],
    "filtered": ['mwu_bonferroni', 'kruskal_bonferroni'],
    "normalized": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
    "clr_transformed": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
    "presence_absence": ["fisher"]
}

# ==================================== HELPER FUNCTIONS ===================================== #

def _init_dict_level(dictionary: Dict, *keys) -> None:
    """Initialize nested dictionary levels efficiently."""
    current = dictionary
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current.setdefault(keys[-1], {})

@lru_cache(maxsize=128)
def _validate_group_column_cached(group_column_name: str, metadata_hash: int) -> bool:
    """Cached validation of group column presence."""
    # This would need to be implemented with actual metadata checking
    # The hash ensures cache invalidation when metadata changes
    return True

def get_enabled_tasks(
    config: Dict, 
    tables: Dict[str, Dict[str, Table]]
) -> List[AnalysisTask]:
    """Get enabled analysis tasks with improved organization."""
    stats_config = config.get('stats', {})
    table_config = stats_config.get('tables', {})
    
    tasks = []
    priority_map = {'fisher': 1, 'ttest': 2, 'mwu_bonferroni': 3, 
                   'kruskal_bonferroni': 4, 'enhanced_stats': 5}
    
    for table_type, levels in tables.items():
        table_type_config = table_config.get(table_type, {})
        if not table_type_config.get('enabled', False):
            continue

        enabled_levels = [
            l for l in table_type_config.get('levels', levels.keys()) 
            if l in levels
        ]
        enabled_tests = [
            t for t in table_type_config.get('tests', DEFAULT_TESTS.get(table_type, [])) 
            if t in TEST_CONFIGS
        ]

        for level in enabled_levels:
            for test in enabled_tests:
                # Group column will be added later when running analysis
                task = AnalysisTask(
                    table_type=table_type,
                    level=level,
                    test_name=test,
                    group_column="",  # Will be set when running
                    priority=priority_map.get(test, 99)
                )
                tasks.append(task)
    
    return sorted(tasks, key=lambda x: x.priority)

@lru_cache(maxsize=64)
def get_group_column_values_cached(group_column_name: str, group_type: str, metadata_id: str):
    """Cached version of group column value extraction."""
    # This would need metadata passed differently for caching to work
    # For now, keeping the original logic but with caching structure
    pass

def get_group_column_values(group_column: Dict, metadata: pd.DataFrame) -> Optional[List]:
    """Optimized group column value extraction."""
    col = group_column['name']
    
    # Use cached values if provided
    if 'values' in group_column and group_column['values']:
        return group_column['values']
    
    # Handle boolean columns
    if group_column.get('type') == 'bool':
        return [True, False]
    
    # Extract unique values from metadata
    if col in metadata.columns:
        unique_vals = metadata[col].dropna().unique()
        return unique_vals.tolist() if len(unique_vals) > 0 else None
    
    return None

def create_output_directory(base_path: Path, *subdirs: str) -> Path:
    """Create output directory structure efficiently."""
    output_dir = base_path
    for subdir in subdirs:
        output_dir = output_dir / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

# ==================================== MAIN CLASS ===================================== #

class OptimizedStatisticalAnalysis:
    """Enhanced Statistical Analysis class with performance optimizations."""
    
    def __init__(
        self,
        config: Dict,
        tables: Dict,
        metadata: Dict,
        mode: str,
        group_columns: List,
        project_dir: Union[str, Path],
        max_workers: int = 4,
        enable_caching: bool = True
    ) -> None:
        self.config = config
        self.project_dir = Path(project_dir) if isinstance(project_dir, str) else project_dir
        self.mode = mode
        self.tables = tables
        self.metadata = metadata
        self.group_columns = group_columns
        self.max_workers = max_workers
        self.enable_caching = enable_caching
        
        # Performance tracking
        self.execution_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Add NFC facilities if enabled
        self._add_nfc_facilities()
        
        self.results: Dict = {}
        self.advanced_results: Dict = {}
        
        # Initialize with validation
        validation_results = self.validate_configuration()
        if validation_results['errors']:
            logger.error(f"Configuration errors found: {validation_results['errors']}")
            raise ValueError("Configuration validation failed")
        
        # Run analysis
        self._run_parallel_analysis()

    def _add_nfc_facilities(self) -> None:
        """Add NFC facilities group column if enabled."""
        if (self.config.get("nfc_facilities", {}).get('enabled', False) and 
            'facility_match' in self.metadata["raw"]["genus"].columns):
            self.group_columns.append({
                'name': 'facility_match', 
                'type': 'bool', 
                'values': [True, False]
            })

    def _run_parallel_analysis(self) -> None:
        """Run statistical analysis in parallel for all group columns."""
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each group column
            future_to_group = {}
            
            for group_column in self.group_columns:
                col = group_column['name']
                vals = get_group_column_values(group_column, self.metadata["raw"]["genus"])
                
                if vals is None:
                    logger.warning(f"No valid values found for group column: {col}")
                    continue
                    
                future = executor.submit(self._run_for_group, col, vals)
                future_to_group[future] = col
            
            # Collect results
            for future in as_completed(future_to_group):
                group_col = future_to_group[future]
                try:
                    self.results[group_col] = future.get(timeout=300)  # 5 minute timeout
                    logger.info(f"Completed analysis for group: {group_col}")
                except TimeoutError:
                    logger.error(f"Analysis timeout for group: {group_col}")
                    self.results[group_col] = {}
                except Exception as e:
                    logger.error(f"Analysis failed for group {group_col}: {e}")
                    self.results[group_col] = {}
        
        total_time = time.time() - start_time
        self.execution_times['total_analysis'] = total_time
        logger.info(f"Total analysis time: {total_time:.2f}s")

    def _run_for_group(self, group_column: str, group_column_values: List[Any]) -> Dict:
        """Optimized analysis for a specific group column."""
        start_time = time.time()
        
        # Get base tasks (without group column set)
        base_tasks = get_enabled_tasks(self.config, self.tables)
        if not base_tasks:
            return {}
        
        # Create tasks with group column
        tasks = [
            AnalysisTask(
                table_type=task.table_type,
                level=task.level,
                test_name=task.test_name,
                group_column=group_column,
                priority=task.priority
            )
            for task in base_tasks
        ]
        
        group_stats = {}
        failed_tasks = []
        
        # Pre-validate and cache aligned data
        aligned_data_cache = {}
        
        with get_progress_bar() as progress:
            stats_desc = f"Running statistics for '{group_column}'"
            stats_task = progress.add_task(_format_task_desc(stats_desc), total=len(tasks))
            
            # Process tasks in batches to optimize memory usage
            batch_size = min(self.max_workers * 2, len(tasks))
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batch))) as executor:
                    # Submit batch tasks
                    future_to_task = {}
                    for task in batch:
                        future = executor.submit(
                            self._execute_single_task, 
                            task, group_column_values, aligned_data_cache
                        )
                        future_to_task[future] = task
                    
                    # Collect batch results
                    for future in as_completed(future_to_task):
                        task = future_to_task[future]
                        try:
                            result = future.get(timeout=60)  # 1 minute per task
                            if result is not None:
                                # Store result
                                _init_dict_level(group_stats, task.table_type, task.level)
                                group_stats[task.table_type][task.level][task.test_name] = result
                                
                                # Save result
                                self._save_result(task, result)
                            
                        except Exception as e:
                            logger.error(f"Task failed {task.table_type}/{task.level}/{task.test_name}: {e}")
                            failed_tasks.append(task)
                        finally:
                            progress.update(stats_task, advance=1)
        
        # Log performance stats
        execution_time = time.time() - start_time
        self.execution_times[f'group_{group_column}'] = execution_time
        
        if failed_tasks:
            logger.warning(f"Failed tasks for {group_column}: {len(failed_tasks)}")
        
        return group_stats

    def _get_aligned_data(
        self, 
        table_type: str, 
        level: str, 
        cache: Dict
    ) -> Tuple[Table, pd.DataFrame]:
        """Get aligned table and metadata with caching."""
        cache_key = f"{table_type}_{level}"
        
        if self.enable_caching and cache_key in cache:
            self.cache_hits += 1
            return cache[cache_key]
        
        self.cache_misses += 1
        table = self.tables[table_type][level]
        metadata = self.metadata[table_type][level]
        table_aligned, metadata_aligned = update_table_and_metadata(table, metadata)
        
        if self.enable_caching:
            cache[cache_key] = (table_aligned, metadata_aligned)
        
        return table_aligned, metadata_aligned

    def _execute_single_task(
        self, 
        task: AnalysisTask, 
        group_column_values: List[Any],
        aligned_data_cache: Dict
    ) -> Optional[pd.DataFrame]:
        """Execute a single statistical test task."""
        test_config = TEST_CONFIGS[task.test_name]
        
        try:
            # Get aligned data
            table_aligned, metadata_aligned = self._get_aligned_data(
                task.table_type, task.level, aligned_data_cache
            )
            
            # Validate sample size
            if len(metadata_aligned) < test_config.min_samples:
                logger.warning(
                    f"Insufficient samples for {task.test_name} "
                    f"({len(metadata_aligned)} < {test_config.min_samples})"
                )
                return None
            
            # Validate group structure
            if task.group_column not in metadata_aligned.columns:
                logger.warning(f"Group column {task.group_column} not found in metadata")
                return None
            
            group_counts = metadata_aligned[task.group_column].value_counts()
            if not test_config.supports_multigroup and len(group_counts) != 2:
                logger.warning(f"Test {task.test_name} requires exactly 2 groups")
                return None
            
            # Execute test based on type
            result = self._execute_test(
                test_config, table_aligned, metadata_aligned, 
                task.group_column, group_column_values
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return None

    def _execute_test(
        self, 
        test_config: TestConfig, 
        table: Table, 
        metadata: pd.DataFrame,
        group_column: str, 
        group_column_values: List[Any]
    ) -> pd.DataFrame:
        """Execute specific statistical test with optimized parameter handling."""
        
        # Special handling for different test types
        if test_config.key in ['enhanced', 'diffabund']:
            return test_config.func(
                table=table,
                metadata=metadata,
                group_column=group_column
            )
        
        elif test_config.key == 'network':
            # Network analysis returns tuple
            if hasattr(test_config.func, '__call__'):
                corr_matrix, edges_df = test_config.func(table=table)
                return edges_df
            
        elif test_config.requires_continuous:
            # Handle continuous variable tests
            continuous_vars = self.config.get('stats', {}).get('continuous_variables', [])
            if not continuous_vars:
                logger.warning("No continuous variables configured for correlation analysis")
                return pd.DataFrame()
            
            # Use first available continuous variable
            for var in continuous_vars:
                if var in metadata.columns:
                    return test_config.func(
                        table=table,
                        metadata=metadata,
                        continuous_column=var
                    )
            
            logger.warning("No valid continuous variables found in metadata")
            return pd.DataFrame()
            
        else:
            # Standard statistical tests
            return test_config.func(
                table=table,
                metadata=metadata,
                group_column=group_column,
                group_column_values=group_column_values
            )

    def _save_result(self, task: AnalysisTask, result: pd.DataFrame) -> None:
        """Save analysis result to file."""
        if result is None or result.empty:
            return
        
        try:
            output_dir = create_output_directory(
                self.project_dir / 'final' / 'stats',
                task.group_column, task.table_type, task.level
            )
            output_path = output_dir / f'{task.test_name}.tsv'
            result.to_csv(output_path, sep='\t', index=True)
            
            # Save correlation matrix separately for network analysis
            if task.test_name == 'network_analysis':
                # This would need to be handled in _execute_test to return both
                pass
                
        except Exception as e:
            logger.error(f"Failed to save result for {task.test_name}: {e}")

    def run_core_microbiome_analysis(
        self, 
        prevalence_threshold: float = 0.8, 
        abundance_threshold: float = 0.01
    ) -> Dict:
        """Optimized core microbiome analysis with parallel processing."""
        start_time = time.time()
        core_results = {}
        
        # Create tasks for parallel execution
        tasks = []
        for group_column in self.group_columns:
            col = group_column['name']
            for table_type in self.tables:
                for level in self.tables[table_type]:
                    tasks.append((col, table_type, level))
        
        # Execute in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            with get_progress_bar() as progress:
                main_desc = "Running core microbiome analysis"
                main_task = progress.add_task(_format_task_desc(main_desc), total=len(tasks))
                
                # Submit tasks
                future_to_task = {}
                for col, table_type, level in tasks:
                    future = executor.submit(
                        self._core_microbiome_single_task,
                        col, table_type, level, prevalence_threshold, abundance_threshold
                    )
                    future_to_task[future] = (col, table_type, level)
                
                # Collect results
                for future in as_completed(future_to_task):
                    col, table_type, level = future_to_task[future]
                    try:
                        core_features = future.get()
                        if core_features is not None:
                            _init_dict_level(core_results, col, table_type, level)
                            core_results[col][table_type][level] = core_features
                            
                            # Save results
                            self._save_core_microbiome_results(col, table_type, level, core_features)
                            
                    except Exception as e:
                        logger.error(f"Core microbiome analysis failed for {col}/{table_type}/{level}: {e}")
                    finally:
                        progress.update(main_task, advance=1)
        
        self.advanced_results['core_microbiome'] = core_results
        self.execution_times['core_microbiome'] = time.time() - start_time
        return core_results

    def _core_microbiome_single_task(
        self, col: str, table_type: str, level: str,
        prevalence_threshold: float, abundance_threshold: float
    ) -> Optional[Dict]:
        """Execute single core microbiome analysis task."""
        try:
            table = self.tables[table_type][level]
            metadata = self.metadata[table_type][level]
            table_aligned, metadata_aligned = update_table_and_metadata(table, metadata)
            
            core_features = core_microbiome(
                table=table_aligned,
                metadata=metadata_aligned,
                group_column=col,
                prevalence_threshold=prevalence_threshold,
                abundance_threshold=abundance_threshold
            )
            
            return core_features
        except Exception as e:
            logger.error(f"Core microbiome task failed: {e}")
            return None

    def _save_core_microbiome_results(
        self, col: str, table_type: str, level: str, core_features: Dict
    ) -> None:
        """Save core microbiome results."""
        try:
            output_dir = create_output_directory(
                self.project_dir / 'final' / 'core_microbiome',
                col, table_type, level
            )
            
            for group, core_df in core_features.items():
                if isinstance(core_df, pd.DataFrame):
                    output_path = output_dir / f'core_features_{group}.tsv'
                    core_df.to_csv(output_path, sep='\t', index=False)
        except Exception as e:
            logger.error(f"Failed to save core microbiome results: {e}")

    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics."""
        total_time = self.execution_times.get('total_analysis', 0)
        
        stats = {
            'total_execution_time': total_time,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'execution_times': self.execution_times,
            'parallel_workers_used': self.max_workers,
            'total_tasks_completed': sum(
                len(level_data) for group_data in self.results.values()
                for table_data in group_data.values()
                for level_data in table_data.values()
            )
        }
        
        if total_time > 0:
            stats['tasks_per_second'] = stats['total_tasks_completed'] / total_time
        
        return stats

    def validate_configuration(self) -> Dict[str, List[str]]:
        """Enhanced configuration validation with performance checks."""
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check memory requirements
        total_features = sum(
            len(table_to_df(table).columns)
            for table_dict in self.tables.values()
            for table in table_dict.values()
        )
        
        if total_features > 10000:
            issues['warnings'].append(
                f"High feature count ({total_features}) may require significant memory"
            )
        
        # Check parallel processing suitability
        total_tasks = len(get_enabled_tasks(self.config, self.tables)) * len(self.group_columns)
        if total_tasks < self.max_workers:
            issues['info'].append(
                f"Few tasks ({total_tasks}) relative to workers ({self.max_workers}) - "
                "consider reducing max_workers"
            )
        
        # Validate test configurations
        for test_name in self.config.get('stats', {}).get('enabled_tests', []):
            if test_name not in TEST_CONFIGS:
                issues['errors'].append(f"Unknown test configuration: {test_name}")
        
        # Check group column validity with sample size warnings
        for group_column in self.group_columns:
            col_name = group_column['name']
            for table_type in self.metadata:
                for level in self.metadata[table_type]:
                    metadata = self.metadata[table_type][level]
                    if col_name in metadata.columns:
                        group_sizes = metadata[col_name].value_counts()
                        min_size = group_sizes.min()
                        
                        if min_size < 3:
                            issues['errors'].append(
                                f"Group column '{col_name}' has groups with <3 samples"
                            )
                        elif min_size < 10:
                            issues['warnings'].append(
                                f"Group column '{col_name}' has small groups (min: {min_size})"
                            )
        
        return issues

    def export_comprehensive_report(self, output_path: Union[str, Path]) -> None:
        """Export comprehensive analysis report with performance metrics."""
        performance_stats = self.get_performance_stats()
        summary_stats = self.get_summary_statistics()
        
        report_lines = [
            "# Comprehensive Statistical Analysis Report",
            f"**Analysis completed in:** {performance_stats['total_execution_time']:.2f} seconds",
            f"**Cache hit rate:** {performance_stats['cache_hit_rate']:.2%}",
            f"**Parallel workers:** {performance_stats['parallel_workers_used']}",
            f"**Tasks per second:** {performance_stats.get('tasks_per_second', 0):.2f}",
            "",
            "## Analysis Summary",
            f"- Total tests executed: {summary_stats['total_tests_run']}",
            f"- Group columns analyzed: {', '.join(summary_stats['group_columns_analyzed'])}",
            f"- Total significant features: {sum(summary_stats['significant_features_by_test'].values())}",
            "",
            "## Performance Breakdown",
        ]
        
        for analysis_type, exec_time in performance_stats['execution_times'].items():
            report_lines.append(f"- {analysis_type}: {exec_time:.2f}s")
        
        report_lines.extend([
            "",
            "## Test Results Summary",
        ])
        
        for test_name, count in summary_stats['significant_features_by_test'].items():
            test_display_name = TEST_CONFIGS.get(test_name, {}).name or test_name
            report_lines.append(f"- {test_display_name}: {count} significant features")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Comprehensive report exported to {output_path}")

    # Additional optimized methods would go here...
    def get_summary_statistics(self) -> Dict:
        """Generate optimized summary statistics."""
        summary = {
            'total_tests_run': 0,
            'significant_features_by_test': {},
            'effect_sizes_summary': {},
            'group_columns_analyzed': list(self.results.keys())
        }
        
        # Use vectorized operations where possible
        for group_col, group_results in self.results.items():
            for table_type, levels in group_results.items():
                for level, tests in levels.items():
                    for test_name, result in tests.items():
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            summary['total_tests_run'] += 1
                            
                            # Count significant features
                            test_count = summary['significant_features_by_test'].get(test_name, 0)
                            summary['significant_features_by_test'][test_name] = test_count + len(result)
                            
                            # Efficient effect size calculation
                            test_config = TEST_CONFIGS.get(test_name)
                            if test_config and test_config.effect_col in result.columns:
                                effects = result[test_config.effect_col].dropna()
                                if len(effects) > 0:
                                    summary['effect_sizes_summary'][test_name] = {
                                        'mean': float(effects.mean()),
                                        'std': float(effects.std()),
                                        'min': float(effects.min()),
                                        'max': float(effects.max())
                                    }
        
        return summary
