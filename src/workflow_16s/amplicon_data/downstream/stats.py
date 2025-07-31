# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

# Third‑Party Imports
import pandas as pd
import numpy as np
from biom.table import Table
from scipy.stats import (
    ttest_ind, mannwhitneyu, kruskal, f_oneway, fisher_exact, 
    spearmanr, shapiro, levene
)
from statsmodels.stats.multitest import multipletests

from sklearn.cluster import KMeans
from tqdm import tqdm

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.downstream.input import update_table_and_metadata
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence, table_to_df,
    merge_table_with_meta
)
from workflow_16s.utils.io import export_h5py
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ==================================== FUNCTIONS ===================================== #

def _init_dict_level(dictionary: Dict, *keys) -> None:
    """Initialize nested dictionary levels."""
    current = dictionary
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    if keys[-1] not in current:
        current[keys[-1]] = {}

def get_enabled_tasks(
    config: Dict, 
    tables: Dict[str, Dict[str, Table]]
):
    # Configuration setup
    KNOWN_TESTS = {'fisher', 'ttest', 'mwu_bonferroni', 'kruskal_bonferroni', 
                   'enhanced_stats', 'differential_abundance', 'anova', 
                   'spearman_correlation', 'network_analysis'}
    DEFAULT_TESTS = {
        "raw": ["ttest"],
        "filtered": ['mwu_bonferroni', 'kruskal_bonferroni'],
        "normalized": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
        "clr_transformed": ['ttest', 'mwu_bonferroni', 'kruskal_bonferroni'],
        "presence_absence": ["fisher"]
    }
    
    stats_config = config.get('stats', {})
    table_config = stats_config.get('tables', {})

    tasks = []
    for table_type, levels in tables.items():
        table_type_config = table_config.get(table_type, {})
        if not table_type_config.get('enabled', False):
            continue

        enabled_levels = [
            l for l in table_type_config.get('levels', levels.keys()) 
            if l in levels
        ]
        enabled_tests = [
            t for t in table_type_config.get('tests', DEFAULT_TESTS[table_type]) 
            if t in KNOWN_TESTS
        ]

        for level in enabled_levels:
            for test in enabled_tests:
                tasks.append((table_type, level, test))
    return tasks

def validate_inputs(
    table: Union[Dict, Any, pd.DataFrame],
    metadata: Optional[pd.DataFrame] = None,
    group_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Validate and standardize inputs for analysis functions."""
    df = table_to_df(table)
    
    if df.empty:
        raise ValueError("Input table is empty")
    
    if df.isnull().all().all():
        raise ValueError("Input table contains only null values")
    
    if metadata is not None:
        if not isinstance(metadata, pd.DataFrame):
            raise ValueError("Metadata must be a pandas DataFrame")
        
        if group_column and group_column not in metadata.columns:
            raise ValueError(f"Group column '{group_column}' not found in metadata")
        
        common_samples = df.index.intersection(metadata.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples between table and metadata")
        
        if len(common_samples) < len(df.index) * 0.5:
            warnings.warn(
                f"Only {len(common_samples)}/{len(df.index)} samples have metadata. "
                "Consider checking sample ID matching."
            )
    
    return df, metadata

def enhanced_statistical_tests(
    table: Union[Dict, Table, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    test_type: str = 'auto',
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05,
    effect_size_threshold: float = 0.5
) -> pd.DataFrame:
    """Enhanced statistical testing with automatic test selection and effect sizes."""
    df, metadata = validate_inputs(table, metadata, group_column)
    merged = merge_table_with_meta(df, metadata, group_column)
    
    groups = merged[group_column].unique()
    n_groups = len(groups)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for comparison")
    
    results = []
    
    for feature in tqdm(df.columns, desc="Statistical testing"):
        group_data = []
        for group in groups:
            data = merged[merged[group_column] == group][feature].dropna()
            if len(data) >= 3:
                group_data.append(data)
        
        if len(group_data) < 2:
            continue
        
        # Test for normality and equal variances if auto mode
        normality_ok = True
        equal_var_ok = True
        
        if test_type == 'auto':
            for data in group_data:
                if len(data) < 50:
                    _, p_norm = shapiro(data)
                    if p_norm < 0.05:
                        normality_ok = False
                        break
            
            if normality_ok:
                _, p_levene = levene(*group_data)
                if p_levene < 0.05:
                    equal_var_ok = False
        
        # Choose appropriate test
        if n_groups == 2:
            if test_type == 'parametric' or (test_type == 'auto' and normality_ok):
                stat, p_val = ttest_ind(*group_data, equal_var=equal_var_ok)
                test_name = "Welch's t-test" if not equal_var_ok else "Student's t-test"
                
                pooled_std = np.sqrt((np.var(group_data[0], ddof=1) + 
                                    np.var(group_data[1], ddof=1)) / 2)
                effect_size = (np.mean(group_data[0]) - np.mean(group_data[1])) / pooled_std
                
            else:
                stat, p_val = mannwhitneyu(*group_data, alternative='two-sided')
                test_name = "Mann-Whitney U"
                
                n1, n2 = len(group_data[0]), len(group_data[1])
                effect_size = 1 - (2 * stat) / (n1 * n2)
        
        else:
            if test_type == 'parametric' or (test_type == 'auto' and normality_ok):
                stat, p_val = f_oneway(*group_data)
                test_name = "One-way ANOVA"
                
                all_data = np.concatenate(group_data)
                grand_mean = np.mean(all_data)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
                ss_total = sum((x - grand_mean)**2 for x in all_data)
                effect_size = ss_between / ss_total if ss_total > 0 else 0
                
            else:
                stat, p_val = kruskal(*group_data)
                test_name = "Kruskal-Wallis"
                
                n_total = sum(len(g) for g in group_data)
                effect_size = stat / (n_total - 1) if n_total > 1 else 0
        
        means = [np.mean(g) for g in group_data]
        medians = [np.median(g) for g in group_data]
        
        results.append({
            'feature': feature,
            'test': test_name,
            'statistic': stat,
            'p_value': p_val,
            'effect_size': effect_size,
            'mean_values': means,
            'median_values': medians,
            'n_groups': len(group_data),
            'total_samples': sum(len(g) for g in group_data)
        })
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    _, p_adj, _, _ = multipletests(results_df['p_value'], method=correction_method)
    results_df['p_adj'] = p_adj
    
    significant = results_df[
        (results_df['p_adj'] <= alpha) & 
        (np.abs(results_df['effect_size']) >= effect_size_threshold)
    ]
    
    return significant.sort_values('p_adj')

def differential_abundance_analysis(
    table: Union[Dict, Any, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    method: str = 'deseq2_like',
    alpha: float = 0.05,
    fold_change_threshold: float = 1.5,
    min_prevalence: float = 0.1
) -> pd.DataFrame:
    """Comprehensive differential abundance analysis with multiple methods."""
    from scipy.stats import ttest_ind, mannwhitneyu
    
    df, metadata = validate_inputs(table, metadata, group_column)
    merged = merge_table_with_meta(df, metadata, group_column)
    
    # Filter by prevalence
    prevalence = (df > 0).mean()
    df_filt = df.loc[:, prevalence >= min_prevalence]
    merged_filt = merge_table_with_meta(df_filt, metadata, group_column)
    
    groups = merged_filt[group_column].unique()
    if len(groups) != 2:
        raise ValueError("Differential abundance analysis requires exactly 2 groups")
    
    results = []
    
    for feature in tqdm(df_filt.columns, desc=f"DA analysis ({method})"):
        group1_data = merged_filt[merged_filt[group_column] == groups[0]][feature]
        group2_data = merged_filt[merged_filt[group_column] == groups[1]][feature]
        
        if len(group1_data) < 3 or len(group2_data) < 3:
            continue
        
        # Calculate fold change
        mean1 = group1_data.mean()
        mean2 = group2_data.mean()
        fold_change = (mean1 + 1e-8) / (mean2 + 1e-8)
        log2_fc = np.log2(fold_change)
        
        if abs(log2_fc) < np.log2(fold_change_threshold):
            continue
        
        # Statistical testing
        if method == 'wilcoxon':
            statistic, p_value = mannwhitneyu(group1_data, group2_data)
        elif method == 'ttest':
            statistic, p_value = ttest_ind(group1_data, group2_data)
        elif method == 'deseq2_like':
            statistic, p_value = _deseq2_like_test(group1_data, group2_data)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        results.append({
            'feature': feature,
            'log2_fold_change': log2_fc,
            'fold_change': fold_change,
            'mean_group1': mean1,
            'mean_group2': mean2,
            'statistic': statistic,
            'p_value': p_value,
            'prevalence': prevalence[feature]
        })
    
    if not results:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    
    _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['p_adj'] = p_adj
    
    significant = results_df[results_df['p_adj'] <= alpha]
    significant = significant.sort_values('p_adj')
    
    return significant

def _deseq2_like_test(group1: pd.Series, group2: pd.Series) -> Tuple[float, float]:
    """Simplified DESeq2-like differential expression test."""
    def variance_stabilize(x):
        return np.arcsinh(x)
    
    vs1 = variance_stabilize(group1)
    vs2 = variance_stabilize(group2)
    
    return ttest_ind(vs1, vs2)

def core_microbiome(
    table: Union[Dict, Any, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    prevalence_threshold: float = 0.8,
    abundance_threshold: float = 0.01
) -> Dict[str, pd.DataFrame]:
    """Identify core microbiome for each group."""
    df, metadata = validate_inputs(table, metadata, group_column)
    merged = merge_table_with_meta(df, metadata, group_column)
    
    rel_abundance = df.div(df.sum(axis=1), axis=0)
    merged_rel = merge_table_with_meta(rel_abundance, metadata, group_column)
    
    core_features = {}
    
    for group in merged[group_column].unique():
        group_data = merged_rel[merged_rel[group_column] == group]
        group_features = group_data.drop(columns=[group_column])
        
        prevalence = (group_features > 0).mean()
        mean_abundance = group_features.mean()
        
        core_mask = (
            (prevalence >= prevalence_threshold) & 
            (mean_abundance >= abundance_threshold)
        )
        
        core_df = pd.DataFrame({
            'feature': core_mask.index[core_mask],
            'prevalence': prevalence[core_mask],
            'mean_abundance': mean_abundance[core_mask],
            'group': group
        })
        
        core_features[group] = core_df.sort_values('mean_abundance', ascending=False)
    
    return core_features

def microbial_network_analysis(
    table: Union[Dict, Table, pd.DataFrame],
    method: str = 'sparcc',
    threshold: float = 0.3,
    min_prevalence: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct microbial co-occurrence networks."""
    df, _ = validate_inputs(table)
    
    prevalence = (df > 0).mean()
    df_filt = df.loc[:, prevalence >= min_prevalence]
    
    logger.info(f"Network analysis: {df_filt.shape[1]} features after prevalence filtering")
    
    if method == 'sparcc':
        corr_matrix = df_filt.corr(method='spearman')
    elif method in ['spearman', 'pearson']:
        corr_matrix = df_filt.corr(method=method)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    edges = []
    n_features = len(corr_matrix)
    
    for i in range(n_features):
        for j in range(i + 1, n_features):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                edges.append({
                    'source': corr_matrix.index[i],
                    'target': corr_matrix.index[j],
                    'correlation': corr_val,
                    'abs_correlation': abs(corr_val),
                    'edge_type': 'positive' if corr_val > 0 else 'negative'
                })
    
    edges_df = pd.DataFrame(edges).sort_values('abs_correlation', ascending=False)
    
    return corr_matrix, edges_df

from workflow_16s.stats.tests import (
    anova, fisher_exact_bonferroni, kruskal_bonferroni, mwu_bonferroni, ttest,
    spearman_correlation
)

def get_group_column_values(group_column, metadata):
    """Your existing get_group_column_values function"""
    col = group_column['name']
    if 'values' in group_column and group_column['values']:
        return group_column['values']
    elif group_column['type'] == 'bool':
        return [True, False]
    else:
        if group_column['name'] in metadata.columns:
            return metadata[group_column['name']].unique()
        else:
            return None

class StatisticalAnalysis:
    """Enhanced Statistical Analysis class with integrated advanced functions."""
    
    TestConfig = {
        "fisher": {
            "key": "fisher",
            "func": fisher_exact_bonferroni,
            "name": "Fisher exact (Bonferroni)",
            "effect_col": "proportion_diff",
            "alt_effect_col": "odds_ratio",
        },
        "ttest": {
            "key": "ttest",
            "func": ttest,
            "name": "Student t‑test",
            "effect_col": "mean_difference",
            "alt_effect_col": "cohens_d",
        },
        "mwu_bonferroni": {
            "key": "mwub",
            "func": mwu_bonferroni,
            "name": "Mann–Whitney U (Bonferroni)",
            "effect_col": "effect_size_r",
            "alt_effect_col": "median_difference",
        },
        "kruskal_bonferroni": {
            "key": "kwb",
            "func": kruskal_bonferroni,
            "name": "Kruskal–Wallis (Bonferroni)",
            "effect_col": "epsilon_squared",
            "alt_effect_col": None,
        },
        "enhanced_stats": {
            "key": "enhanced",
            "func": enhanced_statistical_tests,
            "name": "Enhanced Statistical Tests",
            "effect_col": "effect_size",
            "alt_effect_col": None,
        },
        "differential_abundance": {
            "key": "diffabund",
            "func": differential_abundance_analysis,
            "name": "Differential Abundance Analysis",
            "effect_col": "log2_fold_change",
            "alt_effect_col": "fold_change",
        },
        "anova": {
            "key": "anova",
            "func": anova,
            "name": "One-way ANOVA",
            "effect_col": "eta_squared",
            "alt_effect_col": None,
        },
        "spearman_correlation": {
            "key": "spearman",
            "func": spearman_correlation,
            "name": "Spearman Correlation",
            "effect_col": "rho",
            "alt_effect_col": None,
        },
        "network_analysis": {
            "key": "network",
            "func": microbial_network_analysis,
            "name": "Network Analysis",
            "effect_col": "correlation",
            "alt_effect_col": "abs_correlation",
        }
    }
    
    def __init__(
        self,
        config: Dict,
        tables: Dict,
        metadata: Dict,
        mode: str,
        group_columns: List,
        project_dir: Union[str, Path]
    ) -> None:
        self.config, self.project_dir, self.mode = config, project_dir, mode
        self.tables, self.metadata = tables, metadata
        self.group_columns = group_columns 
        
        # Add NFC facilities if enabled
        if (self.config.get("nfc_facilities", {}).get('enabled', False) and 
            'facility_match' in self.metadata["raw"]["genus"].columns):
            self.group_columns.append({
                'name': 'nfc_facilities', 
                'type': 'bool', 
                'values': [True, False]
            })
            
        self.results: Dict = {}
        self.advanced_results: Dict = {}  # Store advanced analysis results
        
        # Run analysis for each group column
        for group_column in self.group_columns:
            col = group_column['name']
            vals = get_group_column_values(group_column, self.metadata["raw"]["genus"])
            print(f"Processing group column: {col}")
            print(f"Values: {vals}")
            self.results[col] = self._run_for_group(col, vals)

    def _run_for_group(self, group_column: str, group_column_values: List[Any]):
        """Run statistical analysis for a specific group column."""
        tasks = get_enabled_tasks(self.config, self.tables)
        if not tasks:
            return {}
        
        group_stats = {}
        
        with get_progress_bar() as progress:
            stats_desc = f"Running statistics for '{group_column}'"
            stats_task = progress.add_task(_format_task_desc(stats_desc), total=len(tasks))
    
            for table_type, level, test in tasks:
                test_desc = (
                    f"{table_type.replace('_', ' ').title()} ({level.title()})"
                    f" → {self.TestConfig[test]['name']}"
                )
                progress.update(stats_task, description=_format_task_desc(test_desc))
    
                # Initialize data storage
                _init_dict_level(group_stats, table_type, level)
                data_storage = group_stats[table_type][level]
                
                # Initialize output directory and path
                output_dir = self.project_dir.final / 'stats' / group_column / table_type / level
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f'{test}.tsv'
    
                try:
                    # Prepare data
                    table = self.tables[table_type][level]
                    metadata = self.metadata[table_type][level]
                    table_aligned, metadata_aligned = update_table_and_metadata(table, metadata)
                    
                    # Run statistical test
                    test_func = self.TestConfig[test]["func"]
                    
                    # Handle different function signatures
                    if test in ['enhanced_stats', 'differential_abundance']:
                        result = test_func(
                            table=table_aligned,
                            metadata=metadata_aligned,
                            group_column=group_column
                        )
                    elif test == 'network_analysis':
                        # Network analysis returns tuple, handle differently
                        corr_matrix, edges_df = test_func(table=table_aligned)
                        result = edges_df
                        # Save correlation matrix separately
                        corr_path = output_dir / f'{test}_correlation_matrix.tsv'
                        corr_matrix.to_csv(corr_path, sep='\t')
                    elif test == 'spearman_correlation':
                        # Skip if no continuous variables configured
                        continuous_vars = self.config.get('stats', {}).get('continuous_variables', [])
                        if not continuous_vars:
                            continue
                        # Use first continuous variable (you might want to iterate through all)
                        result = test_func(
                            table=table_aligned,
                            metadata=metadata_aligned,
                            continuous_column=continuous_vars[0]
                        )
                    else:
                        # Standard statistical tests
                        result = test_func(
                            table=table_aligned,
                            metadata=metadata_aligned,
                            group_column=group_column,
                            group_column_values=group_column_values
                        )
                    
                    # Store and save results
                    data_storage[test] = result
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        result.to_csv(output_path, sep='\t', index=True)
                    
                except Exception as e:
                    logger.error(f"Test '{test}' failed for {table_type}/{level}: {str(e)}")
                    data_storage[test] = None
                    
                finally:
                    progress.update(stats_task, advance=1)
                    
            progress.update(stats_task, description=_format_task_desc(stats_desc))
            
        return group_stats

    def run_core_microbiome_analysis(self, prevalence_threshold: float = 0.8, 
                                   abundance_threshold: float = 0.01) -> Dict:
        """Run core microbiome analysis for all groups."""
        core_results = {}
        
        for group_column in self.group_columns:
            col = group_column['name']
            core_results[col] = {}
            
            for table_type in self.tables:
                for level in self.tables[table_type]:
                    table = self.tables[table_type][level]
                    metadata = self.metadata[table_type][level]
                    table_aligned, metadata_aligned = update_table_and_metadata(table, metadata)
                    
                    try:
                        core_features = core_microbiome(
                            table=table_aligned,
                            metadata=metadata_aligned,
                            group_column=col,
                            prevalence_threshold=prevalence_threshold,
                            abundance_threshold=abundance_threshold
                        )
                        
                        _init_dict_level(core_results, col, table_type, level)
                        core_results[col][table_type][level] = core_features
                        
                        # Save results
                        output_dir = self.project_dir.final / 'core_microbiome' / col / table_type / level
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        for group, core_df in core_features.items():
                            output_path = output_dir / f'core_features_{group}.tsv'
                            core_df.to_csv(output_path, sep='\t', index=False)
                            
                    except Exception as e:
                        logger.error(f"Core microbiome analysis failed for {col}/{table_type}/{level}: {e}")
        
        self.advanced_results['core_microbiome'] = core_results
        return core_results

    def get_effect_size(self, test_name: str, row: pd.Series) -> Optional[float]:
        """Get effect size from test results."""
        if test_name not in self.TestConfig:
            return None
        test_config = self.TestConfig[test_name]
        for col in (test_config["effect_col"], test_config["alt_effect_col"]):
            if col and col in row:
                return row[col]
        return None

    def get_summary_statistics(self) -> Dict:
        """Generate summary statistics across all analyses."""
        summary = {
            'total_tests_run': 0,
            'significant_features_by_test': {},
            'effect_sizes_summary': {},
            'group_columns_analyzed': list(self.results.keys())
        }
        
        for group_col, group_results in self.results.items():
            for table_type, levels in group_results.items():
                for level, tests in levels.items():
                    for test_name, result in tests.items():
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            summary['total_tests_run'] += 1
                            
                            # Count significant features
                            if test_name not in summary['significant_features_by_test']:
                                summary['significant_features_by_test'][test_name] = 0
                            summary['significant_features_by_test'][test_name] += len(result)
                            
                            # Effect size summary
                            effect_col = self.TestConfig.get(test_name, {}).get('effect_col')
                            if effect_col and effect_col in result.columns:
                                if test_name not in summary['effect_sizes_summary']:
                                    summary['effect_sizes_summary'][test_name] = {
                                        'mean': 0, 'std': 0, 'min': 0, 'max': 0
                                    }
                                
                                effects = result[effect_col].dropna()
                                if len(effects) > 0:
                                    summary['effect_sizes_summary'][test_name] = {
                                        'mean': float(effects.mean()),
                                        'std': float(effects.std()),
                                        'min': float(effects.min()),
                                        'max': float(effects.max())
                                    }
        
        return summary

    def export_results_summary(self, output_path: Union[str, Path]) -> None:
        """Export a comprehensive summary of all results."""
        summary = self.get_summary_statistics()
        
        # Create summary report
        report_lines = [
            "# Statistical Analysis Summary Report",
            f"Total tests executed: {summary['total_tests_run']}",
            f"Group columns analyzed: {', '.join(summary['group_columns_analyzed'])}",
            "",
            "## Significant Features by Test Type",
        ]
        
        for test_name, count in summary['significant_features_by_test'].items():
            test_display_name = self.TestConfig.get(test_name, {}).get('name', test_name)
            report_lines.append(f"- {test_display_name}: {count} significant features")
        
        report_lines.extend([
            "",
            "## Effect Size Summaries",
        ])
        
        for test_name, stats in summary['effect_sizes_summary'].items():
            test_display_name = self.TestConfig.get(test_name, {}).get('name', test_name)
            report_lines.extend([
                f"### {test_display_name}",
                f"- Mean effect size: {stats['mean']:.4f}",
                f"- Standard deviation: {stats['std']:.4f}",
                f"- Range: {stats['min']:.4f} to {stats['max']:.4f}",
                ""
            ])
        
        # Write summary report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Summary report exported to {output_path}")

    def run_batch_correlation_analysis(self, continuous_variables: List[str]) -> Dict:
        """Run correlation analysis for multiple continuous variables."""
        correlation_results = {}
        
        for var in continuous_variables:
            correlation_results[var] = {}
            
            for table_type in self.tables:
                for level in self.tables[table_type]:
                    # Check if variable exists in metadata
                    metadata = self.metadata[table_type][level]
                    if var not in metadata.columns:
                        continue
                    
                    table = self.tables[table_type][level]
                    table_aligned, metadata_aligned = update_table_and_metadata(table, metadata)
                    
                    try:
                        result = spearman_correlation(
                            table=table_aligned,
                            metadata=metadata_aligned,
                            continuous_column=var
                        )
                        
                        _init_dict_level(correlation_results, var, table_type, level)
                        correlation_results[var][table_type][level] = result
                        
                        # Save results
                        output_dir = self.project_dir.final / 'correlations' / var / table_type / level
                        output_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_dir / 'spearman_correlations.tsv'
                        result.to_csv(output_path, sep='\t', index=False)
                        
                    except Exception as e:
                        logger.error(f"Correlation analysis failed for {var}/{table_type}/{level}: {e}")
        
        self.advanced_results['correlations'] = correlation_results
        return correlation_results

    def run_network_analysis_batch(self, methods: List[str] = ['sparcc', 'spearman'], 
                                 threshold: float = 0.3) -> Dict:
        """Run network analysis for multiple correlation methods."""
        network_results = {}
        
        for method in methods:
            network_results[method] = {}
            
            for table_type in self.tables:
                for level in self.tables[table_type]:
                    table = self.tables[table_type][level]
                    table_aligned, _ = update_table_and_metadata(
                        table, self.metadata[table_type][level]
                    )
                    
                    try:
                        corr_matrix, edges_df = microbial_network_analysis(
                            table=table_aligned,
                            method=method,
                            threshold=threshold
                        )
                        
                        _init_dict_level(network_results, method, table_type, level)
                        network_results[method][table_type][level] = {
                            'correlation_matrix': corr_matrix,
                            'edges': edges_df
                        }
                        
                        # Save results
                        output_dir = self.project_dir.final / 'networks' / method / table_type / level
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        corr_path = output_dir / 'correlation_matrix.tsv'
                        edges_path = output_dir / 'network_edges.tsv'
                        
                        corr_matrix.to_csv(corr_path, sep='\t')
                        edges_df.to_csv(edges_path, sep='\t', index=False)
                        
                        # Generate network statistics
                        network_stats = self._calculate_network_statistics(edges_df)
                        stats_path = output_dir / 'network_statistics.tsv'
                        pd.DataFrame([network_stats]).to_csv(stats_path, sep='\t', index=False)
                        
                    except Exception as e:
                        logger.error(f"Network analysis failed for {method}/{table_type}/{level}: {e}")
        
        self.advanced_results['networks'] = network_results
        return network_results

    def _calculate_network_statistics(self, edges_df: pd.DataFrame) -> Dict:
        """Calculate basic network statistics from edge list."""
        if edges_df.empty:
            return {
                'total_edges': 0,
                'positive_edges': 0,
                'negative_edges': 0,
                'mean_correlation': 0,
                'unique_nodes': 0
            }
        
        total_edges = len(edges_df)
        positive_edges = (edges_df['correlation'] > 0).sum()
        negative_edges = (edges_df['correlation'] < 0).sum()
        mean_correlation = edges_df['correlation'].mean()
        
        # Count unique nodes
        unique_nodes = len(
            set(edges_df['source'].tolist() + edges_df['target'].tolist())
        )
        
        return {
            'total_edges': total_edges,
            'positive_edges': positive_edges,
            'negative_edges': negative_edges,
            'mean_correlation': mean_correlation,
            'unique_nodes': unique_nodes,
            'density': total_edges / (unique_nodes * (unique_nodes - 1) / 2) if unique_nodes > 1 else 0
        }

    def run_comprehensive_analysis(self, **kwargs) -> Dict:
        """Run all available advanced analyses."""
        comprehensive_results = {}
        
        logger.info("Starting comprehensive statistical analysis...")
        
        # 1. Core microbiome analysis
        logger.info("Running core microbiome analysis...")
        try:
            core_results = self.run_core_microbiome_analysis(
                prevalence_threshold=kwargs.get('prevalence_threshold', 0.8),
                abundance_threshold=kwargs.get('abundance_threshold', 0.01)
            )
            comprehensive_results['core_microbiome'] = core_results
        except Exception as e:
            logger.error(f"Core microbiome analysis failed: {e}")
        
        # 2. Correlation analysis
        continuous_vars = kwargs.get('continuous_variables', 
                                   self.config.get('stats', {}).get('continuous_variables', []))
        if continuous_vars:
            logger.info("Running correlation analysis...")
            try:
                correlation_results = self.run_batch_correlation_analysis(continuous_vars)
                comprehensive_results['correlations'] = correlation_results
            except Exception as e:
                logger.error(f"Correlation analysis failed: {e}")
        
        # 3. Network analysis
        logger.info("Running network analysis...")
        try:
            network_methods = kwargs.get('network_methods', ['sparcc', 'spearman'])
            network_threshold = kwargs.get('network_threshold', 0.3)
            network_results = self.run_network_analysis_batch(
                methods=network_methods,
                threshold=network_threshold
            )
            comprehensive_results['networks'] = network_results
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
        
        # 4. Generate summary
        logger.info("Generating comprehensive summary...")
        try:
            summary_path = self.project_dir.final / 'comprehensive_analysis_summary.md'
            self.export_results_summary(summary_path)
            comprehensive_results['summary_path'] = str(summary_path)
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
        
        # Store all results
        self.advanced_results.update(comprehensive_results)
        
        logger.info("Comprehensive analysis completed!")
        return comprehensive_results

    def get_top_features_across_tests(self, n_features: int = 10) -> pd.DataFrame:
        """Get top features that appear consistently across multiple tests."""
        feature_counts = {}
        feature_effects = {}
        
        for group_col, group_results in self.results.items():
            for table_type, levels in group_results.items():
                for level, tests in levels.items():
                    for test_name, result in tests.items():
                        if isinstance(result, pd.DataFrame) and not result.empty:
                            for _, row in result.iterrows():
                                feature = row.get('feature', '')
                                if feature:
                                    # Count occurrences
                                    if feature not in feature_counts:
                                        feature_counts[feature] = 0
                                        feature_effects[feature] = []
                                    
                                    feature_counts[feature] += 1
                                    
                                    # Store effect sizes
                                    effect_size = self.get_effect_size(test_name, row)
                                    if effect_size is not None:
                                        feature_effects[feature].append(abs(effect_size))
        
        # Create summary DataFrame
        summary_data = []
        for feature, count in feature_counts.items():
            effects = feature_effects[feature]
            summary_data.append({
                'feature': feature,
                'test_count': count,
                'mean_effect_size': np.mean(effects) if effects else 0,
                'max_effect_size': np.max(effects) if effects else 0,
                'effect_size_std': np.std(effects) if len(effects) > 1 else 0
            })
        
        summary_df = pd.DataFrame(summary_data)
        if not summary_df.empty:
            # Sort by test count and mean effect size
            summary_df = summary_df.sort_values(
                ['test_count', 'mean_effect_size'], 
                ascending=[False, False]
            ).head(n_features)
        
        return summary_df

    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate the analysis configuration and return issues."""
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check if tables and metadata are aligned
        for table_type in self.tables:
            for level in self.tables[table_type]:
                try:
                    table = self.tables[table_type][level]
                    metadata = self.metadata[table_type][level]
                    update_table_and_metadata(table, metadata)
                except Exception as e:
                    issues['errors'].append(
                        f"Table/metadata alignment failed for {table_type}/{level}: {e}"
                    )
        
        # Check group columns
        for group_column in self.group_columns:
            col_name = group_column['name']
            found_in_any = False
            
            for table_type in self.metadata:
                for level in self.metadata[table_type]:
                    metadata = self.metadata[table_type][level]
                    if col_name in metadata.columns:
                        found_in_any = True
                        
                        # Check for sufficient group sizes
                        group_sizes = metadata[col_name].value_counts()
                        small_groups = group_sizes[group_sizes < 3]
                        if len(small_groups) > 0:
                            issues['warnings'].append(
                                f"Small group sizes for '{col_name}' in {table_type}/{level}: "
                                f"{dict(small_groups)}"
                            )
            
            if not found_in_any:
                issues['errors'].append(f"Group column '{col_name}' not found in any metadata")
        
        # Check continuous variables if specified
        continuous_vars = self.config.get('stats', {}).get('continuous_variables', [])
        for var in continuous_vars:
            found_in_any = False
            for table_type in self.metadata:
                for level in self.metadata[table_type]:
                    metadata = self.metadata[table_type][level]
                    if var in metadata.columns:
                        found_in_any = True
                        # Check if actually numeric
                        if not pd.api.types.is_numeric_dtype(metadata[var]):
                            issues['warnings'].append(
                                f"Continuous variable '{var}' in {table_type}/{level} "
                                "is not numeric"
                            )
            
            if not found_in_any:
                issues['warnings'].append(
                    f"Continuous variable '{var}' not found in any metadata"
                )
        
        # Configuration completeness
        if not self.config.get('stats', {}).get('tables', {}):
            issues['warnings'].append("No statistical tests configured in config['stats']['tables']")
        
        return issues

    def get_analysis_recommendations(self) -> List[str]:
        """Provide analysis recommendations based on data characteristics."""
        recommendations = []
        
        # Analyze data characteristics
        total_samples = 0
        total_features = 0
        
        for table_type in self.tables:
            for level in self.tables[table_type]:
                table = self.tables[table_type][level]
                df = table_to_df(table)
                total_samples += len(df)
                total_features += len(df.columns)
        
        avg_samples = total_samples / (len(self.tables) * max(1, len(self.tables.get(list(self.tables.keys())[0], {}))))
        avg_features = total_features / (len(self.tables) * max(1, len(self.tables.get(list(self.tables.keys())[0], {}))))
        
        # Sample size recommendations
        if avg_samples < 20:
            recommendations.append(
                "Small sample size detected. Consider using non-parametric tests "
                "(Mann-Whitney U, Kruskal-Wallis) instead of parametric tests."
            )
        
        if avg_samples > 100:
            recommendations.append(
                "Large sample size detected. Both parametric and non-parametric tests "
                "should be reliable. Consider using enhanced statistical tests for "
                "automatic test selection."
            )
        
        # Feature recommendations
        if avg_features > 1000:
            recommendations.append(
                "High-dimensional data detected. Consider using differential abundance "
                "analysis with appropriate multiple testing correction."
            )
        
        # Group structure recommendations
        group_structures = []
        for group_column in self.group_columns:
            col_name = group_column['name']
            for table_type in self.metadata:
                for level in self.metadata[table_type]:
                    metadata = self.metadata[table_type][level]
                    if col_name in metadata.columns:
                        n_groups = metadata[col_name].nunique()
                        group_structures.append(n_groups)
                        break
        
        if any(n > 2 for n in group_structures):
            recommendations.append(
                "Multiple groups detected. Consider using ANOVA or Kruskal-Wallis tests "
                "for overall group differences, followed by post-hoc pairwise comparisons."
            )
        
        # Network analysis recommendations
        if avg_features > 50:
            recommendations.append(
                "Sufficient features for network analysis. Consider running microbial "
                "co-occurrence network analysis to identify feature interactions."
            )
        
        return recommendations
