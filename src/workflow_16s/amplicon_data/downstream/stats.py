# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
from biom.table import Table

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.downstream.input import update_table_and_metadata
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence, table_to_df
)
from workflow_16s.utils.io import export_h5py
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ==================================== FUNCTIONS ===================================== #

def get_enabled_tasks(
    config: Dict, 
    tables: Dict[str, Dict[str, Table]]
):
    # Configuration setup
    KNOWN_TESTS = {'fisher', 'ttest', 'mwu_bonferroni', 'kruskal_bonferroni'}
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


def log_test_results(
    result: pd.DataFrame, 
    table_type: str, 
    level: str, 
    test: str
) -> None:
    """Log statistical test results"""
    sig_mask = result["p_value"] < 0.05
    n_sig = sig_mask.sum()
    
    logger.debug(f"Found {n_sig} significant features for {table_type}/{level}/{test}")
    
    if n_sig == 0:
        logger.debug(f"Top 5 features by p-value ({test}):")
        top_features = result.nsmallest(5, "p_value")
        
        for _, row in top_features.iterrows():
            feat = row.get('feature', 'N/A')
            p_val = row.get('p_value', float('nan'))
            effect = row.get('effect_size', float('nan'))
            logger.debug(f"  {feat}: p={p_val:.3e}, effect={effect:.3f}")
          

def validate_inputs(
    table: Union[Dict, Any, pd.DataFrame],
    metadata: Optional[pd.DataFrame] = None,
    group_column: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Validate and standardize inputs for analysis functions.
    
    Args:
        table: Input abundance table
        metadata: Sample metadata DataFrame
        group_column: Column name for grouping variable
        
    Returns:
        Tuple of (standardized_table, metadata)
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Convert table to DataFrame
    df = table_to_df(table)
    
    # Basic validation
    if df.empty:
        raise ValueError("Input table is empty")
    
    if df.isnull().all().all():
        raise ValueError("Input table contains only null values")
    
    # Validate metadata if provided
    if metadata is not None:
        if not isinstance(metadata, pd.DataFrame):
            raise ValueError("Metadata must be a pandas DataFrame")
        
        if group_column and group_column not in metadata.columns:
            raise ValueError(f"Group column '{group_column}' not found in metadata")
        
        # Check for sample overlap
        common_samples = df.index.intersection(metadata.index)
        if len(common_samples) == 0:
            raise ValueError("No common samples between table and metadata")
        
        if len(common_samples) < len(df.index) * 0.5:
            warnings.warn(
                f"Only {len(common_samples)}/{len(df.index)} samples have metadata. "
                "Consider checking sample ID matching."
            )
    
    return df, metadata


def differential_abundance_analysis(
    table: Union[Dict, Any, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    method: str = 'deseq2_like',
    alpha: float = 0.05,
    fold_change_threshold: float = 1.5,
    min_prevalence: float = 0.1
) -> pd.DataFrame:
    """Comprehensive differential abundance analysis with multiple methods.
    
    Args:
        table: Input abundance table
        metadata: Sample metadata
        group_column: Column containing group labels
        method: Analysis method ('deseq2_like', 'ancom_like', 'wilcoxon', 'ttest')
        alpha: Significance threshold
        fold_change_threshold: Minimum fold change for biological significance
        min_prevalence: Minimum prevalence threshold
        
    Returns:
        DataFrame with differential abundance results
    """
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
        
        # Skip if insufficient data
        if len(group1_data) < 3 or len(group2_data) < 3:
            continue
        
        # Calculate fold change
        mean1 = group1_data.mean()
        mean2 = group2_data.mean()
        fold_change = (mean1 + 1e-8) / (mean2 + 1e-8)  # Add pseudocount
        log2_fc = np.log2(fold_change)
        
        # Skip if fold change is too small
        if abs(log2_fc) < np.log2(fold_change_threshold):
            continue
        
        # Statistical testing
        if method == 'wilcoxon':
            statistic, p_value = mannwhitneyu(group1_data, group2_data)
        elif method == 'ttest':
            statistic, p_value = ttest_ind(group1_data, group2_data)
        elif method == 'deseq2_like':
            # Simplified DESeq2-like analysis with variance stabilization
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
    
    # Multiple testing correction
    _, p_adj, _, _ = multipletests(results_df['p_value'], method='fdr_bh')
    results_df['p_adj'] = p_adj
    
    # Filter significant results
    significant = results_df[results_df['p_adj'] <= alpha]
    significant = significant.sort_values('p_adj')
    
    return significant


def _deseq2_like_test(group1: pd.Series, group2: pd.Series) -> Tuple[float, float]:
    """Simplified DESeq2-like differential expression test."""
    # Variance stabilization (similar to DESeq2's approach)
    def variance_stabilize(x):
        return np.arcsinh(x)
    
    vs1 = variance_stabilize(group1)
    vs2 = variance_stabilize(group2)
    
    # Perform t-test on variance-stabilized data
    return ttest_ind(vs1, vs2)


def core_microbiome(
    table: Union[Dict, Any, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    prevalence_threshold: float = 0.8,
    abundance_threshold: float = 0.01
) -> Dict[str, pd.DataFrame]:
    """Identify core microbiome for each group.
    
    Args:
        table: Input abundance table
        metadata: Sample metadata
        group_column: Column containing group labels
        prevalence_threshold: Minimum prevalence within group
        abundance_threshold: Minimum relative abundance
        
    Returns:
        Dictionary mapping group names to core feature DataFrames
    """
    df, metadata = validate_inputs(table, metadata, group_column)
    merged = merge_table_with_meta(df, metadata, group_column)
    
    # Convert to relative abundance
    rel_abundance = df.div(df.sum(axis=1), axis=0)
    merged_rel = merge_table_with_meta(rel_abundance, metadata, group_column)
    
    core_features = {}
    
    for group in merged[group_column].unique():
        group_data = merged_rel[merged_rel[group_column] == group]
        group_features = group_data.drop(columns=[group_column])
        
        # Calculate prevalence and mean abundance for this group
        prevalence = (group_features > 0).mean()
        mean_abundance = group_features.mean()
        
        # Identify core features
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


def k_means(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    n_clusters: int = constants.DEFAULT_N_CLUSTERS, 
    random_state: int = constants.DEFAULT_RANDOM_STATE,
    verbose: bool = False
) -> pd.Series:
    """Apply K-means clustering and return cluster labels."""
    table = table_to_df(table)
    table_with_column = merge_table_with_meta(table, metadata, group_column)
    
    kmeans = KMeans(
        n_clusters, 
        random_state=random_state
    ).fit(table_with_column.drop(group_column, axis=1))

    results = pd.Series(
        kmeans.labels_, 
        index=table_with_column.index, 
        name='kmeans_cluster'
    )
    return results


def enhanced_statistical_tests(
    table: Union[Dict, Table, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    test_type: str = 'auto',
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05,
    effect_size_threshold: float = 0.5
) -> pd.DataFrame:
    """Enhanced statistical testing with automatic test selection and effect sizes.
    
    Args:
        table: Input abundance table
        metadata: Sample metadata
        group_column: Column containing group labels
        test_type: Statistical test ('auto', 'parametric', 'nonparametric')
        correction_method: Multiple testing correction method
        alpha: Significance threshold
        effect_size_threshold: Minimum effect size for practical significance
        
    Returns:
        DataFrame with comprehensive statistical results
    """
    df, metadata = validate_inputs(table, metadata, group_column)
    merged = merge_table_with_meta(df, metadata, group_column)
    
    groups = merged[group_column].unique()
    n_groups = len(groups)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for comparison")
    
    results = []
    
    for feature in tqdm(df.columns, desc="Statistical testing"):
        # Collect group data
        group_data = []
        for group in groups:
            data = merged[merged[group_column] == group][feature].dropna()
            if len(data) >= 3:  # Minimum sample size
                group_data.append(data)
        
        if len(group_data) < 2:
            continue
        
        # Test for normality and equal variances if auto mode
        normality_ok = True
        equal_var_ok = True
        
        if test_type == 'auto':
            # Shapiro-Wilk test for normality (if n < 50)
            for data in group_data:
                if len(data) < 50:
                    _, p_norm = stats.shapiro(data)
                    if p_norm < 0.05:
                        normality_ok = False
                        break
            
            # Levene's test for equal variances
            if normality_ok:
                _, p_levene = levene(*group_data)
                if p_levene < 0.05:
                    equal_var_ok = False
        
        # Choose appropriate test
        if n_groups == 2:
            if test_type == 'parametric' or (test_type == 'auto' and normality_ok):
                # t-test
                stat, p_val = ttest_ind(*group_data, equal_var=equal_var_ok)
                test_name = "Welch's t-test" if not equal_var_ok else "Student's t-test"
                
                # Cohen's d
                pooled_std = np.sqrt((np.var(group_data[0], ddof=1) + 
                                    np.var(group_data[1], ddof=1)) / 2)
                effect_size = (np.mean(group_data[0]) - np.mean(group_data[1])) / pooled_std
                
            else:
                # Mann-Whitney U test
                stat, p_val = mannwhitneyu(*group_data, alternative='two-sided')
                test_name = "Mann-Whitney U"
                
                # Rank biserial correlation
                n1, n2 = len(group_data[0]), len(group_data[1])
                effect_size = 1 - (2 * stat) / (n1 * n2)
        
        else:  # Multiple groups
            if test_type == 'parametric' or (test_type == 'auto' and normality_ok):
                # ANOVA
                stat, p_val = f_oneway(*group_data)
                test_name = "One-way ANOVA"
                
                # Eta squared
                all_data = np.concatenate(group_data)
                grand_mean = np.mean(all_data)
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
                ss_total = sum((x - grand_mean)**2 for x in all_data)
                effect_size = ss_between / ss_total if ss_total > 0 else 0
                
            else:
                # Kruskal-Wallis test
                stat, p_val = kruskal(*group_data)
                test_name = "Kruskal-Wallis"
                
                # Epsilon squared
                n_total = sum(len(g) for g in group_data)
                effect_size = stat / (n_total - 1) if n_total > 1 else 0
        
        # Calculate additional statistics
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
    
    # Multiple testing correction
    _, p_adj, _, _ = multipletests(results_df['p_value'], method=correction_method)
    results_df['p_adj'] = p_adj
    
    # Filter for significance and effect size
    significant = results_df[
        (results_df['p_adj'] <= alpha) & 
        (np.abs(results_df['effect_size']) >= effect_size_threshold)
    ]
    
    return significant.sort_values('p_adj')


def ttest(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = constants.DEFAULT_GROUP_COLUMN_VALUES,
    equal_var: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs independent t-tests for two groups."""
    table = table_to_df(table)
    table_with_column = merge_table_with_meta(table, metadata, group_column)
    
    results = []
    for feature in table_with_column.columns.drop(group_column):
        # Subset groups
        mask_group1 = (table_with_column[group_column] == group_column_values[0])
        mask_group2 = (table_with_column[group_column] == group_column_values[1])
        
        group1_values = table_with_column.loc[mask_group1, feature].dropna()
        group2_values = table_with_column.loc[mask_group2, feature].dropna()
        
        # Skip features with < 2 samples in either group
        if len(group1_values) < 2 or len(group2_values) < 2:
            continue
            
        try:
            t_stat, p_val = ttest_ind(group1_values, group2_values, equal_var=equal_var)
        except ValueError:
            continue  # Handle cases with invalid variance calculations
            
        # Calculate effect size (Cohen's d)
        n1, n2 = len(group1_values), len(group2_values)
        mean_diff = group1_values.mean() - group2_values.mean()
        std1 = group1_values.std(ddof=1)
        std2 = group2_values.std(ddof=1)
        
        # Pooled standard deviation for Cohen's d
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
        cohen_d = mean_diff / pooled_std if pooled_std != 0 else 0.0

        if constants.debug_mode:
            print(f"{feature}: {t_stat}, {p_val}, {mean_diff}, {cohen_d}")
            
        results.append({
            'feature': feature,
            't_statistic': t_stat,
            'p_value': max(p_val, 1e-10),  # Prevent zero p-values
            'mean_difference': mean_diff,
            'cohens_d': cohen_d
        })
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        if verbose:
            logger.error(
                    f"No features passed for groups: {group_column_values} "
                    f"in column '{group_column}'"
            )
        return pd.DataFrame(columns=['feature', 't_statistic', 'p_value'])

    # Filter invalid p-values and sort
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df[results_df['p_value'] <= 0.05]
    results_df = results_df.sort_values('p_value', ascending=True)
    return results_df


def mwu_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = constants.DEFAULT_GROUP_COLUMN_VALUES,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs Mann-Whitney U tests with Bonferroni correction for two groups."""
    table = table_to_df(table)
    table_with_column = merge_table_with_meta(table, metadata, group_column)
    
    # Total features tested (for Bonferroni)
    total_features = len(table_with_column.columns.drop(group_column))
    threshold = 0.01 / total_features
    
    results = []
    for feature in table_with_column.columns.drop(group_column):
        # Subset groups safely
        mask_group1 = (table_with_column[group_column] == group_column_values[0])
        mask_group2 = (table_with_column[group_column] == group_column_values[1])
        
        group1_values = table_with_column.loc[mask_group1, feature].dropna()
        group2_values = table_with_column.loc[mask_group2, feature].dropna()
        
        # Skip features with empty groups
        if len(group1_values) < 1 or len(group2_values) < 1:
            continue
        
        # Perform MWU test
        u_stat, p_val = mannwhitneyu(
            group1_values, 
            group2_values, 
            alternative='two-sided'
        )
        
        # Effect size and median difference
        n1, n2 = len(group1_values), len(group2_values)
        r = 1 - (2 * u_stat) / (n1 * n2)
        median_diff = group1_values.median() - group2_values.median()

        if constants.debug_mode:
            print(f"{feature}: {u_stat}, {p_val}, {median_diff}, {r}")
            
        results.append({
            'feature': feature,
            'u_statistic': u_stat,
            'p_value': max(p_val, 1e-10),  # Cap p-values
            'median_difference': median_diff,
            'effect_size_r': r
        })
        
    results_df = pd.DataFrame(results)
    if results_df.empty:
        if verbose:
            logger.error(
                f"No features passed Mann-Whitney U tests with Bonferroni correction "
                f"for groups: {group_column_values} "
                f"in column '{group_column}'"
            )
        return pd.DataFrame(columns=['feature', 'u_statistic', 'p_value'])

    # Filter invalid p-values and sort
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df[results_df['p_value'] <= 0.05]
    results_df = results_df.sort_values('p_value', ascending=True)
    
    # Apply Bonferroni threshold
    results_df = results_df[results_df['p_value'] <= threshold]
    return results_df


def microbial_network_analysis(
    table: Union[Dict, Table, pd.DataFrame],
    method: str = 'sparcc',
    threshold: float = 0.3,
    min_prevalence: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct microbial co-occurrence networks.
    
    Args:
        table: Input abundance table
        method: Correlation method ('sparcc', 'spearman', 'pearson')
        threshold: Minimum correlation threshold for edges
        min_prevalence: Minimum prevalence for including features
        
    Returns:
        Tuple of (correlation_matrix, network_edges)
    """
    df, _ = validate_inputs(table)
    
    # Filter by prevalence
    prevalence = (df > 0).mean()
    df_filt = df.loc[:, prevalence >= min_prevalence]
    
    logger.info(f"Network analysis: {df_filt.shape[1]} features after prevalence filtering")
    
    if method == 'sparcc':
        # Simplified SparCC-like approach
        # In practice, you'd use the actual SparCC algorithm
        corr_matrix = df_filt.corr(method='spearman')
    elif method in ['spearman', 'pearson']:
        corr_matrix = df_filt.corr(method=method)
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Create network edges
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


def kruskal_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs Kruskal-Wallis H-test with Bonferroni correction for ≥3 groups."""
    table = table_to_df(table)
    table_with_column = merge_table_with_meta(table, metadata, group_column)
    
    # Get unique groups if group_column_values not specified
    if group_column_values is None:
        group_column_values = table_with_column[group_column].unique().tolist()
    
    # Pre-calculate Bonferroni threshold
    total_features = len(table_with_column.columns.drop(group_column))
    threshold = 0.01 / total_features
    
    results = []
    for feature in table_with_column.columns.drop(group_column):
        # Collect data for all groups
        groups = []
        for group_value in group_column_values:
            mask = (table_with_column[group_column] == group_value)
            group_data = table_with_column.loc[mask, feature].dropna()
            if len(group_data) > 0:  # Skip empty groups
                groups.append(group_data)
        
        # Skip feature if < 2 groups have data
        if len(groups) < 2:
            continue
        
        try:
            h_stat, p_val = kruskal(*groups)
        except ValueError:
            continue  # Handle identical values in all groups
            
        # Calculate effect size (epsilon squared)
        n_total = sum(len(g) for g in groups)
        epsilon_sq = h_stat / (n_total - 1)
        
        if constants.debug_mode:
            print(f"{feature}: {h_stat}, {p_val}, {epsilon_sq}")
            
        results.append({
            'feature': feature,
            'h_statistic': h_stat,
            'p_value': max(p_val, 1e-10),
            'epsilon_squared': epsilon_sq,
            'groups_tested': len(groups)
        })
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        if verbose:
            logger.error(
                f"No features passed Kruskal-Wallis H-test with Bonferroni correction "
                f"for groups: {group_column_values} "
                f"in column '{group_column}'"
            )
        return pd.DataFrame(columns=['feature', 't_statistic', 'p_value'])

    # Filter invalid p-values and sort
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df[results_df['p_value'] <= 0.05]
    results_df = results_df.sort_values('p_value', ascending=True)
    
    # Apply Bonferroni correction
    results_df = results_df[results_df['p_value'] <= threshold]
    return results_df


def anova(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs one-way ANOVA for ≥3 groups."""
    table = table_to_df(table)
    table_with_column = merge_table_with_meta(table, metadata, group_column)
    
    # Get unique groups if group_column_values not specified
    if group_column_values is None:
        group_column_values = table_with_column[group_column].unique().tolist()
    
    results = []
    for feature in table_with_column.columns.drop(group_column):
        # Collect data for all groups
        groups = []
        for group_value in group_column_values:
            mask = (table_with_column[group_column] == group_value)
            group_data = table_with_column.loc[mask, feature].dropna()
            if len(group_data) >= 2:  # Require ≥ 2 samples per group
                groups.append(group_data.values)
        
        # Skip feature if < 2 groups have sufficient data
        if len(groups) < 2:
            continue
            
        try:
            # Perform one-way ANOVA
            f_stat, p_val = f_oneway(*groups)
            
            # Calculate effect size (eta squared)
            all_data = np.concatenate(groups)
            ss_between = sum([len(g) * (np.mean(g) - np.mean(all_data))**2 
                              for g in groups])
            ss_total = sum((x - np.mean(all_data))**2 for x in all_data)
            eta_sq = ss_between / ss_total if ss_total != 0 else 0.0
            
        except (ValueError, ZeroDivisionError):
            continue  # Handle degenerate cases
            
        results.append({
            'feature': feature,
            'f_statistic': f_stat,
            'p_value': max(p_val, 1e-10),
            'eta_squared': eta_sq,
            'groups_tested': len(groups)
        })
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        if verbose:
            logger.error(
                f"No features passed one-way ANOVA for groups: {group_column_values} "
                f"in column '{group_column}'"
            )
        return pd.DataFrame(columns=['feature', 'f_statistic', 'p_value'])

    # Filter and sort results
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df[results_df['p_value'] <= 0.05]
    results_df = results_df.sort_values('p_value', ascending=True)
    return results_df


def fisher_exact_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str,
    group_column_values: List[Union[bool, int, str]],
    alpha: float = 0.01,
    min_samples: int = 5,
    debug_mode: bool = constants.debug_mode,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs Fisher's Exact Tests with Bonferroni correction for presence-absence data."""
    # Convert to DataFrame and merge with metadata
    table_df = table_to_df(table)
    merged_df = merge_table_with_meta(table_df, metadata, group_column)
    
    # Total features for Bonferroni correction
    total_features = len(merged_df.columns) - 1  # Exclude group column
    threshold = alpha / total_features
    
    results = []
    for feature in merged_df.columns.drop(group_column):
        # Subset groups
        mask_group1 = (merged_df[group_column] == group_column_values[0])
        mask_group2 = (merged_df[group_column] == group_column_values[1])
        
        group1 = merged_df.loc[mask_group1, feature].dropna()
        group2 = merged_df.loc[mask_group2, feature].dropna()
        
        # Skip small groups
        if len(group1) < min_samples or len(group2) < min_samples:
            continue
            
        # Build 2x2 contingency table
        a = (group1 == 1).sum()  # Group1 present
        b = (group2 == 1).sum()  # Group2 present
        c = (group1 == 0).sum()  # Group1 absent
        d = (group2 == 0).sum()  # Group2 absent
        
        # Skip invariant features
        if (a + b == 0) or (c + d == 0):
            continue
            
        # Perform Fisher's Exact Test
        try:
            odds_ratio, p_val = fisher_exact(
                [[a, b], [c, d]], alternative='two-sided'
            )
        except ValueError:
            continue  # Skip invalid tables
            
        # Calculate proportions
        prop1 = a / (a + c) if (a + c) > 0 else 0
        prop2 = b / (b + d) if (b + d) > 0 else 0
        prop_diff = prop1 - prop2

        results.append({
            'feature': feature,
            'p_value': max(p_val, 1e-10),
            'odds_ratio': odds_ratio,
            'proportion_diff': prop_diff,
            f'prop_{group_column_values[0]}': prop1,
            f'prop_{group_column_values[1]}': prop2,
            f'present_{group_column_values[0]}': a,
            f'absent_{group_column_values[0]}': c,
            f'present_{group_column_values[1]}': b,
            f'absent_{group_column_values[1]}': d
        })
        
        if constants.debug_mode:
            print(
                f"{feature}: OR={odds_ratio:.3f}, "
                f"p={p_val:.4f}, "
                f"diff={prop_diff:.2f}"
            )
            
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    if results_df.empty:
        if verbose:
            logger.warning(
                "No significant features found after Fisher's Exact "
                f"Tests with Bonferroni correction"
            )
        return pd.DataFrame()
    
    # Apply Bonferroni correction
    results_df = results_df.sort_values('p_value', ascending=True)
    results_df['p_adj'] = results_df['p_value'] * total_features
    results_df['p_adj'] = results_df['p_adj'].clip(upper=1.0)  # Cap at 1.0
    
    # Filter significant results
    results_df = results_df[results_df['p_value'] <= threshold]
    return results_df


def spearman_correlation(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    continuous_column: str,
    alpha: float = 0.01
) -> pd.DataFrame:
    """Calculate Spearman correlations between features and a continuous metadata variable."""
    df = table_to_df(table)
    merged = merge_table_with_meta(df, metadata, continuous_column)
    
    results = []
    for feature in tqdm(
        merged.columns.drop(continuous_column), 
        desc="Calculating correlations"
    ):
        # Remove NA values pairwise
        valid_idx = merged[[feature, continuous_column]].dropna().index
        if len(valid_idx) < 3:
            continue
            
        subset = merged.loc[valid_idx]
        rho, p_val = spearmanr(subset[feature], subset[continuous_column])
        
        results.append({
            'feature': feature,
            'rho': rho,
            'p_value': p_val,
            'n_samples': len(valid_idx)
        })
    
    result_df = pd.DataFrame(results)
    result_df['p_adj'] = result_df['p_value'] * len(result_df)
    results_df = result_df[result_df['p_adj'] <= alpha].sort_values(
        'rho', key=abs, ascending=False
    )
    return results_df


def microbiome_age_prediction(
    table: Union[Dict, Table, pd.DataFrame],
    metadata: pd.DataFrame,
    age_column: str,
    feature_selection_method: str = 'correlation',
    top_n_features: int = 50
) -> Dict[str, Any]:
    """Predict biological age from microbiome composition.
    
    Args:
        table: Input abundance table
        metadata: Sample metadata with age information
        age_column: Column containing age values
        feature_selection_method: Method for feature selection
        top_n_features: Number of top features to use
        
    Returns:
        Dictionary with model results and selected features
    """
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    
    df, metadata = validate_inputs(table, metadata, age_column)
    merged = merge_table_with_meta(df, metadata, age_column)
    
    # Remove samples with missing age
    merged_clean = merged.dropna(subset=[age_column])
    
    if len(merged_clean) < 20:
        raise ValueError("Insufficient samples with age data for prediction")
    
    X = merged_clean.drop(columns=[age_column])
    y = merged_clean[age_column]
    
    # Feature selection
    if feature_selection_method == 'correlation':
        # Select features most correlated with age
        correlations = X.corrwith(y).abs().sort_values(ascending=False)
        selected_features = correlations.head(top_n_features).index.tolist()
    elif feature_selection_method == 'variance':
        # Select features with highest variance
        variances = X.var().sort_values(ascending=False)
        selected_features = variances.head(top_n_features).index.tolist()
    else:
        raise ValueError(f"Unknown feature selection method: {feature_selection_method}")
    
    X_selected = X[selected_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_selected, y, cv=5, scoring='r2')
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'selected_features': selected_features,
        'feature_importance': feature_importance,
        'test_mae': mae,
        'test_r2': r2,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'predictions': pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        })
    }

  

class StatisticalAnalysis:
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
        if self.config.get("nfc_facilities", {}).get('enabled', False) and 'facility_match' in self.metadata["raw"]["genus"].columns:
            self.group_columns.append({'name': 'nfc_facilities', 'type': 'bool', 'values': [True, False]})
        self.results: Dict = {}
        for group_column in self.group_columns:
            col, vals = group_column['name'], group_column['values']
            self.results[group_column['name']] = self._run_for_group(col, vals)

    def _run_for_group(
        self, 
        group_column: str,
        group_column_values: List[Any]
    ):
        # Check which table_type/level/test combinations are enabled
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
                    result = self.TestConfig[test]["func"](
                        table=table_aligned,
                        metadata=metadata_aligned,
                        group_column=group_column,
                        group_column_values=group_column_values
                    )
                    
                    # Store and save results
                    data_storage[test] = result
                    result.to_csv(output_path, sep='\t', index=True)
                    
                    #if verbose:
                        # Log significant features
                    #    if isinstance(result, pd.DataFrame) and "p_value" in result.columns:
                    #        self._log_test_results(result, table_type, level, test)
                    
                except Exception as e:
                    logger.error(
                        f"Test '{test}' failed for {table_type}/{level}: {str(e)}"
                    )
                    data_storage[test] = None
                    
                finally:
                    progress.update(stats_task, advance=1)
        progress.update(stats_task, description=_format_task_desc(stats_desc))    
        return group_stats

    def get_effect_size(self, test_name: str, row: pd.Series) -> Optional[float]:
        if test_name not in self.TestConfig:
            return None
        test_config = self.TestConfig[test_name]
        for col in (test_config["effect_col"], test_config["alt_effect_col"]):
            if col and col in row:
                return row[col]
        return None
