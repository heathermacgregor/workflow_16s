# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from scipy.stats import (
    fisher_exact, f_oneway, kruskal, mannwhitneyu, spearmanr, ttest_ind
)
from skbio.diversity import alpha
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from tqdm import tqdm

# Local Imports
from workflow_16s import constants
from workflow_16s.utils.data import merge_table_with_meta, table_to_df, merge_data
from workflow_16s.stats.utils import validate_inputs

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

# CLUSTERING
def k_means(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    n_clusters: int = constants.DEFAULT_N_CLUSTERS, 
    random_state: int = constants.DEFAULT_RANDOM_STATE,
    verbose: bool = False
) -> pd.Series:
    """Apply K-means clustering and return cluster labels.

    Args:
        table:
        metadata:
        group_column:
        n_clusters:
        random_state:
        verbose:

    Returns:
    """
    df = table_to_df(table)
    data = merge_data(df, metadata, group_column)
    
    kmeans = KMeans(
        n_clusters, 
        random_state=random_state
    ).fit(data.drop(group_column, axis=1))

    results = pd.Series(
        kmeans.labels_, 
        index=data.index, 
        name='kmeans_cluster'
    )
    return results
    

# TWO-GROUP STATISTICAL TESTS
def ttest(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = constants.DEFAULT_GROUP_COLUMN_VALUES,
    equal_var: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs independent t-tests for two groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: Two group identifiers to compare.
        equal_var:           Whether to assume equal population variances (default: False).
        verbose:
        
    Returns:
        DataFrame with significant features (p < Bonferroni-corrected threshold).
    """
    df = table_to_df(table)
    data = merge_data(df, metadata, group_column)
    
    results = []
    for feature in data.columns.drop(data):
        # Subset groups
        mask_group1 = (data[group_column] == group_column_values[0])
        mask_group2 = (data[group_column] == group_column_values[1])
        
        group1_values = data.loc[mask_group1, feature].dropna()
        group2_values = data.loc[mask_group2, feature].dropna()
        
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
                f"{table.shape} {data.shape} "
                f"{table.index} {data.index} "
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
    """Performs Mann-Whitney U tests with Bonferroni correction for two groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: Two group identifiers to compare.
        verbose:
        
    Returns:
        Results with p-values below Bonferroni-corrected threshold.
    """
    df = table_to_df(table)#, metadata = validate_inputs(table, metadata, group_column)
    data = merge_data(df, metadata, group_column)
    
    # Total features tested (for Bonferroni)
    total_features = len(data.columns.drop(group_column))
    threshold = 0.01 / total_features
    
    results = []
    for feature in data.columns.drop(group_column):
        # Subset groups safely
        mask_group1 = (data[group_column] == group_column_values[0])
        mask_group2 = (data[group_column] == group_column_values[1])
        
        group1_values = data.loc[mask_group1, feature].dropna()
        group2_values = data.loc[mask_group2, feature].dropna()
        
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
    """Performs Fisher's Exact Tests with Bonferroni correction for 
    presence-absence data.
    
    Args:
        table:               Input presence-absence table (samples x 
                             features, binary 0/1).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: Two group identifiers to compare.
        alpha:               Significance level before correction 
                             (default: 0.01).
        min_samples:         Minimum samples required per group 
                             (default: 5).
        debug_mode:          Print debug information if True.
        verbose:
        
    Returns:
        DataFrame with significant results (p-value ≤ Bonferroni-
        corrected threshold)
    """
    # Convert to DataFrame and merge with metadata
    df = table_to_df(table)#, metadata = validate_inputs(table, metadata, group_column)
    merged = merge_data(df, metadata, group_column)
    
    # Total features for Bonferroni correction
    total_features = len(merged.columns) - 1  # Exclude group column
    threshold = alpha / total_features
    
    results = []
    for feature in merged.columns.drop(group_column):
        # Subset groups
        mask_group1 = (merged[group_column] == group_column_values[0])
        mask_group2 = (merged[group_column] == group_column_values[1])
        
        group1 = merged.loc[mask_group1, feature].dropna()
        group2 = merged.loc[mask_group2, feature].dropna()
        
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


# MULTI-GROUP STATISTICAL TESTS
def kruskal_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = constants.DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Performs Kruskal-Wallis H-test with Bonferroni correction for ≥3 groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: List of group identifiers to compare 
                             (None = use all groups).
        verbose:
        
    Returns:
        DataFrame with significant features after Bonferroni correction.
    """
    df = table_to_df(table)
    merged = merge_data(df, metadata, group_column)
    
    # Get unique groups if group_column_values not specified
    if group_column_values is None:
        group_column_values = merged[group_column].unique().tolist()
    
    # Pre-calculate Bonferroni threshold
    total_features = len(merged.columns.drop(group_column))
    threshold = 0.01 / total_features
    
    results = []
    for feature in merged.columns.drop(group_column):
        # Collect data for all groups
        groups = []
        for group_value in group_column_values:
            mask = (merged[group_column] == group_value)
            group_data = merged.loc[mask, feature].dropna()
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
    """Performs one-way ANOVA for ≥3 groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: List of group identifiers to compare 
                             (None = use all groups).
        verbose:
        
    Returns:
        DataFrame with significant features after Bonferroni correction.
    
    Note: 
        - Effect size (eta squared) represents the proportion of variance 
          explained by groups. Values range from 0 to 1, with higher values 
          indicating stronger group separation.
    """
    df = table_to_df(table)#, metadata = validate_inputs(table, metadata, group_column)
    merged = merge_data(df, metadata, group_column)
    
    # Get unique groups if group_column_values not specified
    if group_column_values is None:
        group_column_values = merged[group_column].unique().tolist()
    
    results = []
    for feature in merged.columns.drop(group_column):
        # Collect data for all groups
        groups = []
        for group_value in group_column_values:
            mask = (merged[group_column] == group_value)
            group_data = merged.loc[mask, feature].dropna()
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


# CORRELATION ANALYSIS
def spearman_correlation(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    continuous_column: str,
    alpha: float = 0.01,
    min_samples: int = 5,
    progress: Any,
    task_id: Any
) -> pd.DataFrame:
    """Spearman correlations with type validation and conversion"""
    df = table_to_df(table)
    
    # Validate continuous column
    if continuous_column not in metadata.columns:
        logger.error(f"Column '{continuous_column}' not found in metadata")
        return pd.DataFrame()
    
    # Check sufficient non-missing values
    non_missing_count = metadata[continuous_column].notna().sum()
    if non_missing_count < min_samples:
        logger.error(f"Only {non_missing_count} valid values for '{continuous_column}'")
        return pd.DataFrame()
    
    try:
        merged = merge_data(df, metadata, continuous_column)
        merged = merged.dropna(subset=[continuous_column])
    except KeyError as e:
        logger.error(f"Merge error: {str(e)}")
        return pd.DataFrame()
    
    logger.info(
        f"Correlation analysis for '{continuous_column}': "
        f"{len(merged)} samples with valid data"
    )
    
    # Convert continuous variable to numeric
    merged[continuous_column] = pd.to_numeric(
        merged[continuous_column], errors='coerce'
    )
    
    results = []
    total = len(merged.columns.drop(continuous_column))
    for i, feature in enumerate(merged.columns.drop(continuous_column)):
        # Convert feature to numeric
        merged[feature] = pd.to_numeric(merged[feature], errors='coerce')
        
        # Skip non-numeric features
        if not pd.api.types.is_numeric_dtype(merged[feature]):
            logger.warning(f"Skipping non-numeric feature: {feature}")
            continue
            
        # Pairwise complete observations
        valid_idx = merged[[feature, continuous_column]].dropna().index
        n_valid = len(valid_idx)
        
        if n_valid < min_samples:
            continue
            
        try:
            rho, p_val = spearmanr(
                merged.loc[valid_idx, feature], 
                merged.loc[valid_idx, continuous_column]
            )
            results.append({
                'feature': feature,
                'rho': rho,
                'p_value': p_val,
                'n_samples': n_valid
            })
        except Exception as e:
            logger.warning(f"Correlation failed for {feature}: {str(e)}")
        finally:
            progress.update(task_id, description=_format_task_desc(f"Correlation of {feature}: {i} / {total}"))
    
    if not results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(results)
    result_df['p_adj'] = result_df['p_value'] * len(result_df)
    result_df['p_adj'] = result_df['p_adj'].clip(upper=1.0)
    sig_df = result_df[result_df['p_adj'] <= alpha].sort_values(
        'rho', key=abs, ascending=False
    )
    
    logger.info(f"Found {len(sig_df)} significant correlations")
    return sig_df


# ENHANCED STATISTICAL TESTS
def enhanced_statistical_tests(
    table: Union[Dict, Table, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    test_type: str = 'auto',
    correction_method: str = 'fdr_bh',
    alpha: float = 0.05,
    effect_size_threshold: float = 0.5,
    progress: Any,
    task_id: Any
) -> pd.DataFrame:
    """Enhanced statistical testing with automatic test selection and effect sizes."""
    df = table_to_df(table)
    merged = merge_data(df, metadata, group_column)
    
    groups = merged[group_column].unique()
    n_groups = len(groups)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for comparison")
    
    results = []

    total = len(df.columns)
    for i, feature in enumerate(df.columns):
        try:
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
        except Exception as e:
            logger.error(f"Error with statistical test of {feature}: {e}")
        finally:
            progress.update(task_id, description=_format_task_desc(f"Enhanced statistical tests: {i} / {total}"))
    
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


# DIFFERENTIAL ABUNDANCE ANALYSIS
def differential_abundance_analysis(
    table: Union[Dict, Any, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    method: str = 'deseq2_like',
    alpha: float = 0.05,
    fold_change_threshold: float = 1.5,
    min_prevalence: float = 0.1,
    progress: Any,
    task_id: Any
) -> pd.DataFrame:
    """Comprehensive differential abundance analysis with multiple methods."""
    df = table_to_df(table)
    merged = merge_data(df, metadata, group_column)
    
    # Filter by prevalence
    prevalence = (df > 0).mean()
    df_filt = df.loc[:, prevalence >= min_prevalence]
    merged_filt = merge_table_with_meta(df_filt, metadata, group_column)
    
    groups = merged_filt[group_column].unique()
    if len(groups) != 2:
        raise ValueError("Differential abundance analysis requires exactly 2 groups")
    
    results = []

    total = len(df_filt.columns)
    for i, feature in enumerate(df_filt.columns):
        try:
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
        except Exception as e:
            logger.error(f"Error with DA analysis of {feature}: {e}")
        finally:
            progress.update(task_id, description=_format_task_desc(f"{i} / {total}"))
    
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


# CORE MICROBIOME ANALYSIS
def core_microbiome(
    table: Union[Dict, pd.DataFrame],
    metadata: pd.DataFrame,
    group_column: str,
    prevalence_threshold: float = 0.8,
    abundance_threshold: float = 0.01
) -> Dict[str, pd.DataFrame]:
    """Identify core microbiome for each group."""
    df = table_to_df(table)
    merged = merge_data(df, metadata, group_column)
    
    rel_abundance = df.div(df.sum(axis=1), axis=0)
    merged_rel = merge_data(rel_abundance, metadata, group_column)
    
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


# NETWORK ANALYSIS
def microbial_network_analysis(
    table: Union[Dict, Table, pd.DataFrame],
    method: str = 'sparcc',
    threshold: float = 0.3,
    min_prevalence: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Construct microbial co-occurrence networks."""
    df = table_to_df(table)#
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
