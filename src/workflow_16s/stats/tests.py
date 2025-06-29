# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table
from scipy.spatial.distance import pdist, squareform
from scipy.stats import (
    fisher_exact, 
    f_oneway, 
    kruskal, 
    mannwhitneyu, 
    spearmanr, 
    ttest_ind
)
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.stats.utils import merge_table_with_metadata, table_to_dataframe

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ================================= DEFAULT VALUES =================================== #

DEFAULT_N_CLUSTERS = 10
DEFAULT_RANDOM_STATE = 0

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

debug_mode = False

# ==================================== FUNCTIONS ===================================== #

def k_means(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    n_clusters: int = DEFAULT_N_CLUSTERS, 
    random_state: int = DEFAULT_RANDOM_STATE,
    verbose: bool = False
) -> pd.Series:
    """
    Apply K-means clustering and return cluster labels.
    """
    table = table_to_dataframe(table)
    table_with_column = merge_table_with_metadata(table, metadata, group_column)
    
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
    

def ttest(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = DEFAULT_GROUP_COLUMN_VALUES,
    equal_var: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs independent t-tests for two groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: Two group identifiers to compare.
        equal_var:           Whether to assume equal population variances (default: False).
        
    Returns:
        DataFrame with significant features (p < Bonferroni-corrected threshold).
    """
    table = table_to_dataframe(table)
    table_with_column = merge_table_with_metadata(table, metadata, group_column)
    
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

        if debug_mode:
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
                    f"{table.shape} {table_with_column.shape} "
                    f"{table.index} {table_with_column.index} "
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
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = DEFAULT_GROUP_COLUMN_VALUES,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs Mann-Whitney U tests with Bonferroni correction for two groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: Two group identifiers to compare.
        
    Returns:
        Results with p-values below Bonferroni-corrected threshold.
    """
    table = table_to_dataframe(table)
    table_with_column = merge_table_with_metadata(table, metadata, group_column)
    
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

        if debug_mode:
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


def kruskal_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs Kruskal-Wallis H-test with Bonferroni correction for ≥3 groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: List of group identifiers to compare 
                             (None = use all groups).
        
    Returns:
        DataFrame with significant features after Bonferroni correction.
    """
    table = table_to_dataframe(table)
    table_with_column = merge_table_with_metadata(table, metadata, group_column)
    
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
        
        if debug_mode:
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
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs one-way ANOVA for ≥3 groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: List of group identifiers to compare 
                             (None = use all groups).
        
    Returns:
        DataFrame with significant features after Bonferroni correction.
    
    Note: Effect size (eta squared) represents the proportion of variance 
    explained by groups.
    Values range from 0 to 1, with higher values indicating stronger group 
    separation.
    """
    table = table_to_dataframe(table)
    table_with_column = merge_table_with_metadata(table, metadata, group_column)
    
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
    debug_mode: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Performs Fisher's Exact Tests with Bonferroni correction for 
    presence-absence data.
    
    Args:
        table: Input presence-absence table (samples x features, binary 0/1)
        metadata: Sample metadata DataFrame
        group_column: Metadata column containing group labels
        group_column_values: Two group identifiers to compare
        alpha: Significance level before correction (default: 0.01)
        min_samples: Minimum samples required per group (default: 5)
        debug_mode: Print debug information if True
        
    Returns:
        DataFrame with significant results (p-value ≤ Bonferroni-
        corrected threshold)
    """
    # Convert to DataFrame and merge with metadata
    table_df = table_to_dataframe(table)
    merged_df = merge_table_with_metadata(table_df, metadata, group_column)
    
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
        
        if debug_mode:
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
    """
    Calculate Spearman correlations between features and a continuous 
    metadata variable.
    
    Args:
        table: Input abundance table
        metadata: Sample metadata
        continuous_column: Metadata column with continuous values
        alpha: Significance threshold
        
    Returns:
        DataFrame with correlation results
    """
    df = table_to_dataframe(table)
    merged = merge_table_with_metadata(df, metadata, continuous_column)
    
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
    results_df = result_df[result_df['p_adj'] <= alpha].sort_values('rho', key=abs, ascending=False)
    return results_df


def calculate_distance_matrix(
    table: Union[Dict, Table, pd.DataFrame],
    metric: str = 'braycurtis'
) -> DistanceMatrix:
    """
    Calculate distance matrix from abundance table.
    
    Args:
        table:  Input abundance table.
        metric: Distance metric (default: braycurtis).
        
    Returns:
        skbio DistanceMatrix object
    """
    df = table_to_dataframe(table)
    ids = df.index.tolist()
    dist_array = pdist(df.values, metric=metric)
    return DistanceMatrix(squareform(dist_array), ids)
    

def run_ordination(
    table: Union[Dict, Table, pd.DataFrame],
    method: str = 'pca',
    n_components: int = 2,
    random_state: int = DEFAULT_RANDOM_STATE
) -> pd.DataFrame:
    """
    Perform dimensionality reduction.
    
    Args:
        table:        Input abundance table.
        method:       'pca', 'pcoa', 'tsne', or 'umap'.
        n_components: Number of dimensions to keep.
        random_state: Random seed.
        
    Returns:
        DataFrame with ordination coordinates
    """
    df = table_to_dataframe(table)
    scaled = StandardScaler().fit_transform(df)
    
    if method == 'pca':
        model = PCA(n_components=n_components, random_state=random_state)
        results = model.fit_transform(scaled)
    elif method == 'pcoa':
        dm = calculate_distance_matrix(df)
        results = PCoA(dm).scores(scores_df).samples.values[:, :n_components]
    elif method == 'tsne':
        model = TSNE(n_components=n_components, random_state=random_state)
        results = model.fit_transform(scaled)
    elif method == 'umap':
        model = UMAP(n_components=n_components, random_state=random_state)
        results = model.fit_transform(scaled)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return pd.DataFrame(
        results, 
        index=df.index, 
        columns=[f"{method.upper()}{i+1}" for i in range(n_components)]
    )
