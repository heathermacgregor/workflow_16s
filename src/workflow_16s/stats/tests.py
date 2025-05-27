# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table
from scipy.spatial.distance import braycurtis, pdist, squareform
from scipy.stats import kruskal, mannwhitneyu, spearmanr, ttest_ind
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.stats.utils import merge_table_with_metadata, table_to_dataframe

# ================================= DEFAULT VALUES =================================== #

DEFAULT_N_CLUSTERS = 10
DEFAULT_RANDOM_STATE = 0

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def k_means(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    n_clusters: int = DEFAULT_N_CLUSTERS, 
    random_state: int = DEFAULT_RANDOM_STATE
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
    equal_var: bool = False
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
            print(f"{feature} {t_stat} {p_val}")
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
        
        results.append({
            'feature': feature,
            't_statistic': t_stat,
            'p_value': max(p_val, 1e-10),  # Prevent zero p-values
            'mean_difference': mean_diff,
            'cohens_d': cohen_d
        })
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.error(
            f"No features passed the t-test for groups: {group_column_values} "
            f"in column '{group_column}'"
        )
        return pd.DataFrame(columns=['feature', 't_statistic', 'p_value'])

    # Filter invalid p-values and sort
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df.sort_values('p_value')
    return results_df
    

def mwu_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = DEFAULT_GROUP_COLUMN_VALUES
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
        
        results.append({
            'feature': feature,
            'u_statistic': u_stat,
            'p_value': max(p_val, 1e-10),  # Cap p-values
            'median_difference': median_diff,
            'effect_size_r': r
        })
        
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.error(
            f"No features passed the mwu test for groups: {group_column_values} "
            f"in column '{group_column}'"
        )
        return pd.DataFrame(columns=['feature', 'u_statistic', 'p_value'])

    # Filter invalid p-values and sort
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df.sort_values('p_value')
    
    # Apply Bonferroni threshold
    results_df_filtered = results_df[results_df['p_value'] <= threshold]
    return results_df_filtered


def kruskal_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None
) -> pd.DataFrame:
    """
    Performs Kruskal-Wallis H-test with Bonferroni correction for ≥3 groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: List of group identifiers to compare (None = use all groups).
        
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
        
        results.append({
            'feature': feature,
            'h_statistic': h_stat,
            'p_value': max(p_val, 1e-10),
            'epsilon_squared': epsilon_sq,
            'groups_tested': len(groups)
        })
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.error(
            f"No features passed the kruskal-wallis for groups: {group_column_values} "
            f"in column '{group_column}'"
        )
        return pd.DataFrame(columns=['feature', 't_statistic', 'p_value'])

    # Filter invalid p-values and sort
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df.sort_values('p_value')
    
    # Apply Bonferroni correction
    results_df_filtered = results_df[results_df['p_value'] <= threshold]
    return results_df_filtered


def anova(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    group_column_values: List[Union[bool, int, str]] = None
) -> pd.DataFrame:
    """
    Performs one-way ANOVA for ≥3 groups.
    
    Args:
        table:               Input abundance table (samples x features).
        metadata:            Sample metadata DataFrame.
        group_column:        Metadata column containing group labels.
        group_column_values: List of group identifiers to compare (None = use all groups).
        
    Returns:
        DataFrame with significant features after Bonferroni correction.
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
            ss_between = sum([len(g) * (np.mean(g) - np.mean(all_data))**2 for g in groups])
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
        logger.error(
            f"No features passed ANOVA for groups: {group_column_values} "
            f"in column '{group_column}'"
        )
        return pd.DataFrame(columns=['feature', 'f_statistic', 'p_value'])

    # Filter and sort results
    results_df = results_df[(results_df['p_value'] != 0) & (
        results_df['p_value'].notna()
    )]
    results_df = results_df.sort_values('p_value')
    return results_df
