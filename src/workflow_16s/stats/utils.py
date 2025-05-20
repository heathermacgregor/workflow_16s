# ===================================== IMPORTS ====================================== #

# Standard Library Imports
from pathlib import Path
from typing import Any, Bool, Dict, List, Tuple, Union

# Third-Party Imports
from biom import Table
import numpy as np
import pandas as pd
from scipy.spatial.distance import braycurtis, pdist, squareform
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu, spearmanr, ttest_ind
from skbio.stats.composition import clr as CLR
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from umap import UMAP

# ================================= DEFAULT VALUES =================================== #

DEFAULT_MIN_REL_ABUNDANCE = 1
DEFAULT_MIN_SAMPLES = 10
DEFAULT_MIN_COUNTS = 1000

DEFAULT_PA_THRESHOLD = 0.99

DEFAULT_N_CLUSTERS = 10
DEFAULT_RANDOM_STATE = 0

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

# ==================================== FUNCTIONS ===================================== #

def table_to_dataframe(
    table: Union[Dict, Table]
) -> pd.DataFrame:
    """Convert a feature table to a pandas DataFrame."""
    # Convert table to DataFrame 
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True) # features x samples
        table = table.T                        # samples  x features
    if isinstance(table, Dict):
        table = pd.DataFrame(table)            # samples  x features

    return table

def filter_table(
    table: Union[Dict, Table, pd.DataFrame], 
    min_rel_abundance: float = DEFAULT_MIN_REL_ABUNDANCE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_counts: int = DEFAULT_MIN_COUNTS
) -> pd.DataFrame:
    """
    Filters features and samples from an abundance table based on abundance 
    thresholds.
    
    Args:
        table:             Input abundance table. 
        min_rel_abundance: Minimum relative abundance percentage (0-100) for 
                           feature retention.
        min_samples:       Minimum number of samples a feature must appear in.
        min_counts:        Minimum total counts required per sample.
        
    Returns:
        table:             Filtered table in samples x features format.
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)
    
    table = filter_features(table, min_rel_abundance, min_samples)
    table = feature_samples(table, min_counts)
    return table


def filter_features(
    table: pd.DataFrame,
    min_rel_abundance: float = DEFAULT_MIN_REL_ABUNDANCE,
    min_samples: int = DEFAULT_MIN_SAMPLES
) -> pd.DataFrame:
    """
    Filter for columns (samples) where at least one row (OTU) has a relative 
    abundance of at least X% and for rows (OTUs) that are present in at least 
    Y columns (samples).
    
    Args:
        table:             Input abundance table (samples x features).
        min_rel_abundance: Minimum relative abundance.
        min_samples:       Minimum samples.
        
    Returns:
        table:             Filtered table ready for downstream analysis.
    """
    table = table.loc[:, table.max(axis=0) >= min_rel_abundance / 100]           
    table = table.loc[(table > 0).sum(axis=1) > min_samples, :]   
    return table


def filter_samples(
    table: pd.DataFrame, 
    min_counts: int = DEFAULT_MIN_COUNTS
) -> pd.DataFrame:                  
    """
    Filter for columns (samples) that have at least X counts total.

    Args:
        table:      Input abundance table (samples x features).
        min_counts: Minimum counts.
        
    Returns:
        table:      Filtered table ready for downstream analysis.
    """ 
    return table.loc[:, (table.sum(axis=0) > min_counts)] 


def preprocess_table(
    table: Union[Dict, Table, pd.DataFrame], 
    filter: bool = False,
    normalize: bool = True,
    clr_transform: bool = True
) -> pd.DataFrame:
    """
    Preprocesses abundance table with normalization and CLR transformation.
    
    Args:
        table:         Input abundance table (samples x features).
        normalize:     Whether to normalize samples to relative abundances.
        clr_transform: Whether to apply centered log-ratio (CLR) transformation.
        
    Returns:
        table:         Processed table ready for downstream analysis.
    """
    if not isinstance(table, pd.DataFrame):
        table_0 = table
        table = table_to_dataframe(table)
    if isinstance(table_0, Table):
        table = table.T

    if filter:
        table = filter_table(table)
        
    if normalize:
        table = normalize_table(table, axis=0)
        
    if clr_transform:
        table = clr_transform_table(table)
        
    return table
    
    
def normalize_table(
    table: pd.DataFrame, 
    axis: int = 0
) -> pd.DataFrame:
    """
    Normalize by column (sample) to get relative abundances for each sample.
    
    Args:
        table:
        axis:

    Returns:
        table:
    """
    table_n = table.apply(lambda x: x / x.sum(), axis=axis)
    return table_n


def clr_transform_table(
    table: pd.DataFrame
) -> pd.DataFrame:
    """
    Applies centered log-ratio (CLR) transformation with pseudocount addition.
    
    Args:
        table:  Input abundance table (samples x features).
        
    Returns:
        clr_df: CLR-transformed table preserving feature/sample labels.
    """
    np_clr = CLR(table + 0.00001) 
    table_clr = pd.DataFrame(np_clr, index=table.index, columns=table.columns)
    return table_clr


def presence_absence(
    table: Union[Dict, Table, pd.DataFrame],
    threshold: float = DEFAULT_PA_THRESHOLD
) -> pd.DataFrame:
    """
    Converts to presence/absence table while retaining top abundant 
    features.
    
    Args:
        table:          Input abundance table (features x samples).
        threshold:      Cumulative abundance threshold (0-1) for 
                        feature retention.
        
    Returns:
        filtered_table: Binary presence/absence table of retained features.
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)
    table = table.T # features x samples
    
    # Get total counts per feature
    feature_sums = np.array(table.sum(axis='observation')).flatten() 
    # Sort features from most to least abundant 
    sorted_idx = np.argsort(feature_sums)[::-1] 
    # Get cumulative proportion of total counts contributed by the top features
    cumulative = np.cumsum(feature_sums[sorted_idx]) / feature_sums.sum() 
    # Get the smallest number of top features needed to account for ≥99% of the total counts
    stop_idx = np.searchsorted(cumulative, threshold) + 1 
    keep_ids = [table.ids(axis='observation')[i] for i in sorted_idx[:stop_idx]]
    # Convert to presence/absence
    pa_table = table.pa(inplace=False) 
    filtered_table_table = pa_table.filter(keep_ids, axis='observation')
    filtered_table = filtered_table_table.to_dataframe(dense=True)
    return filtered_table


def k_means(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame, 
    col: str = DEFAULT_GROUP_COLUMN,
    n_clusters: int = DEFAULT_N_CLUSTERS, 
    random_state: int = DEFAULT_RANDOM_STATE
):
    """
    Applies K-means clustering and adds cluster labels to metadata.
    
    Args:
        table:          Input abundance table (samples x features).
        metadata:       Sample metadata DataFrame.
        col:            Metadata column name to preserve in output.
        n_clusters:     Number of clusters for K-means.
        random_state:   Random seed for reproducibility.
        
    Returns:
        table_with_col: Table with added 'kmeans_cluster' column.
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)

    table_with_col = table.join(metadata[[col]])

    # Apply K-means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(table_with_col.drop(col, axis=1))
    table_with_col['kmeans_cluster'] = kmeans.labels_
    return table_with_col


# Statistical tests
def t_test(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    col: str = DEFAULT_GROUP_COLUMN,
    col_values: List[Union[Bool, int, str]] = DEFAULT_GROUP_COLUMN_VALUES
) -> pd.DataFrame:
    """
    Performs independent t-tests between groups for all features.
    
    Args:
        table:      Input abundance table (samples x features).
        metadata:   Sample metadata DataFrame.
        col:        Metadata column containing group labels.
        col_values: Two group identifiers to compare.
        
    Returns:
        results:    Results sorted by p-value with test statistics.
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)

    table_with_col = table.join(metadata[[col]])
  
    results = []
    for feature in table_with_col.columns.drop(col):
        group_1 = metadata_df[metadata_df[col] == col_values[0]][feature]
        group_2 = metadata_df[metadata_df[col] == col_values[1]][feature]
        t_statistic, p_value = ttest_ind(group_1, group_2)
        results.append({
            'feature': col, 
            't_statistic': t_statistic, 
            'p_value': p_value
        })
        
    results = pd.DataFrame(results)
    results = results.sort_values(by='p_value', ascending=True)
    return results


def mwu_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    col: str = DEFAULT_GROUP_COLUMN,
    col_values: List[Union[Bool, int, str]] = DEFAULT_GROUP_COLUMN_VALUES
) -> pd.DataFrame:
    """
    Performs Mann-Whitney U tests with Bonferroni correction.
    
    Args:
        table:      Input abundance table (samples x features).
        metadata:   Sample metadata DataFrame.
        col:        Metadata column containing group labels.
        col_values: Two group identifiers to compare.
        
    Returns:
        results:    Filtered results meeting Bonferroni-corrected threshold.
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)

    table_with_col = table.join(metadata[[col]])
  
    group_labels = table_with_col[col].values
    group_1 = table_with_col[group_labels == col_values[0]] # Contaminated
    group_2 = table_with_col[group_labels == col_values[1]] # Pristine

    # Perform Mann-Whitney U test for each OTU
    results = []
    for feature in tqdm(
      table_with_col.columns.drop(col), 
      desc="Calculating MWU (Bonferroni)..."
    ):
        group_1_feature = group_1[feature].dropna()
        group_2_feature = group_2[feature].dropna()

        u_stat, p_val = mannwhitneyu(
          group_1_feature, 
          group_2_feature, 
          alternative='two-sided'
        )
        s, p = kruskal(group_1_feature, group_2_feature)
        
        # Effect size (r) computation
        n1, n2 = len(group_1_feature), len(group_2_feature)
        effect_size_r = 1 - ((2 * u_stat) / (n1 * n2))
        
        # Mean difference for context
        mean_diff = np.mean(group_1_feature) - np.mean(group_2_feature)
        
        results.append({
            'feature': feature,
            'u_statistic': u_stat,
            'p_value': p_val,
            'kruskal_s': s,
            'kruskal_p_value': p,
            'mean_difference': mean_diff,
            'effect_size_r': effect_size_r
        })
    
    results = pd.DataFrame(results)

    # Cap extremely small p-values to avoid numerical precision issues
    results['p_value'] = np.maximum(results['p_value'], 1e-10)
    results['kruskal_p_value'] = np.maximum(results['kruskal_p_value'], 1e-10)

    # Apply Bonferroni correction
    threshold = 0.01 / len(results)
    results = results[
    (results['p_value'] <= threshold) & (results['kruskal_p_value'] <= threshold)
    ]
    return results


def variability_explained(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    col: str = DEFAULT_GROUP_COLUMN   
) -> pd.DataFrame:
    """
    Calculate the variability explained by a metadata column for each feature 
    in the data.

    Args:
        table:    DataFrame of OTU abundances.
        metadata: DataFrame with sample metadata.
        col:      Column in metadata used as the explanatory variable.
    
    Returns:
        results:  DataFrame with variability explained (R^2) for each 
                  feature.
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)

    table_with_col = table.join(metadata[[col]])
    
    # Extract explanatory variable
    explanatory_variable = table_with_col[col].values.reshape(-1, 1)
    
    # Calculate R^2 for each feature in df
    results = []
    for feature in tqdm(
      table_with_col.columns.drop(col), 
      desc="Calculating explained variability..."
    ):
        try:
            response_variable = table_with_col[feature].values.reshape(-1, 1)
            
            # Fit linear model
            model = LinearRegression()
            model.fit(explanatory_variable, response_variable)
            
            # Compute R^2
            predictions = model.predict(explanatory_variable)
            r2 = r2_score(response_variable, predictions)
            
            results.append({
              'feature': feature, 
              'r^2': r2
            })
        except Exception as e:
            print(f"Skipping feature '{feature}' due to error: {e}")
            continue
    
    results = pd.DataFrame(results).sort_values(by='r^2', ascending=False)
    return results
