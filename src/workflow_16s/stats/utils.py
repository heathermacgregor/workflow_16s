# ===================================== IMPORTS ====================================== #
from pathlib import Path
from typing import Any, Bool, Dict, List, Tuple, Union
import numpy as np
import pandas as pd

from biom import Table

from scipy import stats
from scipy.stats import mannwhitneyu, kruskal, spearmanr, ttest_ind
from scipy.spatial.distance import pdist, squareform, braycurtis

from skbio.stats.composition import clr as CLR
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from umap import UMAP

from tqdm import tqdm

# ================================= GLOBAL VARIABLES ================================= #


# ==================================== FUNCTIONS ===================================== #

def filter_table(
    table: Union[Dict, Table, pd.DataFrame], 
    min_rel_abundance: float = 1,
    min_samples: int = 10,
    min_counts: int = 1000
) -> pd.DataFrame:
    """"""
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
        # Transpose it (samples x features)
        table = table.T
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
    
    table = filter_features(table, min_rel_abundance, min_samples)
    table = feature_samples(table, min_counts)
    return table


def filter_features(
    table: pd.DataFrame,
    min_rel_abundance: float = 1,
    min_samples: int = 10
) -> pd.DataFrame:
    """Filter for columns (samples) where at least one row (OTU) has a relative abundance 
    of at least X% and for rows (OTUs) that are present in at least Y columns (samples)"""
    table = table.loc[:, table.max(axis=0) >= min_rel_abundance / 100]           
    table = table.loc[(table > 0).sum(axis=1) > min_samples, :]   
    return table


def filter_samples(
    table: pd.DataFrame, 
    min_counts: int = 1000
) -> pd.DataFrame:                  
    """Filter for columns (samples) that have at least X counts total""" 
    return table.loc[:, (table.sum(axis=0) > min_counts)] 


def preprocess_table(
    table: Union[Dict, Table, pd.DataFrame], 
    normalize: bool = True,
    clr_transform: bool = True
) -> pd.DataFrame:
    """"""
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
        # Transpose it (samples x features)
        table = table.T
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
    
    if normalize:
        table = normalize_table(table, axis=0)
    if clr_transform:
        table = clr_transform_table(table)
    return table
    
    
def normalize_table(
    table: pd.DataFrame, 
    axis: int = 0
) -> pd.DataFrame:
    """Normalize by column (sample) to get relative abundances for each sample"""
    return table.apply(lambda x: x / x.sum(), axis=axis)


def clr_transform_table(
    table: pd.DataFrame
) -> pd.DataFrame:
    """"""
    np_clr = CLR(table + 0.00001)  
    return pd.DataFrame(np_clr, index=table.index, columns=table.columns)


def presence_absence(
    table: Union[Dict, Table, pd.DataFrame],
    threshold: float = 0.99
) -> pd.DataFrame:
    """"""
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
        # Transpose it (samples x features)
        table = table.T
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
    # Transpose back
    table = table.T
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
    col: str = 'nuclear_contamination_status',
    n_clusters: int = 10, 
    random_state: int = 0
):
    """"""
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
    # Transpose it (samples x features)
    table = table.T

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
    col: str = 'nuclear_contamination_status',
    col_values: List[Union[Bool, int, str]] = [True, False]
) -> pd.DataFrame:
    """"""
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
    # Transpose it (samples x features)
    table = table.T

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
    col: str = 'nuclear_contamination_status',
    col_values: List[Union[Bool, int, str]] = [True, False]
) -> pd.DataFrame:
    """"""
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
    # Transpose it (samples x features)
    table = table.T

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
    results = results[(results['p_value'] <= threshold) & (results['kruskal_p_value'] <= threshold)]
    return results


def variability_explained(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    col: str = 'nuclear_contamination_status'    
) -> pd.DataFrame:
    """Calculate the variability explained by a metadata column for each feature in the data

    Args:
        table: DataFrame of OTU abundances
        metadata: DataFrame with sample metadata
        col: Column in metadata used as the explanatory variable
    
    Returns:
        DataFrame with variability explained (R^2) for each feature
    """
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
    # Transpose it (samples x features)
    table = table.T

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
            # Handle any errors during processing
            print(f"Skipping feature {feature} due to error: {e}")
            continue
    
    results = pd.DataFrame(results).sort_values(by='r^2', ascending=False)
    return results
