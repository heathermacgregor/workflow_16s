# ===================================== IMPORTS ====================================== #

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from biom import Table
from scipy import stats
from scipy.spatial.distance import braycurtis, pdist, squareform
from scipy.stats import kruskal, mannwhitneyu, spearmanr, ttest_ind
from skbio.stats.composition import clr
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import logging

logger = logging.getLogger('workflow_16s')

# ================================= DEFAULT VALUES =================================== #

DEFAULT_MIN_REL_ABUNDANCE = 1
DEFAULT_MIN_SAMPLES = 10
DEFAULT_MIN_COUNTS = 1000
DEFAULT_PA_THRESHOLD = 0.99
DEFAULT_N_CLUSTERS = 10
DEFAULT_RANDOM_STATE = 0
DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]
DEFAULT_PSEUDOCOUNT = 1e-5

# ================================ CORE FUNCTIONALITY ================================ #

def table_to_dataframe(table: Union[Dict, Table, pd.DataFrame]) -> pd.DataFrame:
    """Convert BIOM Table/dict to samples × features DataFrame."""
    if isinstance(table, pd.DataFrame):
        return table
    if isinstance(table, Table):
        return table.to_dataframe(dense=True).T
    if isinstance(table, dict):
        return pd.DataFrame(table)
    raise TypeError("Input must be BIOM Table, dict, or DataFrame.")


def merge_table_with_metadata(
    table: pd.DataFrame,
    metadata: pd.DataFrame, 
    group_column: str
) -> pd.DataFrame:
    """Merge abundance table with metadata column after index sanitization."""
    # Preserve original index names
    table_index_name = table.index.name or 'index'
    meta_index_name = metadata.index.name or 'index'
    
    # Reset indexes for merging
    table = table.reset_index().rename(columns={table_index_name: 'temp_index'})
    metadata = metadata.reset_index().rename(columns={meta_index_name: 'temp_index'})
    
    # Sanitize IDs
    table['temp_index'] = table['temp_index'].astype(str).str.strip().str.lower()
    metadata['temp_index'] = metadata['temp_index'].astype(str).str.strip().str.lower()
    
    # Perform merge
    merged = pd.merge(
        table,
        metadata[[group_column, 'temp_index']],
        on='temp_index',
        how='inner'
    ).set_index('temp_index')
    
    # Restore original index name
    merged.index.name = table_index_name
    
    # Validate merge
    if merged[group_column].isna().any():
        missing = merged[group_column].isna().sum()
        raise ValueError(
            f"{missing} samples have NaN in '{group_column}' after merge. "
            "Check metadata completeness."
        )
        
    return merged
    

def filter_table(
    table: Union[Dict, Table, pd.DataFrame], 
    min_rel_abundance: float = DEFAULT_MIN_REL_ABUNDANCE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    min_counts: int = DEFAULT_MIN_COUNTS
) -> pd.DataFrame:
    """Filter features and samples based on abundance thresholds."""
    df = table_to_dataframe(table)
    df = filter_features(df, min_rel_abundance, min_samples)
    df = filter_samples(df, min_counts)
    return df
    

def filter_features(
    table: pd.DataFrame,
    min_rel_abundance: float,
    min_samples: int
) -> pd.DataFrame:
    """Filter features by relative abundance and sample presence."""
    min_abs = min_rel_abundance / 100
    feature_mask = (table.max(axis=0) >= min_abs) & (table.astype(bool).sum(axis=0) >= min_samples)
    return table.loc[:, feature_mask]
    

def filter_samples(table: pd.DataFrame, min_counts: int) -> pd.DataFrame:
    """Filter samples by total counts."""
    sample_mask = table.sum(axis=1) >= min_counts
    return table.loc[sample_mask]
    

def preprocess_table(
    table: Union[Dict, Table, pd.DataFrame], 
    apply_filter: bool = False,
    normalize: bool = True,
    clr_transform: bool = True,
    pseudocount: float = DEFAULT_PSEUDOCOUNT
) -> pd.DataFrame:
    """Preprocess table with filtering, normalization, and CLR."""
    df = table_to_dataframe(table)
    
    if apply_filter:
        df = filter_table(df)
    if normalize:
        df = normalize_table(df)
    if clr_transform:
        df = clr_transform_table(df, pseudocount)
    return df
    

def normalize_table(table: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
    """Convert to relative abundances along specified axis."""
    return table.div(table.sum(axis=axis), axis=1-axis)
    

def clr_transform_table(table: pd.DataFrame, pseudocount: float) -> pd.DataFrame:
    """Apply centered log-ratio transformation."""
    return pd.DataFrame(
        clr(table + pseudocount),
        index=table.index,
        columns=table.columns
    )
    

def presence_absence(
    table: Union[Dict, Table, pd.DataFrame],
    threshold: float = DEFAULT_PA_THRESHOLD
) -> pd.DataFrame:
    """Binarize table keeping features comprising threshold fraction of total abundance."""
    df = table_to_dataframe(table).T  # Features × samples
    sorted_features = df.sum(axis=1).sort_values(ascending=False)
    cumulative = sorted_features.cumsum() / sorted_features.sum()
    keep = cumulative[cumulative <= threshold].index
    return df.reindex(keep).gt(0).astype(int).T
    

def k_means(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    n_clusters: int = DEFAULT_N_CLUSTERS, 
    random_state: int = DEFAULT_RANDOM_STATE
) -> pd.Series:
    """Apply K-means clustering and return cluster labels."""
    df = table_to_dataframe(table)
    merged = merge_table_with_metadata(df, metadata, group_column)
    
    kmeans = KMeans(
        n_clusters, 
        random_state=random_state
    ).fit(merged.drop(group_column, axis=1))
    return pd.Series(
        kmeans.labels_, 
        index=merged.index, 
        name='kmeans_cluster'
    )
    

def _base_statistical_test(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    test_func: callable,
    **test_kwargs
) -> pd.DataFrame:
    """Base function for statistical tests."""
    merged = merge_table_with_metadata(table, metadata, group_column)
    features = merged.columns.drop(group_column)
    results = []
    
    for feature in features:
        groups = [
            g.dropna() for _, g in merged.groupby(group_column)[feature]
        ]
        if len(groups) < 2:
            continue

        try:
            stat, p = test_func(*groups, **test_kwargs)
            results.append({
                'feature': feature,
                'statistic': stat,
                'p_value': max(p, 1e-10)  # Prevent log(0) in downstream analysis
            })
        except Exception as e:
            logger.debug(f"Test failed for {feature}: {str(e)}")
        
    return pd.DataFrame(results).dropna(subset=['p_value'])

# ============================== STATISTICAL INTERFACE =============================== #

def t_test(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
) -> pd.DataFrame:
    """Independent t-tests between groups."""
    df = table_to_dataframe(table)
    return _base_statistical_test(
        df, metadata, group_column,
        test_func=ttest_ind,
        equal_var=False
    )


def mwu_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
) -> pd.DataFrame:
    """Mann-Whitney U tests with Bonferroni correction."""
    results = _base_statistical_test(
        table_to_dataframe(table), metadata, group_column,
        test_func=mannwhitneyu,
        alternative='two-sided'
    )
    threshold = 0.01 / len(results)
    return results[results.p_value <= threshold].sort_values('p_value')


def kruskal_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    **kwargs
) -> pd.DataFrame:
    """Kruskal-Wallis test with Bonferroni correction."""
    df = table_to_dataframe(table)
    results = _base_statistical_test(
        df, metadata, group_column, 
        test_func=kruskal,
        **kwargs
    )
    threshold = 0.01 / len(results)
    return results[results.p_value <= threshold].sort_values('p_value')
    

def variability_explained(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
) -> pd.DataFrame:
    """Calculate R²."""
    merged = merge_table_with_metadata(table_to_dataframe(table), metadata, group_column)
    X = merged[[group_column]]
    features = merged.columns.drop(group_column)
    r_squared = []
    
    for feature in features:
        y = merged[[feature]]
        model = LinearRegression().fit(X, y)
        r_squared.append({
            'feature': feature,
            'r_squared': model.score(X, y)
        })
    
    return pd.DataFrame(r_squared).sort_values('r_squared', ascending=False)
