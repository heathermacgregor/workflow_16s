# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import warnings
from typing import Any, Dict, Optional, Union

# Third-Party Imports
import numpy as np
import pandas as pd
from biom import Table
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

# Set thread environment variables BEFORE importing UMAP
import os
os.environ['NUMBA_NUM_THREADS'] = '1'  # Default safe value
os.environ['OMP_NUM_THREADS'] = '1'
from umap import UMAP

# ========================== INITIALIZATION & CONFIGURATION ========================== #

warnings.filterwarnings("ignore") # Suppress warnings

# ================================= GLOBAL VARIABLES ================================= #

DEFAULT_METRIC = 'braycurtis'
DEFAULT_N_PCA = 20
DEFAULT_N_PCOA = None
DEFAULT_N_TSNE = 3
DEFAULT_N_UMAP = 3
DEFAULT_RANDOM_STATE = 0
DEFAULT_CPU_LIMIT = 1  # Default CPU limit for parallel operations

# ==================================== FUNCTIONS ===================================== #

def table_to_dataframe(table: Union[Dict[Any, Any], Table]) -> pd.DataFrame:
    """
    Convert a BIOM Table or dictionary to a pandas DataFrame.
    
    Args:
        table: Input feature table, either a BIOM Table or a dict-like
               object where keys are sample identifiers and values are feature
               counts or abundances.
    
    Returns:
        df:    Pandas DataFrame of shape (n_samples, n_features) with samples
               as rows and features as columns.
    
    Raises:
        ValueError: If input type is not supported.
    """
    if isinstance(table, Table):
        # Convert BIOM Table to DataFrame (features x samples), then transpose
        df = table.to_dataframe(dense=True).T
    elif isinstance(table, dict):
        # Construct DataFrame from dict with samples as rows
        df = pd.DataFrame.from_dict(table, orient='index')
    elif isinstance(table, pd.DataFrame):
        # Return as-is if already a DataFrame
        df = table
    else:
        raise ValueError("Unsupported input type for table conversion")
    
    # Validate dataframe
    if df.empty:
        raise ValueError("Input table is empty")
    return df


def distance_matrix(
    table: Union[Dict[Any, Any], Table, pd.DataFrame],
    metric: str = DEFAULT_METRIC
) -> np.ndarray:
    """
    Compute pairwise distance matrix from a feature table.
    
    Args:
        table:  Input feature table as a dict-like, BIOM Table, or DataFrame
                (samples x features or features x samples).
        metric: Distance metric name accepted by scipy.spatial.distance.pdist.
    
    Returns:
        dm:     2D numpy array representing the pairwise distance matrix.
    
    Raises:
        ValueError: If fewer than 2 samples are provided.
    """
    df = table_to_dataframe(table)
    
    # Validate sample size
    if df.shape[0] < 2:
        raise ValueError("At least 2 samples required for distance calculation")
    
    # Use more efficient pairwise_distances instead of pdist+squareform
    dm = pairwise_distances(df.values, metric=metric)
    return dm


def pcoa(
    table: Union[Dict, Table, pd.DataFrame], 
    metric: str = DEFAULT_METRIC, 
    n_dimensions: Optional[int] = DEFAULT_N_PCOA
) -> PCoA:
    """
    Perform Principal Coordinate Analysis (PCoA) on a feature table.
    
    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame.
        metric:       Distance metric for computing pairwise distances.
        n_dimensions: Number of principal coordinates to compute.
    
    Returns:
        pcoa_result: PCoA result object containing:
            - samples: DataFrame of PCoA coordinates
            - proportion_explained: Explained variance ratios
    
    Raises:
        ValueError: If fewer than 2 samples are provided.
    """
    # Convert to dataframe with samples as rows
    df = table_to_dataframe(table)
    
    # Validate sample size
    if df.shape[0] < 2:
        raise ValueError("At least 2 samples required for PCoA")
    
    sample_ids = df.index.tolist()
    
    # Fix for duplicate sample IDs
    if len(set(sample_ids)) != len(sample_ids):
        seen = {}
        new_ids = []
        for id_ in sample_ids:
            if id_ in seen:
                seen[id_] += 1
                new_ids.append(f"{id_}_{seen[id_]}")
            else:
                seen[id_] = 1
                new_ids.append(id_)
        sample_ids = new_ids
    
    # Compute pairwise distances using optimized method
    dist_matrix = pairwise_distances(df.values, metric=metric)
    
    # Create DistanceMatrix with sample IDs
    dist_matrix = DistanceMatrix(dist_matrix, ids=sample_ids)
    
    # Set safe dimension limit
    max_dims = min(df.shape[0] - 1, df.shape[1])
    if n_dimensions is None:
        n_dimensions = max_dims
    else:
        n_dimensions = min(n_dimensions, max_dims)
    
    # Perform PCoA with safe dimension setting
    pcoa_result = PCoA(dist_matrix, number_of_dimensions=n_dimensions)
    
    # Ensure consistent PC naming
    pcoa_result.samples.columns = [f"PCo{i+1}" for i in range(n_dimensions)]
    
    return pcoa_result


def pca(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = DEFAULT_N_PCA
) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis (PCA) on a feature table.
    
    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples).
        n_components: Number of principal components to compute.
    
    Returns:
        Dictionary containing:
            - 'components': DataFrame of component scores (samples x components)
            - 'exp_var_ratio': Explained variance ratios per component
            - 'exp_var_cumul': Cumulative explained variance ratios
            - 'loadings': PCA loadings (features x components)
    
    Raises:
        ValueError: If fewer than 2 samples are provided or n_components is invalid.
    """
    df = table_to_dataframe(table)
    
    # Validate input
    if df.shape[0] < 2:
        raise ValueError("At least 2 samples required for PCA")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    
    # Set safe component limit
    n_components = min(n_components, df.shape[0] - 1, df.shape[1] - 1)
    
    # Standardize features
    scaled = StandardScaler().fit_transform(df.values)
    scaled_df = pd.DataFrame(scaled, index=df.index, columns=df.columns)

    # Fit PCA
    pca_model = PCA(n_components=n_components)
    scores = pca_model.fit_transform(scaled_df.values)

    # Prepare results
    components_df = pd.DataFrame(
        scores,
        index=df.index,
        columns=[f"PC{i+1}" for i in range(n_components)]
    )
    exp_var_ratio = pca_model.explained_variance_ratio_
    exp_var_cumul = np.cumsum(exp_var_ratio)
    loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)

    return {
        'components': components_df,
        'exp_var_ratio': exp_var_ratio,
        'exp_var_cumul': exp_var_cumul,
        'loadings': loadings
    }


def tsne(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = DEFAULT_N_TSNE,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_CPU_LIMIT
) -> pd.DataFrame:
    """
    Compute t-distributed Stochastic Neighbor Embedding (t-SNE) reduction.
    
    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples).
        n_components: Dimension of the embedded space.
        random_state: Random seed for reproducibility.
        n_jobs:       Number of CPU cores to use.
    
    Returns:
        tsne_df:      DataFrame of shape (n_samples, n_components) with t-SNE 
                      coordinates.
    
    Raises:
        ValueError: If fewer than 2 samples are provided or n_components is invalid.
    """
    df = table_to_dataframe(table)
    
    # Validate input
    if df.shape[0] < 2:
        raise ValueError("At least 2 samples required for t-SNE")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    
    # Set safe component limit
    n_components = min(n_components, df.shape[0] - 1)

    # Validate data quality
    if np.isnan(df.values).any():
        raise ValueError("Input data contains NaNs")
    if not np.isfinite(df.values).all():
        raise ValueError("Input data contains infinite values")
    
    tsne_arr = TSNE(
        n_components=n_components, 
        random_state=random_state,
        n_jobs=n_jobs
    ).fit_transform(df.values)
    
    tsne_df = pd.DataFrame(
        tsne_arr,
        index=df.index,
        columns=[f"TSNE{i+1}" for i in range(n_components)]  # Fixed to uppercase
    )
    return tsne_df


def umap(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = DEFAULT_N_UMAP,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_CPU_LIMIT
) -> pd.DataFrame:
    """
    Compute Uniform Manifold Approximation and Projection (UMAP) reduction.
    
    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples).
        n_components: Dimension of the embedded space.
        random_state: Random seed for reproducibility.
        n_jobs:       Number of CPU cores to use.
    
    Returns:
        umap_df:      DataFrame of shape (n_samples, n_components) with UMAP 
                      coordinates.
    
    Raises:
        ValueError: If fewer than 2 samples are provided or n_components is invalid.
    """
    df = table_to_dataframe(table)
    
    # Validate input
    if df.shape[0] < 2:
        raise ValueError("At least 2 samples required for UMAP")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    
    # Set safe component limit
    n_components = min(n_components, df.shape[0] - 1)

    # Handle threading conflicts gracefully
    try:
        # Match n_jobs to environment threads
        numba_threads = int(os.environ.get('NUMBA_NUM_THREADS', '1'))
        reducer = UMAP(
            n_components=n_components,
            init='random',
            random_state=random_state,
            n_jobs=min(n_jobs, numba_threads)
        umap_arr = reducer.fit_transform(df.values)
    except RuntimeError as e:
        if "Cannot set NUMBA_NUM_THREADS" in str(e):
            # Graceful fallback to single-threaded
            warnings.warn(f"Threading conflict: {str(e)}. Falling back to single-thread")
            reducer = UMAP(
                n_components=n_components,
                init='random',
                random_state=random_state,
                n_jobs=1)
            umap_arr = reducer.fit_transform(df.values)
        else:
            raise
    
    umap_df = pd.DataFrame(
        umap_arr,
        index=df.index,
        columns=[f"UMAP{i+1}" for i in range(n_components)]
    )
    return umap_df
    
