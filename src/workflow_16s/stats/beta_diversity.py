# ===================================== IMPORTS ====================================== #

# Standard Library
import os
import logging
import warnings
from typing import Any, Dict, Optional, Union

# Third-Party
import multiprocessing
import numpy as np
import pandas as pd
from biom import Table
from multiprocessing import get_context
from scipy.spatial.distance import pdist, squareform
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
# Force thread-safe environment BEFORE importing UMAP
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from umap import UMAP

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.data import table_to_df

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
warnings.filterwarnings("ignore") 

# ================================= DEFAULT VALUES =================================== #

DEFAULT_METRIC = 'braycurtis'
DEFAULT_N_PCA = 20
DEFAULT_N_PCOA = None
DEFAULT_N_TSNE = 3
DEFAULT_N_UMAP = 3
DEFAULT_RANDOM_STATE = 0
DEFAULT_CPU_LIMIT = 1

# =============================== HELPER FUNCTIONS ==================================== #

def validate_min_samples(df: pd.DataFrame, min_samples: int = 2) -> None:
    """Validate sufficient samples for analysis."""
    if len(df) < min_samples:
        raise ValueError(f"At least {min_samples} samples required")
        

def validate_component_count(n_components: int) -> None:
    """Validate requested component count."""
    if n_components < 1:
        raise ValueError("n_components must be ≥ 1")
        

def safe_component_limit(df: pd.DataFrame, requested: int) -> int:
    """Determine safe number of components based on data dimensions."""
    max_components = min(len(df) - 1, df.shape[1])
    return min(requested, max_components)


def create_result_dataframe(
    data: np.ndarray, 
    index: pd.Index, 
    prefix: str, 
    n_components: int
) -> pd.DataFrame:
    """Create standardized result DataFrame with named components."""
    columns = [f"{prefix}{i+1}" for i in range(n_components)]
    return pd.DataFrame(data, index=index, columns=columns)


def handle_duplicate_ids(ids: list) -> list:
    """Resolve duplicate sample IDs by appending suffixes."""
    seen = {}
    new_ids = []
    for sample_id in ids:
        count = seen.get(sample_id, 0) + 1
        seen[sample_id] = count
        new_ids.append(f"{sample_id}_{count}" if count > 1 else sample_id)
    return new_ids

# =============================== CORE FUNCTIONALITY ================================== #

def distance_matrix(
    table: Union[Dict, Table, pd.DataFrame],
    metric: str = DEFAULT_METRIC
) -> np.ndarray:
    """Compute pairwise distance matrix from feature abundance data."""
    df = table_to_df(table)
    validate_min_samples(df, min_samples=2)
    return pairwise_distances(df.values, metric=metric)
    

def pcoa(
    table: Union[Dict, Table, pd.DataFrame], 
    metric: str = DEFAULT_METRIC, 
    n_dimensions: Optional[int] = DEFAULT_N_PCOA
) -> PCoA:
    """Perform Principal Coordinate Analysis (PCoA)."""
    df = table_to_df(table)
    validate_min_samples(df, min_samples=2)
    
    # Generate unique sample IDs
    sample_ids = handle_duplicate_ids(df.index.tolist())
    
    distance_array = pairwise_distances(df.values, metric=metric)
    if not np.isfinite(distance_array).all():
        raise ValueError("Distance matrix contains non-finite values")
    if not (distance_array >= 0).all():
        raise ValueError("Distance matrix contains negative values")
    if not np.allclose(distance_array, distance_array.T):
        raise ValueError("Distance matrix is not symmetric")
    distance_matrix = DistanceMatrix(distance_array, ids=sample_ids)
    
    # CORRECTED: Proper component limit for PCoA (n_samples-1)
    max_dims = len(df) - 1
    n_dimensions = n_dimensions or max_dims
    n_dimensions = min(n_dimensions, max_dims)
    
    pcoa_result = PCoA(distance_matrix, number_of_dimensions=n_dimensions)
    
    # CORRECTED: Use actual dimensions in result (accounts for negative eigenvalues)
    actual_dims = pcoa_result.samples.shape[1]
    new_axis_names = [f"PCo{i+1}" for i in range(actual_dims)]
    
    # Update names in all relevant attributes
    pcoa_result.samples.columns = new_axis_names
    pcoa_result.samples.index = sample_ids  # Ensure unique IDs in index
    
    # CORRECTED: Update proportion explained to match new axis names
    if hasattr(pcoa_result, 'proportion_explained'):
        pcoa_result.proportion_explained.index = new_axis_names
    
    return pcoa_result
    

def pca(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = DEFAULT_N_PCA
) -> Dict[str, Any]:
    """Perform Principal Component Analysis (PCA)."""
    df = table_to_df(table)
    validate_min_samples(df, min_samples=2)
    validate_component_count(n_components)
    
    n_components = safe_component_limit(df, n_components)
    
    # Generate unique sample IDs
    sample_ids = handle_duplicate_ids(df.index.tolist())
    
    scaled_data = StandardScaler().fit_transform(df.values)
    pca_model = PCA(n_components=n_components)
    scores = pca_model.fit_transform(scaled_data)
    
    return {
        # Use unique IDs for index
        'components': create_result_dataframe(scores, sample_ids, "PC", n_components),
        'exp_var_ratio': pca_model.explained_variance_ratio_,
        'exp_var_cumul': np.cumsum(pca_model.explained_variance_ratio_),
        'loadings': pca_model.components_.T * np.sqrt(pca_model.explained_variance_)
    }


def tsne(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = DEFAULT_N_TSNE,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_CPU_LIMIT
) -> pd.DataFrame:
    """Compute t-Distributed Stochastic Neighbor Embedding (t-SNE)."""
    df = table_to_df(table)
    validate_min_samples(df, min_samples=2)
    validate_component_count(n_components)
    
    if not np.isfinite(df.values).all():
        raise ValueError("Input data contains non-finite values")
    
    n_components = safe_component_limit(df, n_components)
    
    # Generate unique sample IDs
    sample_ids = handle_duplicate_ids(df.index.tolist())
    tsne_model = TSNE(
        n_components=n_components,
        random_state=random_state,
        n_jobs=n_jobs
    )
    embeddings = tsne_model.fit_transform(df.values)
    # Use unique IDs for index
    result = create_result_dataframe(embeddings, sample_ids, "TSNE", n_components)
    return result


def umap(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = DEFAULT_N_UMAP,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_CPU_LIMIT
) -> pd.DataFrame:
    """Compute Uniform Manifold Approximation and Projection (UMAP)."""
    df = table_to_df(table)
    print(type(df))
    validate_min_samples(df, min_samples=2)
    validate_component_count(n_components)
    n_components = safe_component_limit(df, n_components)
    
    # Generate unique sample IDs
    sample_ids = handle_duplicate_ids(df.index.tolist())
    print(type(sample_ids))
    # Execute in isolated process using top-level function
    with get_context("spawn").Pool(1) as pool:
        embeddings = pool.apply(
            _run_umap_isolated,
            (df.values, n_components, random_state)
        )
    
    # Use unique IDs for index
    reults = create_result_dataframe(embeddings, sample_ids, "UMAP", n_components)
    print(type(results))
    return results


def _run_umap_isolated(
    data: np.ndarray,
    n_components: int,
    random_state: int
) -> np.ndarray:
    os.environ['NUMBA_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    reducer = UMAP(
        n_components=n_components,
        init='random',
        random_state=random_state,
        n_jobs=1  # Force single-threaded in worker
    )
    return reducer.fit_transform(data)
