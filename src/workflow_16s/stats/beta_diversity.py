# ===================================== IMPORTS ====================================== #

# Standard Library
import os
import logging
import warnings
from typing import Any, Dict, Optional, Union

# Third-Party
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

# ============================== CRITICAL FIX FOR UMAP =============================== #
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
    """
    Validate that the input contains sufficient samples for analysis.
    
    Args:
        df:          Input data as pandas DataFrame with samples as rows.
        min_samples: Minimum required number of samples.
        
    Raises:
        ValueError: If number of samples is less than required minimum.
    """
    if len(df) < min_samples:
        raise ValueError(f"At least {min_samples} samples required")
        

def validate_component_count(n_components: int) -> None:
    """
    Validate that the requested number of components is valid.
    
    Args:
        n_components: Requested number of components.
        
    Raises:
        ValueError: If n_components is less than 1.
    """
    if n_components < 1:
        raise ValueError("n_components must be ≥ 1")
        

def safe_component_limit(df: pd.DataFrame, requested: int) -> int:
    """
    Determine safe number of components based on data dimensions.
    
    Computes the maximum possible components given:
    - For dimensionality reduction: min(n_samples - 1, n_features)
    - For distance-based methods: n_samples - 1
    
    Args:
        df:        Input data as pandas DataFrame.
        requested: Originally requested number of components.
        
    Returns:
        Safe number of components to compute (min(requested, max_possible)).
    """
    max_components = min(len(df) - 1, df.shape[1])
    return min(requested, max_components)


def create_result_dataframe(
    data: np.ndarray, 
    index: pd.Index, 
    prefix: str, 
    n_components: int
) -> pd.DataFrame:
    """
    Create standardized result DataFrame with named components.
    
    Args:
        data:         Embedding array of shape (n_samples, n_components).
        index:        Sample identifiers for DataFrame index.
        prefix:       Component name prefix (e.g., 'PC', 'UMAP').
        n_components: Number of components in the result.
        
    Returns:
        DataFrame with named components (prefix + number) and sample index.
    """
    columns = [f"{prefix}{i+1}" for i in range(n_components)]
    return pd.DataFrame(data, index=index, columns=columns)


def handle_duplicate_ids(ids: list) -> list:
    """
    Resolve duplicate sample IDs by appending numerical suffixes.
    
    Example: 
        Input: ['A', 'B', 'A'] → Output: ['A_1', 'B', 'A_2']
    
    Args:
        ids: List of original sample identifiers.
        
    Returns:
        List of unique identifiers with duplicates disambiguated.
    """
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
    """
    Compute pairwise distance matrix from feature abundance data.
    
    Args:
        table:  Input data in supported format (BIOM, dict, or DataFrame).
        metric: Distance metric (default: 'braycurtis'). Valid options include:
                'euclidean', 'jaccard', 'braycurtis', 'cityblock', etc.
                
    Returns:
        Symmetric distance matrix of shape (n_samples, n_samples).
        
    Raises:
        ValueError: If fewer than 2 samples are provided.
    """
    df = table_to_df(table)
    validate_min_samples(df, min_samples=2)
    return pairwise_distances(df.values, metric=metric)
    

def pcoa(
    table: Union[Dict, Table, pd.DataFrame], 
    metric: str = DEFAULT_METRIC, 
    n_dimensions: Optional[int] = DEFAULT_N_PCOA
) -> PCoA:
    """
    Perform Principal Coordinate Analysis (PCoA) on feature data.
    
    Also known as Metric Multidimensional Scaling (MDS). This method:
    1. Computes a distance matrix using the specified metric
    2. Performs eigenvalue decomposition on the distance matrix
    3. Returns principal coordinates and variance explained
    
    Args:
        table:        Input data in supported format.
        metric:       Distance metric (default: 'braycurtis').
        n_dimensions: Number of principal coordinates to return. 
                      If None, computes all possible components.
                     
    Returns:
        skbio PCoA result object containing:
        - samples: DataFrame of coordinates (n_samples × n_dimensions)
        - proportion_explained: Variance explained per component
        - eigenvalues: Component eigenvalues
        
    Raises:
        ValueError: For insufficient samples or invalid component count.
    """
    df = table_to_df(table)
    validate_min_samples(df, min_samples=2)
    
    # Handle duplicate sample IDs
    sample_ids = handle_duplicate_ids(df.index.tolist())
    
    # Compute distance matrix
    distance_array = pairwise_distances(df.values, metric=metric)
    distance_matrix = DistanceMatrix(distance_array, ids=sample_ids)
    
    # Determine safe number of dimensions
    n_dimensions = n_dimensions or len(df)
    n_dimensions = safe_component_limit(df, n_dimensions)
    
    # Perform PCoA
    pcoa_result = PCoA(distance_matrix, number_of_dimensions=n_dimensions)
    
    # Standardize component names
    pcoa_result.samples.columns = [f"PCo{i+1}" for i in range(n_dimensions)]
    return pcoa_result
    

def pca(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = DEFAULT_N_PCA
) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis (PCA) on feature data.
    
    This method:
    1. Standardizes features (mean=0, variance=1)
    2. Computes principal components via SVD
    3. Returns component scores and loadings
    
    Args:
        table:        Input data in supported format.
        n_components: Number of principal components to compute.
        
    Returns:
        Dictionary with:
        - 'components': DataFrame of component scores (n_samples × n_components)
        - 'exp_var_ratio': Explained variance ratio per component
        - 'exp_var_cumul': Cumulative explained variance
        - 'loadings': Feature loadings (n_features × n_components)
        
    Raises:
        ValueError: For insufficient samples or invalid component count.
    """
    df = table_to_df(table)
    validate_min_samples(df, min_samples=2)
    validate_component_count(n_components)
    
    # Determine safe component count
    n_components = safe_component_limit(df, n_components)
    
    # Standardize and transform
    scaled_data = StandardScaler().fit_transform(df.values)
    pca_model = PCA(n_components=n_components)
    scores = pca_model.fit_transform(scaled_data)
    
    return {
        'components': create_result_dataframe(scores, df.index, "PC", n_components),
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
    """
    Compute t-Distributed Stochastic Neighbor Embedding (t-SNE).
    
    Suitable for high-dimensional data visualization. This method:
    1. Models pairwise similarities in high-dimensional space
    2. Optimizes low-dimensional embedding to preserve local structures
    
    Args:
        table:        Input data in supported format.
        n_components: Dimension of embedding space (typically 2-3).
        random_state: Seed for reproducible results.
        n_jobs:       CPU cores to use (-1 for all available).
        
    Returns:
        DataFrame of t-SNE coordinates (n_samples × n_components)
        
    Raises:
        ValueError: For insufficient samples, invalid components, or data issues
    """
    df = table_to_df(table)
    validate_min_samples(df, min_samples=2)
    validate_component_count(n_components)
    
    # Validate data quality
    if not np.isfinite(df.values).all():
        raise ValueError("Input data contains non-finite values")
    
    n_components = safe_component_limit(df, n_components)
    
    # Compute t-SNE embedding
    tsne_model = TSNE(
        n_components=n_components,
        random_state=random_state,
        n_jobs=n_jobs
    )
    embeddings = tsne_model.fit_transform(df.values)
    return create_result_dataframe(embeddings, df.index, "TSNE", n_components)


def umap(
    table: Union[Dict, Table, pd.DataFrame],
    n_components: int = DEFAULT_N_UMAP,
    random_state: int = DEFAULT_RANDOM_STATE,
    n_jobs: int = DEFAULT_CPU_LIMIT
) -> pd.DataFrame:
    """
    Compute Uniform Manifold Approximation and Projection (UMAP).
    
    Preserves both local and global data structures. This method:
    1. Constructs topological representation of data
    2. Optimizes low-dimensional embedding
    
    Args:
        table:        Input data in supported format.
        n_components: Dimension of embedding space (typically 2-3).
        random_state: Seed for reproducible results.
        n_jobs:       CPU cores to use.
        
    Returns:
        DataFrame of UMAP coordinates (n_samples × n_components).
        
    Raises:
        ValueError: For insufficient samples or invalid components.
        RuntimeError: For threading issues (handled internally).
    """
    df = table_to_df(table)
    validate_min_samples(df, min_samples=2)
    validate_component_count(n_components)
    n_components = safe_component_limit(df, n_components)
    
    try:
        # Attempt UMAP with requested thread count
        reducer = UMAP(
            n_components=n_components,
            init='random',
            random_state=random_state,
            n_jobs=n_jobs
        )
        embeddings = reducer.fit_transform(df.values)
    except RuntimeError as e:
        if "threading" in str(e).lower():
            # Fallback to single-threaded execution
            reducer = UMAP(
                n_components=n_components,
                init='random',
                random_state=random_state,
                n_jobs=1
            )
            embeddings = reducer.fit_transform(df.values)
        else:
            raise
    
    return create_result_dataframe(embeddings, df.index, "UMAP", n_components)
