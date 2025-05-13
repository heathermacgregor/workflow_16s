# ===================================== IMPORTS ====================================== #

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd

from biom import Table

from scipy.stats import mannwhitneyu, kruskal, spearmanr, ttest_ind
from scipy.spatial.distance import pdist, squareform

from skbio.stats.composition import clr as CLR
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import PCoAResults, pcoa as PCoA

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from umap import UMAP

# ================================= GLOBAL VARIABLES ================================= #

# ==================================== FUNCTIONS ===================================== #

def table_to_dataframe(
    table: Union[Dict[Any, Any], Table]
) -> pd.DataFrame:
    """
    Convert a BIOM Table or a mapping to a pandas DataFrame.

    Args:
        table: Input feature table, either a BIOM Table or a dict-like
               object where keys are sample identifiers and values are feature
               counts or abundances

    Returns:
        A pandas DataFrame of shape (n_samples, n_features)
    """
    if isinstance(table, Table):
        # Convert BIOM Table to DataFrame (features x samples), then transpose
        df = table.to_dataframe(dense=True).T
    else:
        # Construct DataFrame directly from dict-like mapping
        df = pd.DataFrame(table)
    return df


def distance_matrix(
    table: Union[Dict[Any, Any], Table, pd.DataFrame],
    metric: str = 'braycurtis'
) -> np.ndarray:
    """
    Compute a pairwise distance matrix from a feature table.

    Args:
        table:  Input feature table as a dict-like, BIOM Table, or DataFrame
                (samples x features or features x samples)
        metric: Distance metric name accepted by scipy.spatial.distance.pdist

    Returns:
        A 2D numpy array representing the pairwise distance matrix
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)
    # Compute condensed distance vector and convert to square form
    dm = squareform(pdist(table.values, metric=metric))
    return dm


def pcoa(
    table: Union[Dict[Any, Any], Table, pd.DataFrame],
    metric: str = 'braycurtis',
    n_dimensions: Optional[int] = None
) -> PCoAResults:
    """
    Perform Principal Coordinates Analysis (PCoA) on a feature table.

    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples)
        metric:       Distance metric name for computing the distance matrix
        n_dimensions: Number of dimensions to return; if None, returns all

    Returns:
        A PCoAResults object containing eigenvalues, coordinates, and
        proportion of variance explained
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)
    # Compute distance matrix
    dm = distance_matrix(table, metric=metric)
    dm_df = pd.DataFrame(dm, index=table.index, columns=table.index)
    # Run PCoA
    if n_dimensions:
        return PCoA(dm_df, number_of_dimensions=n_dimensions)
    return PCoA(dm_df)


def pca(
    table: Union[Dict[Any, Any], Table, pd.DataFrame],
    n_components: int = 20
) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis (PCA) on a feature table.

    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples)
        n_components: Number of principal components to compute

    Returns:
        A dictionary with the following keys:
            - 'components': DataFrame of component scores (samples x components).
            - 'exp_var_ratio': Array of explained variance ratios per component.
            - 'exp_var_cumul': Cumulative explained variance ratios.
            - 'loadings': Array of PCA loadings (features x components).
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)

    # Standardize features
    scaled = StandardScaler().fit_transform(table.values)
    scaled_df = pd.DataFrame(scaled, index=table.index, columns=table.columns)

    # Fit PCA
    pca_model = PCA(n_components=n_components)
    scores = pca_model.fit_transform(scaled_df.values)

    # Prepare results
    components_df = pd.DataFrame(
        scores,
        index=table.index,
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
    table: Union[Dict[Any, Any], Table, pd.DataFrame],
    n_components: int = 3,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Compute t-distributed Stochastic Neighbor Embedding (t-SNE) reduction.

    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples)
        n_components: Dimension of the embedded space
        random_state: Random seed for reproducibility

    Returns:
        tsne_df:      DataFrame of shape (n_samples, n_components) with TSNE 
                      coordinates
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)

    tsne_arr = TSNE(
        n_components=n_components, 
        random_state=random_state
    ).fit_transform(
        table.values
    )
    tsne_df = pd.DataFrame(
        tsne_arr,
        index=table.index,
        columns=[f"TSNE{i+1}" for i in range(n_components)]
    )
    return tsne_df


def umap(
    table: Union[Dict[Any, Any], Table, pd.DataFrame],
    n_components: int = 3,
    random_state: int = 0
) -> pd.DataFrame:
    """
    Compute Uniform Manifold Approximation and Projection (UMAP) reduction.

    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples)
        n_components: Dimension of the embedded space
        random_state: Random seed for reproducibility

    Returns:
        umap_df:       DataFrame of shape (n_samples, n_components) with UMAP 
                       coordinates
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)

    umap_arr = UMAP(
        n_components=n_components,
        init='random',
        random_state=random_state
    ).fit_transform(table.values)
    umap_df = pd.DataFrame(
        umap_arr,
        index=table.index,
        columns=[f"UMAP{i+1}" for i in range(n_components)]
    )
    return umap_df
