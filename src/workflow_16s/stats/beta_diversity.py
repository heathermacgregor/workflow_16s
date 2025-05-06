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

def distance_matrix(
    table: Union[Dict, Table, pd.DataFrame], 
    metric: str = 'braycurtis'
):
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
        # Transpose it (samples x features)
        table = table.T
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
    
    dm = squareform(pdist(table, metric=metric))
    return dm#DistanceMatrix(dm, ids=table.index) 


def pcoa(
    table: Union[Dict, Table, pd.DataFrame], 
    metric: str = 'braycurtis',
    n: int = None
):
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
        # Transpose it (samples x features)
        table = table.T
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
      
    dm = distance_matrix(table, metric=metric)
    dm_df = pd.DataFrame(dm, index=table.index, columns=df.index)
    if n:
        return PCoA(dm_df, number_of_dimensions=n) 
    else:
        return PCoA(dm_df)


def pca(
    table: Union[Dict, Table, pd.DataFrame], 
    n: int = 20
):
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
        # Transpose it (samples x features)
        table = table.T
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
      
    table_standard = pd.DataFrame(
        StandardScaler().fit_transform(table), 
        columns=list(table.columns)
    )
    
    pca = PCA(n_components=n)
    
    components = pca.fit_transform(table_standard)
    components = pd.DataFrame(
        components, 
        index=table.index, 
        columns=[f"PC{i+1}" for i in range(n)]
    )
    
    exp_var_ratio = pca.explained_variance_ratio_
    logger.info(
        'Explained variation per principal component: {}'.format(exp_var_ratio)
    )
    
    exp_var_cumul = np.cumsum(pca.explained_variance_ratio_)
    
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    results = {
        'components': components, 
        'exp_var_ratio': exp_var_ratio, 
        'exp_var_cumul': exp_var_cumul, 
        'loadings': loadings
    }
    return results


# Other MDS
def tsne(
    table: Union[Dict, Table, pd.DataFrame], 
    n_components: int = 3, 
    random_state: int = 0
) -> pd.DataFrame:
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
        # Transpose it (samples x features)
        table = table.T
    if isinstance(table, Dict):
        table = pd.DataFrame(table)
        
    tsne_obj = TSNE(
        n_components=n_components,
        random_state=random_state
    ).fit_transform(table)
    
    tsne_df = pd.DataFrame(
      tsne_obj, 
      index=table.index, 
      columns=[f'TSNE{n+1}' for n in range(n_components)]
    )
    return tsne_df


def umap(
    table: Union[Dict, Table, pd.DataFrame], 
    n_components: int = 3, 
    random_state: int = 0
) -> pd.DataFrame:
    # Convert table to DataFrame if necessary
    if isinstance(table, Table):
        table = table.to_dataframe(dense=True)
        # Transpose it (samples x features)
        table = table.T
    if isinstance(table, Dict):
        table = pd.DataFrame(table)

    umap_obj = UMAP(
        n_components=n_components, 
        init='random', 
        random_state=random_state
    ).fit_transform(table)
    
    umap_df = pd.DataFrame(
        umap_obj, 
        index=table.index, 
        columns=[f'UMAP{n+1}' for n in range(n_components)]
    )
    return umap_df
