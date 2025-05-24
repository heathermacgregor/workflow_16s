# ===================================== IMPORTS ====================================== #

# Standard Library Imports
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

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
import logging
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

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

# ==================================== FUNCTIONS ===================================== #

def table_to_dataframe(
    table: Union[Dict, Table]
) -> pd.DataFrame:
    """Convert a BIOM Table or dictionary to a samples × features DataFrame."""
    if isinstance(table, Table):
        df = table.to_dataframe(dense=True).T  # Samples × features
    elif isinstance(table, dict):
        df = pd.DataFrame(table)               # Samples × features 
    else:
        raise TypeError("Input must be a BIOM Table or dictionary.")
    return df

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
        table = table_to_dataframe(table)

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
DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

def t_test(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    col: str = DEFAULT_GROUP_COLUMN,
    col_values: List[Union[bool, int, str]] = DEFAULT_GROUP_COLUMN_VALUES
) -> pd.DataFrame:
    """
    Performs independent t-tests between groups for all features.
    
    Args:
        table:      Input abundance table (samples x features).
        metadata:   Sample metadata DataFrame (must contain the same samples as 
                    the table).
        col:        Metadata column containing group labels.
        col_values: Two group identifiers to compare.
        
    Returns:
        results:    Results sorted by p-value with test statistics, excluding 
                    features with p=0 or NaN.
    """
    # Convert input to DataFrame if necessary
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)
    
    # Sanitize indices (critical fix)
    table.index = table.index.astype(str).str.strip().str.lower()
    metadata = metadata.copy()  # Avoid modifying original metadata
    metadata.index = metadata.index.astype(str).str.strip().str.lower()

    # Check for column name conflict
    if col in table.columns:
        raise ValueError(
            f"Column '{col}' already exists in table. Choose different group column."
        )

    # Validate index alignment
    common_indices = table.index.intersection(metadata.index)
    if not common_indices.size:
        logger.error(f"Table samples: {table.index.tolist()[:5]}...")
        logger.error(f"Metadata samples: {metadata.index.tolist()[:5]}...")
        logger.error(
            "No common indices between table and metadata after sanitization."
        )
        table = table.T
        table.index = table.index.astype(str).str.strip().str.lower()

    # Merge with sanitized indices
    table_with_col = table.merge(
        metadata[[col]], 
        left_index=True, 
        right_index=True, 
        how='inner'  # Stricter merge to exclude samples without metadata
    )
    # Validate successful merge
    if table_with_col[col].isna().any():
        missing = table_with_col[col].isna().sum()
        raise ValueError(
            f"{missing} samples have NaN in '{col}' after merge. "
            f"Check metadata completeness."
        )

    # Rest of the t-test logic remains unchanged
    results = []
    for feature in table_with_col.columns.drop(col):
        mask_group1 = (table_with_col[col] == col_values[0])
        mask_group2 = (table_with_col[col] == col_values[1])
        
        group1_values = table_with_col.loc[mask_group1, feature].dropna()
        group2_values = table_with_col.loc[mask_group2, feature].dropna()
        
        if len(group1_values) < 1 or len(group2_values) < 1:
            continue
            
        t_stat, p_val = ttest_ind(group1_values, group2_values, equal_var=False)
        results.append(
            {'feature': feature, 't_statistic': t_stat, 't_test_p_value': p_val}
        )
    
    # Results processing
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.error(f"No features passed t-test for {col_values} in '{col}'")
        return pd.DataFrame(columns=['feature', 't_statistic', 't_test_p_value'])
    logger.info(results_df[
        results_df['t_test_p_value'].notna() & 
        (results_df['t_test_p_value'] > 0)
    ].sort_values('t_test_p_value').head())
    return results_df[
        results_df['t_test_p_value'].notna() & 
        (results_df['t_test_p_value'] > 0)
    ].sort_values('t_test_p_value')
    


from typing import Union, List, Optional, Dict
import pandas as pd
from scipy.stats import mannwhitneyu
import logging

logger = logging.getLogger(__name__)

def mwu_bonferroni(
    table: Union[Dict, 'Table', pd.DataFrame], 
    metadata: pd.DataFrame,
    col: str,
    col_values: List[Union[bool, int, str]],
    progress: Optional['Progress'] = None,
    parent_task_id: Optional['TaskID'] = None,
    level: Optional[str] = None
) -> pd.DataFrame:
    """
    Performs Mann-Whitney U tests with Bonferroni correction for two groups.

    Args:
        table:      Input abundance table (samples x features).
        metadata:   Sample metadata DataFrame.
        col:        Metadata column containing group labels.
        col_values: Two group identifiers to compare.
        progress:   Optional progress reporter.
        parent_task_id: Optional parent task ID for progress.
        level:      Optional feature level label for progress.

    Returns:
        A DataFrame of features with Bonferroni-corrected p-values below threshold.
    """
    # Convert input to DataFrame if necessary
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)

    # Sanitize indices
    table.index = table.index.astype(str).str.strip().str.lower()
    metadata = metadata.copy()
    metadata.index = metadata.index.astype(str).str.strip().str.lower()

    # Prevent column name conflicts
    if col in table.columns:
        raise ValueError(
            f"Column '{col}' already exists in the table. Choose a different group column."
        )

    # Align indices
    common_indices = table.index.intersection(metadata.index)
    if not common_indices.size:
        logger.error(f"Table samples: {table.index.tolist()[:5]}...")
        logger.error(f"Metadata samples: {metadata.index.tolist()[:5]}...")
        raise ValueError("No common indices between table and metadata after sanitization.")

    # Merge table with group column
    table_with_col = table.merge(
        metadata[[col]],
        left_index=True,
        right_index=True,
        how='inner'
    )

    if table_with_col[col].isna().any():
        raise ValueError(
            f"{table_with_col[col].isna().sum()} samples have NaN in '{col}' after merge. "
            f"Check metadata completeness."
        )

    # Setup
    features = table_with_col.columns.drop(col)
    total_features = len(features)
    threshold = 0.01 / total_features

    results = []
    subtask_id = None

    try:
        # Add progress subtask if applicable
        if progress and parent_task_id and level:
            desc = f"[dim]├─ MWU: {level[:15]}..." if len(level) > 15 else f"[dim]├─ MWU: {level}"
            subtask_id = progress.add_task(
                description=desc,
                total=total_features,
                parent=parent_task_id,
                transient=True
            )

        for feature in features:
            # Select group values
            mask1 = table_with_col[col] == col_values[0]
            mask2 = table_with_col[col] == col_values[1]
            group1 = table_with_col.loc[mask1, feature].dropna()
            group2 = table_with_col.loc[mask2, feature].dropna()

            if len(group1) < 1 or len(group2) < 1:
                continue

            # MWU test
            u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
            n1, n2 = len(group1), len(group2)
            effect_size_r = 1 - (2 * u_stat) / (n1 * n2)
            median_diff = group1.median() - group2.median()

            results.append({
                'feature': feature,
                'u_statistic': u_stat,
                'mwu_bonferroni_p_value': max(p_val, 1e-10),
                'median_difference': median_diff,
                'effect_size_r': effect_size_r
            })

            if progress and subtask_id:
                progress.advance(subtask_id)

    finally:
        if progress and subtask_id:
            progress.remove_task(subtask_id)

    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.error(f"No features passed the MWU test for groups {col_values} in column '{col}'")
        return pd.DataFrame(columns=['feature', 'u_statistic', 'mwu_bonferroni_p_value'])

    results_df = results_df[
        (results_df['mwu_bonferroni_p_value'] != 0) &
        (results_df['mwu_bonferroni_p_value'].notna())
    ]
    results_df.sort_values('mwu_bonferroni_p_value', inplace=True)

    # Filter with Bonferroni threshold
    results_filtered = results_df[results_df['mwu_bonferroni_p_value'] <= threshold]
    return results_filtered.sort_values('mwu_bonferroni_p_value')



def kruskal_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    col: str = DEFAULT_GROUP_COLUMN,
    col_values: List[Union[bool, int, str]] = None,
    progress: Optional[Progress] = None,
    parent_task_id: Optional[TaskID] = None,
    level: Optional[str] = None
) -> pd.DataFrame:
    """
    Performs Kruskal-Wallis H-test with Bonferroni correction for ≥3 groups.
    
    Args:
        table:      Input abundance table (samples x features)
        metadata:   Sample metadata DataFrame
        col:        Metadata column containing group labels
        col_values: List of group identifiers to compare (None = use all groups)
        
    Returns:
        results:    DataFrame with significant features after Bonferroni correction
    """
    # Convert input to DataFrame if necessary
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)
    
    # Sanitize indices (critical fix)
    table.index = table.index.astype(str).str.strip().str.lower()
    metadata = metadata.copy()  # Avoid modifying original metadata
    metadata.index = metadata.index.astype(str).str.strip().str.lower()

    # Check for column name conflict
    if col in table.columns:
        raise ValueError(f"Column '{col}' already exists in table. Choose different group column.")

    # Validate index alignment
    common_indices = table.index.intersection(metadata.index)
    if not common_indices.size:
        logger.error(f"Table samples: {table.index.tolist()[:5]}...")
        logger.error(f"Metadata samples: {metadata.index.tolist()[:5]}...")
        logger.error("No common indices between table and metadata after sanitization.")
        table = table.T
        table.index = table.index.astype(str).str.strip().str.lower()

    # Merge with sanitized indices
    table_with_col = table.merge(
        metadata[[col]], 
        left_index=True, 
        right_index=True, 
        how='inner'  # Stricter merge to exclude samples without metadata
    )
    # Validate successful merge
    if table_with_col[col].isna().any():
        missing = table_with_col[col].isna().sum()
        raise ValueError(f"{missing} samples have NaN in '{col}' after merge. Check metadata completeness.")

    
    # Get unique groups if col_values not specified
    if col_values is None:
        col_values = table_with_col[col].unique().tolist()
    
    # Pre-calculate Bonferroni threshold
    total_features = len(table_with_col.columns.drop(col))
    threshold = 0.01 / total_features
    
    features = table_with_col.columns.drop(col)
    total_features = len(features)
    
    results = []
    subtask_id = None
    
    try:
        # Create transient subtask if progress available
        if progress and parent_task_id is not None and level:
            subtask_desc = f"[dim]├─ Kruskal: {level[:15]}..." if len(level) > 15 else f"[dim]├─ Kruskal: {level}"
            subtask_id = progress.add_task(
                description=subtask_desc,
                total=total_features,
                parent=parent_task_id,
                transient=True
            )

        for feature in features:
            # Collect data for all groups
            groups = []
            for group_val in col_values:
                mask = (table_with_col[col] == group_val)
                group_data = table_with_col.loc[mask, feature].dropna()
                if len(group_data) > 0:  # Skip empty groups
                    groups.append(group_data)
            
            # Skip feature if <2 groups have data
            if len(groups) < 2:
                continue
            
            try:
                h_stat, p_val = kruskal(*groups)
            except ValueError:
                continue  # Handle identical values in all groups
                
            # Calculate effect size (epsilon squared)
            n_total = sum(len(g) for g in groups)
            epsilon_sq = h_stat / (n_total - 1)
            
            # Update progress if available
            if progress and subtask_id is not None:
                progress.advance(subtask_id)

            results.append({
                'feature': feature,
                'h_statistic': h_stat,
                'kruskal_bonferroni_p_value': max(p_val, 1e-10),
                'epsilon_squared': epsilon_sq,
                'groups_tested': len(groups)
            })
            
    finally:
        # Cleanup progress task
        if progress and subtask_id is not None:
            progress.remove_task(subtask_id)
        
        
    
    results_df = pd.DataFrame(results)
    if results_df.empty:
        logger.error(f"No features passed the t-test for groups: {col_values} in column '{col}'")
        return pd.DataFrame(columns=['feature', 't_statistic', 'kruskal_bonferroni_p_value'])

    results_df = results_df[(results_df['kruskal_bonferroni_p_value'] != 0) & (results_df['kruskal_bonferroni_p_value'].notna())]
    
    # Apply Bonferroni correction
    results_df = results_df[results_df['kruskal_bonferroni_p_value'] <= threshold]
    return results_df.sort_values('kruskal_bonferroni_p_value')


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
    # Convert input to DataFrame if necessary
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)
    
    # Sanitize indices (critical fix)
    table.index = table.index.astype(str).str.strip().str.lower()
    metadata = metadata.copy()  # Avoid modifying original metadata
    metadata.index = metadata.index.astype(str).str.strip().str.lower()

    # Check for column name conflict
    if col in table.columns:
        raise ValueError(f"Column '{col}' already exists in table. Choose different group column.")

    # Validate index alignment
    common_indices = table.index.intersection(metadata.index)
    if not common_indices.size:
        logger.error(f"Table samples: {table.index.tolist()[:5]}...")
        logger.error(f"Metadata samples: {metadata.index.tolist()[:5]}...")
        logger.error("No common indices between table and metadata after sanitization.")
        table = table.T
        table.index = table.index.astype(str).str.strip().str.lower()

    # Merge with sanitized indices
    table_with_col = table.merge(
        metadata[[col]], 
        left_index=True, 
        right_index=True, 
        how='inner'  # Stricter merge to exclude samples without metadata
    )
    # Validate successful merge
    if table_with_col[col].isna().any():
        missing = table_with_col[col].isna().sum()
        raise ValueError(
            f"{missing} samples have NaN in '{col}' after merge. "
            f"Check metadata completeness."
        )
    
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
            logger.error(f"Skipping feature '{feature}' due to error: {e}")
            continue
    
    results = pd.DataFrame(results).sort_values(by='r^2', ascending=False)
    return results
