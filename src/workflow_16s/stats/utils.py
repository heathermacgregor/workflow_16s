# ===================================== IMPORTS ====================================== #
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from biom import Table
from rich.progress import (
    Progress, 
    BarColumn, 
    TextColumn, 
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
    SpinnerColumn,
    TaskID
)
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

# ================================= PROGRESS SETUP =================================== #

def create_progress() -> Progress:
    """Create a pre-configured Rich Progress instance."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        expand=True
    )

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

def _base_statistical_test(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    test_name: str,
    test_func: callable,
    progress: Optional[Progress] = None,
    parent_task_id: Optional[TaskID] = None,
    level: Optional[str] = None,
    **test_kwargs
) -> pd.DataFrame:
    """Base function for statistical tests with Rich progress integration."""
    merged = merge_table_with_metadata(table, metadata, group_column)
    features = merged.columns.drop(group_column)
    results = []
    
    # Progress management
    auto_progress = False
    if progress is None:
        progress = create_progress()
        auto_progress = True
        progress.start()

    task_desc = f"[bold cyan]{test_name}[/] ({level or 'all features'})"
    task_id = progress.add_task(
        description=task_desc,
        total=len(features),
        parent=parent_task_id
    )

    try:
        for feature in features:
            groups = [g.dropna() for _, g in merged.groupby(group_column)[feature]]
            if len(groups) < 2:
                progress.advance(task_id)
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
            
            progress.advance(task_id)
        
        # Update style for completed task
        progress.update(task_id, description=f"[dim]{task_desc}")
    
    finally:
        if auto_progress:
            progress.stop()

    return pd.DataFrame(results).dropna(subset=['p_value'])

# ============================== STATISTICAL INTERFACE =============================== #

def t_test(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    progress: Optional[Progress] = None,
    parent_task_id: Optional[TaskID] = None,
    level: Optional[str] = None
) -> pd.DataFrame:
    """Independent t-tests between groups with Rich progress."""
    df = table_to_dataframe(table)
    return _base_statistical_test(
        df, metadata, group_column,
        test_name="Student's t-test",
        test_func=ttest_ind,
        progress=progress,
        parent_task_id=parent_task_id,
        level=level,
        equal_var=False
    )

def mwu_bonferroni(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    progress: Optional[Progress] = None,
    parent_task_id: Optional[TaskID] = None,
    level: Optional[str] = None
) -> pd.DataFrame:
    """Mann-Whitney U tests with Bonferroni correction."""
    results = _base_statistical_test(
        table_to_dataframe(table), metadata, group_column,
        test_name="Mann-Whitney U",
        test_func=mannwhitneyu,
        progress=progress,
        parent_task_id=parent_task_id,
        level=level,
        alternative='two-sided'
    )
    threshold = 0.01 / len(results)
    return results[results.p_value <= threshold].sort_values('p_value')

def variability_explained(
    table: Union[Dict, Table, pd.DataFrame], 
    metadata: pd.DataFrame,
    group_column: str = DEFAULT_GROUP_COLUMN,
    progress: Optional[Progress] = None,
    parent_task_id: Optional[TaskID] = None,
    level: Optional[str] = None
) -> pd.DataFrame:
    """Calculate R² with Rich progress integration."""
    merged = merge_table_with_metadata(table_to_dataframe(table), metadata, group_column)
    X = merged[[group_column]]
    features = merged.columns.drop(group_column)
    r_squared = []
    
    auto_progress = False
    if progress is None:
        progress = create_progress()
        auto_progress = True
        progress.start()

    task_desc = f"[bold cyan]R² Analysis[/] ({level or 'all features'})"
    task_id = progress.add_task(
        description=task_desc,
        total=len(features),
        parent=parent_task_id
    )

    try:
        for feature in features:
            y = merged[[feature]]
            model = LinearRegression().fit(X, y)
            r_squared.append({
                'feature': feature,
                'r_squared': model.score(X, y)
            })
            progress.advance(task_id)
        
        progress.update(task_id, description=f"[dim]{task_desc}")
    
    finally:
        if auto_progress:
            progress.stop()

    return pd.DataFrame(r_squared).sort_values('r_squared', ascending=False)
