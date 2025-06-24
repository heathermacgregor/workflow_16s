# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from functools import reduce

# Third-Party Imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from biom import load_table
from biom.table import Table
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
    TaskID
)
from scipy.spatial.distance import pdist, squareform
from scipy.stats import kruskal, mannwhitneyu, spearmanr, ttest_ind
from skbio.stats.composition import clr as CLR
from skbio.stats.distance import DistanceMatrix
from skbio.stats.ordination import pcoa as PCoA
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.biom import (
    collapse_taxa, 
    convert_to_biom, 
    export_h5py, 
    presence_absence, 
    filter_presence_absence
)
from workflow_16s.utils.progress import create_progress
from workflow_16s.utils.file_utils import (
    import_merged_table_biom,
    import_merged_meta_tsv
)
from workflow_16s.utils import df_utils
from workflow_16s.utils.dir_utils import SubDirs
from workflow_16s.stats import beta_diversity 
from workflow_16s.stats.utils import (
    clr_transform_table,
    filter_table, 
    merge_table_with_metadata,
    normalize_table,
    table_to_dataframe
)
from workflow_16s.stats.tests import (
    fisher_exact_bonferroni, 
    kruskal_bonferroni, 
    mwu_bonferroni, 
    ttest 
)
from workflow_16s.figures.html_report import HTMLReport
from workflow_16s.figures.merged.merged import (
    mds as plot_mds, 
    pca as plot_pca, 
    pcoa as plot_pcoa, 
    sample_map_categorical
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
warnings.filterwarnings("ignore")  # Suppress warnings

# ================================= DEFAULT VALUES =================================== #

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

DEFAULT_PROGRESS_TEXT_N = 50

DEFAULT_GROUP_COLUMN = 'nuclear_contamination_status'
DEFAULT_GROUP_COLUMN_VALUES = [True, False]

# Ordination constants
DEFAULT_METRIC = 'braycurtis'
DEFAULT_N_PCA = 20
DEFAULT_N_PCOA = None
DEFAULT_N_TSNE = 3
DEFAULT_N_UMAP = 3
DEFAULT_RANDOM_STATE = 0

# ============================= STATISTICAL ANALYZER CLASS ============================ #
class StatisticalAnalyzer:
    """Handles all statistical analyses for amplicon data"""
    
    TEST_CONFIG = {
        'fisher': {
            'key': 'fisher',
            'func': fisher_exact_bonferroni,
            'name': 'Fisher test (w/ Bonferroni)',
            'effect_col': 'proportion_diff',
            'alt_effect_col': 'odds_ratio'
        },
        'ttest': {
            'key': 'ttest',
            'func': ttest,
            'name': 't-test',
            'effect_col': 'mean_difference',
            'alt_effect_col': 'cohens_d'
        },
        'mwu_bonferroni': {
            'key': 'mwub',
            'func': mwu_bonferroni,
            'name': 'Mann-Whitney U test (w/ Bonferroni)',
            'effect_col': 'effect_size_r',
            'alt_effect_col': 'median_difference'
        },
        'kruskal_bonferroni': {
            'key': 'kwb',
            'func': kruskal_bonferroni,
            'name': 'Kruskal-Wallis test (w/ Bonferroni)',
            'effect_col': 'epsilon_squared',
            'alt_effect_col': None
        }
    }
    
    def __init__(self, cfg: Dict, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
    
    def run_tests(
        self,
        table: Table,
        metadata: pd.DataFrame,
        group_column: str,
        group_values: List[Any],
        enabled_tests: List[str],
        progress: Optional[Progress] = None,
        task_id: Optional[TaskID] = None
    ) -> Dict[str, Any]:
        """
        Run statistical tests on a feature table
        
        Args:
            table:         BIOM feature table
            metadata:      Sample metadata
            group_column:  Column in metadata to group by
            group_values:  Values to compare in group_column
            enabled_tests: List of tests to run
            progress:      Rich progress object
            task_id:       Parent task ID
            
        Returns:
            Dictionary of test results keyed by test name
        """
        results = {}
        total_tests = len(enabled_tests)
        
        if progress and task_id:
            main_task = progress.add_task(
                f"[white]Running statistical tests", 
                total=total_tests,
                parent=task_id
            )
        
        for test_name in enabled_tests:
            if test_name not in self.TEST_CONFIG:
                continue
                
            config = self.TEST_CONFIG[test_name]
            test_key = config['key']
            
            if self.verbose:
                logger.info(f"Running {config['name']}...")
                
            results[test_key] = config['func'](
                table=table,
                metadata=metadata,
                group_column=group_column,
                group_column_values=group_values,
            )
            
            if progress and task_id:
                progress.update(main_task, advance=1)
                
        return results

    def get_effect_size(self, test_name: str, result_row: pd.Series) -> Optional[float]:
        """
        Extract effect size from statistical test results
        
        Args:
            test_name:  Name of statistical test
            result_row: Row from results DataFrame
            
        Returns:
            Effect size value or None if not found
        """
        if test_name not in self.TEST_CONFIG:
            return None
            
        config = self.TEST_CONFIG[test_name]
        effect_col = config['effect_col']
        alt_effect_col = config['alt_effect_col']
        
        if effect_col in result_row:
            return result_row[effect_col]
        elif alt_effect_col and alt_effect_col in result_row:
            return result_row[alt_effect_col]
        return None


# ================================= PLOTTER CLASS ================================== #
class Plotter:
    """Handles all visualization tasks for amplicon data"""
    
    def __init__(self, cfg: Dict, output_dir: Path, verbose: bool = False):
        self.cfg = cfg
        self.output_dir = output_dir
        self.verbose = verbose
        self.figures = {}
        self.ordination_results = {}
    
    def generate_sample_map(
        self, 
        metadata: pd.DataFrame,
        color_columns: List[str] = ['dataset_name', 'nuclear_contamination_status'],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate sample location maps
        
        Args:
            metadata:      Sample metadata with location data
            color_columns: Columns to use for coloring points
            **kwargs:      Additional plot arguments
            
        Returns:
            Dictionary of generated figures keyed by color column
        """
        figures = {}
        for color_col in color_columns:
            fig, _ = sample_map_categorical(
                metadata=metadata,
                output_dir=self.output_dir,
                color_col=color_col,
                **kwargs
            )
            figures[color_col] = fig
        return figures
    
    def compute_pca(
        self,
        table: Table,
        n_components: int = DEFAULT_N_PCA
    ) -> Dict[str, Any]:
        """
        Compute Principal Component Analysis (PCA) on a feature table.
        
        Args:
            table:        Input BIOM table
            n_components: Number of principal components to compute
            
        Returns:
            Dictionary containing PCA results
        """
        df = table_to_dataframe(table).T
        return pca(df, n_components)
    
    def compute_pcoa(
        self,
        table: Table,
        metric: str = DEFAULT_METRIC,
        n_dimensions: Optional[int] = DEFAULT_N_PCOA
    ) -> PCoA:
        """
        Compute Principal Coordinates Analysis (PCoA) on a feature table.
        
        Args:
            table:        Input BIOM table
            metric:       Distance metric
            n_dimensions: Number of dimensions to return
            
        Returns:
            PCoA results object
        """
        df = table_to_dataframe(table).T
        return pcoa(df, metric, n_dimensions)
    
    def compute_tsne(
        self,
        table: Table,
        n_components: int = DEFAULT_N_TSNE,
        random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
        """
        Compute t-SNE on a feature table.
        
        Args:
            table:        Input BIOM table
            n_components: Dimension of the embedded space
            random_state: Random seed
            
        Returns:
            DataFrame with t-SNE coordinates
        """
        df = table_to_dataframe(table).T
        return tsne(df, n_components, random_state)
    
    def compute_umap(
        self,
        table: Table,
        n_components: int = DEFAULT_N_UMAP,
        random_state: int = DEFAULT_RANDOM_STATE
    ) -> pd.DataFrame:
        """
        Compute UMAP on a feature table.
        
        Args:
            table:        Input BIOM table
            n_components: Dimension of the embedded space
            random_state: Random seed
            
        Returns:
            DataFrame with UMAP coordinates
        """
        df = table_to_dataframe(table).T
        return umap(df, n_components, random_state)
    
    def generate_pcoa_plot(
        self,
        pcoa_result: PCoA,
        metadata: pd.DataFrame,
        color_col: str = 'dataset_name',
        symbol_col: str = 'nuclear_contamination_status',
        transformation: Optional[str] = None,
        x: int = 1,
        y: int = 2,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Generate PCoA plot from precomputed results
        
        Args:
            pcoa_result:     PCoA results object
            metadata:        Sample metadata
            color_col:        Column for coloring points
            symbol_col:       Column for point symbols
            transformation:  Data transformation applied
            x:               Component for x-axis
            y:               Component for y-axis
            
        Returns:
            Tuple of figure and color dictionary
        """
        components = pcoa_result.samples
        proportion_explained = pcoa_result.proportion_explained
        
        fig, colordict = plot_pcoa(
            components=components,
            proportion_explained=proportion_explained,
            metadata=metadata,
            color_col=color_col,
            symbol_col=symbol_col,
            output_dir=self.output_dir,
            transformation=transformation,
            x=x,
            y=y,
            **kwargs
        )
        return fig, colordict
    
    def generate_pca_plot(
        self,
        pca_result: Dict[str, Any],
        metadata: pd.DataFrame,
        color_col: str = 'dataset_name',
        symbol_col: str = 'nuclear_contamination_status',
        transformation: Optional[str] = None,
        x: int = 1,
        y: int = 2,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Generate PCA plot from precomputed results
        
        Args:
            pca_result:      PCA results dictionary
            metadata:        Sample metadata
            color_col:        Column for coloring points
            symbol_col:       Column for point symbols
            transformation:  Data transformation applied
            x:               Component for x-axis
            y:               Component for y-axis
            
        Returns:
            Tuple of figure and color dictionary
        """
        components = pca_result['components']
        proportion_explained = pca_result['exp_var_ratio']
        
        fig, colordict = plot_pca(
            components=components,
            proportion_explained=proportion_explained,
            metadata=metadata,
            color_col=color_col,
            symbol_col=symbol_col,
            output_dir=self.output_dir,
            transformation=transformation,
            x=x,
            y=y,
            **kwargs
        )
        return fig, colordict
    
    def generate_mds_plot(
        self,
        coordinates: pd.DataFrame,
        metadata: pd.DataFrame,
        mode: str = 'UMAP',
        group_col: str = 'dataset_name',
        symbol_col: str = 'nuclear_contamination_status',
        transformation: Optional[str] = None,
        x: int = 1,
        y: int = 2,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Generate MDS plot (t-SNE or UMAP) from precomputed coordinates
        
        Args:
            coordinates:     Coordinates DataFrame
            metadata:        Sample metadata
            mode:            Reduction method ('UMAP' or 't-SNE')
            group_col:        Column for grouping points
            symbol_col:       Column for point symbols
            transformation:  Data transformation applied
            x:               Dimension for x-axis
            y:               Dimension for y-axis
            
        Returns:
            Tuple of figure and color dictionary
        """
        fig, colordict = plot_mds(
            df=coordinates,
            metadata=metadata,
            group_col=group_col,
            symbol_col=symbol_col,
            output_dir=self.output_dir,
            transformation=transformation,
            mode=mode,
            x=x,
            y=y,
            **kwargs
        )
        return fig, colordict
    
    def generate_ordination_plot(
        self,
        method: str,
        table: Table,
        metadata: pd.DataFrame,
        color_col: str = 'dataset_name',
        symbol_col: str = 'nuclear_contamination_status',
        transformation: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Generate ordination plot (PCA, PCoA, t-SNE, UMAP) with proper calculation and plotting separation
        
        Args:
            method:         Ordination method ('pca', 'pcoa', 'tsne', 'umap')
            table:          BIOM feature table
            metadata:       Sample metadata
            color_col:      Column for coloring points
            symbol_col:     Column for point symbols
            transformation: Data transformation applied
            
        Returns:
            Generated figure
        """
        # Compute ordination
        if method == 'pcoa':
            pcoa_result = self.compute_pcoa(table)
            self.ordination_results['pcoa'] = pcoa_result
            fig, _ = self.generate_pcoa_plot(
                pcoa_result, metadata, color_col, symbol_col, transformation, **kwargs
            )
        elif method == 'pca':
            pca_result = self.compute_pca(table)
            self.ordination_results['pca'] = pca_result
            fig, _ = self.generate_pca_plot(
                pca_result, metadata, color_col, symbol_col, transformation, **kwargs
            )
        elif method == 'tsne':
            coordinates = self.compute_tsne(table)
            self.ordination_results['tsne'] = coordinates
            fig, _ = self.generate_mds_plot(
                coordinates, metadata, 't-SNE', color_col, symbol_col, transformation, **kwargs
            )
        elif method == 'umap':
            coordinates = self.compute_umap(table)
            self.ordination_results['umap'] = coordinates
            fig, _ = self.generate_mds_plot(
                coordinates, metadata, 'UMAP', color_col, symbol_col, transformation, **kwargs
            )
        else:
            raise ValueError(f"Unsupported ordination method: {method}")
        
        return fig


# ================================== TOP FEATURES ANALYZER ================================== #
class TopFeaturesAnalyzer:
    """Identifies top features associated with experimental groups"""
    
    def __init__(self, cfg: Dict, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
    
    def analyze(
        self, 
        stats_results: Dict[str, Dict[str, Dict[str, pd.DataFrame]]],
        group_column: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Identify top features associated with group differences
        
        Args:
            stats_results: Nested dictionary of statistical results
            group_column:  Metadata column defining groups
            
        Returns:
            Tuple of (contaminated_features, pristine_features)
        """
        contaminated_features = []
        pristine_features = []
        analyzer = StatisticalAnalyzer(self.cfg, self.verbose)
        
        # Process each taxonomic level
        for level in ['phylum', 'class', 'order', 'family', 'genus']:
            level_features = {}
            
            # Collect results across all table types and tests
            for table_type, tests in stats_results.items():
                for test_name, test_results in tests.items():
                    if level not in test_results:
                        continue
                    
                    # Process each feature in results
                    for _, row in test_results[level].iterrows():
                        feature = row['feature']
                        p_value = row['p_value']
                        
                        # Skip non-significant results
                        if pd.isna(p_value) or p_value > 0.05:
                            continue
                            
                        # Get effect size
                        effect = analyzer.get_effect_size(test_name, row)
                        if effect is None:
                            continue
                            
                        # Track best result per feature
                        current = level_features.get(feature)
                        if not current or p_value < current['p_value']:
                            level_features[feature] = {
                                'p_value': p_value,
                                'effect': effect,
                                'table_type': table_type,
                                'test': test_name
                            }
            
            # Classify features by effect direction
            for feature, result in level_features.items():
                feature_data = {
                    'feature': feature,
                    'level': level,
                    'table_type': result['table_type'],
                    'test': result['test'],
                    'effect': result['effect'],
                    'p_value': result['p_value']
                }
                
                if result['effect'] > 0:
                    contaminated_features.append(feature_data)
                else:
                    pristine_features.append(feature_data)
        
        # Sort by effect size magnitude and significance
        key_func = lambda x: (-abs(x['effect']), x['p_value'])
        contaminated_features.sort(key=key_func)
        pristine_features.sort(key=key_func)
        
        return contaminated_features, pristine_features


# ================================== UTILITY FUNCTIONS ================================== #    

def filter_and_reorder_biom_and_metadata(
    table: Table,
    metadata_df: pd.DataFrame,
    sample_column: str = '#sampleid'
) -> tuple[Table, pd.DataFrame]:
    """
    Filters and reorders a BIOM table and metadata DataFrame so that sample IDs 
    match exactly and are aligned.

    Args:
        table:         The BIOM table (features x samples).
        metadata_df:   Metadata DataFrame with a sample ID column.
        sample_column: Column name in metadata_df with sample IDs.

    Returns:
        Tuple containing the filtered table and metadata_df.
    """
    # Lowercase and deduplicate sample IDs in metadata
    metadata_df = metadata_df.copy()
    metadata_df[sample_column] = metadata_df[sample_column].astype(str).str.lower()
    metadata_df = metadata_df.drop_duplicates(subset=[sample_column])

    # Lowercase biom sample IDs, mapping to original IDs
    original_sample_ids = table.ids(axis='sample')
    lowercase_to_original = {}
    for sid in original_sample_ids:
        key = sid.lower()
        if key in lowercase_to_original:
            raise ValueError(
                f"Duplicate lowercase sample ID found in BIOM table: "
                f"'{key}' from '{sid}' and '{lowercase_to_original[key]}'"
            )
        lowercase_to_original[key] = sid

    # Get shared sample IDs (in lowercase)
    metadata_ids = metadata_df[sample_column].tolist()
    shared_ids = [sid for sid in metadata_ids if sid in lowercase_to_original]

    # Filter metadata by shared IDs
    metadata_df = metadata_df.set_index(sample_column).loc[shared_ids].reset_index()

    # Map back to original-case sample IDs for biom
    ordered_biom_sample_ids = [lowercase_to_original[sid] 
                               for sid in metadata_df[sample_column]]

    # Filter and reorder BIOM table
    table_filtered = table.filter(
        ordered_biom_sample_ids, axis='sample', inplace=False
    )
    sample_index = {sid: i 
                    for i, sid in enumerate(table_filtered.ids(axis='sample'))}
    reordered_indices = [sample_index[sid] 
                         for sid in ordered_biom_sample_ids]
    data = table_filtered.matrix_data.toarray()
    reordered_data = data[:, reordered_indices]

    table_reordered = Table(
        reordered_data,
        observation_ids=table_filtered.ids(axis='observation'),
        sample_ids=ordered_biom_sample_ids
    )

    return table_reordered, metadata_df

def table_to_dataframe(
    table: Union[Dict[Any, Any], Table]
) -> pd.DataFrame:
    """
    Convert a BIOM Table or a mapping to a pandas DataFrame.

    Args:
        table: Input feature table, either a BIOM Table or a dict-like
               object where keys are sample identifiers and values are feature
               counts or abundances.

    Returns:
        df:    A pandas DataFrame of shape (n_samples, n_features).
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
    metric: str = DEFAULT_METRIC
) -> np.ndarray:
    """
    Compute a pairwise distance matrix from a feature table.

    Args:
        table:  Input feature table as a dict-like, BIOM Table, or DataFrame
                (samples x features or features x samples).
        metric: Distance metric name accepted by scipy.spatial.distance.pdist.

    Returns:
        dm:     A 2D numpy array representing the pairwise distance matrix.
    """
    if not isinstance(table, pd.DataFrame):
        table = table_to_dataframe(table)
    # Compute condensed distance vector and convert to square form
    dm = squareform(pdist(table.values, metric=metric))
    return dm

def pcoa(
    table: Union[Dict[Any, Any], Table, pd.DataFrame],
    metric: str = DEFAULT_METRIC,
    n_dimensions: Optional[int] = DEFAULT_N_PCOA
) -> PCoA:
    """
    Perform Principal Coordinates Analysis (PCoA) on a feature table.

    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples).
        metric:       Distance metric name for computing the distance matrix.
        n_dimensions: Number of dimensions to return; if None, returns all.

    Returns:
        A PCoAResults object containing eigenvalues, coordinates, and
        proportion of variance explained.
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
    n_components: int = DEFAULT_N_PCA
) -> Dict[str, Any]:
    """
    Perform Principal Component Analysis (PCA) on a feature table.

    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples).
        n_components: Number of principal components to compute.

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
    n_components: int = DEFAULT_N_TSNE,
    random_state: int = DEFAULT_RANDOM_STATE
) -> pd.DataFrame:
    """
    Compute t-distributed Stochastic Neighbor Embedding (t-SNE) reduction.

    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples).
        n_components: Dimension of the embedded space.
        random_state: Random seed for reproducibility.

    Returns:
        tsne_df:      DataFrame of shape (n_samples, n_components) with TSNE 
                      coordinates.
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
    n_components: int = DEFAULT_N_UMAP,
    random_state: int = DEFAULT_RANDOM_STATE
) -> pd.DataFrame:
    """
    Compute Uniform Manifold Approximation and Projection (UMAP) reduction.

    Args:
        table:        Input feature table as a dict-like, BIOM Table, or DataFrame
                      (samples x features or features x samples).
        n_components: Dimension of the embedded space.
        random_state: Random seed for reproducibility.

    Returns:
        umap_df:       DataFrame of shape (n_samples, n_components) with UMAP 
                       coordinates.
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


# ================================== AMPLICON DATA CLASS ================================== #
class AmpliconData:
    """Main class for processing amplicon sequencing data"""
    
    MODE_CONFIG = {
        'asv': ('table', 'asv'),
        'genus': ('table_6', 'l6')
    }
    
    def __init__(self, cfg, project_dir, mode='genus', verbose=False):
        self.cfg = cfg
        self.project_dir = project_dir
        self.mode = mode
        self.verbose = verbose
        self.meta = None
        self.table = None
        self.tables = {}
        self.stats = {}
        self.figures = {}
        self.ordination_results = {}
        self.top_contaminated_features = []
        self.top_pristine_features = []
        
        self._validate_mode()
        self._set_output_paths()
        self._load_data()
        self._process_data()
        self._run_analyses()
    
    def _validate_mode(self):
        """Validate processing mode"""
        if self.mode not in self.MODE_CONFIG:
            raise ValueError(f"Invalid processing mode: {self.mode}")
    
    def _set_output_paths(self):
        """Set output paths for processed data"""
        table_dir, output_dir = self.MODE_CONFIG[self.mode]
        self.figure_output_dir = Path(self.project_dir.figures)
        self.table_output_path = (
            Path(self.project_dir.data) / 'merged' / 'table' / 
            output_dir / 'feature-table.biom'
        )
        self.meta_output_path = (
            Path(self.project_dir.data) / 'merged' / 'metadata' / 
            'sample-metadata.tsv'
        )
    
    def _load_data(self):
        """Load metadata and feature tables"""
        self._load_metadata()
        self._load_biom_table()
        self._filter_and_align_data()
    
    def _load_metadata(self):
        """Load and merge metadata from multiple sources"""
        meta_paths = self._get_metadata_paths()
        self.meta = import_merged_meta_tsv(
            meta_paths, 
            None,
            self.verbose
        )
    
    def _get_metadata_paths(self) -> List[Path]:
        """Get paths to metadata files corresponding to BIOM tables"""
        meta_paths = []
        for biom_path in self._get_biom_paths():
            # Handle both file paths and directory paths
            dataset_dir = biom_path.parent if biom_path.is_file() else biom_path
            
            # Extract relevant path components
            tail_parts = dataset_dir.parts[-6:-1]
            
            # Construct metadata path
            meta_path = Path(self.project_dir.metadata_per_dataset).joinpath(
                *tail_parts, "sample-metadata.tsv"
            )
            if meta_path.exists():
                meta_paths.append(meta_path)
            
        if self.verbose:
            logger.info(f"Found {RED}{len(meta_paths)}{RESET} metadata files")
        return meta_paths
    
    def _load_biom_table(self):
        """Load and merge BIOM feature tables"""
        biom_paths = self._get_biom_paths()
        if not biom_paths:
            error_text = "No BIOM files found matching pattern"
            logger.error(error_text)
            raise FileNotFoundError(error_text)   
            
        self.table = import_merged_table_biom(
            biom_paths, 
            'table',
            self.verbose
        )
    
    def _get_biom_paths(self) -> List[Path]:
        """
        Get paths to BIOM feature tables using a glob pattern.
        
        Returns:
            List of Path objects to BIOM files
        """
        pattern = '/'.join(
          ['*', '*', '*', '*', 'FWD_*_REV_*', self.MODE_CONFIG[self.mode][0], 
           'feature-table.biom']
        )
        biom_paths = glob.glob(
            str(Path(self.project_dir.qiime_data_per_dataset) / pattern), 
            recursive=True
        )
        if self.verbose:
            logger.info(f"Found {RED}{len(biom_paths)}{RESET} feature tables")
        return [Path(p) for p in biom_paths]
    
    def _filter_and_align_data(self):
        """Filter and align BIOM table with metadata"""
        original_n_samples = self.table.shape[1]
        self.table, self.meta = filter_and_reorder_biom_and_metadata(
            table=self.table, metadata_df=self.meta, sample_column="#sampleid"
        )
        
        logger.info(
            f"Loaded (samples x features) metadata table with "
            f"{RED}{self.meta.shape[0]}{RESET} samples "
            f"and {RED}{self.meta.shape[1]}{RESET} columns"
        )

        if self.cfg["figures"]["map"]:
            self.plotter = Plotter(self.cfg, self.figure_output_dir, self.verbose)
            self.figures["map"] = self.plotter.generate_sample_map(self.meta)
            
        feature_type = 'genera' if self.mode == 'genus' else 'ASVs'
        logger.info(
            f"Loaded (features x samples) feature table with "
            f"{RED}{self.table.shape[1]} ({original_n_samples}){RESET} samples "
            f"and {RED}{self.table.shape[0]}{RESET} {feature_type}"
        )
    
    def _process_data(self):
        """Process data through filtering and transformation pipeline"""
        self._apply_preprocessing()
        self._collapse_taxa()
        self._create_presence_absence()
        self._save_tables()
    
    def _apply_preprocessing(self):
        """
        Apply filtering, normalization, and CLR transformation to the table before 
        collapsing.
        """
        # Start with the original table
        table = self.table
        self.tables["raw"] = {}
        self.tables["raw"][self.mode] = table
        
        filtering_enabled = self.cfg['features']['filter']
        normalization_enabled = (
            self.cfg['features']['filter'] and 
            self.cfg['features']['normalize']
        )
        clr_transformation_enabled = (
            self.cfg['features']['filter'] and 
            self.cfg['features']['normalize'] and 
            self.cfg['features']['clr_transform']
        )
        enabled_steps = [
            filtering_enabled, normalization_enabled, clr_transformation_enabled
        ]
        n_enabled_steps = sum(enabled_steps)

        with create_progress() as progress:
            main_task = progress.add_task(
                f"[white]Preprocessing {self.mode} tables".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=n_enabled_steps
            )
            # Apply filtering if enabled
            if filtering_enabled:
                if self.verbose:
                    logger.info("Applying filtering to table...")
                filtered_table = filter_table(table)
                self.tables["filtered"] = {}
                self.tables["filtered"][self.mode] = filtered_table
                progress.update(main_task, advance=1)
                table = filtered_table  # Use filtered table for next steps
            
            # Apply normalization if enabled (requires prior filtering)
            if normalization_enabled:
                if self.verbose:
                    logger.info("Applying normalization to table...")
                normalized_table = normalize_table(table, axis=1)
                self.tables["normalized"] = {}
                self.tables["normalized"][self.mode] = normalized_table
                progress.update(main_task, advance=1)
                table = normalized_table  # Use normalized table for next step
            
            # Apply CLR transformation if enabled (requires prior normalization)
            if clr_transformation_enabled:
                if self.verbose:
                    logger.info("Applying CLR transformation to table...")
                clr_transformed_table = clr_transform_table(table)
                self.tables["clr_transformed"] = {}
                self.tables["clr_transformed"][self.mode] = clr_transformed_table
                progress.update(main_task, advance=1)
    
    def _collapse_taxa(self):
        """Generate tables by collapsing taxa at different levels"""
        tax_levels = ['phylum', 'class', 'order', 'family', 'genus']
        
        # Process all table types
        for table_type in list(self.tables.keys()):
            if table_type not in self.tables:
                continue
                
            current_table = self.tables[table_type][self.mode]
            self.tables[table_type] = self._run_processing_step(
                process_name=f"Collapsing {table_type} taxonomy",
                process_func=collapse_taxa,
                levels=tax_levels,
                func_args=(),
                get_source=lambda level: current_table,
                log_template=f"Collapsed {table_type} to {{level}} level"
            )
    
    def _create_presence_absence(self):
        """Generate presence/absence tables if enabled in config"""
        if not self.cfg['features']['presence_absence']:
            return
            
        # Only create presence/absence for raw tables
        self.tables["presence_absence"] = self._run_processing_step(
            process_name="Converting to presence/absence",
            process_func=presence_absence,
            levels=['phylum', 'class', 'order', 'family', 'genus'],
            func_args=(),
            get_source=lambda level: self.tables["raw"][level]
        )
    
    def _run_processing_step(
        self,
        process_name: str,
        process_func: Callable,
        levels: List[str],
        func_args: tuple,
        get_source: Callable,
        log_template: Optional[str] = None,
        log_action: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Execute a processing step across multiple taxonomic levels.
        
        Args:
            process_name: Name of the processing step for logging
            process_func: Function to execute for each level
            levels:       Taxonomic levels to process
            func_args:    Additional arguments for process_func
            get_source:   Function to get input table for a level
            log_template: Template for logging messages
            log_action:   Action name for simple logging
            
        Returns:
            Dictionary of processed tables keyed by taxonomic level
        """
        processed_tables = {}
        
        if self.verbose:
            # Verbose mode: Use logging
            logger.info(f"{process_name}...")
            for level in levels:
                source_table = get_source(level)
                processed = process_func(source_table, level, *func_args)
                processed_tables[level] = processed
                self._log_level_action(level, log_template, log_action)
        else:
            # Non-verbose mode: Use progress bars
            with create_progress() as progress:
                task = progress.add_task(
                    f"[white]{process_name}...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                    total=len(levels)
                )
                for level in levels:
                    source_table = get_source(level)
                    processed = process_func(source_table, level, *func_args)
                    processed_tables[level] = processed
                    progress.update(task, advance=1)
                    
        return processed_tables

    def _log_level_action(self, level: str, template: Optional[str] = None, 
                         action: Optional[str] = None):
        """
        Log action for a specific taxonomic level.
        
        Args:
            level:    Taxonomic level being processed
            template: String template for logging (uses {level} placeholder)
            action:   Simple action description
        """
        if template:
            logger.info(template.format(level=level))
        elif action:
            logger.info(f"{level} {action}")
    
    def _save_tables(self):
        """Save all generated tables to appropriate directories"""
        # Calculate total tables to save
        total_tables = sum(len(level_tables) for level_tables in self.tables.values())
        
        with create_progress() as progress:
            task = progress.add_task(
                "[white]Saving tables...".ljust(DEFAULT_PROGRESS_TEXT_N), 
                total=total_tables
            )
            base_dir = Path(self.project_dir.data) / 'merged' / 'table'
            base_dir.mkdir(parents=True, exist_ok=True)
            for table_type, level_tables in self.tables.items():
                type_dir = base_dir / table_type
                type_dir.mkdir(parents=True, exist_ok=True)
                for level, table in level_tables.items():
                    level_dir = type_dir / level
                    level_dir.mkdir(parents=True, exist_ok=True)
                    output_path = level_dir / "feature-table.biom"
                    # Save table
                    export_h5py(table, output_path)
                    if self.verbose:
                        n_features, n_samples = table.shape
                        shape_str = f"[{n_features}, {n_samples}]"
                        logger.info(
                            f"Wrote {table_type} {level} table {shape_str} to '{output_path}'"
                        )
                    # Update progress after each table save
                    progress.update(task, advance=1)
    
    def _run_analyses(self):
        """Run statistical analyses and visualizations"""
        self._run_statistical_analyses()
        self._identify_top_features()
    
    def _run_statistical_analyses(self):
        """Run all configured statistical analyses"""
        self.stats_analyzer = StatisticalAnalyzer(self.cfg, self.verbose)
        
        # Create plotter if not already created
        if not hasattr(self, 'plotter'):
            self.plotter = Plotter(self.cfg, self.figure_output_dir, self.verbose)
        
        # Calculate total tasks
        total_main_tasks = 0
        total_plots = 0
        for table_type, tables in self.tables.items():
            enabled_tests = self._get_enabled_tests(table_type)
            if not enabled_tests:
                continue
            for level in tables:
                total_main_tasks += 1
                # Count plots for this level
                for test_name in enabled_tests:
                    if test_name in ['pca', 'pcoa', 'tsne', 'umap']:
                        total_plots += 1
        
        with create_progress() as progress:
            # Create main tasks
            main_task = progress.add_task(
                "[white]Running statistical analyses".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=total_main_tasks
            )
            plot_task = progress.add_task(
                "[cyan]Generating ordination plots".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=total_plots
            )
            
            for table_type, tables in self.tables.items():
                enabled_tests = self._get_enabled_tests(table_type)
                
                # Skip if no tests enabled for this table type
                if not enabled_tests:
                    continue
                
                # Initialize stats dictionary for this table type
                if table_type not in self.stats:
                    self.stats[table_type] = {}
                    
                for level, table in tables.items():
                    # Create a subtask for this level
                    level_task = progress.add_task(
                        f"[white]{table_type} {level}".ljust(DEFAULT_PROGRESS_TEXT_N),
                        total=len(enabled_tests),
                        parent=main_task
                    )
                    
                    # Run statistical tests
                    test_results = self.stats_analyzer.run_tests(
                        table=table,
                        metadata=self.meta,
                        group_column=DEFAULT_GROUP_COLUMN,
                        group_values=DEFAULT_GROUP_COLUMN_VALUES,
                        enabled_tests=enabled_tests,
                        progress=progress,
                        task_id=level_task
                    )
                    
                    # Store results for this level
                    self.stats[table_type][level] = test_results
                    
                    # Generate ordination plots
                    for test_name in enabled_tests:
                        if test_name in ['pca', 'pcoa', 'tsne', 'umap']:
                            try:
                                fig = self.plotter.generate_ordination_plot(
                                    test_name, table, self.meta,
                                    transformation=table_type
                                )
                                key = f"{test_name}_{table_type}_{level}"
                                self.figures[key] = fig
                                if self.verbose:
                                    logger.info(f"Generated {key} plot")
                            except Exception as e:
                                logger.error(f"{test_name} failed for {table_type}/{level}: {str(e)}")
                            finally:
                                # Update plot progress
                                progress.update(plot_task, advance=1)
                    
                    # Update main progress
                    progress.update(main_task, advance=1)
            
            # Save statistical results
            self._save_statistical_results()
    
    def _get_enabled_tests(self, table_type: str) -> List[str]:
        """Get enabled tests for a table type from config"""
        # Ensure config exists for this table type
        if table_type not in self.cfg['stats']:
            return []
            
        return [
            test for test in [
                'ttest', 'mwu_bonferroni', 'kruskal_bonferroni', 
                'pca', 'pcoa', 'tsne', 'umap'
            ] if self.cfg['stats'][table_type].get(test, False)
        ]
    
    def _save_statistical_results(self):
        """Save statistical analysis results to CSV files"""
        total_files = 0
        # Count total files to save for progress tracking
        for table_type, levels in self.stats.items():
            for level, tests in levels.items():
                total_files += len(tests)
        
        with create_progress() as progress:
            task = progress.add_task(
                "[white]Saving statistical results...".ljust(DEFAULT_PROGRESS_TEXT_N),
                total=total_files
            )
            base_dir = Path(self.project_dir.tables) / 'stats' / 'tests'
            base_dir.mkdir(parents=True, exist_ok=True) 
            for table_type, levels in self.stats.items():
                type_dir = base_dir / table_type
                type_dir.mkdir(parents=True, exist_ok=True)
                for level, tests in levels.items():
                    level_dir = type_dir / level
                    level_dir.mkdir(parents=True, exist_ok=True)
                    for test_name, result_df in tests.items():
                        output_path = level_dir / f"{test_name}.csv"
                        # Sort by p-value before saving
                        if 'p_value' in result_df.columns:
                            result_df = result_df.sort_values(by='p_value', ascending=True)
                        # Save DataFrame to CSV
                        result_df.to_csv(output_path, index=False)
                            
                        if self.verbose:
                            logger.info(
                                f"Saved {table_type}/{level}/{test_name} stats "
                                f"to {output_path}"
                            )
                                
                        # Update progress
                        progress.update(task, advance=1)
    
    def _identify_top_features(self):
        """Identify top differentiating features"""
        analyzer = TopFeaturesAnalyzer(self.cfg, self.verbose)
        self.top_contaminated_features, self.top_pristine_features = analyzer.analyze(
            self.stats, DEFAULT_GROUP_COLUMN
        )
        self._save_top_features()
    
    def _save_top_features(self):
        """Save top feature associations to CSV files"""
        # Create output directory
        output_dir = Path(self.project_dir.tables) / "top_features"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current timestamp for filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create DataFrames from feature lists
        contam_df = pd.DataFrame(self.top_contaminated_features)
        pristine_df = pd.DataFrame(self.top_pristine_features)
        
        # Add direction column
        if not contam_df.empty:
            contam_df['direction'] = 'contaminated'
            contam_path = output_dir / f"contaminated_features_{timestamp}.csv"
            contam_df.to_csv(contam_path, index=False)
            logger.info(f"Saved {len(contam_df)} contaminated features to {contam_path}")
        
        if not pristine_df.empty:
            pristine_df['direction'] = 'pristine'
            pristine_path = output_dir / f"pristine_features_{timestamp}.csv"
            pristine_df.to_csv(pristine_path, index=False)
            logger.info(f"Saved {len(pristine_df)} pristine features to {pristine_path}")
        
        # Create and save combined report
        if not contam_df.empty or not pristine_df.empty:
            combined_df = pd.concat([contam_df, pristine_df], ignore_index=True)
            combined_path = output_dir / f"all_significant_features_{timestamp}.csv"
            combined_df.to_csv(combined_path, index=False)
            logger.info(f"Saved {len(combined_df)} total features to {combined_path}")
            
            # Also save a version without timestamp for easy access
            latest_path = output_dir / "latest_significant_features.csv"
            combined_df.to_csv(latest_path, index=False)
            logger.info(f"Saved latest features to {latest_path}")
        else:
            logger.warning("No significant features found to save")


# ================================= MAIN EXECUTION ================================== #
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger('workflow_16s')
    
    # Sample configuration - should be replaced with real configuration
    cfg = {
        'features': {
            'filter': True,
            'normalize': True,
            'clr_transform': True,
            'presence_absence': True
        },
        'stats': {
            'raw': {
                'ttest': True,
                'mwu_bonferroni': True,
                'kruskal_bonferroni': True,
                'pca': True,
                'pcoa': True,
                'tsne': True,
                'umap': True
            },
            'filtered': {
                'ttest': True,
                'mwu_bonferroni': True,
                'kruskal_bonferroni': True,
                'pca': True,
                'pcoa': True
            },
            'normalized': {
                'ttest': True,
                'pca': True
            },
            'clr_transformed': {
                'ttest': True,
                'pca': True
            },
            'presence_absence': {
                'fisher': True
            }
        },
        'figures': {
            'map': True
        }
    }
    
    # Sample project directory structure
    class ProjectDirs:
        def __init__(self, base_path):
            self.base = Path(base_path)
            self.data = self.base / "data"
            self.figures = self.base / "figures"
            self.tables = self.base / "tables"
            self.qiime_data_per_dataset = self.base / "qiime_data_per_dataset"
            self.metadata_per_dataset = self.base / "metadata_per_dataset"
    
    # Create project directories
    base_path = Path.cwd() / "amplicon_analysis"
    project_dir = ProjectDirs(base_path)
    
    # Ensure directories exist
    project_dir.data.mkdir(parents=True, exist_ok=True)
    project_dir.figures.mkdir(parents=True, exist_ok=True)
    project_dir.tables.mkdir(parents=True, exist_ok=True)
    project_dir.qiime_data_per_dataset.mkdir(parents=True, exist_ok=True)
    project_dir.metadata_per_dataset.mkdir(parents=True, exist_ok=True)
    
    # Run the analysis
    try:
        logger.info("Starting amplicon data analysis")
        amplicon_data = AmpliconData(cfg, project_dir, mode='genus', verbose=True)
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise
