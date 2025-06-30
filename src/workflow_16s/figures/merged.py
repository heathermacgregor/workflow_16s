# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party Imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.figures.figures import (
    plotly_show_and_save,
    largecolorset,
    plot_legend,
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
sns.set_style('whitegrid')  # Set seaborn style globally
warnings.filterwarnings("ignore") # Suppress warnings

# ================================= GLOBAL VARIABLES ================================= #

DEFAULT_HEIGHT = 1000
DEFAULT_WIDTH = 1100

DEFAULT_COLOR_COL = 'dataset_name'
DEFAULT_SYMBOL_COL = 'nuclear_contamination_status'

DEFAULT_METRIC = 'braycurtis'

DEFAULT_PROJECTION = 'natural earth'
DEFAULT_LATITUDE_COL = 'latitude_deg'
DEFAULT_LONGITUDE_COL = 'longitude_deg'
DEFAULT_SIZE_MAP = 5
DEFAULT_OPACITY_MAP = 0.3

DEFAULT_FEATURE_TYPE = 'ASV'

DEFAULT_FEATURE_TYPE_ANCOM = 'l6'
DEFAULT_COLOR_COL_ANCOM = 'p'

# ================================== CORE HELPERS =================================== #

def _validate_metadata(
    metadata: pd.DataFrame, 
    required_cols: List[str]
) -> None:
    """
    Validate presence of required columns in metadata.
    
    Args:
        metadata:      DataFrame containing sample metadata.
        required_cols: List of column names required for visualization.
        
    Raises:
        ValueError: If any required columns are missing.
    """
    missing = [col for col in required_cols if col not in metadata.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def _prepare_visualization_data(
    components: pd.DataFrame,
    metadata: pd.DataFrame,
    color_col: str,
    symbol_col: str,
    placeholder: str = 'unknown',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Prepare merged component and metadata data for visualization.
    
    Args:
        components:  DataFrame with ordination components.
        metadata:    DataFrame with sample metadata.
        color_col:   Column to use for point coloring.
        symbol_col:  Column to use for point symbols.
        placeholder: Value to use for missing metadata.
        verbose:     Enable debug logging.
        
    Returns:
        Merged DataFrame ready for visualization
        
    Raises:
        ValueError: If no common samples exist between datasets
    """
    # Create copies to avoid modifying originals
    comp_copy = components.copy()
    meta_copy = metadata.copy()
    
    # Standardize indices to lowercase strings with whitespace trimming
    comp_copy.index = comp_copy.index.astype(str).str.strip().str.lower()
    
    # Handle metadata index - prefer '#sampleid' column if available
    if '#sampleid' in meta_copy.columns:
        meta_copy['#sampleid'] = meta_copy['#sampleid'].astype(str).str.strip().str.lower()
        meta_copy.index = meta_copy['#sampleid']
        if verbose:
            logger.debug("Set metadata index from '#sampleid' column")
    else:
        if verbose:
            logger.warning(
                "Metadata missing '#sampleid' column - using existing index"
            )
        meta_copy.index = meta_copy.index.astype(str).str.strip().str.lower()
        
    if verbose:
        # Log sample IDs for debugging
        logger.debug(f"Components index (first 5): {comp_copy.index.tolist()[:5]}")
        logger.debug(f"Metadata index (first 5): {meta_copy.index.tolist()[:5]}")
    
    # Find common samples
    common_idx = comp_copy.index.intersection(meta_copy.index)
    if verbose:
        logger.info(
            f"Found {len(common_idx)} common samples between components and metadata"
        )
    
    # Handle no common samples case with detailed diagnostics
    if len(common_idx) == 0:
        comp_samples = set(comp_copy.index)
        meta_samples = set(meta_copy.index)
        
        comp_only = comp_samples - meta_samples
        meta_only = meta_samples - comp_samples

        if verbose:
            logger.critical(
                "CRITICAL ERROR: No common samples between components and metadata!"
            )
            logger.critical(
                f"Components-only samples ({len(comp_only)}): "
                f"{list(comp_only)[:5]}{'...' if len(comp_only) > 5 else ''}"
            )
            logger.critical(
                f"Metadata-only samples ({len(meta_only)}): "
                f"{list(meta_only)[:5]}{'...' if len(meta_only) > 5 else ''}"
            )
        
        # Look for partial matches
        partial_matches = []
        for comp_id in list(comp_samples)[:10]:  # Check first 10
            for meta_id in meta_samples:
                if comp_id in meta_id or meta_id in comp_id:
                    partial_matches.append(f"{comp_id} ~ {meta_id}")
                    break
        
        if partial_matches:
            if verbose:
                logger.critical(f"Possible partial matches: {partial_matches[:5]}")
        
        raise ValueError("No common samples between components and metadata")
    
    # Filter to common samples
    meta_filtered = meta_copy.loc[common_idx].copy()
    comp_filtered = comp_copy.loc[common_idx].copy()
    
    # Handle missing metadata columns
    for col in [color_col, symbol_col]:
        if col not in meta_filtered.columns:
            if verbose:
                logger.warning(
                    f"Column '{col}' missing from metadata. "
                    f"Creating placeholder column."
                )
            meta_filtered[col] = placeholder
    
    # Merge components with metadata
    merged = comp_filtered.join(
        meta_filtered[[color_col, symbol_col]], 
        how='inner'
    )
    
    # Fill missing values in metadata columns
    for col in [color_col, symbol_col]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(placeholder)
    if verbose:
        logger.debug(f"Merged data shape: {merged.shape}")
    return merged


def _create_colordict(
    data: Union[pd.Series, pd.DataFrame], 
    color_set: List[str] = largecolorset
) -> Dict[str, str]:
    """
    Create consistent color mapping for categories.
    
    Args:
        data:      Series or single-column DataFrame containing categorical values.
        color_set: List of colors to use for mapping.
        
    Returns:
        Dictionary mapping categories to colors.
    """
    # Handle DataFrame input (extract first column)
    if isinstance(data, pd.DataFrame):
        if data.shape[1] != 1:
            print(data)
            raise ValueError("Color data must be a single column")
        data = data.iloc[:, 0]
    
    categories = sorted(data.astype(str).unique())
    return {c: color_set[i % len(color_set)] for i, c in enumerate(categories)}


def _save_figure_and_legend(
    fig: go.Figure,
    colordict: Dict[str, str],
    color_col: str,
    output_dir: Path,
    file_stem: str,
    show: bool,
    verbose: bool
) -> None:
    """
    Save figure and corresponding legend.
    
    Args:
        fig:        Plotly figure to save.
        colordict:  Color mapping dictionary.
        color_col:  Name of the coloring column.
        output_dir: Directory to save outputs.
        file_stem:  Base filename without extension.
        show:       Display figure interactively.
        verbose:    Enable debug logging.
    """
    plotly_show_and_save(fig, show, output_dir / file_stem, ['png', 'html'], verbose)
    plot_legend(colordict, color_col, output_dir / f"{file_stem}.legend.png")


def _create_base_scatter_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    symbol_col: str,
    colordict: Dict[str, str],
    hover_data: List[str]
) -> go.Figure:
    """
    Create standardized scatter plot configuration.
    
    Args:
        data:       DataFrame containing visualization data.
        x_col:      Column name for x-axis values.
        y_col:      Column name for y-axis values.
        color_col:  Column name for coloring points.
        symbol_col: Column name for point symbols.
        colordict:  Color mapping dictionary.
        hover_data: Additional columns to show in hover info.
        
    Returns:
        Configured Plotly scatter plot
    """
    return px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        symbol=symbol_col,
        color_discrete_map=colordict,
        hover_data=hover_data,
        opacity=0.8,
        size_max=10
    )


def _apply_common_layout(
    fig: go.Figure,
    x_title: str,
    y_title: str,
    title: str = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH
) -> go.Figure:
    """
    Apply consistent layout to figures.
    
    Args:
        fig:     Plotly figure to configure.
        x_title: Label for x-axis.
        y_title: Label for y-axis.
        title:   Overall plot title.
        height:  Figure height in pixels.
        width:   Figure width in pixels.
        
    Returns:
        Configured Plotly figure.
    """
    layout_updates = {
        'template': 'heather',
        'height': height,
        'width': width,
        'plot_bgcolor': '#fff',
        'font_size': 45,
        'showlegend': False,
        'xaxis': {
            'showticklabels': False,
            'zeroline': True,
            'showline': True,
            'linewidth': 2,
            'linecolor': 'black',
            'mirror': True
        },
        'yaxis': {
            'showticklabels': False,
            'zeroline': True,
            'showline': True,
            'linewidth': 2,
            'linecolor': 'black',
            'mirror': True
        }
    }
    
    if title:
        layout_updates.update({
            'title_text': title,
            'title_x': 0.5
        })
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
        **layout_updates
    )
    return fig


# ================================ VISUALIZATIONS ================================== #

def create_geographical_map(
    metadata: pd.DataFrame, 
    color_col: str = DEFAULT_COLOR_COL,
    lat_col: str = DEFAULT_LATITUDE_COL,
    lon_col: str = DEFAULT_LONGITUDE_COL,
    projection: str = DEFAULT_PROJECTION,
    output_dir: Union[Path, None] = None, 
    show: bool = False,
    verbose: bool = False,
    size: int = DEFAULT_SIZE_MAP,
    opacity: float = DEFAULT_OPACITY_MAP
) -> Tuple[go.Figure, Dict]:
    """
    Generate interactive geographical map of samples.
    
    Args:
        metadata:   DataFrame containing geographic coordinates.
        color_col:  Column to use for coloring points.
        lat_col:    Column containing latitude values.
        lon_col:    Column containing longitude values.
        projection: Map projection type.
        output_dir: Directory to save outputs.
        show:       Display figure interactively.
        verbose:    Enable debug logging.
        size:       Marker size.
        opacity:    Marker opacity.
        
    Returns:
        Tuple containing figure and color mapping dictionary.
    """
    # Preprocess data
    metadata = metadata.copy()
    metadata[color_col] = metadata[color_col].fillna('other').replace('', 'other')
    metadata = metadata.sort_values(color_col)
    
    # Count samples per category
    cat_counts = metadata[color_col].value_counts().reset_index()
    cat_counts.columns = [color_col, 'sample_count']
    metadata = metadata.merge(cat_counts, on=color_col, how='left')
    
    # Create visualization
    colordict = _create_colordict(metadata[color_col])
    
    fig = px.scatter_geo(
        metadata,
        lat=lat_col,
        lon=lon_col,
        color=color_col,
        color_discrete_map=colordict,
        hover_data={color_col: True, 'sample_count': ':.0f'}
    )
    
    # Configure map
    fig.update_geos(
        projection_type=projection,
        resolution=50,
        showcoastlines=True, coastlinecolor="#b5b5b5",
        showland=True, landcolor="#e8e8e8",
        showlakes=True, lakecolor="#fff",
        showrivers=True, rivercolor="#fff",
    )
    
    # Update layout
    fig.update_layout(margin=dict(l=5, r=5, t=5, b=5))
    fig.update_traces(marker=dict(size=size, opacity=opacity))
    
    # Save output
    if output_dir:
        file_stem = f"sample_map.{color_col}"
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_figure_and_legend(
            fig, colordict, color_col, output_dir, file_stem, show, verbose
        )
        
    return fig, colordict


def create_ordination_plot(
    components: pd.DataFrame,
    metadata: pd.DataFrame,
    ordination_type: str,
    proportion_explained: np.ndarray = None,
    color_col: str = DEFAULT_COLOR_COL,
    symbol_col: str = DEFAULT_SYMBOL_COL,
    dimensions: Tuple[int, int] = (1, 2),
    transformation: str = None,
    output_dir: Union[Path, None] = None,
    show: bool = False,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """
    Generate ordination plot (PCA/PCoA/MDS).
    
    Args:
        components:           DataFrame with ordination results.
        metadata:             DataFrame with sample metadata.
        ordination_type:      Type of ordination (PCA, PCoA, MDS).
        proportion_explained: Variance explained per dimension.
        color_col:            Column to use for coloring points.
        symbol_col:           Column to use for point symbols.
        dimensions:           Tuple of dimensions to plot (x,y).
        transformation:       Data transformation applied.
        output_dir:           Directory to save outputs.
        show:                 Display figure interactively.
        verbose:              Enable debug logging.
        
    Returns:
        Tuple containing figure and color mapping dictionary.
    """
    # Validate inputs
    _validate_metadata(metadata, [color_col, symbol_col, '#sampleid'])
    if not isinstance(color_col, str):
        raise TypeError(f"color_col must be a string, got {type(color_col)}")
    
    # Prepare data
    data = _prepare_visualization_data(
        components, metadata, color_col, symbol_col, verbose=verbose
    )
    data['sample_id'] = data.index
    
    # Create colormap
    colordict = _create_colordict(data[color_col])
    
    # Determine axis columns and titles
    prefix_map = {
        'PCA': 'PC',
        'PCoA': 'PCo',
        'MDS': ordination_type
    }
    prefix = prefix_map.get(ordination_type, ordination_type)
    
    x_dim, y_dim = dimensions
    x_col = f'{prefix}{x_dim}'
    y_col = f'{prefix}{y_dim}'
    
    # Verify dimension columns exist
    if x_col not in data.columns:
        available_dims = [col for col in data.columns if col.startswith(prefix)]
        raise ValueError(
            f"Column '{x_col}' not found. Available: "
            f"{available_dims[:5]}{'...' if len(available_dims) > 5 else ''}"
        )
    
    if y_col not in data.columns:
        available_dims = [col for col in data.columns if col.startswith(prefix)]
        raise ValueError(
            f"Column '{y_col}' not found. Available: "
            f"{available_dims[:5]}{'...' if len(available_dims) > 5 else ''}"
        )
    
    # Create axis titles
    if proportion_explained is not None and len(proportion_explained) >= max(x_dim, y_dim):
        x_title = f"{x_col} ({proportion_explained[x_dim-1]*100:.1f}%)"
        y_title = f"{y_col} ({proportion_explained[y_dim-1]*100:.1f}%)"
    else:
        x_title, y_title = x_col, y_col

    hover_data = ['sample_id', color_col, symbol_col]
    # Create plot
    fig = _create_base_scatter_plot(
        data,
        x_col,
        y_col,
        color_col,
        symbol_col,
        colordict,
        hover_data=hover_data
    )
    
    # Apply layout
    title = f'{ordination_type}: {transformation.title() if transformation else "Raw Data"}'
    fig = _apply_common_layout(fig, x_title, y_title, title)
    
    # Save output
    if output_dir:
        plot_dir = output_dir / ordination_type.lower()
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"{ordination_type.lower()}.{transformation or 'raw'}.{x_dim}-{y_dim}.{color_col}"
        _save_figure_and_legend(
            fig, colordict, color_col, plot_dir, file_stem, show, verbose
        )
        
    return fig, colordict


def create_heatmap(
    data: pd.DataFrame,
    feature_type: str = DEFAULT_FEATURE_TYPE,
    output_dir: Union[Path, None] = None,
    show: bool = False
) -> go.Figure:
    """
    Generate feature abundance heatmap.
    
    Args:
        data:         Abundance matrix (features x samples).
        feature_type: Type of features (ASV, OTU, etc.).
        output_dir:   Directory to save outputs.
        show:         Display figure interactively.
        
    Returns:
        Plotly heatmap figure
    """
    fig = px.imshow(
        data,
        color_continuous_scale='viridis',
        labels={'x': 'Samples', 'y': feature_type, 'color': 'Abundance'},
        title=f"{feature_type} Abundance Heatmap"
    )
    
    fig.update_layout(
        template='heather',
        height=1200,
        xaxis_showticklabels=False,
        yaxis_showticklabels=False
    )
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"heatmap.{feature_type.lower()}"
        plotly_show_and_save(fig, show, output_dir / file_stem)
    
    return fig


def create_ubiquity_plot(
    cm: np.ndarray, 
    pm: np.ndarray, 
    ubi_c: np.ndarray, 
    ubi_p: np.ndarray, 
    contaminated: List[str], 
    pristine: List[str], 
    transformation: str = None,
    output_dir: Union[Path, None] = None,
    show: bool = False
) -> go.Figure:
    """
    Generate ubiquity comparison plot.
    
    Args:
        cm:             Mean abundances in contaminated samples.
        pm:             Mean abundances in pristine samples.
        ubi_c:          Ubiquity values in contaminated samples.
        ubi_p:          Ubiquity values in pristine samples.
        contaminated:   IDs of contaminated samples.
        pristine:       IDs of pristine samples.
        transformation: Data transformation applied.
        output_dir:     Directory to save outputs.
        show:           Display figure interactively.
        
    Returns:
        Plotly scatter figure
    """
    # Calculate marker sizes and hover text
    sizes = [(v/len(contaminated)) + (ubi_p[i]/len(pristine)) for i, v in enumerate(ubi_c)]
    text = [
        f'Ubiq C = {v/len(contaminated):.3g}<br>Ubiq P = {ubi_p[i]/len(pristine):.3g}'
        f'<br>Mean C = {cm[i]:.3g}<br>Mean P = {pm[i]:.3g}'
        for i, v in enumerate(ubi_c)
    ]
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cm,
        y=pm,
        mode='markers',
        marker_size=10 * np.array(sizes),
        text=text
    ))
    
    fig.update_layout(
        template='heather',
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        title='Feature Ubiquity Comparison',
        xaxis_title='Contaminated',
        yaxis_title='Pristine'
    )

    if output_dir:
        plot_dir = output_dir / 'ubiquity'
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"ubiquity.{transformation or 'raw'}"
        plotly_show_and_save(fig, show, plot_dir / file_stem)
    
    return fig


def create_violin_plot(
    data: pd.DataFrame,
    feature: str,
    status_col: str = DEFAULT_SYMBOL_COL,
    output_dir: Union[Path, None] = None,
    sub_dir: str = 'violin',
    show: bool = False
) -> go.Figure:
    """
    Generate feature distribution violin plot.
    
    Args:
        data:       DataFrame containing feature abundances and metadata.
        feature:    Feature name to visualize.
        status_col: Column containing contamination status.
        output_dir: Directory to save outputs.
        sub_dir:    Subdirectory for output.
        show:       Display figure interactively.
        
    Returns:
        Plotly violin figure
    """
    plot_data = data.reset_index().dropna(subset=[feature, status_col])
    
    fig = px.violin(
        plot_data, 
        y=feature, 
        x=status_col,
        box=True,
        points="all",
        title=f"{feature.replace('_', ' ').title()} Distribution",
        hover_data=['index', 'dataset_name']
    )
    
    fig.update_layout(
        template='heather',
        xaxis_title="Contamination Status",
        yaxis_title=feature.replace('_', ' ').title()
    )

    if output_dir:
        plot_dir = output_dir / sub_dir
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f'{status_col}.{feature}'.lower()
        plotly_show_and_save(fig, show, plot_dir / file_stem)
    
    return fig


def create_ancom_plot(
    data: pd.DataFrame,
    min_W: float,
    feature_type: str = "l6",
    output_dir: Union[Path, None] = None,
    show: bool = False,
    reverse_x_axis: bool = True
) -> Tuple[go.Figure, Dict]:
    """
    Generate ANCOM volcano plot.
    
    Args:
        data:           ANCOM results DataFrame.
        min_W:          Significance threshold for W statistic.
        feature_type:   Taxonomic level (l2-l7).
        output_dir:     Directory to save outputs.
        show:           Display figure interactively.
        reverse_x_axis: Flip CLR values direction.
        
    Returns:
        Tuple containing figure and empty dictionary
    """
    # Optionally reverse CLR values
    if reverse_x_axis:
        data = data.assign(clr=-data['clr'])
    
    fig = px.scatter(
        data, 
        x='clr', 
        y='W', 
        hover_data=['Feature'],
        color='p',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        template='heather',
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
        xaxis_title='CLR',
        yaxis_title='W statistic'
    )
    
    # Add significance threshold
    fig.add_shape(
        type='line',
        y0=min_W,
        y1=min_W,
        x0=data['clr'].min(),
        x1=data['clr'].max(),
        line=dict(color='black', dash='dash', width=4)
    )

    if output_dir:
        plot_dir = output_dir / 'ancom'
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"ancom.{feature_type.lower()}"
        plotly_show_and_save(fig, show, plot_dir / file_stem)
    
    return fig, {}


def create_correlation_heatmap(
    data: pd.DataFrame,
    feature_type: str = DEFAULT_FEATURE_TYPE,
    output_dir: Union[Path, None] = None,
    show: bool = False
) -> go.Figure:
    """
    Generate correlation matrix heatmap.
    
    Args:
        data:         Correlation matrix DataFrame.
        feature_type: Type of features (ASV, OTU, etc.).
        output_dir:   Directory to save outputs.
        show:         Display figure interactively.
        
    Returns:
        Plotly heatmap figure.
    """
    fig = px.imshow(
        data, 
        color_continuous_scale='bluered', 
        title=f"{feature_type} Correlation Matrix"
    )
    
    fig.update_layout(
        template='heather',
        height=1200,
        coloraxis_colorbar=dict(
            thickness=30,
            len=0.85,
            x=1.05,
            y=0.5,
            yanchor='middle',
            tickfont=dict(size=14)
    ))

    if output_dir:
        plot_dir = output_dir / 'correlation'
        plot_dir.mkdir(parents=True, exist_ok=True)
        file_stem = f"correlation.{feature_type.lower()}"
        plotly_show_and_save(fig, show, plot_dir / file_stem)
    
    return fig


# ================================ API ENDPOINTS ==================================== #

# Simplified API functions using the new modular components
def sample_map_categorical(
    metadata: pd.DataFrame, 
    show: bool = False,
    output_dir: Union[str, Path, None] = None, 
    projection_type: str = DEFAULT_PROJECTION, 
    height: int = DEFAULT_HEIGHT, 
    size: int = DEFAULT_SIZE_MAP, 
    opacity: float = DEFAULT_OPACITY_MAP,
    lat: str = DEFAULT_LATITUDE_COL, 
    lon: str = DEFAULT_LONGITUDE_COL,
    color_col: str = DEFAULT_COLOR_COL,
    limit_axes: bool = False,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """API endpoint for geographical sample map"""
    # Convert output_dir to Path if provided
    output_path = Path(output_dir) if output_dir else None
    
    return create_geographical_map(
        metadata=metadata,
        color_col=color_col,
        lat_col=lat,
        lon_col=lon,
        projection=projection_type,
        output_dir=output_path,
        show=show,
        verbose=verbose,
        size=size,
        opacity=opacity
    )


def pca(
    components: pd.DataFrame, 
    proportion_explained: np.ndarray, 
    metadata: pd.DataFrame,
    color_col: str = DEFAULT_COLOR_COL, 
    color_map: Dict = None,
    symbol_col: str = DEFAULT_SYMBOL_COL,
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    x: int = 1, 
    y: int = 2,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """API endpoint for PCA plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_ordination_plot(
        components=components,
        metadata=metadata,
        ordination_type='PCA',
        proportion_explained=proportion_explained,
        color_col=color_col,
        symbol_col=symbol_col,
        dimensions=(x, y),
        transformation=transformation,
        output_dir=output_path,
        show=show,
        verbose=verbose
    )


def pcoa(
    components: pd.DataFrame, 
    proportion_explained: np.ndarray, 
    metadata: pd.DataFrame,
    metric: str = DEFAULT_METRIC,
    color_map: Dict = None,
    color_col: str = DEFAULT_COLOR_COL, 
    symbol_col: str = DEFAULT_SYMBOL_COL,
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    x: int = 1, 
    y: int = 2,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """API endpoint for PCoA plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_ordination_plot(
        components=components,
        metadata=metadata,
        ordination_type='PCoA',
        proportion_explained=proportion_explained,
        color_col=color_col,
        symbol_col=symbol_col,
        dimensions=(x, y),
        transformation=transformation,
        output_dir=output_path,
        show=show,
        verbose=verbose
    )


def mds(
    df: pd.DataFrame, 
    metadata: pd.DataFrame,
    color_col: str, 
    symbol_col: str,
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    mode: str = 'UMAP',
    x: int = 1, 
    y: int = 2,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """API endpoint for MDS plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_ordination_plot(
        components=df,
        metadata=metadata,
        ordination_type=mode,
        color_col=color_col,
        symbol_col=symbol_col,
        dimensions=(x, y),
        transformation=transformation,
        output_dir=output_path,
        show=show,
        verbose=verbose
    )


def heatmap_feature_abundance(
    table: pd.DataFrame, 
    show: bool = False,
    output_dir: Union[str, Path] = None,
    feature_type: str = DEFAULT_FEATURE_TYPE,
) -> go.Figure:
    """API endpoint for feature abundance heatmap"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_heatmap(
        data=table,
        feature_type=feature_type,
        output_dir=output_path,
        show=show
    )


def plot_ubiquity(
    cm: np.ndarray, 
    pm: np.ndarray, 
    ubi_c: np.ndarray, 
    ubi_p: np.ndarray, 
    contaminated: List[str], 
    pristine: List[str], 
    show: bool = False,
    output_dir: Union[str, Path] = None,
    transformation: str = None
) -> go.Figure:
    """API endpoint for ubiquity plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_ubiquity_plot(
        cm=cm,
        pm=pm,
        ubi_c=ubi_c,
        ubi_p=ubi_p,
        contaminated=contaminated,
        pristine=pristine,
        transformation=transformation,
        output_dir=output_path,
        show=show
    )


def violin_feature(
    df: pd.DataFrame, 
    feature: str, 
    output_dir: Union[str, Path], 
    sub_output_dir: str = 'faprotax',
    status_col: str = DEFAULT_SYMBOL_COL, 
    show: bool = False
) -> go.Figure:
    """API endpoint for violin plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_violin_plot(
        data=df,
        feature=feature,
        status_col=status_col,
        output_dir=output_path,
        sub_dir=sub_output_dir,
        show=show
    )


def ancom(
    data: pd.DataFrame,
    min_W: float,
    output_dir: Union[str, Path] = None,
    color_col: str = DEFAULT_COLOR_COL_ANCOM,
    show: bool = False,
    reverse_x_axis: bool = True,
    feature_type: str = DEFAULT_FEATURE_TYPE_ANCOM
) -> Tuple[go.Figure, Any]:
    """API endpoint for ANCOM plot"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_ancom_plot(
        data=data,
        min_W=min_W,
        feature_type=feature_type,
        output_dir=output_path,
        show=show,
        reverse_x_axis=reverse_x_axis
    )


def plot_correlation_matrix(
    data: pd.DataFrame,
    show: bool = False,
    output_dir: Union[str, Path] = None,
    feature_type: str = DEFAULT_FEATURE_TYPE
) -> go.Figure:
    """API endpoint for correlation matrix"""
    output_path = Path(output_dir) if output_dir else None
    
    return create_correlation_heatmap(
        data=data,
        feature_type=feature_type,
        output_dir=output_path,
        show=show
    )
    
