# ===================================== IMPORTS ====================================== #

# Standard Library Imports
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party Imports
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import logging

# Local Imports
import workflow_16s.figures.figures
from workflow_16s.figures.figures import (
    PlotlyFigure,
    plotly_show_and_save,
    largecolorset,
    marker_color_map,
    plot_legend,
)

# ================================= GLOBAL VARIABLES ================================= #

sns.set_style('whitegrid')  # Set seaborn style globally
logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def _validate_metadata(metadata: pd.DataFrame, required_cols: List[str]) -> None:
    """Validate presence of required columns in metadata."""
    missing = [col for col in required_cols if col not in metadata.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in metadata: {', '.join(missing)}"
        )
      

def _prepare_visualization_data(
    components: pd.DataFrame,
    metadata: pd.DataFrame,
    color_col: str,
    symbol_col: str,
    placeholder: str = 'unknown'
) -> pd.DataFrame:
    """Prepare merged component and metadata data for visualization."""
    print(components.shape)
    print(metadata.shape)
    common_idx = components.index.intersection(metadata.index)
    metadata = metadata.loc[common_idx].copy()
    components = components.loc[common_idx].copy()
    
    metadata = metadata.dropna(subset=[color_col, symbol_col], how='all').fillna(placeholder)
    return components.join(metadata[[color_col, symbol_col]], how='inner')
  

def _create_scatter_figure(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str,
    symbol_col: str,
    color_map: Dict[str, str],
    hover_data: List[str] = None,
    marker_size: int = 8
) -> go.Figure:
    """Create standardized scatter plot figure."""
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        symbol=symbol_col,
        hover_data=hover_data or data.index,
        color_discrete_map=color_map,
    )
    fig.update_traces(
        marker_size=marker_size,
        marker=dict(line=dict(width=0.1, color='black'))
    )
    return fig
  

def _configure_axes(
    fig: go.Figure,
    x_title: str,
    y_title: str,
    linewidth: int = 4,
    mirror: bool = True
) -> None:
    """Configure plot axes with standardized settings."""
    fig.update_xaxes(
        title_text=x_title,
        showline=True,
        linewidth=linewidth,
        linecolor='black',
        mirror=mirror,
        automargin=True
    )
    fig.update_yaxes(
        title_text=y_title,
        showline=True,
        linewidth=linewidth,
        linecolor='black', 
        mirror=mirror,
        automargin=True
    )

def sample_map_categorical(
    metadata: pd.DataFrame, 
    show: bool = False,
    output_dir: Union[str, Path, None] = None, 
    projection_type: str = 'natural earth', 
    height: int = 800, 
    size: int = 5, 
    opacity: float = 0.3,
    lat: str = 'latitude_deg', 
    lon: str = 'longitude_deg',
    color_col: str = 'project_name',
):
    """"""
    metadata[color_col] = metadata[color_col].replace('', np.nan)  # first convert empty strings to NaN
    metadata[color_col] = metadata[color_col].fillna('other')      # then fill NaN with ''
    metadata = metadata.sort_values(by=color_col, ascending=True)
    
    # Group by 'color_col' and count samples
    cat_counts = metadata[color_col].value_counts().reset_index()
    cat_counts.columns = [color_col, 'sample_count']
    
    # Merge the counts back into the original metadata
    metadata = metadata.merge(cat_counts, on=color_col, how='left')

    # Create a color mapping for datasets
    color_mapping = {c: largecolorset[i % len(largecolorset)] 
                     for i, c in enumerate(cat_counts[color_col])}
    
    # Print the assigned colors
    for cat, assigned_color in color_mapping.items():
        logger.info(f"[{assigned_color}]    {cat}")
        
    #legend = plot_legend(color_mapping, color_col)
    
    # Plot the points on a map
    fig = px.scatter_geo(
        metadata, 
        lat=lat, 
        lon=lon, 
        color=color_col, 
        color_discrete_map=color_mapping,  
        hover_name=color_col, 
        hover_data={'sample_count': True}  
    )
    
    fig.update_geos(
        projection_type=projection_type,  
        resolution=50,
        showcoastlines=True, coastlinecolor="#b5b5b5",
        showland=True, landcolor="#e8e8e8",
        showlakes=True, lakecolor="#fff",
        showrivers=True, rivercolor="#fff",
        # Set axis limits based on the range of latitude and longitude values
        #lonaxis_range=[metadata[lon].min() - 20, metadata[lon].max() + 20], 
        #lataxis_range=[metadata[lat].min() - 20, metadata[lat].max() + 20]
    )
    
    fig.update_layout(
        template='heather',
        margin=dict(l=5, r=5, t=5, b=5), # Set the layout with increased width and height
        showlegend=False, 
        font_size=12
    )  
    
    # Update marker size to make the dots smaller and semi-transparent
    fig.update_traces(marker=dict(size=size, opacity=opacity)) 

    #fig.show()
    if output_dir:
        output_path = Path(output_dir) / f"sample_map.{color_col}"
        plotly_show_and_save(fig=fig, show=show, output_path=output_path)

        colordict = color_mapping
        legend_path = Path(output_dir) / f'legend.{color_col}.png'
        plot_legend(colordict, color_col, legend_path)
    return fig#, legend
    

def heatmap_feature_abundance(
    table: pd.DataFrame, 
    show: bool = False,
    output_dir: Union[str, Path] = None,
    feature_type: str = "ASV",
) -> go.Figure:
    """
    Generate an interactive heatmap visualization of feature abundance across samples.
    
    Args:
        table:        Feature abundance matrix with features as rows and samples 
                      as columns.
        show:         Whether to display the figure immediately. Defaults to False.
        output_dir:   Directory path to save the figure. Defaults to None.
        feature_type: Type of features displayed (e.g., 'ASV', 'OTU'). Defaults 
                      to "ASV".

    Returns:
        Interactive Plotly figure object displaying the heatmap.
    """
    fig = px.imshow(
        table,
        color_continuous_scale='viridis',
        labels={'x': 'Samples', 'y': feature_type, 'color': 'Abundance'},
        title=f"Heatmap of {feature_type} Abundance"
    )

    fig.update_layout(
        template='heather',
        height=1200,
        xaxis_showticklabels=False,
        yaxis_showticklabels=False
    )
    
    if output_dir:
        output_path = Path(output_dir) / f"heatmap.{feature_type.lower()}"
        plotly_show_and_save(fig=fig, show=show, output_path=output_path)
    
    return fig
  

def pcoa(
    components: pd.DataFrame, 
    proportion_explained: np.ndarray, 
    metadata: pd.DataFrame,
    metric: str = 'braycurtis',
    color_col: str = 'dataset_name', 
    symbol_col: str = 'nuclear_contamination_status',
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    x: int = 1, 
    y: int = 2
) -> Tuple[go.Figure, Any]:
    """
    Generate a PCoA plot with metadata annotations and explanatory variance percentages.
    
    Args:
        components:           PCoA coordinates matrix with samples as rows and 
                              components as columns.
        proportion_explained: Array of variance explained percentages for each 
                              component.
        metadata:             Metadata DataFrame containing color and symbol columns.
        metric:               Beta diversity metric used for PCoA. Defaults to 
                              'braycurtis'.
        color_col:            Metadata column name for point colors. Defaults to 
                              'dataset_name'.
        symbol_col:           Metadata column name for point markers. Defaults to 
                              'nuclear_contamination_status'.
        show:                 Whether to display the figure immediately. Defaults 
                              to False.
        output_dir:           Directory path to save outputs. Defaults to None.
        transformation:       Data transformation applied prior to PCoA. Defaults 
                              to None.
        x:                    Component number for x-axis. Defaults to 1.
        y:                    Component number for y-axis. Defaults to 2.

    Returns:
        Tuple containing Plotly figure object and legend figure.
        
    Raises:
        ValueError: If specified color_col or symbol_col are missing from metadata.
    """
    _validate_metadata(metadata, [color_col, symbol_col])
    
    data = _prepare_visualization_data(components, metadata, color_col, symbol_col)
    colordict, _ = marker_color_map(data, color_col, continuous_color_set=False)
    
    x_col = f'PC{x}'
    y_col = f'PC{y}'
    x_title = f"PCo{x} ({round(100 * proportion_explained[x-1], 2)}%)"
    y_title = f"PCo{y} ({round(100 * proportion_explained[y-1], 2)}%)"

    data['index'] = data.index
    
    fig = _create_scatter_figure(
        data=data,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        symbol_col=symbol_col,
        color_map=colordict,
        hover_data=['index', color_col]
    )
    
    fig.update_layout(
        template='heather',
        height=1000,
        width=1100,
        plot_bgcolor='#fff',
        font_size=45,
        showlegend=False,
        xaxis=dict(showticklabels=False, zeroline=True),
        yaxis=dict(showticklabels=False, zeroline=True)
    )
    _configure_axes(fig, x_title, y_title, linewidth=7)

    if output_dir:
        file_stem = (
            f"pcoa.{f'{transformation}.' if transformation else ''}"
            f"{metric}.{x}-{y}.{color_col}.{symbol_col}"
        )
        plotly_show_and_save(
            fig, show, Path(output_dir) / 'pcoa' / file_stem
        )
        legend_path = Path(output_dir) / 'pcoa' / f'legend.{color_col}.png'
        plot_legend(colordict, color_col, legend_path)
    
    return fig, colordict
  

def pca(
    components: pd.DataFrame, 
    proportion_explained: np.ndarray, 
    metadata: pd.DataFrame,
    color_col: str = 'dataset_name', 
    symbol_col: str = 'nuclear_contamination_status',
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    x: int = 1, 
    y: int = 2
) -> Tuple[go.Figure, Any]:
    """
    Generate a PCA plot with metadata annotations and variance explained percentages.
    
    Args:
        components:           PCA coordinates matrix with samples as rows and 
                              components as columns.
        proportion_explained: Array of variance explained percentages for each 
                              component.
        metadata:             Metadata DataFrame containing color and symbol columns.
        color_col:            Metadata column name for point colors. Defaults 
                              to 'dataset_name'.
        symbol_col:           Metadata column name for point markers. Defaults 
                              to 'nuclear_contamination_status'.
        show:                 Whether to display the figure immediately. Defaults 
                              to False.
        output_dir:           Directory path to save outputs. Defaults to None.
        transformation:       Data transformation applied prior to PCA. Defaults 
                              to None.
        x:                    Component number for x-axis. Defaults to 1.
        y:                    Component number for y-axis. Defaults to 2.

    Returns:
        Tuple containing Plotly figure object and legend figure.
    """
    _validate_metadata(metadata, [color_col, symbol_col])
    
    data = _prepare_visualization_data(components, metadata, color_col, symbol_col)
    colordict, _ = marker_color_map(data, color_col, continuous_color_set=False)
    
    x_col = f'PC{x}'
    y_col = f'PC{y}'
    x_title = f"PC{x} ({round(100 * proportion_explained[x-1], 2)}%)"
    y_title = f"PC{y} ({round(100 * proportion_explained[y-1], 2)}%)"

    data['index'] = data.index

    fig = _create_scatter_figure(
        data=data,
        x_col=x_col,
        y_col=y_col,
        color_col=color_col,
        symbol_col=symbol_col,
        color_map=colordict,
        hover_data=['index', color_col]
    )
    
    fig.update_layout(
        template='heather',
        height=800,
        width=800,
        showlegend=False,
        font_family="Helvetica",
        font_color="black",
        font_size=15,
        title_text='PCA',
        title_x=0.5
    )
    _configure_axes(fig, x_title, y_title)

    if output_dir:
        file_stem = (
            f"pca.{f'{transformation}.' if transformation else ''}"
            f"{x}-{y}.{color_col}.{symbol_col}"
        )
        plotly_show_and_save(
            fig, show, Path(output_dir) / 'pca' / file_stem
        )
        legend_path = Path(output_dir) / 'pca' / f'legend_{color_col}.png'
        plot_legend(colordict, color_col, legend_path)
    
    return fig, colordict


def mds(
    df: pd.DataFrame, 
    metadata: pd.DataFrame,
    group_col: str, 
    symbol_col: str,
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    mode: str = 'UMAP',
    x: int = 1, 
    y: int = 2
) -> Tuple[go.Figure, Any]:
    """
    Generate a multidimensional scaling plot (e.g., UMAP, t-SNE) with metadata 
    annotations.
    
    Args:
        df:             MDS/UMAP coordinates matrix with samples as rows and 
                        dimensions as columns.
        metadata:       Metadata DataFrame containing group and symbol annotations.
        group_col:      Metadata column name for grouping/coloring points.
        symbol_col:     Metadata column name for point markers.
        show:           Whether to display the figure immediately. Defaults to False.
        output_dir:     Directory path to save outputs. Defaults to None.
        transformation: Data transformation applied prior to MDS. Defaults to None.
        mode:           Dimensionality reduction method used. Defaults to 'UMAP'.
        x:              Dimension number for x-axis. Defaults to 1.
        y:              Dimension number for y-axis. Defaults to 2.

    Returns:
        Tuple containing Plotly figure object and legend figure.
    """
    _validate_metadata(metadata, [group_col, symbol_col])
    
    # Align indices and handle missing values
    df.index = df.index.astype(str)
    metadata.index = metadata.index.astype(str)
    data = _prepare_visualization_data(df, metadata, group_col, symbol_col)
    
    if f'{mode}{x}' not in data.columns or f'{mode}{y}' not in data.columns:
        raise ValueError(f"Missing {mode} coordinates in data columns")

    colordict, _ = marker_color_map(data, group_col, continuous_color_set=False)

    data['index'] = data.index
    
    fig = _create_scatter_figure(
        data=data,
        x_col=f'{mode}{x}',
        y_col=f'{mode}{y}',
        color_col=group_col,
        symbol_col=symbol_col,
        color_map=colordict,
        hover_data=['index', group_col]
    )
    
    fig.update_layout(
        template='heather',
        height=800,
        width=800,
        showlegend=False,
        title_text=mode,
        title_x=0.5
    )
    _configure_axes(fig, f'{mode}{x}', f'{mode}{y}')

    if output_dir: 
        file_stem = (
            f"{mode}.{f'{transformation}.' if transformation else ''}"
            f"{x}-{y}.{group_col}.{symbol_col}"
        )
        plotly_show_and_save(
            fig, show, Path(output_dir) / mode.lower() / file_stem
        )
        legend_path = Path(output_dir) / mode.lower() / f'legend_{group_col}.png'
        plot_legend(colordict, group_col, legend_path)
    
    return fig, colordict
  

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
    """
    Generate a ubiquity plot comparing feature prevalence between contaminated 
    and pristine groups.
    
    Args:
        cm:             Mean abundance values for contaminated group.
        pm:             Mean abundance values for pristine group.
        ubi_c:          Ubiquity scores for contaminated group.
        ubi_p:          Ubiquity scores for pristine group.
        contaminated:   Sample IDs in contaminated group.
        pristine:       Sample IDs in pristine group.
        show:           Whether to display the figure immediately. Defaults 
                        to False.
        output_dir:     Directory path to save outputs. Defaults to None.
        transformation: Data transformation applied. Defaults to None.

    Returns:
        Interactive Plotly figure object showing ubiquity relationships.
    """
    sizes = np.array(
        [(v / len(contaminated)) + (ubi_p[i] / len(pristine)) 
         for i, v in enumerate(ubi_c)]
    )
    
    text = [
        f'Ubiq C = {v/len(contaminated):.3g}<BR>Ubiq P = {ubi_p[i]/len(pristine):.3g}'
        f'<BR><BR>Mean C = {cm[i]:.3g}<BR>Mean P = {pm[i]:.3g}'
        for i, v in enumerate(ubi_c)
    ]
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cm,
        y=pm,
        mode='markers',
        marker_size=10*sizes,
        text=text
    ))
    
    fig.update_layout(
        template='heather',
        height=600,
        width=800,
        title_text='Enrichment of significant changers',
        title_x=0.5,
        xaxis_title='Contaminated',
        yaxis_title='Pristine'
    )

    if output_dir:
        file_stem = (
            f"ubiquity.{f'{transformation}.' if transformation else ''}"
        )
        plotly_show_and_save(
            fig, show, Path(output_dir) / 'ubiquity' / file_stem
        )
    
    return fig
  

def violin_feature(
    df: pd.DataFrame, 
    feature: str, 
    output_dir: Union[str, Path], 
    sub_output_dir: str = 'faprotax',
    status_col: str = 'nuclear_contamination_status', 
    show: bool = False
) -> go.Figure:
    """
    Generate a violin plot showing distribution of a feature across contamination 
    status groups.
    
    Args:
        df:             DataFrame containing feature values and metadata.
        feature:        Name of feature column to plot.
        output_dir:     Base directory path for saving outputs.
        sub_output_dir: Subdirectory for specific output type. Defaults to 'faprotax'.
        status_col:     Metadata column containing contamination status. Defaults 
                        to 'nuclear_contamination_status'.
        show:           Whether to display the figure immediately. Defaults to False.

    Returns:
        Interactive Plotly figure object displaying violin plot.
    """
    df = df.reset_index().dropna(subset=[feature, status_col])
    
    fig = px.violin(
        df, 
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
        plotly_show_and_save(
            fig, show, Path(output_dir) / sub_output_dir / f'violin.{status_col}.{feature}'.lower()
        )
    
    return fig
  

def ancom(
    data: pd.DataFrame,
    min_W: float,
    output_dir: Union[str, Path] = None,
    color_col: str = 'p',
    show: bool = False,
    reverse_x_axis: bool = True,
    feature_type: str = "l6"
) -> Tuple[go.Figure, Any]:
    """
    Generate an ANCOM volcano plot showing differentially abundant features.
    
    Args:
        data:           ANCOM results DataFrame containing CLR values and W statistics.
        min_W:          Minimum W statistic threshold for significance.
        output_dir:     Directory path to save outputs. Defaults to None.
        color_col:      Column name for color coding points. Defaults to 'p'.
        show:           Whether to display the figure immediately. Defaults to False.
        reverse_x_axis: Whether to invert CLR values on x-axis. Defaults to True.
        feature_type:   Taxonomic level or feature type analyzed. Defaults to "l6".

    Returns:
        Tuple containing ANCOM plot figure and legend figure.
    """
    colordict, colormap = marker_color_map(data, color_col)
    
    if reverse_x_axis:
        data['clr'] = -data['clr']
    
    fig = px.scatter(
        data, 
        x='clr', 
        y='W', 
        hover_data=['Feature'],
        color=color_col,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        template='heather',
        width=1100,
        height=1000,
        paper_bgcolor='#fff'
    )
    _configure_axes(fig, 'clr', 'W', linewidth=7)
    
    # Add significance threshold line
    fig.add_shape(
        type='line',
        x0=data['clr'].min(),
        y0=min_W,
        x1=data['clr'].max(),
        y1=min_W,
        line=dict(color='black', dash='dash', width=4)
    )

    if output_dir:
        plotly_show_and_save(
            fig, show, Path(output_dir) / 'ancom' / f"ancom.{feature_type.lower()}"
        )
        plot_legend(
            colordict, color_col, 
            Path(output_dir) / 'ancom' / f'legend.{feature_type.lower()}.png'
        )
    
    return fig, colordict
  

def plot_correlation_matrix(
    data: pd.DataFrame,
    show: bool = False,
    output_dir: Union[str, Path] = None,
    feature_type: str = "ASV"
) -> go.Figure:
    """
    Generate an interactive correlation matrix heatmap for features.
    
    Args:
        data:         Correlation matrix DataFrame.
        show:         Whether to display the figure immediately. Defaults to False.
        output_dir:   Directory path to save outputs. Defaults to None.
        feature_type: Type of features analyzed. Defaults to "ASV".

    Returns:
        Interactive Plotly figure object displaying correlation matrix.
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
        )
    )

    if output_dir:
        plotly_show_and_save(
            fig, show, Path(output_dir) / 'correlation' / f"correlation.{feature_type.lower()}"
        )
    
    return fig
