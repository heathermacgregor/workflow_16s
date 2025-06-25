# ===================================== IMPORTS ====================================== #

# Standard Library Imports
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

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

# Local Imports
import workflow_16s.figures.figures
from workflow_16s.figures.figures import (
    PlotlyFigure,
    plotly_show_and_save,
    largecolorset,
    marker_color_map,
    plot_legend,
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

sns.set_style('whitegrid')  # Set seaborn style globally
logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def _validate_metadata(
    metadata: pd.DataFrame, 
    required_cols: List[str]
) -> None:
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
    placeholder: str = 'unknown',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Prepare merged component and metadata data for visualization with robust diagnostics.
    
    Returns:
        Merged DataFrame containing components and requested metadata columns
        
    Raises:
        ValueError: If no common samples found between components and metadata
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
            logger.warning("Metadata missing '#sampleid' column - using existing index")
        meta_copy.index = meta_copy.index.astype(str).str.strip().str.lower()
        
    if verbose:
        # Log sample IDs for debugging
        logger.debug(f"Components index (first 5): {comp_copy.index.tolist()[:5]}")
        logger.debug(f"Metadata index (first 5): {meta_copy.index.tolist()[:5]}")
    
    # Find common samples
    common_idx = comp_copy.index.intersection(meta_copy.index)
    if verbose:
        logger.info(f"Found {len(common_idx)} common samples between components and metadata")
    
    # Handle no common samples case with detailed diagnostics
    if len(common_idx) == 0:
        comp_samples = set(comp_copy.index)
        meta_samples = set(meta_copy.index)
        
        comp_only = comp_samples - meta_samples
        meta_only = meta_samples - comp_samples

        if verbose:
            logger.critical("CRITICAL ERROR: No common samples between components and metadata!")
            logger.critical(f"Components-only samples ({len(comp_only)}): {list(comp_only)[:5]}{'...' if len(comp_only) > 5 else ''}")
            logger.critical(f"Metadata-only samples ({len(meta_only)}): {list(meta_only)[:5]}{'...' if len(meta_only) > 5 else ''}")
        
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
                logger.warning(f"Column '{col}' missing from metadata. Creating placeholder column.")
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
    color_col: str = 'dataset_name',
    limit_axes: bool = False,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """
    Generate an interactive geographical map of samples colored by categorical metadata.
    """
    # Convert empty strings to NaN then to 'other'
    metadata[color_col] = metadata[color_col].replace('', np.nan) 
    metadata[color_col] = metadata[color_col].fillna('other')      
    metadata = metadata.sort_values(by=color_col, ascending=True)
    
    # Count samples per category
    cat_counts = metadata[color_col].value_counts().reset_index()
    cat_counts.columns = [color_col, 'sample_count']
    metadata = metadata.merge(cat_counts, on=color_col, how='left')

    # Create color mapping using sorted unique categories
    categories = sorted(metadata[color_col].astype(str).unique())
    colordict = {c: largecolorset[i % len(largecolorset)] 
                 for i, c in enumerate(categories)}

    if verbose:
        for cat, color in colordict.items():
            logger.info(f"[{color}]    {cat}")
        
    # Create plot
    fig = px.scatter_geo(
        metadata, 
        lat=lat, 
        lon=lon, 
        color=color_col, 
        color_discrete_map=colordict,  
        hover_name=color_col, 
        hover_data={'sample_count': True}  
    )
    
    # Configure map
    fig.update_geos(
        projection_type=projection_type,  
        resolution=50,
        showcoastlines=True, coastlinecolor="#b5b5b5",
        showland=True, landcolor="#e8e8e8",
        showlakes=True, lakecolor="#fff",
        showrivers=True, rivercolor="#fff",
    )
    
    # Set axis limits if requested
    if limit_axes:
        fig.update_geos(
            lonaxis_range=[metadata[lon].min() - 20, metadata[lon].max() + 20], 
            lataxis_range=[metadata[lat].min() - 20, metadata[lat].max() + 20]
        )
    
    # Update layout
    fig.update_layout(
        template='heather',
        margin=dict(l=5, r=5, t=5, b=5),
        showlegend=False, 
        font_size=40,
        xaxis=dict(showticklabels=False, zeroline=True),
        yaxis=dict(showticklabels=False, zeroline=True)
    )  
    
    # Update marker appearance
    fig.update_traces(marker=dict(size=size, opacity=opacity)) 

    # Save output and legend if requested
    if output_dir:
        output_path = Path(output_dir)
        file_stem = f"sample_map.{color_col}"
        plotly_show_and_save(fig=fig, show=show, output_path=output_path / file_stem)

        # Save legend
        legend_path = output_path / f"{file_stem}.legend.png"
        plot_legend(colordict, color_col, legend_path)
        
    return fig, colordict


def pca(
    components: pd.DataFrame, 
    proportion_explained: np.ndarray, 
    metadata: pd.DataFrame,
    color_col: str = 'dataset_name', 
    color_map: Dict = None,
    symbol_col: str = 'nuclear_contamination_status',
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    x: int = 1, 
    y: int = 2,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """
    Generate a PCA plot with comprehensive error handling and diagnostics.
    """
    try:
        # Validate required metadata columns
        required_columns = [color_col, symbol_col, '#sampleid']
        missing = [col for col in required_columns if col not in metadata.columns]
        if missing:
            if verbose:
                logger.error(f"Missing required metadata columns: {', '.join(missing)}")
            raise ValueError(f"Missing columns: {', '.join(missing)}")
        
        # Prepare visualization data
        logger.info("Preparing PCA visualization data...")
        data = _prepare_visualization_data(components, metadata, color_col, symbol_col, verbose)
        
        # Handle empty data case
        if data.empty:
            if verbose:
                logger.error("No data available for PCA plotting after merging")
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for plotting",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=24, color="red")
            )
            fig.update_layout(
                title="PCA Plot - Data Unavailable",
                template='heather'
            )
            return fig, {}
        
        # Create color mapping using sorted unique categories
        categories = sorted(data[color_col].astype(str).unique())
        colordict = {c: largecolorset[i % len(largecolorset)] 
                     for i, c in enumerate(categories)}
        
        # Add explicit sample ID column
        data['sample_id'] = data.index
        
        # Verify PC columns exist
        x_col = f'PC{x}'
        y_col = f'PC{y}'
        
        if x_col not in data.columns:
            available_pcs = [col for col in data.columns if col.startswith('PC')]
            if verbose:
                logger.error(f"Missing x-axis column '{x_col}'. Available PC columns: {available_pcs}")
            raise ValueError(f"Column {x_col} not found in PCA components")
        
        if y_col not in data.columns:
            available_pcs = [col for col in data.columns if col.startswith('PC')]
            if verbose:
                logger.error(f"Missing y-axis column '{y_col}'. Available PC columns: {available_pcs}")
            raise ValueError(f"Column {y_col} not found in PCA components")
        
        # Create plot
        logger.info(f"Creating PCA plot with {len(data)} samples")
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            symbol=symbol_col,
            color_discrete_map=colordict,
            hover_data=['sample_id', color_col, symbol_col],
            opacity=0.8,
            size_max=10
        )
        
        # Add variance explained to axis labels
        if proportion_explained is not None and len(proportion_explained) >= max(x, y):
            x_pct = proportion_explained[x-1] * 100
            y_pct = proportion_explained[y-1] * 100
            x_title = f"PC{x} ({x_pct:.1f}%)"
            y_title = f"PC{y} ({y_pct:.1f}%)"
        else:
            logger.warning("Proportion explained array missing or too short")
            x_title = f"PC{x}"
            y_title = f"PC{y}"

        fig.update_layout(
            template='heather',
            height=1000,
            width=1100,
            plot_bgcolor='#fff',
            font_size=45,
            showlegend=False,
            title_text=f'PCA: {transformation.title() if transformation else "Raw Data"}',
            title_x=0.5,
            xaxis_title=x_title,
            yaxis_title=y_title,
            xaxis=dict(showticklabels=False, zeroline=True),
            yaxis=dict(showticklabels=False, zeroline=True)
        )
        
        # Save output and legend if requested
        if output_dir:
            output_path = Path(output_dir) / 'pca'
            output_path.mkdir(parents=True, exist_ok=True)
            file_stem = f"pca.{transformation or 'raw'}.{x}-{y}.{color_col}.{symbol_col}"
            plotly_show_and_save(fig, show, output_path / file_stem, ['png', 'html'], verbose)
            logger.info(f"Saved PCA plot to {output_path / file_stem}")
            
            # Save legend
            legend_path = output_path / f"{file_stem}.legend.png"
            plot_legend(colordict, color_col, legend_path)
        
        return fig, colordict
    
    except Exception as e:
        logger.exception(f"Critical error generating PCA plot: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)[:100]}...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="red")
        )
        fig.update_layout(
            title="PCA Plot - Generation Failed",
            template='heather'
        )
        return fig, {}
  

def pcoa(
    components: pd.DataFrame, 
    proportion_explained: np.ndarray, 
    metadata: pd.DataFrame,
    metric: str = 'braycurtis',
    color_map: Dict = None,
    color_col: str = 'dataset_name', 
    symbol_col: str = 'nuclear_contamination_status',
    show: bool = False,
    output_dir: Union[str, Path] = None, 
    transformation: str = None,
    x: int = 1, 
    y: int = 2,
    verbose: bool = False
) -> Tuple[go.Figure, Dict]:
    """
    Generate a PCoA plot with metadata annotations.
    """
    # Validate metadata columns
    _validate_metadata(metadata, [color_col, symbol_col])
    
    # Prepare visualization data
    data = _prepare_visualization_data(components, metadata, color_col, symbol_col, verbose)
    
    # Create color mapping using sorted unique categories
    categories = sorted(data[color_col].astype(str).unique())
    colordict = {c: largecolorset[i % len(largecolorset)] 
                 for i, c in enumerate(categories)}
    
    # Add explicit sample ID column
    data['sample_id'] = data.index
    
    # Create plot
    fig = px.scatter(
        data,
        x=f'PCo{x}',
        y=f'PCo{y}',
        color=color_col,
        symbol=symbol_col,
        color_discrete_map=colordict,
        hover_data=['sample_id', color_col],
        opacity=0.8
    )
    
    # Configure axes
    if proportion_explained is not None and len(proportion_explained) >= max(x, y):
        x_title = f"PCo{x} ({proportion_explained[x-1]:.2%})"
        y_title = f"PCo{y} ({proportion_explained[y-1]:.2%})"
    else:
        x_title = f"PCo{x}"
        y_title = f"PCo{y}"
    
    fig.update_layout(
        template='heather',
        height=1000,
        width=1100,
        plot_bgcolor='#fff',
        font_size=45,
        showlegend=False,
        xaxis_title=x_title,
        yaxis_title=y_title,
        xaxis=dict(showticklabels=False, zeroline=True),
        yaxis=dict(showticklabels=False, zeroline=True)
    )
    
    # Save output and legend if requested
    if output_dir:
        output_path = Path(output_dir) / 'pcoa'
        output_path.mkdir(parents=True, exist_ok=True)
        file_stem = f"pcoa.{transformation or 'raw'}.{x}-{y}.{color_col}.{symbol_col}"
        plotly_show_and_save(fig, show, output_path / file_stem, ['png', 'html'], verbose)
        
        # Save legend
        legend_path = output_path / f"{file_stem}.legend.png"
        plot_legend(colordict, color_col, legend_path)
    
    return fig, colordict
    

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
    """
    Generate a multidimensional scaling plot (t-SNE or UMAP).
    """
    # Validate metadata columns
    _validate_metadata(metadata, [color_col, symbol_col])
    
    # Prepare visualization data
    data = _prepare_visualization_data(df, metadata, color_col, symbol_col)
    
    # Create color mapping using sorted unique categories
    categories = sorted(data[color_col].astype(str).unique())
    colordict = {c: largecolorset[i % len(largecolorset)] 
                 for i, c in enumerate(categories)}
    
    # Add explicit sample ID column
    data['sample_id'] = data.index
    
    # Create plot
    fig = px.scatter(
        data,
        x=f'{mode}{x}',
        y=f'{mode}{y}',
        color=color_col,
        symbol=symbol_col,
        color_discrete_map=colordict,
        hover_data=['sample_id', color_col, symbol_col],
        opacity=0.8
    )
    
    # Configure layout
    fig.update_layout(
        template='heather',
        height=1000,
        width=1100,
        plot_bgcolor='#fff',
        font_size=45,
        showlegend=False,
        title_text=mode,
        title_x=0.5,
        xaxis_title=f'{mode}{x}',
        yaxis_title=f'{mode}{y}',
        xaxis=dict(showticklabels=False, zeroline=True),
        yaxis=dict(showticklabels=False, zeroline=True)
    )
    
    # Save output and legend if requested
    if output_dir: 
        output_path = Path(output_dir) / mode.lower()
        output_path.mkdir(parents=True, exist_ok=True)
        file_stem = f"{mode}.{transformation or 'raw'}.{x}-{y}.{color_col}.{symbol_col}"
        plotly_show_and_save(fig, show, output_path / file_stem, ['png', 'html'], verbose)
        
        # Save legend
        legend_path = output_path / f"{file_stem}.legend.png"
        plot_legend(colordict, color_col, legend_path)
    
    return fig, colordict


def heatmap_feature_abundance(
    table: pd.DataFrame, 
    show: bool = False,
    output_dir: Union[str, Path] = None,
    feature_type: str = "ASV",
) -> go.Figure:
    """
    Generate an interactive heatmap visualization of feature abundance.
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
    Generate a ubiquity plot comparing feature prevalence.
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
        title_text='Feature Ubiquity Comparison',
        title_x=0.5,
        xaxis_title='Contaminated',
        yaxis_title='Pristine'
    )

    if output_dir:
        output_path = Path(output_dir) / 'ubiquity'
        output_path.mkdir(parents=True, exist_ok=True)
        file_stem = f"ubiquity.{transformation or 'raw'}"
        plotly_show_and_save(fig, show, output_path / file_stem)
    
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
    Generate a violin plot for a specific feature.
    """
    # Reset index and handle missing values
    df = df.reset_index().dropna(subset=[feature, status_col])
    
    # Create plot
    fig = px.violin(
        df, 
        y=feature, 
        x=status_col,
        box=True,
        points="all",
        title=f"{feature.replace('_', ' ').title()} Distribution",
        hover_data=['index', 'dataset_name']
    )
    
    # Configure layout
    fig.update_layout(
        template='heather',
        xaxis_title="Contamination Status",
        yaxis_title=feature.replace('_', ' ').title()
    )

    # Save output
    if output_dir:
        output_path = Path(output_dir) / sub_output_dir
        output_path.mkdir(parents=True, exist_ok=True)
        file_stem = f'violin.{status_col}.{feature}'.lower()
        plotly_show_and_save(fig, show, output_path / file_stem)
    
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
    Generate an ANCOM volcano plot.
    """
    # Create color mapping
    colordict, colormap = marker_color_map(data, color_col)
    
    # Optionally reverse CLR values
    if reverse_x_axis:
        data['clr'] = -data['clr']
    
    # Create plot
    fig = px.scatter(
        data, 
        x='clr', 
        y='W', 
        hover_data=['Feature'],
        color=color_col,
        color_continuous_scale='viridis'
    )
    
    # Configure layout
    fig.update_layout(
        template='heather',
        width=1100,
        height=1000,
        paper_bgcolor='#fff',
        xaxis_title='CLR',
        yaxis_title='W statistic'
    )
    
    # Add significance threshold
    fig.add_shape(
        type='line',
        x0=data['clr'].min(),
        y0=min_W,
        x1=data['clr'].max(),
        y1=min_W,
        line=dict(color='black', dash='dash', width=4)
    )

    # Save output
    if output_dir:
        output_path = Path(output_dir) / 'ancom'
        output_path.mkdir(parents=True, exist_ok=True)
        file_stem = f"ancom.{feature_type.lower()}"
        plotly_show_and_save(fig, show, output_path / file_stem)
    
    return fig, colordict
  

def plot_correlation_matrix(
    data: pd.DataFrame,
    show: bool = False,
    output_dir: Union[str, Path] = None,
    feature_type: str = "ASV"
) -> go.Figure:
    """
    Generate a correlation matrix heatmap.
    """
    fig = px.imshow(
        data, 
        color_continuous_scale='bluered', 
        title=f"{feature_type} Correlation Matrix"
    )
    
    # Configure layout
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

    # Save output
    if output_dir:
        output_path = Path(output_dir) / 'correlation'
        output_path.mkdir(parents=True, exist_ok=True)
        file_stem = f"correlation.{feature_type.lower()}"
        plotly_show_and_save(fig, show, output_path / file_stem)
    
    return fig
