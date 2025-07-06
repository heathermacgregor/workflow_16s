# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Union

# Third Party Imports
import colorcet as cc
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
warnings.filterwarnings("ignore") # Suppress warnings

# ================================= GLOBAL VARIABLES ================================= #

largecolorset = list(
  cc.glasbey + cc.glasbey_light + cc.glasbey_warm + cc.glasbey_cool + cc.glasbey_dark
)

# Define the plot template
pio.templates["heather"] = go.layout.Template(
  layout={
    'title': {
      'font': {
        'family': 'HelveticaNeue-CondensedBold, Helvetica, Sans-serif',
        'size': 40,
        'color': '#000' # Black
      }
    },
    'font': {
      'family': 'Helvetica Neue, Helvetica, Sans-serif',
      'size': 26,
      'color' : '#000'
    },
    'paper_bgcolor': 'rgba(0, 0, 0, 0)', # Transparent
    'plot_bgcolor': '#fff', # White
    'colorway': largecolorset,
    'xaxis': {
      'showgrid': False,
      'zeroline': True,
      'showline': True,
      'linewidth': 7,
      'linecolor': 'black',
      'automargin': True,
      'mirror': True
    },
    'yaxis': {
      'showgrid': False,
      'zeroline': True,
      'showline': True,
      'linewidth': 7,
      'linecolor': 'black',
      'automargin': True,
      'mirror': True
    }
  }
)
pio.templates.default = "heather" 

# ==================================== FUNCTIONS ===================================== #

def plotly_show_and_save(
    fig,
    show: bool = False,
    output_path: Union[str, Path] = None,
    save_as: List[str] = ['png', 'html'],
    scale: int = 3,
    engine: str = 'kaleido',
    verbose: bool = False,
    **write_kwargs
):
    """
    Save a Plotly figure to PNG and/or HTML formats and optionally display it.
    
    Args:
        fig:            Plotly Figure object to be saved/displayed.
        show:           Whether to display the figure (default: False).
        output_path:    Base output path for files. Actual files will have format-
                        specific extensions appended (.png, .html). Directory will 
                        be created if needed.
        save_as:        List of formats to save ('png', 'html'). 
                        Default: ['png', 'html']
        scale:          DPI‑like scale factor for raster outputs.
        engine:         Backend used for static image export ('kaleido', 'orca').
        verbose:        If True, logs success messages; errors are always logged.
        **write_kwargs: Extra args forwarded to `fig.write_image` / `fig.write_html`.
    
    Notes:
        - Saving PNG files requires kaleido: install with `pip install -U kaleido`.
        - File extensions are automatically handled (e.g., 'plot' becomes 'plot.png').
    """
    if output_path:
        output_path = Path(output_path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convenience for optional INFO logging
        log_ok = (lambda msg: logger.info(msg)) if verbose else (lambda *_: None)

        static_exts = {"png", "jpg", "jpeg", "pdf", "svg", "eps"}
        for ext in static_exts.intersection(save_as):
            target = output_path.with_suffix(f".{ext}")
            try:
                fig.write_image(
                    str(target),
                    format=ext,
                    scale=scale,
                    engine=engine,
                    **write_kwargs,
                )
                log_ok(f"Saved figure to '{target}'.")
            except Exception as e:
                logger.error(
                    f"Failed to save figure: {str(e)}. "
                    "Make sure the export engine is installed "
                    "(e.g. `pip install -U kaleido`)."
                )
        
        if 'html' in save_as:
            target = output_path.with_suffix('.html')
            try:
                fig.write_html(str(target), **write_kwargs)
                log_ok(f"Saved figure to '{target}'.")
            except Exception as e:
                logger.error(f"Failed to save figure: {str(e)}")
              

#TODO: Delete this class if it is unused
class PlotlyFigure:
    def __init__(
        self,
        fig,
        output_path: Union[str, Path] = None,
        show: bool = False
    ):
        self.fig = fig
        self.output_path = Path(output_path)
        self.show = show

        plotly_show_and_save(
            fig=self.fig,
            show=self.show,
            output_path=self.output_path    
        )


def marker_color_map(
    df: pd.DataFrame, 
    col: str, 
    continuous_color_set: bool = False
) -> None:
    """
    Generate color mappings for markers based on a DataFrame column.
    
    Creates either continuous (numeric) or categorical color mappings for 
    data visualization. For continuous data, uses a viridis colormap and 
    normalizes values. For categorical data, assigns unique colors from a 
    large color set.
    
    Args:
        df:                   DataFrame containing the data to color-map.
        col:                  Column name in `df` to base color mapping on.
        continuous_color_set: If True, treats column as continuous/numeric data. 
                              If False (default), treats as categorical data.
    
    Returns:
        tuple: 
            - For continuous: (ScalarMappable, list) 
                - ScalarMappable: Matplotlib mappable object for colorbar 
                  creation.
                - list: RGBA color strings for each data point.
            - For categorical: (dict, pd.Series)
                - dict: Color mapping dictionary {category: rgba_color}.
                - Series: RGBA color strings for each data point (aligned 
                  with df index).
    
    Raises:
        ValueError: If `continuous_color_set=True` but column contains 
        non-numeric data.
    
    Notes:
        - Handles NaN values by assigning transparent black (rgba(0,0,0,0)).
        - Continuous mode adds a 'marker_color' column to the input DataFrame.
        - Categorical mode does not modify the input DataFrame.
    """
    nan_color = (0, 0, 0, 0)
    if continuous_color_set:
        try:
            cmap = plt.cm.viridis      
            norm = mcolors.Normalize(
              vmin=df[col].min(), vmax=df[col].max()
            )    
            scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)    
            # Map column data to colors
            color_map_col = [
              scalar_mappable.to_rgba(value) if not np.isnan(value) 
              else nan_color for value in df[col]]         
            color_map_col = [
              'rgba({}, {}, {}, {})'.format(x[0], x[1], x[2], x[3]) 
              for x in color_map_col
            ]
            df['marker_color'] = color_map_colw
            return scalar_mappable, color_map_col
        
        except Exception as e:
            error_message = (
              f"ERROR: 'continuous_color_set' can only be used on "
              f"numeric data. Please check your input for '{col}'."
            )
            logger.error(error_message)
            raise ValueError(error_message)
            
    else:
        color_map_dict = {
            value: color if pd.notnull(value) else 'rgba({0}, {0}, {0}, {0})' 
            for value, color in zip(df[col].unique(), largecolorset)
        }
        color_map_col = df[col].map(color_map_dict)
        return color_map_dict, color_map_col
        

def create_color_mapping(
    metadata: pd.DataFrame, 
    color_col: str
) -> Dict:
    """
    Create a color mapping for unique values in a metadata column.
    
    Args:
        metadata:  Metadata containing the color column.
        color_col: Column to use for color mapping.
    
    Returns:
        A dictionary mapping unique values to colors.
    """
    unique_values = metadata[color_col].unique()
    return {
      value: largecolorset[i % len(largecolorset)] 
      for i, value in enumerate(unique_values)
    }


def plot_legend(
    color_dict: Dict[str, str],
    max_height: int = 600  # pixels (default max height before multi-column)
) -> go.Figure:
    """
    Creates a Plotly legend figure from a dictionary of labels and colors.
    
    Args:
        color_dict:  Dictionary with labels as keys and hex colors as values.
        max_height:  Maximum pixel height before creating additional columns.
    
    Returns:
        Plotly Figure object containing only the legend
    """
    if not color_dict:
        return go.Figure()  # Return empty figure if no items
    
    # Reverse the order of legend items
    items = list(color_dict.items())[::-1]
    n = len(items)
    
    # Calculate layout parameters
    row_height = 30  # pixels per row
    max_rows = max(1, min(n, max_height // row_height))
    n_cols = (n + max_rows - 1) // max_rows  # Ceiling division
    n_rows = min(n, max_rows)
    
    # Create figure
    fig = go.Figure()
    for label, color in items:
        fig.add_trace(
            go.Scatter(
                x=[None], 
                y=[None],
                mode='markers',
                marker=dict(color=color, size=15),
                name=label,
                showlegend=True
            )
        )
    
    # Configure legend layout
    fig.update_layout(
        legend=dict(
            title=None,
            orientation='v',
            itemsizing='constant',
            itemwidth=30,
            traceorder='normal',  # Maintains reversed item order
            bordercolor='black',
            borderwidth=1
        ),
        template="heather",
        width=200 * n_cols,  # Adjust width based on columns
        height=row_height * n_rows + 100,  # Dynamic height
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    # Hide axes
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False, mirror=True)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False, mirror=True)
    
    return fig


def attach_legend_to_figure(
    main_fig: go.Figure,
    legend_fig: go.Figure,
    main_width: float = 0.8,
    legend_width: float = 0.2
) -> go.Figure:
    """
    Attaches a legend figure to the right of a main Plotly figure.
    
    Args:
        main_fig:     The main Plotly figure (geo plot).
        legend_fig:   The legend Plotly figure.
        main_width:   Width proportion for main figure (0-1).
        legend_width: Width proportion for legend (0-1).
    
    Returns:
        Combined Plotly figure with main plot and legend side-by-side.
    """
    # Create subplot figure with geo and cartesian subplots
    combined_fig = make_subplots(
        rows=1, 
        cols=2,
        column_widths=[main_width, legend_width],
        specs=[[{"type": "geo"}, {"type": "xy"}]],  # Specify geo and cartesian types
        horizontal_spacing=0.01
    )
    
    # Add main figure traces to geo subplot
    for trace in main_fig.data:
        combined_fig.add_trace(trace, row=1, col=1)
    
    # Add legend figure traces to cartesian subplot
    for trace in legend_fig.data:
        combined_fig.add_trace(trace, row=1, col=2)
    
    # Update layout from main figure
    combined_fig.update_layout(
        title=main_fig.layout.title,
        template=main_fig.layout.template,
        showlegend=False,  # We're using custom legend
        # Transfer geo layout settings
        geo=main_fig.layout.geo,
        margin=main_fig.layout.margin
    )
    
    # Configure legend column
    combined_fig.update_xaxes(
        showgrid=False, 
        showticklabels=False, 
        zeroline=False,
        row=1, 
        col=2
    )
    combined_fig.update_yaxes(
        showgrid=False, 
        showticklabels=False, 
        zeroline=False,
        row=1, 
        col=2
    )
    
    # Handle dimensions safely
    main_width_val = main_fig.layout.width or 800  # Default if not set
    main_height_val = main_fig.layout.height or 600
    legend_width_val = legend_fig.layout.width or 200
    
    combined_fig.update_layout(
        height=main_height_val,
        width=main_width_val + legend_width_val
    )
    
    return combined_fig
