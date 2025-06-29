# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
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
      'zeroline': True
      'showline': True,
      'linewidth': 7,
      'linecolor': 'black',
      'automargin': True,
      'mirror': True
    },
    'yaxis': {
      'showgrid': False,
      'zeroline': True
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
    verbose: bool = False
):
    """
    Save a Plotly figure to PNG and/or HTML formats and optionally display it.
    
    Args:
        fig:         Plotly Figure object to be saved/displayed.
        show:        Whether to display the figure (default: False).
        output_path: Base output path for files. Actual files will have format-
                     specific extensions appended (.png, .html). Directory will 
                     be created if needed.
        save_as:     List of formats to save (supported: 'png', 'html'). 
                     Default: ['png','html']
        verbose:     If True, logs success messages; errors are always logged.
    
    Notes:
        - PNG saving requires kaleido: install with `pip install -U kaleido`
        - File extensions are automatically handled (e.g., 'plot' becomes 'plot.png')
        - Directory creation errors will propagate (not caught in exception handling)
    """
    if output_path:
        output_path = Path(output_path)
        output_dir = output_path.parent.resolve()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if 'png' in save_as:
            png_file = output_path.with_suffix('.png')  # Replace with .png
            try:
                fig.write_image(
                    str(png_file),
                    format='png', 
                    scale=3,
                    engine='kaleido'
                )
                if verbose:
                    logger.info(f"Saved figure to '{png_file}'.")
            except Exception as e:
                logger.error(f"Failed to save PNG: {str(e)}")
                if verbose:
                    logger.info("Install kaleido: pip install -U kaleido")
        
        if 'html' in save_as:
            html_file = output_path.with_suffix('.html')  # Replace with .html
            try:
                fig.write_html(str(html_file))
                if verbose:
                    logger.info(f"Saved figure to '{html_file}'.")
            except Exception as e:
                logger.error(f"Failed to save HTML: {str(e)}")
              

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
                  creation
                - list: RGBA color strings for each data point
            - For categorical: (dict, pd.Series)
                - dict: Color mapping dictionary {category: rgba_color}
                - Series: RGBA color strings for each data point (aligned 
                  with df index)
    
    Raises:
        ValueError: If `continuous_color_set=True` but column contains 
        non-numeric data
    
    Notes:
        - Handles NaN values by assigning transparent black (rgba(0,0,0,0))
        - Continuous mode adds a 'marker_color' column to the input DataFrame
        - Categorical mode does not modify the input DataFrame
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
    color_dict: dict[str, str], 
    show: bool = False,
    output_path: str | Path = None
) -> None:
    """
    Creates a legend image from a dictionary where keys are labels and 
    values are hex colors.
    
    Args:
        color_dict:  Dictionary with labels as keys and hex colors as 
                     values.
        show:        If True, displays the legend using plt.show().
        output_path: Path to save the generated legend image (optional).
    """
    n = len(color_dict)  # Number of items in the legend
    
    # Create figure
    fig, ax = plt.subplots(figsize=(3, n * 0.5))
    ax.axis('off')  # Remove axes
    
    # Add legend items
    for i, (label, color) in enumerate(color_dict.items()):
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color))
        ax.text(1.2, i + 0.5, label, va='center', fontsize=16, color='#000')
    
    # Set plot limits
    ax.set_xlim(0, 3)
    ax.set_ylim(0, n)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()
    
    plt.close(fig)  # Close the figure to free memory
