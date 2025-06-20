# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

# Third Party Imports
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
import colorcet as cc
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import kaleido

# ================================= GLOBAL VARIABLES ================================= #

logger = logging.getLogger('workflow_16s')

largecolorset = list(
  cc.glasbey + cc.glasbey_light + cc.glasbey_warm + cc.glasbey_cool + cc.glasbey_dark
)

# Define the plot template
pio.templates["heather"] = go.layout.Template(
    layout={
        'title': {
            'font': {
                'family': 'HelveticaNeue-CondensedBold, Helvetica, Sans-serif',
                'size'  : 40,
                'color' : '#000'
            }
        },
        'font': {
            'family': 'Helvetica Neue, Helvetica, Sans-serif',
            'size'  : 26,
            'color' : '#000'
        },
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        'plot_bgcolor' : '#fff',
        'colorway'     : largecolorset,
        'xaxis': {
            'showgrid'  : False,
            'showline'  : True,
            'linewidth' : 7,
            'linecolor' : 'black',
            'automargin': True,
            'mirror'    : True
        },
        'yaxis': {
            'showgrid'  : False,
            'showline'  : True,
            'linewidth' : 7,
            'linecolor' : 'black',
            'automargin': True,
            'mirror'    : True
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
    verbose: bool = True
):
    if show:
        fig.update_layout(
            plot_bgcolor='#fff', 
            paper_bgcolor='#fff', 
            height=1000
        )
        fig.show()
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)'
        )   
    
    if output_path:
        output_path = Path(output_path)
        output_dir = output_path.parent.resolve()
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if 'png' in save_as:
            try:
                fig.write_image(
                    str(output_path.with_suffix('.png')),  # Better path handling
                    format='png', 
                    scale=3,
                    engine='kaleido'  # Explicitly specify engine
                )
                if verbose:
                    logger.info(f"Saved figure to '{output_path.with_suffix('.png')}'.")
            except Exception as e:
                logger.error(f"Failed to save PNG: {str(e)}")
                if verbose:
                    logger.info("You may need to install kaleido: pip install -U kaleido")
        
        if 'html' in save_as:
            try:
                fig.write_html(str(output_path.with_suffix('.html')))
                if verbose:
                    logger.info(f"Saved figure to '{output_path.with_suffix('.html')}'.")
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
    Color map markers based on a column.
    
    Args:
        df: 
        col:
        continuous_color_set:

    Returns:
        color_map_dict:
        color_map_col:
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
    color_col: str, 
    show: bool = False,
    output_path: Union[str, Path] = None
) -> None:
    """
    Creates a legend image from a dictionary where the keys are 
    labels and the values are hex colors.
    
    Args:
        color_dict:  A dictionary where the keys are strings 
                     (labels) and values are hex colors.
        output_file: The path to save the generated legend image.
    """
    n = len(color_dict) # Number of items in the legend
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(3, n * 0.5))
    ax.axis('off') # Remove axes
    
    # Iterate over the dictionary and add each label and color to the plot
    for i, (label, color) in enumerate(color_dict.items()):
        ax.add_patch(
          plt.Rectangle((0, i), 1, 1, color=color)
        )
        ax.text(
          1.2, i + 0.5, 
          label, 
          va='center', 
          fontsize=16, 
          color='#000'
        )
    
    # Set the limits of the plot to fit all items
    ax.set_xlim(0, 3)
    ax.set_ylim(0, n)

    
    # Save the figure
    try:
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
        if show:
            plt.show()
    except:
        if output_path:
            plotly_show_and_save(
                fig,
                show=show,
                output_path=output_path   
            )
    plt.close()
    return fig
