# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party Imports
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import textwrap

# Local Imports
from workflow_16s import constants
from workflow_16s.figures.figures import (
    attach_legend_to_figure, largecolorset, plot_legend, plotly_show_and_save    
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')
sns.set_style('whitegrid')  # Set seaborn style globally

# ================================= GLOBAL VARIABLES ================================= #


def fig_to_json(fig, output_path):
    # Convenience for optional INFO logging
    log_ok = (lambda msg: logger.debug(msg)) if verbose else (lambda *_: None)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_json(output_path, engine="json")
        #fig.write_html(output_path, include_plotlyjs="cdn")
        log_ok(f"Saved figure to '{output_path}'")
    except Exception as e:
        logger.error(f"Failed to save figure: {e}")


import pandas as pd
import logging
from typing import Optional, Union, List
from functools import wraps
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DataPrepError(Exception):
    """Raised when data preparation encounters unrecoverable issues."""
    pass

@contextmanager
def prep_context(verbose: bool = False):
    """Context manager for data preparation with optional verbose logging."""
    if verbose:
        level = logger.getEffectiveLevel()
        logger.setLevel(logging.DEBUG)
    try:
        yield
    finally:
        if verbose:
            logger.setLevel(level)

def prep_step(description: str):
    """Decorator to log preparation steps."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.debug(f"→ {description}")
            result = func(*args, **kwargs)
            logger.debug(f"✓ {description}")
            return result
        return wrapper
    return decorator

class DataPrep:
    """Elegant data preparation for visualization with fluent interface."""
    
    def __init__(self, data: pd.DataFrame, metadata: pd.DataFrame, verbose: bool = False):
        self.data = data.copy()
        self.metadata = metadata.copy()
        self.verbose = verbose
        self.placeholder = 'unknown'
    
    def with_placeholder(self, value: str) -> 'DataPrep':
        """Set placeholder value for missing data."""
        self.placeholder = value
        return self
    
    @prep_step("Normalizing sample indices")
    def _normalize_indices(self) -> 'DataPrep':
        """Normalize indices to lowercase strings, handling special cases."""
        # Data index
        self.data.index = self.data.index.astype(str).str.strip().str.lower()
        
        # Metadata index - prefer #sampleid if available
        if '#sampleid' in self.metadata.columns:
            self.metadata.index = (self.metadata['#sampleid']
                                 .astype(str).str.strip().str.lower())
        else:
            self.metadata.index = self.metadata.index.astype(str).str.strip().str.lower()
        
        return self
    
    @prep_step("Removing duplicate samples")
    def _remove_duplicates(self) -> 'DataPrep':
        """Remove duplicate indices, keeping first occurrence."""
        initial_data_size = len(self.data)
        initial_meta_size = len(self.metadata)
        
        self.data = self.data[~self.data.index.duplicated(keep='first')]
        self.metadata = self.metadata[~self.metadata.index.duplicated(keep='first')]
        
        if self.verbose:
            data_removed = initial_data_size - len(self.data)
            meta_removed = initial_meta_size - len(self.metadata)
            if data_removed:
                logger.info(f"Removed {data_removed} duplicate data samples")
            if meta_removed:
                logger.info(f"Removed {meta_removed} duplicate metadata samples")
        
        return self
    
    @prep_step("Finding sample intersection")
    def _find_intersection(self) -> 'DataPrep':
        """Find and validate common samples between datasets."""
        self.common_samples = self.data.index.intersection(self.metadata.index)
        
        if len(self.common_samples) == 0:
            self._diagnose_mismatch()
            raise DataPrepError("No common samples found between data and metadata")
        
        if self.verbose:
            logger.info(f"Found {len(self.common_samples)} common samples")
        
        return self
    
    def _diagnose_mismatch(self):
        """Provide diagnostic information for sample mismatches."""
        data_samples = set(self.data.index[:10])  # Sample for diagnosis
        meta_samples = set(self.metadata.index)
        
        # Look for partial matches
        matches = [(d, m) for d in data_samples for m in meta_samples 
                  if d in m or m in d]
        
        logger.error("Sample ID mismatch detected")
        logger.error(f"Data samples: {list(data_samples)}")
        logger.error(f"Metadata samples: {list(meta_samples)[:10]}")
        if matches:
            logger.error(f"Potential matches: {matches[:3]}")
    
    @prep_step("Preparing metadata columns")
    def _prepare_columns(self, required_cols: List[str]) -> 'DataPrep':
        """Ensure required columns exist with appropriate defaults."""
        for col in required_cols:
            if col not in self.metadata.columns:
                if self.verbose:
                    logger.warning(f"Missing column '{col}' - using placeholder")
                self.metadata[col] = self.placeholder
        
        return self
    
    @prep_step("Merging datasets")
    def _merge(self, color_col: str, symbol_col: str) -> pd.DataFrame:
        """Merge data with metadata on common samples."""
        # Filter to common samples
        data_filtered = self.data.loc[self.common_samples]
        meta_filtered = self.metadata.loc[self.common_samples]
        
        # Remove conflicting columns from data
        conflicts = [col for col in [color_col, symbol_col] if col in data_filtered.columns]
        if conflicts:
            data_filtered = data_filtered.drop(columns=conflicts)
        
        # Merge and fill missing values
        merged = data_filtered.join(meta_filtered[[color_col, symbol_col]])
        merged[[color_col, symbol_col]] = merged[[color_col, symbol_col]].fillna(self.placeholder)
        
        if self.verbose:
            logger.info(f"Final dataset shape: {merged.shape}")
        
        return merged
    
    def prepare(self, color_col: str, symbol_col: str) -> pd.DataFrame:
        """Execute the complete preparation pipeline."""
        with prep_context(self.verbose):
            return (self
                   ._normalize_indices()
                   ._remove_duplicates()
                   ._find_intersection()
                   ._prepare_columns([color_col, symbol_col])
                   ._merge(color_col, symbol_col))

def prepare_visualization_data(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    color_col: str,
    symbol_col: str,
    placeholder: str = 'unknown',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Elegantly prepare data for visualization by merging datasets on common samples.
    
    Handles the common data preparation tasks:
    • Index normalization and cleanup
    • Duplicate sample removal  
    • Sample intersection validation
    • Missing metadata column creation
    • Clean dataset merging
    
    Args:
        data: Primary dataset with samples as index
        metadata: Sample metadata with matching identifiers
        color_col: Column for visualization colors
        symbol_col: Column for visualization symbols
        placeholder: Value for missing metadata (default: 'unknown')
        verbose: Enable detailed logging (default: False)
    
    Returns:
        Clean merged DataFrame ready for visualization
        
    Raises:
        DataPrepError: When no common samples exist or other critical issues
        
    Example:
        >>> viz_data = prepare_visualization_data(
        ...     pca_results, sample_metadata, 
        ...     color_col='treatment', symbol_col='timepoint'
        ... )
    """
    if [col for col in ['#sample_id', color_col, symbol_col] if col not in metadata.columns]:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = DataPrep(data, metadata, verbose).with_placeholder(placeholder).prepare(color_col, symbol_col)
    df['sample_id'] = df.index
    df = df.loc[:, ~df.columns.duplicated()]
    return df

# Backward compatibility
_prepare_visualization_data = prepare_visualization_data


class BetaDiversityPlot:
    def __init__(self, ):

    def test():
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
        # Prepare data
        data = prepare_visualization_data(components, metadata, color_col, symbol_col)
        
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
        x_col, y_col = f'{prefix}{x_dim}', f'{prefix}{y_dim}'

        # Create plot
        fig = _create_base_scatter_plot(
            data,
            x_col,
            y_col,
            color_col,
            symbol_col,
            colordict,
            hover_data=['sample_id', color_col, symbol_col]
        )
        
        # Apply layout
        title = f'{ordination_type}: {transformation.title() if transformation else "Raw Data"}'
        if proportion_explained is not None and len(proportion_explained) >= max(x_dim, y_dim):
            x_title = f"{x_col} ({proportion_explained[x_dim-1]*100:.1f}%)"
            y_title = f"{y_col} ({proportion_explained[y_dim-1]*100:.1f}%)"
        else:
            x_title, y_title = x_col, y_col
        fig = _apply_common_layout(fig, x_title, y_title, title)
    
        fig.update_layout(
            width=1600,
            title=dict(font=dict(size=24)),
            xaxis=dict(title=dict(font=dict(size=20)), scaleanchor="y", scaleratio=1.0),
            yaxis=dict(title=dict(font=dict(size=20)))
        )
        
        # Save output
        if output_dir:
            file_stem = output_dir / ordination_type.lower() / f"{ordination_type.lower()}.{transformation or 'raw'}.{x_dim}-{y_dim}.{color_col}"            
            fig_to_json(fig, f"{file_stem}.json")
            
        return fig, colordict
