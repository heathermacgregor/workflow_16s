# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from typing import Union

# Third-Party Imports
import pandas as pd
from biom import Table
from scipy.sparse import issparse
from skbio.stats.composition import clr as CLR

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s import constants
from workflow_16s.utils.table_conversion import to_biom

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ========================== TABLE NORMALIZATION & TRANSFORM ========================= #

def normalize(
    table: Union[dict, Table, pd.DataFrame], 
    axis: int = 1
) -> Table:
    """Normalize table to relative abundance with strict type enforcement.
    
    Args:
        table: Input table.
        axis:  Normalization axis (0=features, 1=samples).
        
    Returns:
        Normalized BIOM Table.
        
    Raises:
        ValueError: For invalid axis values.
    """
    if axis not in (0, 1):
        raise ValueError(f"Invalid axis: {axis}. Must be 0 (features) or 1 (samples)")
      
    biom_table = to_biom(table)
    
    if axis == 1:  # Sample-wise normalization (convert to relative abundance)
        return biom_table.norm(axis='sample')
    else:  # Feature-wise normalization
        return biom_table.norm(axis='observation')
        

def clr(
    table: Union[dict, Table, pd.DataFrame], 
    pseudocount: float = constants.DEFAULT_PSEUDOCOUNT
) -> Table:
    """Apply centered log-ratio (CLR) transformation to table.
    
    Args:
        table:       Input table.
        pseudocount: Small value to add to avoid log(0).
        
    Returns:
        CLR-transformed BIOM Table.
    """
    biom_table = to_biom(table)
    
    # Convert to dense array (samples x features)
    if issparse(biom_table.matrix_data):
        dense_data = biom_table.matrix_data.toarray().T
    else:
        dense_data = biom_table.matrix_data.T
    
    # Apply CLR transformation
    clr_data = CLR(dense_data + pseudocount)
    
    # Transpose back to features x samples
    clr_data = clr_data.T
    
    # Create new BIOM Table with original metadata
    return Table(
        data=clr_data,
        observation_ids=biom_table.ids(axis='observation'),
        sample_ids=biom_table.ids(axis='sample'),
        observation_metadata=biom_table.metadata(axis='observation'),
        sample_metadata=biom_table.metadata(axis='sample')
    )