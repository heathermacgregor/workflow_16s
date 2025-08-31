# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from typing import Dict, Union

# Third-Party Imports
import pandas as pd
from biom import Table

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================ TABLE CONVERSION ================================== #

def table_to_df(table: Union[Dict, Table, pd.DataFrame]) -> pd.DataFrame:
    """Convert various table formats to samples × features DataFrame.
    
    Handles:
    - Pandas DataFrame (returns unchanged)
    - BIOM Table (transposes to samples × features)
    - Dictionary (converts to DataFrame)
    
    Args:
        table: Input table in various formats.
        
    Returns:
        DataFrame in samples × features orientation.
        
    Raises:
        TypeError: For unsupported input types
    """
    if isinstance(table, pd.DataFrame):  # samples × features
        return table
    if isinstance(table, Table):     # features × samples
        return table.to_dataframe(dense=True).T
    if isinstance(table, dict):          # samples × features
        return pd.DataFrame(table)
    raise TypeError("Input must be BIOM Table, dict, or DataFrame.")


def to_biom(table: Union[dict, Table, pd.DataFrame]) -> Table:
    """Convert various table formats to BIOM Table with features × samples 
    orientation.
    
    Args:
        table: Input table in various formats.
        
    Returns:
        BIOM Table object.
        
    Raises:
        TypeError: For unsupported input types.
    """
    if isinstance(table, Table):
        return table
    elif isinstance(table, pd.DataFrame):
        # Convert from samples × features to features × samples for BIOM
        return Table(table.values.T, 
                    observation_ids=table.columns, 
                    sample_ids=table.index)
    elif isinstance(table, dict):
        df = pd.DataFrame(table)
        return Table(df.values.T, 
                    observation_ids=df.columns, 
                    sample_ids=df.index)
    else:
        raise TypeError("Input must be BIOM Table, dict, or DataFrame.")