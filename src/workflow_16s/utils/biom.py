# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from biom import load_table
from biom.table import Table

# Local Imports
from workflow_16s import constants

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ==================================== FUNCTIONS ===================================== #

def export_h5py(
    table: Table,
    output_path: Union[str, Path]
):
    """"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        table.to_hdf5(f, generated_by=f"Table")
        

def convert_to_biom(
    table: Union[pd.DataFrame, Table]
) -> Table:
    """Convert pandas DataFrame to BIOM Table.
    
    Args:
        table: Input DataFrame containing feature counts.
    
    Returns:
        BIOM Table representation of the DataFrame.
    """
    if isinstance(table, Table):
        return table
    
    observation_ids = table.index.astype(str).tolist()
    sample_ids = table.columns.astype(str).tolist()
    data = table.values
    
    return Table(
        data=data,
        observation_ids=observation_ids,
        sample_ids=sample_ids,
        type="OTU table"
    )
  

def collapse_taxa(
    table: Union[pd.DataFrame, Table], 
    target_level: str, 
    verbose: bool = True
) -> Table:
    """Collapse feature table to specified taxonomic level.
    
    Args:
        table:        Input BIOM Table or DataFrame.
        target_level: Taxonomic level to collapse to (phylum/class/order/family).
        output_dir:   Directory to save collapsed table.
    
    Returns:
        Collapsed BIOM Table.
    
    Raises:
        ValueError: For invalid target_level.
    """
    table = table.copy()
    table = convert_to_biom(table)
        
    if target_level not in constants.levels:
        raise ValueError(
            f"Invalid `target_level`: {target_level}. "
            f"Expected one of {list(constants.levels.keys())}")

    level_idx = constants.levels[target_level]

    # Create taxonomy mapping
    id_map = {}
    for taxon in table.ids(axis='observation').astype(str):
        parts = taxon.split(';')
        truncated = ';'.join(
            parts[:level_idx + 1]
        ) if len(parts) >= level_idx + 1 else 'Unclassified'
        id_map[taxon] = truncated

    # Collapse table
    collapsed_table = table.collapse(
        lambda id, _: id_map.get(id, 'Unclassified'),
        norm=False,
        axis='observation',
        include_collapsed_metadata=False
    ).remove_empty()
    
    return collapsed_table
  

def presence_absence(
    table: Union[Table, pd.DataFrame], 
    target_level: str, 
) -> Table:
    """Convert table to presence/absence format and filter by abundance.
    
    Args:
        table:        Input BIOM Table or DataFrame.
        target_level: Taxonomic level for output naming.
        output_dir:   Directory to save output.
    
    Returns:
        Presence/absence BIOM Table filtered by abundance.
    """
    table = table.copy()
    table = convert_to_biom(table)
    
    # Filter by abundance
    feature_sums = np.array(table.sum(axis='observation')).flatten()
    sorted_idx = np.argsort(feature_sums)[::-1]
    cumulative = np.cumsum(feature_sums[sorted_idx]) / feature_sums.sum()
    stop_idx = np.searchsorted(cumulative, 0.99) + 1
    keep_ids = [table.ids(axis='observation')[i] for i in sorted_idx[:stop_idx]]
    
    # Convert to presence/absence
    pa_table = table.pa(inplace=False)
    pa_table_filtered = pa_table.filter(keep_ids, axis='observation')
    pa_df_filtered = pa_table_filtered.to_dataframe(dense=True)

    # Save output
    pa_table = Table(
        pa_df_filtered.values,
        pa_df_filtered.index,
        pa_df_filtered.columns,
        table_id='Presence Absence BIOM Table'
    )
    
    return pa_table


def filter_presence_absence(
    table: Table, 
    metadata: pd.DataFrame, 
    col: str = 'nuclear_contamination_status', 
    prevalence_threshold: float = 0.05, 
    group_threshold: float = 0.05
) -> Table:
    """Filter presence/absence table based on prevalence and group differences.
    
    Args:
        table:                Input BIOM Table.
        metadata:             Sample metadata DataFrame.
        col:                  Metadata column to group by.
        prevalence_threshold: Minimum prevalence across all samples.
        group_threshold:      Minimum prevalence difference between groups.
    
    Returns:
        Filtered BIOM Table
    """
    df = table.to_dataframe(dense=True).T
    metadata = metadata.set_index("run_accession.1")
    df_with_meta = df.join(metadata[[col]], how='inner')

    # Apply prevalence filter
    if prevalence_threshold:
        species_data = df_with_meta.drop(columns=[col])
        prev = species_data.mean(axis=0)
        filtered_species = prev[prev >= prevalence_threshold].index
        df_with_meta = df_with_meta[filtered_species.union(pd.Index([col]))]

    # Apply group filter
    if group_threshold:
        groups = df_with_meta.groupby(col)
        if True not in groups.groups or False not in groups.groups:
            raise ValueError(f"Metadata column '{col}' must have True/False groups")
        sum_per_group = groups.sum(numeric_only=True)
        n_samples = groups.size()
        percentages = sum_per_group.div(n_samples, axis=0)
        mask = (percentages.loc[True] >= group_threshold) & (percentages.loc[False] >= group_threshold)
        selected_species = mask[mask].index
        df_with_meta = df_with_meta[selected_species.union(pd.Index([col]))]

    return Table(
        df_with_meta.drop(columns=[col]).values.T,
        df_with_meta.columns.tolist(),
        df_with_meta.index.tolist(),
        table_id='Filtered Presence/Absence Table'
    )
