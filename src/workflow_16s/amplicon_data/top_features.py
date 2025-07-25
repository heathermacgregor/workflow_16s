# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level
from workflow_16s.stats.tests import (
    fisher_exact_bonferroni, kruskal_bonferroni, mwu_bonferroni, ttest
)
from workflow_16s.utils.data import update_table_and_meta
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

def top_features_plots(
    output_dir,
    config,
    top_features,
    tables,
    meta,
    nfc_facilities,
    verbose
):
    # Create output directory for top features
    output_dir = output_dir / 'top_features'
    output_dir.mkdir(parents=True, exist_ok=True)

    n = config.get('violin_plots', {}).get('n', 50)

    for col, vals in top_features.items():
        for val, features in vals.items():
            group_key = f"{col}={val}"
            with get_progress_bar() as progress:
                groupval_desc = f"Processing '{col}'={val} features"
                groupval_task = progress.add_task(_format_task_desc(groupval_desc), total=len(features))
                for feature in features[:n]:
                    table_type = feature['table_type']
                    level = feature['level']
                    feature_name = feature['feature']
                    try:
                        feature['figures'] = {}
                                        
                        # Get the table and convert to DataFrame
                        biom_table = tables[table_type][level]
                        table = table_to_df(biom_table)[[feature_name]]
                        meta_ids = meta['#sampleid'].astype(str).str.strip().str.lower()
                        table_ids = table.index.astype(str).str.strip().str.lower()
                        shared_ids = set(table_ids) & set(meta_ids)
                    
                        group_map = meta.assign(norm_id=meta_ids).set_index("norm_id")[col]
                        # Create normalized table index
                        table_normalized_index = table.index.astype(str).str.strip().str.lower()
                        # Map group values using normalized IDs
                        table[group_col] = table_normalized_index.map(group_map)
                                        
                        # Verify feature exists
                        if feature_name not in table.columns:
                            logger.warning(f"Feature '{feature_name}' not found in {table_type}/{level} table")
                            continue
    
                        if data.config.get('violin_plots', {}).get('enabled', False):
                            # Create output directory
                            feature_output_dir = output_dir / col / val / table_type / level
                            feature_output_dir.mkdir(parents=True, exist_ok=True)
                            try:
                                # Generate violin plot
                                fig = violin_feature(
                                    df=table,
                                    feature=feature_name,
                                    output_dir=feature_output_dir,
                                    status_col=col
                                )
                                feature['figures']['violin'] = fig
                            except Exception as e:
                                logger.error(f"Failed violin plot for {feature_name} at {level} level: {e}")
                                feature['figures']['violin'] = None
    
                        if data.config.get('feature_maps', {}).get('enabled', False):
                            try:
                                # Generate feature abundance map
                                fig_map = create_feature_abundance_map(
                                    metadata=meta,
                                    feature_abundance=table[[feature_name]],
                                    feature_name=feature_name,
                                    nfc_facilities_data=nfc_facilities,
                                    output_dir=feature_output_dir,
                                    show=False,
                                    verbose=verbose
                                )
                                feature['figures']['abundance_map'] = fig_map
                            except Exception as e:
                                logger.error(f"Failed feature map for {feature_name} at {level} level: {e}")
                                feature['figures']['abundance_map'] = None
                              
                    except Exception as e:
                        logger.error(
                            f"Failed plots for {feature_name} at {table_type}/{level}: {str(e)}"
                        )
                        feature['figures'] = None
                    finally:
                        progress.update(groupval_task, advance=1)
        progress.update(groupval_task, description=_format_task_desc(groupval_desc))    
        return top_features
