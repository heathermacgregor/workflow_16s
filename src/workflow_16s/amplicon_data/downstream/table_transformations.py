# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
from biom.table import Table

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.helpers import _init_dict_level, _ProcessingMixin
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence, table_to_df, 
    update_table_and_meta, to_biom
)
from workflow_16s.utils.io import (
    export_h5py, import_merged_metadata_tsv, import_merged_table_biom
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.utils.nfc_facilities import find_nearby_nfc_facilities
from workflow_16s.amplicon_data.downstream.load import update_table_and_metadata
# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

class DownstreamTableTransformations:
    ModeConfig = {
        "any": ("asv", "table", "asv"), 
        "genus": ("genus", "table_6", "l6")
    }
    def __init__(
        self,
        config: Dict,
        tables: Dict,#Table,
        metadata: Dict,#pd.DataFrame,
        mode: str,
        project_dir: Union[str, Path]
    ) -> None:
        self.config = config
        self.mode = mode
        self.project_dir = project_dir
        self.tables, self.metadata = tables, metadata

        self.base_level, _, _ = self.ModeConfig[self.mode]
        self._collapse_taxonomy()

    def _collapse_taxonomy(self) -> None:
        with get_progress_bar() as progress:
            ct_desc = "Collapsing taxonomy"
            ct_task = progress.add_task(_format_task_desc(ct_desc), len(constants.levels))   
            for level in constants.levels:
                level_desc = f"{ct_desc} → {level.title()}"
                progress.update(ct_task, description=_format_task_desc(level_desc))
                try:
                    self.tables[level], self.metadata[level] = {}, {}
                    base_table, base_metadata = self.tables["genus"]["raw"], self.metadata["genus"]["raw"]
                    self.tables[level]["raw"] = collapse_taxa(base_table, level, progress, ct_task)
                except Exception as e:
                    logger.error(f"Taxonomic collapse failed for {level}: {e}")
                    self.tables[level]["raw"] = None
                finally:
                    progress.update(ct_task, advance=1)
                    
            progress.update(ct_task, description=_format_task_desc(ct_desc))

    def _apply_preprocessing(self) -> None:
        features_config = self.config.get("features", {})
        table = self.tables["raw"][self.mode]

        if features_config.get("filter", True):
            table = filter(table)
            table, metadata = update_table_and_metadata(table, metadata)
            self.tables.setdefault("filtered", {})[self.mode] = table

        if features_config.get("normalize", True):
            table = normalize(table, axis=1)
            self.tables.setdefault("normalized", {})[self.mode] = table

        if features_config.get("clr_transform", True):
            table = clr(table)
            self.tables.setdefault("clr_transformed", {})[self.mode] = table


class _TableProcessor(_ProcessingMixin):
    """Processes feature tables through various transformations and taxonomical collapses."""
    
    def __init__(
        self,
        config: Dict,
        table: Table,
        mode: str,
        meta: pd.DataFrame,
        output_dir: Path,
        project_dir: Any,
        verbose: bool,
    ) -> None:
        self.config, self.mode, self.verbose = config, mode, verbose
        self.project_dir, self.output_dir = project_dir, output_dir
        self.meta = meta
        self.tables: Dict[str, Dict[str, Table]] = {"raw": {mode: table}}
        self.levels = ["phylum", "class", "order", "family", "genus"]
        
        self._apply_preprocessing()
        self._collapse_taxa()
        self._create_presence_absence()
        self._save_tables()

    def _apply_preprocessing(self) -> None:
        feat_config = self.config.get("features", {})
        table = self.tables["raw"][self.mode]

        if feat_config.get("filter", True):
            table = filter(table)
            self.tables.setdefault("filtered", {})[self.mode] = table

        if feat_config.get("normalize", True):
            table = normalize(table, axis=1)
            self.tables.setdefault("normalized", {})[self.mode] = table

        if feat_config.get("clr_transform", True):
            table = clr(table)
            self.tables.setdefault("clr_transformed", {})[self.mode] = table

    def _collapse_taxa(self) -> None:
        with get_progress_bar() as progress:
            ct_desc = "Collapsing taxonomy"
            ct_task = progress.add_task(
                _format_task_desc(ct_desc), 
                total=len(self.tables) * len(self.levels)
            )   
            
            for table_type in list(self.tables.keys()):
                table_desc = f"{table_type.replace('_', ' ').title()}"
                table_task = progress.add_task(
                    _format_task_desc(table_desc),
                    parent=ct_task,
                    total=len(self.levels)
                )
                
                base_table = self.tables[table_type][self.mode]
                processed = {}
                for level in self.levels:
                    level_desc = f"{table_desc} → {level.title()}"
                    progress.update(
                        table_task, 
                        description=_format_task_desc(level_desc)
                    )
                    try:
                        start_time = time.perf_counter()
                        processed[level] = collapse_taxa(
                            base_table, 
                            level, 
                            progress, table_task
                        )
                        duration = time.perf_counter() - start_time
                        logger.debug(
                            f"Collapsed {table_type} to {level} in {duration:.2f}s"
                        )
                    except Exception as e:
                        logger.error(
                            f"Taxonomic collapse failed for {table_type}/{level}: {e}"
                        )
                        processed[level] = None
                    finally:
                        progress.update(table_task, advance=1)
                        progress.update(ct_task, advance=1)
                    
                self.tables[table_type] = processed
                progress.remove_task(table_task)
    
    def _create_presence_absence(self) -> None:
        if not self.config.get("features", {}).get("presence_absence", False):
            return
        with get_progress_bar() as progress:
            pa_desc = "Converting to Presence/Absence"
            pa_task = progress.add_task(
                _format_task_desc(pa_desc),
                total=len(self.levels)  
            )
            processed = {}
            
            for level in self.levels:
                level_desc = f"Converting to Presence/Absence → {level.capitalize()}"
                progress.update(
                    pa_task, 
                    description=_format_task_desc(level_desc)
                )
                try:
                    collapsed_table = self.tables["raw"][level]
                    start_time = time.perf_counter()
                    processed[level] = presence_absence(collapsed_table)
                    duration = time.perf_counter() - start_time
                    if self.verbose:
                        logger.debug(
                            f"Created Presence/Absence table for {level} in {duration:.2f}s"
                        )
                except Exception as e:
                    logger.error(f"Presence/Absence failed for {level}: {e}")
                    processed[level] = None
                finally:
                    progress.update(pa_task, advance=1)
                
            self.tables["presence_absence"] = processed
            progress.update(
                pa_task, 
                description=_format_task_desc(pa_desc)
            )
            
    def _save_tables(self) -> None:
        # Create directory if it doesn't exist
        base = Path(self.project_dir.data) / "merged" / "table"
        base.mkdir(parents=True, exist_ok=True)
    
        # Prepare export tasks
        export_tasks = []
        for table_type, levels in self.tables.items():
            tdir = base / table_type
            tdir.mkdir(parents=True, exist_ok=True)
            for level, table in levels.items():
                out = tdir / f"{level}.biom"  # Simplified filename
                out.parent.mkdir(parents=True, exist_ok=True)
                export_tasks.append((table, out))
    
        # Use ThreadPoolExecutor for parallel exports
        max_workers = self.config.get("threads", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for table, out_path in export_tasks:
                future = executor.submit(export_h5py, table, out_path)
                futures[future] = (table_type, level, out_path)
    
            # Track progress
            with get_progress_bar() as progress:
                task = progress.add_task(
                    _format_task_desc("Exporting tables"), 
                    total=len(export_tasks)
                )
                for future in as_completed(futures):
                    table_type, level, out_path = futures[future]
                    try:
                        future.result()
                        if self.verbose:
                            logger.debug(
                                f"Exported {table_type}/{level} to {out_path}"
                            )
                    except Exception as e:
                        logger.error(f"Failed to export {out_path}: {str(e)}")
                    finally:
                        progress.update(task, advance=1)
