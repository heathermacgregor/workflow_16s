# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import polars as pl
from biom.table import Table

# Local Imports
from workflow_16s import constants
from workflow_16s.amplicon_data.downstream.input import update_table_and_metadata
from workflow_16s.utils.data import (
    clr, collapse_taxa, filter, normalize, presence_absence
)
from workflow_16s.utils.io import export_h5py
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

class PrepData:
    ModeConfig = {
        "any": ("asv", "table", "asv"), 
        "genus": ("genus", "table_6", "l6")
    }
    
    def __init__(
        self,
        config: Dict,
        tables: Dict,
        metadata: Dict,
        mode: str,
        project_dir: Union[str, Path],
        verbose: bool = False
    ) -> None:
        self.config, self.project_dir, self.mode = config, project_dir, mode
        self.tables, self.metadata = tables, metadata
        self.verbose = verbose
        
        # Convert tables to Polars DataFrames for faster processing
        self._convert_to_polars()
        
        # Collapse raw tables at all taxonomy levels
        self._collapse_taxonomy("raw")

        # Apply preprocessing in parallel using Polars
        self._apply_preprocessing_parallel()

        # Save tables
        self._save_tables()

    def _convert_to_polars(self) -> None:
        """Convert biom tables to Polars DataFrames for faster processing"""
        for table_type in list(self.tables.keys()):
            for level in list(self.tables[table_type].keys()):
                table = self.tables[table_type][level]
                # Convert biom table to Polars DataFrame
                df = table.to_dataframe(dense=True).T
                self.tables[table_type][level] = pl.from_pandas(df)
                
                # Convert metadata to Polars if it's a pandas DataFrame
                if hasattr(self.metadata[table_type][level], 'to_pandas'):
                    self.metadata[table_type][level] = pl.from_pandas(
                        self.metadata[table_type][level].to_pandas()
                    )

    def _collapse_taxonomy(self, table_type: str = "raw") -> None:
        base_level = self.ModeConfig[self.mode][0]
        base_table = self.tables.setdefault(table_type, {}).get(base_level)
        base_metadata = self.metadata.setdefault(table_type, {}).get(base_level)

        if base_table is None or base_metadata is None:
            raise ValueError(
                f"Missing base table or metadata for {base_level} level in {table_type}"
            )

        with get_progress_bar() as progress:
            ct_desc = "Collapsing taxonomy"
            ct_task = progress.add_task(_format_task_desc(ct_desc), total=len(constants.levels))
            
            for level in constants.levels:
                level_desc = f"{ct_desc} {table_type} → {level.title()}"
                progress.update(ct_task, description=_format_task_desc(level_desc))
                
                if level == base_level:
                    table = base_table
                    metadata = base_metadata
                else:
                    table = collapse_taxa(base_table, level, progress, ct_task)
                    table, metadata = update_table_and_metadata(table, base_metadata)

                self.tables.setdefault(table_type, {})[level] = table
                self.metadata.setdefault(table_type, {})[level] = metadata
                progress.update(ct_task, advance=1)
                
            progress.update(ct_task, description=_format_task_desc(ct_desc))

    def _apply_preprocessing_parallel(self) -> None:
        """Apply preprocessing steps in parallel using Polars"""
        features_config = self.config.get("features", {})
        levels = list(self.tables['raw'].keys())
        
        with ThreadPoolExecutor(max_workers=self.config.get("threads", 4)) as executor:
            futures = {}
            for level in levels:
                table = self.tables['raw'][level]
                metadata = self.metadata['raw'][level]
                
                # Submit preprocessing tasks
                future = executor.submit(
                    self._process_single_level,
                    table, metadata, level, features_config
                )
                futures[future] = level

            # Process completed tasks
            with get_progress_bar() as progress:
                task = progress.add_task(
                    _format_task_desc("Preprocessing data"), 
                    total=len(futures)
                )
                for future in as_completed(futures):
                    level = futures[future]
                    try:
                        result = future.result()
                        for table_type, (table, metadata) in result.items():
                            self.tables.setdefault(table_type, {})[level] = table
                            self.metadata.setdefault(table_type, {})[level] = metadata
                    except Exception as e:
                        logger.error(f"Preprocessing failed for level {level}: {e}")
                    finally:
                        progress.update(task, advance=1)

    def _process_single_level(
        self,
        table: pl.DataFrame,
        metadata: pl.DataFrame,
        level: str,
        features_config: Dict[str, Any]
    ) -> Dict[str, Tuple[pl.DataFrame, pl.DataFrame]]:
        """Process a single taxonomic level with all preprocessing steps"""
        results = {}
        current_table = table
        current_metadata = metadata

        steps = [
            ("filter", filter, "filtered"),
            ("normalize", normalize, "normalized"),
            ("clr_transform", clr, "clr_transformed")
        ]

        for config_key, func, table_type in steps:
            if features_config.get(config_key, True):
                try:
                    current_table = func(current_table)
                    current_table, current_metadata = update_table_and_metadata(
                        current_table, current_metadata
                    )
                    results[table_type] = (current_table, current_metadata)
                    
                    # Create presence/absence table if requested
                    if features_config.get("presence_absence", False):
                        pa_table = presence_absence(current_table)
                        pa_table, pa_metadata = update_table_and_metadata(
                            pa_table, current_metadata
                        )
                        results[f"{table_type}_presence_absence"] = (pa_table, pa_metadata)
                        
                except Exception as e:
                    logger.error(f"{config_key} failed for {level}: {e}")
                    continue

        return results

    def _save_tables(self) -> None:
        base = Path(self.project_dir.data) / "merged" / "table"
        base.mkdir(parents=True, exist_ok=True)
    
        export_tasks = []
        for table_type, levels in self.tables.items():
            tdir = base / table_type
            tdir.mkdir(parents=True, exist_ok=True)
            for level, table in levels.items():
                # Convert back to biom table before saving
                biom_table = self._polars_to_biom(table)
                out = tdir / f"{level}.biom"
                export_tasks.append((biom_table, out))
    
        # Parallel export
        max_workers = self.config.get("threads", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(export_h5py, table, out_path): (table_type, level, out_path)
                for table, out_path in export_tasks
            }
    
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
                            logger.debug(f"Exported {table_type}/{level} to {out_path}")
                    except Exception as e:
                        logger.error(f"Failed to export {out_path}: {str(e)}")
                    finally:
                        progress.update(task, advance=1)

    def _polars_to_biom(self, df: pl.DataFrame) -> Table:
        """Convert Polars DataFrame back to biom Table"""
        pandas_df = df.to_pandas().T
        return Table(
            pandas_df.values,
            observation_ids=pandas_df.index.tolist(),
            sample_ids=pandas_df.columns.tolist()
        )
