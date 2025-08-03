# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

class _DataLoader(_ProcessingMixin):
    """Loads and processes microbiome data from BIOM files and metadata."""
    
    MODE_CONFIG = {
        "asv": ("table", "asv"), 
        "genus": ("table_6", "l6")
    }
    TAXONOMY_LEVELS = ["phylum", "class", "order", "family", "genus"]

    def __init__(
        self, 
        config: Dict, 
        project_dir: Any, 
        mode: str, 
        existing_subsets: Dict[str, Dict[str, Path]] = None,
        verbose: bool = False
    ):
        self.cfg, self.project_dir, self.mode = config, project_dir, mode
        self.existing_subsets, self.verbose = existing_subsets, verbose
        
        self._validate_mode()
        self._load_metadata()
        self._load_biom_table()
        self._filter_and_align()
        
    # Type hints
    meta: pd.DataFrame
    nfc_facilities: pd.DataFrame
    meta_nfc_facilities: pd.DataFrame
    table: Table

    def _validate_mode(self) -> None:
        if self.mode not in self.MODE_CONFIG:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _get_metadata_paths(self) -> List[Path]:
        metadata_paths = [paths["metadata"] 
                          for subset_id, paths in self.existing_subsets.items()]
        if self.verbose:
            logger.info(f"Found {len(metadata_paths)} metadata files")
        return metadata_paths

    def _get_metadata_paths_glob(self) -> List[Path]:
        paths: List[Path] = []
        for bi in self._get_biom_paths_glob():
            ds_dir = bi.parent if bi.is_file() else bi
            tail = ds_dir.parts[-6:-1]
            mp = Path(self.project_dir.metadata_per_dataset).joinpath(
                *tail, "sample-metadata.tsv"
            )
            if mp.exists():
                paths.append(mp)
        if self.verbose:
            logger.info(f"Found {len(paths)} metadata files")
        return paths

    def _load_metadata(self) -> None:
        if self.existing_subsets is None:
            paths = self._get_metadata_paths_glob()
        else:
            paths = self._get_metadata_paths()
            
        self.meta = import_merged_metadata_tsv(paths, None, self.verbose)

        # Remove duplicated columns
        if self.meta.columns.duplicated().any():
            duplicated_columns = self.meta.columns[self.meta.columns.duplicated()].tolist()
            logger.debug(
                f"Found duplicate columns in metadata: {duplicated_columns}. "
                "Removing duplicates."
            )
            self.meta = self.meta.loc[:, ~self.meta.columns.duplicated()]
            
        # NFC facilities handling
        if self.cfg.get("nfc_facilities", {}).get("enabled", False):
            nfc_meta, self.nfc_facilities, self.meta_nfc_facilities = find_nearby_nfc_facilities(
                cfg=self.cfg,
                meta=self.meta, 
                output_dir=self.project_dir.final
            )
            # Merge NFC results without overwriting original metadata
            self.meta = pd.concat([self.meta, nfc_meta], axis=1)
        else:
            self.nfc_facilities, self.meta_nfc_facilities = None, None
            
    def _get_biom_paths(self) -> List[Path]:
        table_dir, _ = self.MODE_CONFIG[self.mode]
        biom_paths = [paths[table_dir] for subset_id, paths in self.existing_subsets.items()]
        if self.verbose:
            logger.info(f"Found {len(biom_paths)} feature tables")
        return biom_paths

    def _get_biom_paths_glob(self) -> List[Path]:
        table_dir, _ = self.MODE_CONFIG[self.mode]
        if self.cfg["target_subfragment_mode"] != 'any' or self.mode != 'genus':
            pattern = "/".join([
                "*", "*", "*", self.cfg["target_subfragment_mode"], 
                "FWD_*_REV_*", table_dir, "feature-table.biom",
            ])
        else:
            pattern = "/".join([
                "*", "*", "*", "*", 
                "FWD_*_REV_*", table_dir, "feature-table.biom",
            ])
        globbed = glob.glob(str(Path(
            self.project_dir.qiime_data_per_dataset
        ) / pattern), recursive=True)
        if self.verbose:
            logger.info(f"Found {len(globbed)} feature tables")
        return [Path(p) for p in globbed]

    def _load_biom_table(self) -> None:
        if self.existing_subsets is None:
            biom_paths = self._get_biom_paths_glob()
        else:
            biom_paths = self._get_biom_paths()
            
        if not biom_paths:
            raise FileNotFoundError("No BIOM files found")
        self.table = import_merged_table_biom(biom_paths, "table", self.verbose)
        
        # Validate genus-level table format
        if self.mode == "genus":
            obs_ids = self.table.ids(axis='observation')
            if obs_ids.size > 0:
                first_id = obs_ids[0]
                if not first_id.startswith('g__'):
                    logger.warning(
                        f"First feature ID in genus table: '{first_id}' "
                        "doesn't start with 'g__'. May not be genus-level."
                    )
    
    def _filter_and_align(self) -> None:
        self.table, self.meta = update_table_and_meta(
            self.table, self.meta, 
            self.cfg.get("metadata_id_column", "#sampleid")
        )
        ftype = "genera" if self.mode == "genus" else "ASVs"
        logger.info(
            f"{'Loaded metadata:':<30}{self.meta.shape[0]:>6} samples "
            f"× {self.meta.shape[1]:>5} cols"
        )
        logger.info(
            f"{'Loaded features:':<30}{self.table.shape[1]:>6} samples "
            f"× {self.table.shape[0]:>5} {ftype}"
        )


class _TableProcessor(_ProcessingMixin):
    """Processes feature tables through various transformations and taxonomical collapses."""
    
    TAXONOMY_LEVELS = ["phylum", "class", "order", "family", "genus"]
    
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
        """Collapses taxonomy only from ASV-level to higher ranks"""
        with get_progress_bar() as progress:
            total_tasks = len(self.tables) * len(self.TAXONOMY_LEVELS)
            task = progress.add_task(
                _format_task_desc("Collapsing taxonomy"), 
                total=total_tasks
            )
            
            for table_type in list(self.tables.keys()):
                base_table = self.tables[table_type][self.mode]
                processed = {}
                
                for level in self.TAXONOMY_LEVELS:
                    # Update task description
                    desc = f"{table_type.title()} → {level.title()}"
                    progress.update(task, description=_format_task_desc(desc))
                    
                    # Skip invalid collapses
                    if self.mode != "asv" and level != "genus":
                        logger.warning(
                            f"Skipping {level} collapse - requires ASV input. "
                            f"Current mode: {self.mode}"
                        )
                        processed[level] = None
                        progress.update(task, advance=1)
                        continue
                    
                    try:
                        # Handle genus mode differently
                        if self.mode == "genus" and level == "genus":
                            processed["genus"] = base_table
                            if self.verbose:
                                logger.debug(f"Using existing genus table for {table_type}")
                        else:
                            start_time = time.perf_counter()
                            processed[level] = collapse_taxa(base_table, level)
                            duration = time.perf_counter() - start_time
                            if self.verbose:
                                logger.debug(
                                    f"Collapsed {table_type} to {level} "
                                    f"({base_table.shape} → {processed[level].shape}) "
                                    f"in {duration:.2f}s"
                                )
                    except Exception as e:
                        logger.error(
                            f"Taxonomic collapse failed for {table_type}/{level}: {e}"
                        )
                        processed[level] = None
                    finally:
                        progress.update(task, advance=1)
                
                self.tables[table_type] = processed
    
    def _create_presence_absence(self) -> None:
        if not self.config.get("features", {}).get("presence_absence", False):
            return
            
        with get_progress_bar() as progress:
            total_tasks = len(self.TAXONOMY_LEVELS)
            task = progress.add_task(
                _format_task_desc("Creating Presence/Absence"),
                total=total_tasks
            )
            processed = {}
            
            for level in self.TAXONOMY_LEVELS:
                # Skip invalid levels
                if self.mode != "asv" and level != "genus":
                    progress.update(task, advance=1)
                    continue
                    
                # Update description
                desc = f"Presence/Absence → {level.title()}"
                progress.update(task, description=_format_task_desc(desc))
                
                try:
                    collapsed_table = self.tables["raw"].get(level)
                    if collapsed_table is None:
                        logger.warning(
                            f"No table found for {level}, skipping Presence/Absence"
                        )
                        continue
                        
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
                    progress.update(task, advance=1)
                
            self.tables["presence_absence"] = processed
            
    def _save_tables(self) -> None:
        base = Path(self.project_dir.data) / "merged" / "table"
        base.mkdir(parents=True, exist_ok=True)
    
        export_tasks = []
        for table_type, levels in self.tables.items():
            tdir = base / table_type
            tdir.mkdir(parents=True, exist_ok=True)
            for level, table in levels.items():
                # Skip null tables from skipped operations
                if table is None:
                    continue
                    
                out = tdir / f"{level}.biom"
                out.parent.mkdir(parents=True, exist_ok=True)
                export_tasks.append((table, out))
    
        max_workers = self.config.get("threads", 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for table, out_path in export_tasks:
                future = executor.submit(export_h5py, table, out_path)
                futures[future] = (table_type, level, out_path)
    
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
