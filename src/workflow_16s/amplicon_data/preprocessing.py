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

# ================================== LOCAL IMPORTS =================================== #

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
    """
    Loads and processes microbiome data from BIOM files and metadata.
    """
    
    MODE_CONFIG = {
        "asv": ("table", "asv"), 
        "genus": ("table_6", "l6")
    }

    def __init__(
        self, 
        cfg: Dict, 
        project_dir: Any, 
        mode: str, 
        existing_subsets: Dict[str, Dict[str, Path]] = None,
        verbose: bool = False
    ):
        self.cfg, self.project_dir, self.mode, self.existing_subsets, self.verbose = cfg, project_dir, mode, existing_subsets, verbose
        self._validate_mode()
        self._load_metadata()
        self._load_biom_table()
        self._filter_and_align()

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
            (f"Found {len(metadata_paths)} metadata files")
        return metadata_paths

    def _get_metadata_paths_glob(self) -> List[Path]:
        paths: List[Path] = []
        for bi in self._get_biom_paths_glob():
            ds_dir = bi.parent if bi.is_file() else bi
            tail = ds_dir.parts[-6:-1]
            mp = Path(self.project_dir.metadata_per_dataset).joinpath(*tail, "sample-metadata.tsv")
            if mp.exists():
                paths.append(mp)
        if self.verbose:
            logger.info(f"Found {len(paths)} metadata files")
        return paths

    def _load_metadata(self) -> None:
        if self.existing_subsets == None:
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
            
        # If enabled, find samples within a threshold distance from NFC facilities
        if self.cfg.get("nfc_facilities", {}).get("enabled", False):
            self.meta, self.nfc_facilities, self.meta_nfc_facilities = find_nearby_nfc_facilities(cfg=self.cfg, meta=self.meta)
            
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
        if self.existing_subsets == None:
            biom_paths = self._get_biom_paths_glob()
        else:
            biom_paths = self._get_biom_paths()
            
        if not biom_paths:
            raise FileNotFoundError("No BIOM files found")
        self.table = import_merged_table_biom(biom_paths, "table", self.verbose)
    
    def _filter_and_align(self) -> None:
        self.table, self.meta = update_table_and_meta(
            self.table, self.meta, 
            self.cfg.get("metadata_id_column", "#sampleid")
        )
        ftype = "genera" if self.mode == "genus" else "ASVs"
        logger.info(
            f"{'Loaded metadata:':<30}{self.meta.shape[0]:>6} samples × {self.meta.shape[1]:>5} cols"
        )
        logger.info(
            f"{'Loaded features:':<30}{self.table.shape[1]:>6} samples × {self.table.shape[0]:>5} {ftype}"
        )


class _TableProcessor(_ProcessingMixin):
    """
    Processes feature tables through various transformations and taxonomical collapses.
    """
    
    def __init__(
        self,
        cfg: Dict,
        table: Table,
        mode: str,
        meta: pd.DataFrame,
        output_dir: Path,
        project_dir: Any,
        verbose: bool,
    ) -> None:
        self.cfg, self.mode, self.verbose = cfg, mode, verbose
        self.meta = meta
        self.output_dir = output_dir
        self.project_dir = project_dir
        self.levels = ["phylum", "class", "order", "family", "genus"]
        self.tables: Dict[str, Dict[str, Table]] = {"raw": {mode: table}}
        self._apply_preprocessing()
        self._collapse_taxa()
        self._create_presence_absence()
        self._save_tables()

    def _apply_preprocessing(self) -> None:
        feat_cfg = self.cfg.get("features", {})
        table = self.tables["raw"][self.mode]

        if feat_cfg.get("filter", True):
            table = filter(table)
            self.tables.setdefault("filtered", {})[self.mode] = table

        if feat_cfg.get("normalize", True):
            table = normalize(table, axis=1)
            self.tables.setdefault("normalized", {})[self.mode] = table

        if feat_cfg.get("clr_transform", True):
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
                        logger.debug(f"Collapsed {table_type} to {level} in {duration:.2f}s")
                    except Exception as e:
                        logger.error(f"Taxonomic collapse failed for {table_type}/{level}: {e}")
                        processed[level] = None
                    finally:
                        progress.update(table_task, advance=1)
                        progress.update(ct_task, advance=1)
                    
                self.tables[table_type] = processed
                progress.remove_task(table_task)
    
    def _create_presence_absence(self) -> None:
        if not self.cfg.get("features", {}).get("presence_absence", False):
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
                        logger.debug(f"Created Presence/Absence table for {level} in {duration:.2f}s")
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

        export_tasks = []
        for table_type, levels in self.tables.items():
            tdir = base / table_type
            tdir.mkdir(parents=True, exist_ok=True)
            for level, table in levels.items():
                out = tdir / level / "feature-table.biom"
                out.parent.mkdir(parents=True, exist_ok=True)
                export_tasks.append((table, out))

        with () as executor:
            futures = []
            for table, out in export_tasks:
                futures.append(executor.submit(export_h5py, table, out))

            for future in futures:
                future.result()
