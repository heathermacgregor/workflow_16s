# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import time
import threading  
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third‑Party Imports
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s import constants
from workflow_16s.amplicon_data.alpha_diversity import AlphaDiversity
from workflow_16s.amplicon_data.beta_diversity import Ordination
from workflow_16s.amplicon_data.feature_selection import FeatureSelection
from workflow_16s.amplicon_data.maps import Maps
from workflow_16s.amplicon_data.statistical_analyses import (
    run_statistical_tests_for_group, TopFeaturesAnalyzer
)
from workflow_16s.amplicon_data.top_features import top_features_plots
from workflow_16s.function.faprotax import (
    faprotax_functions_for_taxon, get_faprotax_parsed
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.amplicon_data.downstream.input import DownstreamDataLoader as InputData
from workflow_16s.amplicon_data.downstream.tables import PrepData 
from workflow_16s.amplicon_data.downstream.stats import StatisticalAnalysis

# ==================================== FUNCTIONS ===================================== #

logger = logging.getLogger("workflow_16s")
# Global lock for UMAP operations to prevent thread conflicts
umap_lock = threading.Lock()

# ================================= DEFAULT VALUES =================================== #

class FunctionalAnnotation:
    def __init__(
        self,
        config: Dict
    ):
        if self.config.get("faprotax", {}).get('enabled', False):
            self.db = get_faprotax_parsed()
        self._faprotax_cache: Dict[str, Any] = {}

    def _get_cached_faprotax(self, taxon: str) -> List[str]:
        if taxon not in self._faprotax_cache:
            self._faprotax_cache[taxon] = faprotax_functions_for_taxon(taxon, db, include_references=False)
    
    def _annotate_features(self, features):
        features = list(features)
        # Initialize results array
        results = [None] * len(features)

        with ThreadPoolExecutor() as executor:
            future_to_idx = {executor.submit(self._get_cached_faprotax, taxon): idx for idx, taxon in enumerate(features)}
            with get_progress_bar() as progress:
                task = progress.add_task(description=_format_task_desc("Annotating most important features"), total=len(features)
                for future in as_completed(future_to_idx):
                    idx = future_to_idx(future)
                    results[idx] = future.result()
                    progress.update(task, advance=1)
        
        # Create taxon map
        taxon_map = dict(zip(features, results))
        
        # Annotate features across all groups and conditions
        for feature in features:
            feature["faprotax_function"] = taxon_map.get(feature["feature"], [])
        return result

class Downstream:
    """Main class for orchestrating 16S amplicon data analysis pipeline."""
    ModeConfig = {
        "asv": ("table", "asv"), 
        "genus": ("table_6", "l6")
    }
    def __init__(
        self, 
        config: Dict, 
        project_dir: Any, 
        mode: str = constants.DEFAULT_MODE, 
        existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None,
        verbose: bool = False
    ):
        self.config, self.project_dir, self.verbose = config, project_dir, verbose
        self.existing_subsets = existing_subsets
        self.mode = 'genus' if self.config.get("target_subfragment_mode", constants.DEFAULT_MODE) == 'any' else 'asv'
        self._validate_mode()

        self.group_columns = self.config.get("group_columns", [])
        
        # Initialize result containers
        self.metadata: Dict[str, Any] = {}
        self.tables: Dict[str, Any] = {}
        self.maps: Optional[Dict[str, Any]] = None
        self.stats: Optional[Dict[str, Any]] = None
        self.alpha_diversity: Optional[Dict[str, Any]] = None
        self.beta_diversity: Optional[Dict[str, Any]] = None
        self.most_important_features: Optional[Dict[str, Any]] = None
        self.catboost_models: Optional[Dict[str, Any]] = None
        
        logger.info("Running downstream analysis pipeline...")
        self._execute_pipeline()
      
    def _validate_mode(self) -> None:
        if self.mode not in self.ModeConfig:
            raise ValueError(f"Invalid mode: {self.mode}")
          
    def _execute_pipeline(self):
        """Execute the analysis pipeline in sequence."""
        self.metadata, self.tables, self.nfc_facilities = self._load_data()
        self.metadata, self.tables = self._prep_data()
        self._run_analysis()
        
        if self.verbose:
            logger.info("AmpliconData analysis finished.")

    def _load_data(self):
        data = InputData(self.config, self.mode, self.project_dir, self.existing_subsets)
        return data.metadata, data.tables, data.nfc_facilities
    
    def _prep_data(self):
        data = PrepData(self.config, self.tables, self.metadata, self.mode, self.project_dir)
        return data.metadata, data.tables

    def _run_analysis(self):
        self.maps = self._plot_sample_maps()
        self.stats = self._stats()
        self.alpha_diversity = self._alpha_diversity()
        self.beta_diversity = self._beta_diversity()
        self.catboost_models = self._catboost_feature_selection()

    def _plot_sample_maps(self):
        if not self.config.get("maps", {}).get('enabled', False):
            return
        maps = Maps(self.config, self.metadata, Path(self.output_dir) / 'sample_maps', self.verbose)
        maps.generate_sample_maps(nfc_facility_data=self.nfc_facilities)s
        return maps.maps

    def _stats(self):
        if not self.config.get("stats", {}).get('enabled', False):
            return

        stats = StatisticalAnalysis(
            config=self.config,
            tables=self.tables,
            metadata=self.metadata,
            mode=self.mode,
            group_columns=self.group_columns,
            project_dir=self.project_dir
        )
        return stats.results

    def _alpha_diversity(self):
        if not self.config.get("alpha_diversity", {}).get('enabled', False):
            return
        alpha = AlphaDiversity(self.config, self.metadata, self.tables)
        alpha.run(output_dir=self.output_dir)
        return alpha.results

    def _beta_diversity(self):
        if not self.config.get("beta_diversity", {}).get('enabled', False):
            return
        beta = Ordination(self.config, self.metadata, self.tables, self.verbose)
        beta.run(output_dir=self.output_dir)
        return beta.results

    def _catboost_feature_selection(self):
        if not self.config.get("machine_learning", {}).get('enabled', False):
            return
        cb = FeatureSelection(self.config, self.metadata, self.tables, self.verbose)
        cb.run(output_dir=self.output_dir)
        return cb.moodels

    def _top_features(self):
        placeholder = ''

        if self.config.get("violin_plots", {}).get('enabled', False) or self.config.get("feature_maps", {}).get('enabled', False):
            features_plots = top_features_plots(
                output_dir=self.output_dir, 
                config=self.config, 
                top_features=self.top_features, 
                tables=self.tables, 
                meta=self.metadata, 
                nfc_facilities=self.nfc_facilities, 
                verbose=self.verbose
            )

    def _functional_annotation(self):
        placeholder = ''
        faprotax = FunctionalAnnotation(self.config)
        annotations = faprotax._annotate_features()
