# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import glob
import logging
import os
import time
import warnings
import threading  # Added for thread locking
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
from workflow_16s.amplicon_data.helpers import _init_dict_level, _ProcessingMixin
from workflow_16s.amplicon_data.maps import Maps
from workflow_16s.amplicon_data.preprocessing import _DataLoader, _TableProcessor
from workflow_16s.amplicon_data.statistical_analyses import run_statistical_tests_for_group, TopFeaturesAnalyzer
from workflow_16s.function.faprotax import (
    faprotax_functions_for_taxon, get_faprotax_parsed
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.utils.nfc_facilities import find_nearby_nfc_facilities

# ========================== INITIALISATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")
warnings.filterwarnings("ignore")

# Global lock for UMAP operations to prevent thread conflicts
umap_lock = threading.Lock()

# ================================= DEFAULT VALUES =================================== #

class AmpliconData:
    """Main class for orchestrating 16S amplicon data analysis pipeline"""
    
    def __init__(
        self, 
        cfg: Dict, 
        project_dir: Any, 
        mode: str = constants.DEFAULT_MODE, 
        existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None,
        verbose: bool = False
    ):
        self.config = cfg
        self.project_dir = project_dir
        self.mode = mode
        self.existing_subsets = existing_subsets
        self.verbose = verbose
        
        # Initialize result containers
        self.tables: Dict[str, Any] = {}
        self.maps: Optional[Dict[str, Any]] = None
        
        self.stats: Dict[str, Any] = {}
        self.top_features = {}
        self.ordination: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.alpha_diversity: Dict[str, Any] = {}
        
        logger.info("Running amplicon data analysis pipeline...")
        self._execute_pipeline()

    def _execute_pipeline(self):
        """Execute the analysis pipeline in sequence."""
        self._load_data()
        self._process_tables()
        self._generate_sample_maps()
        self._run_analysis()
        
        if self.verbose:
            logger.info("AmpliconData analysis finished.")

    def _load_data(self):
        data_loader = _DataLoader(
            config=self.config, 
            mode=self.mode, 
            existing_subsets=self.existing_subsets,
            project_dir=self.project_dir, 
            verbose=self.verbose
        )
        self.meta = data_loader.meta
        self.table = data_loader.table
        self.nfc_facilities = data_loader.nfc_facilities
        self.meta_nfc_facilities = data_loader.meta_nfc_facilities

    def _process_tables(self):
        processor = _TableProcessor(
            config=self.config,
            mode=self.mode,
            meta=self.meta,
            table=self.table,
            project_dir=self.project_dir,
            output_dir=Path(self.project_dir.final),
            verbose=self.verbose
        )
        self.tables = processor.tables

    def _generate_sample_maps(self):
        self.maps = Maps(
            config=self.config, 
            meta=self.meta,
            output_dir=Path(self.project_dir.final),
            verbose=self.verbose
        ).generate_sample_maps(
            nfc_facility_data=self.nfc_facilities if self.nfc_facilities else None
        )

    def _run_analysis(self):
        analyzer = _AnalysisManager(
            config=self.config,
            tables=self.tables,
            meta=self.meta,
            output_dir=Path(self.project_dir.final),
            verbose=self.verbose
        )
        
        # Collect results
        self.stats = analyzer.stats
        self.alpha_diversity = analyzer.alpha_diversity
        self.ordination = analyzer.beta_diversity
        self.models = analyzer.models
        self.top_features = analyzer.top_features


class _AnalysisManager(_ProcessingMixin):
    def __init__(
        self,
        config: Dict,
        tables: Dict[str, Dict[str, Table]],
        meta: pd.DataFrame,
        output_dir: Optional[Path] = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.tables = tables
        self.meta = meta
        self.output_dir = output_dir
        self.verbose = verbose

        self.group_column = self.config.get('group_column', constants.DEFAULT_GROUP_COLUMN)
        self.group_column_values = self.config.get('group_column_values', constants.DEFAULT_GROUP_COLUMN_VALUES)
        
        self.stats: Dict[str, Any] = {}  # Nested dict: group -> 
        self.top_features: Dict[str, Dict[Any, List]] = {}  # Nested dict: group -> condition -> features
        self.alpha_diversity: Dict = {}

        self.run()
        
    def run(self) -> None:
        self._run_statistical_tests()
        self._identify_top_features()  
        if self.config.get('faprotax', False):
            self._annotate_top_features()
        self._run_alpha_diversity()

    # ALPHA DIVERSITY
    def _run_alpha_diversity(self) -> None:
        alpha = AlphaDiversity(
            config=self.config,
            meta=self.meta,
            tables=self.tables
        )
        alpha.run(
            output_dir=self.output_dir
        )
        self.alpha_diversity = alpha.results

    # BETA DIVERSITY
    def _run_beta_diversity(self) -> None:
        logger.info("placeholder")
        beta = Ordination(
            config=self.config,
            meta=self.meta,
            tables=self.tables,
            verbose=self.verbose
        )
        beta.run(
            output_dir=self.output_dir
        )
        self.beta_diversity = beta.results

    # STATISTICS
    def _run_statistical_tests(self) -> None:
        """Run statistical tests for primary and special cases"""
        # Primary group
        self.stats[self.group_column] = run_statistical_tests_for_group(
            config=self.config,  
            tables=self.tables,
            meta=self.meta,
            group_column=self.group_column,
            group_column_values=self.group_column_values,
            output_dir=self.output_dir,
            verbose=self.verbose
        )
        # Special case: NFC facility matching
        if 'facility_match' in self.meta.columns:
            self.stats['facility_match'] = run_statistical_tests_for_group(
                config=self.config,
                tables=self.tables,
                meta=self.meta,
                group_column='facility_match',
                group_column_values=[True, False],  
                output_dir=self.output_dir,
                verbose=self.verbose
            )

    def _identify_top_features(self) -> None:
        """Identify top features for each group condition"""
        # Process primary group
        self._process_group_features(
            group_column=self.group_column,
            group_values=self.group_column_values
        )
        # Special case: NFC facility matching
        if 'facility_match' in self.meta.columns and 'facility_match' in self.stats:
            self._process_group_features(
                group_column='facility_match',
                group_values=[True, False]
            )

    def _process_group_features(self, group_column: str, group_values: List[Any]) -> None:
        """Helper to identify top features for a specific group"""
        if group_column not in self.stats or not self.stats[group_column]:
            logger.warning(
              f"No statistics calculated for group '{group_column}'. Skipping top features."
            )
            return

        # Initialize storage for this group
        self.top_features[group_column] = {}
        # Analyze top features for both conditions in the group
        analyzer = TopFeaturesAnalyzer(self.config, self.verbose) 
        features_cond1, features_cond2 = analyzer.analyze(
            self.stats[group_column], 
            group_column
        )
        
        # Store results
        self.top_features[group_column][group_values[0]] = features_cond1
        self.top_features[group_column][group_values[1]] = features_cond2
        
        logger.info(
            f"Top features for {group_column}: "
            f"{group_values[0]} ({len(features_cond1)}), "
            f"{group_values[1]} ({len(features_cond2)})"
        )

    # FUNCTIONAL ANNOTATION
    def _get_cached_faprotax(
        self, 
        taxon: str
    ) -> List[str]:
        if taxon not in self._faprotax_cache:
            self._faprotax_cache[taxon] = faprotax_functions_for_taxon(
                taxon, 
              self.config.get('faprotax_db_path', constants.DEFAULT_FAPROTAX_DB), 
              include_references=False
            )
        return self._faprotax_cache[taxon]

    def _annotate_top_features(self) -> None:
        all_taxa = set()
        for group_dict in self.top_features.values():
            for features in group_dict.values():
                for feature in features:
                    all_taxa.add(feature["feature"])

        if not all_taxa:
            if self.verbose:
                logger.info("No top features to annotate")
            return

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self._get_cached_faprotax, all_taxa))

        taxon_map = dict(zip(all_taxa, results))

        # Annotate features across all groups and conditions
        for group_dict in self.top_features.values():
            for condition, features in group_dict.items():
                for feature in features:
                    feature["faprotax_functions"] = taxon_map.get(feature["feature"], [])

