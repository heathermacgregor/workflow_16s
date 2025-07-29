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
from workflow_16s.amplicon_data.helpers import _init_dict_level, _ProcessingMixin
from workflow_16s.amplicon_data.feature_selection import FeatureSelection
from workflow_16s.amplicon_data.maps import Maps
from workflow_16s.amplicon_data.preprocessing import _DataLoader, _TableProcessor
from workflow_16s.amplicon_data.statistical_analyses import (
    run_statistical_tests_for_group, TopFeaturesAnalyzer
)
from workflow_16s.amplicon_data.top_features import top_features_plots
from workflow_16s.function.faprotax import (
    faprotax_functions_for_taxon, get_faprotax_parsed
)
from workflow_16s.utils.progress import get_progress_bar, _format_task_desc
from workflow_16s.utils.nfc_facilities import find_nearby_nfc_facilities

# ==================================== FUNCTIONS ===================================== #

logger = logging.getLogger("workflow_16s")
# Global lock for UMAP operations to prevent thread conflicts
umap_lock = threading.Lock()

# ================================= DEFAULT VALUES =================================== #

from workflow_16s.amplicon_data.downstream.load import DownstreamDataLoader

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
        self.config = config
        
        self.verbose = verbose
      
        self.project_dir = project_dir
        self.existing_subsets = existing_subsets

        self.mode = 'genus' if self.config.get("target_subfragment_mode", constants.DEFAULT_MODE) == 'any' else 'asv'
        self._validate_mode()
        
        # Initialize result containers
        self.maps: Optional[Dict[str, Any]] = None
        
        self.stats: Dict[str, Any] = {}
        self.top_features = {}
        self.ordination: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.alpha_diversity: Dict[str, Any] = {}

        
        logger.info("Running downstream analysis pipeline...")
        self._execute_pipeline()
      
    def _validate_mode(self) -> None:
        if self.mode not in self.ModeConfig:
            raise ValueError(f"Invalid mode: {self.mode}")
          
    def _execute_pipeline(self):
        """Execute the analysis pipeline in sequence."""
        self.metadata, self.tables, self.nfc_facilities = self._load_data()
        print(self.metadata)
        #self._process_tables()
        #self._run_analysis()
        
        if self.verbose:
            logger.info("AmpliconData analysis finished.")

    def _load_data(self):
        data_loader = DownstreamDataLoader(self.config, self.mode, self.project_dir, self.existing_subsets)
        return data_loader.metadata, data_loader.tables, data_loader.nfc_facilities
    """
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
    """
