from workflow_16s.constants import MODE, GROUP_COLUMNS
from workflow_16s.downstream.load_data import load_data
from workflow_16s.downstream.prep_data import prep_data

class Config:
    """Configuration container for downstream analysis parameters."""
    
    def __init__(self, config: Dict):
        self.config = config
        
    def is_enabled(self, module: str) -> bool:
        """Check if a specific analysis module is enabled."""
        return self.config.get(module, {}).get('enabled', False)
    
    def get_parameter(self, module: str, parameter: str, default: Any = None) -> Any:
        """Get a specific parameter for an analysis module."""
        return self.config.get(module, {}).get(parameter, default)


class Results:
    """Container for organizing analysis results."""
    
    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.tables: Dict[str, Any] = {}
        self.maps: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.alpha_diversity: Dict[str, Any] = {}
        self.ordination: Dict[str, Any] = {}
        self.top_features: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.analysis_statistics: Dict[str, Any] = {}
      

class DownstreamAnalyzer:
    """Main class for orchestrating 16S amplicon data analysis pipeline."""
    ModeConfig = {
        "asv": ("table", "asv"), 
        "genus": ("table_6", "l6")
    }
    
    def __init__(
        self, 
        config: Dict, 
        project_dir: Any, 
        existing_subsets: Optional[Dict[str, Dict[str, Path]]] = None,
        verbose: bool = False,
        # Result loading parameters
        load_existing_results: bool = True,
        max_result_age_hours: Optional[float] = None,
        force_recalculate_stats: List[str] = None,
        invalidate_results_patterns: List[str] = None
    ):
        # Initialize core attributes
        self.config = Config(config)
        self.verbose = verbose
        self.project_dir = project_dir
        self.output_dir = self.project_dir.final
        self.existing_subsets = existing_subsets

        # Result loading configuration
        self.load_existing_results = load_existing_results
        self.max_result_age_hours = max_result_age_hours
        self.force_recalculate_stats = force_recalculate_stats or []
        self.invalidate_results_patterns = invalidate_results_patterns or []

        # Initialize analysis settings
        self._setup_mode()
        self.group_columns = config.get("group_columns", GROUP_COLUMNS)
        
        # Initialize result containers
        self.results = Results()
        
        # Initialize analysis components
        self.functional_annotation = FunctionalAnnotation(config)
        
        # Execute pipeline
        self._execute_pipeline()
    
    def _setup_mode(self) -> None:
        """Setup analysis mode based on configuration."""
        default_mode = self.config.config.get("target_subfragment_mode", MODE)
        self.mode = 'genus' if default_mode == 'any' else 'asv'
        
        if self.mode not in self.ModeConfig:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {list(self.MODE_CONFIG.keys())}")
    
    def _execute_pipeline(self) -> None:
        """Execute the complete analysis pipeline."""
        logger.info("Starting downstream analysis pipeline...")
        
        try:
            # Data loading and preparation
            self._load_data()
            self._prep_data()
            
            # Run analyses based on configuration
            self._run_enabled_modules()
            
            # Generate summary
            self._log_analysis_summary()
            
            logger.info("Downstream analysis pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    def _load_data(self) -> None:
        logger.info("Loading data...")
        data = load_data(
            config=self.config.config, 
            project_dir=self.project_dir, 
            existing_subsets=self.existing_subsets
        )
        
        self.results.metadata = data.metadata
        self.results.tables = data.tables
        self.nfc_facilities = data.nfc_facilities
      
    def _prep_data(self) -> None:
        logger.info("Prepping data...")
        data = prep_data(
            config=self.config,
            metadata=self.results.metadata,
            tables=self.results.tables,
            project_dir=self.project_dir
        )
      
        self.results.metadata = data.metadata
        self.results.tables = data.tables

    def _run_enabled_modules(self) -> None:
        """Run all enabled analysis modules."""
        analysis_modules = [
            ('maps', self._run_sample_maps),
            ('stats', self._run_statistical_analysis),
            ('alpha_diversity', self._run_alpha_diversity),
            ('ordination', self._run_beta_diversity),
            ('ml', self._run_ml_feature_selection),
            ('top_features', self._run_top_features_analysis),
        ]
        
        for module_name, module_func in analysis_modules:
            if self.config.is_enabled(module_name):
                logger.info(f"Running {module_name} analysis...")
                try:
                    analysis_method()
                except Exception as e:
                    logger.error(f"Error in {module_name} analysis: {e}")
                    if self.verbose:
                        raise
            else:
                logger.info(f"Skipping '{module_name}' analysis: disabled in configuration")
