# ================================= DEFAULT VALUES =================================== #
DEFAULT_PROGRESS_TEXT_N = 65
DEFAULT_N = 65
DEFAULT_DATASET_COLUMN = "dataset_name"
DEFAULT_GROUP_COLUMN = "nuclear_contamination_status"
DEFAULT_SYMBOL_COL = DEFAULT_GROUP_COLUMN
DEFAULT_GROUP_COLUMN_VALUES = [True, False]
DEFAULT_MODE = 'genus'
DEFAULT_ALPHA_METRICS = [
    'shannon', 'observed_features', 'simpson',
    'pielou_evenness', 'chao1', 'ace', 
    'gini_index', 'goods_coverage', 'heip_evenness', 
    'dominance'       
]
PHYLO_METRICS = ['faith_pd', 'pd_whole_tree']
DEFAULT_GROUP_COLUMNS = [
    {
        'name': "nuclear_contamination_status",
        'type': "bool",
        'values': [True, False]
    },
]
debug_mode = False

# FIGURES
DEFAULT_HEIGHT = 1000
DEFAULT_WIDTH = 1100

DEFAULT_COLOR_COL = 'dataset_name'
DEFAULT_SYMBOL_COL = 'nuclear_contamination_status'

DEFAULT_METRIC = 'braycurtis'

DEFAULT_PROJECTION = 'natural earth'
DEFAULT_LATITUDE_COL = 'latitude_deg'
DEFAULT_LONGITUDE_COL = 'longitude_deg'
DEFAULT_SIZE_MAP = 5
DEFAULT_OPACITY_MAP = 0.3

DEFAULT_FEATURE_TYPE = 'ASV'

DEFAULT_FEATURE_TYPE_ANCOM = 'l6'
DEFAULT_COLOR_COL_ANCOM = 'p'
