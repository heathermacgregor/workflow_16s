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
