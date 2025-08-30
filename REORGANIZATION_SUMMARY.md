# ===================================== REORGANIZATION SUMMARY ====================================== #
"""
This file documents the reorganization of the workflow_16s codebase for improved maintainability.

COMPLETED REORGANIZATIONS:

1. JavaScript Modules (src/workflow_16s/figures/):
   - tables.js (477 lines) -> Split into focused modules:
     - js/table-core.js        - Core table initialization
     - js/table-sorting.js     - Table sorting functionality  
     - js/table-pagination.js  - Pagination controls
     - js/table-resizing.js    - Column resizing
     - js/plotly-utils.js      - Plotly-specific utilities
     - js/section-utils.js     - Section collapse/expand
     - js/main.js              - Main initialization
   - Maintained backward compatibility through consolidated tables.js

2. Python Utility Modules (src/workflow_16s/utils/):
   - file_utils.py (690 lines) -> Split into focused modules:
     - file_ops.py             - Basic file operations
     - dataset_loading.py      - Dataset loading functions
     - biom_utils.py           - BIOM table handling  
     - metadata_utils.py       - Metadata processing
     - sequence_utils.py       - Sequence handling
     - path_utils.py           - Path generation and discovery
     - taxonomy_utils.py       - Taxonomy and FAPROTAX handling
   - Maintained backward compatibility through updated file_utils.py

   - data.py (791 lines) -> Split into focused modules:
     - table_conversion.py     - Table format conversion utilities
     - table_filtering.py      - Table filtering operations
     - table_processing.py     - Table processing operations (normalize, clr)
     - feature_classification.py - Feature ID classification
   - Maintained backward compatibility through updated data.py

BENEFITS:
- Improved code organization and maintainability
- Smaller, focused modules are easier to understand and modify
- Reduced coupling between different functional areas
- Better separation of concerns
- Maintained full backward compatibility
- Cleaner import structure for new code

POTENTIAL FUTURE IMPROVEMENTS:
- io.py (643 lines) contains some duplicated functionality with file_utils.py modules
- df_utils.py (457 lines) could potentially be split further
- Consider consolidating overlapping functionality between io.py and file_utils.py modules

USAGE RECOMMENDATIONS:
For new code, consider importing directly from the specific modules:
    from workflow_16s.utils.biom_utils import import_table_biom
    from workflow_16s.utils.metadata_utils import import_metadata_tsv
    from workflow_16s.utils.table_processing import normalize, clr

This makes dependencies more explicit and improves code organization.
"""