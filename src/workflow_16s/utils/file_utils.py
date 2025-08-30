# ===================================== IMPORTS ====================================== #
"""
Reorganized file utilities module providing backward compatibility.

This module has been split into focused sub-modules for better maintainability:
- file_ops.py:       Basic file operations (safe_delete, missing_output_files)
- dataset_loading.py: Dataset loading functions (load_datasets_list, load_datasets_info, fetch_first_match)
- biom_utils.py:      BIOM table handling (import_table_biom, import_merged_table_biom, filter_and_reorder_biom_and_metadata)
- metadata_utils.py:  Metadata handling (import_metadata_tsv, import_merged_metadata_tsv, write_metadata_tsv)
- sequence_utils.py:  Sequence handling (import_seqs_fasta)
- path_utils.py:      Path generation and file discovery (processed_dataset_files, find_required_qiime_output_files)
- taxonomy_utils.py:  Taxonomy handling (import_faprotax_tsv, Taxonomy class)

For new code, consider importing directly from the specific modules.
"""

# Standard Library Imports
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd
from biom.table import Table

# ================================== LOCAL IMPORTS =================================== #

# Import all functions from the new modules for backward compatibility
from workflow_16s.utils.file_ops import (
    safe_delete,
    missing_output_files
)

from workflow_16s.utils.dataset_loading import (
    load_datasets_list,
    load_datasets_info,
    fetch_first_match
)

from workflow_16s.utils.biom_utils import (
    import_table_biom,
    import_merged_table_biom,
    filter_and_reorder_biom_and_metadata,
    _normalize_metadata,
    _create_biom_id_mapping
)

from workflow_16s.utils.metadata_utils import (
    import_metadata_tsv,
    import_merged_metadata_tsv,
    write_metadata_tsv,
    manual_meta,
    write_manifest_tsv
)

from workflow_16s.utils.sequence_utils import (
    import_seqs_fasta
)

from workflow_16s.utils.path_utils import (
    processed_dataset_files,
    find_required_qiime_output_files
)

from workflow_16s.utils.taxonomy_utils import (
    import_faprotax_tsv,
    Taxonomy
)

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger('workflow_16s')

# ================================= DEFAULT VALUES =================================== #

DEFAULT_PROGRESS_TEXT_N: int = 65 
DEFAULT_GROUP_COLUMN: str = 'nuclear_contamination_status'  
DEFAULT_GROUP_COLUMN_VALUES: List[bool] = [True, False]  

# ANSI color codes for terminal output
RED: str = "\033[91m"
GREEN: str = "\033[92m"
YELLOW: str = "\033[93m"
RESET: str = "\033[0m"

# ==================================== BACKWARD COMPATIBILITY =================================== #

# All the functions are now imported from their respective modules above.
# This ensures backward compatibility while providing a cleaner modular structure.

# The original monolithic file_utils.py contained the following sections that have been split:
# 1. File Operations -> file_ops.py
# 2. Dataset Loading -> dataset_loading.py  
# 3. File Path Handling -> path_utils.py
# 4. Metadata Handling -> metadata_utils.py
# 5. BIOM Table Handling -> biom_utils.py
# 6. BIOM-Metadata Alignment -> biom_utils.py
# 7. Sequence Handling -> sequence_utils.py
# 8. FAPROTAX and Taxonomy Handling -> taxonomy_utils.py

# ==================================== USAGE RECOMMENDATIONS =================================== #

"""
For new code, consider importing directly from the specific modules for better clarity:

    from workflow_16s.utils.biom_utils import import_table_biom
    from workflow_16s.utils.metadata_utils import import_metadata_tsv
    from workflow_16s.utils.file_ops import safe_delete

This makes dependencies more explicit and improves code organization.
"""