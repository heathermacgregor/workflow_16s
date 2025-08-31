# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import logging
import re
from typing import Dict, Iterable

# ================================== LOCAL IMPORTS =================================== #

from workflow_16s.utils.progress import get_progress_bar, _format_task_desc

# ========================== INITIALIZATION & CONFIGURATION ========================== #

logger = logging.getLogger("workflow_16s")

# ================================= DEFAULT VALUES =================================== #

FEATURE_PATTERNS = {
    "taxonomic": re.compile(
        r'^d__[\w]+(;p__[\w]+)?(;c__[\w]+)?(;o__[\w]+)?'
        r'(;f__[\w]+)?(;g__[\w]+)?(;s__[\w]+)?$'
    ),
    "hashes": re.compile(r'^[a-f0-9]{32}$|^[a-f0-9]{64}$'),
    "raw_sequences": re.compile(r'^[ACGTRYSWKMBDHVN]+$', re.IGNORECASE)
}

# ==================================== FUNCTIONS ===================================== #

def classify_feature_format(
    cols: Iterable[str], 
    verbose: bool = False
) -> Dict[str, int]:
    """Classify feature IDs into taxonomic, hash, sequence, or unknown types.
    
    Uses regex patterns to identify:
    - Taxonomic strings (e.g., 'd__Bacteria;p__Firmicutes')
    - QIIME-style hashes (32/64 character hex strings)
    - IUPAC nucleotide sequences
    - Unknown patterns
    
    Args:
        cols:    Feature IDs to classify.
        verbose: Verbosity flag.
        
    Returns:
        Dictionary with counts for each category.
    """
    counts = {k: 0 for k in FEATURE_PATTERNS}
    counts["unknown"] = 0

    if verbose:
        for col in map(str, cols):
            col = col.strip()
            if not col:
                counts["unknown"] += 1
                continue
                
            matched = False
            for name, pattern in FEATURE_PATTERNS.items():
                if pattern.match(col):
                    counts[name] += 1
                    matched = True
                    break
                    
            if not matched:
                counts["unknown"] += 1
                
        # Print classification summary
        total = sum(counts.values())
        if total > 0:
            dominant = max(counts, key=counts.get)
            confidence = counts[dominant] / total
            logger.info(
                f"Feature classification: {dominant} "
                f"({confidence:.0%} confidence)"
            )
    else: 
        with get_progress_bar() as progress:
            task_desc = "Classifying feature IDs..."
            task = progress.add_task(
                _format_task_desc(task_desc), 
                total=len(list(cols))
            )
            for col in map(str, cols):
                try:
                    col = col.strip()
                    if not col:
                        counts["unknown"] += 1
                        continue
                        
                    matched = False
                    for name, pattern in FEATURE_PATTERNS.items():
                        if pattern.match(col):
                            counts[name] += 1
                            matched = True
                            break
                            
                    if not matched:
                        counts["unknown"] += 1
                        
                except Exception:
                    counts["unknown"] += 1
                finally:
                    progress.update(task, advance=1)
    
    return counts