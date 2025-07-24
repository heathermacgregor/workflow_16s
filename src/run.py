"""
16S rRNA Analysis Pipeline 
Full Implementation
Comprehensive workflow for microbial community analysis from raw data to processed 
results.
"""
# ===================================== IMPORTS ====================================== #

# Standard Library Imports
import itertools
import logging
import os
import re
import shutil
import subprocess
import sys
import warnings
from collections import Counter
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-Party Imports
import pandas as pd
import numba

os.environ['NUMBA_NUM_THREADS'] = '8'  # Match your n_jobs setting
numba.config.NUMBA_NUM_THREADS = 8

# ================================== LOCAL IMPORTS =================================== #

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from workflow_16s import ena
from workflow_16s.config import get_config
from workflow_16s.figures.html_report import generate_html_report
from workflow_16s.figures.html_report_test import Section
from workflow_16s.logger import setup_logging 
from workflow_16s.metadata.per_dataset import SubsetDataset
from workflow_16s.qiime.workflows.execute_workflow import (
    execute_per_dataset_qiime_workflow as execute_qiime
)
from workflow_16s.sequences.sequence_processing import process_sequences
from workflow_16s.utils import df_utils, dir_utils, file_utils, misc_utils
#from workflow_16s.utils.amplicon_data import AmpliconData
from workflow_16s.amplicon_data.analysis import AmpliconData
from workflow_16s.utils.io import (
    dataset_first_match, import_metadata_tsv, import_table_biom, load_datasets_info, 
    load_datasets_list, safe_delete, write_manifest_tsv, write_metadata_tsv
)

# ================================ CUSTOM TMP CONFIG ================================= #

import workflow_16s.custom_tmp_config

# ========================== INITIALIZATION & CONFIGURATION ========================== #

warnings.filterwarnings("ignore") # Suppress warnings
pd.set_option('display.max_colwidth', None)
pd.set_option('future.no_silent_downcasting', True)

# ================================= DEFAULT VALUES =================================== #

DEFAULT_CONFIG = (
    Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
    / "references"
    / "config.yaml"
)
DEFAULT_PER_DATASET = (
    Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))) 
    / "src" / "workflow_16s" / "qiime" / "workflows" / "per_dataset_run.py"
)
DEFAULT_CLASSIFIER = "silva-138-99-515-806"
DEFAULT_N = 20
DEFAULT_MAX_WORKERS_ENA = 16
DEFAULT_MAX_WORKERS_SEQKIT = 8
ENA_PATTERN = re.compile(r"^PRJ[EDN][A-Z]\d{4,}$", re.IGNORECASE)

# =================================== MAIN WORKFLOW ================================== #

def get_existing_subsets(cfg, logger) -> Dict[str, Dict[str, Path]]:
    """
    Identify existing subsets with required QIIME outputs without running upstream 
    processing.
    
    Args:
        cfg:    Configuration dictionary from the workflow.
        logger: Logger instance for logging messages.
        
    Returns:
        Dictionary mapping subset IDs to dictionaries of file paths.
    """
    project_dir = dir_utils.SubDirs(cfg["project_dir"])
    classifier = cfg["qiime2"]["per_dataset"]["taxonomy"].get(
        "classifier", DEFAULT_CLASSIFIER
    )
    datasets = load_datasets_list(cfg["dataset_list"])
    datasets_info = load_datasets_info(cfg["dataset_info"])
    existing_subsets = {}

    # Define required files and their keys
    required_files = {
        "metadata": "sample-metadata.tsv",
        "table": "table/feature-table.biom",
        "rep_seqs": "rep-seqs/dna-sequences.fasta",
        "taxonomy": f"{classifier}/taxonomy/taxonomy.tsv",
    }
    if cfg["target_subfragment_mode"] == "any":
        required_files["table_6"] = "table_6/feature-table.biom"

    # Process each dataset to get expected subsets
    for dataset in datasets:
        try:
            # Get dataset info
            dataset_info = dataset_first_match(dataset, datasets_info)

            # Generate potential subsets
            subsets = SubsetDataset(cfg)
            subsets.process(dataset, dataset_info)
            
            for subset in subsets.success:
                # Generate consistent subset ID
                sanitize = lambda s: re.sub(r"[^a-zA-Z0-9-]", "_", s)
                subset_id = (
                    subset["dataset"] + '.' 
                    + subset["instrument_platform"].lower() + '.' 
                    + subset["library_layout"].lower() + '.' 
                    + subset["target_subfragment"].lower() + '.' 
                    + f"FWD_{sanitize(subset['pcr_primer_fwd_seq'])}_" 
                    + f"REV_{sanitize(subset['pcr_primer_rev_seq'])}"
                )
                
                # Get directory paths for this subset
                subset_dirs = project_dir.subset_dirs(subset=subset)
                subset_files = {}
                all_files_exist = True
                
                # Check each required file
                for file_key, rel_path in required_files.items():
                    if file_key == "metadata":
                        file_path = subset_dirs["metadata"] / rel_path
                    else:
                        file_path = subset_dirs["qiime"] / rel_path
                    
                    if not file_path.exists():
                        all_files_exist = False
                        break
                    subset_files[file_key] = file_path
                
                if all_files_exist:
                    existing_subsets[subset_id] = subset_files
                    logger.debug(f"Found existing outputs for subset: {subset_id}")
        
        except Exception as e:
            logger.error(f"❌ Error processing dataset {dataset} for existing subsets: {str(e)}")
    
    logger.info(f"Found {len(existing_subsets)} completed subsets with all required outputs")
    return existing_subsets
    
    
def upstream(cfg, logger) -> None:
    """Orchestrate entire analysis workflow."""
    success_subsets, fail_subsets = [], []
    qiime_outputs = {}
    try:
        qiime_hard_rerun = cfg["qiime2"]["per_dataset"].get("hard_rerun", False)
        classifier = cfg["qiime2"]["per_dataset"]["taxonomy"]["classifier"]
        project_dir = dir_utils.SubDirs(cfg["project_dir"])
        datasets = file_utils.load_datasets_list(cfg["dataset_list"])
        datasets_info = file_utils.load_datasets_info(cfg["dataset_info"])
        
        for dataset in datasets:
            try:
                # Partition datasets by processing requirements 
                dataset_info = dataset_first_match(dataset, datasets_info)

                subsets = SubsetDataset(cfg)
                subsets.process(dataset, dataset_info)

                for subset in subsets.success:
                    try:
                        sanitize = lambda s: re.sub(r"[^a-zA-Z0-9-]", "_", s)
                        # Subset identifier:
                        # dataset.instrument_platform.library_layout.target_subfragment.FWD_SEQ_REV_SEQ
                        subset_id = (
                            subset["dataset"] + '.' 
                            + subset["instrument_platform"].lower() + '.' 
                            + subset["library_layout"].lower() + '.' 
                            + subset["target_subfragment"].lower() + '.' 
                            + f"FWD_{sanitize(subset['pcr_primer_fwd_seq'])}_" 
                            + f"REV_{sanitize(subset['pcr_primer_rev_seq'])}"
                        )

                        subset_dirs = project_dir.subset_dirs(subset=subset)

                        # Write the sample metadata TSV file
                        metadata = subset["metadata"]
                        metadata_path = subset_dirs["metadata"] / "sample-metadata.tsv"
                        write_metadata_tsv(metadata, metadata_path)

                        # If hard_rerun is not enabled, skip QIIME if the necessary outputs already exist
                        if not qiime_hard_rerun:
                            required_paths = {
                                "metadata": metadata_path,
                                "manifest": manifest_path,
                                "table": subset_dirs["qiime"] / "table" / "feature-table.biom",
                                "rep_seqs": subset_dirs["qiime"] / "rep-seqs" / "dna-sequences.fasta",
                                "taxonomy": subset_dirs["qiime"] / classifier / "taxonomy" / "taxonomy.tsv",
                                "table_6": subset_dirs["qiime"] / "table_6" / "feature-table.biom",
                            }
                            if all(p.exists() for p in required_paths.values()):
                                qiime_outputs[subset_id] = required_paths
                                success_subsets.append(subset_id)
                                logger.info(
                                    f"⏭️  Skipping processing for "
                                    f"{subset_id.replace('.', '/')} "
                                    f"- existing outputs found"
                                )
                                continue

                        seq_paths, seq_stats = process_sequences(
                            cfg=cfg,
                            subset=subset,
                            subset_dirs=subset_dirs,
                            info=dataset_info,
                        )

                        # Write the manifest TSV file
                        manifest_path = subset_dirs["qiime"] / "manifest.tsv"
                        write_manifest_tsv(seq_paths, manifest_path)

                        qiime_dir = subset_dirs["qiime"]
                        qiime_outputs = execute_qiime(
                            cfg, subset, qiime_dir, metadata_path, manifest_path
                        )

                        qiime_outputs[subset["dataset"]] = qiime_outputs
                        success_subsets.append(subset["dataset"])

                        # Check if clean_fastq is enabled
                        clean_fastq = cfg.get("clean_fastq", {}).get("enabled", True)
                        dataset_type = dataset_info.get('dataset_type', '').upper()
                        if clean_fastq and dataset_type == 'ENA':
                            dir_types = ["raw_seqs", "trimmed_seqs"]
                            for dir_type in dir_types:
                                dir_path = subset_dirs[dir_type]
                                if not dir_path.exists():
                                    continue
                                for fastq_file in dir_path.glob("*.fastq.gz"):
                                    safe_delete(fastq_file)
                            logger.info(
                                f"Cleaned up intermediate files for subset: "
                                f"{subset['dataset']}"
                            )

                    except Exception as subset_error:
                        logger.error(f"❌ Failed processing subset {subset['dataset']}: {str(subset_error)}")
                        fail_subsets.append((subset["dataset"], str(subset_error)))

            except Exception as dataset_error:
                logger.error(f"❌ Failed processing dataset {dataset}: {str(dataset_error)}")
                fail_subsets.append((dataset, str(dataset_error)))

        n_success_subsets = len(success_subsets)
        n_total_subsets = len(success_subsets) + len(fail_subsets)
        logger.info(
            f"📢 Processing complete! Succeeded for {n_success_subsets} of {n_total_subsets} subsets"
        )
        if fail_subsets:
            fail_subsets_report = '\n'.join(["ℹ️ Failure details:"] + [f"    • {dataset}: {error}" 
                                                                      for dataset, error in fail_subsets])
            logger.info(fail_subsets_report)

        metadata_dfs = [import_metadata_tsv(i['metadata']) 
                        for i in qiime_outputs.values()]
        metadata_df = pd.concat(metadata_dfs)
        # Calculate the percentage of non-null values for each column
        completeness = metadata_df.sort_index(axis=1).notna().mean() * 100
        logger.info(f"\n{completeness}")

        table_type = 'table_6' if cfg['target_subfragment_mode'] == 'any' else 'table'
        table_dfs = [import_table_biom(i[table_type]) 
                     for i in qiime_outputs.values()]
        table_df = pd.concat(table_dfs)
        logger.info(f"Feature table shape: {table_df.shape}")
        return success_subsets
        
    except Exception as global_error:
        logger.critical(
            f"❌ Fatal pipeline error: {str(global_error)}", 
            exc_info=True
        )
        raise

def downstream(cfg, logger) -> None:
    project_dir = dir_utils.SubDirs(cfg["project_dir"])
    if not cfg.get("upstream", {}).get("enabled", False):
        existing_subsets = None
        if cfg.get("downstream", {}).get("find_subsets", False):
            existing_subsets = get_existing_subsets(cfg, logger)
            logger.info(f"Found {len(existing_subsets)} completed subsets")
            
    data = AmpliconData(
        cfg=cfg,
        project_dir=project_dir,
        mode='genus' if cfg["target_subfragment_mode"] == 'any' else 'asv',
        existing_subsets=existing_subsets,
        verbose=False        
    )
    def print_dict_structure(d, parent_key):
        for k, v in d.items():
            new_key = f"{parent_key}.{k}"
            if isinstance(v, dict):
                print_dict_structure(v, new_key)
            else:
                logger.info(f"{new_key}: {type(v)}")
    
    # Assuming 'data' is the object to analyze
    for attr_name, attr_value in data.__dict__.items():
        if not attr_name.startswith('__'):
            logger.info(f"{attr_name}: {type(attr_value)}")
            if isinstance(attr_value, dict):
                print_dict_structure(attr_value, attr_name)
        
    report_path = Path(project_dir.final) / "analysis_report.html"
    generate_html_report(
        amplicon_data=data,
        output_path=report_path
    )
    logger.info(f"HTML report generated at: {report_path}")
    

def main(config_path: Path = DEFAULT_CONFIG) -> None:
    """Orchestrate entire analysis workflow."""    
    cfg = get_config(config_path)
    project_dir = dir_utils.SubDirs(cfg["project_dir"])
    logger = setup_logging(project_dir.logs)

    if cfg.get("upstream", {}).get("enabled", False):
        upstream(cfg, logger)
    if cfg.get("downstream", {}).get("enabled", False):
        downstream(cfg, logger)


if __name__ == "__main__":
    main()
