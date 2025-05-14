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
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# ================================== LOCAL IMPORTS =================================== #

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from workflow_16s import ena
from workflow_16s.config import get_config
from workflow_16s.logger import setup_logging 
from workflow_16s.metadata.per_dataset import SubsetDataset
from workflow_16s.sequences.utils import BasicStats, CutAdapt, FastQC, SeqKit
from workflow_16s.utils import dir_utils, file_utils, misc_utils

# ================================ CUSTOM TMP CONFIG ================================= #

import workflow_16s.custom_tmp_config

# ========================== INITIALIZATION & CONFIGURATION ========================== #

# Suppress warnings
warnings.filterwarnings("ignore")

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

# ==================================== FUNCTIONS ===================================== #

def find_required_files(test):
    """Check for required output files in QIIME directories"""
    targets = [
        ("feature-table.biom", "table"),
        ("feature-table.biom", "table_6"),
        ("dna-sequences.fasta", "rep-seqs"),
        ("taxonomy.tsv", "taxonomy"),
        ("sample-metadata.tsv", None)
    ]
    qiime_base = test.get('qiime')
    metadata_base = test.get('metadata')
    found = {}
    
    for fname, subdir in targets:
        if subdir:
            base = qiime_base
            pattern = f"{subdir}/{fname}"
        else:
            base = metadata_base
            pattern = fname
            
        if not base:
            continue
            
        for p in Path(base).rglob(pattern):
            if p.is_file():
                key = f"{subdir}/{fname}" if subdir else fname
                found[key] = p.resolve()
                break
                
    required_keys = [f"{subdir}/{fname}" if subdir else fname 
                    for fname, subdir in targets]
    return found if all(k in found for k in required_keys) else None

# ================================= QIIME EXECUTION ================================== #

def get_conda_env_path(env_name_substring):
    """"""
    try:
        result = subprocess.run(["conda", "env", "list"], capture_output=True, 
                                text=True, check=True)
        for line in result.stdout.splitlines():
            if env_name_substring in line:
                return line.split()[-1]
        raise ValueError(
            f"Conda environment containing '{env_name_substring}' not found."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error finding conda environment: {e}")

def execute_per_dataset_qiime_workflow(
    qiime_dir: Union[str, Path],
    metadata_path: Union[str, Path],
    manifest_path: Union[str, Path],
    subset: Dict[str, Union[str, Path, bool, dict]],
    cfg: Dict[str, Any],
    seq_paths: List[Path],
    logger: logging.Logger,
) -> Dict[str, Path]:
    """
    Execute QIIME2 workflow with comprehensive error handling.
    """
    qiime_env_path = get_conda_env_path("qiime2-amplicon-2024.10")
    qiime_config = cfg["qiime2"]["per_dataset"]
    
    # Get configured script path and check existence
    script_path = Path(qiime_config["script_path"])
    if not script_path.exists():
        logger.warning(f"Script not found at '{script_path}', using default")
        script_path = DEFAULT_PER_DATASET
    
    command = [
        "conda", "run",
        "--prefix", qiime_env_path,
        "python", str(script_path),  
        "--qiime_dir", str(qiime_dir),
        "--metadata_tsv", str(metadata_path),
        "--manifest_tsv", str(manifest_path),
        "--library_layout", str(subset["library_layout"]).lower(),
        "--instrument_platform", str(subset["instrument_platform"]).lower(),
        "--fwd_primer", str(subset["pcr_primer_fwd_seq"]),
        "--rev_primer", str(subset["pcr_primer_rev_seq"]),
        "--classifier_dir", str(qiime_config["taxonomy"]["classifier_dir"]),
        "--classifier", str(qiime_config["taxonomy"]["classifier"]),
        "--classify_method", 
        str(qiime_config["taxonomy"]["classify_method"]).lower(),
        "--retain_threshold", str(qiime_config["filter"]["retain_threshold"]),
        "--chimera_method", str(qiime_config["denoise"]["chimera_method"]),
        "--denoise_algorithm", str(qiime_config["denoise"]["denoise_algorithm"]),
    ]
    
    if qiime_config.get("hard_rerun", False):
        command.append("--hard_rerun")
    if qiime_config.get("trim", {}).get("run", False):
        command.append("--trim_sequences")

    try:
        command_str = str(' '.join(command)).replace(" --", " \\\n--")
        command_str = command_str.replace(" python ", " \\\npython")
        logger.info(
            f"\nExecuting QIIME2 command:\n"
            f"{command_str}"
        )
        result = subprocess.run(
            command, 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
        logger.debug("QIIME STDOUT:\n%s", result.stdout)
        logger.debug("QIIME STDERR:\n%s", result.stderr)
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"QIIME2 execution failed with code {e.returncode}:\n"
            f"Command: {e.cmd}\n"
            f"Error output:\n{e.stderr}"
        )
        logger.error(error_msg)
        raise RuntimeError("QIIME2 workflow failure") from e

    expected_outputs = [
        metadata_path,
        manifest_path,
        qiime_dir / "table" / "feature-table.biom",
        qiime_dir / "rep-seqs" / "dna-sequences.fasta",
        qiime_dir / qiime_config["taxonomy"]["classifier"] / "taxonomy" 
        / "taxonomy.tsv",
        qiime_dir / "table_6" / "feature-table.biom",
    ]
    missing_outputs = file_utils.missing_output_files(expected_outputs)
    if missing_outputs:
        raise RuntimeError(
            f"Missing required QIIME outputs: \n"
            f"{'\n  • '.join(map(str, missing_outputs))}"
        )
    return {
        "metadata": metadata_path,
        "manifest": manifest_path,
        "table": expected_outputs[2],
        "rep_seqs": expected_outputs[3],
        "taxonomy": expected_outputs[4],
        "table_6": expected_outputs[5],
    }

# ================================ SEQUENCE PROCESSING =============================== #

def process_sequences(
    cfg: Dict[str, Any],
    subset_dirs: Dict[str, Path],
    subset: Dict[str, Any],
    info: Any,
    logger: logging.Logger,
) -> Tuple[List[Path], pd.DataFrame]:
    """Handle ENA sequence processing pipeline."""
    run_fastqc = cfg.get("run_fastqc", False)
    run_seqkit = cfg.get("run_seqkit", False)
    run_cutadapt = cfg.get("run_cutadapt", False)

    dataset_type = info.get('dataset_type', '').upper()

    if dataset_type == 'ENA':
        # Initialize ENA sequence fetcher
        fetcher = ena.api.SequenceFetcher(
            fastq_dir=subset_dirs["raw_seqs"], 
            max_workers=cfg.get("max_workers", DEFAULT_MAX_WORKERS_ENA)
        )
    
        # Download sequences and process them if samples are pooled
        if not subset["sample_pooling"]:
            raw_seqs_paths = fetcher.download_run_fastq_concurrent(
                subset["metadata"].set_index("run_accession", drop=False)
            )
        elif subset["sample_pooling"]:
            raw_seqs_paths = fetcher.download_run_fastq_concurrent(
                subset["ena_runs"]
            )
            processor = ena.api.PooledSamplesProcessor(
                metadata_df=subset["metadata"],
                output_dir=Path(subset_dirs["raw_seqs"]) / 'sorted'
            )
            processor.process_all(subset_dirs["raw_seqs"])
            raw_seqs_paths = processor.sample_file_map    
    else:
        raise ValueError(
            f"Dataset type '{dataset_type}' not recognized. "
            f"Expected 'ENA'."
        )
        
    seq_analyzer = BasicStats()
    raw_stats = seq_analyzer.calculate_statistics(raw_seqs_paths)
    raw_df = pd.DataFrame(
        [{"Metric": k, "Raw": v} for k, v in raw_stats["overall"].items()]
    )

    if run_fastqc:
        FastQC(
            fastq_paths=raw_seqs_paths, output_dir=subset_dirs["raw_seqs"]
        ).run_pipeline()

    if run_seqkit:
        raw_stats_seqkit = SeqKit(
            max_workers=DEFAULT_MAX_WORKERS_SEQKIT
        ).analyze_samples(raw_seqs_paths)
        stats = raw_stats_seqkit["overall"]
        report = (
            f"\n=== Summary ===\n"
            f"{'Total Samples'.ljust(DEFAULT_N)}: {stats['total_samples']}\n"
            f"{'Total Files'.ljust(DEFAULT_N)}: {stats['total_files']}\n"
            f"{'Total Sequences'.ljust(DEFAULT_N)}: {stats['total_sequences']:,}\n"
            f"{'Total Bases'.ljust(DEFAULT_N)}: {stats['total_bases']:,}\n\n"
            "=== Length Distribution ===\n"
            f"{'Average Length'.ljust(DEFAULT_N)}: {stats['avg_length']:.2f}\n"
            f"{'Minimum Length'.ljust(DEFAULT_N)}: {stats['min_length']}\n"
            f"{'Maximum Length'.ljust(DEFAULT_N)}: {stats['max_length']}\n\n"
            "=== Most Common Lengths ===\n"
            + "".join(
                f"{rank:>2}. {length:3} bp - {count:>9,} sequences\n"
                for rank, (length, count) in enumerate(
                    stats["most_common_lengths"], start=1
                )
            )
        )
        logger.info(report)
        
    # Set final paths
    processed_paths = raw_seqs_paths
    
    stats_df = pd.DataFrame()
    if run_cutadapt:
        trimmed_seqs_paths, cutadapt_results, cutadapt_proc_time = CutAdapt(
            fastq_dir=subset_dirs["raw_seqs"],
            trimmed_fastq_dir=subset_dirs["trimmed_seqs"],
            primer_fwd=subset["pcr_primer_fwd_seq"],
            primer_rev=subset["pcr_primer_rev_seq"],
            start_trim=cfg["cutadapt"]["start_trim"],
            end_trim=cfg["cutadapt"]["end_trim"],
            start_q_cutoff=cfg["cutadapt"]["start_q_cutoff"],
            end_q_cutoff=cfg["cutadapt"]["end_q_cutoff"],
            min_seq_length=cfg["cutadapt"]["min_seq_length"],
            cores=cfg["cutadapt"]["n_cores"],
            rerun=True,
            region=subset["target_subfragment"],
        ).run(fastq_paths=raw_seqs_paths)
        # Update final paths
        processed_paths = trimmed_seqs_paths
        trimmed_stats = seq_analyzer.calculate_statistics(trimmed_seqs_paths)
        trimmed_df = pd.DataFrame(
            [{"Metric": k, "Trimmed": v} for k, v in trimmed_stats["overall"].items()]
        )
        stats_df = pd.merge(raw_df, trimmed_df, on="Metric")
        stats_df["Percent Change"] = (
            (stats_df["Trimmed"] - stats_df["Raw"]) / stats_df["Raw"]
        ) * 100
        numeric_cols = ["Raw", "Trimmed", "Percent Change"]
        stats_df[numeric_cols] = stats_df[numeric_cols].applymap(
            lambda x: (f"{x:.2f}%" if isinstance(x, float) and x != x
                       else f"{x:.2f}" if x else ""))
        stats_df = stats_df.dropna(axis=1, how="all")

    if run_cutadapt and run_fastqc:
        FastQC(
            fastq_paths=processed_paths, 
            output_dir=subset_dirs["trimmed_seqs"]
        ).run_pipeline()

    if run_cutadapt and run_seqkit:
        SeqKit(
            max_workers=DEFAULT_MAX_WORKERS_SEQKIT
        ).analyze_samples(processed_paths)

    if run_cutadapt:
        return processed_paths, stats_df
        
    else:
        return raw_seqs_paths, pd.DataFrame()


# =================================== MAIN WORKFLOW ================================== #

def main(config_path: Path = DEFAULT_CONFIG) -> None:
    """Orchestrate entire analysis workflow."""    
    try:
        cfg = get_config(config_path)
        per_dataset_hard_rerun = cfg["qiime2"]["per_dataset"].get("hard_rerun", False)
        classifier = cfg["qiime2"]["per_dataset"]["taxonomy"]["classifier"]
        
        project_dir = dir_utils.SubDirs(cfg["project_dir"])
        logger = setup_logging(project_dir.logs)
        
        datasets = file_utils.load_datasets_list(cfg["dataset_list"])
        datasets_info = file_utils.load_datasets_info(cfg["dataset_info"])
        try:
            success_subsets = []
            success_subsets_qiime_outputs = {}
            failed_subsets = []

            # Iterate through datasets
            for dataset in datasets:
                try:
                    # Break the datasets into subsets that need to be processed 
                    # differentially 
                    info = file_utils.fetch_first_match(datasets_info, dataset)
                    dataset_type = info.get('dataset_type', '').upper()
                    
                    subsets = SubsetDataset(cfg)
                    subsets.process(dataset, info)
                    
                    for subset in subsets.success:
                        try:
                            sanitize = lambda s: re.sub(r"[^a-zA-Z0-9-]", "_", s)
                            subset_name = (
                                subset["dataset"] + '.' 
                                + subset["instrument_platform"].lower() + '.' 
                                + subset["library_layout"].lower() + '.' 
                                + subset["target_subfragment"].lower() + '.' 
                                + f"FWD_{sanitize(subset['pcr_primer_fwd_seq'])}_" 
                                + f"REV_{sanitize(subset['pcr_primer_rev_seq'])}"
                            )
                            
                            subset_dirs = project_dir.subset_dirs(subset=subset)

                            metadata_path = subset_dirs["metadata"] / "sample-metadata.tsv"
                            manifest_path = subset_dirs["qiime"] / "manifest.tsv"

                            # Write the sample metadata file
                            file_utils.write_metadata_tsv(subset["metadata"], metadata_path)

                            # If QIIME2 is not in hard rerun mode, check whether 
                            # the necessary outputs for downstream processing already exist.
                            # If they do, we can skip this step.
                            if not per_dataset_hard_rerun:
                                required_paths = {
                                    "metadata": metadata_path,
                                    "manifest": manifest_path,
                                    "table": subset_dirs["qiime"] / "table" 
                                    / "feature-table.biom",
                                    "rep_seqs": subset_dirs["qiime"] / "rep-seqs" 
                                    / "dna-sequences.fasta",
                                    "taxonomy": subset_dirs["qiime"] / classifier 
                                    / "taxonomy" / "taxonomy.tsv",
                                    "table_6": subset_dirs["qiime"] / "table_6" 
                                    / "feature-table.biom",
                                }
                                if all(p.exists() for p in required_paths.values()):
                                    success_subsets_qiime_outputs[subset_name] = required_paths
                                    success_subsets.append(subset_name)
                                    logger.info(
                                        f"⏭️  Skipping processing for "
                                        f"{subset_name.replace('.', '/')} "
                                        f"- existing outputs found"
                                    )
                                    continue
                                    
                            seq_paths, stats = process_sequences(
                                cfg, subset_dirs, subset, info, logger
                            )

                            # Write the manifest file
                            file_utils.write_manifest_tsv(seq_paths, manifest_path)

                            qiime_outputs = execute_per_dataset_qiime_workflow(
                                subset_dirs["qiime"], metadata_path, manifest_path, 
                                subset, cfg, seq_paths, logger
                            )

                            success_subsets_qiime_outputs[subset["dataset"]] = qiime_outputs
                            success_subsets.append(subset["dataset"])

                            def safe_delete(file_path):
                                try:
                                    file_path.unlink(missing_ok=True)
                                except Exception as e:
                                    logger.warning(f"Error deleting {file_path}: {e}")

                            
                            if cfg.get("clean_fastq", True) and dataset_type == 'ENA':
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
                            logger.error(
                                f"❌ Failed processing subset {subset['dataset']}: "
                                f"{str(subset_error)}"
                            )
                            failed_subsets.append((subset["dataset"], str(subset_error)))

                except Exception as dataset_error:
                    logger.error(
                        f"❌ Failed processing dataset {dataset}: {str(dataset_error)}"
                    )
                    failed_subsets.append((dataset, str(dataset_error)))

            logger.info(
                f"📢 Processing complete!\n"
                f"    Success: {len(success_subsets)}\n"
                f"    Failure: {len(failed_subsets)}"
            )
            if failed_subsets:
                failed_subsets_report = '\n'.join(
                    ["ℹ️ Failure details:"] 
                    + [f"    • {dataset}: {error}" 
                       for dataset, error in failed_subsets]
                )
                logger.info(failed_subsets_report)

            metadata_dfs = [file_utils.import_metadata_tsv(i['metadata']) 
                            for i in success_subsets_qiime_outputs.values()]
            metadata_df = pd.concat(metadata_dfs)
            
            # Sort the DataFrame columns alphabetically
            metadata_df = metadata_df.sort_index(axis=1)
            # Calculate the percentage of non-null values for each column
            completeness = metadata_df.notna().mean() * 100
            logger.info(f"\n{completeness}")

            table_dfs = [file_utils.import_features_biom(i['table_6']) 
                         for i in success_subsets_qiime_outputs.values()]
            table_df = pd.concat(table_dfs)
            logger.info(f"Feature table shape: {table_df.shape}")
            #for df in table_dfs: 
            #    logger.info(f"Feature table shape: {df.shape}")

        except Exception as global_error:
            logger.critical(
                f"❌ Fatal pipeline error: {str(global_error)}", 
                exc_info=True
            )
            raise
    except Exception as e:
        print(f"Critical initialization error: {str(e)}")

if __name__ == "__main__":
    main()
