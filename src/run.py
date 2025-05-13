"""
16S rRNA Analysis Pipeline - Full Implementation
Comprehensive workflow for microbial community analysis from raw data to processed results
"""

# ===================================== IMPORTS ====================================== #

# Standard library imports
import os
import sys
import re
import shutil
import subprocess
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import Counter
import itertools
from pprint import pprint

# Third-party imports
import pandas as pd

# Rich progress bar imports
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# Custom module imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from workflow_16s.config import get_config
from workflow_16s.logger import setup_logging
from workflow_16s import ena
from workflow_16s.utils import file_utils, misc_utils, dir_utils
from workflow_16s.metadata.per_dataset import SubsetDataset
from workflow_16s.sequences.utils import CutAdapt, BasicStats, FastQC, SeqKit

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
DEFAULT_MAX_WORKERS_SEQKIT = 8
ENA_PATTERN = re.compile(r"^PRJ[EDN][A-Z]\d{4,}$", re.IGNORECASE)

# Initialize logging
logger = logging.getLogger("workflow_16s")

# Suppress warnings
warnings.filterwarnings("ignore")

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

def execute_per_dataset_qiime_workflow(
    qiime_dir: Union[str, Path],
    metadata_path: Union[str, Path],
    manifest_path: Union[str, Path],
    subset: Dict[str, Union[str, Path, bool, dict]],
    cfg: Dict[str, Any],
    seq_paths: List[Path],
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Execute QIIME2 workflow with comprehensive error handling."""
    qiime_config = cfg["qiime2"]["per_dataset"]
    
    def get_conda_env_path(env_name_substring):
        try:
            result = subprocess.run(
                ["conda", "env", "list"], 
                capture_output=True, 
                text=True, 
                check=True
            )
            for line in result.stdout.splitlines():
                if env_name_substring in line:
                    return line.split()[-1]
            raise ValueError(
                f"Conda environment containing '{env_name_substring}' not found."
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error finding conda environment: {e}")

    qiime_env_path = get_conda_env_path("qiime2-amplicon-2024.10")
    command = [
        "conda", "run", 
        "--prefix", qiime_env_path, 
        "python", str(qiime_config["script_path"]),
        "--qiime_dir", str(qiime_dir),
        "--metadata_tsv", str(metadata_path),
        "--manifest_tsv", str(manifest_path),
        "--library_layout", str(subset["library_layout"]).lower(),
        "--instrument_platform", str(subset["instrument_platform"]).lower(),
        "--fwd_primer", str(subset["pcr_primer_fwd_seq"]),
        "--rev_primer", str(subset["pcr_primer_rev_seq"]),
        "--classifier_dir", str(qiime_config["taxonomy"]["classifier_dir"]),
        "--classifier", str(qiime_config["taxonomy"]["classifier"]),
        "--classify_method", str(qiime_config["taxonomy"]["classify_method"]).lower(),
        "--retain_threshold", str(qiime_config["filter"]["retain_threshold"]),
        "--chimera_method", str(qiime_config["denoise"]["chimera_method"]),
        "--denoise_algorithm", str(qiime_config["denoise"]["denoise_algorithm"]),
    ]
    if qiime_config.get("hard_rerun", False):
        command.append("--hard_rerun")
    if qiime_config.get("trim", {}).get("run", False):
        command.append("--trim_sequences")

    try:
        logger.info(f"\nExecuting QIIME command: {str(' '.join(command)).replace(" --", " \\\n--")}")
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.debug("QIIME STDOUT:\n%s", result.stdout)
        logger.debug("QIIME STDERR:\n%s", result.stderr)
    except subprocess.CalledProcessError as e:
        error_msg = (f"QIIME execution failed with code {e.returncode}:\nCommand: {e.cmd}\nError output:\n{e.stderr}")
        logger.error(error_msg)
        raise RuntimeError("QIIME workflow failure") from e

    expected_outputs = [
        metadata_path,
        manifest_path,
        qiime_dir / "table" / "feature-table.biom",
        qiime_dir / "rep-seqs" / "dna-sequences.fasta",
        qiime_dir / qiime_config["taxonomy"]["classifier"] / "taxonomy" / "taxonomy.tsv",
        qiime_dir / "table_6" / "feature-table.biom",
    ]
    missing_outputs = file_utils.missing_output_files(expected_outputs)
    if missing_outputs:
        raise RuntimeError(f"Missing required QIIME outputs: {', '.join(map(str, missing_outputs))}")
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
    logger: logging.Logger,
) -> Tuple[List[Path], pd.DataFrame]:
    """Handle sequence processing pipeline."""
    fetcher = ena.api.SequenceFetcher(
        fastq_dir=subset_dirs["raw_seqs"], max_workers=cfg.get("max_workers", 16)
    )
    if not subset["sample_pooling"]:
        raw_seqs_paths = fetcher.download_run_fastq_concurrent(
            subset["metadata"].set_index("run_accession", drop=False)
        )
    else:
        raw_seqs_paths = fetcher.download_run_fastq_concurrent(
            subset["ena_runs"]
        )
        processor = ena.api.PooledSamplesProcessor(
            metadata_df=subset["metadata"],
            output_dir=Path(subset_dirs["raw_seqs"]) / 'sorted'
        )
        
        # Run full processing pipeline
        processor.process_all(subset_dirs["raw_seqs"])
        
        # Access the mapping dictionary directly
        raw_seqs_paths = processor.sample_file_map    

    process_seqs = cfg.get("run_cutadapt", False)
    if process_seqs:
        seq_analyzer = BasicStats()
        raw_stats = seq_analyzer.calculate_statistics(raw_seqs_paths)
        raw_df = pd.DataFrame(
            [{"Metric": k, "Raw": v} for k, v in raw_stats["overall"].items()]
        )

    if cfg["run_fastqc"]:
        FastQC(
            fastq_paths=raw_seqs_paths, output_dir=subset_dirs["raw_seqs"]
        ).run_pipeline()

    if cfg["run_seqkit"]:
        raw_stats_seqkit = SeqKit(max_workers=DEFAULT_MAX_WORKERS_SEQKIT).analyze_samples(raw_seqs_paths)
        stats = raw_stats_seqkit["overall"]
        output = (
            f"\n=== Summary ===\n{'Total Samples'.ljust(DEFAULT_N)}: {stats['total_samples']}\n"
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
        logger.info(output)

    processed_paths = raw_seqs_paths
    stats_df = pd.DataFrame()
    if cfg["run_cutadapt"]:
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
        processed_paths = trimmed_seqs_paths
        if process_seqs:
            trimmed_stats = seq_analyzer.calculate_statistics(trimmed_seqs_paths)
            trimmed_df = pd.DataFrame(
                [
                    {"Metric": k, "Trimmed": v}
                    for k, v in trimmed_stats["overall"].items()
                ]
            )
            stats_df = pd.merge(raw_df, trimmed_df, on="Metric")
            stats_df["Percent Change"] = (
                (stats_df["Trimmed"] - stats_df["Raw"]) / stats_df["Raw"]
            ) * 100
            numeric_cols = ["Raw", "Trimmed", "Percent Change"]
            stats_df[numeric_cols] = stats_df[numeric_cols].applymap(
                lambda x: (
                    f"{x:.2f}%"
                    if isinstance(x, float) and x != x
                    else f"{x:.2f}" if x else ""
                )
            )
            stats_df = stats_df.dropna(axis=1, how="all")

    if cfg["run_fastqc"] and cfg["run_cutadapt"]:
        FastQC(
            fastq_paths=processed_paths, output_dir=subset_dirs["trimmed_seqs"]
        ).run_pipeline()

    if cfg["run_seqkit"] and cfg["run_cutadapt"]:
        SeqKit(max_workers=DEFAULT_MAX_WORKERS_SEQKIT).analyze_samples(processed_paths)

    if process_seqs:
        return processed_paths, stats_df
    else:
        return raw_seqs_paths, pd.DataFrame()


# =================================== MAIN WORKFLOW ================================== #

def main(config_path: Path = DEFAULT_CONFIG) -> None:
    """Orchestrate entire analysis workflow."""
    try:
        cfg = get_config(config_path)
        project_dir = dir_utils.SubDirs(cfg["project_dir"])
        logger = setup_logging(project_dir.logs)
        datasets = file_utils.load_datasets_list(cfg["dataset_list"])
        datasets_info = file_utils.load_datasets_info(cfg["dataset_info"])
        try:
            success_subsets = []
            success_qiime_outputs = {}
            failed_subsets = []

            # Iterate through datasets
            for dataset in datasets:
                try:
                    # Break the datasets into subsets that need to be processed 
                    # differentially 
                    subsets = SubsetDataset(cfg)
                    subsets.process(
                        dataset, 
                        file_utils.fetch_first_match(datasets_info, dataset)
                    )
                    
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
                            # the necessary outputs for downstream processing already exist
                            if not cfg["qiime2"].get("hard_rerun", False):
                                classifier = cfg["qiime2"]["per_dataset"]["taxonomy"]["classifier"]
                                required_paths = {
                                    "metadata": metadata_path,
                                    "manifest": manifest_path,
                                    "table": subset_dirs["qiime"] / "table" / "feature-table.biom",
                                    "rep_seqs": subset_dirs["qiime"] / "rep-seqs" / "dna-sequences.fasta",
                                    "taxonomy": subset_dirs["qiime"] / classifier / "taxonomy" / "taxonomy.tsv",
                                    "table_6": subset_dirs["qiime"] / "table_6" / "feature-table.biom",
                                }
                                if all(p.exists() for p in required_paths.values()):
                                    success_qiime_outputs[subset_name] = required_paths
                                    success_subsets.append(subset_name)
                                    logger.info(f"⏭️ Skipping processing for {subset_name.replace('.', '/')} - existing outputs found")
                                    continue
                                    
                            seq_paths, stats = process_sequences(cfg, subset_dirs, subset, logger)

                            # Write the manifest file
                            file_utils.write_manifest_tsv(seq_paths, manifest_path)

                            qiime_outputs = execute_per_dataset_qiime_workflow(
                                subset_dirs["qiime"], metadata_path, manifest_path, subset, cfg, seq_paths, logger
                            )

                            success_qiime_outputs[subset["dataset"]] = qiime_outputs
                            success_subsets.append(subset["dataset"])

                            if cfg.get("clean_fastq", True):
                                for dir_type in ["raw_seqs", "trimmed_seqs"]:
                                    dir_path = subset_dirs[dir_type]
                                    if dir_path.exists():
                                        for fq in dir_path.glob("*.fastq.gz"):
                                            try: fq.unlink(missing_ok=True)
                                            except Exception as e: logger.warning(f"Error deleting {fq}: {str(e)}")
                                logger.info(f"Cleaned up intermediate files for {subset['dataset']}")

                        except Exception as subset_error:
                            logger.error(f"❌ Failed processing subset {subset['dataset']}: {str(subset_error)}")
                            failed_subsets.append((subset["dataset"], str(subset_error)))

                except Exception as dataset_error:
                    logger.error(f"❌ Failed processing dataset {dataset}: {str(dataset_error)}")
                    failed_subsets.append((dataset, str(dataset_error)))

            logger.info(
                f"📢 Processing complete!\n"
                f"    Success:  {len(success_subsets)}"
                f"    Failures: {len(failed_subsets)}"
            )
            if failed_subsets:
                logger.info("ℹ️ Failure details:")
                for dataset, error in failed_subsets: logger.info(f"- {dataset}: {error}")

            metadata_dfs = [file_utils.import_metadata_tsv(i['metadata']) for i in success_qiime_outputs.values()]
            metadata_df = pd.concat(metadata_dfs)
            # Sort the DataFrame columns alphabetically
            metadata_df = metadata_df.sort_index(axis=1)
            
            # Calculate the percentage of non-null values for each column
            completeness = metadata_df.notna().mean() * 100
            
            logger.info(f"\n{completeness}")
            #for col in metadata_df.columns:
            #    logger.info(f"{col}: {metadata_df[col].value_counts()}")

            table_dfs = [file_utils.import_features_biom(i['table_6']) for i in success_qiime_outputs.values()]
            table_df = pd.concat(table_dfs)
            logger.info(f"Feature table shape: {table_df.shape}")
            #for df in table_dfs: 
            #    logger.info(f"Feature table shape: {df.shape}")

        except Exception as global_error:
            logger.critical(f"❌ Fatal pipeline error: {str(global_error)}", exc_info=True)
            raise
    except Exception as e:
        print(f"Critical initialization error: {str(e)}")

if __name__ == "__main__":
    main()
