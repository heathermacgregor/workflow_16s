"""
16S rRNA Analysis Pipeline - Full Implementation
Comprehensive workflow for microbial community analysis from raw data to processed results
"""

# ===================================== IMPORTS ====================================== #

# Standard library imports

import os, sys
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

# Custom module imports

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from workflow_16s.config import get_config
from workflow_16s.logger import setup_logging
from workflow_16s import ena
from workflow_16s.utils import file_utils, misc_utils, dir_utils
from workflow_16s.metadata.per_dataset import SubsetDataset
from workflow_16s.sequences.utils import CutAdapt, BasicStats, FastQC, SeqKit

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


# ================================= GLOBAL VARIABLES ================================= #

DEFAULT_CONFIG = (
    Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))) 
    / "references" 
    / "config.yaml"
)
DEFAULT_PER_DATASET = (
    Path(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))) 
    / "src" 
    / "qiime"
    / "workflows"
    / "per_dataset_run.py"
)
DEFAULT_CLASSIFIER = "silva-138-99-515-806"
ENA_PATTERN = re.compile(r"^PRJ[EDN][A-Z]\d{4,}$", re.IGNORECASE)

# Initialize logging

logger = logging.getLogger("workflow_16s")
warnings.filterwarnings("ignore")


from pathlib import Path

def find_required_files(test):
    """
    Expects test to be a dict with keys:
      - 'qiime': Path to Qiime2 output directory
      - 'metadata': Path to metadata directory

    Returns a dict mapping filenames to Path, or None if any are missing.
    """
    targets = {
        "feature-table.biom": "table",
        "dna-sequences.fasta":  "rep-seqs",
        "taxonomy.tsv":        "taxonomy",
        "sample-metadata.tsv": None      # no subdir → look under metadata
    }

    qiime_base    = test.get('qiime')
    metadata_base = test.get('metadata')
    found = {}

    for fname, subdir in targets.items():
        # choose where to search and how to pattern-match
        if subdir:
            base    = qiime_base
            pattern = f"{subdir}/{fname}"
        else:
            base    = metadata_base
            pattern = fname

        if not base:
            continue

        # rglob is recursive; pattern with no leading wildcards still matches anywhere under base
        for p in Path(base).rglob(pattern):
            if p.is_file():
                found[fname] = p.resolve()
                break

    # make sure we found *all* targets
    if set(found) == set(targets):
        return found

    # else: report None (or you could raise an exception / log missing keys)
    return None



# ================================= QIIME EXECUTION ================================== #


def execute_per_dataset_qiime_workflow(
    subset_dirs: Dict[str, Path],
    subset: Dict[str, Union[str, Path, bool, dict]],
    cfg: Dict[str, Any],
    seq_paths: List[Path],
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Execute QIIME2 workflow with comprehensive error handling and output validation.

    Args:
        dataset_dirs: Directory paths for QIIME processing
        params: Processing parameters dictionary
        cfg: Main configuration dictionary
        seq_paths: Paths to input sequence files
        logger: Configured logger instance

    Returns:
        Dictionary of key output file paths

    Raises:
        RuntimeError: If any critical workflow step fails
    """
    # Create directory structure with sanitized primer sequences

    sanitize = lambda s: re.sub(r"[^a-zA-Z0-9-]", "_", s)
    qiime_dir = (
        subset_dirs["qiime"]
        / subset["instrument_platform"].lower()
        / subset["library_layout"].lower()
        / subset["target_subfragment"].lower()
        / f"FWD_{sanitize(subset['pcr_primer_fwd_seq'])}_REV_{sanitize(subset['pcr_primer_rev_seq'])}"
    )
    qiime_dir.mkdir(parents=True, exist_ok=True)

    metadata_dir = (
        subset_dirs["metadata"]
        / subset["instrument_platform"].lower()
        / subset["library_layout"].lower()
        / subset["target_subfragment"].lower()
        / f"FWD_{sanitize(subset['pcr_primer_fwd_seq'])}_REV_{sanitize(subset['pcr_primer_rev_seq'])}"
    )
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Generate input files

    metadata_path = metadata_dir / "sample-metadata.tsv"
    manifest_path = qiime_dir / "manifest.tsv"
    file_utils.write_metadata_tsv(subset["metadata"], metadata_path)
    file_utils.write_manifest_tsv(seq_paths, manifest_path)

    # Build QIIME command components

    qiime_config = cfg["QIIME 2"]["Per-Dataset"]
    command = [
        "conda",
        "run",
        "--prefix",
        str(cfg["QIIME 2"]["Conda Environment"]),
        "python",
        str(DEFAULT_PER_DATASET),#str(qiime_config["path"]),
        "--qiime_dir",
        str(qiime_dir),
        "--metadata_tsv",
        str(metadata_path),
        "--manifest_tsv",
        str(manifest_path),
        "--library_layout",
        str(subset["library_layout"].lower()),
        "--instrument_platform",
        str(subset["instrument_platform"].lower()),
        "--fwd_primer",
        str(subset["pcr_primer_fwd_seq"]),
        "--rev_primer",
        str(subset["pcr_primer_rev_seq"]),
        "--classifier_dir",
        str(qiime_config["Taxonomic Classification"]["classifier_dir"]),
        "--classifier",
        str(qiime_config["Taxonomic Classification"]["classifier"]),
        "--classify_method",
        str(qiime_config["Taxonomic Classification"]["classify_method"].lower()),
        "--retain_threshold",
        str(qiime_config["Filter"]["retain_threshold"]),
        "--chimera_method",
        str(qiime_config["Denoise Sequences"]["chimera_method"]),
        "--denoise_algorithm",
        str(qiime_config["Denoise Sequences"]["denoise_algorithm"]),
    ]

    # Add conditional flags

    if qiime_config.get("hard_rerun", False):
        command.append("--hard_rerun")
    if qiime_config.get("Trim Sequences", {}).get("run", False):
        command.append("--trim_sequences")
    # Execute command with enhanced logging

    try:
        logger.info(f"Executing QIIME command: {' '.join(command)}")
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.debug("QIIME STDOUT:\n%s", result.stdout)
        logger.debug("QIIME STDERR:\n%s", result.stderr)
    except subprocess.CalledProcessError as e:
        error_msg = (
            f"QIIME execution failed with code {e.returncode}:\n"
            f"Command: {e.cmd}\n"
            f"Error output:\n{e.stderr}"
        )
        logger.error(error_msg)
        raise RuntimeError("QIIME workflow failure") from e
    
    # Validate output files
    
    expected_outputs = [
        metadata_path,
        manifest_path,
        qiime_dir / "table" / "feature-table.biom",
        qiime_dir / "rep-seqs" / "dna-sequences.fasta",
        qiime_dir
        / qiime_config["Taxonomic Classification"]["classifier"]
        / "taxonomy"
        / "taxonomy.tsv",
    ]

    missing_outputs = file_utils.missing_output_files(expected_outputs)
    if missing_outputs:
        raise RuntimeError(
            f"Missing required QIIME outputs: {', '.join(map(str, missing_outputs))}"
        )
    return {
        "metadata": metadata_path,
        "manifest": manifest_path,
        "table": expected_outputs[2],
        "rep_seqs": expected_outputs[3],
        "taxonomy": expected_outputs[4],
    }


# ================================ SEQUENCE PROCESSING =============================== #


def process_sequences(
    cfg: Dict[str, Any],
    subset_dirs: Dict[str, Path],
    subset: Dict[str, Any],
    logger: logging.Logger,
) -> Tuple[List[Path], pd.DataFrame]:
    """Handle sequence processing pipeline including download and quality control.

    Returns:
        Tuple containing paths to processed sequences and statistics DataFrame
    """
    # Download raw sequences

    fetcher = ena.api.SequenceFetcher(
        fastq_dir=subset_dirs["raw_seqs"], max_workers=cfg.get("max_workers", 16)
    )
    raw_seqs_paths = fetcher.download_run_fastq_concurrent(
        subset["metadata"].set_index("run_accession", drop=False)
    )

    # Initial quality assessment

    seq_analyzer = BasicStats()
    raw_stats = seq_analyzer.calculate_statistics(raw_seqs_paths)
    raw_df = pd.DataFrame(
        [{"Metric": k, "Raw": v} for k, v in raw_stats["overall"].items()]
    )

    if cfg["FastQC"]["run"]:
        FastQC(
            fastq_paths=raw_seqs_paths,
            output_dir=subset_dirs["raw_seqs"],
            fastqc_path="fastqc"#cfg["FastQC"]["path"],
        ).run_pipeline()
    if cfg["SeqKit"]["run"]:
        raw_stats_seqkit = SeqKit(max_workers=8).analyze_samples(raw_seqs_paths)
        pprint(raw_stats_seqkit["overall"])
    # Sequence trimming with Cutadapt

    if cfg["Cutadapt"]["run"]:
        trimmed_seqs_paths, cutadapt_results, cutadapt_proc_time = CutAdaptPipeline(
            fastq_dir=subset_dirs["raw_seqs"],
            trimmed_fastq_dir=subset_dirs["trimmed_seqs"],
            primer_fwd=subset["pcr_primer_fwd_seq"],
            primer_rev=subset["pcr_primer_rev_seq"],
            start_trim=cfg["Cutadapt"]["start_trim"],
            end_trim=cfg["Cutadapt"]["end_trim"],
            start_q_cutoff=cfg["Cutadapt"]["start_q_cutoff"],
            end_q_cutoff=cfg["Cutadapt"]["end_q_cutoff"],
            min_seq_length=cfg["Cutadapt"]["min_seq_length"],
            cores=cfg["Cutadapt"]["cores"],
            rerun=True,
            region=subset["target_subfragment"],
        ).run(fastq_paths=raw_seqs_paths)
        print(cutadapt_proc_time)

        # Post-trimming analysis

        trimmed_stats = seq_analyzer.calculate_statistics(trimmed_seqs_paths)
        trimmed_df = pd.DataFrame(
            [{"Metric": k, "Trimmed": v} for k, v in trimmed_stats["overall"].items()]
        )
        processed_paths = trimmed_seqs_paths

        # Generate comparative statistics

        stats_df = pd.merge(raw_df, trimmed_df, on="Metric")
        stats_df["Percent Change"] = (
            (stats_df["Trimmed"] - stats_df["Raw"]) / stats_df["Raw"]
        ) * 100
    else:
        processed_paths = raw_seqs_paths
        stats_df = raw_df
        stats_df["Trimmed"] = None
        stats_df["Percent Change"] = None
    # Formatting for reporting

    numeric_cols = ["Raw", "Trimmed", "Percent Change"]
    stats_df[numeric_cols] = stats_df[numeric_cols].applymap(
        lambda x: (
            f"{x:.2f}%" if isinstance(x, float) and x != x else f"{x:.2f}" if x else ""
        )
    )

    logger.info(
        f"{subset['dataset']} Statistics\n"
        f"Platform: {subset['instrument_platform']}\n"
        f"Layout: {subset['library_layout']}\n"
        f"{stats_df.to_string(index=False)}"
    )

    if cfg["FastQC"]["run"] and cfg["Cutadapt"]["run"]:
        FastQC(
            fastq_paths=processed_paths,
            output_dir=subset_dirs["trimmed_seqs"],
            fastqc_path="fastqc",#cfg["FastQC"]["path"],
        ).run_pipeline()
    if cfg["SeqKit"]["run"] and cfg["Cutadapt"]["run"]:
        raw_stats_seqkit = SeqKit(max_workers=8).analyze_samples(processed_paths)
        pprint(raw_stats_seqkit["overall"])
    return processed_paths, stats_df


# =================================== MAIN WORKFLOW ================================== #


def main(config_path: Path = DEFAULT_CONFIG) -> None:
    """Orchestrate entire analysis workflow with enhanced error handling."""
    try:
        # Extract information from the config YAML file
        cfg = get_config(config_path)
        
        # Set up directory structure for the designated project
        project_dir = dir_utils.SubDirs(cfg["Project Directory"])
        
        # Set up logging
        logger = setup_logging(project_dir.logs)

        # Load the list of datasets to process
        datasets = file_utils.load_datasets_list(cfg["Dataset List"])
        
        # Load the dataframe of dataset information
        datasets_info = file_utils.load_datasets_info(cfg["Dataset Information"])
    
        try:
            # Iterate through datasets
            success_subsets = []
            success_qiime_outputs = {}
            failed_subsets = []

            for dataset in datasets:
                try:
                    subsets = SubsetDataset(cfg)
                    subsets.process(
                        dataset, file_utils.fetch_first_match(datasets_info, dataset)
                    )

                    for subset in subsets.success:
                        try:
                            subset_dirs = project_dir.dataset_dirs(
                                dataset=subset["dataset"]
                            )

                            seq_paths, stats = process_sequences(
                                cfg=cfg,
                                subset_dirs=subset_dirs,
                                subset=subset,
                                logger=logger,
                            )

                            qiime_outputs = execute_per_dataset_qiime_workflow(
                                subset_dirs=subset_dirs,
                                subset=subset,
                                cfg=cfg,
                                seq_paths=seq_paths,
                                logger=logger,
                            )

                            success_qiime_outputs[subset["datasets"]] = qiime_outputs
                            success_subsets.append(subset["dataset"])

                            # Cleanup intermediate files if configured
                            if cfg.get("Clean Up FASTQ", True):
                                for dir_type in ["raw_seqs", "trimmed_seqs"]:
                                    dir_path = subset_dirs[dir_type]
                                    if dir_path.exists():
                                        for fq in dir_path.glob("*.fastq.gz"):
                                            try:
                                                fq.unlink(missing_ok=True)
                                            except Exception as e:
                                                logger.warning(
                                                    f"Error deleting {fq}: {str(e)}"
                                                )
                                logger.info(
                                    f"Cleaned up intermediate files for {subset['dataset']}"
                                )

                        except Exception as subset_error:
                            logger.error(
                                f"Failed processing subset {subset['dataset']}: {str(subset_error)}"
                            )

                            failed_subsets.append((subset["dataset"], str(subset_error)))

                except Exception as dataset_error:
                    logger.error(
                        f"Failed processing dataset {dataset}: {str(dataset_error)}"
                    )

                    failed_subsets.append((dataset, str(dataset_error)))

            # Final report
            logger.info(
                f"Processing complete. Success: {len(success_subsets)}, Failures: {len(failed_subsets)}"
            )

            print(success_qiime_subsets)

            if failed_subsets:
                logger.info("Failure details:")
                for dataset, error in failed_subsets:
                    logger.info(f"- {dataset}: {error}")



            success_qiime_subsets = [find_required_files(project_dir.dataset_dirs(dataset=dataset)) for dataset in datasets]
            success_qiime_subsets = [item for item in success_qiime_subsets if item is not None]
            print(success_qiime_subsets)

            metadata_dfs = [file_utils.import_metadata_tsv(i["sample-metadata.tsv"]) for i in success_qiime_subsets]
            metadata_df = pd.concat(metadata_dfs)
            metadata_df = metadata_df.drop(['bam_galaxy', 'bam_bytes', 'bam_aspera', 'bam_md5', 'bam_ftp', 'secondary_sample_accession', 'submission_accession', 'secondary_study_accession', 'library_name'], axis=1)
            print(metadata_df)
            for col in metadata_df.columns:
                logger.info(col)
                logger.info(metadata_df[col].value_counts())

            table_dfs = [file_utils.import_features_biom(i["feature-table.biom"]) for i in success_qiime_subsets]
            for df in table_dfs:
                print(df.shape)

            taxonomy_dicts = [file_utils.Taxonomy(i["taxonomy.tsv"]).taxonomy for i in success_qiime_subsets]
            for d in taxonomy_dicts:
                print(len(d))

        except Exception as global_error:
            logger.critical(f"Fatal pipeline error: {str(global_error)}", exc_info=True)
            raise
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()
