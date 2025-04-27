# ================================== IMPORTS ================================== #

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import qiime2
from qiime2 import Artifact, Metadata

# ================================== LOCAL IMPORTS =================================== #

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from logger import setup_logging
from utils import create_dir, get_average_lengths, get_truncation_lengths

from api.api_io import construct_file_path, output_files_exist, load_with_print
from api.api import (
    import_seqs_from_manifest,
    trim_sequences,
    filter_samples_for_denoising,
    denoise_sequences,
    classify_taxonomy,
)

# Suppress warnings
warnings.filterwarnings("ignore")

# ==================================== FUNCTIONS ===================================== #


class Dataset:
    """16S rRNA sequencing data processing workflow for microbiome analysis.

    Parameters:
        params: Configuration parameters for the workflow
        qiime_dir: Root directory for QIIME2 artifacts
        file_registry: Path registry for input/output files
        metadata: Sample metadata for analysis
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        self.params = params
        self.qiime_dir = Path(params["qiime_dir"])
        self.file_registry: Dict[str, Path] = {}
        self.metadata: Optional[Metadata] = None
        self._create_directories()
        self._setup()

    def _setup(self) -> None:
        """Initialize workflow components."""
        self._validate_inputs()
        self._create_directories()
        self._setup_file_registry()
        self._load_metadata()

    def _validate_inputs(self) -> None:
        """Verify required input files exist before processing."""
        required_files = {
            "manifest": self.params["manifest_tsv"],
            "metadata": self.params["metadata_tsv"],
        }
        for name, path in required_files.items():
            if not Path(path).exists():
                raise FileNotFoundError(f"{name.capitalize()} file not found: {path}")

    def _create_directories(self) -> None:
        """Create output directory structure for QIIME2 artifacts."""
        create_dir(self.qiime_dir)
        create_dir(self.qiime_dir / "demux-stats")
        create_dir(self.qiime_dir / "trimmed-seqs_demux-stats")

    def _setup_file_registry(self) -> None:
        """Define all input/output file paths for workflow components."""
        self.file_registry = {
            "manifest": self.params["manifest_tsv"],
            "metadata": self.params["metadata_tsv"],
            "seqs": construct_file_path(self.qiime_dir, "seqs"),
            "trimmed-seqs": construct_file_path(self.qiime_dir, "trimmed-seqs"),
            "rep-seqs": construct_file_path(self.qiime_dir, "rep-seqs"),
            "table": construct_file_path(self.qiime_dir, "table"),
            "stats": construct_file_path(self.qiime_dir, "stats"),
            "taxonomy": construct_file_path(self.qiime_dir, "taxonomy"),
            "alignment": construct_file_path(self.qiime_dir, "alignment"),
            "tree": construct_file_path(self.qiime_dir, "tree"),
        }

    def _load_metadata(self) -> None:
        """Load and validate QIIME2 metadata file for sample information."""
        print(f"Importing metadata from {self.file_registry['metadata']}...")
        self.metadata = qiime2.Metadata.load(str(self.file_registry["metadata"]))

    def run_workflow(self) -> None:
        """Execute main processing pipeline with error handling and restart capability."""
        try:
            if not self._output_files_exist(["rep-seqs", "table", "stats"]):
                self._process_sequences()
        except Exception as e:
            print(f"Workflow failed: {e}")
            raise

    def _process_sequences(self) -> None:
        """Core sequence processing pipeline: import, trim, filter, denoise, classify."""
        layout = self.params["library_layout"].lower()
        if layout not in {"paired", "single"}:
            raise ValueError("Library layout must be 'single' or 'paired'")

        if self.params["trim_sequences"]:
            seqs = self._import_sequences()
            print("Successfully imported sequences")

            stats = self._calculate_sequence_stats(seqs)
            print("Calculated initial sequence statistics")

            trim_length = stats["trunc_len_f"]

            seqs = self._trim_sequences(
                seqs=seqs,
                trim_length=trim_length,
                minimum_length=100,
                n_cores=32,
                save_intermediates=True,
            )
            print("Completed sequence trimming")

            stats = self._calculate_sequence_stats(seqs)
            print("Updated stats post-trimming")

            counts_file = (
                self.qiime_dir
                / "trimmed-seqs_demux-stats"
                / "per-sample-fastq-counts.tsv"
            )
        else:
            seqs = self._import_sequences()
            print("Successfully imported sequences")

            stats = self._calculate_sequence_stats(seqs)
            print("Calculated trimmed sequence statistics")

            counts_file = (
                self.qiime_dir / "demux-stats" / "per-sample-fastq-counts.tsv"
            )

        seqs = filter_samples_for_denoising(seqs=seqs, counts_file=counts_file)
        print("Filtered low-count samples")

        if layout == "paired":
            trunc_params = (stats["trunc_len_f"], stats["trunc_len_r"])
        else:
            trunc_params = (stats["trunc_len"], 0)

        rep_seqs, table, stats = self._denoise_sequences(seqs, *trunc_params)
        print("Completed denoising pipeline")

        taxonomy = self._taxonomic_classification(rep_seqs)
        print("Assigned taxonomy to features")

    def _import_sequences(self) -> Any:
        """Import sequences from manifest file or load existing artifact."""
        if self._output_files_exist(["seqs"]):
            try:
                return load_with_print(self.qiime_dir, self.file_registry["seqs"].stem)
            except Exception as e:
                print(f"Reload failed: {e}. Reimporting sequences.")

        return self._import_seqs_from_manifest()

    def _import_seqs_from_manifest(self) -> Any:
        """Import raw sequence data using QIIME2 manifest format."""
        print(f"Importing sequences from: {self.file_registry['manifest']}")
        try:
            return import_seqs_from_manifest(
                output_dir=self.qiime_dir,
                manifest_tsv=self.file_registry["manifest"],
                library_layout=self.params["library_layout"],
            )
        except Exception as e:
            raise RuntimeError(f"Sequence import failed: {e}")

    def _calculate_sequence_stats(self, seqs: Artifact) -> Dict[str, float]:
        """Calculate sequence length metrics from demultiplexing statistics."""
        stats_path = self.qiime_dir / "demux-stats"

        avg_len_f, avg_len_r = get_average_lengths(
            forward_file=stats_path / "forward-seven-number-summaries.tsv",
            reverse_file=stats_path / "reverse-seven-number-summaries.tsv",
        )

        trunc_len_f, trunc_len_r = get_truncation_lengths(
            forward_file=stats_path / "forward-seven-number-summaries.tsv",
            reverse_file=stats_path / "reverse-seven-number-summaries.tsv",
            quality_threshold=25,
        )

        print(f"Average lengths    - F: {avg_len_f}, R: {avg_len_r}")
        print(f"Truncation lengths - F: {trunc_len_f}, R: {trunc_len_r}")

        return {
            "avg_len_f": avg_len_f,
            "avg_len_r": avg_len_r,
            "trunc_len_f": trunc_len_f,
            "trunc_len_r": trunc_len_r,
        }

    def _trim_sequences(
        self,
        seqs: Artifact,
        trim_length: int,
        minimum_length: int,
        n_cores: int,
        save_intermediates: bool,
    ) -> Any:
        """Trim adapter sequences and quality filter with restart capability."""
        if self._output_files_exist(["trimmed-seqs"]):
            try:
                print("Reloading trimmed sequences")
                return load_with_print(
                    self.qiime_dir, self.file_registry["trimmed-seqs"].stem
                )
            except Exception as e:
                print(f"Reload failed: {e}. Reprocessing sequences.")

        return self._perform_trimming(
            seqs, trim_length, minimum_length, n_cores, save_intermediates
        )

    def _perform_trimming(
        self,
        seqs: Artifact,
        trim_length: int,
        minimum_length: int,
        n_cores: int,
        save_intermediates: bool,
    ) -> Any:
        """Execute primer removal and quality trimming using CutAdapt."""
        print(
            "Trimming parameters:\n"
            f"  Primers: {self.params['fwd_primer']}/{self.params['rev_primer']}\n"
            f"  Trim length: {trim_length}\n"
            f"  Minimum length: {minimum_length}"
        )
        return trim_sequences(
            output_dir=self.qiime_dir,
            seqs=seqs,
            library_layout=self.params["library_layout"],
            fwd_primer_seq=self.params["fwd_primer"],
            rev_primer_seq=self.params["rev_primer"],
            minimum_length=minimum_length,
            n_cores=n_cores,
            save_intermediates=save_intermediates,
        )

    def _denoise_sequences(
        self,
        seqs: Artifact,
        trunc_len_f: int,
        trunc_len_r: int,
    ) -> Tuple[Artifact, Artifact, Artifact]:
        """Perform ASV/OTU clustering and chimera removal."""
        if self._output_files_exist(["rep-seqs", "table", "stats"]):
            try:
                print("Loading cached denoising results")
                return (
                    load_with_print(self.qiime_dir, "rep-seqs"),
                    load_with_print(self.qiime_dir, "table"),
                    load_with_print(self.qiime_dir, "stats"),
                )
            except Exception as e:
                print(f"Reload failed: {e}. Reprocessing denoising.")

        return self._perform_denoising(seqs, trunc_len_f, trunc_len_r)

    def _perform_denoising(
        self,
        seqs: Artifact,
        trunc_len_f: int,
        trunc_len_r: int,
    ) -> Tuple[Artifact, Artifact, Artifact]:
        """Execute DADA2 or Deblur denoising algorithm."""
        print(f"Starting {self.params['denoise_algorithm']} denoising")
        return denoise_sequences(
            output_dir=self.qiime_dir,
            seqs=seqs,
            library_layout=self.params["library_layout"].lower(),
            instrument_platform=self.params["instrument_platform"].lower(),
            trunc_len_f=trunc_len_f,
            trunc_len_r=trunc_len_r,
            chimera_method=self.params["chimera_method"],
            denoise_algorithm=self.params["denoise_algorithm"],
            n_threads=12,
        )

    def _taxonomic_classification(self, rep_seqs: Artifact) -> Artifact:
        """Assign taxonomy using pre-trained classifier."""
        if self._output_files_exist(["taxonomy"]):
            try:
                print("Loading cached taxonomy")
                return load_with_print(self.qiime_dir, "taxonomy")
            except Exception as e:
                print(f"Reload failed: {e}. Reclassifying taxonomy.")

        return self._assign_taxonomy(rep_seqs)

    def _assign_taxonomy(self, rep_seqs: Artifact) -> Artifact:
        """Execute taxonomic classification using q2-feature-classifier."""
        print(f"Using classifier: {self.params['classifier']}")
        return classify_taxonomy(
            output_dir=self.qiime_dir,
            rep_seqs=rep_seqs,
            classifier_dir=self.params["classifier_dir"],
            classifier=self.params["classifier"],
        )[0]

    def _output_files_exist(self, keys: List[str]) -> bool:
        """Check if all specified output files exist."""
        return all(self.file_registry[key].exists() for key in keys)


class WorkflowRunner:
    """Orchestrates QIIME 2 per-dataset workflow execution.

    Parameters:
        args: Input parameters for workflow configuration
        workflow: Dataset processing instance
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        self.args = args
        self.workflow: Optional[Dataset] = None

    def execute(self) -> bool:
        """Execute complete workflow and return success status."""
        try:
            self.workflow = Dataset(self.args)
            self.workflow.run_workflow()
            print("Workflow completed successfully")
            return True
        except Exception as e:
            print(f"Workflow execution failed: {e}")
            return False
