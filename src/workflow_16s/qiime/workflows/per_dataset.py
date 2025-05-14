# ===================================== IMPORTS ====================================== #

# Standard library imports
import os
import sys
import glob
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party library imports
import qiime2
from qiime2 import Artifact, Metadata

# ================================== LOCAL IMPORTS =================================== #

import os

# Set the environment variable BEFORE importing tempfile
os.environ["TMPDIR"] = "/opt/tmp"  # Unix/Linux
# os.environ["TEMP"] = "C:\\my_temp"  # Windows (use TEMP or TMP instead)

# Create the directory if it doesn't exist
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

# Now import tempfile (order matters!)
import tempfile

# Test if it worked
print(tempfile.gettempdir())  # Should output "/opt/tmp"

# Suppress warnings
warnings.filterwarnings("ignore")

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from utils import create_dir, get_average_lengths, get_truncation_lengths
from api.api_io import construct_file_path, output_files_exist, load_with_print
from api.api import (
    import_seqs_from_manifest,
    trim_sequences,
    filter_samples_for_denoising,
    denoise_sequences,
    classify_taxonomy,
    collapse_to_genus,
)

# ================================= GLOBAL VARIABLES ================================= #

DEFAULT_N = 20
DEFAULT_MIN_READS = 1000
DEFAULT_MIN_LENGTH = 100
DEFAULT_RETRY_TRUNC_LENGTHS = [(250, 220), (150, 150), (250, 0)]
DEFAULT_N_THREADS = 12

# ==================================== CLASSES ====================================== #

class Dataset:
    """
    16S rRNA sequencing data processing workflow for per-dataset microbiome 
    analysis.

    Attributes:
        params:        Dictionary of configuration parameters.
        qiime_dir:     Path to QIIME2 output directory.
        file_registry: Dictionary mapping file types to paths.
        metadata:      QIIME2 Metadata object containing sample information.
    """

    def __init__(self, params: Dict[str, Any]) -> None:
        """Initialize dataset with processing parameters."""
        self.params = params
        self.qiime_dir = Path(params["qiime_dir"])
        self.file_registry: Dict[str, Path] = {}
        self.metadata: Optional[Metadata] = None
        self._setup()

    def _setup(self) -> None:
        """Initialize workflow components."""
        self._validate_inputs()
        self._create_directories()
        self._setup_file_registry()

    def _validate_inputs(self) -> None:
        """Verify required input files exist."""
        required_files = {
            "manifest": self.params["manifest_tsv"],
            "metadata": self.params["metadata_tsv"],
        }
        missing = [
            name 
            for name, path in required_files.items() if not Path(path).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing required files: {', '.join(missing)}"
            )

    def _create_directories(self) -> None:
        """Create directory structure for QIIME2 outputs."""
        dirs_to_create = [
            self.qiime_dir,
            self.qiime_dir / "demux-stats",
            self.qiime_dir / "trimmed-seqs_demux-stats",
        ]
        for directory in dirs_to_create:
            create_dir(directory)

    def _setup_file_registry(self) -> None:
        """Initialize paths for all input/output files."""
        self.file_registry = {
            "manifest": Path(self.params["manifest_tsv"]),
            "metadata": Path(self.params["metadata_tsv"]),
            "seqs": construct_file_path(self.qiime_dir, "seqs"),
            "trimmed-seqs": construct_file_path(self.qiime_dir, "trimmed-seqs"),
            "rep-seqs": construct_file_path(self.qiime_dir, "rep-seqs"),
            "table": construct_file_path(self.qiime_dir, "table"),
            "stats": construct_file_path(self.qiime_dir, "stats"),
            "taxonomy": construct_file_path(self.qiime_dir, "taxonomy"),
            "alignment": construct_file_path(self.qiime_dir, "alignment"),
            "tree": construct_file_path(self.qiime_dir, "tree"),
            "collapsed_table": construct_file_path(self.qiime_dir, "table_6"),
        }

    def run_workflow(self) -> None:
        """Execute main processing pipeline with error handling."""
        try:
            required_outputs = [
                "rep-seqs",
                "table",
                "stats",
                "taxonomy",
                "collapsed_table",
            ]
            if not self._output_files_exist(required_outputs):
                self._process_sequences()
        except Exception as e:
            self._handle_processing_error(e)

    def _handle_processing_error(self, error: Exception) -> None:
        """Handle and clean up after processing errors."""
        print(f"  ❌ Workflow failed: {str(error)}")
        self.clean_qiime_dir()
        raise RuntimeError("Workflow execution aborted") from error

    def _process_sequences(self) -> None:
        """Core sequence processing pipeline."""
        layout = self.params["library_layout"].lower()
        self._validate_library_layout(layout)

        if self.params["trim_sequences"]:
            seqs, counts_file = self._process_with_trimming()
        else:
            seqs, counts_file = self._process_without_trimming()

        seqs = filter_samples_for_denoising(seqs, counts_file, DEFAULT_MIN_READS)
        print("  ✅ Filtered low-count samples")

        trunc_params = self._calculate_truncation_params(seqs, layout)
        rep_seqs, table, stats = self._denoise_sequences(seqs, *trunc_params)
        print("  ✅ Completed denoising pipeline")

        taxonomy = self._taxonomic_classification(rep_seqs)
        print("  ✅ Assigned taxonomy to features")

        self._collapse_to_genus(table, taxonomy)
        print("  ✅ Collapsed table to genus level")

    def _validate_library_layout(self, layout: str) -> None:
        """Validate library layout parameter."""
        if layout not in {"paired", "single"}:
            raise ValueError(
                f"❓ Library layout '{layout}' not recognized. "
                f"Expected 'single' or 'paired'."
            )

    def _process_with_trimming(self) -> Tuple[Artifact, Path]:
        """Process sequence data with trimming step."""
        seqs = self._import_sequences()
        print("  ✅ Imported raw sequences")

        stats = self._calculate_sequence_stats(seqs)
        print("  ✅ Calculated raw sequence statistics")
        
        seqs = self._trim_sequences(
            seqs=seqs,
            trim_length=stats["trunc_len_f"],
            minimum_length=DEFAULT_MIN_LENGTH,
            n_cores=self.params.get("trim_cores", 32),
            save_intermediates=True,
        )
        print("  ✅ Completed sequence trimming")

        stats = self._calculate_sequence_stats(seqs)
        print("  ✅ Updated statistics post-trimming")

        return (
            seqs,
            self.qiime_dir / "trimmed-seqs_demux-stats" / "per-sample-fastq-counts.tsv",
        )

    def _process_without_trimming(self) -> Tuple[Artifact, Path]:
        """Process pre-trimmed sequence data."""
        seqs = self._import_sequences()
        print("  ✅ Imported trimmed sequences")

        stats = self._calculate_sequence_stats(seqs)
        print("  ✅ Calculated trimmed sequence statistics")

        return seqs, self.qiime_dir / "demux-stats" / "per-sample-fastq-counts.tsv"

    def _calculate_truncation_params(
        self, seqs: Artifact, layout: str
    ) -> Tuple[int, int]:
        """Determine truncation parameters based on library layout."""
        stats = self._calculate_sequence_stats(seqs)
        return (
            (stats["trunc_len_f"], stats["trunc_len_r"])
            if layout == "paired"
            else (stats["trunc_len_f"], 0)
        )

    def _import_sequences(self) -> Artifact:
        """Import or load sequence artifact."""
        # Load existing output files if the 'hard_rerun' flag is absent
        if not self.params["hard_rerun"] and self._output_files_exist(["seqs"]):
            return self._load_existing_artifact("seqs")
        return self._import_seqs_from_manifest()

    def _load_existing_artifact(self, artifact_key: str) -> Artifact:
        """Load existing QIIME2 artifact with validation."""
        try:
            return load_with_print(
                self.qiime_dir, self.file_registry[artifact_key].stem
            )
        except Exception as e:
            print(f"  ❌ Failed to load {artifact_key}: {e}")
            raise

    def _import_seqs_from_manifest(self) -> Artifact:
        """Import sequences using QIIME2 manifest format."""
        manifest_tsv = self.file_registry['manifest']
        print(f"  🔄 Importing sequences from '{manifest_tsv}'")
        try:
            return import_seqs_from_manifest(
                output_dir=self.qiime_dir,
                manifest_tsv=self.file_registry["manifest"],
                library_layout=self.params["library_layout"],
            )
        except Exception as e:
            raise RuntimeError(f"  ❌ Sequence import failed: {str(e)}") from e

    def _calculate_sequence_stats(self, seqs: Artifact) -> Dict[str, float]:
        """Calculate sequence statistics from demux results."""
        stats_path = self.qiime_dir / "demux-stats"

        avg_len_f, avg_len_r = get_average_lengths(
            forward_file=stats_path / "forward-seven-number-summaries.tsv",
            reverse_file=stats_path / "reverse-seven-number-summaries.tsv",
        )

        trunc_len_f, trunc_len_r = get_truncation_lengths(
            forward_file=stats_path / "forward-seven-number-summaries.tsv",
            reverse_file=stats_path / "reverse-seven-number-summaries.tsv",
            quality_threshold=self.params.get("quality_threshold", 25),
        )

        print(
            f"    • {'Average Length'.ljust(DEFAULT_N)}: {avg_len_f} / {avg_len_r}"
            f"    • {'Truncation Length'.ljust(DEFAULT_N)}: {trunc_len_f} / {trunc_len_r}"
        )

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
    ) -> Artifact:
        """Trim sequences with restart capability."""
        # Load existing output files if the 'hard_rerun' flag is absent
        if not self.params["hard_rerun"] and self._output_files_exist(["trimmed-seqs"]):
            return self._load_existing_artifact("trimmed-seqs")
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
    ) -> Artifact:
        """Execute primer removal and quality trimming."""
        print(
            "  🔄 Trimming sequences with [CutAdapt]\n"
            f"    • Primers:        {self.params['fwd_primer']}\n"
            f"                      {self.params['rev_primer']}\n"
            f"    • Trim Length:    {trim_length}\n"
            f"    • Minimum Length: {minimum_length}"
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
        """Perform denoising with fallback strategies."""
        # Load existing output files if the 'hard_rerun' flag is absent
        if not self.params["hard_rerun"] and self._output_files_exist(
            ["rep-seqs", "table", "stats"]
        ):
            return self._load_denoising_artifacts()

        for fallback_trunc in DEFAULT_RETRY_TRUNC_LENGTHS:
            try:
                return self._perform_denoising(seqs, trunc_len_f, trunc_len_r)
            except Exception as e:
                print(f"  ⚠️ Denoising failed with error: {e}")
                print(f"  🔄 Retrying with truncation lengths: {fallback_trunc}")
                trunc_len_f, trunc_len_r = fallback_trunc

        raise RuntimeError("All denoising attempts failed")

    def _load_denoising_artifacts(self) -> Tuple[Artifact, Artifact, Artifact]:
        """Load existing denoising results."""
        print("  🔄 Loading cached denoising results...")
        return (
            self._load_existing_artifact("rep-seqs"),
            self._load_existing_artifact("table"),
            self._load_existing_artifact("stats"),
        )

    def _perform_denoising(
        self,
        seqs: Artifact,
        trunc_len_f: int,
        trunc_len_r: int,
    ) -> Tuple[Artifact, Artifact, Artifact]:
        """Execute denoising algorithm."""
        self._validate_denoise_params()
        print(
            f"  🔄 Denoising sequences with: {self.params['denoise_algorithm']}\n"
            f"    • Library Layout:      {self.params['library_layout']}\n"
            f"    • Instrument Platform: {self.params['instrument_platform']}\n"
            f"    • Trunc Length:        {trunc_len_f}"
            f"{'' if (trunc_len_r == 0 or trunc_len_r != trunc_len_r) else f' / {trunc_len_r}'}\n"
            f"    • Chimera Method:      {self.params['chimera_method']}"
        )
        return denoise_sequences(
            output_dir=self.qiime_dir,
            seqs=seqs,
            library_layout=self.params["library_layout"],
            instrument_platform=self.params["instrument_platform"],
            trunc_len_f=trunc_len_f,
            trunc_len_r=trunc_len_r,
            chimera_method=self.params["chimera_method"],
            denoise_algorithm=self.params["denoise_algorithm"],
            n_threads=self.params.get("denoise_threads", DEFAULT_N_THREADS),
        )

    def _validate_denoise_params(self) -> None:
        """Validate denoising parameters."""
        valid_algorithms = {"dada2", "deblur"}
        if self.params["denoise_algorithm"].lower() not in valid_algorithms:
            raise ValueError(
                f"❓ Denoising algorithm '{self.params['denoise_algorithm']}' "
                f"not recognized. Expected one of {valid_algorithms}"
            )

    def _taxonomic_classification(self, rep_seqs: Artifact) -> Artifact:
        """Assign taxonomy using classifier."""
        # Load existing output files if the 'hard_rerun' flag is absent
        if not self.params["hard_rerun"] and self._output_files_exist(["taxonomy"]):
            return self._load_existing_artifact("taxonomy")
        return self._assign_taxonomy(rep_seqs)

    def _assign_taxonomy(self, rep_seqs: Artifact) -> Artifact:
        """Execute taxonomic classification."""
        print(f"  🔄 Classifying taxonomy with [{self.params['classifier']}]")
        return classify_taxonomy(
            output_dir=self.qiime_dir,
            rep_seqs=rep_seqs,
            classifier_dir=self.params["classifier_dir"],
            classifier=self.params["classifier"],
        )[0]

    def _collapse_to_genus(self, table: Artifact, taxonomy: Artifact) -> Artifact:
        """Collapse features to genus level."""
        # Load existing output files if the 'hard_rerun' flag is absent
        if not self.params["hard_rerun"] and self._output_files_exist(["collapsed_table"]):
            return self._load_existing_artifact("collapsed_table")
        return collapse_to_genus(
            output_dir=self.qiime_dir,
            table=table,
            taxonomy=taxonomy,
        )

    def _output_files_exist(self, keys: List[str]) -> bool:
        """Check existence of specified output files."""
        return all(self.file_registry[key].exists() for key in keys)

    def clean_qiime_dir(self) -> None:
        """
        Clean up intermediate files from QIIME2 output directories.

        Args:
            qiime_dir: Root directory containing QIIME2 artifacts.
        """
        print(f"  ♻️ Cleaning up...")
        qiime_path = Path(self.args["qiime_dir"])
        subdirs_to_clean = ["demux-stats", "trimmed-seqs_demux-stats"]
        files_to_remove = ["data.jsonp"]
        extensions_to_remove = ["*.html", "*.pdf"]
        dirs_to_remove = ["dist", "q2templateassets"]

        for subdir in subdirs_to_clean:
            subdir_path = qiime_path / subdir
            if not subdir_path.exists():
                continue

            # Remove specified files
            for file_name in files_to_remove:
                file_path = subdir_path / file_name
                if file_path.exists():
                    file_path.unlink()

            # Remove files by extension
            for ext in extensions_to_remove:
                for file_path in subdir_path.glob(ext):
                    file_path.unlink()

            # Remove entire directories
            for dir_name in dirs_to_remove:
                dir_path = subdir_path / dir_name
                if dir_path.exists() and dir_path.is_dir():
                    shutil.rmtree(dir_path)



class WorkflowRunner:
    """
    Orchestrate workflow execution with cleanup handling.

    Attributes:
        args:     Dictionary of workflow parameters.
        workflow: Initialized Dataset processing instance.
    """

    def __init__(self, args: Dict[str, Any]) -> None:
        self.args = args
        self.workflow: Optional[Dataset] = None

    def execute(self) -> bool:
        """Execute workflow and return success status."""
        try:
            self.workflow = Dataset(self.args)
            self.workflow.run_workflow()
            self.clean_qiime_dir()
            print("  ✅ Workflow completed successfully!")
            return True
        except Exception as e:
            self.clean_qiime_dir()
            print(f"  ❌ Workflow execution failed: {str(e)}")
            return False

    def clean_qiime_dir(self) -> None:
        """
        Clean up intermediate files from QIIME2 output directories.

        Args:
            qiime_dir: Root directory containing QIIME2 artifacts.
        """
        print(f"  ♻️ Cleaning up...")
        qiime_path = Path(self.args["qiime_dir"])
        subdirs_to_clean = ["demux-stats", "trimmed-seqs_demux-stats"]
        files_to_remove = ["data.jsonp"]
        extensions_to_remove = ["*.html", "*.pdf"]
        dirs_to_remove = ["dist", "q2templateassets"]

        for subdir in subdirs_to_clean:
            subdir_path = qiime_path / subdir
            if not subdir_path.exists():
                continue

            # Remove specified files
            for file_name in files_to_remove:
                file_path = subdir_path / file_name
                if file_path.exists():
                    file_path.unlink()

            # Remove files by extension
            for ext in extensions_to_remove:
                for file_path in subdir_path.glob(ext):
                    file_path.unlink()

            # Remove entire directories
            for dir_name in dirs_to_remove:
                dir_path = subdir_path / dir_name
                if dir_path.exists() and dir_path.is_dir():
                    shutil.rmtree(dir_path)
