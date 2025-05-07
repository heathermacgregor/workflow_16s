# ======================== IMPORT REQUIRED LIBRARIES ======================== #


from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from collections import Counter
import pandas as pd
import re

from workflow_16s.utils import misc_utils, dir_utils, file_utils
import workflow_16s.sequences.analyze as seq_analyze
from workflow_16s.ena.metadata import ENAMetadata

import logging

# ========================== GLOBAL CONFIGURATION ========================== #


logger = logging.getLogger("workflow_16s")

DEFAULT_16S_PRIMERS = {
    "V1-V2": {
        "fwd": {
            "name": "",
            "full_name": "",
            "position": (0, 0),
            "seq": "AGAGTTTGATCMTGGCTCAG",
            "ref": "",
        },
        "rev": {
            "name": "",
            "full_name": "",
            "position": (0, 0),
            "seq": "TGCTGCCTCCCGTAGGAGT",
            "ref": "",
        },
    },
    "V2-V3": {
        "fwd": {
            "name": "",
            "full_name": "",
            "position": (0, 0),
            "seq": "ACTCCTACGGGAGGCAGCAG",
            "ref": "",
        },
        "rev": {
            "name": "",
            "full_name": "",
            "position": (0, 0),
            "seq": "TTACCGCGGCTGCTGGCAC",
            "ref": "",
        },
    },
    "V3-V4": {
        "fwd": {
            "name": "Bakt_341F",
            "full_name": "S-D-Bact-0341-b-S-17",
            "position": (341, 357),
            "seq": "CCTACGGGNGGCWGCAG",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21472016/",
        },
        "rev": {
            "name": "Bakt_805R",
            "full_name": "S-D-Bact-0785-a-A-21",
            "position": (785, 805),
            "seq": "GACTACHVGGGTATCTAATCC",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21472016/",
        },
    },
    "V4": {
        "fwd": {
            "name": "U515F",
            "full_name": "S-*-Univ-0515-a-S-19",
            "position": (515, 533),
            "seq": "GTGCCAGCMGCCGCGGTAA",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21349862/",
        },
        "rev": {
            "name": "806R",
            "full_name": "S-D-Bact-0787-b-A-20",
            "position": (787, 808),
            "seq": "GGACTACHVGGGTWTCTAAT",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/21349862/",
        },
    },
    "V4-V5": {
        "fwd": {
            "name": "515F-Y",
            "full_name": "",
            "position": (515, 533),
            "seq": "GTGYCAGCMGCCGCGGTAA",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/26271760/",
        },
        "rev": {
            "name": "926R",
            "full_name": "S-D-Bact-0907-a-A-19",
            "position": (907, 926),
            "seq": "CCGYCAATTYMTTTRAGTTT",
            "ref": "https://pubmed.ncbi.nlm.nih.gov/26271760/",
        },
    },
    "V6-V8": {
        "fwd": {
            "name": "",
            "full_name": "",
            "position": (0, 0),
            "seq": "AAACTYAAAKGAATTGACGG",
            "ref": "",
        },
        "rev": {
            "name": "",
            "full_name": "",
            "position": (0, 0),
            "seq": "ACGGGCGGTGTGTACAAG",
            "ref": "",
        },
    },
}

# ========================== CORE PROCESSING CLASS ========================== #


def fetch_manual_meta(config, dataset: str):
    """Retrieves and processes manually-collected dataset metadata.

    Args:
        dataset: ENA Project Accession number (e.g., PRJEB1234)

    Returns:
        Dictionary containing metadata, run characteristics, and filtered run lists.
        Returns None if invalid dataset format or metadata retrieval fails.
    """
    manual_metadata_tsv = Path(config["manual_meta_dir"]) / f"{dataset}.tsv"
    if manual_metadata_tsv.is_file():
        return pd.read_csv(
            manual_metadata_tsv, sep="\t", encoding="utf8", low_memory=False
        )
    else:
        return pd.DataFrame({})


class SubsetDataset:
    """Central processing unit for dataset analysis with automated and manual modes.

    Features:
        - Automated primer estimation and metadata validation
        - Manual metadata and primer configuration
        - Error tracking and success/failure reporting

    Attributes:
        config: Configuration dictionary for processing parameters.
        dirs: Subdirectories structure handler.
        success: List of successfully processed datasets with parameters.
        failed: List of failed datasets with error information.
    """

    # ENA-related configurations

    ENA_PATTERN = re.compile(r"^PRJ[EDN][A-Z]\d{4,}$", re.IGNORECASE)
    ENA_METADATA_UNNECESSARY_COLUMNS = [
        "sra_bytes",
        "sra_aspera",
        "sra_galaxy",
        "sra_md5",
        "sra_ftp",
        "fastq_bytes",
        "fastq_aspera",
        "fastq_galaxy",
        "fastq_md5",
        "collection_date_start",
        "collection_date_end",
        "location_start",
        "location_end",
        "ncbi_reporting_standard",
        "datahub",
        "tax_lineage",
        "tax_id",
        "scientific_name",
        "isolation_source",
        "first_created",
        "first_public",
        "last_updated",
        "status",
    ]
    ENA_METADATA_COLUMNS_TO_RENAME = {"lat": "latitude_deg", "lon": "longitude_deg"}

    def __init__(self, config: Dict[str, Any]):
        """Initialize processor with configuration and directory setup."""
        self.config = config
        self.dirs = dir_utils.SubDirs(self.config["project_dir"])
        self.success: List[Dict] = []
        self.failed: List[Dict] = []

    def _determine_target_fragment(self, estimates: Dict[str, Any]) -> str:
        """Determine target fragment from primer estimation results.

        Args:
            estimates: Dictionary mapping targets to estimated subfragments.

        Returns:
            Selected target subfragment (e.g., 'V4').

        Raises:
            ValueError: If no valid subfragment can be determined.
        """
        unique = {v[0] for v in estimates.values()}
        if len(unique) == 1:
            return unique.pop()
        if "16S" in estimates:
            return estimates["16S"][0]
        raise ValueError("No valid target subfragment identified")

    def _process_group(
        self,
        group: pd.DataFrame,
        dataset: str,
        layout: str,
        platform: str,
        target_subfragment: str,
        fwd_primer: str,
        rev_primer: str,
    ) -> Dict[str, Any]:
        """Construct parameters dictionary for a metadata group."""
        return {
            "dataset": dataset,
            "metadata": group,
            "n_runs": len(group),
            "library_layout": layout,
            "instrument_platform": platform,
            "target_subfragment": target_subfragment,
            "pcr_primer_fwd_seq": fwd_primer,
            "pcr_primer_rev_seq": rev_primer,
        }

    def _infer_library_layout(
        self,
        metadata: pd.DataFrame,
        info: Dict,  # Unused but preserved for interface consistency
    ) -> pd.DataFrame:
        """
        Infer library layout from FASTQ FTP URLs in metadata.

        Args:
            metadata: DataFrame containing sequencing metadata.
            info: Dataset info dictionary (unused, preserved for compatibility).

        Returns:
            Updated metadata with corrected library_layout column.
        """
        metadata = metadata.copy()
        metadata["fastq_ftp"] = metadata["fastq_ftp"].fillna("")

        url_counts = [
            len([url for url in ftp_urls.strip().split(";") if url])
            for ftp_urls in metadata["fastq_ftp"]
        ]

        library_layout = [
            "paired" if count == 2 else "single" if count == 1 else "unknown"
            for count in url_counts
        ]

        new_layout = pd.Series(
            library_layout, index=metadata.index, name="library_layout"
        )
        original_lower = metadata["library_layout"].str.lower()

        if not original_lower.equals(new_layout.str.lower()):
            mismatches = metadata[original_lower != new_layout.str.lower()]
            if not mismatches.empty:
                logger.debug(
                    f"Library layout mismatch in {len(mismatches)} rows.\n"
                    f"Differences:\n{mismatches[['library_layout']].join(new_layout.rename('new_layout'))}"
                )
            metadata["library_layout"] = new_layout
        return metadata

    def _process_citations(self, info: Dict) -> List[str]:
        """Extract citations from publication URLs in dataset info."""
        citations = []
        urls = str(info.get("publication_url", "")).strip(";").split(";")
        for url in urls:
            if not url:
                continue
            citation = misc_utils.get_citation(url, style="apa")
            citations.append(citation if citation else url)
        return citations

    def _extract_primers_from_metadata(
        self, meta: pd.DataFrame, info: Dict
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract and validate primers from metadata columns.

        Args:
            meta: Metadata DataFrame.
            info: Dataset info with potential primer sequences.

        Returns:
            Tuple of forward and reverse primer sequences.

        Raises:
            ValueError: If metadata primers conflict with info primers.
        """
        fwd_primer, rev_primer = None, None

        if {"pcr_primer_fwd_seq", "pcr_primer_rev_seq"}.issubset(meta.columns):
            fwd_unique = meta["pcr_primer_fwd_seq"].nunique() == 1
            rev_unique = meta["pcr_primer_rev_seq"].nunique() == 1

            if fwd_unique and rev_unique:
                fwd_primer = meta["pcr_primer_fwd_seq"].iloc[0]
                rev_primer = meta["pcr_primer_rev_seq"].iloc[0]

                info_fwd = info.get("pcr_primer_fwd_seq")
                info_rev = info.get("pcr_primer_rev_seq")

                if info_fwd and info_fwd != fwd_primer:
                    raise ValueError(
                        f"Metadata forward primer {fwd_primer} "
                        f"doesn't match info {info_fwd}"
                    )
                if info_rev and info_rev != rev_primer:
                    raise ValueError(
                        f"Metadata reverse primer {rev_primer} "
                        f"doesn't match info {info_rev}"
                    )
        return (
            fwd_primer or info.get("pcr_primer_fwd_seq"),
            rev_primer or info.get("pcr_primer_rev_seq"),
        )

    def auto(self, dataset: str, meta: pd.DataFrame, ena_runs: Dict):
        """Automatically estimate primers and process metadata groups."""
        estimates = {}

        # Target genes (Default: '16S', 'unknown')

        target_genes = self.config["validate_sequences"]["run_targets"]

        for gene in target_genes:
            runs = ena_runs.get(gene, [])
            runs = [run for run in runs if run in meta["run_accession"].values]

            if not runs:
                continue
            results = seq_analyze.estimate_16s_subfragment(
                metadata=ena_data["metadata"],
                runs=runs,
                run_label=gene,
                n_runs=self.config["validate_sequences"]["n_runs"],
                output_dir=self.dirs.metadata_per_dataset / dataset,
                fastq_dir=self.dirs.seq_data_per_dataset
                / dataset
                / "sequence_validation",
            )

            if gene == "unknown":
                results = {k: v for k, v in results.items() if v[1] >= 10}
            if results:
                ((subfragment, _),) = Counter(results.values()).most_common(1)
                estimates[target] = subfragment
        target_subfragment = self._determine_target_fragment(estimates)

        fwd_primer = DEFAULT_16S_PRIMERS[target_subfragment]["fwd"]["seq"]
        rev_primer = DEFAULT_16S_PRIMERS[target_subfragment]["rev"]["seq"]

        group_columns = ["library_layout", "instrument_platform"]
        for (layout, platform), group in meta.groupby(group_columns, dropna=False):
            if group.empty:
                continue
            params = self._process_group(
                group,
                dataset,
                layout,
                platform,
                target_subfragment,
                fwd_primer,
                rev_primer,
            )
            self.success.append(params)

    def manual(self, dataset: str, info: Dict, meta: pd.DataFrame):
        """Process dataset with manually provided primers and metadata."""
        group_columns = ["library_layout", "instrument_platform"]

        # Primer extraction and validation

        fwd_primer, rev_primer = self._extract_primers_from_metadata(meta, info)

        # Target subfragment handling

        target_subfragment = info.get("target_subfragment")
        if (
            "target_subfragment" in meta.columns
            and meta["target_subfragment"].nunique() == 1
        ):
            target_subfragment = meta["target_subfragment"].iloc[0]
            group_columns.append("target_subfragment")
        # Handle potential primer columns in metadata

        if {"pcr_primer_fwd_seq", "pcr_primer_rev_seq"}.issubset(meta.columns):
            if (
                meta["pcr_primer_fwd_seq"].nunique() > 1
                or meta["pcr_primer_rev_seq"].nunique() > 1
            ):
                group_columns.extend(["pcr_primer_fwd_seq", "pcr_primer_rev_seq"])
        # Process each metadata group

        for cols, group in meta.groupby(group_columns, dropna=False):
            if group.empty:
                continue
            sample_subset = {
                "dataset": dataset,
                "metadata": group,
                "n_runs": len(group),
                "target_subfragment": target_subfragment,
                "pcr_primer_fwd_seq": fwd_primer,
                "pcr_primer_rev_seq": rev_primer,
            }
            for i, col in enumerate(group_columns):
                sample_subset[col] = cols[i]
            self.success.append(sample_subset)

    def process(self, dataset: str, info: Dict):
        """Process a dataset with automated or manual primer configuration.
        Handles metadata retrieval, validation, and error tracking."""
        try:
            # Publication info processing

            citations = self._process_citations(info)

            # Log dataset information

            dataset_info_text = (
                  f"\n[Dataset]             {dataset.upper()}".ljust(50)
                + f"\n[Type]                {info.get('dataset_type', '').upper()}".ljust(50)
                + f"\n[Sequencing Platform] {info.get('instrument_platform', '').upper()} ({info.get('instrument_model', '')})".ljust(50)
                + f"\n[Library Layout]      {info.get('library_layout', '').upper()}".ljust(50)
                + f"\n[Primers]             {info.get('pcr_primer_fwd', '')} ({info.get('pcr_primer_fwd_seq', '')})".ljust(50)
                + f"\n                      {info.get('pcr_primer_rev', '')} ({info.get('pcr_primer_rev_seq', '')})".ljust(50)
                + f"\n[Target]              {info.get('target_gene', '')} {info.get('target_subfragment', '')}".ljust(50)
                + f"\n[Publications]        {citations[0]}".ljust(50)
            )
            
            if len(citations) > 1:
                for citation in citations[1:]:
                    dataset_info_text += f"\n                      {citation}".ljust(50)

            logger.info(dataset_info_text)

            # Metadata retrieval

            if self.ENA_PATTERN.match(dataset):
                ena_data = ENAMetadata(email=self.config["ena_email"])
                ena_data.process_dataset(dataset, info)
                ena_meta = ena_data.df
                ena_runs = ena_data.runs
            else:
                ena_meta = pd.DataFrame()
                ena_runs = {}
            manual_meta = fetch_manual_meta(self.config, dataset)

            # Metadata validation and combination

            meta = self._combine_metadata(dataset, ena_meta, manual_meta, info)
            meta = self._infer_library_layout(meta, info)

            # Primer processing mode

            if self.config["pcr_primers_mode"] == "estimate":
                self.auto(dataset, meta, ena_runs)
            else:
                self.manual(dataset, info, meta)
        except Exception as e:
            logger.error(f"Dataset {dataset} failed: {str(e)}", exc_info=True)
            self.failed.append({"dataset": dataset, "error": str(e)})
            raise

    def _combine_metadata(
        self,
        dataset: str,
        ena_meta: pd.DataFrame,
        manual_meta: pd.DataFrame,
        info: Dict,
    ) -> pd.DataFrame:
        """Combine and validate ENA/manual metadata."""
        if not ena_meta.empty and info.get("dataset_type") != "ENA":
            raise ValueError(
                f"ENA metadata present for non-ENA dataset type: "
                f"{info.get('dataset_type')}"
            )
        if ena_meta.empty and not manual_meta.empty:
            return manual_meta
        elif not ena_meta.empty and manual_meta.empty:
            return ena_meta
        else:
            combined = self.combine_ena_and_manual_metadata(
                dataset, ena_meta, manual_meta
            )
            if combined.empty:
                raise ValueError(f"No valid samples after metadata merge for {dataset}")
            return combined

    def combine_ena_and_manual_metadata(
        self, dataset: str, ena_meta: pd.DataFrame, manual_meta: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge ENA and manual metadata with conflict resolution."""
        # Column standardization

        ena_meta.columns = ena_meta.columns.str.lower().str.strip()
        manual_meta.columns = manual_meta.columns.str.lower().str.strip()

        # Column presence validation

        for df, df_type in [(ena_meta, "ENA"), (manual_meta, "manual")]:
            if "run_accession" not in df.columns:
                raise ValueError(
                    f"{df_type} metadata for {dataset} missing 'run_accession' column"
                )
        # Column conflict resolution

        manual_meta, ena_meta = self._resolve_column_conflicts(manual_meta, ena_meta)

        # ENA metadata cleanup

        ena_meta = ena_meta.drop(
            columns=ena_meta.columns.intersection(self.ENA_METADATA_UNNECESSARY_COLUMNS)
        )
        ena_meta = ena_meta.rename(
            columns={
                col: self.ENA_METADATA_COLUMNS_TO_RENAME[col]
                for col in ena_meta.columns.intersection(
                    self.ENA_METADATA_COLUMNS_TO_RENAME
                )
            }
        )

        # Merge metadata

        meta = manual_meta.merge(ena_meta, on="run_accession", how="left")
        if "dataset_id" not in meta.columns:
            meta["dataset_id"] = f"ENA_{dataset}"
        return meta

    @staticmethod
    def _resolve_column_conflicts(
        manual_meta: pd.DataFrame, ena_meta: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Resolve column name conflicts between manual and ENA metadata."""
        common_cols = set(ena_meta.columns) & set(manual_meta.columns) - {
            "run_accession"
        }
        ena_processed = ena_meta.copy()

        for col in common_cols:
            if manual_meta[col].equals(ena_processed[col]):
                ena_processed = ena_processed.drop(columns=col)
            else:
                ena_processed = ena_processed.rename(columns={col: f"{col}_ena"})
        return manual_meta, ena_processed
