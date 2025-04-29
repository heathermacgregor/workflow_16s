# workflow_16s


<pre> 
  workflow_16s/ 
  ├── references/ 
  │ ├── classifier/ 
  │ │ └── silva-138-99-515-806/ 
  │ ├── conda_envs/
  │ ├── manual_metadata/ 
  │ ├── config.yaml
  │ ├── datasets.tsv
  │ └── datasets.txt
  ├── src/ 
  │ ├── workflow_16s/
  │ │ ├── ena/
  │ │ ├── figures/
  │ │ ├── metadata/
  │ │ ├── qiime/
  │ │ ├── sequences/
  │ │ ├── utils/
  │ │ ├── __init__.py 
  │ │ ├── config.py 
  │ │ └── logger.py 
  │ ├── __init__.py 
  │ └── run.py 
  ├── README.md
  └── setup.sh 
  </pre>

## Configuration File (`config.yaml`)

This YAML configuration file controls all aspects of the pipeline. Below is an overview of each section and its parameters.

### ENA
- **`email`**: Email address used to access ENA (European Nucleotide Archive). Required for some API requests.

### Dataset List
- **`default`**: Path to a plain text file containing ENA accessions for datasets to be downloaded.  
  _Default: `./datasets.txt`_

### Dataset Information
- **`default`**: Path to a TSV file where dataset metadata will be saved.  
  _Default: `./datasets.tsv`_

### Project Directory
- **`default`**: Root directory where intermediate and output files will be stored.  
  _Default: `../../test`_

### Manual Metadata Directory
- **`default`**: Path to a directory containing manually curated metadata files (if available).  
  _Default: `../../manual_metadata`_

### PCR Primers
- **`default`**: Defines how PCR primers are determined.  
  - `"manual"`: Use primers provided manually in metadata.

### Target Subfragment
- Specifies how to select and trim target 16S subfragments.  
  - `"any"`: Use dataset-specific primers and compare at the **genus level**.
  - If a specific subfragment is provided (e.g., `V3-V4`): only include datasets targeting that subfragment, and compare at the **ASV level** using standard primers.  
  _Default: `any`_

### Clean Up FASTQ
- **`default`**: Whether to remove intermediate FASTQ files after processing.  
  _Default: `true`_

---

## Sequence Validation

Parameters for checking whether the sequencing data appears valid:

- **`n_runs`**: Number of validation attempts.
- **`run_targets`**: List of expected sequencing targets. E.g., `16S`, `unknown`.

---

## BLAST

- **`path`**: Location of the BLAST database for identifying sequences.  
  _E.g., `./blast/silva_16s/SILVA_16S_db`_

---

## FastQC / SeqKit / Cutadapt

These tools are used to inspect and clean raw reads.

### FastQC
- **`run`**: Whether to generate FastQC quality reports.

### SeqKit
- **`run`**: Whether to run basic sequence statistics.

### Cutadapt
- **`run`**: Whether to trim adapters and low-quality bases.
- **`start_trim` / `end_trim`**: Fixed number of bases to trim from the start/end.
- **`start_q_cutoff` / `end_q_cutoff`**: Quality threshold for trimming.
- **`min_seq_length`**: Minimum sequence length after trimming.
- **`cores`**: Number of threads to use.

---

## QIIME 2 Processing

Defines how QIIME 2 is used to process and classify datasets.

### Per-Dataset
- **`path`**: Path to the script for per-dataset QIIME 2 processing.
- **`hard_rerun`**: If `true`, re-runs processing even if output already exists.
- **`n_threads`**: Number of threads to use.

#### Trim Sequences
- **`run`**: If `true`, trims sequences (use `false` if Cutadapt was run).
- **`trim_length`**: Truncate sequences to this length.
- **`n_cores`**: Threads to use.
- **`save_intermediates`**: If `true`, saves intermediate QIIME artifacts.

#### Denoise Sequences
- **`denoise_algorithm`**: Denoising method (e.g., `DADA2`).
- **`chimera_method`**: Chimera detection strategy (e.g., `consensus`).
- **`n_threads`**: Threads to use.
- **`trim_left_f/r`**: Trim bases from start of forward/reverse reads.
- **`trunc_q`**: Truncate reads at first quality score below this threshold.
- **`max_ee` / `max_ee_f/r`**: Maximum expected errors.

#### Taxonomic Classification
- **`run`**: If `true`, performs taxonomic classification.
- **`hard_rerun`**: Re-runs classification even if results exist.
- **`classifier_dir`**: Path to pre-trained classifier.
- **`classifier`**: Classifier filename.
- **`classify_method`**: Classification method (e.g., `sklearn`).
- **`maxaccepts`**: Max BLAST hits to accept.
- **`perc_identity`**: Minimum sequence identity.
- **`query_cov`**: Minimum query coverage.

#### Filter
- **`retain_threshold`**: Minimum number of sequences required to retain a sample.


# Known issues:
- Datasets with SingleEnd sequences fail per-dataset QIIME workflow.
