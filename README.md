# workflow_16s

A modular, extensible **16S rRNA microbiome analysis pipeline** supporting preprocessing, taxonomy assignment, diversity analysis, machine learning, and visualization. Designed for reproducible research.

> See an example output file [here](https://heathermacgregor.github.io/workflow_16s/info/Example_Report.html).

---

## Purpose

This repository provides:

- A configurable **upstream pipeline**: trimming, quality control, and QIIME2 taxonomic processing.
- Advanced **downstream analysis**: diversity metrics, statistical testing, ML-based feature selection, functional prediction, and geospatial mapping.
- Flexible control for grouping and metadata‑driven comparisons.

Originally used for the “nuclear contamination” study, the pipeline is fully adaptable to other case studies.

---

## Repository Structure


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
  ├── README.md # This file
  └── setup.sh 
  </pre>


---

## Configuration

See a breakdown of the default example [here](https://github.com/heathermacgregor/workflow_16s/blob/main/info/config.md).

### Key YAML Settings

- **Hardware**: CPU thread limits (`cpu.limit`).
- **Paths**: Dataset list, metadata, manual metadata directory, project output path.
- **Credentials**: ENA API email for obtaining sequence metadata or download authorization.
- **Execution flags**: Toggle `upstream` and `downstream` processing; optionally enable subset analysis.

### Core Pipeline Modules

#### Upstream
- PCR primer control, subfragment targeting, BLAST database location.
- Tools like `fastqc`, `seqkit`, `validate_16s`, and `cutadapt` for preprocessing.
- QIIME2 settings: trimming, denoising (e.g. DADA2), taxonomy classification, filtering, and cleanup.

#### Downstream
- **Metadata grouping**: Define sample groups using columns (e.g. `nuclear_contamination_status`).
- **Feature preprocessing**: Support for filtering, normalization, CLR transformation, and presence/absence conversion.
- **Statistical testing**: Mann‑Whitney U, Kruskal‑Wallis, Fisher’s exact, and t‑tests across multiple feature representations.
- **Diversity analyses**:
  - *Alpha*: Shannon, Simpson, Pielou’s evenness, Heip, etc., with visualization and optional correlation.
  - *Beta/Ordination*: PCoA, PCA, t-SNE, UMAP across metrics like Bray‑Curtis, Jaccard, Euclidean.
- **Functional prediction**: FaProTax integration for ecological role inference.
- **Mapping & visualization**: Geospatial sample maps, violin plots, feature maps colored by metadata.
- **Machine learning**: Feature selection via RFE, chi‑squared, LASSO, SHAP; permutation importance; performance tracking.

---

## Installation & Setup

```bash
git clone https://github.com/heathermacgregor/workflow_16s.git
cd workflow_16s
bash setup.sh
```

---

## Usage

```bash
bash run.sh [--config PATH_TO_CUSTOM_CONFIG_YAML]
```

## To-Do
- Add to HTML report sections on the original datasets, linking to ENA BioProjects when relevant.
- Add **batch correction methods** that are appropriate for microbial community data (e.g. [ConQuR](https://github.com/wdl2459/ConQuR)).
- Add support for ASV mode (only genus mode is currently available).
