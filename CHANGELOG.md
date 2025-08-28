# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive project restructuring following bioinformatics best practices
- Standard repository files (LICENSE, CONTRIBUTING.md, CODE_OF_CONDUCT.md)
- Requirements.txt for Python dependency management
- Docker containerization support
- GitHub Actions CI/CD pipeline
- Basic testing framework structure
- Enhanced documentation with workflow diagrams
- Installation verification scripts

### Changed
- Reorganized project structure into standard directories (data/, results/, config/, scripts/, docs/, tests/)
- Moved configuration files to dedicated config/ directory
- Moved scripts to dedicated scripts/ directory
- Enhanced README.md with better installation and usage instructions

### Fixed
- Improved error handling in setup and run scripts
- Enhanced path management for different project structures

## [1.0.0] - Initial Release

### Added
- Modular 16S rRNA gene analysis pipeline
- QIIME 2 integration for sequence processing
- Comprehensive downstream analysis capabilities
- Alpha and beta diversity analysis
- Statistical testing framework
- Machine learning feature selection
- Interactive HTML report generation
- Geospatial mapping capabilities
- Nuclear facility proximity analysis
- Conda environment management
- Automated setup and execution scripts

### Features
- **Upstream Processing**:
  - ENA data retrieval and processing
  - Sequence quality assessment (FastQC, SeqKit)
  - PCR primer prediction and validation
  - Sequence trimming with CutAdapt
  - QIIME 2 workflow execution
  - Taxonomic classification with SILVA database

- **Downstream Analysis**:
  - Alpha diversity metrics (Shannon, Simpson, Pielou's evenness)
  - Beta diversity and ordination (PCoA, PCA, t-SNE, UMAP)
  - Statistical testing (Mann-Whitney U, Kruskal-Wallis, Fisher's exact)
  - Feature selection using CatBoost and other ML methods
  - Functional prediction with FAPROTAX
  - Geospatial sample mapping
  - Interactive visualizations

- **Technical Features**:
  - Configurable YAML-based settings
  - Comprehensive logging and progress tracking
  - Error handling and validation
  - Multi-dataset processing capability
  - Automated environment setup