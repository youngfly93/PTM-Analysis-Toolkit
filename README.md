# PTM Analysis Toolkit

This repository contains a collection of Python scripts for analysing post–translational modifications (PTMs) in proteomics datasets. The toolkit focuses on peptide level properties, comparison of modified versus unmodified peptides and tumour versus normal sample analysis. Most utilities are written in Chinese but the overall structure is outlined below.

## Core projects

The `github_upload_package` directory hosts the main scripts and documentation. The three principal analysis projects are:

1. **`peptide_group_analysis.py`** – groups peptides by detection pattern (PTM-only, PTM+WT and WT-only) to explore physicochemical differences.
2. **`peptide_group_analysis_modified.py`** – an updated grouping logic that considers any type of modification when classifying peptides.
3. **`tumor_vs_normal_ptm_analysis.py`** – compares the relative effect of PTMs between tumour and normal samples.

Supporting modules include `peptide_properties_analyzer.py` for calculating peptide physicochemical properties and `tumor_analysis_core.py` which provides shared data handling routines. A stand‑alone tool `site_score_all.py` performs site score calculations for peptide modifications.

## Installation

Python 3.8+ is recommended. Required packages are listed in `github_upload_package/requirements.txt`:

```bash
pip install -r github_upload_package/requirements.txt
```

## Quick start

Run any of the core scripts directly. For example:

```bash
python github_upload_package/peptide_group_analysis.py
python github_upload_package/peptide_group_analysis_modified.py
python github_upload_package/tumor_vs_normal_ptm_analysis.py
```

Results and figures will be saved to their corresponding output directories as described in the documentation.

## Documentation

Detailed Chinese documentation can be found in `github_upload_package/README.md` and other `*.md` files within that directory. They include usage guides, project structure information and upload instructions.

## License

This project is distributed under the terms of the MIT License. See `github_upload_package/LICENSE` for details.

