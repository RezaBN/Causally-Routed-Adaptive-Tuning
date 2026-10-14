# CRAT — Supplementary Materials

This directory contains supplementary resources for **Causally-Routed Adaptive Tuning (CRAT)**. The materials complement the main repository by providing an architectural diagram, scripts and generated visualizations, underlying tabular exports, and detailed reports on the principal comparisons, ablation studies, and sensitivity analyses.

> **Navigation:** [Repository home](../README.md) · [Architecture diagram](#1-crat-block-diagram) · [Visualization package](#2-crat-visualization-package) · [Supplementary analyses](#3-supplementary-results-analysis-documents) · [Reproducing visualizations](#reproducing-the-visualizations)

## Contents at a glance

| Resource | Description | Access |
| --- | --- | --- |
| **CRAT Block Diagram** | CRAT architecture illustration and its accompanying caption. | [Open folder](CRAT%20Block%20Diagram/) |
| **CRAT_visualization_package** | Python plotting and table-generation code, input CSV exports, captions, configuration files, and generated outputs. | [Open package](CRAT_visualization_package/) |
| **Supplementary Results Analysis Documents** | Three PDF reports covering comparisons, ablations, and sensitivity analyses. | [Open folder](Supplementary%20Results%20Analysis%20Documents/) |

## 1. CRAT Block Diagram

The [`CRAT Block Diagram/`](CRAT%20Block%20Diagram/) directory contains:

- **[CRAT Diagram.pdf](CRAT%20Block%20Diagram/CRAT%20Diagram.pdf)** — the CRAT block diagram in PDF format.
- **[CRAT Diagram Caption.txt](CRAT%20Block%20Diagram/CRAT%20Diagram%20Caption.txt)** — the corresponding figure caption.

Use the PDF and caption together when referencing the architecture in a manuscript, presentation, or supplementary discussion.

## 2. CRAT Visualization Package

The [`CRAT_visualization_package/`](CRAT_visualization_package/) directory groups the code, data inputs, and rendered outputs for the project’s experimental visualizations and tables.

### Code, instructions, and metadata

| File | Purpose |
| --- | --- |
| [`README.md`](CRAT_visualization_package/README.md) | Package-level overview and documentation. |
| [`RUN_ME.txt`](CRAT_visualization_package/RUN_ME.txt) | Instructions for running the package; consult this for the supported commands and options. |
| [`requirements.txt`](CRAT_visualization_package/requirements.txt) | Python package dependencies. |
| [`crat_visualize.py`](CRAT_visualization_package/crat_visualize.py) | Main visualization script. |
| [`crat_data.py`](CRAT_visualization_package/crat_data.py) | Data-processing functionality. |
| [`crat_plots.py`](CRAT_visualization_package/crat_plots.py) | Figure-generation functionality. |
| [`crat_tables.py`](CRAT_visualization_package/crat_tables.py) | Table-generation functionality. |
| [`captions.json`](CRAT_visualization_package/captions.json) | Machine-readable captions. |
| [`style_default.json`](CRAT_visualization_package/style_default.json) | Default visualization styling. |
| [`table_catalog.json`](CRAT_visualization_package/table_catalog.json) | Table catalog metadata. |
| [`VALIDATION.txt`](CRAT_visualization_package/VALIDATION.txt), [`validation.json`](CRAT_visualization_package/validation.json) | Packaged validation information. |
| [`source_checksums.json`](CRAT_visualization_package/source_checksums.json) | Recorded source checksum metadata. |

Additional examples and configuration files are available under [`examples/`](CRAT_visualization_package/examples/).

### Data inputs

The [`data/`](CRAT_visualization_package/data/) directory includes CSV exports for **English–French (EN–FR), English–Persian (EN–FA), and English–Chinese (EN–ZH)** experiments. The recorded filenames cover main comparisons involving **Llama 2** and **Mistral**, as well as post-hoc ablation and sensitivity results. See the package-level documentation for input expectations and data interpretation.

### Generated outputs

The [`output/`](CRAT_visualization_package/output/) directory is organized by analysis:

| Subdirectory | Scope |
| --- | --- |
| [`comparison/`](CRAT_visualization_package/output/comparison/) | Main comparisons, including quality metrics, baseline comparisons, and resource-related summaries. |
| [`ablation/`](CRAT_visualization_package/output/ablation/) | Ablation comparisons, quality changes, training and inference resources, and routing-related outputs. |
| [`sensitivity/`](CRAT_visualization_package/output/sensitivity/) | Parameter sensitivity, routing choices, head budgets, top-*k* analyses, and robustness-related outputs. |
| [`integrated/`](CRAT_visualization_package/output/integrated/) | Cross-analysis or consolidated figures and tables. |

Within these subdirectories, outputs are grouped by function:

- **`figures/`** — plotted figures, supplied where available as **PDF**, **PNG**, and **SVG**.
- **`tables/`** — presentation-ready and machine-readable tables, supplied where available as **CSV**, **HTML**, **PDF**, **PNG**, **SVG**, and **LaTeX (`.tex`)**.
- **`descriptions/`** — text descriptions and captions for the corresponding figures and tables.
- **`data_tables/`** — supporting tabular analyses and intermediate or derived summaries.

Also see [`output/all_descriptions.txt`](CRAT_visualization_package/output/all_descriptions.txt), [`output/data_notes.txt`](CRAT_visualization_package/output/data_notes.txt), and [`output/run_manifest.json`](CRAT_visualization_package/output/run_manifest.json) for package-wide descriptive, data, and run metadata.

## 3. Supplementary Results Analysis Documents

The [`Supplementary Results Analysis Documents/`](Supplementary%20Results%20Analysis%20Documents/) directory contains three standalone PDF reports:

| Document | Focus | PDF |
| --- | --- | --- |
| **Supplementary A — CRAT Results Comparison** | Main experimental comparisons and results. | [Read Supplementary A](Supplementary%20Results%20Analysis%20Documents/Supplementary%20A.%20CRAT_results_comparison.pdf) |
| **Supplementary B — CRAT Ablation Analysis** | Ablation experiments and component-level analysis. | [Read Supplementary B](Supplementary%20Results%20Analysis%20Documents/Supplementary%20B.%20CRAT_ablation_analysis.pdf) |
| **Supplementary C — CRAT Sensitivity Analysis** | Sensitivity tests and parameter-dependent behavior. | [Read Supplementary C](Supplementary%20Results%20Analysis%20Documents/Supplementary%20C.%20CRAT_sensitivity_analysis.pdf) |

For a detailed interpretation of results, consult the relevant PDF alongside its corresponding figures and tables in the visualization package. The PDFs are supplementary analyses; the scripts and exported data provide the complementary computational materials.

## Reproducing the visualizations

1. Navigate to [`CRAT_visualization_package/`](CRAT_visualization_package/) in a local clone of this repository.
2. Create or activate a suitable Python environment, then install the listed dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Read **[`RUN_ME.txt`](CRAT_visualization_package/RUN_ME.txt)** and the package **[`README.md`](CRAT_visualization_package/README.md)** for the supported execution commands, inputs, and options. Use those instructions rather than assuming command-line flags.
4. Compare generated artifacts with the checked-in materials under [`output/`](CRAT_visualization_package/output/). Consult the validation and checksum metadata where relevant.

**Reproducibility note:** Availability of scripts, data tables, and generated files does not itself establish that every experiment can be rerun in an arbitrary environment. Consult the package instructions for environment requirements and any limitations of the provided inputs.

## Using and citing these materials

When reporting results, identify the relevant **Supplementary A/B/C** document and, when applicable, the specific figure or table from the visualization package. Refer to the CRAT paper and the repository’s citation and licensing information (if provided at the repository root) for formal attribution and reuse conditions.

File and directory names are retained as distributed to support traceability between the supplementary reports, captions, scripts, and generated outputs.
