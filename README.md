# Causally-Routed Adaptive Tuning (CRAT)

**Research repository for Causally-Routed Adaptive Tuning (CRAT): experimental artifacts, model comparisons, analysis code, and supplementary results.**

This repository brings together the materials associated with **CRAT**, a research approach to adaptive tuning and routing. It provides experiment-level records for **Llama 2** and **Mistral**, as well as supporting analyses of translation quality, computational efficiency, routing behavior, ablation variants, and sensitivity to experimental settings.

The repository is intended to help readers **inspect the reported evidence, trace analyses to their source exports, and regenerate the supplied figures and tables**. For a detailed account of the method, consult the associated CRAT manuscript and the architecture diagram linked below; this README is a navigation and reproducibility guide, not a replacement for the methodological specification.

**Start here:** [Architecture diagram](supplementary/CRAT%20Block%20Diagram/CRAT%20Diagram.pdf) · [Supplementary overview](supplementary/README.md) · [Visualization package](supplementary/CRAT_visualization_package/README.md) · [Supplementary analyses](#supplementary-analysis-reports)

---

## Repository organization

```text
Causally-Routed-Adaptive-Tuning/
├── Llama2/                           # Llama 2 experiment outputs and summaries
├── Mistral/                          # Mistral experiment materials
├── scripts/                          # Project scripts
├── supplementary/
│   ├── README.md                     # Guide to the supplementary materials
│   ├── CRAT Block Diagram/
│   │   ├── CRAT Diagram.pdf
│   │   └── CRAT Diagram Caption.txt
│   ├── CRAT_visualization_package/
│   │   ├── README.md
│   │   ├── RUN_ME.txt
│   │   ├── requirements.txt
│   │   ├── crat_visualize.py
│   │   ├── crat_data.py
│   │   ├── crat_plots.py
│   │   ├── crat_tables.py
│   │   ├── data/
│   │   └── output/
│   │       ├── comparison/
│   │       ├── ablation/
│   │       ├── sensitivity/
│   │       └── integrated/
│   └── Supplementary Results Analysis Documents/
│       ├── Supplementary A. CRAT_results_comparison.pdf
│       ├── Supplementary B. CRAT_ablation_analysis.pdf
│       └── Supplementary C. CRAT_sensitivity_analysis.pdf
└── README.md                         # This file
```

| Location | What to find |
| --- | --- |
| [`Llama2/`](Llama2/) | Llama 2 runs, configuration and metadata exports, comparison results, training logs, model/adaptor-related artifacts, and aggregated or post-hoc analyses. |
| [`Mistral/`](Mistral/) | Model-specific experimental materials for the Mistral family. |
| [`scripts/`](scripts/) | Project scripts; inspect individual scripts and their arguments before execution. |
| [`supplementary/`](supplementary/README.md) | Architecture figure, reproducible visualization resources, and three detailed PDF reports. |

> **Naming note:** The directory is spelled **`Llama2`** (not `Lama2`). Use the spelling in the repository when writing paths or commands.

## Study and experimental artifacts

The checked-in filenames identify experiments on the **OPUS-100** setting for three English-to-target translation directions:

| Direction | Language pair |
| --- | --- |
| EN–FA | English → Persian |
| EN–FR | English → French |
| EN–ZH | English → Chinese |

The available result exports cover the following kinds of analyses:

- **Main comparisons:** CRAT and alternative adaptation strategies, with quality- and resource-related exports.
- **Ablation studies:** component or objective variants and their effects on quality, efficiency, and routing behavior.
- **Sensitivity analyses:** behavior under different parameter settings, including analyses labeled for top-*k*, head budgets, sparsity, and routing/gating choices.
- **Post-hoc and summary analyses:** per-run/per-seed records, consolidated CSV tables, metadata, and quality-evaluation exports, including COMET-labeled results.

For example, the `Llama2/` records include runs labeled **base, CRAT, CRAT-CE, CRAT-CE-KL, CRAT-no-sparse, full SFT, targeted SFT, LoRA, rsLoRA, and DoRA**. These are **artifact labels**, not full definitions of the corresponding optimization objectives; refer to the run configurations, project scripts, and manuscript for experimental details.

Useful entry points for Llama 2 results include:

- [`Llama2/seed42_aggregate/opus100/`](Llama2/seed42_aggregate/opus100/) — consolidated main-comparison summaries by translation direction.
- [`Llama2/seed42_posthoc/opus100/`](Llama2/seed42_posthoc/opus100/) — post-hoc analyses, including COMET-, routing-, and efficiency-related CSV exports.

Folder contents may evolve as additional experiments are incorporated. File names and metadata should be treated as the authoritative identifiers of the available runs.

## CRAT architecture

The architecture illustration is available as a PDF, with a separate caption:

- **[CRAT block diagram (PDF)](supplementary/CRAT%20Block%20Diagram/CRAT%20Diagram.pdf)**
- **[CRAT diagram caption (TXT)](supplementary/CRAT%20Block%20Diagram/CRAT%20Diagram%20Caption.txt)**

Read the figure together with the manuscript for the intended interpretation of individual modules and connections.

## Supplementary analysis reports

Three companion reports provide more detailed treatment of the reported experiments:

| Report | Subject | Document |
| --- | --- | --- |
| **Supplementary A** | Main results and comparisons | [CRAT results comparison (PDF)](supplementary/Supplementary%20Results%20Analysis%20Documents/Supplementary%20A.%20CRAT_results_comparison.pdf) |
| **Supplementary B** | Component-level ablation analysis | [CRAT ablation analysis (PDF)](supplementary/Supplementary%20Results%20Analysis%20Documents/Supplementary%20B.%20CRAT_ablation_analysis.pdf) |
| **Supplementary C** | Sensitivity and robustness analyses | [CRAT sensitivity analysis (PDF)](supplementary/Supplementary%20Results%20Analysis%20Documents/Supplementary%20C.%20CRAT_sensitivity_analysis.pdf) |

Each report should be consulted together with the corresponding plots, tables, source CSV exports, and captions in the [visualization package](supplementary/CRAT_visualization_package/README.md).

## Figures, tables, and visualization code

The [`supplementary/CRAT_visualization_package/`](supplementary/CRAT_visualization_package/) directory contains the plotting and table-generation utilities used to organize the experimental visualizations.

| Resource | Location |
| --- | --- |
| Package guide | [`README.md`](supplementary/CRAT_visualization_package/README.md) |
| Execution instructions | [`RUN_ME.txt`](supplementary/CRAT_visualization_package/RUN_ME.txt) |
| Dependencies | [`requirements.txt`](supplementary/CRAT_visualization_package/requirements.txt) |
| Main entry script | [`crat_visualize.py`](supplementary/CRAT_visualization_package/crat_visualize.py) |
| Supporting modules | [`crat_data.py`](supplementary/CRAT_visualization_package/crat_data.py), [`crat_plots.py`](supplementary/CRAT_visualization_package/crat_plots.py), [`crat_tables.py`](supplementary/CRAT_visualization_package/crat_tables.py) |
| Source tabular exports | [`data/`](supplementary/CRAT_visualization_package/data/) |
| Generated material | [`output/`](supplementary/CRAT_visualization_package/output/) |
| Validation information | [`VALIDATION.txt`](supplementary/CRAT_visualization_package/VALIDATION.txt), [`validation.json`](supplementary/CRAT_visualization_package/validation.json) |
| Source checksum metadata | [`source_checksums.json`](supplementary/CRAT_visualization_package/source_checksums.json) |

The output is organized into **comparison**, **ablation**, **sensitivity**, and **integrated** analyses. Where provided, figures are available in **PDF, PNG, and SVG**; tables include formats such as **CSV, HTML, PDF, PNG, SVG, and LaTeX**. The package also contains text descriptions and captions to support interpretation and reuse.

## Getting started

### 1. Obtain the repository

If you have access to the repository, clone it using your authenticated GitHub account:

```bash
git clone https://github.com/RezaBN/Causally-Routed-Adaptive-Tuning.git
cd Causally-Routed-Adaptive-Tuning
```

If you already have a local clone, use that copy rather than cloning again.

### 2. Set up the visualization dependencies

From the repository root, create and activate an optional Python virtual environment:

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source .venv/bin/activate
```

Install the visualization package dependencies:

```bash
python -m pip install -r supplementary/CRAT_visualization_package/requirements.txt
```

### 3. Follow the supported visualization commands

Before running the scripts, consult:

1. [`supplementary/CRAT_visualization_package/RUN_ME.txt`](supplementary/CRAT_visualization_package/RUN_ME.txt)
2. [`supplementary/CRAT_visualization_package/README.md`](supplementary/CRAT_visualization_package/README.md)

These files document the package-specific workflow. This root README intentionally does **not** assume undocumented command-line options or claim that the complete model-training environment can be recreated from the visualization dependencies alone.

### 4. Inspect or compare the outputs

Use the supplied results and artifacts under [`output/`](supplementary/CRAT_visualization_package/output/) and consult the description files, [`run_manifest.json`](supplementary/CRAT_visualization_package/output/run_manifest.json), and validation metadata when checking provenance or comparing generated files.

## Reproducibility and interpretation

The repository provides several complementary layers of evidence:

1. **Experiment-level records** under the model-specific directories, including configurations, results, metadata, and available logs.
2. **Source and derived tables** in the supplementary visualization package.
3. **Rendered figures and tables**, with accompanying captions/descriptions.
4. **Extended scientific discussion** in Supplementary A, B, and C.

For reproducible reporting, identify the **model family, language direction, experimental variant, seed (where applicable), dataset setting, source file, and repository commit**. Confirm any numerical claims against the relevant source CSV and the accompanying supplementary report rather than relying on filenames or figure titles alone.

**Scope:** The presence of output artifacts and plotting scripts should not be interpreted as a guarantee that every training run can be reproduced without additional hardware, data-access permissions, model licenses, environment configuration, or dependencies. Refer to the scripts, original run configurations, and manuscript for the exact experimental procedure.

## Citation and responsible reuse

If you use CRAT results or assets in scholarly work, cite the associated CRAT manuscript when its bibliographic details are available. To make an analysis auditable, additionally record the **repository URL, commit identifier, and specific supplementary document or figure/table** used.

Model weights, benchmark datasets, evaluation tools, and any third-party components remain subject to their respective licenses and terms. Check the repository's license information, if provided, before assuming permission to redistribute or modify its contents.
