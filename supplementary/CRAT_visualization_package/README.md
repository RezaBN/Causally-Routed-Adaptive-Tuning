# CRAT publication figures and tables

Run one main program to reproduce the **15 figures and 29 displayed tables** from the baseline comparison, ablation, sensitivity, and integrated analyses. The integrated analysis includes the CRAT-without-KL resource and quality figure. All 27 original CSV files are included in `data/` without modification.

The program changes presentation, not the experimental results. It recomputes differences and resource percentages from primary measurements, exports the reconciliation tables, and preserves missing measurements.

## Quick start

Extract the entire ZIP, open a terminal in the extracted folder, and run:

```bash
python -m pip install -r requirements.txt
python crat_visualize.py
```

Use Python 3.11–3.13; the package was tested with Python 3.12. No GPU, model checkpoint, PyTorch, LaTeX installation, Word, or internet connection is needed after the dependencies are installed. Keep the helper `.py` files and the two catalog `.json` files beside `crat_visualize.py`.

The default run writes to `output/`. It generates vector **PDF and SVG**, **600-dpi PNG**, and editable table **CSV, LaTeX and HTML** files. Descriptions are separate UTF-8 `.txt` files. A fresh run replaces generated files with the same names; it never changes `data/`. Prefer a new `--output-dir` when comparing styles.

## Default appearance

| Setting | Default |
|---|---:|
| Font family | Times New Roman |
| Axis labels and body text | 7 pt |
| Tick labels | 6 pt |
| Legends and annotations | 6.5 pt |
| Short panel labels | 8 pt |
| Visual table text | 7 pt |
| Figure and table width | 7 inches |
| Figure height | A separate aspect ratio for each figure |
| Raster resolution | 600 dpi |
| Color palette | Colorblind-oriented categorical palette |

“Times New Roman” is the standard font-family name corresponding to your request. If it is unavailable, the program prints a notice and uses DejaVu Serif. The actual font is recorded in `run_manifest.json`. It never silently claims to have used Times New Roman. Font files are not bundled.

To require an installed Times New Roman font:

```bash
python crat_visualize.py --font-family "Times New Roman" --strict-font
```

To register font files yourself, repeat `--font-file` for regular/bold/italic faces as needed. Example on Windows, if these files exist:

```bash
python crat_visualize.py --font-family "Times New Roman" --font-file "C:/Windows/Fonts/times.ttf" --font-file "C:/Windows/Fonts/timesbd.ttf" --strict-font
```

On macOS or Linux, provide the corresponding local TTF/OTF paths. The PDF embeds the font used; default SVG files outline text to preserve its appearance on another computer. Use `--svg-text` if you prefer editable SVG text and can ensure the font is installed where it will be opened. LaTeX fragments inherit the typeface of your paper.

## Change font sizes, colors and dimensions

This command changes the base font size and figure width. Related font defaults shift with the base size unless explicitly overridden:

```bash
python crat_visualize.py --font-family "Times New Roman" --font-size 8 --figure-width 7.2 --output-dir output_larger
```

For independent font controls:

```bash
python crat_visualize.py --font-size 7 --tick-font-size 6 --legend-font-size 6.5 --annotation-font-size 6.5 --panel-font-size 8 --table-font-size 7
```

Set an exact figure size, in inches:

```bash
python crat_visualize.py --only "integrated/Figure_4*" --figsize 7.2 2.8 --output-dir output_no_kl
```

`--figure-width` preserves each recipe's aspect ratio. `--figsize WIDTH HEIGHT` or `--figure-height` overrides that behavior. Files retain the requested canvas dimensions; there is no automatic tight crop that changes the final physical size. Very small dimensions can crowd labels; review the export and any `layout_notices.txt` entries.

Choose grayscale or a custom categorical cycle:

```bash
python crat_visualize.py --palette grayscale --output-dir output_grayscale
python crat_visualize.py --colors "#999999" "#8C564B" "#666666" "#0072B2" "#CC79A7" "#D69F00" "#009E73" --crat-color "#006B55" --output-dir output_custom
```

Custom cycles set method, pair, role and resource colors. For precise colors of particular elements, use the `colors` mapping in a JSON configuration instead. Three colors are the minimum; seven give distinct colors for all seven comparison methods. Method symbols remain distinct when using grayscale.

Change export formats or raster resolution:

```bash
python crat_visualize.py --formats pdf svg png tiff --dpi 600
```

PDF and SVG are vector graphics and remain sharp when resized. DPI controls PNG/TIFF resolution. Prefer the vector PDF for submission; PNG is convenient for Word and previews. Do not shrink a figure after insertion if doing so would make the 6–8 pt text too small.

## Per-figure settings through JSON

An editable `style_default.json` is included. Generate another copy with:

```bash
python crat_visualize.py --write-config my_style.json
python crat_visualize.py --config my_style.json
```

The example `examples/style_custom.json` demonstrates exact colors, font sizes and individual dimensions. For example:

```json
{
  "font_family": "Times New Roman",
  "font_size": 7,
  "tick_font_size": 6,
  "legend_font_size": 6.5,
  "annotation_font_size": 6.5,
  "panel_font_size": 8,
  "table_font_size": 7,
  "figure_sizes": {
    "integrated/Figure_4_CRAT_without_KL": [7.2, 2.8],
    "sensitivity/Figure_3_topk_quality": [7.2, 5.0]
  },
  "colors": {
    "methods": {"CRAT": "#006B55", "Targeted SFT": "#606060"},
    "memory": "#367C64",
    "training_time": "#8471A8",
    "targeted": "#226888",
    "lora": "#A55337",
    "negative": "#A95436",
    "neutral": "#FAFAF7",
    "positive": "#217B68"
  }
}
```

Other color mappings include `pairs`, `roles`, `sequential_low`, `sequential_high`, `grid`, `text`, `reference`, `table_header`, `table_band`, `table_highlight`, and `table_border`. Command-line flags override JSON settings. Per-ID `figure_sizes` take precedence over the global width/height. `table_sizes` accepts the same `[width, height]` format, but a height too small for the text raises a clear error instead of silently reducing the font.

## Select a section or individual outputs

```bash
python crat_visualize.py --list
python crat_visualize.py --sections comparison
python crat_visualize.py --sections ablation sensitivity
python crat_visualize.py --sections integrated
python crat_visualize.py --kind figures
python crat_visualize.py --kind tables
python crat_visualize.py --only "integrated/Figure_4*" "comparison/Table_1*"
```

Quote wildcard selections so the terminal does not expand them. `--only` filters output IDs, not CSV rows: no model, language pair, metric or uncertainty values are dropped from the selected item. Ablation processing also loads the six comparison CSVs to reconcile the displayed reference. Integrated outputs load all three analyses. Other section selections load only their dependencies.

| Section | Figures | Tables |
|---|---:|---:|
| `comparison` | 4 | 8 |
| `ablation` | 3 | 7 |
| `sensitivity` | 4 | 10 |
| `integrated` | 4 | 4 |
| Total | 15 | 29 |

The integrated figures intentionally repeat the corresponding comparison, ablation and sensitivity evidence where the paper does so. The standalone analysis numbering and appendix numbering are retained in the captions and file IDs.

## Outputs and descriptions

Within each selected section:

- `figures/`: figure PDF, SVG and raster exports.
- `tables/`: visual tables, formatted CSV tables, editable HTML and LaTeX fragments.
- `data_tables/`: normalized and derived numeric tables at full available precision.
- `descriptions/`: one `.txt` caption/notes file per figure or displayed table.

At the output root:

- `all_descriptions.txt`: every caption and note from the current run in one text file.
- `run_manifest.json`: source hashes, selected output IDs, dimensions, actual font, exact colors and library versions.
- `data_notes.txt`: key measurement and comparison limitations.
- `layout_notices.txt`: detected text near or beyond a canvas edge, if any.

**No overall title, caption, interpretive paragraph or explanatory footnote is drawn inside a figure or visual table.** Necessary axis labels, metric/pair panel labels, data values, legends, reference markers and table headers remain. For example, the no-KL quality plots show a zero reference line; its interpretation and the unmatched-budget caveat are in the separate caption.

Tables retain reported values and precision conventions used in the sections. In table CSVs, a language-pair or parameter band appears as a label row with blank remaining cells. `NR` means unreported. Use `data_tables/` for full-precision numerical analysis rather than parsing rounded display tables. Long tables can be paginated with `--table-rows-per-page`; group labels and column headers repeat as needed.

For LaTeX, include `\usepackage{booktabs}` and insert the generated fragment with `\input{...}`. Supply its caption separately from the `.txt` file. The fragments set the selected point size but deliberately do not force your manuscript's font family.

## CSV inputs and scientific conventions

`data/` contains six main-comparison CSVs, six ablation CSVs and fifteen sensitivity CSVs. The code reads their supplied schemas, including the different column names in the Mistral EN-FR comparison export. It does not need a Word file or any previously generated table. A `source_checksums.json` identifies the original files.

To use another directory containing the same filenames and schemas:

```bash
python crat_visualize.py --data-dir /path/to/csv_files --output-dir /path/to/new_output
```

This is a reproducer for the supplied experiment design: the seven methods/variants, two main-comparison backbones, three language pairs and named sensitivity sweeps are explicit in the code. It is not a general schema-free CSV plotter. Changing the study design requires updating those definitions. If substituting new results, recheck captions and experimental metadata; statements about seeds, training budgets, defaults and reference provenance are properties of these experiments, not facts inferred automatically from arbitrary scores.

The numerical conventions are:

- Main-comparison quality differences use CRAT minus the baseline; the strongest baseline is selected separately by setting and metric.
- Cost ratios use CRAT divided by baseline. Ablation savings use `100 × (1 − ablation/full)`.
- Ablation quality-decrease heatmaps use displayed full mean minus ablated mean. Recomputing these differences resolves arithmetic, not reference provenance.
- Macro quality weights the six model-pair settings equally, separately for each metric. Ranks are descriptive.
- Error bars are reported SDs. No missing SD, paired covariance, confidence interval, p-value or seed observation is generated.
- Sensitivity ranges use either the complete grid or the intersection across pairs, as specified in each caption. Differences stay within the named sweep.
- Sensitivity COMET uses the equally weighted directional average, not the pooled field. Missing EN-ZH COMET values stay missing.
- Peak memory is labeled GiB according to the supplied implementation. Training duration and generation latency retain their different measurement scopes.
- No-KL quality comparisons with Mistral baselines remain contextual because the workloads differ and auxiliary provenance is unresolved.

Known delta discrepancies are saved for review; they are not substituted for primary measurements. Duplicate rows, missing required files and inconsistent source schemas cause validation errors.

## Source files

- `crat_visualize.py`: main command, configuration, fonts, table rendering, exports and manifests.
- `crat_data.py`: original-CSV normalization, derived quantities and reconciliation.
- `crat_plots.py`: the 15 figure recipes.
- `crat_tables.py`: the 29 table definitions.
- `captions.json`: figure captions, written out as text during a run.
- `table_catalog.json`: table IDs.

The included `example_output/` contains a complete verified default run. Its actual font is DejaVu Serif because Times New Roman was unavailable in the validation environment. Your run will use Times New Roman when it is installed or supplied.
