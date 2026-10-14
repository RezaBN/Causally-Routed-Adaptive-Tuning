"""Recompute the 29 paper tables from the normalized CSV measurements.

Captions and notes are collected separately from the rendered table bodies.
Numeric formatting follows the corresponding research sections.
"""
import pandas as pd
import numpy as np

PAIRS=['EN-FR','EN-FA','EN-ZH']
METHODS=['Base','Full SFT','Targeted SFT','LoRA','rsLoRA','DoRA','CRAT']
VARIANTS=['CRAT_full','no_kl','no_causal','no_morph','no_sparse','no_integration','static_role']
ROLES=['source','indicator','position','integration']
METRICS=['BLEU','chrF','COMET']
PARAMS=['lambda_kl','lambda_causal','lambda_morph','lambda_sparse','kl_temperature','integration_rank','heads_per_direction','top_k']
DEFAULT={'lambda_kl':.1,'lambda_causal':.1,'lambda_morph':.1,'lambda_sparse':.002,'kl_temperature':2.,'integration_rank':16.,'heads_per_direction':64.,'top_k':2.}
LABELS={'lambda_kl':'KL coefficient','lambda_causal':'Causal coefficient','lambda_morph':'Morphology coefficient','lambda_sparse':'Sparsity coefficient','kl_temperature':'KL temperature','integration_rank':'Integration rank','heads_per_direction':'Heads per direction','top_k':'Evaluation top k'}


def comparison_tables(data,b):
    D=data['comparison']['D'];M=data['comparison']['M'];A=data['comparison']['A']
    def table(headers,rows,widths,bands=None,boldcells=None,cratrows=None,**unused):
        return b.table(headers,rows,widths,bands,cratrows,boldcells)

    caption=b.caption
    note=b.note

    def macro():
        caption('Table 1. Unweighted macro averages over the six model-pair settings and descriptive ranks.')
        rows = []
        for m in METHODS:
            r = M.loc[m]
            rows.append([m, f'{r.BLEU:.3f}', f'{r.chrF:.3f}', f'{r.COMET:.5f}', f'{r.mean_rank:.2f}', str(int(r.first_place_cells))])
        table(['Method', 'BLEU', 'chrF', 'COMET', 'Mean\nrank', 'First\nplaces'], rows, [1.66, 1.05, 1.05, 1.15, 1.0, 1.1], boldcells={(6, c) for c in range(1, 6)}, cratrows={6})
        note('Higher is better for quality and first-place count; lower is better for mean rank. Rank averages cover 18 metric-setting cells. Macro scores are arithmetic averages of reported means, not pooled corpus scores. Bold marks the best value.')

    def quality(model, num):
        caption(f'Table {num}. {model} 7B translation quality on OPUS100. Values are mean ± reported SD over six seeds.')
        rows = []
        bands = set()
        bold = set()
        cr = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            d = D[(D.model == model) & (D.pair == pair)].set_index('method')
            for m in METHODS:
                ri = len(rows)
                r = d.loc[m]
                label = m + (' †' if model == 'Llama 2' and pair == 'EN-FR' and (m == 'CRAT') else '')
                rows.append([label] + [f"{r[k]:.{n}f} ± {r[k + '_sd']:.{n}f}" for k, n in [('BLEU', 3), ('chrF', 3), ('COMET', 5)]])
                if m == 'CRAT':
                    cr.add(ri)
                for ci, k in enumerate(['BLEU', 'chrF', 'COMET'], 1):
                    if r[k] == d[k].max():
                        bold.add((ri, ci))
        table(['Method', 'BLEU', 'chrF', 'COMET'], rows, [1.48, 1.75, 1.75, 2.08], bands, bold, cr, font=10.5)
        note('Higher is better. Bold marks the highest unrounded mean within each pair and metric; shaded rows identify CRAT. Rounded scores can appear tied even when the unrounded ranking differs. SD is not a confidence interval.')
        if model == 'Llama 2':
            note('† Llama 2 EN-FR CRAT used 158 updates and 1,582,308 tokens; the adapted baselines used 316 updates and 3,164,616 tokens. This setting is not matched for training workload.')

    def training(model, num):
        caption(f'Table S{num}. {model} 7B trainable parameters, peak memory, and reported training cost.')
        rows = []
        bands = set()
        cr = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            d = D[(D.model == model) & (D.pair == pair)].set_index('method')
            for m in METHODS:
                r = d.loc[m]
                if m == 'CRAT':
                    cr.add(len(rows))
                label = m + (' †' if model == 'Llama 2' and pair == 'EN-FR' and (m == 'CRAT') else '')
                rows.append([label, f'{r.trainable_params / 1000000.0:,.2f}', f'{r.trainable_pct:.2f}', f'{r.peak_vram_GB:.2f}', f'{r.train_seconds / 60:.2f}', f'{r.train_tokens_per_second:,.1f}', str(int(r.updates))])
        table(['Method', 'Trainable\n(M)', 'Trainable\n(%)', 'Peak\nVRAM*', 'Train\n(min)', 'Train\n(tokens/s)', 'Updates'], rows, [1.42, 1.0, 0.9, 0.82, 0.86, 1.09, 0.87], bands, cratrows=cr, font=10)
        note("M denotes millions of parameters. Trainable percentage uses each method's registered total parameter count. Resource entries are reported means; available resource SDs are retained in the companion CSV. Base training entries are zero because no adaptation is performed.")
        note('* The source field is named peak_vram_GB; the supplied implementation uses 1,073,741,824 bytes per GiB. It records the peak over the method run, not an isolated inference-memory measurement. Numerical values are unchanged.')
        if model == 'Llama 2':
            note('† EN-FR CRAT has half the updates and training tokens of every adapted baseline. Other Llama 2 pairs have 312 updates. The effective batch size field is 128 for all rows.')
        else:
            note('All adapted EN-FR and EN-ZH rows have 316 updates; EN-FA rows have 312. The effective batch size field is 128 for all rows. The Mistral EN-FR export additionally reports resource SDs.')

    def inference(model, num):
        caption(f'Table S{num}. {model} 7B generation duration, output volume, throughput, and latency.')
        rows = []
        bands = set()
        cr = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            d = D[(D.model == model) & (D.pair == pair)].set_index('method')
            for m in METHODS:
                r = d.loc[m]
                if m == 'CRAT':
                    cr.add(len(rows))
                count = f'{r.generated_tokens:,.1f}' if r.generated_tokens % 1 else f'{r.generated_tokens:,.0f}'
                rows.append([m, f'{r.generation_seconds:,.1f}', count, f'{r.generation_tokens_per_second:.1f}', f'{r.latency_mean_seconds:.3f}', f'{r.latency_p50_seconds:.3f}', f'{r.latency_p95_seconds:.3f}', f'{r.benchmark_tokens_per_second:.1f}'])
        table(['Method', 'Gen.\n(s)', 'Generated\ntokens', 'Batch\ntokens/s', 'Mean\n(s)', 'p50\n(s)', 'p95\n(s)', 'Single\ntokens/s'], rows, [1.32, 0.79, 0.98, 0.82, 0.82, 0.72, 0.72, 0.79], bands, cratrows=cr, font=9.8)
        note('Gen. is total timed generation duration over the evaluation workload. Batch tokens/s is generation_tokens_per_second. Mean, p50, p95, and Single tokens/s come from the separate single-example latency benchmark. They are not alternative summaries of the batched workload. All entries are the reported means; p50 and p95 are means of run-level percentile statistics when aggregated across seeds.')
        note('The supplied generate_batch timer starts after tokenization, source features, and CRAT router computation. These timings exclude that setup work. Different output token counts also affect raw duration; tokenizer differences limit cross-backbone interpretation of tokens/s.')

    def audit():
        caption('Table S5. Reconciliation of the Mistral EN-FA delta_chrF_vs_CRAT column.')
        rows = []
        for _, r in A.iterrows():
            rows.append([r['method'], f'{r.reported_value:+.6f}', f'{r.recomputed_baseline_minus_CRAT:+.6f}'])
        table(['Method', 'Supplied delta', 'Recomputed delta'], rows, [2.56, 2.25, 2.25], font=10.5)
        note('Both delta columns in this reconciliation use baseline chrF minus CRAT chrF. A negative value therefore favors CRAT. The primary chrF means were retained. For example, 27.09303414 minus 27.17611266 equals -0.08307852 for Targeted SFT; the supplied +0.077451539 has the wrong sign. All six non-CRAT chrF differences in this file fail this reconciliation. Other supplied additive quality deltas reconcile within 0.000001.')

    b.begin('Table_1_macro')
    macro()

    b.begin('Table_2_quality_llama2')
    quality('Llama 2',2)

    b.begin('Table_3_quality_mistral')
    quality('Mistral',3)

    b.begin('Table_S1_training_llama2')
    training('Llama 2',1)

    b.begin('Table_S2_training_mistral')
    training('Mistral',2)

    b.begin('Table_S3_inference_llama2')
    inference('Llama 2',3)

    b.begin('Table_S4_inference_mistral')
    inference('Mistral',4)

    b.begin('Table_S5_delta_reconciliation')
    audit()


def ablation_tables(data,b):
    Q=data['ablation']['Q'];C=data['ablation']['C'];R=data['ablation']['R'];A=data['ablation']['A'];REF=data['ablation']['REF']
    def table(headers,rows,widths,bands=None,fullrows=None,boldcells=None,**unused):
        return b.table(headers,rows,widths,bands,fullrows,boldcells)

    caption=b.caption
    note=b.note

    def quality():
        caption('Table 1. Translation quality for all ablations. Values are reported mean ± SD.')
        rows = []
        bands = set()
        full = set()
        bold = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            for v in VARIANTS:
                rr = Q.loc[pair, v]
                ri = len(rows)
                rows.append([v] + [f"{rr[m]:.{digits}f} ± {rr[m + '_sd']:.{digits}f}" for m, digits in [('BLEU', 3), ('chrF', 3), ('COMET', 5)]])
                if v == 'CRAT_full':
                    full.add(ri)
                    bold.update(((ri, ci) for ci in [1, 2, 3]))
        table(['Variant', 'BLEU', 'chrF', 'COMET'], rows, [1.46, 1.8, 1.8, 2.0], bands, full, bold, font=10.5)
        note('Higher is better. Bold values identify the highest displayed mean within each pair and metric. SD is the reported standard deviation, not a confidence interval. The full-model reference discrepancy in Appendix A applies to all comparisons with CRAT_full; the ablation seed count and model identifier are not encoded in these CSVs.')

    def training():
        caption('Table 2. Trainable parameters and training resource measurements.')
        rows = []
        bands = set()
        full = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            for v in VARIANTS:
                rr = C.loc[pair, v]
                ri = len(rows)
                rows.append([v, f'{rr.trainable_params / 1000000.0:.3f}', f'{rr.trainable_pct:.3f}', f'{rr.peak_vram_GB:.3f}', f'{rr.train_seconds / 60:.2f}', f'{rr.train_tokens_per_second:,.1f}', f'{rr.train_time_reduction_pct_vs_CRAT:+.2f}'])
                if v == 'CRAT_full':
                    full.add(ri)
        table(['Variant', 'Trainable\n(M)', 'Trainable\n(%)', 'Peak\n(GiB)', 'Training\n(min)', 'Training\ntokens/s', 'Time saved\n(%)'], rows, [1.35, 0.88, 0.9, 0.85, 0.9, 1.12, 1.06], bands, full, font=10.1)
        note('M denotes millions. Trainable percentage uses the registered total parameter count of each variant. Positive time savings indicate a shorter duration relative to the displayed CRAT_full resource row; negative values indicate a longer duration. These percentages have been recomputed from primary measurements.')
        note('The source field peak_vram_GB is labeled here in GiB because the supplied implementation divides allocated bytes by 2 to the power 30. It is a method-run peak, not an isolated inference-memory measurement. The training timer excludes earlier discovery and preparation and includes validation in the supplied main scripts. The ablation runner must confirm that it uses the same measurement boundaries.')

    def inference():
        caption('Table 3. Generation duration, output volume, throughput, and single-example latency.')
        rows = []
        bands = set()
        full = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            for v in VARIANTS:
                rr = C.loc[pair, v]
                ri = len(rows)
                rows.append([v, f'{rr.generation_seconds:.1f}', f'{rr.generated_tokens:,.0f}', f'{rr.generation_tokens_per_second:.2f}', f'{rr.latency_mean_seconds:.3f}', f'{rr.latency_p50_seconds:.3f}', f'{rr.latency_p95_seconds:.3f}', f'{rr.benchmark_tokens_per_second:.2f}'])
                if v == 'CRAT_full':
                    full.add(ri)
        table(['Variant', 'Gen.\n(s)', 'Output\ntokens', 'Batch\ntokens/s', 'Mean\n(s)', 'p50\n(s)', 'p95\n(s)', 'Single\ntokens/s'], rows, [1.24, 0.72, 0.91, 0.82, 0.82, 0.75, 0.82, 0.98], bands, full, font=10)
        note('Batch tokens/s comes from the complete evaluation generation workload. Mean, p50, p95, and Single tokens/s come from the separate single-example latency benchmark in the supplied implementation. p50 and p95 denote the reported 50th and 95th percentiles; the CSVs do not specify whether these are pooled percentiles or aggregates of per-run percentiles. Lower duration and latency, and higher throughput, are preferable.')

    def routing():
        caption('Table 4. Evaluation mean soft gates and diagnostic threshold counts.')
        rows = []
        bands = set()
        full = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            for v in VARIANTS[:-1]:
                rr = R.loc[pair, v]
                ri = len(rows)
                pattern = ' / '.join((f"{rr['eval_active_' + role]:g}" for role in ROLES))
                rows.append([v] + [f"{rr['eval_gate_mean_' + role]:.4f}" for role in ROLES] + [pattern, f'{rr.eval_soft_threshold_count:.1f}'])
                if v == 'CRAT_full':
                    full.add(ri)
        table(['Variant', 'Source', 'Indicator', 'Position', 'Integration', 'Active fractions\nS / I / P / G', 'Mean count\nabove 0.5'], rows, [1.3, 0.9, 0.9, 0.9, 0.9, 1.25, 0.91], bands, full, font=10)
        note('S, I, P, and G identify source, indicator, position, and integration. Active fractions are the reported proportions of soft gates strictly greater than 0.5, in that order. The last column is their sum. It does not measure actual top-k execution. Static_role has missing routing diagnostics for all pairs and is omitted from this table; missing values have not been imputed.')

    def references():
        caption('Table A1. Displayed and implied CRAT_full quality references.')
        rows = []
        for pair in PAIRS:
            rr = REF.loc[pair]
            for m in ['BLEU', 'chrF', 'COMET']:
                shown = rr[m + '_displayed_full']
                implied = rr[m + '_implied_reference']
                rows.append([pair, m, f'{shown:.6f}', f'{implied:.6f}', f'{shown - implied:+.6f}'])
        table(['Pair', 'Metric', 'Displayed full', 'Implied reference', 'Displayed minus\nimplied'], rows, [1.04, 0.88, 1.65, 1.72, 1.77], font=10.4)
        note('Implied reference is the mean across the six reconstructed reference values. It is presented only to diagnose the mismatch. No replacement of the displayed primary means has been made.')

    def sign_flips():
        caption('Table A2. Quality comparisons whose sign changes after recomputation.')
        rows = []
        for _, rr in A[A.sign_flip].iterrows():
            rows.append([rr['pair'], rr['ablation'], rr['metric'], f'{rr.source_delta:+.6f}', f'{rr.recomputed_delta:+.6f}'])
        table(['Pair', 'Variant', 'Metric', 'Supplied delta', 'Recomputed delta'], rows, [1.0, 1.6, 0.9, 1.78, 1.78], font=10.3)
        note('Here both deltas use ablated mean minus full-model mean, matching the source delta-column convention. Positive values favor the ablation. Figure 1 instead shows score decreases, using the opposite sign.')

    def train_routing():
        caption('Table B1. Training mean soft gates and the logged gate statistic.')
        rows = []
        bands = set()
        full = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            for v in VARIANTS[:-1]:
                rr = R.loc[pair, v]
                ri = len(rows)
                rows.append([v] + [f"{rr['train_gate_mean_' + role]:.4f}" for role in ROLES] + [f'{rr.train_gate_statistic:.5f}'])
                if v == 'CRAT_full':
                    full.add(ri)
        table(['Variant', 'Source', 'Indicator', 'Position', 'Integration', 'Logged gate\nstatistic'], rows, [1.46, 0.98, 0.98, 0.98, 0.98, 1.68], bands, full, font=10.3)
        note('The final column preserves the numeric values of train_router_entropy, with a neutral label because the implementation does not compute categorical or full Bernoulli entropy. Static_role diagnostics are missing in every pair. Training means aggregate recorded training snapshots rather than only the selected final checkpoint.')

    b.begin('Table_1_quality')
    quality()

    b.begin('Table_2_training')
    training()

    b.begin('Table_3_inference')
    inference()

    b.begin('Table_4_evaluation_routing')
    routing()

    b.begin('Table_A1_quality_references')
    references()

    b.begin('Table_A2_delta_sign_reversals')
    sign_flips()

    b.begin('Table_B1_training_routing')
    train_routing()


def sensitivity_tables(data,b):
    D=data['sensitivity']['D'];R=data['sensitivity']['R'];DI=data['sensitivity']['DI']
    def table(headers,rows,widths,bands=None,refrows=None,**unused):
        return b.table(headers,rows,widths,bands,refrows,None)

    caption=b.caption
    note=b.note

    def fmt(value, digits=3):
        return 'NR' if pd.isna(value) else f'{value:.{digits}f}'

    def grid():
        caption('Table 1. Tested parameter grids and defaults in the supplied main implementations.')
        rows = []
        for p in PARAMS:
            grids = []
            for pair in ['EN-FR', 'EN-FA']:
                vals = D[(D.pair == pair) & (D.sensitivity_parameter == p)].sensitivity_value
                grids.append(', '.join((f'{v:g}' for v in vals)))
            rows.append([LABELS[p], f'{DEFAULT[p]:g}'] + grids)
        table(['Parameter', 'Code\ndefault', 'EN-FR values', 'EN-FA and EN-ZH values'], rows, [1.9, 0.8, 2.05, 2.31], font=10.4)
        note('EN-FA and EN-ZH use the same grids. The code-default sparsity coefficient, 0.002, is absent from the exports. Code defaults identify comparison values; they do not establish that all other configuration fields were identical across sweeps.')

    def ranges():
        caption('Table 2. Maximum minus minimum mean quality over each complete reported grid.')
        rows = []
        for p in PARAMS:
            row = [LABELS[p]]
            for pair in PAIRS:
                for m in ['BLEU', 'chrF']:
                    x = R[(R.pair == pair) & (R.parameter == p) & (R.metric == m)].iloc[0]
                    row.append(f"{x['range']:.3f}")
            rows.append(row)
        table(['Parameter', 'EN-FR\nBLEU', 'EN-FR\nchrF', 'EN-FA\nBLEU', 'EN-FA\nchrF', 'EN-ZH\nBLEU', 'EN-ZH\nchrF'], rows, [1.84, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87], font=10.2)
        note('A smaller range means less variation in the reported means on that grid. Grid widths and counts differ across pairs for several parameters. Figure 1 therefore repeats the comparison using only parameter values shared by all three pairs.')

    def heads():
        caption('Table 3. Quality and resource cost under different head-selection budgets.')
        rows = []
        bands = set()
        refs = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            for _, r in D[(D.pair == pair) & (D.sensitivity_parameter == 'heads_per_direction')].iterrows():
                if r.sensitivity_value == 64:
                    refs.add(len(rows))
                rows.append([f'{r.sensitivity_value:g}', f'{r.trainable_params_mean / 1000000.0:.3f}', fmt(r.BLEU_mean_mean), fmt(r.chrF_mean_mean), fmt(r.COMET_mean_mean, 5), f'{r.peak_vram_GB_mean:.3f}', f'{r.train_seconds_mean / 60:.2f}', f'{r.generation_tokens_per_second_mean:.2f}'])
        table(['Heads per\ndirection', 'Trainable\n(M)', 'BLEU', 'chrF', 'COMET', 'Peak\n(GiB)', 'Train\n(min)', 'Generation\ntokens/s'], rows, [0.77, 0.94, 0.79, 0.79, 1.03, 0.87, 0.85, 1.02], bands, refs, font=10.1)
        note('M denotes millions of trainable parameters. Shading identifies the within-sweep reference of 64 heads. NR means not reported. Resource entries are reported means; EN-FR quality variability is retained in Appendix A and Figure 2. Peak measurements in this table include the training-containing workflow.')

    def topk():
        caption('Table 4. Quality and inference measurements across evaluation top-k settings.')
        rows = []
        bands = set()
        refs = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            for _, r in D[(D.pair == pair) & (D.sensitivity_parameter == 'top_k')].iterrows():
                if r.sensitivity_value == 2:
                    refs.add(len(rows))
                rows.append([f'{r.sensitivity_value:g}', fmt(r.BLEU_mean_mean), fmt(r.chrF_mean_mean), fmt(r.COMET_mean_mean, 5), fmt(r.generation_tokens_per_second_mean, 2), fmt(r.latency_mean_seconds_mean), fmt(r.latency_p95_seconds_mean)])
        table(['Top k', 'BLEU', 'chrF', 'COMET', 'Generation\ntokens/s', 'Mean latency\n(s)', 'p95 latency\n(s)'], rows, [0.56, 0.92, 0.92, 1.12, 1.18, 1.18, 1.18], bands, refs, font=10.4)
        note('Shading identifies top k equal to 2. Generation throughput uses batched evaluation, whereas latency comes from the separate single-example benchmark in the supplied implementation. p95 is the reported 95th percentile, potentially aggregated over runs; no pooled latency distribution is supplied. The top-k rows contain no training measurements.')

    def directions():
        caption('Table 5. Selected directional differences relative to the corresponding within-sweep reference.')
        selected = [('EN-FA', 'lambda_kl', 0.5, 'KL 0.5 vs 0.1'), ('EN-FA', 'kl_temperature', 1, 'Temperature 1 vs 2'), ('EN-FA', 'heads_per_direction', 96, 'Heads 96 vs 64'), ('EN-ZH', 'heads_per_direction', 32, 'Heads 32 vs 64'), ('EN-FA', 'top_k', 1, 'Top k 1 vs 2'), ('EN-FA', 'top_k', 4, 'Top k 4 vs 2'), ('EN-ZH', 'top_k', 1, 'Top k 1 vs 2')]
        rows = []
        for pair, p, value, label in selected:
            r = DI[(DI.pair == pair) & (DI.parameter == p) & np.isclose(DI.value, value)].iloc[0]
            row = [pair, label]
            for col, digits in [('BLEU_forward_delta', 3), ('BLEU_reverse_delta', 3), ('COMET_forward_delta', 5), ('COMET_reverse_delta', 5)]:
                row.append('NR' if pd.isna(r[col]) else f'{r[col]:+.{digits}f}')
            rows.append(row)
        table(['Pair', 'Contrast', 'BLEU\nforward', 'BLEU\nreverse', 'COMET\nforward', 'COMET\nreverse'], rows, [0.83, 1.8, 1.0, 1.0, 1.215, 1.215], font=10.3)
        note('Forward means English to the paired language; reverse means the paired language to English. Positive differences indicate higher scores. These are differences of reported means, without significance testing. All directional scores and differences are retained in the companion CSVs.')

    def topk_routing():
        caption('Table 6. Selected roles and retained soft-gate mass for EN-FR and EN-FA.')
        rows = []
        bands = set()
        role_names = {1: 'Position', 2: 'Position and source', 3: 'Position, source, integration', 4: 'All four roles'}
        for pair in ['EN-FR', 'EN-FA']:
            bands.add(len(rows))
            rows.append([pair])
            for _, r in D[(D.pair == pair) & (D.sensitivity_parameter == 'top_k')].iterrows():
                rows.append([f'{r.sensitivity_value:g}', role_names[int(r.sensitivity_value)], f'{100 * r.topk_retained_gate_mass_fraction_mean:.2f}', f"{r['eval_mean_active_roles@0.5_mean']:.1f}", f'{r.topk_router_entropy_mean:.5f}'])
        table(['Top k', 'Selected roles', 'Retained gate\nmass (%)', 'Soft count\nabove 0.5', 'Logged gate\nstatistic'], rows, [0.6, 2.7, 1.36, 1.2, 1.2], bands, font=10, padding=30)
        note('Selected-role rates are exactly 0 or 1 at the exported precision, and their sum equals top k. Retained mass refers to soft-gate magnitudes, not categorical probabilities or fractions of model FLOPs. The last column preserves topk_router_entropy without asserting categorical entropy. Equivalent top-k selection and mass fields are absent for EN-ZH.')

    def quality(pair, num):
        caption(f'Table A{num}. Complete {pair} quality means' + (' and reported standard deviations.' if pair == 'EN-FR' else '.'))
        rows = []
        bands = set()
        refs = set()
        for p in PARAMS:
            bands.add(len(rows))
            rows.append([LABELS[p]])
            for _, r in D[(D.pair == pair) & (D.sensitivity_parameter == p)].iterrows():
                if np.isclose(r.sensitivity_value, DEFAULT[p]):
                    refs.add(len(rows))
                row = [f'{r.sensitivity_value:g}']
                for m, digits in [('BLEU', 3), ('chrF', 3), ('COMET', 5)]:
                    v = r[m + '_mean_mean']
                    sd = r.get(m + '_mean_std', np.nan)
                    row.append('NR' if pd.isna(v) else f'{v:.{digits}f} ± {sd:.{digits}f}' if pair == 'EN-FR' and pd.notna(sd) else f'{v:.{digits}f}')
                rows.append(row)
        table(['Value', 'BLEU', 'chrF', 'COMET'], rows, [0.9, 1.94, 1.94, 2.28], bands, refs, font=10.2, padding=30)
        note('Shading marks the corresponding code-default value when that row is present. Higher scores are better. All entries are primary reported means; COMET uses the equally weighted directional average. Comparisons are within the named sweep. Complete precision and direction-specific scores are preserved in the companion CSVs.')
        if pair == 'EN-FR':
            note('The ± quantities are reported standard deviations. The SD/SEM arithmetic implies two aggregated observations, whose identities are not specified; no six-seed assumption is made. The figures use ±1 SD, not the supplied nominal 95% confidence intervals.')
        elif pair == 'EN-FA':
            note('Variability is not reported. Non-top-k COMET rows cover 999 segments; top-k COMET rows cover 1,000. The displayed COMET column consistently uses COMET_mean_mean rather than the slightly different pooled COMET_mean.')
        else:
            note('NR means not reported: COMET is available only for the four top-k settings. Standard deviations and confidence intervals are absent from these EN-ZH exports. Missing values are not imputed.')

    def sparse_reference():
        caption('Table B1. References implied by the exported sparsity differences from default.')
        rows = []
        for pair in PAIRS:
            vals = []
            for m in ['BLEU', 'chrF']:
                r = R[(R.pair == pair) & (R.parameter == 'lambda_sparse') & (R.metric == m)].iloc[0]
                vals.append(f'{r.implied_default_from_min:.6f}')
            rows.append([pair, '0.002 absent'] + vals)
        table(['Pair', 'Code-default row', 'Implied BLEU reference', 'Implied chrF reference'], rows, [1.06, 1.65, 2.17, 2.18], font=10.5)
        note('Values are reconstructed from the reported extrema and default-delta columns, not measured rows supplied for 0.002. Their use is restricted to diagnosing the missing reference.')

    b.begin('Table_1_parameter_grids')
    grid()

    b.begin('Table_2_quality_ranges')
    ranges()

    b.begin('Table_3_head_budgets')
    heads()

    b.begin('Table_4_topk_quality_cost')
    topk()

    b.begin('Table_5_directional_differences')
    directions()

    b.begin('Table_6_topk_routing')
    topk_routing()

    b.begin('Table_A1_quality_en_fr')
    quality('EN-FR',1)

    b.begin('Table_A2_quality_en_fa')
    quality('EN-FA',2)

    b.begin('Table_A3_quality_en_zh')
    quality('EN-ZH',3)

    b.begin('Table_B1_sparsity_reference')
    sparse_reference()


def integrated_tables(data,b):
    D=data['comparison']['D'];S=data['sensitivity']['D'];R=data['sensitivity']['common']
    def table(headers,rows,widths,bands=None,refs=None,bold=None,**unused):
        return b.table(headers,rows,widths,bands,refs,bold)

    caption=b.caption
    note=b.note

    def macro():
        caption('Table 1. Macro-average quality and descriptive ranks across six model-pair settings.')
        means = D.groupby('method', observed=True)[METRICS].mean()
        ranked = []
        for _, g in D.groupby(['model', 'pair'], observed=True):
            for metric in METRICS:
                ranks = g[metric].rank(ascending=False, method='min')
                ranked.extend([{'method': m, 'rank': r} for m, r in zip(g.method, ranks)])
        ranks = pd.DataFrame(ranked)
        rows = []
        for m in METHODS:
            rr = ranks[ranks.method == m]['rank']
            r = means.loc[m]
            rows.append([m, f'{r.BLEU:.3f}', f'{r.chrF:.3f}', f'{r.COMET:.5f}', f'{rr.mean():.2f}', f'{int((rr == 1).sum())}'])
        table(['Method', 'BLEU', 'chrF', 'COMET', 'Mean rank', 'First places'], rows, [1.4, 1.12, 1.12, 1.3, 1.06, 1.0], refs={6}, bold={(6, j) for j in range(6)}, font=10.5)
        note('Macro means weight the six settings equally within each metric. Rank averages cover 18 metric-setting comparisons; lower mean rank is better. The metrics are correlated. Bold identifies the best aggregate values. Individual means and SDs appear in Supplementary Analysis A, Tables 2 and 3.')

    def resources():
        caption('Table 2. CRAT parameter counts and measured costs in the main comparison.')
        rows = []
        bands = set()
        for model in ['Llama 2', 'Mistral']:
            bands.add(len(rows))
            rows.append([model + ' 7B'])
            for pair in PAIRS:
                r = D[(D.model == model) & (D.pair == pair) & (D.method == 'CRAT')].iloc[0]
                rows.append([pair + (' †' if model == 'Llama 2' and pair == 'EN-FR' else ''), f'{r.trainable_params / 1000000.0:.2f}', f'{r.peak_vram_GB:.2f}', f'{r.train_seconds / 60:.2f}', f'{r.generation_tokens_per_second:.2f}', f'{r.latency_mean_seconds:.3f}', f'{r.latency_p95_seconds:.3f}'])
        table(['Pair', 'Trainable\n(M)', 'Peak\n(GiB)', 'Training\n(min)', 'Generation\ntokens/s', 'Mean latency\n(s)', 'p95 latency\n(s)'], rows, [0.9, 1.0, 0.85, 0.85, 1.1, 1.1, 1.2], bands, font=10.1)
        note('M denotes millions of parameters. Peak memory covers the method run and uses 2³⁰ bytes per GiB. Generation throughput is measured in batches; latency comes from a separate single-example benchmark. Entries are reported means, including means of run-level p95 values where aggregated. † CRAT uses half the baseline updates and training tokens in Llama 2 EN-FR.')

    def ranges():
        caption('Table 3. Quality ranges on parameter grids shared by all three language pairs.')
        params = ['lambda_kl', 'lambda_causal', 'lambda_morph', 'lambda_sparse', 'kl_temperature', 'integration_rank', 'heads_per_direction', 'top_k']
        labels = ['KL coefficient', 'Causal-prior coefficient', 'Morphology coefficient', 'Sparsity coefficient', 'KL temperature', 'Integration rank', 'Heads per direction', 'Evaluation top *k*']
        rows = []
        for p, label in zip(params, labels):
            row = [label]
            for pair in PAIRS:
                r = R[(R.pair == pair) & (R.parameter == p)].iloc[0]
                row.extend([f'{r.BLEU_range_on_common_grid:.3f}', f'{r.chrF_range_on_common_grid:.3f}'])
            rows.append(row)
        table(['Parameter', 'EN-FR\nBLEU', 'EN-FR\nchrF', 'EN-FA\nBLEU', 'EN-FA\nchrF', 'EN-ZH\nBLEU', 'EN-ZH\nchrF'], rows, [1.78, 0.87, 0.87, 0.87, 0.87, 0.87, 0.87], font=10.1)
        note('Each range is the maximum minus minimum of reported means. Common values are 0.05/0.1/0.2 for KL, causal-prior, and morphology coefficients; 0.001/0.005/0.01 for sparsity; 2/4/8 for temperature; 8/16/32 for rank; 32/64/96 for head budget; and 1/2/3/4 for top *k*. A smaller range indicates less variation over that grid, not stronger statistical evidence of robustness.')

    def heads():
        caption('Table 4. Quality and resource costs across head-selection budgets in the sensitivity experiments.')
        rows = []
        bands = set()
        refs = set()
        for pair in PAIRS:
            bands.add(len(rows))
            rows.append([pair])
            for _, r in S[(S.pair == pair) & (S.sensitivity_parameter == 'heads_per_direction')].sort_values('sensitivity_value').iterrows():
                if r.sensitivity_value == 64:
                    refs.add(len(rows))
                rows.append([f'{r.sensitivity_value:g}', f'{r.trainable_params_mean / 1000000.0:.3f}', f'{r.BLEU_mean_mean:.3f}', f'{r.chrF_mean_mean:.3f}', 'NR' if pd.isna(r.COMET_mean_mean) else f'{r.COMET_mean_mean:.5f}', f'{r.peak_vram_GB_mean:.3f}', f'{r.train_seconds_mean / 60:.2f}'])
        table(['Heads per\ndirection', 'Trainable\n(M)', 'BLEU', 'chrF', 'COMET', 'Peak\n(GiB)', 'Training\n(min)'], rows, [0.8, 1.05, 0.85, 0.85, 1.2, 1.05, 1.2], bands, refs, font=10.2)
        note('Entries are reported means; shading identifies the within-sweep reference of 64 heads. NR denotes unreported COMET values. These references are specific to the head-budget sweep and are not interchangeable with main-comparison or coefficient-sweep defaults. Uncertainty estimates are unavailable for EN-FA and EN-ZH.')

    b.begin('Table_1_macro')
    macro()

    b.begin('Table_2_CRAT_resources')
    resources()

    b.begin('Table_3_common_grid_ranges')
    ranges()

    b.begin('Table_4_head_budgets')
    heads()
