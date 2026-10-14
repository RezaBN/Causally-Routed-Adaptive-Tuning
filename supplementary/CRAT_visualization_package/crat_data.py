"""CSV normalization and arithmetic used by crat_visualize.py.

No training, network access, or language-model libraries are required.
Quality differences are recomputed from primary means. The supplied deltas
are retained only for reconciliation. Missing measurements stay missing.
"""
from pathlib import Path
import hashlib
import math
import numpy as np
import pandas as pd

def comparison_data(SRC, OUT):
    TAB = OUT / 'data_tables'
    TAB.mkdir(parents=True, exist_ok=True)
    METHODS = ['Base', 'Full SFT', 'Targeted SFT', 'LoRA', 'rsLoRA', 'DoRA', 'CRAT']
    MODELS = ['Llama 2', 'Mistral']
    PAIRS = ['EN-FR', 'EN-FA', 'EN-ZH']
    METRICS = ['BLEU', 'chrF', 'COMET']
    ALIASES = {'Base Llama-2-7B': 'Base', 'Base Mistral-7B': 'Base', 'Base Mistral': 'Base', 'base': 'Base', 'full_sft': 'Full SFT', 'Full-SFT': 'Full SFT', 'Targeted head-SFT': 'Targeted SFT', 'targeted_sft': 'Targeted SFT', 'lora': 'LoRA', 'RSLoRA': 'rsLoRA', 'rslora': 'rsLoRA', 'dora': 'DoRA', 'crat': 'CRAT'}
    records, source_deltas = ([], [])
    for path in sorted(SRC.glob('*main-comparison.csv')):
        raw = pd.read_csv(path)
        if not (len(raw) == 7 and raw.method.nunique() == 7):
            raise ValueError(f'CSV validation failed: len(raw) == 7 and raw.method.nunique() == 7; context: {path}')
        model = 'Llama 2' if 'LlaMA2' in path.name else 'Mistral'
        pair = path.name[:5]
        for _, row in raw.iterrows():
            rec = {'model': model, 'pair': pair, 'method': ALIASES.get(row.method, row.method), 'n_seeds': 6, 'source_file': path.name, 'source_method': row.method}
            for key, val in row.items():
                if key == 'method':
                    continue
                if key.endswith('_std_across_seeds'):
                    key = key[:-len('_std_across_seeds')] + '_sd'
                elif key.endswith('_mean_across_seeds'):
                    key = key[:-len('_mean_across_seeds')]
                for metric in METRICS:
                    if key == metric + '_mean_std':
                        key = metric + '_sd'
                    elif key == metric + '_mean':
                        key = metric
                rec[key] = float(val)
            records.append(rec)
    df = pd.DataFrame(records)
    if not (len(df) == 42 and (not df.duplicated(['model', 'pair', 'method']).any())):
        raise ValueError("CSV validation failed: len(df) == 42 and (not df.duplicated(['model', 'pair', 'method']).any())")
    primary = ['model', 'pair', 'method', 'n_seeds'] + sum(([m, m + '_sd'] for m in METRICS), []) + ['trainable_params', 'total_params', 'trainable_pct', 'peak_vram_GB', 'train_seconds', 'updates', 'effective_batch_size', 'train_tokens', 'train_tokens_per_second', 'generation_seconds', 'generated_tokens', 'generation_tokens_per_second', 'latency_mean_seconds', 'latency_p50_seconds', 'latency_p95_seconds', 'benchmark_tokens_per_second', 'source_file', 'source_method']
    primary += [c for c in df if c.endswith('_sd') and c not in primary and ('vs_CRAT' not in c)]
    if not df[primary[:10]].notna().all().all():
        raise ValueError('CSV validation failed: df[primary[:10]].notna().all().all()')
    for k in ['model', 'pair', 'method']:
        order = {'model': MODELS, 'pair': PAIRS, 'method': METHODS}[k]
        df[k] = pd.Categorical(df[k], categories=order, ordered=True)
    df = df.sort_values(['model', 'pair', 'method']).reset_index(drop=True)
    df[primary].to_csv(TAB / 'normalized_primary_results.csv', index=False, float_format=None)
    margins, contrasts, ranks, costs, anomalies = ([], [], [], [], [])
    for (model, pair), g in df.groupby(['model', 'pair'], observed=True):
        g = g.set_index('method')
        cr = g.loc['CRAT']
        for metric in METRICS:
            rank = g[metric].rank(method='min', ascending=False)
            for method in METHODS:
                ranks.append({'model': model, 'pair': pair, 'method': method, 'metric': metric, 'rank': int(rank[method])})
            others = g.loc[[m for m in METHODS if m != 'CRAT']]
            best = str(others[metric].idxmax())
            margins.append({'model': model, 'pair': pair, 'metric': metric, 'CRAT_mean': cr[metric], 'CRAT_sd': cr[metric + '_sd'], 'best_baseline': best, 'baseline_mean': g.loc[best, metric], 'baseline_sd': g.loc[best, metric + '_sd'], 'CRAT_minus_best': cr[metric] - g.loc[best, metric], 'CRAT_rank': int(rank['CRAT'])})
            for method in METHODS[:-1]:
                contrasts.append({'model': model, 'pair': pair, 'metric': metric, 'baseline': method, 'CRAT_minus_baseline': cr[metric] - g.loc[method, metric]})
            col = f'delta_{metric}_vs_CRAT'
            for method in METHODS:
                val = g.loc[method, col]
                recomputed = g.loc[method, metric] - cr[metric]
                err = val - recomputed
                if abs(err) > 1e-06:
                    anomalies.append({'source_file': g.loc[method, 'source_file'], 'model': model, 'pair': pair, 'method': method, 'column': col, 'reported_value': val, 'recomputed_baseline_minus_CRAT': recomputed, 'discrepancy': err})
        for method in METHODS[1:-1]:
            b = g.loc[method]
            rec = {'model': model, 'pair': pair, 'baseline': method}
            for k in ['trainable_params', 'peak_vram_GB', 'train_seconds', 'generation_seconds', 'latency_mean_seconds', 'latency_p50_seconds', 'latency_p95_seconds']:
                rec[k + '_CRAT_reduction_pct'] = 100 * (1 - cr[k] / b[k])
                rec[k + '_CRAT_ratio'] = cr[k] / b[k]
            for k in ['train_tokens_per_second', 'generation_tokens_per_second', 'benchmark_tokens_per_second']:
                rec[k + '_CRAT_gain_pct'] = 100 * (cr[k] / b[k] - 1)
                rec[k + '_CRAT_ratio'] = cr[k] / b[k]
            rec['CRAT_to_baseline_train_tokens'] = cr.train_tokens / b.train_tokens
            rec['CRAT_to_baseline_updates'] = cr.updates / b.updates
            costs.append(rec)
    mar = pd.DataFrame(margins)
    con = pd.DataFrame(contrasts)
    rank = pd.DataFrame(ranks)
    cost = pd.DataFrame(costs)
    anomaly = pd.DataFrame(anomalies)
    for name, data in [('CRAT_vs_strongest_baseline', mar), ('CRAT_vs_each_baseline', con), ('metric_ranks', rank), ('CRAT_cost_comparisons', cost), ('source_delta_discrepancies', anomaly)]:
        data.to_csv(TAB / (name + '.csv'), index=False, float_format=None)
    macro = df.groupby('method', observed=True)[METRICS].mean().reindex(METHODS)
    macro['mean_rank'] = rank.groupby('method')['rank'].mean()
    macro['first_place_cells'] = rank[rank['rank'] == 1].groupby('method').size().reindex(METHODS, fill_value=0)
    macro.to_csv(TAB / 'macro_summary.csv', float_format=None)
    return {'D': df[primary].copy(), 'M': macro, 'A': anomaly, 'G': mar, 'C': cost, 'rank': rank}

def ablation_data(SRC, OUT, comparison):
    TABLES = OUT / 'data_tables'
    TABLES.mkdir(parents=True, exist_ok=True)
    PAIRS = ['EN-FR', 'EN-FA', 'EN-ZH']
    VARIANTS = ['CRAT_full', 'no_kl', 'no_causal', 'no_morph', 'no_sparse', 'no_integration', 'static_role']
    METRICS = ['BLEU', 'chrF', 'COMET']
    ROLES = ['source', 'indicator', 'position', 'integration']
    frames, routes, manifest = ([], [], [])
    for pair in PAIRS:
        for kind in ['efficiency', 'routing']:
            path = SRC / f'{pair.lower()}_ablation_posthoc_opus100_{kind}.csv'
            frame = pd.read_csv(path)
            if not list(frame.ablation) == VARIANTS:
                raise ValueError('CSV validation failed: list(frame.ablation) == VARIANTS')
            if not not frame.ablation.duplicated().any():
                raise ValueError('CSV validation failed: not frame.ablation.duplicated().any()')
            frame.insert(0, 'pair', pair)
            (frames if kind == 'efficiency' else routes).append(frame)
            manifest.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(frame)})
    d = pd.concat(frames, ignore_index=True)
    r = pd.concat(routes, ignore_index=True)
    d.to_csv(TABLES / 'source_efficiency_combined.csv', index=False)
    r.to_csv(TABLES / 'source_routing_combined.csv', index=False)
    if not not d.isna().any().any():
        raise ValueError('CSV validation failed: not d.isna().any().any()')
    if not r.loc[r.ablation != 'static_role'].notna().all().all():
        raise ValueError("CSV validation failed: r.loc[r.ablation != 'static_role'].notna().all().all()")
    if not r.loc[r.ablation == 'static_role'].iloc[:, 2:].isna().all().all():
        raise ValueError("CSV validation failed: r.loc[r.ablation == 'static_role'].iloc[:, 2:].isna().all().all()")
    delta_audit, quality, effects, costs, reference = ([], [], [], [], [])
    cost_map = {'params_reduction_pct_vs_CRAT': ('trainable_params', 'reduction'), 'vram_reduction_pct_vs_CRAT': ('peak_vram_GB', 'reduction'), 'train_time_reduction_pct_vs_CRAT': ('train_seconds', 'reduction'), 'generation_time_reduction_pct_vs_CRAT': ('generation_seconds', 'reduction'), 'latency_mean_reduction_pct_vs_CRAT': ('latency_mean_seconds', 'reduction'), 'latency_p50_reduction_pct_vs_CRAT': ('latency_p50_seconds', 'reduction'), 'latency_p95_reduction_pct_vs_CRAT': ('latency_p95_seconds', 'reduction'), 'train_throughput_gain_pct_vs_CRAT': ('train_tokens_per_second', 'gain'), 'generation_throughput_gain_pct_vs_CRAT': ('generation_tokens_per_second', 'gain'), 'benchmark_throughput_gain_pct_vs_CRAT': ('benchmark_tokens_per_second', 'gain')}
    checks = []
    for pair, group in d.groupby('pair', sort=False):
        full = group.set_index('ablation').loc['CRAT_full']
        for col in ['train_tokens', 'updates', 'effective_batch_size']:
            if not group[col].nunique() == 1:
                raise ValueError('CSV validation failed: group[col].nunique() == 1')
        main = comparison['D']
        prior = main[(main.model == 'Mistral') & (main.pair == pair) & (main.method == 'CRAT')].iloc[0]
        ref = {'pair': pair}
        for m in METRICS:
            ref[m + '_matches_main'] = bool(np.isclose(full[m + '_mean'], prior[m], rtol=0, atol=1e-10))
            ref[m + '_sd_matches_main'] = bool(np.isclose(full[m + '_mean_std'], prior[m + '_sd'], rtol=0, atol=1e-10))
            implied = group.loc[group.ablation != 'CRAT_full', m + '_mean'] - group.loc[group.ablation != 'CRAT_full', 'delta_' + m + '_vs_CRAT']
            ref[m + '_displayed_full'] = full[m + '_mean']
            ref[m + '_implied_reference'] = float(implied.mean())
            ref[m + '_implied_reference_range'] = float(implied.max() - implied.min())
        for col in ['train_tokens', 'updates', 'train_seconds', 'trainable_params']:
            ref['ablation_' + col] = float(full[col])
            ref['main_' + col] = float(prior[col])
            ref[col + '_ratio_ablation_main'] = float(full[col] / prior[col])
        reference.append(ref)
        for _, row in group.iterrows():
            q = {'pair': pair, 'ablation': row.ablation}
            e = {'pair': pair, 'ablation': row.ablation}
            c = {'pair': pair, 'ablation': row.ablation}
            for m in METRICS:
                q[m] = row[m + '_mean']
                q[m + '_sd'] = row[m + '_mean_std']
                change = float(row[m + '_mean'] - full[m + '_mean'])
                e[m + '_delta'] = change
                e[m + '_drop'] = -change
                e[m + '_relative_drop_pct'] = -100 * change / full[m + '_mean']
                e[m + '_sd_ratio'] = row[m + '_mean_std'] / full[m + '_mean_std']
                source = float(row['delta_' + m + '_vs_CRAT'])
                delta_audit.append({'pair': pair, 'ablation': row.ablation, 'metric': m, 'source_delta': source, 'recomputed_delta': change, 'difference': source - change, 'mismatch': not np.isclose(source, change, rtol=0, atol=1e-06), 'sign_flip': source * change < 0})
            for name, (col, mode) in cost_map.items():
                calc = 100 * (1 - row[col] / full[col] if mode == 'reduction' else row[col] / full[col] - 1)
                c[name] = calc
                checks.append({'pair': pair, 'ablation': row.ablation, 'field': name, 'source': row[name], 'recomputed': calc, 'error': row[name] - calc})
            for col in ['trainable_params', 'total_params', 'trainable_pct', 'peak_vram_GB', 'train_seconds', 'train_tokens_per_second', 'generation_seconds', 'generated_tokens', 'generation_tokens_per_second', 'latency_mean_seconds', 'latency_p50_seconds', 'latency_p95_seconds', 'benchmark_tokens_per_second']:
                c[col] = row[col]
            c['removed_trainable_params'] = int(full.trainable_params - row.trainable_params)
            c['vram_saved_GiB'] = full.peak_vram_GB - row.peak_vram_GB
            c['train_minutes_saved'] = (full.train_seconds - row.train_seconds) / 60
            quality.append(q)
            effects.append(e)
            costs.append(c)
            for name, calc in [('trainable_pct', 100 * row.trainable_params / row.total_params), ('train_tokens_per_second', row.train_tokens / row.train_seconds), ('generation_tokens_per_second', row.generated_tokens / row.generation_seconds)]:
                checks.append({'pair': pair, 'ablation': row.ablation, 'field': name, 'source': row[name], 'recomputed': calc, 'error': row[name] - calc})
    q = pd.DataFrame(quality)
    e = pd.DataFrame(effects)
    c = pd.DataFrame(costs)
    a = pd.DataFrame(delta_audit)
    refs = pd.DataFrame(reference)
    check = pd.DataFrame(checks)
    for name, frame in [('quality', q), ('recomputed_quality_effects', e), ('recomputed_efficiency', c), ('source_quality_delta_audit', a), ('reference_reconciliation', refs), ('efficiency_arithmetic_checks', check)]:
        frame.to_csv(TABLES / (name + '.csv'), index=False)
    rg = []
    for pair, group in r.groupby('pair', sort=False):
        full = group.set_index('ablation').loc['CRAT_full']
        for _, row in group.iterrows():
            rr = {'pair': pair, 'ablation': row.ablation}
            for role in ROLES:
                for phase in ['train', 'eval']:
                    col = phase + '_gate_mean_' + role
                    rr[col] = row[col]
                    rr[col + '_delta'] = row[col] - full[col]
                rr['eval_active_' + role] = row['eval_gate_active_' + role + '@0.5']
            rr['train_gate_statistic'] = row.train_router_entropy
            rr['train_gate_statistic_delta'] = row.train_router_entropy - full.train_router_entropy
            rr['eval_soft_threshold_count'] = row['eval_mean_active_roles@0.5']
            if row.ablation != 'static_role':
                if not abs(sum((rr['eval_active_' + role] for role in ROLES)) - rr['eval_soft_threshold_count']) < 1e-07:
                    raise ValueError("CSV validation failed: abs(sum((rr['eval_active_' + role] for role in ROLES)) - rr['eval_soft_threshold_count']) < 1e-07")
            rg.append(rr)
    rg = pd.DataFrame(rg)
    rg.to_csv(TABLES / 'routing_analysis.csv', index=False)
    macro = e.groupby('ablation', sort=False).mean(numeric_only=True).reset_index()
    macro.to_csv(TABLES / 'descriptive_macro_quality_effects.csv', index=False)
    return {'Q': q.set_index(['pair', 'ablation']), 'C': c.set_index(['pair', 'ablation']), 'R': rg.set_index(['pair', 'ablation']), 'A': a, 'REF': refs.set_index('pair'), 'E': e.set_index(['pair', 'ablation'])}

def sensitivity_data(SRC, OUT):
    TAB = OUT / 'data_tables'
    TAB.mkdir(parents=True, exist_ok=True)
    PAIRS = ['EN-FR', 'EN-FA', 'EN-ZH']
    PARAMS = ['lambda_kl', 'lambda_causal', 'lambda_morph', 'lambda_sparse', 'kl_temperature', 'integration_rank', 'heads_per_direction', 'top_k']
    DEFAULT = {'lambda_kl': 0.1, 'lambda_causal': 0.1, 'lambda_morph': 0.1, 'lambda_sparse': 0.002, 'kl_temperature': 2.0, 'integration_rank': 16.0, 'heads_per_direction': 64.0, 'top_k': 2.0}
    LABELS = {'lambda_kl': 'KL coefficient', 'lambda_causal': 'Causal coefficient', 'lambda_morph': 'Morphology coefficient', 'lambda_sparse': 'Sparsity coefficient', 'kl_temperature': 'KL temperature', 'integration_rank': 'Integration rank', 'heads_per_direction': 'Heads per direction', 'top_k': 'Evaluation top k'}
    KEYS = ['sensitivity_parameter', 'sensitivity_value']
    data = []
    audits = []
    manifest = []
    robust = []
    uncertainty = []
    directions = []
    comet_audit = []
    all_raw = {}
    for pair in PAIRS:
        fs = {}
        for kind in ['summary', 'efficiency', 'routing_summary', 'parameter_robustness', 'comet']:
            path = SRC / f'{pair.lower()}_sensitivity_posthoc_opus100_{kind}.csv'
            d = pd.read_csv(path)
            fs[kind] = d
            manifest.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'rows': len(d), 'columns': len(d.columns)})
            if not not d.duplicated(KEYS if kind != 'parameter_robustness' else ['sensitivity_parameter']).any():
                raise ValueError("CSV validation failed: not d.duplicated(KEYS if kind != 'parameter_robustness' else ['sensitivity_parameter']).any()")
            copy = d.copy()
            copy.insert(0, 'pair', pair)
            all_raw.setdefault(kind, []).append(copy)
        s = fs['summary'].copy().set_index(KEYS)
        for kind in ['efficiency', 'routing_summary']:
            other = fs[kind].set_index(KEYS)
            if not set(other.index) == set(s.index):
                raise ValueError('CSV validation failed: set(other.index) == set(s.index)')
            for col in other.columns.intersection(s.columns):
                aa = s[col]
                bb = other[col]
                if pd.api.types.is_numeric_dtype(aa) and pd.api.types.is_numeric_dtype(bb):
                    valid = aa.notna() & bb.notna()
                    error = (aa[valid] - bb[valid]).abs()
                    for ix in error[error > 1e-08].index:
                        audits.append({'pair': pair, 'kind': kind, 'parameter': ix[0], 'value': ix[1], 'field': col, 'summary': aa.loc[ix], 'other': bb.loc[ix], 'difference': aa.loc[ix] - bb.loc[ix]})
                    if not aa.isna().equals(bb.isna()):
                        raise ValueError(f"CSV validation failed: aa.isna().equals(bb.isna()); context: {(pair, kind, col, 'missingness')}")
                elif not aa.equals(bb):
                    raise ValueError(f'CSV validation failed: aa.equals(bb); context: {(pair, kind, col)}')
        c = fs['comet'].set_index(KEYS)
        if 'COMET_mean_mean' not in s:
            s['COMET_mean_mean'] = np.nan
        for ix, row in c.iterrows():
            present = s.loc[ix, 'COMET_mean_mean']
            if pd.notna(present):
                comet_audit.append({'pair': pair, 'parameter': ix[0], 'value': ix[1], 'summary_COMET': present, 'comet_export_macro': row.COMET_mean_mean, 'difference': present - row.COMET_mean_mean})
            else:
                s.loc[ix, 'COMET_mean_mean'] = row.COMET_mean_mean
            s.loc[ix, 'COMET_pooled'] = row.COMET_mean
            s.loc[ix, 'COMET_num_segments'] = row.COMET_num_segments_mean
            lang = pair[3:].lower()
            for short, long in [('forward', 'en-' + lang), ('reverse', lang + '-en')]:
                col = 'COMET_' + long + '_mean'
                s.loc[ix, 'COMET_' + short] = row[col]
            if 'COMET_mean_std' in row:
                if 'COMET_mean_std' not in s:
                    s['COMET_mean_std'] = np.nan
                if pd.isna(s.loc[ix, 'COMET_mean_std']):
                    s.loc[ix, 'COMET_mean_std'] = row.COMET_mean_std
        s = s.reset_index()
        s.insert(0, 'pair', pair)
        for m in ['BLEU', 'chrF']:
            err = s[m + '_mean_mean'] - (s[m + '_forward_mean'] + s[m + '_reverse_mean']) / 2
            if not err.abs().max() < 1e-07:
                raise ValueError('CSV validation failed: err.abs().max() < 1e-07')
        if 'COMET_forward' in s:
            err = s.COMET_mean_mean - (s.COMET_forward + s.COMET_reverse) / 2
            if not err.dropna().abs().max() < 1e-08:
                raise ValueError('CSV validation failed: err.dropna().abs().max() < 1e-08')
        if pair == 'EN-FR':
            for col in s:
                if not col.endswith('_sem'):
                    continue
                stem = col[:-4]
                sdcol = stem + '_std'
                mcol = stem + '_mean'
                if sdcol not in s:
                    continue
                for _, row in s.iterrows():
                    sd = row[sdcol]
                    se = row[col]
                    if not (pd.notna(sd) and pd.notna(se) and (se > 0)):
                        continue
                    implied_n = (sd / se) ** 2
                    mult = (row.get(stem + '_ci95_high', np.nan) - row[mcol]) / se
                    uncertainty.append({'pair': pair, 'parameter': row.sensitivity_parameter, 'value': row.sensitivity_value, 'field': stem, 'implied_n': implied_n, 'ci_multiplier': mult, 'student_t_df1_975': math.tan(math.pi * (0.975 - 0.5))})
        rb = fs['parameter_robustness'].set_index('sensitivity_parameter')
        for param, g in s.groupby('sensitivity_parameter', sort=False):
            ref = g[np.isclose(g.sensitivity_value, DEFAULT[param])]
            for m in ['BLEU', 'chrF', 'COMET']:
                vals = g[m + '_mean_mean'].dropna()
                if vals.empty:
                    continue
                best = s.loc[vals.idxmax()]
                worst = s.loc[vals.idxmin()]
                rec = {'pair': pair, 'parameter': param, 'n_settings': len(g), 'metric': m, 'minimum': float(vals.min()), 'maximum': float(vals.max()), 'range': float(vals.max() - vals.min()), 'best_setting': best.sensitivity_value, 'worst_setting': worst.sensitivity_value, 'default_setting': DEFAULT[param], 'default_present': not ref.empty, 'default_mean': float(ref.iloc[0][m + '_mean_mean']) if not ref.empty else np.nan}
                if pd.notna(rec['default_mean']):
                    rec['best_minus_default'] = rec['maximum'] - rec['default_mean']
                    rec['worst_minus_default'] = rec['minimum'] - rec['default_mean']
                if m != 'COMET':
                    r = rb.loc[param]
                    rec['source_range'] = r['delta_' + m + '_vs_default_range']
                    rec['range_error'] = rec['range'] - rec['source_range']
                    rec['implied_default_from_min'] = rec['minimum'] - r['delta_' + m + '_vs_default_min']
                    rec['implied_default_from_max'] = rec['maximum'] - r['delta_' + m + '_vs_default_max']
                    rec['implied_reference_spread'] = abs(rec['implied_default_from_min'] - rec['implied_default_from_max'])
                    rec['source_best_label'] = r['delta_' + m + '_vs_default_best_setting']
                    rec['source_worst_label'] = r['delta_' + m + '_vs_default_worst_setting']
                    if not int(r.n_settings) == len(g):
                        raise ValueError('CSV validation failed: int(r.n_settings) == len(g)')
                robust.append(rec)
        data.append(s)
    D = pd.concat(data, ignore_index=True)
    RB = pd.DataFrame(robust)
    for kind, frames in all_raw.items():
        pd.concat(frames, ignore_index=True).to_csv(TAB / f'source_{kind}_combined.csv', index=False)
    D.to_csv(TAB / 'normalized_sensitivity_results.csv', index=False)
    RB.to_csv(TAB / 'recomputed_parameter_robustness.csv', index=False)
    pd.DataFrame(audits, columns=['pair', 'kind', 'parameter', 'value', 'field', 'summary', 'other', 'difference']).to_csv(TAB / 'cross_export_discrepancies.csv', index=False)
    pd.DataFrame(comet_audit).to_csv(TAB / 'COMET_overlap_checks.csv', index=False)
    U = pd.DataFrame(uncertainty)
    U.to_csv(TAB / 'EN_FR_uncertainty_arithmetic.csv', index=False)
    diffs = []
    for (pair, param), g in D.groupby(['pair', 'sensitivity_parameter'], sort=False):
        ref = g[np.isclose(g.sensitivity_value, DEFAULT[param])]
        if ref.empty:
            continue
        base = ref.iloc[0]
        for _, row in g.iterrows():
            rec = {'pair': pair, 'parameter': param, 'value': row.sensitivity_value, 'reference_value': DEFAULT[param]}
            for m in ['BLEU', 'chrF', 'COMET']:
                rec[m + '_delta'] = row[m + '_mean_mean'] - base[m + '_mean_mean']
            for m in ['BLEU', 'chrF']:
                for direction in ['forward', 'reverse']:
                    rec[m + '_' + direction + '_delta'] = row[m + '_' + direction + '_mean'] - base[m + '_' + direction + '_mean']
            for direction in ['forward', 'reverse']:
                col = 'COMET_' + direction
                rec[col + '_delta'] = row.get(col, np.nan) - base.get(col, np.nan)
            for col in ['trainable_params', 'peak_vram_GB', 'train_seconds', 'generation_seconds', 'generated_tokens', 'generation_tokens_per_second', 'latency_mean_seconds', 'latency_p50_seconds', 'latency_p95_seconds', 'benchmark_tokens_per_second']:
                rec[col + '_change_pct'] = 100 * (row[col + '_mean'] / base[col + '_mean'] - 1)
            diffs.append(rec)
    DI = pd.DataFrame(diffs)
    DI.to_csv(TAB / 'within_sweep_changes.csv', index=False)
    common = []
    for param in PARAMS:
        grid = sorted(set.intersection(*(set(D[(D.pair == pair) & (D.sensitivity_parameter == param)].sensitivity_value) for pair in PAIRS)))
        for pair in PAIRS:
            g = D[(D.pair == pair) & (D.sensitivity_parameter == param) & D.sensitivity_value.isin(grid)]
            rec = {'pair': pair, 'parameter': param, 'common_values': ', '.join((f'{x:g}' for x in grid))}
            for m in ['BLEU', 'chrF', 'COMET']:
                rec[m + '_range_on_common_grid'] = g[m + '_mean_mean'].max() - g[m + '_mean_mean'].min()
            common.append(rec)
    pd.DataFrame(common).to_csv(TAB / 'robustness_on_common_grids.csv', index=False)
    defaults = []
    for pair in PAIRS:
        for param in PARAMS:
            g = D[(D.pair == pair) & (D.sensitivity_parameter == param) & np.isclose(D.sensitivity_value, DEFAULT[param])]
            if len(g):
                defaults.append(g.iloc[0])
    pd.DataFrame(defaults).to_csv(TAB / 'nominal_default_rows.csv', index=False)
    return {'D': D, 'R': RB, 'DI': DI, 'common': pd.DataFrame(common)}
