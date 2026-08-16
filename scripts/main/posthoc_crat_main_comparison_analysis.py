#!/usr/bin/env python3
"""Post-hoc analysis for the CRAT *main comparison* experiment.

This script is aligned to the current Trillium main-experiment layout::

    $ROOT/outputs/main_comparison/
      mistral7b_<dataset>_<pairtag>_seed<seed>_<method>/
        <pair>_seed<seed>_<method>_comparison_results.csv
        <pair>_seed<seed>_<method>_translation_samples.csv
        <pair>_component_discovery.json          # copied when available
        <pair>_morph_cache_<backend>.pkl         # copied when available
        <method>/
          <pair>_seed<seed>_<method>_metadata.json
          <pair>_seed<seed>_<method>_loss_log.csv
          ...

Pair-shared expensive artifacts are read from::

    $ROOT/outputs/_shared/<dataset>/<pair>/
      <pair>_component_discovery.json
      <pair>_morph_cache_<backend>.pkl

For ``--run-mode smoke`` the corresponding ``main_comparison_smoke`` and
``_shared_smoke`` roots are used.

The script DOES NOT retrain models. It provides:

1. Per-seed and across-seed quality/efficiency aggregation.
2. CRAT-relative quality/cost deltas.
3. Routing summaries for CRAT-family methods.
4. Optional reference-based COMET scoring from saved translation samples.
5. Hierarchical paired bootstrap comparisons: CRAT vs every other method.
6. Planned CRAT objective contrasts supported by the new experiment:
     * CRAT(CE+KL) - CRAT(CE)            : KL increment
     * CRAT - CRAT(CE+KL)                : causal+morph+sparsity increment
     * CRAT - CRAT(no sparsity)           : sparsity increment
     * CRAT(no sparsity) - CRAT(CE+KL)    : causal+morph increment
7. Pair-specific shared SIPP/morphology provenance audit.
8. Run-configuration audit to make seed/method settings transparent.

Bootstrap delta convention:
    delta = method_A - method_B
for BLEU, chrF and COMET. Positive values therefore favor method_A.

CRAT-relative table convention:
    quality delta = Method - CRAT
    cost reduction = 100 * (CRAT - Method) / CRAT
so positive cost reduction means the method is cheaper than CRAT.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sacrebleu import corpus_bleu, corpus_chrf


METHODS: Tuple[Tuple[str, str], ...] = (
    ("base", "Base Mistral-7B"),
    ("full_sft", "Full SFT"),
    ("targeted_sft", "Targeted SFT"),
    ("lora", "LoRA"),
    ("rslora", "RSLoRA"),
    ("dora", "DoRA"),
    ("crat", "CRAT"),
    ("crat_no_sparse", "CRAT - sparsity"),
    ("crat_ce", "CRAT (CE only)"),
    ("crat_ce_kl", "CRAT (CE + KL)"),
)
METHOD_LABEL = dict(METHODS)
METHOD_ORDER = {method: i for i, (method, _label) in enumerate(METHODS)}

DISCOVERY_METHODS = {
    "targeted_sft", "crat", "crat_no_sparse", "crat_ce", "crat_ce_kl"
}
MORPH_METHODS = {"crat", "crat_no_sparse"}
ROUTER_METHODS = {"crat", "crat_no_sparse", "crat_ce", "crat_ce_kl"}
ROLES: Tuple[str, ...] = ("source", "indicator", "position", "integration")

EFFICIENCY_METRICS: Tuple[str, ...] = (
    "trainable_params",
    "total_params",
    "trainable_pct",
    "effective_batch_size",
    "peak_vram_GB",
    "train_seconds",
    "updates",
    "train_tokens",
    "train_tokens_per_second",
    "generation_seconds",
    "generated_tokens",
    "generation_tokens_per_second",
    "latency_mean_seconds",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "benchmark_tokens_per_second",
)

ROUTING_METRICS: Tuple[str, ...] = tuple(
    [f"train_gate_mean_{role}" for role in ROLES]
    + ["train_router_entropy"]
    + [f"eval_gate_mean_{role}" for role in ROLES]
    + [f"eval_gate_active_{role}@0.5" for role in ROLES]
    + ["eval_mean_active_roles@0.5"]
)

PLANNED_CONTRASTS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "kl_increment",
        "CRAT (CE+KL) - CRAT (CE)",
        "crat_ce_kl",
        "crat_ce",
    ),
    (
        "structured_objectives_increment",
        "CRAT - CRAT (CE+KL)",
        "crat",
        "crat_ce_kl",
    ),
    (
        "sparsity_increment",
        "CRAT - CRAT (no sparsity)",
        "crat",
        "crat_no_sparse",
    ),
    (
        "causal_morph_increment",
        "CRAT (no sparsity) - CRAT (CE+KL)",
        "crat_no_sparse",
        "crat_ce_kl",
    ),
)

CONFIG_FIELDS: Tuple[str, ...] = (
    "model_name", "source_lang", "target_lang", "bidirectional",
    "dataset_mode", "seed", "discovery_seed", "epochs", "max_steps",
    "learning_rate", "warmup_ratio", "weight_decay", "max_grad_norm",
    "train_batch_size", "gradient_accumulation_steps", "eval_batch_size",
    "max_length", "max_new_tokens", "heads_per_direction",
    "probe_examples_per_direction", "causal_score_examples",
    "component_batch_size", "detection_mode", "specific_subspace_rank",
    "top_integration_layers", "morph_backend", "lambda_causal",
    "lambda_morph", "lambda_sparse", "lambda_kl", "kl_temperature",
    "peft_rank", "peft_alpha", "peft_dropout", "peft_target_modules",
    "gradient_checkpointing", "full_sft_optimizer", "attn_implementation",
    "detection_attn_implementation", "eval_limit_per_direction",
    "benchmark_examples",
)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def parse_csv_list(text: str) -> List[str]:
    values = [x.strip() for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one value is required.")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate values are not allowed: {values}")
    return values


def parse_seed_list(text: str) -> List[int]:
    values = [int(x) for x in parse_csv_list(text)]
    for seed in values:
        if seed < 0 or seed > 4294967295:
            raise ValueError(f"Seed out of uint32 range: {seed}")
    return values


def numeric(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_two(a: float, b: float) -> float:
    arr = np.asarray([a, b], dtype=float)
    good = arr[np.isfinite(arr)]
    return float(good.mean()) if len(good) else float("nan")


def safe_pct_change(new: float, reference: float) -> float:
    if not np.isfinite(new) or not np.isfinite(reference) or reference == 0:
        return float("nan")
    return 100.0 * (new - reference) / reference


def safe_reduction_pct(value: float, reference: float) -> float:
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0:
        return float("nan")
    return 100.0 * (reference - value) / reference


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sample_fingerprint(df: pd.DataFrame) -> str:
    cols = ["seed", "method", "direction", "src", "ref", "pred"]
    payload = df[cols].astype(str).sort_values(cols[:-1]).to_csv(index=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def comparison_root(root: Path, run_mode: str) -> Path:
    return root / "outputs" / (
        "main_comparison_smoke" if run_mode == "smoke" else "main_comparison"
    )


def shared_root(root: Path, run_mode: str) -> Path:
    return root / "outputs" / ("_shared_smoke" if run_mode == "smoke" else "_shared")


def experiment_dir(
    root: Path,
    run_mode: str,
    dataset: str,
    pair_tag: str,
    seed: int,
    method: str,
) -> Path:
    return comparison_root(root, run_mode) / (
        f"mistral7b_{dataset}_{pair_tag}_seed{seed}_{method}"
    )


def artifact_prefix(pair: str, seed: int, method: str) -> str:
    return f"{pair}_seed{seed}_{method}"


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def artifact_candidates(
    run_dir: Path,
    pair: str,
    seed: int,
    method: str,
    kind: str,
) -> List[Path]:
    if kind not in {"comparison_results", "translation_samples"}:
        raise ValueError(f"Unknown artifact kind: {kind}")
    exact = run_dir / f"{artifact_prefix(pair, seed, method)}_{kind}.csv"
    candidates = [exact]
    for path in sorted(run_dir.glob(f"*_{kind}.csv")):
        if path not in candidates:
            candidates.append(path)
    return candidates


def t_critical_975(df: int) -> float:
    table = {
        1: 12.706204736, 2: 4.302652730, 3: 3.182446305,
        4: 2.776445105, 5: 2.570581836, 6: 2.446911851,
        7: 2.364624252, 8: 2.306004135, 9: 2.262157163,
        10: 2.228138852, 11: 2.200985160, 12: 2.178812830,
        13: 2.160368656, 14: 2.144786688, 15: 2.131449546,
        16: 2.119905299, 17: 2.109815578, 18: 2.100922040,
        19: 2.093024054, 20: 2.085963447, 25: 2.059538553,
        30: 2.042272456, 40: 2.021075390, 60: 2.000297822,
        120: 1.979930406,
    }
    if df <= 0:
        return float("nan")
    if df in table:
        return table[df]
    if df < 25:
        return table[20]
    if df < 30:
        return table[25]
    if df < 40:
        return table[30]
    if df < 60:
        return table[40]
    if df < 120:
        return table[60]
    return 1.959963985


def summarize_numeric(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for keys, group in df.groupby(list(group_cols), sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group.columns else len(group)
        for col in metric_cols:
            if col not in group.columns:
                continue
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            n = len(vals)
            row[f"{col}_n"] = int(n)
            if n == 0:
                for suffix in ("mean", "std", "sem", "ci95_low", "ci95_high"):
                    row[f"{col}_{suffix}"] = np.nan
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else np.nan
            sem = std / math.sqrt(n) if n > 1 else np.nan
            crit = t_critical_975(n - 1) if n > 1 else np.nan
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_sem"] = sem
            row[f"{col}_ci95_low"] = mean - crit * sem if n > 1 else np.nan
            row[f"{col}_ci95_high"] = mean + crit * sem if n > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Load completed main-comparison runs
# -----------------------------------------------------------------------------


def load_completed_runs(
    root: Path,
    run_mode: str,
    dataset: str,
    pair: str,
    seeds: Sequence[int],
    methods: Sequence[str],
    allow_missing: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    src, tgt = pair.split("-", 1)
    pair_tag = f"{src}{tgt}"
    fwd, rev = f"{src}-{tgt}", f"{tgt}-{src}"

    result_rows: List[Dict[str, Any]] = []
    sample_frames: List[pd.DataFrame] = []
    problems: List[str] = []

    for seed in seeds:
        for method in methods:
            directory = experiment_dir(root, run_mode, dataset, pair_tag, seed, method)
            result_path = first_existing(
                artifact_candidates(directory, pair, seed, method, "comparison_results")
            )
            sample_path = first_existing(
                artifact_candidates(directory, pair, seed, method, "translation_samples")
            )
            if result_path is None:
                problems.append(f"missing results: seed={seed} method={method} dir={directory}")
                continue
            if sample_path is None:
                problems.append(f"missing translations: seed={seed} method={method} dir={directory}")
                continue

            results = pd.read_csv(result_path)
            require_columns(
                results,
                ["method", f"BLEU_{fwd}", f"BLEU_{rev}", f"chrF_{fwd}", f"chrF_{rev}"],
                result_path,
            )
            hit = results[results["method"].astype(str) == method]
            if hit.empty:
                problems.append(f"method={method!r} not found in {result_path}")
                continue
            raw = hit.iloc[-1].to_dict()

            observed_pair = str(raw.get("pair", pair))
            observed_seed = numeric(raw.get("seed", seed))
            if observed_pair != pair:
                problems.append(f"pair mismatch in {result_path}: {observed_pair} != {pair}")
            if np.isfinite(observed_seed) and int(observed_seed) != int(seed):
                problems.append(f"seed mismatch in {result_path}: {observed_seed} != {seed}")

            row: Dict[str, Any] = {
                "pair": pair,
                "dataset": dataset,
                "run_mode": run_mode,
                "seed": int(seed),
                "method": method,
                "label": METHOD_LABEL[method],
                "run_dir": str(directory),
                "result_file": str(result_path),
                "sample_file": str(sample_path),
                "discovery_seed": numeric(raw.get("discovery_seed")),
                "BLEU_forward": numeric(raw.get(f"BLEU_{fwd}")),
                "BLEU_reverse": numeric(raw.get(f"BLEU_{rev}")),
                "chrF_forward": numeric(raw.get(f"chrF_{fwd}")),
                "chrF_reverse": numeric(raw.get(f"chrF_{rev}")),
                "validation_loss": numeric(raw.get("validation_loss")),
                "empty_rate_forward": numeric(raw.get(f"empty_rate_{fwd}")),
                "empty_rate_reverse": numeric(raw.get(f"empty_rate_{rev}")),
                "punctuation_collapse_rate_forward": numeric(raw.get(f"punctuation_collapse_rate_{fwd}")),
                "punctuation_collapse_rate_reverse": numeric(raw.get(f"punctuation_collapse_rate_{rev}")),
            }
            row["BLEU_mean"] = mean_two(row["BLEU_forward"], row["BLEU_reverse"])
            row["chrF_mean"] = mean_two(row["chrF_forward"], row["chrF_reverse"])
            row["empty_rate_mean"] = mean_two(row["empty_rate_forward"], row["empty_rate_reverse"])
            row["punctuation_collapse_rate_mean"] = mean_two(
                row["punctuation_collapse_rate_forward"], row["punctuation_collapse_rate_reverse"]
            )
            for col in EFFICIENCY_METRICS + ROUTING_METRICS:
                row[col] = numeric(raw.get(col))
            result_rows.append(row)

            samples = pd.read_csv(sample_path)
            require_columns(samples, ["method", "direction", "src", "ref", "pred"], sample_path)
            samples = samples[samples["method"].astype(str) == method].copy()
            if samples.empty:
                problems.append(f"no sample rows for method={method!r} in {sample_path}")
                continue
            samples["seed"] = int(seed)
            samples["method"] = method
            samples["label"] = METHOD_LABEL[method]
            samples["run_dir"] = str(directory)
            samples["sample_file"] = str(sample_path)
            sample_frames.append(samples[[
                "seed", "method", "label", "run_dir", "sample_file",
                "direction", "src", "ref", "pred",
            ]].copy())

    if problems:
        print("\nInput diagnostics:", file=sys.stderr)
        for item in problems:
            print(f"  - {item}", file=sys.stderr)
        if not allow_missing:
            raise SystemExit(
                f"Aborting because {len(problems)} requested main-comparison artifact problem(s) were found. "
                "Use --allow-missing only for intentionally incomplete analyses."
            )

    if not result_rows:
        raise SystemExit("No completed main-comparison results were found.")
    if not sample_frames:
        raise SystemExit("No translation-sample data were found.")

    per_seed = pd.DataFrame(result_rows)
    per_seed["_order"] = per_seed["method"].map(METHOD_ORDER)
    per_seed = per_seed.sort_values(["seed", "_order"]).drop(columns="_order").reset_index(drop=True)
    samples_all = pd.concat(sample_frames, ignore_index=True)

    if not allow_missing:
        expected = len(seeds) * len(methods)
        if len(per_seed) != expected:
            raise SystemExit(f"Expected {expected} result rows, found {len(per_seed)}.")

    return per_seed, samples_all, problems


# -----------------------------------------------------------------------------
# Configuration + shared-artifact audit
# -----------------------------------------------------------------------------


def load_config_audit(
    root: Path,
    run_mode: str,
    dataset: str,
    pair: str,
    seeds: Sequence[int],
    methods: Sequence[str],
) -> pd.DataFrame:
    src, tgt = pair.split("-", 1)
    pair_tag = f"{src}{tgt}"
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            directory = experiment_dir(root, run_mode, dataset, pair_tag, seed, method)
            prefix = artifact_prefix(pair, seed, method)
            path = directory / f"{prefix}_config.json"
            row: Dict[str, Any] = {
                "seed": int(seed), "method": method, "label": METHOD_LABEL[method],
                "config_file": str(path), "config_exists": path.is_file(),
            }
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                for field in CONFIG_FIELDS:
                    value = payload.get(field)
                    if isinstance(value, (list, tuple, dict)):
                        value = json.dumps(value, sort_keys=True)
                    row[field] = value
                row["methods_config"] = json.dumps(payload.get("methods"), sort_keys=True)
            rows.append(row)
    return pd.DataFrame(rows)


def validate_shared_discovery(path: Path, src: str, tgt: str, discovery_seed: int) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path), "exists": path.is_file(), "sha256": sha256_file(path),
        "pair_ok": False, "discovery_seed_ok": False,
    }
    if not path.is_file():
        return info
    payload = json.loads(path.read_text(encoding="utf-8"))
    sig = payload.get("signature", {})
    obs_src = str(sig.get("source_lang", "")).lower()
    obs_tgt = str(sig.get("target_lang", "")).lower()
    info.update({
        "source_lang": obs_src,
        "target_lang": obs_tgt,
        "discovery_seed": sig.get("seed"),
        "signature_hash": payload.get("signature_hash"),
        "selected_head_count": payload.get("selected_head_count"),
        "integration_layers": payload.get("integration_layers"),
    })
    info["pair_ok"] = obs_src == src.lower() and obs_tgt == tgt.lower()
    try:
        info["discovery_seed_ok"] = int(sig.get("seed")) == int(discovery_seed)
    except (TypeError, ValueError):
        info["discovery_seed_ok"] = False
    return info


def validate_shared_morph(path: Path, src: str, tgt: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path), "exists": path.is_file(), "sha256": sha256_file(path),
        "languages": [], "pair_ok": False, "entries": None,
    }
    if not path.is_file():
        return info
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    langs = sorted({str(key).split(":", 1)[0].lower() for key in cache.keys()})
    allowed = {src.lower(), tgt.lower()}
    info["languages"] = langs
    info["entries"] = len(cache)
    info["pair_ok"] = set(langs) == allowed
    return info


def audit_shared_artifact_reuse(
    root: Path,
    run_mode: str,
    dataset: str,
    pair: str,
    seeds: Sequence[int],
    methods: Sequence[str],
    morph_backend: str,
    discovery_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    src, tgt = pair.split("-", 1)
    pair_tag = f"{src}{tgt}"
    shared_dir = shared_root(root, run_mode) / dataset / pair
    shared_morph = shared_dir / f"{pair}_morph_cache_{morph_backend}.pkl"
    shared_discovery = shared_dir / f"{pair}_component_discovery.json"

    shared_info = {
        "shared_dir": str(shared_dir),
        "morph": validate_shared_morph(shared_morph, src, tgt),
        "discovery": validate_shared_discovery(shared_discovery, src, tgt, discovery_seed),
    }
    shared_morph_hash = shared_info["morph"].get("sha256")
    shared_discovery_hash = shared_info["discovery"].get("sha256")

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for method in methods:
            directory = experiment_dir(root, run_mode, dataset, pair_tag, seed, method)
            run_morph = directory / f"{pair}_morph_cache_{morph_backend}.pkl"
            run_discovery = directory / f"{pair}_component_discovery.json"
            morph_hash = sha256_file(run_morph)
            discovery_hash = sha256_file(run_discovery)
            rows.append({
                "seed": int(seed),
                "method": method,
                "label": METHOD_LABEL[method],
                "requires_morph": method in MORPH_METHODS,
                "requires_discovery": method in DISCOVERY_METHODS,
                "run_morph_file": str(run_morph),
                "run_morph_exists": run_morph.is_file(),
                "run_morph_sha256": morph_hash,
                "morph_matches_shared": bool(
                    morph_hash is not None and shared_morph_hash is not None and morph_hash == shared_morph_hash
                ),
                "run_discovery_file": str(run_discovery),
                "run_discovery_exists": run_discovery.is_file(),
                "run_discovery_sha256": discovery_hash,
                "discovery_matches_shared": bool(
                    discovery_hash is not None and shared_discovery_hash is not None and discovery_hash == shared_discovery_hash
                ),
            })
    return pd.DataFrame(rows), shared_info


# -----------------------------------------------------------------------------
# CRAT-relative quality/cost deltas
# -----------------------------------------------------------------------------


def add_crat_relative_deltas(per_seed: pd.DataFrame) -> pd.DataFrame:
    out = per_seed.copy()
    quality_cols = {
        "BLEU_mean": "delta_BLEU_vs_CRAT",
        "chrF_mean": "delta_chrF_vs_CRAT",
        "validation_loss": "delta_validation_loss_vs_CRAT",
    }
    lower_cost_cols = {
        "trainable_params": "params_reduction_pct_vs_CRAT",
        "peak_vram_GB": "vram_reduction_pct_vs_CRAT",
        "train_seconds": "train_time_reduction_pct_vs_CRAT",
        "generation_seconds": "generation_time_reduction_pct_vs_CRAT",
        "latency_mean_seconds": "latency_mean_reduction_pct_vs_CRAT",
        "latency_p50_seconds": "latency_p50_reduction_pct_vs_CRAT",
        "latency_p95_seconds": "latency_p95_reduction_pct_vs_CRAT",
    }
    throughput_cols = {
        "train_tokens_per_second": "train_throughput_gain_pct_vs_CRAT",
        "generation_tokens_per_second": "generation_throughput_gain_pct_vs_CRAT",
        "benchmark_tokens_per_second": "benchmark_throughput_gain_pct_vs_CRAT",
    }

    for seed, idx in out.groupby("seed").groups.items():
        group = out.loc[idx]
        ref = group[group["method"] == "crat"]
        if len(ref) != 1:
            raise RuntimeError(f"Seed {seed}: expected exactly one CRAT row, found {len(ref)}")
        ref = ref.iloc[0]
        for row_idx in idx:
            for col, dst in quality_cols.items():
                out.loc[row_idx, dst] = numeric(out.loc[row_idx, col]) - numeric(ref[col])
            for col, dst in lower_cost_cols.items():
                out.loc[row_idx, dst] = safe_reduction_pct(
                    numeric(out.loc[row_idx, col]), numeric(ref[col])
                )
            for col, dst in throughput_cols.items():
                out.loc[row_idx, dst] = safe_pct_change(
                    numeric(out.loc[row_idx, col]), numeric(ref[col])
                )
    return out


# -----------------------------------------------------------------------------
# COMET
# -----------------------------------------------------------------------------


def load_comet_model(comet_checkpoint: Optional[str], comet_model_name: str):
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError as exc:
        raise RuntimeError(
            "COMET is not installed. Install `unbabel-comet`, or rerun with --skip-comet."
        ) from exc

    if comet_checkpoint:
        checkpoint = Path(comet_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"COMET checkpoint does not exist: {checkpoint}")
        print(f"Loading local COMET checkpoint: {checkpoint}")
        return load_from_checkpoint(str(checkpoint)), str(checkpoint)

    print(f"Resolving COMET model: {comet_model_name}")
    print("NOTE: this may require network access unless already cached.")
    model_path = download_model(comet_model_name)
    return load_from_checkpoint(model_path), comet_model_name


def comet_predict(model, data: List[Dict[str, str]], batch_size: int, gpus: int):
    # Compatibility with both older and newer COMET/Lightning APIs.
    try:
        return model.predict(data, batch_size=batch_size, gpus=gpus)
    except TypeError:
        accelerator = "gpu" if gpus > 0 else "cpu"
        devices = gpus if gpus > 0 else 1
        return model.predict(data, batch_size=batch_size, accelerator=accelerator, devices=devices)


def compute_or_load_comet(
    samples: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    comet_checkpoint: Optional[str],
    comet_model_name: str,
    batch_size: int,
    gpus: int,
    force: bool,
) -> Tuple[pd.DataFrame, str]:
    score_path = output_dir / f"{prefix}_comet_sentence_scores.csv"
    meta_path = output_dir / f"{prefix}_comet_metadata.json"
    fingerprint = sample_fingerprint(samples)
    requested_model = (
        str(Path(comet_checkpoint).expanduser().resolve()) if comet_checkpoint else comet_model_name
    )

    if not force and score_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cached = pd.read_csv(score_path)
            if (
                meta.get("sample_fingerprint") == fingerprint
                and meta.get("model") == requested_model
                and len(cached) == len(samples)
                and "COMET" in cached.columns
            ):
                print(f"Reusing cached COMET scores: {score_path}")
                return cached, requested_model
        except Exception as exc:
            print(f"Ignoring unusable COMET cache ({exc}); recomputing.")

    model, resolved_model = load_comet_model(comet_checkpoint, comet_model_name)
    data = [
        {"src": str(r.src), "mt": str(r.pred), "ref": str(r.ref)}
        for r in samples.itertuples(index=False)
    ]
    print(f"Scoring {len(data):,} translations with COMET ...")
    output = comet_predict(model, data, batch_size=batch_size, gpus=gpus)
    scores = np.asarray(output.scores, dtype=float)
    if len(scores) != len(samples):
        raise RuntimeError(f"COMET returned {len(scores)} scores for {len(samples)} samples.")

    scored = samples.copy()
    scored["COMET"] = scores
    scored.to_csv(score_path, index=False)
    meta = {
        "model": requested_model,
        "resolved_model": resolved_model,
        "sample_fingerprint": fingerprint,
        "n_samples": int(len(scored)),
        "batch_size": int(batch_size),
        "gpus": int(gpus),
        "unbabel_comet_version": package_version("unbabel-comet"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return scored, requested_model


def aggregate_comet(scored: pd.DataFrame, fwd: str, rev: str) -> pd.DataFrame:
    directional = (
        scored.groupby(["seed", "method", "label", "direction"], as_index=False)["COMET"]
        .mean()
        .rename(columns={"COMET": "COMET_direction"})
    )
    rows: List[Dict[str, Any]] = []
    for (seed, method, label), group in directional.groupby(
        ["seed", "method", "label"], sort=False
    ):
        by_dir = dict(zip(group["direction"].astype(str), group["COMET_direction"].astype(float)))
        cf, cr = numeric(by_dir.get(fwd)), numeric(by_dir.get(rev))
        rows.append({
            "seed": int(seed), "method": method, "label": label,
            "COMET_forward": cf, "COMET_reverse": cr,
            "COMET_mean": mean_two(cf, cr),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Paired bootstrap
# -----------------------------------------------------------------------------


def aligned_pair(
    samples: pd.DataFrame,
    seed: int,
    method_a: str,
    method_b: str,
    direction: str,
    include_comet: bool,
) -> pd.DataFrame:
    keys = ["direction", "src", "ref"]
    cols = keys + ["pred"] + (["COMET"] if include_comet else [])
    a = samples[
        (samples["seed"] == seed)
        & (samples["method"] == method_a)
        & (samples["direction"] == direction)
    ][cols].copy()
    b = samples[
        (samples["seed"] == seed)
        & (samples["method"] == method_b)
        & (samples["direction"] == direction)
    ][cols].copy()

    # Duplicate-safe pairing by within-key occurrence.
    a["_occurrence"] = a.groupby(keys, dropna=False).cumcount()
    b["_occurrence"] = b.groupby(keys, dropna=False).cumcount()
    merge_keys = keys + ["_occurrence"]
    rename_a = {"pred": "pred_a"}
    rename_b = {"pred": "pred_b"}
    if include_comet:
        rename_a["COMET"] = "COMET_a"
        rename_b["COMET"] = "COMET_b"
    a = a.rename(columns=rename_a)
    b = b.rename(columns=rename_b)
    merged = a.merge(b, on=merge_keys, how="inner", validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise RuntimeError(
            f"Pairing mismatch: seed={seed} {method_a} vs {method_b} direction={direction}; "
            f"A={len(a)}, B={len(b)}, matched={len(merged)}"
        )
    if merged.empty:
        raise RuntimeError(
            f"No paired samples: seed={seed} {method_a} vs {method_b} direction={direction}"
        )
    return merged.drop(columns="_occurrence").reset_index(drop=True)


def score_bootstrap_sample(
    frame: pd.DataFrame,
    indices: np.ndarray,
    include_comet: bool,
) -> np.ndarray:
    sampled = frame.iloc[indices]
    refs = sampled["ref"].astype(str).tolist()
    pred_a = sampled["pred_a"].astype(str).tolist()
    pred_b = sampled["pred_b"].astype(str).tolist()
    delta_bleu = corpus_bleu(pred_a, [refs]).score - corpus_bleu(pred_b, [refs]).score
    delta_chrf = corpus_chrf(pred_a, [refs]).score - corpus_chrf(pred_b, [refs]).score
    delta_comet = (
        float(sampled["COMET_a"].astype(float).mean() - sampled["COMET_b"].astype(float).mean())
        if include_comet else np.nan
    )
    return np.asarray([delta_bleu, delta_chrf, delta_comet], dtype=float)


def observed_pair_delta(frame: pd.DataFrame, include_comet: bool) -> np.ndarray:
    return score_bootstrap_sample(frame, np.arange(len(frame), dtype=int), include_comet)


def average_deltas(values: Sequence[np.ndarray], include_comet: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = np.empty(3, dtype=float)
    out[0] = float(np.mean(arr[:, 0]))
    out[1] = float(np.mean(arr[:, 1]))
    out[2] = float(np.mean(arr[:, 2])) if include_comet else np.nan
    return out


def percentile_ci(values: np.ndarray, confidence: float) -> Tuple[float, float]:
    alpha = 1.0 - confidence
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def tail_probability(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan
    p_le = float(np.mean(values <= 0.0))
    p_ge = float(np.mean(values >= 0.0))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))


def bootstrap_contrasts(
    samples: pd.DataFrame,
    seeds: Sequence[int],
    directions: Sequence[str],
    contrasts: Sequence[Tuple[str, str, str, str]],
    n_bootstrap: int,
    confidence: float,
    bootstrap_seed: int,
    include_comet: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(bootstrap_seed)
    per_seed_rows: List[Dict[str, Any]] = []
    overall_rows: List[Dict[str, Any]] = []

    aligned: Dict[Tuple[int, str, str, str], pd.DataFrame] = {}
    for _slug, _label, method_a, method_b in contrasts:
        for seed in seeds:
            for direction in directions:
                aligned[(seed, method_a, method_b, direction)] = aligned_pair(
                    samples, seed, method_a, method_b, direction, include_comet
                )

    for slug, label, method_a, method_b in contrasts:
        # Per-seed paired bootstrap.
        for seed in seeds:
            observed_dirs = [
                observed_pair_delta(aligned[(seed, method_a, method_b, d)], include_comet)
                for d in directions
            ]
            observed = average_deltas(observed_dirs, include_comet)
            reps = np.empty((n_bootstrap, 3), dtype=float)
            for b in range(n_bootstrap):
                dir_deltas = []
                for direction in directions:
                    frame = aligned[(seed, method_a, method_b, direction)]
                    idx = rng.integers(0, len(frame), size=len(frame))
                    dir_deltas.append(score_bootstrap_sample(frame, idx, include_comet))
                reps[b] = average_deltas(dir_deltas, include_comet)
            row: Dict[str, Any] = {
                "seed": int(seed), "contrast": slug, "label": label,
                "method_A": method_a, "method_B": method_b,
                "delta_convention": "method_A - method_B",
                "n_bootstrap": int(n_bootstrap), "confidence": confidence,
            }
            for j, metric in enumerate(("BLEU", "chrF", "COMET")):
                row[f"delta_{metric}"] = float(observed[j])
                if metric == "COMET" and not include_comet:
                    row[f"delta_{metric}_ci_low"] = np.nan
                    row[f"delta_{metric}_ci_high"] = np.nan
                    row[f"delta_{metric}_bootstrap_tail_prob"] = np.nan
                else:
                    vals = reps[:, j]
                    lo, hi = percentile_ci(vals, confidence)
                    row[f"delta_{metric}_ci_low"] = lo
                    row[f"delta_{metric}_ci_high"] = hi
                    row[f"delta_{metric}_bootstrap_tail_prob"] = tail_probability(vals)
            per_seed_rows.append(row)

        # Hierarchical seed + sentence bootstrap.
        observed_seed = []
        for seed in seeds:
            observed_dirs = [
                observed_pair_delta(aligned[(seed, method_a, method_b, d)], include_comet)
                for d in directions
            ]
            observed_seed.append(average_deltas(observed_dirs, include_comet))
        observed = average_deltas(observed_seed, include_comet)

        reps = np.empty((n_bootstrap, 3), dtype=float)
        seed_array = np.asarray(seeds, dtype=int)
        for b in range(n_bootstrap):
            sampled_seeds = rng.choice(seed_array, size=len(seed_array), replace=True)
            seed_deltas = []
            for sampled_seed in sampled_seeds:
                dir_deltas = []
                for direction in directions:
                    frame = aligned[(int(sampled_seed), method_a, method_b, direction)]
                    idx = rng.integers(0, len(frame), size=len(frame))
                    dir_deltas.append(score_bootstrap_sample(frame, idx, include_comet))
                seed_deltas.append(average_deltas(dir_deltas, include_comet))
            reps[b] = average_deltas(seed_deltas, include_comet)

        row = {
            "contrast": slug, "label": label,
            "method_A": method_a, "method_B": method_b,
            "delta_convention": "method_A - method_B",
            "n_seeds": len(seeds), "n_bootstrap": int(n_bootstrap),
            "confidence": confidence,
            "bootstrap_type": "hierarchical_paired_seed_and_sentence",
        }
        for j, metric in enumerate(("BLEU", "chrF", "COMET")):
            row[f"delta_{metric}"] = float(observed[j])
            if metric == "COMET" and not include_comet:
                row[f"delta_{metric}_ci_low"] = np.nan
                row[f"delta_{metric}_ci_high"] = np.nan
                row[f"delta_{metric}_bootstrap_tail_prob"] = np.nan
            else:
                vals = reps[:, j]
                lo, hi = percentile_ci(vals, confidence)
                row[f"delta_{metric}_ci_low"] = lo
                row[f"delta_{metric}_ci_high"] = hi
                row[f"delta_{metric}_bootstrap_tail_prob"] = tail_probability(vals)
        overall_rows.append(row)

    return pd.DataFrame(per_seed_rows), pd.DataFrame(overall_rows)


# -----------------------------------------------------------------------------
# Rankings and output formatting
# -----------------------------------------------------------------------------


def add_rankings(summary: pd.DataFrame, include_comet: bool) -> pd.DataFrame:
    out = summary.copy()
    for metric in ["BLEU_mean_mean", "chrF_mean_mean"] + (["COMET_mean_mean"] if include_comet else []):
        if metric in out.columns:
            out[f"rank_{metric.removesuffix('_mean')}"] = out[metric].rank(
                ascending=False, method="min", na_option="bottom"
            )
    if "validation_loss_mean" in out.columns:
        out["rank_validation_loss"] = out["validation_loss_mean"].rank(
            ascending=True, method="min", na_option="bottom"
        )
    return out


def print_compact_summary(summary: pd.DataFrame, include_comet: bool) -> None:
    cols = [
        "method", "label", "n_seeds",
        "BLEU_mean_mean", "BLEU_mean_std",
        "chrF_mean_mean", "chrF_mean_std",
    ]
    if include_comet:
        cols += ["COMET_mean_mean", "COMET_mean_std"]
    cols += ["trainable_params_mean", "peak_vram_GB_mean", "train_seconds_mean"]
    cols = [c for c in cols if c in summary.columns]
    print("\nMain-comparison effectiveness + efficiency summary:")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def print_bootstrap_summary(frame: pd.DataFrame, include_comet: bool, title: str) -> None:
    if frame.empty:
        return
    cols = [
        "label", "method_A", "method_B",
        "delta_BLEU", "delta_BLEU_ci_low", "delta_BLEU_ci_high",
        "delta_chrF", "delta_chrF_ci_low", "delta_chrF_ci_high",
    ]
    if include_comet:
        cols += ["delta_COMET", "delta_COMET_ci_low", "delta_COMET_ci_high"]
    print(f"\n{title} (positive delta favors method_A):")
    print(frame[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Post-hoc quality, efficiency, routing, COMET, bootstrap, and provenance analysis for the CRAT main comparison."
    )
    ap.add_argument("--root", required=True, help="CRAT root, e.g. $SCRATCH/crat")
    ap.add_argument("--pair", default="en-fr")
    ap.add_argument("--dataset", default="opus100")
    ap.add_argument("--seeds", default="42,64,123")
    ap.add_argument(
        "--methods",
        default=",".join(method for method, _ in METHODS),
        help="Comma-separated methods to load. CRAT must be included for CRAT-relative analysis.",
    )
    ap.add_argument("--run-mode", choices=["full", "smoke"], default="full")
    ap.add_argument("--discovery-seed", type=int, default=42)
    ap.add_argument("--morph-backend", default="stanza")
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--strict-cache-audit", action="store_true")
    ap.add_argument("--output-dir", default=None)

    ap.add_argument("--skip-comet", action="store_true")
    ap.add_argument("--comet-model", default="Unbabel/wmt22-comet-da")
    ap.add_argument("--comet-checkpoint", default=None)
    ap.add_argument("--comet-batch-size", type=int, default=32)
    ap.add_argument("--comet-gpus", type=int, default=1)
    ap.add_argument("--force-comet", action="store_true")

    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260816)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    if "-" not in args.pair:
        raise SystemExit("--pair must look like en-fr")
    src, tgt = args.pair.split("-", 1)
    if not src or not tgt:
        raise SystemExit("--pair must look like en-fr")
    seeds = parse_seed_list(args.seeds)
    methods = parse_csv_list(args.methods)
    unknown = [m for m in methods if m not in METHOD_LABEL]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}. Allowed: {list(METHOD_LABEL)}")
    if "crat" not in methods:
        raise SystemExit("CRAT must be included in --methods because it is the primary reference.")
    if not args.skip_bootstrap and args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples must be >= 100; 2000 is recommended.")
    if not 0.0 < args.confidence < 1.0:
        raise SystemExit("--confidence must be between 0 and 1.")

    root = Path(args.root).expanduser().resolve()
    fwd, rev = f"{src}-{tgt}", f"{tgt}-{src}"
    directions = [fwd, rev]
    seed_tag = "-".join(map(str, seeds))
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        comparison_root(root, args.run_mode) / "posthoc" / args.dataset / args.pair
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.pair}_posthoc_{args.dataset}_seeds{seed_tag}_main_comparison"

    print("=" * 90)
    print("CRAT MAIN-COMPARISON POST-HOC ANALYSIS")
    print("=" * 90)
    print(f"Root:       {root}")
    print(f"Run mode:   {args.run_mode}")
    print(f"Dataset:    {args.dataset}")
    print(f"Pair:       {args.pair}")
    print(f"Seeds:      {seeds}")
    print(f"Methods:    {methods}")
    print(f"Output dir: {out_dir}")

    per_seed, samples, input_problems = load_completed_runs(
        root=root, run_mode=args.run_mode, dataset=args.dataset, pair=args.pair,
        seeds=seeds, methods=methods, allow_missing=args.allow_missing,
    )
    print(f"Loaded result rows:  {len(per_seed)}")
    print(f"Loaded translations: {len(samples):,}")

    # Run configuration audit.
    config_audit = load_config_audit(
        root, args.run_mode, args.dataset, args.pair, seeds, methods
    )
    config_audit_path = out_dir / f"{prefix}_config_audit.csv"
    config_audit.to_csv(config_audit_path, index=False)

    # Shared cache provenance audit.
    cache_audit, shared_artifacts = audit_shared_artifact_reuse(
        root=root, run_mode=args.run_mode, dataset=args.dataset, pair=args.pair,
        seeds=seeds, methods=methods, morph_backend=args.morph_backend,
        discovery_seed=args.discovery_seed,
    )
    cache_audit_path = out_dir / f"{prefix}_shared_cache_reuse_audit.csv"
    cache_audit.to_csv(cache_audit_path, index=False)

    discovery_requested = any(m in DISCOVERY_METHODS for m in methods)
    morph_requested = any(m in MORPH_METHODS for m in methods)
    shared_discovery_ok = bool(
        shared_artifacts["discovery"].get("pair_ok")
        and shared_artifacts["discovery"].get("discovery_seed_ok")
    )
    shared_morph_ok = bool(shared_artifacts["morph"].get("pair_ok"))
    required_discovery_rows = cache_audit[cache_audit["requires_discovery"]]
    required_morph_rows = cache_audit[cache_audit["requires_morph"]]
    required_discovery_match = bool(
        required_discovery_rows["discovery_matches_shared"].all()
    ) if len(required_discovery_rows) else True
    required_morph_match = bool(
        required_morph_rows["morph_matches_shared"].all()
    ) if len(required_morph_rows) else True

    print("\nPair-shared artifact audit:")
    print(f"  Shared dir:                 {shared_artifacts['shared_dir']}")
    print(f"  Discovery pair/seed:        {'OK' if shared_discovery_ok else 'NOT VERIFIED'}")
    print(f"  Morphology pair:            {'OK' if shared_morph_ok else 'NOT VERIFIED'}")
    print(f"  Required discovery copies:  {'MATCH' if required_discovery_match else 'MISMATCH/MISSING'}")
    print(f"  Required morphology copies: {'MATCH' if required_morph_match else 'MISMATCH/MISSING'}")

    strict_cache_ok = (
        (not discovery_requested or (shared_discovery_ok and required_discovery_match))
        and (not morph_requested or (shared_morph_ok and required_morph_match))
    )
    if args.strict_cache_audit and not strict_cache_ok:
        raise SystemExit(f"Strict cache audit failed. Inspect {cache_audit_path}")

    # CRAT-relative deltas before optional COMET.
    per_seed = add_crat_relative_deltas(per_seed)

    # Optional COMET.
    include_comet = not args.skip_comet
    scored_samples = samples
    comet_per_seed_path = None
    comet_summary_path = None
    comet_model_id = None
    if include_comet:
        scored_samples, comet_model_id = compute_or_load_comet(
            samples=samples, output_dir=out_dir, prefix=prefix,
            comet_checkpoint=args.comet_checkpoint, comet_model_name=args.comet_model,
            batch_size=args.comet_batch_size, gpus=args.comet_gpus,
            force=args.force_comet,
        )
        comet_per_seed = aggregate_comet(scored_samples, fwd, rev)
        comet_per_seed_path = out_dir / f"{prefix}_comet_per_seed.csv"
        comet_per_seed.to_csv(comet_per_seed_path, index=False)
        per_seed = per_seed.merge(
            comet_per_seed, on=["seed", "method", "label"], how="left", validate="one_to_one"
        )
        per_seed["delta_COMET_vs_CRAT"] = np.nan
        for seed, idx in per_seed.groupby("seed").groups.items():
            ref = per_seed.loc[idx]
            ref = ref[ref["method"] == "crat"]
            if len(ref) != 1:
                raise RuntimeError(f"Seed {seed}: CRAT missing after COMET merge.")
            ref_val = numeric(ref.iloc[0]["COMET_mean"])
            for row_idx in idx:
                per_seed.loc[row_idx, "delta_COMET_vs_CRAT"] = (
                    numeric(per_seed.loc[row_idx, "COMET_mean"]) - ref_val
                )
        comet_summary = summarize_numeric(
            comet_per_seed, ["method", "label"],
            ["COMET_forward", "COMET_reverse", "COMET_mean"],
        )
        comet_summary["_order"] = comet_summary["method"].map(METHOD_ORDER)
        comet_summary = comet_summary.sort_values("_order").drop(columns="_order")
        comet_summary_path = out_dir / f"{prefix}_comet_summary.csv"
        comet_summary.to_csv(comet_summary_path, index=False)
        print(f"COMET model: {comet_model_id}")
    else:
        print("COMET: skipped")

    # Save full per-seed + summary.
    per_seed_path = out_dir / f"{prefix}_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)
    excluded = {"seed", "method", "label", "pair", "dataset", "run_mode", "run_dir", "result_file", "sample_file"}
    numeric_cols = [
        c for c in per_seed.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(per_seed[c])
    ]
    summary = summarize_numeric(per_seed, ["method", "label"], numeric_cols)
    summary["_order"] = summary["method"].map(METHOD_ORDER)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)
    summary = add_rankings(summary, include_comet)
    summary_path = out_dir / f"{prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Focused efficiency and routing files.
    efficiency_cols = [
        "seed", "method", "label", "BLEU_mean", "chrF_mean",
    ]
    if include_comet and "COMET_mean" in per_seed.columns:
        efficiency_cols.append("COMET_mean")
    efficiency_cols += [
        c for c in list(EFFICIENCY_METRICS) + [
            "delta_BLEU_vs_CRAT", "delta_chrF_vs_CRAT", "delta_COMET_vs_CRAT",
            "params_reduction_pct_vs_CRAT", "vram_reduction_pct_vs_CRAT",
            "train_time_reduction_pct_vs_CRAT", "generation_time_reduction_pct_vs_CRAT",
            "latency_mean_reduction_pct_vs_CRAT", "latency_p50_reduction_pct_vs_CRAT",
            "latency_p95_reduction_pct_vs_CRAT", "train_throughput_gain_pct_vs_CRAT",
            "generation_throughput_gain_pct_vs_CRAT", "benchmark_throughput_gain_pct_vs_CRAT",
        ] if c in per_seed.columns
    ]
    efficiency_per_seed = per_seed[efficiency_cols].copy()
    efficiency_per_seed_path = out_dir / f"{prefix}_efficiency_per_seed.csv"
    efficiency_per_seed.to_csv(efficiency_per_seed_path, index=False)
    efficiency_metrics = [c for c in efficiency_per_seed.columns if c not in {"seed", "method", "label"}]
    efficiency_summary = summarize_numeric(
        efficiency_per_seed, ["method", "label"], efficiency_metrics
    )
    efficiency_summary["_order"] = efficiency_summary["method"].map(METHOD_ORDER)
    efficiency_summary = efficiency_summary.sort_values("_order").drop(columns="_order")
    efficiency_summary_path = out_dir / f"{prefix}_efficiency_summary.csv"
    efficiency_summary.to_csv(efficiency_summary_path, index=False)

    routing_per_seed = per_seed[
        ["seed", "method", "label"] + [c for c in ROUTING_METRICS if c in per_seed.columns]
    ].copy()
    routing_per_seed = routing_per_seed[routing_per_seed["method"].isin(ROUTER_METHODS)]
    routing_per_seed_path = out_dir / f"{prefix}_routing_per_seed.csv"
    routing_per_seed.to_csv(routing_per_seed_path, index=False)
    routing_summary = summarize_numeric(
        routing_per_seed, ["method", "label"],
        [c for c in ROUTING_METRICS if c in routing_per_seed.columns],
    )
    routing_summary["_order"] = routing_summary["method"].map(METHOD_ORDER)
    routing_summary = routing_summary.sort_values("_order").drop(columns="_order")
    routing_summary_path = out_dir / f"{prefix}_routing_summary.csv"
    routing_summary.to_csv(routing_summary_path, index=False)

    # Paired bootstrap: CRAT versus every other requested method.
    crat_contrasts = [
        (f"crat_vs_{method}", f"CRAT - {METHOD_LABEL[method]}", "crat", method)
        for method in methods if method != "crat"
    ]
    planned_contrasts = [
        spec for spec in PLANNED_CONTRASTS
        if spec[2] in methods and spec[3] in methods
    ]

    bootstrap_crat_per_seed_path = None
    bootstrap_crat_overall_path = None
    bootstrap_planned_per_seed_path = None
    bootstrap_planned_overall_path = None
    bootstrap_crat_overall = pd.DataFrame()
    bootstrap_planned_overall = pd.DataFrame()
    if not args.skip_bootstrap:
        bootstrap_crat_per_seed, bootstrap_crat_overall = bootstrap_contrasts(
            samples=scored_samples, seeds=seeds, directions=directions,
            contrasts=crat_contrasts, n_bootstrap=args.bootstrap_samples,
            confidence=args.confidence, bootstrap_seed=args.bootstrap_seed,
            include_comet=include_comet,
        )
        bootstrap_crat_per_seed_path = out_dir / f"{prefix}_bootstrap_crat_vs_methods_per_seed.csv"
        bootstrap_crat_overall_path = out_dir / f"{prefix}_bootstrap_crat_vs_methods.csv"
        bootstrap_crat_per_seed.to_csv(bootstrap_crat_per_seed_path, index=False)
        bootstrap_crat_overall.to_csv(bootstrap_crat_overall_path, index=False)

        if planned_contrasts:
            bootstrap_planned_per_seed, bootstrap_planned_overall = bootstrap_contrasts(
                samples=scored_samples, seeds=seeds, directions=directions,
                contrasts=planned_contrasts, n_bootstrap=args.bootstrap_samples,
                confidence=args.confidence,
                bootstrap_seed=args.bootstrap_seed + 7919,
                include_comet=include_comet,
            )
            bootstrap_planned_per_seed_path = out_dir / f"{prefix}_bootstrap_planned_objective_contrasts_per_seed.csv"
            bootstrap_planned_overall_path = out_dir / f"{prefix}_bootstrap_planned_objective_contrasts.csv"
            bootstrap_planned_per_seed.to_csv(bootstrap_planned_per_seed_path, index=False)
            bootstrap_planned_overall.to_csv(bootstrap_planned_overall_path, index=False)
    else:
        print("Paired bootstrap: skipped")

    metadata = {
        "root": str(root), "run_mode": args.run_mode,
        "dataset": args.dataset, "pair": args.pair,
        "seeds": seeds, "methods": methods,
        "primary_reference": "crat",
        "quality_delta_convention": "Method - CRAT",
        "cost_reduction_convention": "100*(CRAT-Method)/CRAT",
        "bootstrap_delta_convention": "method_A - method_B",
        "planned_objective_contrasts": [
            {"slug": s, "label": l, "method_A": a, "method_B": b}
            for s, l, a, b in planned_contrasts
        ],
        "comet_enabled": include_comet,
        "comet_model": comet_model_id,
        "bootstrap_enabled": not args.skip_bootstrap,
        "bootstrap_samples": None if args.skip_bootstrap else args.bootstrap_samples,
        "bootstrap_seed": None if args.skip_bootstrap else args.bootstrap_seed,
        "confidence": None if args.skip_bootstrap else args.confidence,
        "discovery_seed": args.discovery_seed,
        "morph_backend": args.morph_backend,
        "shared_artifacts": shared_artifacts,
        "strict_cache_audit": bool(args.strict_cache_audit),
        "strict_cache_audit_passed": strict_cache_ok,
        "input_problem_count": len(input_problems),
        "config_audit_file": str(config_audit_path),
        "cache_audit_file": str(cache_audit_path),
        "sacrebleu_version": package_version("sacrebleu"),
        "pandas_version": package_version("pandas"),
        "numpy_version": package_version("numpy"),
        "unbabel_comet_version": package_version("unbabel-comet") if include_comet else None,
        "note_on_three_seeds": (
            "Hierarchical bootstrap includes seed resampling, but with only three training seeds "
            "between-seed uncertainty is necessarily estimated imprecisely."
        ),
    }
    metadata_path = out_dir / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print_compact_summary(summary, include_comet)
    print_bootstrap_summary(
        bootstrap_crat_overall, include_comet,
        "Hierarchical paired bootstrap: CRAT vs other methods",
    )
    print_bootstrap_summary(
        bootstrap_planned_overall, include_comet,
        "Hierarchical paired bootstrap: planned CRAT objective contrasts",
    )

    print("\nSaved files:")
    for path in [
        per_seed_path, summary_path,
        efficiency_per_seed_path, efficiency_summary_path,
        routing_per_seed_path, routing_summary_path,
        config_audit_path, cache_audit_path,
        out_dir / f"{prefix}_comet_sentence_scores.csv" if include_comet else None,
        comet_per_seed_path, comet_summary_path,
        bootstrap_crat_per_seed_path, bootstrap_crat_overall_path,
        bootstrap_planned_per_seed_path, bootstrap_planned_overall_path,
        metadata_path,
    ]:
        if path is not None:
            print(f"  {path}")


if __name__ == "__main__":
    main()
