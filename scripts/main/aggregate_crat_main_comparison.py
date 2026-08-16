#!/usr/bin/env python3
"""Aggregate completed CRAT main-comparison runs across training seeds.

Expected layout (created by ``crat_mistral_trillium_pair.sh``)::

    $ROOT/outputs/main_comparison/
      mistral7b_<dataset>_<pairtag>_seed<seed>_<method>/
        <pair>_seed<seed>_<method>_comparison_results.csv
        <pair>_seed<seed>_<method>_translation_samples.csv
        <method>/
          <pair>_seed<seed>_<method>_metadata.json
          <pair>_seed<seed>_<method>_loss_log.csv
          ...

For ``--run-mode smoke`` the same structure is read from
``outputs/main_comparison_smoke``.

This script performs aggregation only; it does not retrain or rescore models.
It writes one per-seed table plus across-seed mean/std/SEM/95% t-CI summaries.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# method, display label
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


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def require_columns(df: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")


def comparison_root(root: Path, run_mode: str) -> Path:
    name = "main_comparison_smoke" if run_mode == "smoke" else "main_comparison"
    return root / "outputs" / name


def run_dir(
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


def result_candidates(directory: Path, pair: str, seed: int, method: str) -> List[Path]:
    """Exact new filename first; permissive glob only as a transparent fallback."""
    exact = directory / f"{artifact_prefix(pair, seed, method)}_comparison_results.csv"
    candidates = [exact]
    for path in sorted(directory.glob("*_comparison_results.csv")):
        if path not in candidates:
            candidates.append(path)
    return candidates


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def mean_two(a: float, b: float) -> float:
    values = np.asarray([a, b], dtype=float)
    good = values[np.isfinite(values)]
    return float(good.mean()) if len(good) else float("nan")


def t_critical_975(df: int) -> float:
    # Exact common values; asymptotic normal fallback.  This avoids a scipy
    # dependency for a simple descriptive aggregation utility.
    table = {
        1: 12.706204736,
        2: 4.302652730,
        3: 3.182446305,
        4: 2.776445105,
        5: 2.570581836,
        6: 2.446911851,
        7: 2.364624252,
        8: 2.306004135,
        9: 2.262157163,
        10: 2.228138852,
        11: 2.200985160,
        12: 2.178812830,
        13: 2.160368656,
        14: 2.144786688,
        15: 2.131449546,
        16: 2.119905299,
        17: 2.109815578,
        18: 2.100922040,
        19: 2.093024054,
        20: 2.085963447,
        25: 2.059538553,
        30: 2.042272456,
        40: 2.021075390,
        60: 2.000297822,
        120: 1.979930406,
    }
    if df in table:
        return table[df]
    if df <= 0:
        return float("nan")
    if df < 25:
        return table[max(k for k in table if k <= 20 and k < df)]
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
            n = int(len(vals))
            row[f"{col}_n"] = n
            if n == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_std"] = np.nan
                row[f"{col}_sem"] = np.nan
                row[f"{col}_ci95_low"] = np.nan
                row[f"{col}_ci95_high"] = np.nan
                continue
            mean = float(vals.mean())
            std = float(vals.std(ddof=1)) if n > 1 else float("nan")
            sem = std / math.sqrt(n) if n > 1 else float("nan")
            crit = t_critical_975(n - 1) if n > 1 else float("nan")
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_sem"] = sem
            row[f"{col}_ci95_low"] = mean - crit * sem if n > 1 else np.nan
            row[f"{col}_ci95_high"] = mean + crit * sem if n > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_main_results(
    root: Path,
    run_mode: str,
    dataset: str,
    pair: str,
    seeds: Sequence[int],
    methods: Sequence[str],
    strict: bool,
) -> Tuple[pd.DataFrame, List[str]]:
    src, tgt = pair.split("-", 1)
    pair_tag = f"{src}{tgt}"
    fwd, rev = f"{src}-{tgt}", f"{tgt}-{src}"

    rows: List[Dict[str, Any]] = []
    problems: List[str] = []

    for seed in seeds:
        for method in methods:
            directory = run_dir(root, run_mode, dataset, pair_tag, seed, method)
            candidates = result_candidates(directory, pair, seed, method)
            path = first_existing(candidates)
            if path is None:
                problems.append(
                    f"missing results: seed={seed} method={method}; checked "
                    + ", ".join(str(p) for p in candidates)
                )
                continue

            frame = pd.read_csv(path)
            require_columns(frame, ["method", f"BLEU_{fwd}", f"BLEU_{rev}", f"chrF_{fwd}", f"chrF_{rev}"], path)
            hit = frame[frame["method"].astype(str) == method]
            if hit.empty:
                problems.append(f"method={method!r} not found in {path}")
                continue
            if len(hit) > 1:
                problems.append(f"multiple rows for method={method!r} in {path}; using last row")
            raw = hit.iloc[-1].to_dict()

            observed_pair = str(raw.get("pair", pair))
            observed_seed = numeric(raw.get("seed", seed))
            if observed_pair != pair:
                problems.append(f"pair mismatch in {path}: observed={observed_pair}, expected={pair}")
            if np.isfinite(observed_seed) and int(observed_seed) != int(seed):
                problems.append(f"seed mismatch in {path}: observed={observed_seed}, expected={seed}")

            row: Dict[str, Any] = {
                "pair": pair,
                "dataset": dataset,
                "run_mode": run_mode,
                "seed": int(seed),
                "method": method,
                "label": METHOD_LABEL[method],
                "run_dir": str(directory),
                "result_file": str(path),
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
            rows.append(row)

    if problems:
        print("\nInput diagnostics:")
        for item in problems:
            print(f"  - {item}")
        if strict:
            raise SystemExit(f"Strict mode failed with {len(problems)} input problem(s).")

    if not rows:
        raise SystemExit("No main-comparison results were found.")

    per_seed = pd.DataFrame(rows)
    per_seed["_order"] = per_seed["method"].map(METHOD_ORDER)
    per_seed = per_seed.sort_values(["seed", "_order"]).drop(columns="_order").reset_index(drop=True)

    if strict:
        expected = len(seeds) * len(methods)
        if len(per_seed) != expected:
            raise SystemExit(f"Expected {expected} method-seed rows, found {len(per_seed)}.")

    return per_seed, problems


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate CRAT main-comparison runs across seeds.")
    ap.add_argument("--root", required=True, help="CRAT root, e.g. $SCRATCH/crat")
    ap.add_argument("--pair", default="en-fr")
    ap.add_argument("--dataset", default="opus100")
    ap.add_argument("--seeds", default="42,64,123")
    ap.add_argument(
        "--methods",
        default=",".join(method for method, _ in METHODS),
        help="Comma-separated subset of main-comparison methods.",
    )
    ap.add_argument("--run-mode", choices=["full", "smoke"], default="full")
    ap.add_argument("--strict", action="store_true", help="Fail on missing/mismatched requested runs.")
    ap.add_argument("--output-dir", default=None, help="Optional output directory override.")
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

    root = Path(args.root).expanduser().resolve()
    per_seed, problems = load_main_results(
        root=root,
        run_mode=args.run_mode,
        dataset=args.dataset,
        pair=args.pair,
        seeds=seeds,
        methods=methods,
        strict=args.strict,
    )

    quality_metrics = [
        "BLEU_forward", "BLEU_reverse", "BLEU_mean",
        "chrF_forward", "chrF_reverse", "chrF_mean",
        "validation_loss",
        "empty_rate_forward", "empty_rate_reverse", "empty_rate_mean",
        "punctuation_collapse_rate_forward", "punctuation_collapse_rate_reverse",
        "punctuation_collapse_rate_mean",
    ]
    metric_cols = quality_metrics + list(EFFICIENCY_METRICS) + list(ROUTING_METRICS)
    summary = summarize_numeric(per_seed, ["method", "label"], metric_cols)
    summary["_order"] = summary["method"].map(METHOD_ORDER)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    base_out = Path(args.output_dir).expanduser().resolve() if args.output_dir else (
        comparison_root(root, args.run_mode) / "aggregate" / args.dataset / args.pair
    )
    base_out.mkdir(parents=True, exist_ok=True)
    seed_tag = "-".join(map(str, seeds))
    prefix = f"{args.pair}_{args.dataset}_seeds{seed_tag}_main_comparison"

    per_seed_path = base_out / f"{prefix}_per_seed.csv"
    summary_path = base_out / f"{prefix}_summary.csv"
    per_seed.to_csv(per_seed_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Useful focused summaries.
    quality_summary = summary[[
        c for c in summary.columns
        if c in {"method", "label", "n_seeds"}
        or any(c.startswith(f"{metric}_") for metric in quality_metrics)
    ]].copy()
    quality_summary_path = base_out / f"{prefix}_quality_summary.csv"
    quality_summary.to_csv(quality_summary_path, index=False)

    efficiency_summary = summary[[
        c for c in summary.columns
        if c in {"method", "label", "n_seeds"}
        or any(c.startswith(f"{metric}_") for metric in EFFICIENCY_METRICS)
    ]].copy()
    efficiency_summary_path = base_out / f"{prefix}_efficiency_summary.csv"
    efficiency_summary.to_csv(efficiency_summary_path, index=False)

    routing_summary = summary[[
        c for c in summary.columns
        if c in {"method", "label", "n_seeds"}
        or any(c.startswith(f"{metric}_") for metric in ROUTING_METRICS)
    ]].copy()
    routing_summary_path = base_out / f"{prefix}_routing_summary.csv"
    routing_summary.to_csv(routing_summary_path, index=False)

    metadata = {
        "root": str(root),
        "pair": args.pair,
        "dataset": args.dataset,
        "run_mode": args.run_mode,
        "seeds": seeds,
        "methods": methods,
        "strict": bool(args.strict),
        "input_problem_count": len(problems),
        "pandas_version": package_version("pandas"),
        "numpy_version": package_version("numpy"),
    }
    metadata_path = base_out / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    compact = [
        "method", "label", "n_seeds",
        "BLEU_mean_mean", "BLEU_mean_std",
        "chrF_mean_mean", "chrF_mean_std",
        "validation_loss_mean",
        "trainable_params_mean", "peak_vram_GB_mean", "train_seconds_mean",
    ]
    compact = [c for c in compact if c in summary.columns]
    print("\nAcross-seed main-comparison summary:")
    print(summary[compact].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nSaved:")
    for path in [per_seed_path, summary_path, quality_summary_path, efficiency_summary_path, routing_summary_path, metadata_path]:
        print(f"  {path}")


if __name__ == "__main__":
    main()
