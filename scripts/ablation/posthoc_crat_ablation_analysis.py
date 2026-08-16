#!/usr/bin/env python3
"""
Post-hoc analysis for completed CRAT ablation experiments.

This script DOES NOT retrain any model. It reads the completed ablation outputs
created by the refined variant-aware ablation pipeline:

  outputs/ablations/mistral7b_<dataset>_<pairtag>_seed<seed>_<ablation>/
      <pair>_seed<seed>_<variant>_comparison_results.csv
      <pair>_seed<seed>_<variant>_translation_samples.csv
      <pair>_morph_cache_<backend>.pkl
      <pair>_component_discovery.json
      <variant>/
          <pair>_seed<seed>_<variant>_metadata.json
          <pair>_seed<seed>_<variant>_loss_log.csv
          <pair>_seed<seed>_<variant>_trainable_state.pt

Pair-shared expensive artifacts are expected under:

  outputs/ablations/_shared/<dataset>/<pair>/
      <pair>_morph_cache_<backend>.pkl
      <pair>_component_discovery.json
      <pair>_probe_examples_<hash>.json

The loader prefers the refined filenames but retains compatibility with the
older named and original generic filenames.

and produces:

1) Extended effectiveness/efficiency aggregation
   - BLEU / chrF / validation loss
   - trainable parameters
   - peak VRAM
   - training time
   - training throughput
   - generation throughput
   - latency mean / p50 / p95
   - benchmark generation throughput
   - deltas/reductions relative to Full CRAT within each seed

2) CRAT-specific routing aggregation
   - train/eval gate means for source/indicator/position/integration
   - eval gate activation rates at threshold 0.5
   - train router entropy
   - eval mean active soft roles at threshold 0.5

3) Optional COMET scoring from translation_samples.csv
   - reference-based COMET sentence scores
   - per-direction and bidirectional means
   - configurable local checkpoint for offline clusters

4) Paired bootstrap Full-CRAT-vs-ablation comparison
   - BLEU delta + 95% CI
   - chrF delta + 95% CI
   - COMET delta + 95% CI when COMET is enabled/available
   - hierarchical paired bootstrap over seeds and test sentences

5) Pair-specific shared-artifact provenance audit
   - validates the shared SIPP discovery language pair and discovery seed
   - validates morphology-cache language keys
   - compares SHA-256 hashes of per-run copies against the pair-shared files

6) Combined-ablation interaction analysis
   - sparse × morphology
   - sparse × integration
   - morphology × integration
   - third-order sparse × morphology × integration non-additivity

Delta convention for bootstrap:
    delta = Full CRAT - Variant
so a positive delta means Full CRAT scores higher.

Quality/cost delta convention in the per-seed tables:
    delta_quality_vs_CRAT = Variant - Full CRAT
so a negative value means the ablation scores lower.

The bootstrap is paired by (seed, direction, src, ref). The same sampled sentence
indices are used for Full CRAT and the compared variant.
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


ABLATIONS: Tuple[Tuple[str, str, str, str], ...] = (
    # slug, display label, internal Python method, externally visible variant name
    ("full", "CRAT", "crat", "full_crat"),
    ("no_kl", "CRAT - KL", "crat", "no_kl"),
    ("no_causal", "CRAT - causal", "crat", "no_causal"),
    ("no_morph", "CRAT - morph", "crat", "no_morph"),
    ("no_sparse", "CRAT - sparse", "crat", "no_sparse"),
    ("no_integration", "CRAT - integration", "crat", "no_integration"),
    ("static_role", "Static-role SFT", "static_role_sft", "static_role_sft"),
    (
        "no_sparse_morph_integration",
        "CRAT - sparse - morph - integration",
        "crat",
        "no_sparse_morph_integration",
    ),
    ("no_sparse_morph", "CRAT - sparse - morph", "crat", "no_sparse_morph"),
    (
        "no_sparse_integration",
        "CRAT - sparse - integration",
        "crat",
        "no_sparse_integration",
    ),
    (
        "no_morph_integration",
        "CRAT - morph - integration",
        "crat",
        "no_morph_integration",
    ),
)


ROLES: Tuple[str, ...] = ("source", "indicator", "position", "integration")

EFFICIENCY_METRICS: Tuple[str, ...] = (
    "trainable_params",
    "total_params",
    "trainable_pct",
    "peak_vram_GB",
    "train_seconds",
    "train_tokens",
    "train_tokens_per_second",
    "generation_seconds",
    "generated_tokens",
    "generation_tokens_per_second",
    "latency_mean_seconds",
    "latency_p50_seconds",
    "latency_p95_seconds",
    "benchmark_tokens_per_second",
    "updates",
    "effective_batch_size",
)

ROUTING_METRICS: Tuple[str, ...] = tuple(
    [f"train_gate_mean_{r}" for r in ROLES]
    + ["train_router_entropy"]
    + [f"eval_gate_mean_{r}" for r in ROLES]
    + [f"eval_gate_active_{r}@0.5" for r in ROLES]
    + ["eval_mean_active_roles@0.5"]
)

QUALITY_METRICS: Tuple[str, ...] = (
    "BLEU_forward",
    "BLEU_reverse",
    "BLEU_mean",
    "chrF_forward",
    "chrF_reverse",
    "chrF_mean",
    "validation_loss",
)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def parse_seed_list(text: str) -> List[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one seed is required.")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate seeds are not allowed: {values}")
    return values


def numeric(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def safe_pct_change(new: float, reference: float) -> float:
    """100 * (new-reference)/reference; NaN if reference is zero/non-finite."""
    if not np.isfinite(new) or not np.isfinite(reference) or reference == 0:
        return float("nan")
    return 100.0 * (new - reference) / reference


def safe_reduction_pct(value: float, reference: float) -> float:
    """100 * (reference-value)/reference; positive means cost reduction."""
    if not np.isfinite(value) or not np.isfinite(reference) or reference == 0:
        return float("nan")
    return 100.0 * (reference - value) / reference


def sample_fingerprint(df: pd.DataFrame) -> str:
    cols = ["seed", "ablation"]
    if "variant" in df.columns:
        cols.append("variant")
    cols += ["direction", "src", "ref", "pred"]
    payload = df[cols].astype(str).sort_values(cols[:-1]).to_csv(index=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def package_version(name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


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
        row["n_seeds"] = int(group["seed"].nunique()) if "seed" in group else len(group)
        for col in metric_cols:
            if col not in group.columns:
                continue
            vals = pd.to_numeric(group[col], errors="coerce")
            good = vals.dropna()
            row[f"{col}_mean"] = float(good.mean()) if len(good) else float("nan")
            row[f"{col}_std"] = float(good.std(ddof=1)) if len(good) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


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


def first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.is_file():
            return path
    return None


def artifact_candidates(
    run_dir: Path,
    pair: str,
    seed: int,
    slug: str,
    internal_method: str,
    variant_name: str,
    kind: str,
) -> List[Path]:
    """Return refined-to-legacy candidate paths for a run-level CSV artifact."""
    if kind not in {"comparison_results", "translation_samples"}:
        raise ValueError(f"Unknown artifact kind: {kind}")
    candidates = [
        # Refined v2 format.
        run_dir / f"{pair}_seed{seed}_{variant_name}_{kind}.csv",
        # Earlier method-aware named format.
        run_dir / f"{pair}_seed{seed}_{slug}_{internal_method}_{kind}.csv",
        # Original generic format.
        run_dir / f"{kind}.csv",
    ]
    # Last-resort compatibility: accept a single unambiguous file of this type.
    globbed = sorted(run_dir.glob(f"*{kind}.csv"))
    for path in globbed:
        if path not in candidates:
            candidates.append(path)
    return candidates


# -----------------------------------------------------------------------------
# Load completed runs
# -----------------------------------------------------------------------------


def experiment_dir(root: Path, dataset: str, pair_tag: str, seed: int, slug: str) -> Path:
    return (
        root
        / "outputs"
        / "ablations"
        / f"mistral7b_{dataset}_{pair_tag}_seed{seed}_{slug}"
    )


def select_variant_rows(
    frame: pd.DataFrame,
    internal_method: str,
    variant_name: str,
) -> pd.DataFrame:
    """Prefer exact method+variant matching; gracefully support legacy files."""
    require = frame["method"].astype(str) == internal_method
    hit = frame[require].copy()
    if hit.empty:
        return hit
    if "variant" in hit.columns:
        exact = hit[hit["variant"].astype(str) == variant_name]
        if not exact.empty:
            return exact
    return hit


def load_completed_runs(
    root: Path,
    dataset: str,
    pair: str,
    pair_tag: str,
    fwd: str,
    rev: str,
    seeds: Sequence[int],
    allow_missing: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    result_rows: List[Dict[str, Any]] = []
    sample_frames: List[pd.DataFrame] = []
    missing_items: List[str] = []

    order = {slug: i for i, (slug, *_rest) in enumerate(ABLATIONS)}

    for seed in seeds:
        for slug, label, internal_method, variant_name in ABLATIONS:
            run_dir = experiment_dir(root, dataset, pair_tag, seed, slug)
            result_candidates = artifact_candidates(
                run_dir, pair, seed, slug, internal_method, variant_name,
                "comparison_results",
            )
            sample_candidates = artifact_candidates(
                run_dir, pair, seed, slug, internal_method, variant_name,
                "translation_samples",
            )
            results_path = first_existing(result_candidates)
            samples_path = first_existing(sample_candidates)

            if results_path is None:
                missing_items.append(
                    "missing results: "
                    f"pair={pair} seed={seed} ablation={slug}; checked "
                    + ", ".join(str(p) for p in result_candidates)
                )
                continue
            if samples_path is None:
                missing_items.append(
                    "missing translations: "
                    f"pair={pair} seed={seed} ablation={slug}; checked "
                    + ", ".join(str(p) for p in sample_candidates)
                )
                continue

            results = pd.read_csv(results_path)
            require_columns(
                results,
                [
                    "method",
                    f"BLEU_{fwd}", f"BLEU_{rev}",
                    f"chrF_{fwd}", f"chrF_{rev}",
                ],
                results_path,
            )
            hit = select_variant_rows(results, internal_method, variant_name)
            if hit.empty:
                missing_items.append(
                    f"method={internal_method!r}, variant={variant_name!r} "
                    f"not found in {results_path}"
                )
                continue
            raw = hit.iloc[-1].to_dict()

            row: Dict[str, Any] = {
                "pair": pair,
                "seed": int(seed),
                "ablation": slug,
                "variant": str(raw.get("variant", variant_name)),
                "label": label,
                "method": internal_method,
                "run_dir": str(run_dir),
                "result_file": str(results_path),
                "sample_file": str(samples_path),
                "BLEU_forward": numeric(raw.get(f"BLEU_{fwd}")),
                "BLEU_reverse": numeric(raw.get(f"BLEU_{rev}")),
                "chrF_forward": numeric(raw.get(f"chrF_{fwd}")),
                "chrF_reverse": numeric(raw.get(f"chrF_{rev}")),
                "validation_loss": numeric(raw.get("validation_loss")),
            }
            row["BLEU_mean"] = float(
                np.nanmean([row["BLEU_forward"], row["BLEU_reverse"]])
            )
            row["chrF_mean"] = float(
                np.nanmean([row["chrF_forward"], row["chrF_reverse"]])
            )

            for col in EFFICIENCY_METRICS + ROUTING_METRICS:
                row[col] = numeric(raw.get(col))
            result_rows.append(row)

            samples = pd.read_csv(samples_path)
            require_columns(
                samples,
                ["method", "direction", "src", "ref", "pred"],
                samples_path,
            )
            samples = select_variant_rows(
                samples, internal_method, variant_name
            ).copy()
            if samples.empty:
                missing_items.append(
                    f"no translation samples for method={internal_method!r}, "
                    f"variant={variant_name!r} in {samples_path}"
                )
                continue

            # Preserve the externally visible variant name even for legacy files.
            samples["variant"] = variant_name
            samples["seed"] = int(seed)
            samples["ablation"] = slug
            samples["label"] = label
            samples["run_dir"] = str(run_dir)
            samples["sample_file"] = str(samples_path)
            sample_frames.append(
                samples[
                    [
                        "seed", "ablation", "variant", "label", "method",
                        "run_dir", "sample_file", "direction", "src", "ref", "pred",
                    ]
                ].copy()
            )

    if missing_items:
        print("\nInput problems detected:", file=sys.stderr)
        for item in missing_items:
            print(f"  - {item}", file=sys.stderr)
        if not allow_missing:
            expected = len(seeds) * len(ABLATIONS)
            raise SystemExit(
                f"Aborting because one or more requested ablation artifacts are missing. "
                f"The refined design expects {expected} runs "
                f"({len(ABLATIONS)} variants x {len(seeds)} seeds). "
                "Use --allow-missing only if incomplete analysis is intentional."
            )

    if not result_rows:
        raise SystemExit("No completed ablation results were found.")
    if not sample_frames:
        raise SystemExit("No translation-sample data were found.")

    per_seed = pd.DataFrame(result_rows)
    per_seed["_order"] = per_seed["ablation"].map(order)
    per_seed = (
        per_seed.sort_values(["seed", "_order"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )
    samples_all = pd.concat(sample_frames, ignore_index=True)

    if not allow_missing:
        expected = len(seeds) * len(ABLATIONS)
        if len(per_seed) != expected:
            raise SystemExit(
                f"Expected {expected} result rows "
                f"({len(ABLATIONS)} variants x {len(seeds)} seeds), "
                f"but found {len(per_seed)}."
            )

    return per_seed, samples_all


# -----------------------------------------------------------------------------
# Pair-shared artifact provenance audit
# -----------------------------------------------------------------------------


def shared_pair_dir(root: Path, dataset: str, pair: str) -> Path:
    return root / "outputs" / "ablations" / "_shared" / dataset / pair


def validate_shared_discovery(
    path: Path,
    src: str,
    tgt: str,
    discovery_seed: int,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
        "pair_ok": False,
        "discovery_seed_ok": False,
    }
    if not path.is_file():
        return info
    payload = json.loads(path.read_text(encoding="utf-8"))
    signature = payload.get("signature", {})
    info.update(
        {
            "source_lang": signature.get("source_lang"),
            "target_lang": signature.get("target_lang"),
            "discovery_seed": signature.get("seed"),
            "signature_hash": payload.get("signature_hash"),
            "selected_head_count": payload.get("selected_head_count"),
            "integration_layers": payload.get("integration_layers"),
        }
    )
    info["pair_ok"] = (
        signature.get("source_lang") == src
        and signature.get("target_lang") == tgt
    )
    try:
        info["discovery_seed_ok"] = int(signature.get("seed")) == int(discovery_seed)
    except (TypeError, ValueError):
        info["discovery_seed_ok"] = False
    return info


def validate_shared_morph(
    path: Path,
    src: str,
    tgt: str,
) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "sha256": sha256_file(path),
        "languages": [],
        "pair_ok": False,
        "entries": None,
    }
    if not path.is_file():
        return info
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    languages = sorted(
        {
            str(key).split(":", 1)[0].lower()
            for key in cache.keys()
        }
    )
    info["languages"] = languages
    info["entries"] = len(cache)
    # Refined bidirectional runs normally contain both languages and must
    # never contain an unrelated language.
    allowed = {src.lower(), tgt.lower()}
    info["pair_ok"] = set(languages).issubset(allowed) and allowed.issubset(set(languages))
    return info


def audit_shared_artifact_reuse(
    root: Path,
    dataset: str,
    pair: str,
    pair_tag: str,
    seeds: Sequence[int],
    morph_backend: str,
    discovery_seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    src, tgt = pair.split("-", 1)
    shared_dir = shared_pair_dir(root, dataset, pair)
    shared_morph = shared_dir / f"{pair}_morph_cache_{morph_backend}.pkl"
    shared_discovery = shared_dir / f"{pair}_component_discovery.json"

    shared_info = {
        "shared_dir": str(shared_dir),
        "morph": validate_shared_morph(shared_morph, src, tgt),
        "discovery": validate_shared_discovery(
            shared_discovery, src, tgt, discovery_seed
        ),
        "probe_files": [
            str(p) for p in sorted(shared_dir.glob(f"{pair}_probe_examples_*.json"))
        ],
    }

    shared_morph_hash = shared_info["morph"].get("sha256")
    shared_discovery_hash = shared_info["discovery"].get("sha256")

    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for slug, label, internal_method, variant_name in ABLATIONS:
            run_dir = experiment_dir(root, dataset, pair_tag, seed, slug)
            run_morph = run_dir / f"{pair}_morph_cache_{morph_backend}.pkl"
            run_discovery = run_dir / f"{pair}_component_discovery.json"
            run_morph_hash = sha256_file(run_morph)
            run_discovery_hash = sha256_file(run_discovery)
            rows.append(
                {
                    "pair": pair,
                    "seed": int(seed),
                    "ablation": slug,
                    "variant": variant_name,
                    "label": label,
                    "method": internal_method,
                    "run_dir": str(run_dir),
                    "run_morph_file": str(run_morph),
                    "run_morph_exists": run_morph.is_file(),
                    "run_morph_sha256": run_morph_hash,
                    "morph_matches_shared": (
                        run_morph_hash is not None
                        and shared_morph_hash is not None
                        and run_morph_hash == shared_morph_hash
                    ),
                    "run_discovery_file": str(run_discovery),
                    "run_discovery_exists": run_discovery.is_file(),
                    "run_discovery_sha256": run_discovery_hash,
                    "discovery_matches_shared": (
                        run_discovery_hash is not None
                        and shared_discovery_hash is not None
                        and run_discovery_hash == shared_discovery_hash
                    ),
                }
            )

    return pd.DataFrame(rows), shared_info


# -----------------------------------------------------------------------------
# Quality-cost trade-off columns
# -----------------------------------------------------------------------------


def add_quality_cost_deltas(per_seed: pd.DataFrame) -> pd.DataFrame:
    out = per_seed.copy()

    # Columns where positive "reduction" is favorable because smaller is cheaper.
    lower_cost_cols = {
        "trainable_params": "params_reduction_pct_vs_CRAT",
        "peak_vram_GB": "vram_reduction_pct_vs_CRAT",
        "train_seconds": "train_time_reduction_pct_vs_CRAT",
        "generation_seconds": "generation_time_reduction_pct_vs_CRAT",
        "latency_mean_seconds": "latency_mean_reduction_pct_vs_CRAT",
        "latency_p50_seconds": "latency_p50_reduction_pct_vs_CRAT",
        "latency_p95_seconds": "latency_p95_reduction_pct_vs_CRAT",
    }

    # Columns where positive gain is favorable because larger is faster.
    throughput_cols = {
        "train_tokens_per_second": "train_throughput_gain_pct_vs_CRAT",
        "generation_tokens_per_second": "generation_throughput_gain_pct_vs_CRAT",
        "benchmark_tokens_per_second": "benchmark_throughput_gain_pct_vs_CRAT",
    }

    for seed, idx in out.groupby("seed").groups.items():
        group = out.loc[idx]
        full = group[group["ablation"] == "full"]
        if len(full) != 1:
            raise RuntimeError(f"Seed {seed}: expected exactly one Full CRAT row, found {len(full)}")
        ref = full.iloc[0]

        for row_idx in idx:
            out.loc[row_idx, "delta_BLEU_vs_CRAT"] = (
                numeric(out.loc[row_idx, "BLEU_mean"]) - numeric(ref["BLEU_mean"])
            )
            out.loc[row_idx, "delta_chrF_vs_CRAT"] = (
                numeric(out.loc[row_idx, "chrF_mean"]) - numeric(ref["chrF_mean"])
            )

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
            "COMET is not installed. Install `unbabel-comet` in the environment, "
            "or rerun this script with --skip-comet."
        ) from exc

    if comet_checkpoint:
        checkpoint = Path(comet_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"COMET checkpoint does not exist: {checkpoint}")
        print(f"Loading local COMET checkpoint: {checkpoint}")
        return load_from_checkpoint(str(checkpoint)), str(checkpoint)

    print(f"Resolving COMET model: {comet_model_name}")
    print("NOTE: this may require network access unless the model is already cached.")
    model_path = download_model(comet_model_name)
    return load_from_checkpoint(model_path), comet_model_name


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
    requested_model = str(Path(comet_checkpoint).expanduser().resolve()) if comet_checkpoint else comet_model_name

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
    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    scores = np.asarray(output.scores, dtype=float)
    if len(scores) != len(samples):
        raise RuntimeError(
            f"COMET returned {len(scores)} sentence scores for {len(samples)} samples."
        )

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
    print(f"Saved COMET sentence scores: {score_path}")
    return scored, requested_model


def aggregate_comet(
    scored: pd.DataFrame,
    fwd: str,
    rev: str,
) -> pd.DataFrame:
    directional = (
        scored.groupby(["seed", "ablation", "label", "direction"], as_index=False)["COMET"]
        .mean()
        .rename(columns={"COMET": "COMET_direction"})
    )

    rows: List[Dict[str, Any]] = []
    for (seed, slug, label), group in directional.groupby(["seed", "ablation", "label"], sort=False):
        by_dir = dict(zip(group["direction"].astype(str), group["COMET_direction"].astype(float)))
        cf = numeric(by_dir.get(fwd))
        cr = numeric(by_dir.get(rev))
        rows.append({
            "seed": int(seed),
            "ablation": slug,
            "label": label,
            "COMET_forward": cf,
            "COMET_reverse": cr,
            "COMET_mean": float(np.nanmean([cf, cr])),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Paired bootstrap
# -----------------------------------------------------------------------------


def aligned_pair(
    scored_or_samples: pd.DataFrame,
    seed: int,
    variant_slug: str,
    direction: str,
    include_comet: bool,
) -> pd.DataFrame:
    base_keys = ["direction", "src", "ref"]
    cols_full = base_keys + ["pred"] + (["COMET"] if include_comet else [])
    cols_var = base_keys + ["pred"] + (["COMET"] if include_comet else [])

    full = scored_or_samples[
        (scored_or_samples["seed"] == seed)
        & (scored_or_samples["ablation"] == "full")
        & (scored_or_samples["direction"] == direction)
    ][cols_full].copy()
    var = scored_or_samples[
        (scored_or_samples["seed"] == seed)
        & (scored_or_samples["ablation"] == variant_slug)
        & (scored_or_samples["direction"] == direction)
    ][cols_var].copy()

    # Duplicate-safe pairing. If identical (direction, src, ref) rows occur,
    # pair them by their within-key occurrence number instead of creating a
    # many-to-many merge.
    full["_occurrence"] = full.groupby(base_keys, dropna=False).cumcount()
    var["_occurrence"] = var.groupby(base_keys, dropna=False).cumcount()
    keys = base_keys + ["_occurrence"]

    rename_full = {"pred": "pred_full"}
    rename_var = {"pred": "pred_variant"}
    if include_comet:
        rename_full["COMET"] = "COMET_full"
        rename_var["COMET"] = "COMET_variant"
    full = full.rename(columns=rename_full)
    var = var.rename(columns=rename_var)

    merged = full.merge(var, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(full) or len(merged) != len(var):
        raise RuntimeError(
            f"Pairing mismatch: seed={seed} variant={variant_slug} direction={direction}; "
            f"full={len(full)}, variant={len(var)}, matched={len(merged)}"
        )
    if merged.empty:
        raise RuntimeError(
            f"No paired samples: seed={seed} variant={variant_slug} direction={direction}"
        )
    return merged.drop(columns=["_occurrence"]).reset_index(drop=True)

def score_bootstrap_sample(frame: pd.DataFrame, indices: np.ndarray, include_comet: bool) -> Tuple[float, float, float]:
    sampled = frame.iloc[indices]
    refs = sampled["ref"].astype(str).tolist()
    full_pred = sampled["pred_full"].astype(str).tolist()
    var_pred = sampled["pred_variant"].astype(str).tolist()

    # Delta convention here: Full CRAT - Variant.
    delta_bleu = (
        corpus_bleu(full_pred, [refs]).score
        - corpus_bleu(var_pred, [refs]).score
    )
    delta_chrf = (
        corpus_chrf(full_pred, [refs]).score
        - corpus_chrf(var_pred, [refs]).score
    )
    if include_comet:
        delta_comet = float(
            sampled["COMET_full"].astype(float).mean()
            - sampled["COMET_variant"].astype(float).mean()
        )
    else:
        delta_comet = float("nan")
    return float(delta_bleu), float(delta_chrf), delta_comet


def observed_pair_delta(frame: pd.DataFrame, include_comet: bool) -> Tuple[float, float, float]:
    idx = np.arange(len(frame), dtype=int)
    return score_bootstrap_sample(frame, idx, include_comet)


def percentile_ci(values: np.ndarray, confidence: float) -> Tuple[float, float]:
    alpha = 1.0 - confidence
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def tail_probability(values: np.ndarray) -> float:
    # Descriptive two-sided bootstrap tail probability around zero.
    if len(values) == 0:
        return float("nan")
    p_le = float(np.mean(values <= 0.0))
    p_ge = float(np.mean(values >= 0.0))
    return float(min(1.0, 2.0 * min(p_le, p_ge)))




def average_direction_or_seed_deltas(values: Sequence[Tuple[float, float, float]] | np.ndarray, include_comet: bool) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    out = np.empty(3, dtype=float)
    out[0] = float(np.mean(arr[:, 0]))
    out[1] = float(np.mean(arr[:, 1]))
    out[2] = float(np.mean(arr[:, 2])) if include_comet else float("nan")
    return out

def paired_bootstrap(
    samples: pd.DataFrame,
    seeds: Sequence[int],
    directions: Sequence[str],
    n_bootstrap: int,
    confidence: float,
    bootstrap_seed: int,
    include_comet: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(bootstrap_seed)
    per_seed_rows: List[Dict[str, Any]] = []
    overall_rows: List[Dict[str, Any]] = []

    variant_specs = [x for x in ABLATIONS if x[0] != "full"]

    # Cache aligned frames to guarantee identical pair handling everywhere.
    aligned: Dict[Tuple[int, str, str], pd.DataFrame] = {}
    for seed in seeds:
        for slug, _label, _method, _variant_name in variant_specs:
            for direction in directions:
                aligned[(seed, slug, direction)] = aligned_pair(
                    samples, seed, slug, direction, include_comet
                )

    # Per-seed paired bootstrap.
    for seed in seeds:
        for slug, label, _method, _variant_name in variant_specs:
            observed_by_direction = [
                observed_pair_delta(aligned[(seed, slug, direction)], include_comet)
                for direction in directions
            ]
            observed = average_direction_or_seed_deltas(observed_by_direction, include_comet)

            reps = np.empty((n_bootstrap, 3), dtype=float)
            for b in range(n_bootstrap):
                direction_deltas = []
                for direction in directions:
                    frame = aligned[(seed, slug, direction)]
                    idx = rng.integers(0, len(frame), size=len(frame))
                    direction_deltas.append(score_bootstrap_sample(frame, idx, include_comet))
                reps[b] = average_direction_or_seed_deltas(direction_deltas, include_comet)

            row: Dict[str, Any] = {
                "seed": seed,
                "ablation": slug,
                "label": label,
                "n_bootstrap": n_bootstrap,
                "confidence": confidence,
            }
            for j, metric in enumerate(("BLEU", "chrF", "COMET")):
                row[f"delta_{metric}_CRAT_minus_variant"] = float(observed[j])
                if metric == "COMET" and not include_comet:
                    row[f"delta_{metric}_ci_low"] = np.nan
                    row[f"delta_{metric}_ci_high"] = np.nan
                    row[f"delta_{metric}_bootstrap_tail_prob"] = np.nan
                    continue
                vals = reps[:, j]
                lo, hi = percentile_ci(vals, confidence)
                row[f"delta_{metric}_ci_low"] = lo
                row[f"delta_{metric}_ci_high"] = hi
                row[f"delta_{metric}_bootstrap_tail_prob"] = tail_probability(vals)
            per_seed_rows.append(row)

    # Hierarchical paired bootstrap:
    # resample seeds with replacement, then resample paired sentences within
    # each sampled seed and direction. This captures both seed and test-set
    # sampling variability while preserving Full-vs-variant pairing.
    for slug, label, _method, _variant_name in variant_specs:
        observed_seed_deltas = []
        for seed in seeds:
            direction_deltas = [
                observed_pair_delta(aligned[(seed, slug, direction)], include_comet)
                for direction in directions
            ]
            observed_seed_deltas.append(average_direction_or_seed_deltas(direction_deltas, include_comet))
        observed = average_direction_or_seed_deltas(observed_seed_deltas, include_comet)

        reps = np.empty((n_bootstrap, 3), dtype=float)
        seed_array = np.asarray(seeds, dtype=int)
        for b in range(n_bootstrap):
            sampled_seeds = rng.choice(seed_array, size=len(seed_array), replace=True)
            seed_deltas = []
            for sampled_seed in sampled_seeds:
                direction_deltas = []
                for direction in directions:
                    frame = aligned[(int(sampled_seed), slug, direction)]
                    idx = rng.integers(0, len(frame), size=len(frame))
                    direction_deltas.append(score_bootstrap_sample(frame, idx, include_comet))
                seed_deltas.append(average_direction_or_seed_deltas(direction_deltas, include_comet))
            reps[b] = average_direction_or_seed_deltas(seed_deltas, include_comet)

        row = {
            "ablation": slug,
            "label": label,
            "n_seeds": len(seeds),
            "n_bootstrap": n_bootstrap,
            "confidence": confidence,
            "bootstrap_type": "hierarchical_paired_seed_and_sentence",
        }
        for j, metric in enumerate(("BLEU", "chrF", "COMET")):
            row[f"delta_{metric}_CRAT_minus_variant"] = float(observed[j])
            if metric == "COMET" and not include_comet:
                row[f"delta_{metric}_ci_low"] = np.nan
                row[f"delta_{metric}_ci_high"] = np.nan
                row[f"delta_{metric}_bootstrap_tail_prob"] = np.nan
                continue
            vals = reps[:, j]
            lo, hi = percentile_ci(vals, confidence)
            row[f"delta_{metric}_ci_low"] = lo
            row[f"delta_{metric}_ci_high"] = hi
            row[f"delta_{metric}_bootstrap_tail_prob"] = tail_probability(vals)
        overall_rows.append(row)

    return pd.DataFrame(per_seed_rows), pd.DataFrame(overall_rows)


# -----------------------------------------------------------------------------
# Combined-ablation interaction / non-additivity analysis
# -----------------------------------------------------------------------------


INTERACTION_VARIANTS: Mapping[str, str] = {
    "full": "full",
    "sparse": "no_sparse",
    "morph": "no_morph",
    "integration": "no_integration",
    "sparse_morph": "no_sparse_morph",
    "sparse_integration": "no_sparse_integration",
    "morph_integration": "no_morph_integration",
    "sparse_morph_integration": "no_sparse_morph_integration",
}


def compute_interaction_effects(
    per_seed: pd.DataFrame,
    include_comet: bool,
) -> pd.DataFrame:
    """
    Measure non-additivity among sparse, morphology, and integration removals.

    For a higher-is-better quality metric Q:
      I_AB = Q(-A,-B) - Q(-A) - Q(-B) + Q(full)

    Positive I_AB means the joint removal is less harmful / more beneficial
    than the sum of the two single-removal effects. Negative I_AB means the
    joint removal is more harmful than an additive expectation.

    The pure third-order interaction is:
      I_ABC = Q(-A,-B,-C) - Q(-A,-B) - Q(-A,-C) - Q(-B,-C)
              + Q(-A) + Q(-B) + Q(-C) - Q(full)
    """
    metrics = ["BLEU_mean", "chrF_mean"]
    if include_comet and "COMET_mean" in per_seed.columns:
        metrics.append("COMET_mean")

    rows: List[Dict[str, Any]] = []
    required = set(INTERACTION_VARIANTS.values())
    for seed, group in per_seed.groupby("seed", sort=False):
        by_slug = {
            str(row.ablation): row
            for row in group.itertuples(index=False)
        }
        if not required.issubset(by_slug):
            continue

        q: Dict[str, Dict[str, float]] = {}
        for key, slug in INTERACTION_VARIANTS.items():
            row = by_slug[slug]
            q[key] = {metric: numeric(getattr(row, metric)) for metric in metrics}

        pair_specs = [
            ("sparse+morph", "sparse_morph", "sparse", "morph"),
            ("sparse+integration", "sparse_integration", "sparse", "integration"),
            ("morph+integration", "morph_integration", "morph", "integration"),
        ]
        for name, pair_key, a, b in pair_specs:
            row: Dict[str, Any] = {
                "seed": int(seed),
                "interaction": name,
                "order": 2,
                "definition": "Q(-A,-B) - Q(-A) - Q(-B) + Q(full)",
            }
            for metric in metrics:
                pair_effect = q[pair_key][metric] - q["full"][metric]
                single_a = q[a][metric] - q["full"][metric]
                single_b = q[b][metric] - q["full"][metric]
                row[f"{metric}_joint_removal_effect_vs_full"] = pair_effect
                row[f"{metric}_nonadditivity"] = (
                    q[pair_key][metric]
                    - q[a][metric]
                    - q[b][metric]
                    + q["full"][metric]
                )
                row[f"{metric}_sum_single_effects"] = single_a + single_b
            rows.append(row)

        row = {
            "seed": int(seed),
            "interaction": "sparse+morph+integration",
            "order": 3,
            "definition": (
                "Q(-A,-B,-C) - Q(-A,-B) - Q(-A,-C) - Q(-B,-C) "
                "+ Q(-A) + Q(-B) + Q(-C) - Q(full)"
            ),
        }
        for metric in metrics:
            row[f"{metric}_joint_removal_effect_vs_full"] = (
                q["sparse_morph_integration"][metric] - q["full"][metric]
            )
            row[f"{metric}_nonadditivity"] = (
                q["sparse_morph_integration"][metric]
                - q["sparse_morph"][metric]
                - q["sparse_integration"][metric]
                - q["morph_integration"][metric]
                + q["sparse"][metric]
                + q["morph"][metric]
                + q["integration"][metric]
                - q["full"][metric]
            )
            row[f"{metric}_sum_single_effects"] = (
                (q["sparse"][metric] - q["full"][metric])
                + (q["morph"][metric] - q["full"][metric])
                + (q["integration"][metric] - q["full"][metric])
            )
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Output formatting
# -----------------------------------------------------------------------------


def print_compact_summary(summary: pd.DataFrame, include_comet: bool) -> None:
    columns = [
        "label",
        "BLEU_mean_mean", "BLEU_mean_std",
        "chrF_mean_mean", "chrF_mean_std",
        "trainable_params_mean",
        "peak_vram_GB_mean",
        "train_seconds_mean",
        "train_tokens_per_second_mean",
        "generation_tokens_per_second_mean",
        "latency_p50_seconds_mean",
        "latency_p95_seconds_mean",
    ]
    if include_comet and "COMET_mean_mean" in summary.columns:
        columns.insert(5, "COMET_mean_mean")
        columns.insert(6, "COMET_mean_std")
    columns = [c for c in columns if c in summary.columns]
    print("\nEffectiveness + efficiency summary:")
    print(summary[columns].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def print_routing_summary(routing_summary: pd.DataFrame) -> None:
    wanted = [
        "label",
        "eval_gate_mean_source_mean",
        "eval_gate_mean_indicator_mean",
        "eval_gate_mean_position_mean",
        "eval_gate_mean_integration_mean",
        "eval_gate_active_source@0.5_mean",
        "eval_gate_active_indicator@0.5_mean",
        "eval_gate_active_position@0.5_mean",
        "eval_gate_active_integration@0.5_mean",
        "train_router_entropy_mean",
        "eval_mean_active_roles@0.5_mean",
    ]
    wanted = [c for c in wanted if c in routing_summary.columns]
    print("\nRouting summary:")
    print(routing_summary[wanted].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


def print_bootstrap_summary(bootstrap: pd.DataFrame, include_comet: bool) -> None:
    cols = [
        "label",
        "delta_BLEU_CRAT_minus_variant", "delta_BLEU_ci_low", "delta_BLEU_ci_high",
        "delta_chrF_CRAT_minus_variant", "delta_chrF_ci_low", "delta_chrF_ci_high",
    ]
    if include_comet:
        cols += [
            "delta_COMET_CRAT_minus_variant", "delta_COMET_ci_low", "delta_COMET_ci_high"
        ]
    print("\nHierarchical paired-bootstrap summary (positive delta = Full CRAT better):")
    print(bootstrap[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Post-hoc effectiveness, efficiency, routing, COMET, and paired-bootstrap analysis for CRAT ablations."
    )
    ap.add_argument("--root", required=True, help="CRAT root, e.g. $SCRATCH/crat")
    ap.add_argument("--pair", default="en-fr", help="Language pair, e.g. en-fr")
    ap.add_argument("--dataset", default="opus100")
    ap.add_argument("--seeds", default="42,64,123")
    ap.add_argument("--discovery-seed", type=int, default=42)
    ap.add_argument("--morph-backend", default="stanza")
    ap.add_argument("--allow-missing", action="store_true", help="Continue if some run artifacts are missing.")
    ap.add_argument(
        "--strict-cache-audit",
        action="store_true",
        help=(
            "Fail when pair-shared SIPP/Stanza artifacts are missing, pair-invalid, "
            "or any per-run copy does not match the pair-shared file."
        ),
    )

    ap.add_argument("--skip-comet", action="store_true", help="Skip COMET scoring but still do all other analyses.")
    ap.add_argument(
        "--comet-model",
        default="Unbabel/wmt22-comet-da",
        help="COMET model name used when --comet-checkpoint is not supplied.",
    )
    ap.add_argument(
        "--comet-checkpoint",
        default=None,
        help="Path to a local COMET .ckpt file. Recommended on offline compute nodes.",
    )
    ap.add_argument("--comet-batch-size", type=int, default=32)
    ap.add_argument("--comet-gpus", type=int, default=1, help="0 for CPU; 1 for one GPU.")
    ap.add_argument("--force-comet", action="store_true", help="Ignore cached COMET sentence scores and recompute.")

    ap.add_argument("--bootstrap-samples", type=int, default=2000)
    ap.add_argument("--bootstrap-seed", type=int, default=20260811)
    ap.add_argument("--confidence", type=float, default=0.95)
    ap.add_argument("--skip-bootstrap", action="store_true")

    args = ap.parse_args()

    if not args.skip_bootstrap and args.bootstrap_samples < 100:
        raise SystemExit("--bootstrap-samples should be at least 100; 2000 is recommended.")
    if not (0.0 < args.confidence < 1.0):
        raise SystemExit("--confidence must be between 0 and 1.")

    root = Path(args.root).expanduser().resolve()
    seeds = parse_seed_list(args.seeds)
    if "-" not in args.pair:
        raise SystemExit("--pair must look like en-fr")
    src, tgt = args.pair.split("-", 1)
    pair_tag = f"{src}{tgt}"
    fwd = f"{src}-{tgt}"
    rev = f"{tgt}-{src}"
    directions = [fwd, rev]

    seed_tag = "-".join(str(seed) for seed in seeds)
    out_dir = (
        root / "outputs" / "ablations" / "posthoc" / args.dataset / args.pair
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{args.pair}_posthoc_{args.dataset}_seeds{seed_tag}"

    print("=" * 90)
    print("CRAT POST-HOC ABLATION ANALYSIS")
    print("=" * 90)
    print(f"Root:       {root}")
    print(f"Dataset:    {args.dataset}")
    print(f"Pair:       {args.pair}")
    print(f"Seeds:      {seeds}")
    print(f"Output dir: {out_dir}")

    # ------------------------------------------------------------------
    # Load all refined ablation runs (11 variants x requested seeds).
    # ------------------------------------------------------------------
    per_seed, samples = load_completed_runs(
        root=root,
        dataset=args.dataset,
        pair=args.pair,
        pair_tag=pair_tag,
        fwd=fwd,
        rev=rev,
        seeds=seeds,
        allow_missing=args.allow_missing,
    )
    print(f"Loaded result rows: {len(per_seed)}")
    print(f"Loaded translations: {len(samples):,}")

    # ------------------------------------------------------------------
    # Audit pair-shared Stanza/SIPP reuse.
    # ------------------------------------------------------------------
    cache_audit, shared_artifacts = audit_shared_artifact_reuse(
        root=root,
        dataset=args.dataset,
        pair=args.pair,
        pair_tag=pair_tag,
        seeds=seeds,
        morph_backend=args.morph_backend,
        discovery_seed=args.discovery_seed,
    )
    cache_audit_path = out_dir / f"{prefix}_shared_cache_reuse_audit.csv"
    cache_audit.to_csv(cache_audit_path, index=False)

    shared_morph_ok = bool(shared_artifacts["morph"].get("pair_ok"))
    shared_discovery_ok = bool(
        shared_artifacts["discovery"].get("pair_ok")
        and shared_artifacts["discovery"].get("discovery_seed_ok")
    )
    all_morph_match = bool(cache_audit["morph_matches_shared"].all())
    all_discovery_match = bool(cache_audit["discovery_matches_shared"].all())

    print("\nPair-shared artifact audit:")
    print(f"  Shared dir:              {shared_artifacts['shared_dir']}")
    print(f"  Shared morphology pair: {'OK' if shared_morph_ok else 'NOT VERIFIED'}")
    print(f"  Shared discovery:        {'OK' if shared_discovery_ok else 'NOT VERIFIED'}")
    print(f"  All run morph copies:    {'MATCH' if all_morph_match else 'MISMATCH/MISSING'}")
    print(f"  All run SIPP copies:     {'MATCH' if all_discovery_match else 'MISMATCH/MISSING'}")

    if args.strict_cache_audit and not (
        shared_morph_ok
        and shared_discovery_ok
        and all_morph_match
        and all_discovery_match
    ):
        raise SystemExit(
            "Strict cache audit failed. Inspect: "
            f"{cache_audit_path}"
        )

    # ------------------------------------------------------------------
    # Extended efficiency + quality/cost trade-off.
    # ------------------------------------------------------------------
    per_seed = add_quality_cost_deltas(per_seed)

    # ------------------------------------------------------------------
    # COMET from saved translations.
    # ------------------------------------------------------------------
    scored_samples = samples
    include_comet = not args.skip_comet
    if include_comet:
        scored_samples, comet_model_id = compute_or_load_comet(
            samples=samples,
            output_dir=out_dir,
            prefix=prefix,
            comet_checkpoint=args.comet_checkpoint,
            comet_model_name=args.comet_model,
            batch_size=args.comet_batch_size,
            gpus=args.comet_gpus,
            force=args.force_comet,
        )
        comet_per_seed = aggregate_comet(scored_samples, fwd=fwd, rev=rev)
        comet_per_seed_path = out_dir / f"{prefix}_comet_per_seed.csv"
        comet_per_seed.to_csv(comet_per_seed_path, index=False)
        comet_summary = summarize_numeric(
            comet_per_seed,
            ["ablation", "label"],
            ["COMET_forward", "COMET_reverse", "COMET_mean"],
        )
        comet_summary_path = out_dir / f"{prefix}_comet_summary.csv"
        comet_summary.to_csv(comet_summary_path, index=False)
        per_seed = per_seed.merge(
            comet_per_seed,
            on=["seed", "ablation", "label"],
            how="left",
            validate="one_to_one",
        )

        # Variant - CRAT COMET delta within seed, same convention as BLEU/chrF
        # in the quality-cost table.
        per_seed["delta_COMET_vs_CRAT"] = np.nan
        for seed, idx in per_seed.groupby("seed").groups.items():
            group = per_seed.loc[idx]
            full = group[group["ablation"] == "full"]
            if len(full) != 1:
                raise RuntimeError(f"Seed {seed}: Full CRAT missing after COMET merge.")
            ref = numeric(full.iloc[0]["COMET_mean"])
            for row_idx in idx:
                per_seed.loc[row_idx, "delta_COMET_vs_CRAT"] = numeric(per_seed.loc[row_idx, "COMET_mean"]) - ref

        print(f"COMET model: {comet_model_id}")
    else:
        comet_per_seed_path = None
        comet_summary_path = None
        print("COMET: skipped")

    # Save complete per-seed table after optional COMET merge.
    per_seed_path = out_dir / f"{prefix}_all_variants_per_seed.csv"
    per_seed.to_csv(per_seed_path, index=False)

    # Summary across seeds for all numeric metrics and quality-cost deltas.
    numeric_cols = [
        c for c in per_seed.columns
        if c not in {"seed", "ablation", "label", "method", "run_dir"}
        and pd.api.types.is_numeric_dtype(per_seed[c])
    ]
    summary = summarize_numeric(per_seed, ["ablation", "label"], numeric_cols)
    summary_path = out_dir / f"{prefix}_all_variants_summary.csv"
    summary.to_csv(summary_path, index=False)

    # Combined sparse/morph/integration non-additivity analysis.
    interaction_per_seed = compute_interaction_effects(
        per_seed, include_comet=include_comet
    )
    if not interaction_per_seed.empty:
        interaction_per_seed_path = (
            out_dir / f"{prefix}_interaction_effects_per_seed.csv"
        )
        interaction_per_seed.to_csv(interaction_per_seed_path, index=False)
        interaction_metrics = [
            c for c in interaction_per_seed.columns
            if c not in {"seed", "interaction", "order", "definition"}
            and pd.api.types.is_numeric_dtype(interaction_per_seed[c])
        ]
        interaction_summary = summarize_numeric(
            interaction_per_seed,
            ["interaction", "order", "definition"],
            interaction_metrics,
        )
        interaction_summary_path = (
            out_dir / f"{prefix}_interaction_effects_summary.csv"
        )
        interaction_summary.to_csv(interaction_summary_path, index=False)
    else:
        interaction_per_seed_path = None
        interaction_summary_path = None

    # Dedicated efficiency files.
    efficiency_cols = [
        "seed", "ablation", "label",
        "BLEU_mean", "chrF_mean",
    ]
    if include_comet and "COMET_mean" in per_seed.columns:
        efficiency_cols.append("COMET_mean")
    efficiency_cols += [
        c for c in (
            list(EFFICIENCY_METRICS)
            + [
                "delta_BLEU_vs_CRAT", "delta_chrF_vs_CRAT", "delta_COMET_vs_CRAT",
                "params_reduction_pct_vs_CRAT",
                "vram_reduction_pct_vs_CRAT",
                "train_time_reduction_pct_vs_CRAT",
                "generation_time_reduction_pct_vs_CRAT",
                "latency_mean_reduction_pct_vs_CRAT",
                "latency_p50_reduction_pct_vs_CRAT",
                "latency_p95_reduction_pct_vs_CRAT",
                "train_throughput_gain_pct_vs_CRAT",
                "generation_throughput_gain_pct_vs_CRAT",
                "benchmark_throughput_gain_pct_vs_CRAT",
            ]
        )
        if c in per_seed.columns
    ]
    efficiency_per_seed = per_seed[efficiency_cols].copy()
    efficiency_per_seed_path = out_dir / f"{prefix}_efficiency_per_seed.csv"
    efficiency_per_seed.to_csv(efficiency_per_seed_path, index=False)
    efficiency_metric_cols = [c for c in efficiency_per_seed.columns if c not in {"seed", "ablation", "label"}]
    efficiency_summary = summarize_numeric(
        efficiency_per_seed,
        ["ablation", "label"],
        efficiency_metric_cols,
    )
    efficiency_summary_path = out_dir / f"{prefix}_efficiency_summary.csv"
    efficiency_summary.to_csv(efficiency_summary_path, index=False)

    # Dedicated routing files.
    routing_cols = ["seed", "ablation", "label"] + [c for c in ROUTING_METRICS if c in per_seed.columns]
    routing_per_seed = per_seed[routing_cols].copy()
    routing_per_seed_path = out_dir / f"{prefix}_routing_per_seed.csv"
    routing_per_seed.to_csv(routing_per_seed_path, index=False)
    routing_summary = summarize_numeric(
        routing_per_seed,
        ["ablation", "label"],
        [c for c in ROUTING_METRICS if c in routing_per_seed.columns],
    )
    routing_summary_path = out_dir / f"{prefix}_routing_summary.csv"
    routing_summary.to_csv(routing_summary_path, index=False)

    # ------------------------------------------------------------------
    # Paired bootstrap: Full CRAT vs each variant.
    # ------------------------------------------------------------------
    if not args.skip_bootstrap:
        bootstrap_per_seed, bootstrap_overall = paired_bootstrap(
            samples=scored_samples,
            seeds=seeds,
            directions=directions,
            n_bootstrap=args.bootstrap_samples,
            confidence=args.confidence,
            bootstrap_seed=args.bootstrap_seed,
            include_comet=include_comet,
        )
        bootstrap_per_seed_path = out_dir / f"{prefix}_bootstrap_vs_full_crat_per_seed.csv"
        bootstrap_overall_path = out_dir / f"{prefix}_bootstrap_vs_full_crat.csv"
        bootstrap_per_seed.to_csv(bootstrap_per_seed_path, index=False)
        bootstrap_overall.to_csv(bootstrap_overall_path, index=False)
    else:
        bootstrap_per_seed = pd.DataFrame()
        bootstrap_overall = pd.DataFrame()
        bootstrap_per_seed_path = None
        bootstrap_overall_path = None
        print("Paired bootstrap: skipped")

    # Metadata/provenance.
    metadata = {
        "root": str(root),
        "dataset": args.dataset,
        "pair": args.pair,
        "seeds": seeds,
        "ablations": [slug for slug, _, _, _ in ABLATIONS],
        "variants": [variant for _, _, _, variant in ABLATIONS],
        "comet_enabled": include_comet,
        "comet_model": None if not include_comet else (
            str(Path(args.comet_checkpoint).expanduser().resolve()) if args.comet_checkpoint else args.comet_model
        ),
        "bootstrap_enabled": not args.skip_bootstrap,
        "bootstrap_samples": None if args.skip_bootstrap else args.bootstrap_samples,
        "bootstrap_seed": None if args.skip_bootstrap else args.bootstrap_seed,
        "bootstrap_confidence": None if args.skip_bootstrap else args.confidence,
        "bootstrap_delta_convention": "Full CRAT - Variant",
        "quality_cost_delta_convention": "Variant - Full CRAT for BLEU/chrF/COMET; positive reduction means lower cost than Full CRAT",
        "discovery_seed": args.discovery_seed,
        "morph_backend": args.morph_backend,
        "shared_artifacts": shared_artifacts,
        "cache_audit_file": str(cache_audit_path),
        "cache_audit_all_morph_match": all_morph_match,
        "cache_audit_all_discovery_match": all_discovery_match,
        "interaction_nonadditivity_definition": (
            "Pair: Q(-A,-B)-Q(-A)-Q(-B)+Q(full); "
            "Third-order: inclusion-exclusion over sparse, morph, integration."
        ),
        "sacrebleu_version": package_version("sacrebleu"),
        "pandas_version": package_version("pandas"),
        "numpy_version": package_version("numpy"),
        "unbabel_comet_version": package_version("unbabel-comet") if include_comet else None,
    }
    metadata_path = out_dir / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    # Console summaries.
    print_compact_summary(summary, include_comet=include_comet)
    print_routing_summary(routing_summary)
    if not bootstrap_overall.empty:
        print_bootstrap_summary(bootstrap_overall, include_comet=include_comet)

    print("\nSaved files:")
    for path in [
        per_seed_path,
        summary_path,
        efficiency_per_seed_path,
        efficiency_summary_path,
        routing_per_seed_path,
        routing_summary_path,
        interaction_per_seed_path,
        interaction_summary_path,
        cache_audit_path,
        out_dir / f"{prefix}_comet_sentence_scores.csv" if include_comet else None,
        comet_per_seed_path,
        comet_summary_path,
        bootstrap_per_seed_path,
        bootstrap_overall_path,
        metadata_path,
    ]:
        if path is not None:
            print(f"  {path}")


if __name__ == "__main__":
    main()
