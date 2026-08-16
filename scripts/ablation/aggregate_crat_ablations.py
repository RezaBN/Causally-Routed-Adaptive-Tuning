#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# slug, display label, internal Python method, externally visible variant name
ABLATIONS = [
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
]


def result_candidates(
    abl_dir: Path,
    pair: str,
    seed: int,
    slug: str,
    internal_method: str,
    variant_name: str,
) -> list[Path]:
    """Prefer the new variant-specific filename, retain legacy compatibility."""
    return [
        abl_dir / f"{pair}_seed{seed}_{variant_name}_comparison_results.csv",
        # Previous named script format.
        abl_dir / f"{pair}_seed{seed}_{slug}_{internal_method}_comparison_results.csv",
        # Original script format.
        abl_dir / "comparison_results.csv",
    ]


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="CRAT root, e.g. $SCRATCH/crat")
    ap.add_argument("--pair", default="en-fr")
    ap.add_argument("--dataset", default="opus100")
    ap.add_argument("--seeds", default="42,64,123")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any requested seed/ablation result is missing.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    src, tgt = args.pair.split("-", 1)
    pair_tag = f"{src}{tgt}"
    fwd = f"{src}-{tgt}"
    rev = f"{tgt}-{src}"

    rows: list[dict] = []
    missing_items: list[str] = []

    for seed in seeds:
        for slug, label, internal_method, variant_name in ABLATIONS:
            abl_dir = (
                root
                / "outputs"
                / "ablations"
                / f"mistral7b_{args.dataset}_{pair_tag}_seed{seed}_{slug}"
            )
            candidates = result_candidates(
                abl_dir,
                args.pair,
                seed,
                slug,
                internal_method,
                variant_name,
            )
            csv_path = first_existing(candidates)

            if csv_path is None:
                msg = (
                    f"MISSING: pair={args.pair} seed={seed} ablation={slug}; checked: "
                    + ", ".join(str(p) for p in candidates)
                )
                print(msg)
                missing_items.append(msg)
                continue

            df = pd.read_csv(csv_path)
            hit = df[df["method"].astype(str) == internal_method]
            if hit.empty:
                msg = (
                    f"MISSING METHOD: pair={args.pair} seed={seed} ablation={slug} "
                    f"internal_method={internal_method}: {csv_path}"
                )
                print(msg)
                missing_items.append(msg)
                continue

            r = hit.iloc[-1].to_dict()
            bleu_f = float(r[f"BLEU_{fwd}"])
            bleu_r = float(r[f"BLEU_{rev}"])
            chrf_f = float(r[f"chrF_{fwd}"])
            chrf_r = float(r[f"chrF_{rev}"])

            rows.append(
                {
                    "pair": args.pair,
                    "seed": seed,
                    "ablation": slug,
                    "variant": r.get("variant", variant_name),
                    "label": label,
                    "internal_method": internal_method,
                    "result_file": str(csv_path),
                    "BLEU_forward": bleu_f,
                    "BLEU_reverse": bleu_r,
                    "BLEU_mean": (bleu_f + bleu_r) / 2.0,
                    "chrF_forward": chrf_f,
                    "chrF_reverse": chrf_r,
                    "chrF_mean": (chrf_f + chrf_r) / 2.0,
                    "validation_loss": r.get("validation_loss", np.nan),
                    "trainable_params": r.get("trainable_params", np.nan),
                    "trainable_pct": r.get("trainable_pct", np.nan),
                    "peak_vram_GB": r.get("peak_vram_GB", np.nan),
                    "train_seconds": r.get("train_seconds", np.nan),
                    "train_tokens_per_second": r.get(
                        "train_tokens_per_second", np.nan
                    ),
                    "generation_seconds": r.get("generation_seconds", np.nan),
                    "generation_tokens_per_second": r.get(
                        "generation_tokens_per_second", np.nan
                    ),
                    "latency_mean_seconds": r.get("latency_mean_seconds", np.nan),
                    "latency_p50_seconds": r.get("latency_p50_seconds", np.nan),
                    "latency_p95_seconds": r.get("latency_p95_seconds", np.nan),
                    "benchmark_tokens_per_second": r.get(
                        "benchmark_tokens_per_second", np.nan
                    ),
                }
            )

    if args.strict and missing_items:
        raise SystemExit(
            f"Strict mode: {len(missing_items)} requested ablation result(s) are missing."
        )
    if not rows:
        raise SystemExit("No ablation results found.")

    order = {slug: i for i, (slug, *_rest) in enumerate(ABLATIONS)}
    per_seed = pd.DataFrame(rows)
    per_seed["_order"] = per_seed["ablation"].map(order)
    per_seed = per_seed.sort_values(["seed", "_order"]).drop(columns=["_order"])

    out_dir = root / "outputs" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_seed_path = (
        out_dir
        / f"ablation_{args.dataset}_{pair_tag}_all_variants_per_seed.csv"
    )
    per_seed.to_csv(per_seed_path, index=False)

    metric_cols = [
        "BLEU_forward",
        "BLEU_reverse",
        "BLEU_mean",
        "chrF_forward",
        "chrF_reverse",
        "chrF_mean",
        "validation_loss",
        "trainable_params",
        "trainable_pct",
        "peak_vram_GB",
        "train_seconds",
        "train_tokens_per_second",
        "generation_seconds",
        "generation_tokens_per_second",
        "latency_mean_seconds",
        "latency_p50_seconds",
        "latency_p95_seconds",
        "benchmark_tokens_per_second",
    ]

    summary_rows: list[dict] = []
    for (slug, variant, label, internal_method), g in per_seed.groupby(
        ["ablation", "variant", "label", "internal_method"], sort=False
    ):
        row = {
            "pair": args.pair,
            "ablation": slug,
            "variant": variant,
            "label": label,
            "internal_method": internal_method,
            "n_seeds": len(g),
        }
        for col in metric_cols:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = vals.mean()
            row[f"{col}_std"] = (
                vals.std(ddof=1) if vals.notna().sum() > 1 else np.nan
            )
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary["_order"] = summary["ablation"].map(order)
    summary = summary.sort_values("_order").drop(columns=["_order"])
    summary_path = (
        out_dir
        / f"ablation_{args.dataset}_{pair_tag}_all_variants_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    # One compact per-variant summary file as well.
    for _, row in summary.iterrows():
        slug = str(row["ablation"])
        variant = str(row["variant"])
        one_path = (
            out_dir
            / f"ablation_{args.dataset}_{pair_tag}_{slug}_{variant}_summary.csv"
        )
        pd.DataFrame([row]).to_csv(one_path, index=False)

    print("\nPer-seed results:")
    print(
        per_seed[
            ["seed", "variant", "label", "BLEU_mean", "chrF_mean", "validation_loss"]
        ].to_string(index=False)
    )
    print("\nAcross-seed summary:")
    print(
        summary[
            [
                "variant",
                "label",
                "n_seeds",
                "BLEU_mean_mean",
                "BLEU_mean_std",
                "chrF_mean_mean",
                "chrF_mean_std",
            ]
        ].to_string(index=False)
    )
    if missing_items:
        print(f"\nWARNING: {len(missing_items)} requested results were missing.")
    print(f"\nSaved: {per_seed_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
