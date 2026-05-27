#!/usr/bin/env python3

"""
Analyze motif-centered PalmSite perturbation results from separate input files.

Inputs:
  1. Manifest TSV from generate_motif_centered_perturbations_v3_unique_control_windows.py
  2. PalmSite logits JSON produced from the perturbed FASTA

This direct-file version does not require a zipped output folder. Pass the manifest
and logits JSON as independent paths. Plain text and .gz inputs are both supported.

The script joins perturbed logits to the perturbation manifest, estimates the original
PalmSite logit from the original PalmSite GFF score stored in the manifest
(control_span_score), computes score/logit changes, summarizes statistics, performs
paired comparisons, and creates vector PDF figures.

Example:
  python analyze_motif_perturbation_stats_files.py \
    --manifest output/10k_motif_centered_w7_palmsite_span_controls.manifest.tsv \
    --logits-json output/10k_motif_centered_w7_palmsite_span_controls.logits.json \
    --outdir motif_perturbation_analysis_w7 \
    --drop-nonfinite
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


TARGET_ORDER = [
    "motif_A",
    "motif_B",
    "motif_C",
    "random_in_span",
    "random_outside_span",
]

TARGET_LABELS = {
    "motif_A": "Motif A",
    "motif_B": "Motif B",
    "motif_C": "Motif C",
    "random_in_span": "Random in-span",
    "random_outside_span": "Random outside-span",
}


@dataclass
class InputHandles:
    manifest_handle: BinaryIO
    logits_handle: BinaryIO
    manifest_name: str
    logits_name: str


def eprint(*args: object) -> None:
    print(*args, file=sys.stderr, flush=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit_from_probability(p: pd.Series, eps: float) -> pd.Series:
    p_clip = p.clip(lower=eps, upper=1.0 - eps)
    return np.log(p_clip / (1.0 - p_clip))



def open_inputs(args: argparse.Namespace) -> InputHandles:
    """Open explicit manifest and logits files.

    This direct-file version intentionally does not require --input-zip. Both input
    files may be plain text or gzip-compressed (.gz).
    """
    if not args.manifest or not args.logits_json:
        raise ValueError("Provide both --manifest and --logits-json")

    def open_plain_or_gzip(path: str) -> BinaryIO:
        if path.endswith(".gz"):
            return gzip.open(path, "rb")
        return open(path, "rb")

    eprint(f"Reading manifest file: {args.manifest}")
    eprint(f"Reading logits JSON file: {args.logits_json}")
    return InputHandles(
        open_plain_or_gzip(args.manifest),
        open_plain_or_gzip(args.logits_json),
        args.manifest,
        args.logits_json,
    )


def read_manifest(handle: BinaryIO) -> pd.DataFrame:
    usecols = [
        "perturb_id",
        "original_id",
        "perturbation_class",
        "target_name",
        "motif",
        "window_size",
        "window_start_1based",
        "window_end_1based",
        "mutation_rate",
        "mutation_mode",
        "replicate",
        "control_window_replicate",
        "n_mutated",
        "seq_len",
        "control_span_source",
        "control_span_start_1based",
        "control_span_end_1based",
        "control_span_feature_type",
        "control_span_score",
        "control_span_raw_score",
    ]
    dtype = {
        "perturb_id": "string",
        "original_id": "string",
        "perturbation_class": "category",
        "target_name": "category",
        "motif": "category",
        "window_size": "int32",
        "window_start_1based": "int32",
        "window_end_1based": "int32",
        "mutation_rate": "float32",
        "mutation_mode": "category",
        "replicate": "Int16",
        "control_window_replicate": "Int16",
        "n_mutated": "int16",
        "seq_len": "int32",
        "control_span_source": "category",
        "control_span_start_1based": "Int32",
        "control_span_end_1based": "Int32",
        "control_span_feature_type": "category",
        "control_span_score": "float64",
        "control_span_raw_score": "string",
    }
    eprint("Reading manifest TSV...")
    df = pd.read_csv(handle, sep="\t", usecols=usecols, dtype=dtype, low_memory=False)
    eprint(f"Manifest rows: {len(df):,}")
    return df


def parse_logits_json_lines(handle: BinaryIO) -> pd.DataFrame:
    """Parse PalmSite logits JSON efficiently when each object item is on one line."""
    base_ids: List[str] = []
    ps: List[float] = []
    logits: List[float] = []
    cal_logits: List[float] = []
    lengths: List[int] = []
    s_idx: List[float] = []
    e_idx: List[float] = []
    mu: List[float] = []
    sigma: List[float] = []

    eprint("Streaming logits JSON...")
    n_seen = 0
    n_kept = 0
    text = io.TextIOWrapper(handle, encoding="utf-8")
    for raw_line in text:
        line = raw_line.strip()
        if not line or line in {"{", "}"}:
            continue
        if line.startswith('"_meta"'):
            continue
        if not line.startswith('"'):
            continue

        if line.endswith(","):
            line = line[:-1]
        try:
            key_json = "{" + line + "}"
            one = json.loads(key_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Could not parse logits JSON line near: {line[:200]}") from exc

        # The wrapped object has exactly one key/value pair.
        _, rec = next(iter(one.items()))
        n_seen += 1

        if rec.get("is_best_base_chunk", True) is not True:
            continue

        base_id = rec.get("base_id")
        if base_id is None:
            chunk_id = rec.get("chunk_id", "")
            base_id = chunk_id.split("|chunk_", 1)[0]

        base_ids.append(str(base_id))
        ps.append(float(rec.get("P", np.nan)))
        logits.append(float(rec.get("logit", np.nan)))
        cal_logits.append(float(rec.get("calibrated_logit", np.nan)))
        lengths.append(int(rec.get("L", -1)))
        s_idx.append(float(rec.get("S_idx", np.nan)))
        e_idx.append(float(rec.get("E_idx", np.nan)))
        mu.append(float(rec.get("mu", np.nan)))
        sigma.append(float(rec.get("sigma", np.nan)))
        n_kept += 1

        if n_kept % 100000 == 0:
            eprint(f"  kept {n_kept:,} best-chunk logits...")

    df = pd.DataFrame(
        {
            "perturb_id": pd.Series(base_ids, dtype="string"),
            "perturbed_P": np.asarray(ps, dtype=np.float64),
            "perturbed_logit": np.asarray(logits, dtype=np.float64),
            "perturbed_calibrated_logit": np.asarray(cal_logits, dtype=np.float64),
            "perturbed_L": np.asarray(lengths, dtype=np.int32),
            "perturbed_S_idx": np.asarray(s_idx, dtype=np.float64),
            "perturbed_E_idx": np.asarray(e_idx, dtype=np.float64),
            "perturbed_mu": np.asarray(mu, dtype=np.float64),
            "perturbed_sigma": np.asarray(sigma, dtype=np.float64),
        }
    )
    eprint(f"Logit records seen: {n_seen:,}; kept best chunks: {len(df):,}")
    return df


def add_effect_columns(df: pd.DataFrame, eps: float) -> pd.DataFrame:
    df = df.copy()
    df["original_P"] = df["control_span_score"].astype(float)
    df["original_calibrated_logit_est"] = logit_from_probability(df["original_P"], eps=eps)
    df["delta_P"] = df["perturbed_P"] - df["original_P"]
    df["P_loss"] = -df["delta_P"]
    df["delta_calibrated_logit"] = df["perturbed_calibrated_logit"] - df["original_calibrated_logit_est"]
    df["logit_loss"] = -df["delta_calibrated_logit"]
    df["score_decreased"] = df["delta_calibrated_logit"] < 0
    df["target_name"] = df["target_name"].astype(str)
    return df


def bootstrap_ci(values: np.ndarray, func=np.mean, n_boot: int = 1000, seed: int = 1) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (np.nan, np.nan)
    if len(values) == 1 or n_boot <= 0:
        val = float(func(values))
        return (val, val)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = func(values[idx], axis=1)
    return (float(np.nanpercentile(boot, 2.5)), float(np.nanpercentile(boot, 97.5)))


def summarize_condition(df_agg: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["window_size", "mutation_rate", "target_name"]
    for keys, g in df_agg.groupby(group_cols, observed=True):
        w, r, target = keys
        x = g["logit_loss"].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        p_loss = g["P_loss"].to_numpy(dtype=float)
        p_loss = p_loss[np.isfinite(p_loss)]
        mean_ci = bootstrap_ci(x, np.mean, n_boot=n_boot, seed=seed)
        median_ci = bootstrap_ci(x, np.median, n_boot=n_boot, seed=seed + 17)
        p_mean_ci = bootstrap_ci(p_loss, np.mean, n_boot=n_boot, seed=seed + 29)
        rows.append(
            {
                "window_size": w,
                "mutation_rate": r,
                "target_name": target,
                "n_originals": int(g["original_id"].nunique()),
                "n_condition_rows": int(len(g)),
                "mean_logit_loss": float(np.nanmean(x)),
                "mean_logit_loss_ci95_low": mean_ci[0],
                "mean_logit_loss_ci95_high": mean_ci[1],
                "median_logit_loss": float(np.nanmedian(x)),
                "median_logit_loss_ci95_low": median_ci[0],
                "median_logit_loss_ci95_high": median_ci[1],
                "q25_logit_loss": float(np.nanpercentile(x, 25)),
                "q75_logit_loss": float(np.nanpercentile(x, 75)),
                "std_logit_loss": float(np.nanstd(x, ddof=1)) if len(x) > 1 else np.nan,
                "mean_P_loss": float(np.nanmean(p_loss)),
                "mean_P_loss_ci95_low": p_mean_ci[0],
                "mean_P_loss_ci95_high": p_mean_ci[1],
                "median_P_loss": float(np.nanmedian(p_loss)),
                "fraction_logit_decreased": float(np.nanmean(g["score_decreased"].to_numpy(dtype=float))),
            }
        )
    out = pd.DataFrame(rows)
    out["target_order"] = out["target_name"].map({v: i for i, v in enumerate(TARGET_ORDER)}).fillna(999).astype(int)
    out = out.sort_values(["window_size", "mutation_rate", "target_order", "target_name"]).drop(columns=["target_order"])
    return out


def benjamini_hochberg(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    p_valid = p[mask]
    if len(p_valid) == 0:
        return out
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    m = len(ranked)
    adjusted = ranked * m / (np.arange(1, m + 1))
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    tmp = np.empty_like(adjusted)
    tmp[order] = adjusted
    out[mask] = tmp
    return out


def paired_comparisons(df_agg: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if stats is None:
        eprint("scipy is unavailable; paired p-values will be omitted.")

    for (w, r), g in df_agg.groupby(["window_size", "mutation_rate"], observed=True):
        pivot = g.pivot_table(index="original_id", columns="target_name", values="logit_loss", aggfunc="mean")
        motifs = [m for m in ["motif_A", "motif_B", "motif_C"] if m in pivot.columns]
        if motifs:
            pivot["motif_mean"] = pivot[motifs].mean(axis=1)

        pairs = []
        for motif in motifs:
            pairs.append((motif, "random_in_span"))
            pairs.append((motif, "random_outside_span"))
        pairs.extend(
            [
                ("motif_mean", "random_in_span"),
                ("motif_mean", "random_outside_span"),
                ("random_in_span", "random_outside_span"),
            ]
        )

        for lhs, rhs in pairs:
            if lhs not in pivot.columns or rhs not in pivot.columns:
                continue
            paired = pivot[[lhs, rhs]].dropna()
            if len(paired) == 0:
                continue
            diff = paired[lhs].to_numpy(dtype=float) - paired[rhs].to_numpy(dtype=float)
            diff = diff[np.isfinite(diff)]
            if len(diff) == 0:
                continue
            p_t = np.nan
            p_w = np.nan
            if stats is not None and len(diff) >= 2:
                try:
                    p_t = float(stats.ttest_rel(paired[lhs], paired[rhs], nan_policy="omit").pvalue)
                except Exception:
                    p_t = np.nan
                try:
                    # For large n, scipy's approximate method is faster and stable.
                    p_w = float(stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", method="approx").pvalue)
                except Exception:
                    p_w = np.nan
            sd = float(np.nanstd(diff, ddof=1)) if len(diff) > 1 else np.nan
            rows.append(
                {
                    "window_size": w,
                    "mutation_rate": r,
                    "lhs": lhs,
                    "rhs": rhs,
                    "comparison": f"{lhs} - {rhs}",
                    "n_pairs": int(len(diff)),
                    "mean_difference_logit_loss": float(np.nanmean(diff)),
                    "median_difference_logit_loss": float(np.nanmedian(diff)),
                    "q25_difference_logit_loss": float(np.nanpercentile(diff, 25)),
                    "q75_difference_logit_loss": float(np.nanpercentile(diff, 75)),
                    "cohen_dz": float(np.nanmean(diff) / sd) if sd and np.isfinite(sd) and sd > 0 else np.nan,
                    "paired_t_pvalue": p_t,
                    "wilcoxon_pvalue": p_w,
                    "interpretation": (
                        "lhs stronger score drop" if np.nanmedian(diff) > 0 else "rhs stronger score drop"
                    ),
                }
            )
    out = pd.DataFrame(rows)
    if len(out):
        out["paired_t_pvalue_bh"] = benjamini_hochberg(out["paired_t_pvalue"].to_numpy(dtype=float))
        out["wilcoxon_pvalue_bh"] = benjamini_hochberg(out["wilcoxon_pvalue"].to_numpy(dtype=float))
        out = out.sort_values(["window_size", "mutation_rate", "comparison"])
    return out


def write_topline_report(
    outdir: Path,
    manifest: pd.DataFrame,
    joined: pd.DataFrame,
    df_agg: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    eps: float,
) -> None:
    lines: List[str] = []
    lines.append("# Motif-centered PalmSite perturbation analysis")
    lines.append("")
    lines.append("## Input summary")
    lines.append("")
    lines.append(f"- Manifest rows: {len(manifest):,}")
    lines.append(f"- Joined perturbation rows: {len(joined):,}")
    lines.append(f"- Unique original proteins in joined data: {joined['original_id'].nunique():,}")
    lines.append(f"- Per-original aggregated rows: {len(df_agg):,}")
    lines.append(f"- Original calibrated logits were estimated from manifest `control_span_score` using clipped logit with eps={eps:g}.")
    lines.append("")

    lines.append("## Generated rows by condition")
    lines.append("")
    count_tbl = (
        joined.groupby(["window_size", "mutation_rate", "target_name"], observed=True)
        .size()
        .reset_index(name="n_rows")
        .sort_values(["window_size", "mutation_rate", "target_name"])
    )
    lines.append(count_tbl.to_markdown(index=False))
    lines.append("")

    lines.append("## Main condition summary")
    lines.append("")
    display_cols = [
        "window_size",
        "mutation_rate",
        "target_name",
        "n_originals",
        "mean_logit_loss",
        "median_logit_loss",
        "q25_logit_loss",
        "q75_logit_loss",
        "mean_P_loss",
        "fraction_logit_decreased",
    ]
    lines.append(summary[display_cols].to_markdown(index=False, floatfmt=".6g"))
    lines.append("")

    if len(comparisons):
        lines.append("## Key paired comparisons")
        lines.append("")
        key = comparisons[comparisons["lhs"].isin(["motif_mean", "random_in_span"])]
        key_cols = [
            "window_size",
            "mutation_rate",
            "comparison",
            "n_pairs",
            "median_difference_logit_loss",
            "cohen_dz",
            "wilcoxon_pvalue_bh",
            "interpretation",
        ]
        lines.append(key[key_cols].to_markdown(index=False, floatfmt=".6g"))
        lines.append("")

    lines.append("## Interpretation note")
    lines.append("")
    lines.append("`logit_loss = original_calibrated_logit_est - perturbed_calibrated_logit`; therefore, larger positive values indicate stronger reduction of the PalmSite score after perturbation.")
    lines.append("")

    (outdir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def prepare_plot_summary(summary: pd.DataFrame) -> pd.DataFrame:
    df = summary.copy()
    df["target_label"] = df["target_name"].map(TARGET_LABELS).fillna(df["target_name"])
    df["target_order"] = df["target_name"].map({v: i for i, v in enumerate(TARGET_ORDER)}).fillna(999).astype(int)
    return df.sort_values(["target_order", "mutation_rate"])


def plot_dose_response(summary: pd.DataFrame, outdir: Path) -> None:
    df = prepare_plot_summary(summary)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for target in [t for t in TARGET_ORDER if t in set(df["target_name"])] + [t for t in sorted(set(df["target_name"])) if t not in TARGET_ORDER]:
        sub = df[df["target_name"] == target].sort_values("mutation_rate")
        if len(sub) == 0:
            continue
        y = sub["mean_logit_loss"].to_numpy(dtype=float)
        ylo = sub["mean_logit_loss_ci95_low"].to_numpy(dtype=float)
        yhi = sub["mean_logit_loss_ci95_high"].to_numpy(dtype=float)
        yerr = np.vstack([y - ylo, yhi - y])
        ax.errorbar(
            sub["mutation_rate"].to_numpy(dtype=float),
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=TARGET_LABELS.get(target, target),
        )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Mutation rate within perturbed window")
    ax.set_ylabel("Mean PalmSite logit loss\n(original − perturbed)")
    ax.set_title("Dose response of motif-centered and control perturbations")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "dose_response_logit_loss.pdf")
    fig.savefig(outdir / "dose_response_logit_loss.png", dpi=300)
    plt.close(fig)


def plot_fraction_decreased(summary: pd.DataFrame, outdir: Path) -> None:
    df = prepare_plot_summary(summary)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for target in [t for t in TARGET_ORDER if t in set(df["target_name"])] + [t for t in sorted(set(df["target_name"])) if t not in TARGET_ORDER]:
        sub = df[df["target_name"] == target].sort_values("mutation_rate")
        if len(sub) == 0:
            continue
        ax.plot(
            sub["mutation_rate"].to_numpy(dtype=float),
            sub["fraction_logit_decreased"].to_numpy(dtype=float),
            marker="o",
            linewidth=1.5,
            label=TARGET_LABELS.get(target, target),
        )
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mutation rate within perturbed window")
    ax.set_ylabel("Fraction with decreased PalmSite logit")
    ax.set_title("Fraction of perturbations reducing PalmSite confidence")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fraction_logit_decreased.pdf")
    fig.savefig(outdir / "fraction_logit_decreased.png", dpi=300)
    plt.close(fig)


def plot_boxplot(df_agg: pd.DataFrame, outdir: Path) -> None:
    # Plot the largest mutation rate by default because it is easiest to interpret.
    max_rate = float(np.nanmax(df_agg["mutation_rate"].to_numpy(dtype=float)))
    sub = df_agg[np.isclose(df_agg["mutation_rate"].astype(float), max_rate)].copy()
    order = [t for t in TARGET_ORDER if t in set(sub["target_name"])] + [t for t in sorted(set(sub["target_name"])) if t not in TARGET_ORDER]
    data = [sub.loc[sub["target_name"] == t, "logit_loss"].dropna().to_numpy(dtype=float) for t in order]
    labels = [TARGET_LABELS.get(t, t) for t in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.boxplot(data, tick_labels=labels, showfliers=False, whis=(5, 95))
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_ylabel("PalmSite logit loss\n(original − perturbed)")
    ax.set_title(f"Per-protein perturbation effects at mutation rate {max_rate:g}")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(outdir / f"boxplot_logit_loss_rate_{max_rate:g}.pdf")
    fig.savefig(outdir / f"boxplot_logit_loss_rate_{max_rate:g}.png", dpi=300)
    plt.close(fig)


def plot_motif_vs_controls(df_agg: pd.DataFrame, outdir: Path) -> None:
    tmp = df_agg.copy()
    tmp["group"] = tmp["target_name"]
    tmp.loc[tmp["target_name"].isin(["motif_A", "motif_B", "motif_C"]), "group"] = "motif_mean"
    # First average motifs per original/rate; controls are already one condition per original/rate.
    group_agg = (
        tmp.groupby(["original_id", "window_size", "mutation_rate", "group"], observed=True)
        .agg(logit_loss=("logit_loss", "mean"))
        .reset_index()
    )
    rows = []
    for (w, r, gname), g in group_agg.groupby(["window_size", "mutation_rate", "group"], observed=True):
        x = g["logit_loss"].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(x, np.mean, n_boot=1000, seed=123)
        rows.append(
            {
                "window_size": w,
                "mutation_rate": r,
                "group": gname,
                "mean": np.nanmean(x),
                "lo": lo,
                "hi": hi,
            }
        )
    s = pd.DataFrame(rows)
    order = ["motif_mean", "random_in_span", "random_outside_span"]
    labels = {
        "motif_mean": "Motif A/B/C mean",
        "random_in_span": "Random in-span",
        "random_outside_span": "Random outside-span",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for group in [x for x in order if x in set(s["group"])] + [x for x in sorted(set(s["group"])) if x not in order]:
        sub = s[s["group"] == group].sort_values("mutation_rate")
        y = sub["mean"].to_numpy(dtype=float)
        yerr = np.vstack([y - sub["lo"].to_numpy(dtype=float), sub["hi"].to_numpy(dtype=float) - y])
        ax.errorbar(
            sub["mutation_rate"].to_numpy(dtype=float),
            y,
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=1.5,
            label=labels.get(group, group),
        )
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_xlabel("Mutation rate within perturbed window")
    ax.set_ylabel("Mean PalmSite logit loss\n(original − perturbed)")
    ax.set_title("Motif-centered perturbation versus control windows")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "motif_mean_vs_controls_logit_loss.pdf")
    fig.savefig(outdir / "motif_mean_vs_controls_logit_loss.png", dpi=300)
    plt.close(fig)


def make_combined_pdf(outdir: Path, image_paths: Sequence[Path]) -> None:
    # Recreate a simple multipage PDF from the already generated raster previews.
    # The individual PDFs above remain the preferred vector outputs.
    with PdfPages(outdir / "motif_perturbation_figures_preview.pdf") as pdf:
        for path in image_paths:
            if not path.exists():
                continue
            img = plt.imread(path)
            fig, ax = plt.subplots(figsize=(8, 5.5))
            ax.imshow(img)
            ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    handles = open_inputs(args)
    manifest = read_manifest(handles.manifest_handle)
    logits = parse_logits_json_lines(handles.logits_handle)

    eprint("Joining manifest and logits...")
    joined = manifest.merge(logits, on="perturb_id", how="inner", validate="one_to_one")
    missing = len(manifest) - len(joined)
    if missing:
        eprint(f"Warning: {missing:,} manifest rows were not matched to a best-chunk logit record.")
    joined = add_effect_columns(joined, eps=args.logit_eps)

    if args.drop_nonfinite:
        before = len(joined)
        joined = joined[np.isfinite(joined["logit_loss"]) & np.isfinite(joined["P_loss"])].copy()
        eprint(f"Dropped {before - len(joined):,} rows with nonfinite effects.")

    eprint("Aggregating replicates/windows per original protein and condition...")
    df_agg = (
        joined.groupby(["original_id", "window_size", "mutation_rate", "target_name"], observed=True)
        .agg(
            perturbation_class=("perturbation_class", "first"),
            motif=("motif", "first"),
            n_rows=("perturb_id", "count"),
            original_P=("original_P", "mean"),
            perturbed_P=("perturbed_P", "mean"),
            P_loss=("P_loss", "mean"),
            original_calibrated_logit_est=("original_calibrated_logit_est", "mean"),
            perturbed_calibrated_logit=("perturbed_calibrated_logit", "mean"),
            delta_calibrated_logit=("delta_calibrated_logit", "mean"),
            logit_loss=("logit_loss", "mean"),
            score_decreased=("score_decreased", "mean"),
            mean_n_mutated=("n_mutated", "mean"),
        )
        .reset_index()
    )
    # At aggregate level, mark decreased if the mean delta is negative.
    df_agg["score_decreased"] = df_agg["delta_calibrated_logit"] < 0

    eprint("Summarizing conditions...")
    summary = summarize_condition(df_agg, n_boot=args.bootstrap, seed=args.seed)
    comparisons = paired_comparisons(df_agg)

    eprint("Writing outputs...")
    summary.to_csv(outdir / "condition_summary.tsv", sep="\t", index=False, lineterminator="\n")
    comparisons.to_csv(outdir / "paired_comparisons.tsv", sep="\t", index=False, lineterminator="\n")
    df_agg.to_csv(outdir / "per_original_condition_effects.tsv.gz", sep="\t", index=False, compression="gzip", lineterminator="\n")

    if args.write_joined:
        joined.to_csv(outdir / "joined_per_perturbation_effects.tsv.gz", sep="\t", index=False, compression="gzip", lineterminator="\n")

    write_topline_report(outdir, manifest, joined, df_agg, summary, comparisons, eps=args.logit_eps)

    eprint("Creating figures...")
    plot_dose_response(summary, outdir)
    plot_fraction_decreased(summary, outdir)
    plot_boxplot(df_agg, outdir)
    plot_motif_vs_controls(df_agg, outdir)
    if args.combined_preview_pdf:
        make_combined_pdf(
            outdir,
            [
                outdir / "dose_response_logit_loss.png",
                outdir / "motif_mean_vs_controls_logit_loss.png",
                outdir / "fraction_logit_decreased.png",
                outdir / f"boxplot_logit_loss_rate_{float(np.nanmax(df_agg['mutation_rate'].to_numpy(dtype=float))):g}.png",
            ],
        )

    eprint(f"Done. Outputs written to: {outdir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze motif-centered PalmSite perturbation logits and manifest.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest", required=True, help="Perturbation manifest TSV path. Plain text or .gz is supported.")
    p.add_argument("--logits-json", required=True, help="PalmSite logits JSON path for perturbed sequences. Plain text or .gz is supported.")
    p.add_argument("--outdir", default="motif_perturbation_analysis", help="Output directory.")
    p.add_argument("--logit-eps", type=float, default=1e-7, help="Clipping epsilon when converting original P to logit.")
    p.add_argument("--bootstrap", type=int, default=1000, help="Bootstrap replicates for 95%% CIs.")
    p.add_argument("--seed", type=int, default=1, help="Random seed for bootstrap CIs.")
    p.add_argument("--write-joined", action="store_true", help="Also write large row-level joined table.")
    p.add_argument("--combined-preview-pdf", action="store_true", help="Also create a rasterized multi-page preview PDF from the PNG figures.")
    p.add_argument("--drop-nonfinite", action="store_true", help="Drop rows with nonfinite effect values.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()

