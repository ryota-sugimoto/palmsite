#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def choose_best(records):
    marked = [r for r in records if bool(r.get("is_best_base_chunk"))]
    pool = marked if marked else records
    return max(pool, key=lambda r: float(r.get("P", -1e9)))


def load_best_by_base(path):
    obj = json.load(open(path))
    grouped = {}
    for k, rec in obj.items():
        if k.startswith("_"):
            continue
        base_id = rec.get("base_id", k)
        grouped.setdefault(base_id, []).append(rec)
    return {base_id: choose_best(recs) for base_id, recs in grouped.items()}


def interval_iou(a0, a1, b0, b1):
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = max(a1, b1) - min(a0, b0) + 1
    return inter / union if union > 0 else np.nan


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(s))


def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser(description="Visualization of PalmSite noise effects.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--logits-json", required=True)
    ap.add_argument("--attention-json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="palmsite_noise_effects")
    ap.add_argument("--prob-threshold", type=float, default=0.5)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    numeric_cols = [
        "rate", "replicate", "n_mutated", "span_start1", "span_end1",
        "original_P", "original_logit", "original_calibrated_logit",
        "temperature", "seed"
    ]
    for col in numeric_cols:
        if col in manifest.columns:
            manifest[col] = pd.to_numeric(manifest[col], errors="coerce")

    best_log = load_best_by_base(args.logits_json)
    log_df = pd.DataFrame([
        {
            "mutant_id": mutant_id,
            "P": float(rec.get("P", np.nan)),
            "logit": float(rec.get("logit", np.nan)),
            "calibrated_logit": float(rec.get("calibrated_logit", np.nan)),
            "temperature_out": float(rec.get("temperature", np.nan)),
        }
        for mutant_id, rec in best_log.items()
    ])

    df = manifest.merge(log_df, on="mutant_id", how="left")
    df["delta_calibrated_logit"] = df["calibrated_logit"] - df["original_calibrated_logit"]
    df["delta_logit"] = df["logit"] - df["original_logit"]
    df["delta_P"] = df["P"] - df["original_P"]

    if args.attention_json:
        best_attn = load_best_by_base(args.attention_json)
        span_start = {}
        span_end = {}
        peak_map = {}
        for mutant_id, rec in best_attn.items():
            span_start[mutant_id] = int(rec.get("orig_start", 0)) + int(rec.get("S_idx", 0)) + 1
            span_end[mutant_id] = int(rec.get("orig_start", 0)) + int(rec.get("E_idx", 0)) + 1
            w = rec.get("w", [])
            peak_map[mutant_id] = int(np.argmax(np.asarray(w, dtype=float))) if len(w) else np.nan

        original_peak = {}
        for sample, sub in df[df["target"] == "original"].groupby("sample_name"):
            mutant_id = sub.iloc[0]["mutant_id"]
            original_peak[sample] = peak_map.get(mutant_id, np.nan)

        df["mutant_span_start1"] = df["mutant_id"].map(span_start)
        df["mutant_span_end1"] = df["mutant_id"].map(span_end)

        df["span_iou"] = [
            interval_iou(int(a), int(b), int(c), int(d))
            if pd.notna(a) and pd.notna(b) and pd.notna(c) and pd.notna(d)
            else np.nan
            for a, b, c, d in zip(
                df["mutant_span_start1"], df["mutant_span_end1"],
                df["span_start1"], df["span_end1"]
            )
        ]

        df["peak_shift"] = [
            abs(int(peak_map.get(mid, np.nan)) - int(original_peak.get(sample, np.nan)))
            if pd.notna(peak_map.get(mid, np.nan)) and pd.notna(original_peak.get(sample, np.nan))
            else np.nan
            for sample, mid in zip(df["sample_name"], df["mutant_id"])
        ]
    else:
        df["span_iou"] = np.nan
        df["peak_shift"] = np.nan

    df["is_positive"] = df["P"] >= args.prob_threshold
    d = df[df["target"] != "original"].copy()

    summary = (
        d.groupby(["sample_name", "target", "rate"], as_index=False)
        .agg(
            n=("mutant_id", "size"),
            mean_delta_calibrated_logit=("delta_calibrated_logit", "mean"),
            mean_delta_P=("delta_P", "mean"),
            mean_span_iou=("span_iou", "mean"),
            mean_peak_shift=("peak_shift", "mean"),
            positive_fraction=("is_positive", "mean"),
        )
    )

    se_rows = []
    for (sample, target, rate), sub in d.groupby(["sample_name", "target", "rate"]):
        se_rows.append({
            "sample_name": sample,
            "target": target,
            "rate": rate,
            "se_delta_calibrated_logit": sem(sub["delta_calibrated_logit"]),
            "se_delta_P": sem(sub["delta_P"]),
            "se_span_iou": sem(sub["span_iou"]),
            "se_peak_shift": sem(sub["peak_shift"]),
            "se_positive_fraction": sem(sub["is_positive"]),
        })
    summary = summary.merge(pd.DataFrame(se_rows), on=["sample_name", "target", "rate"], how="left")

    merged_path = out / f"{args.prefix}.merged_metrics.tsv"
    summary_path = out / f"{args.prefix}.summary_by_target_rate.tsv"
    df.to_csv(merged_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)

    metrics = [
        ("mean_delta_calibrated_logit", "se_delta_calibrated_logit", "Mean Δ calibrated logit"),
        ("positive_fraction", "se_positive_fraction", "Positive fraction"),
    ]
    if args.attention_json:
        metrics += [
            ("mean_span_iou", "se_span_iou", "Mean span IoU"),
            ("mean_peak_shift", "se_peak_shift", "Mean attention peak shift"),
        ]

    for sample in sorted(summary["sample_name"].dropna().unique()):
        sub = summary[summary["sample_name"] == sample].copy()
        for metric, semetric, ylabel in metrics:
            fig, ax = plt.subplots(figsize=(8, 5))
            for target in sorted(sub["target"].dropna().unique()):
                t = sub[sub["target"] == target].sort_values("rate")
                ax.errorbar(
                    t["rate"], t[metric],
                    yerr=t[semetric] if semetric in t.columns else None,
                    marker="o", capsize=3, label=target
                )
            ax.set_xlabel("Mutation rate")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{sample}: {ylabel} vs mutation rate")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / f"{args.prefix}.{safe(sample)}.{metric}.png", dpi=180)
            plt.close(fig)

    for metric, _, ylabel in metrics:
        agg = summary.groupby(["target", "rate"], as_index=False)[metric].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        for target in sorted(agg["target"].dropna().unique()):
            t = agg[agg["target"] == target].sort_values("rate")
            ax.plot(t["rate"], t[metric], marker="o", label=target)
        ax.set_xlabel("Mutation rate")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Overview: {ylabel} vs mutation rate")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"{args.prefix}.overview.{metric}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(d["original_calibrated_logit"], d["calibrated_logit"], s=8)
    mn = float(min(d["original_calibrated_logit"].min(), d["calibrated_logit"].min()))
    mx = float(max(d["original_calibrated_logit"].max(), d["calibrated_logit"].max()))
    ax.plot([mn, mx], [mn, mx], linestyle="--")
    ax.set_xlabel("Original calibrated logit")
    ax.set_ylabel("Mutant calibrated logit")
    ax.set_title("Original vs mutant calibrated logit")
    fig.tight_layout()
    fig.savefig(out / f"{args.prefix}.scatter.original_vs_mutant_calibrated_logit.png", dpi=180)
    plt.close(fig)

    meta = {
        "manifest": args.manifest,
        "logits_json": args.logits_json,
        "attention_json": args.attention_json,
        "samples": sorted(df["sample_name"].dropna().unique().tolist()),
        "targets": sorted(d["target"].dropna().unique().tolist()),
        "n_rows_merged": int(len(df)),
        "n_rows_summary": int(len(summary)),
    }
    with open(out / f"{args.prefix}.meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    print("Wrote:", merged_path)
    print("Wrote:", summary_path)
    print("Output directory:", out)


if __name__ == "__main__":
    main()

