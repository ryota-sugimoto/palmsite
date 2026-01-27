#!/usr/bin/env python3
"""
cv_fpr_plot_single.py

Single-program CV threshold-vs-FPR analysis on negative-only data with fold splits.

Inputs (assumed):
  folds TSV : seq_id <TAB> fold
  scores TSV: seq_id <TAB> value
Notes:
  - scores file may omit sequences; omitted => "no hit" => never predicted positive.
  - duplicates in scores are resolved as:
      metric=score  -> keep max
      metric=evalue -> keep min

Operating points:
  A) CV per-split operating points:
     For each held-out fold, choose threshold on training folds with achieved_train_fpr <= target.
     Evaluate held-out FPR at that threshold.

  B) GLOBAL operating point (requested):
     Choose a single threshold on ALL negatives combined such that achieved_global_fpr <= target.
     Report that threshold and its achieved global FPR, and evaluate per-fold FPR at that threshold.

Outputs:
  - multi-page PDF:
      page 1: GLOBAL curve with global operating points and per-fold FPRs at global thresholds
      pages 2..: one page per held-out fold (train vs held-out + CV operating points)
  - TSV summary:
      rows for CV splits (like before)
      + rows for GLOBAL operating points (one per target)

Axes:
  - y: log(FPR) always
  - x: log for evalue; linear for score
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


@dataclass
class SplitResult:
    mode: str  # "CV" or "GLOBAL"
    metric: str
    heldout_fold: str  # int for CV rows, "ALL" for global rows
    train_folds: str   # e.g. "1,2" or "ALL"
    target_fpr: float
    threshold: float
    threshold_source: str  # "data" or "boundary"
    train_fpr: float       # for GLOBAL rows: achieved global FPR
    heldout_fpr: float     # for GLOBAL rows: blank/NaN (kept for schema) OR could mirror global
    n_train_total: int
    n_train_scored: int
    n_hold_total: int
    n_hold_scored: int
    # For GLOBAL mode: per-fold FPRs at the global threshold (packed string)
    per_fold_fprs: str = ""


def _is_header_or_invalid_float(s: str) -> bool:
    try:
        float(s)
        return False
    except Exception:
        return True


def read_folds_tsv(path: str) -> Tuple[Dict[str, int], Dict[int, int]]:
    seq_to_fold: Dict[str, int] = {}
    fold_sizes: Dict[int, int] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            seq_id, fold_s = parts[0], parts[1]
            if _is_header_or_invalid_float(fold_s):
                continue
            fold = int(float(fold_s))
            seq_to_fold[seq_id] = fold
            fold_sizes[fold] = fold_sizes.get(fold, 0) + 1

    if not seq_to_fold:
        raise ValueError(f"No fold assignments parsed from: {path}")

    return seq_to_fold, fold_sizes


def read_scores_tsv(path: str, metric: str, evalue_floor: float) -> Dict[str, float]:
    if metric not in ("score", "evalue"):
        raise ValueError("metric must be 'score' or 'evalue'")

    best: Dict[str, float] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            seq_id, val_s = parts[0], parts[1]
            if _is_header_or_invalid_float(val_s):
                continue

            v = float(val_s)
            if not math.isfinite(v):
                continue

            if metric == "evalue" and v <= 0.0:
                v = evalue_floor

            if seq_id not in best:
                best[seq_id] = v
            else:
                if metric == "score":
                    if v > best[seq_id]:
                        best[seq_id] = v
                else:
                    if v < best[seq_id]:
                        best[seq_id] = v

    return best


def compute_threshold_fpr_curve(
    finite_scores: List[float],
    n_total: int,
    metric: str,
) -> Tuple[List[float], List[float]]:
    if n_total <= 0:
        return [], []
    if metric not in ("score", "evalue"):
        raise ValueError("metric must be 'score' or 'evalue'")
    if not finite_scores:
        return [], []

    counts = Counter(finite_scores)
    thresholds = sorted(counts.keys())  # ascending
    fprs: List[float] = []

    if metric == "evalue":
        cum = 0
        for thr in thresholds:
            cum += counts[thr]
            fprs.append(cum / n_total)
    else:
        remaining = sum(counts.values())
        for thr in thresholds:
            fprs.append(remaining / n_total)
            remaining -= counts[thr]

    return thresholds, fprs


def fpr_at_threshold(
    finite_scores: List[float],
    n_total: int,
    metric: str,
    threshold: float,
) -> float:
    if n_total <= 0:
        return 0.0
    if not finite_scores:
        return 0.0

    if metric == "score":
        fp = sum(1 for v in finite_scores if v >= threshold)
    else:
        fp = sum(1 for v in finite_scores if v <= threshold)
    return fp / n_total


def pick_operating_threshold(
    thresholds: List[float],
    fprs: List[float],
    target_fpr: float,
    metric: str,
    evalue_floor: float,
) -> Tuple[float, float, str]:
    """
    Conservative: achieved_fpr <= target_fpr and as close as possible to target.

    If impossible (e.g., target < 1/n_total resolution), choose boundary threshold achieving FPR=0:
      - score  -> threshold > max_score
      - evalue -> threshold < min_evalue
    """
    if target_fpr <= 0:
        if metric == "score":
            return math.inf, 0.0, "boundary"
        thr = math.nextafter(evalue_floor, -math.inf)
        return thr, 0.0, "boundary"

    if not thresholds or not fprs:
        if metric == "score":
            return math.inf, 0.0, "boundary"
        thr = math.nextafter(evalue_floor, -math.inf)
        return thr, 0.0, "boundary"

    if metric == "evalue":
        idx = -1
        for i, f in enumerate(fprs):
            if f <= target_fpr:
                idx = i
            else:
                break
        if idx >= 0:
            return thresholds[idx], fprs[idx], "data"
        thr0 = math.nextafter(thresholds[0], -math.inf)
        if thr0 <= 0.0:
            thr0 = math.nextafter(evalue_floor, -math.inf)
        return thr0, 0.0, "boundary"

    else:
        for thr, f in zip(thresholds, fprs):
            if f <= target_fpr:
                return thr, f, "data"
        thr0 = math.nextafter(thresholds[-1], math.inf)
        return thr0, 0.0, "boundary"


def _safe_for_log_y(y: float, y_floor: float) -> float:
    return y if y > y_floor else y_floor


def write_results_tsv(path: str, rows: List[SplitResult]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow([
            "mode",
            "metric",
            "heldout_fold",
            "train_folds",
            "target_fpr",
            "threshold",
            "threshold_source",
            "train_fpr",     # GLOBAL: achieved_global_fpr
            "heldout_fpr",   # CV: achieved held-out FPR; GLOBAL: left as NaN
            "n_train_total",
            "n_train_scored",
            "n_hold_total",
            "n_hold_scored",
            "per_fold_fprs", # GLOBAL: "fold0=...,fold1=...,fold2=..."
        ])
        for r in rows:
            w.writerow([
                r.mode,
                r.metric,
                r.heldout_fold,
                r.train_folds,
                f"{r.target_fpr:.0e}",
                r.threshold,
                r.threshold_source,
                r.train_fpr,
                r.heldout_fpr,
                r.n_train_total,
                r.n_train_scored,
                r.n_hold_total,
                r.n_hold_scored,
                r.per_fold_fprs,
            ])


def plot_global_page(
    ax,
    metric: str,
    fold_ids: List[int],
    fold_sizes: Dict[int, int],
    fold_scores: Dict[int, List[float]],
    targets: List[float],
    evalue_floor: float,
    title: Optional[str],
    y_floor: float,
) -> List[SplitResult]:
    """
    Plot GLOBAL threshold-vs-FPR curve (all folds combined) and annotate global operating points.
    Also compute per-fold FPR at global thresholds and return GLOBAL rows for TSV.
    """
    all_scores: List[float] = []
    for f in fold_ids:
        all_scores.extend(fold_scores.get(f, []))

    n_all_total = sum(fold_sizes[f] for f in fold_ids)
    n_all_scored = len(all_scores)

    thr_all, fpr_all = compute_threshold_fpr_curve(all_scores, n_all_total, metric)

    page_title = title or f"GLOBAL threshold vs FPR ({metric})"
    ax.set_title(page_title)

    ax.set_yscale("log")
    if metric == "evalue":
        ax.set_xscale("log")
        ax.set_xlabel("E-value threshold")
    else:
        ax.set_xlabel("Score threshold")
    ax.set_ylabel("False Positive Rate (FPR)")

    if thr_all and fpr_all:
        ax.plot(thr_all, fpr_all, label="All folds combined")
    else:
        ax.text(0.02, 0.95, "No scored values in ALL folds", transform=ax.transAxes, va="top")

    for t in targets:
        ax.axhline(t, linestyle=":", linewidth=1.0)
        ax.text(
            0.98, t, f" target {t:.0e}",
            transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8
        )

    global_rows: List[SplitResult] = []

    for t in targets:
        thr_g, achieved_g, src = pick_operating_threshold(thr_all, fpr_all, t, metric, evalue_floor)

        # per-fold FPRs at thr_g
        per = []
        for f in fold_ids:
            fpr_f = fpr_at_threshold(fold_scores.get(f, []), fold_sizes[f], metric, thr_g)
            per.append((f, fpr_f))

        # Plot marker
        ax.scatter([thr_g], [_safe_for_log_y(achieved_g, y_floor)], marker="D", zorder=6)

        # Annotate: global + per-fold
        per_txt = ", ".join([f"{f}={v:.2g}" for f, v in per])
        ax.annotate(
            f"GLOBAL thr={thr_g:.3g}\nGLOBAL FPR={achieved_g:.2g}\nper-fold: {per_txt}",
            (thr_g, _safe_for_log_y(achieved_g, y_floor)),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

        global_rows.append(
            SplitResult(
                mode="GLOBAL",
                metric=metric,
                heldout_fold="ALL",
                train_folds="ALL",
                target_fpr=t,
                threshold=thr_g,
                threshold_source=src,
                train_fpr=achieved_g,          # store achieved global FPR here
                heldout_fpr=float("nan"),
                n_train_total=n_all_total,
                n_train_scored=n_all_scored,
                n_hold_total=0,
                n_hold_scored=0,
                per_fold_fprs=",".join([f"fold{f}={v:.6g}" for f, v in per]),
            )
        )

    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(loc="best", frameon=True)
    ax.set_ylim(bottom=y_floor, top=1.0)

    return global_rows


def plot_cv_pages(
    pdf: PdfPages,
    metric: str,
    fold_ids: List[int],
    fold_sizes: Dict[int, int],
    fold_scores: Dict[int, List[float]],
    targets: List[float],
    evalue_floor: float,
    title: Optional[str],
    y_floor: float,
) -> List[SplitResult]:
    """
    Add one page per held-out fold (train vs held-out) with CV operating points.
    """
    rows: List[SplitResult] = []

    for heldout in fold_ids:
        train_folds = [f for f in fold_ids if f != heldout]

        n_train_total = sum(fold_sizes[f] for f in train_folds)
        n_hold_total = fold_sizes[heldout]

        train_scores: List[float] = []
        for f in train_folds:
            train_scores.extend(fold_scores.get(f, []))
        hold_scores = fold_scores.get(heldout, [])

        train_thr, train_fpr = compute_threshold_fpr_curve(train_scores, n_train_total, metric)
        hold_thr, hold_fpr = compute_threshold_fpr_curve(hold_scores, n_hold_total, metric)

        fig = plt.figure(figsize=(8.5, 6.0))
        ax = fig.add_subplot(111)

        ax.set_yscale("log")
        if metric == "evalue":
            ax.set_xscale("log")
            ax.set_xlabel("E-value threshold")
        else:
            ax.set_xlabel("Score threshold")
        ax.set_ylabel("False Positive Rate (FPR)")

        page_title = title or f"CV threshold vs FPR ({metric})"
        ax.set_title(f"{page_title} | train={','.join(map(str, train_folds))} heldout={heldout}")

        if train_thr and train_fpr:
            ax.plot(train_thr, train_fpr, label="Train folds")
        else:
            ax.text(0.02, 0.95, "No scored values in train folds", transform=ax.transAxes, va="top")

        if hold_thr and hold_fpr:
            ax.plot(hold_thr, hold_fpr, linestyle="--", label="Held-out fold")
        else:
            ax.text(0.02, 0.90, "No scored values in held-out fold", transform=ax.transAxes, va="top")

        for t in targets:
            ax.axhline(t, linestyle=":", linewidth=1.0)
            ax.text(
                0.98, t, f" target {t:.0e}",
                transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=8
            )

        for t in targets:
            thr, achieved_train, src = pick_operating_threshold(
                thresholds=train_thr,
                fprs=train_fpr,
                target_fpr=t,
                metric=metric,
                evalue_floor=evalue_floor,
            )
            achieved_hold = fpr_at_threshold(hold_scores, n_hold_total, metric, thr)

            ax.scatter([thr], [_safe_for_log_y(achieved_train, y_floor)], marker="o", zorder=5)
            ax.scatter([thr], [_safe_for_log_y(achieved_hold, y_floor)], marker="x", zorder=6)

            ax.annotate(
                f"thr={thr:.3g}\ntrain={achieved_train:.3g}\nheld={achieved_hold:.3g}",
                (thr, _safe_for_log_y(max(achieved_train, achieved_hold), y_floor)),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )

            rows.append(
                SplitResult(
                    mode="CV",
                    metric=metric,
                    heldout_fold=str(heldout),
                    train_folds=",".join(map(str, train_folds)),
                    target_fpr=t,
                    threshold=thr,
                    threshold_source=src,
                    train_fpr=achieved_train,
                    heldout_fpr=achieved_hold,
                    n_train_total=n_train_total,
                    n_train_scored=len(train_scores),
                    n_hold_total=n_hold_total,
                    n_hold_scored=len(hold_scores),
                    per_fold_fprs="",
                )
            )

        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
        ax.legend(loc="best", frameon=True)
        ax.set_ylim(bottom=y_floor, top=1.0)

        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    return rows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Single-program CV threshold-vs-FPR and GLOBAL operating point estimation on negative-only data."
    )
    ap.add_argument("--folds", required=True, help="TSV: seq_id<TAB>fold (header allowed).")
    ap.add_argument("--scores", required=True, help="TSV: seq_id<TAB>value (header allowed; may omit seq_ids).")
    ap.add_argument(
        "--metric",
        required=True,
        choices=["score", "evalue"],
        help="Use 'score' when higher is more positive; 'evalue' when lower is more positive."
    )
    ap.add_argument(
        "--targets",
        default="1e-5,1e-4",
        help="Comma-separated target FPRs (default: 1e-5,1e-4)."
    )
    ap.add_argument("--out-pdf", required=True, help="Output multi-page PDF path.")
    ap.add_argument("--out-tsv", required=True, help="Output summary TSV path (includes GLOBAL rows).")
    ap.add_argument("--title", default=None, help="Optional plot title prefix.")
    ap.add_argument(
        "--evalue-floor",
        type=float,
        default=1e-300,
        help="For metric=evalue, floor values <=0 to this number for log-x (default: 1e-300)."
    )

    args = ap.parse_args(argv)

    seq_to_fold, fold_sizes = read_folds_tsv(args.folds)
    fold_ids = sorted(fold_sizes.keys())

    targets: List[float] = []
    for x in args.targets.split(","):
        x = x.strip()
        if x:
            targets.append(float(x))
    if not targets:
        raise ValueError("No valid --targets provided.")

    scores_by_seq = read_scores_tsv(args.scores, args.metric, args.evalue_floor)

    # Map scores to folds; missing seq_ids in scores are handled via denominators from fold_sizes.
    fold_scores: Dict[int, List[float]] = {f: [] for f in fold_ids}
    missing_fold = 0
    for seq_id, v in scores_by_seq.items():
        f = seq_to_fold.get(seq_id)
        if f is None:
            missing_fold += 1
            continue
        if f in fold_scores:
            fold_scores[f].append(v)

    if missing_fold > 0:
        print(f"[WARN] {missing_fold} scored seq_ids not found in folds file; ignored.")

    # For log-y, cannot show 0; pick a floor well below your smallest target.
    min_target = min(targets) if targets else 1e-6
    y_floor = max(min_target / 100.0, 1e-12)

    os.makedirs(os.path.dirname(args.out_pdf) or ".", exist_ok=True)

    rows: List[SplitResult] = []

    with PdfPages(args.out_pdf) as pdf:
        # Page 1: GLOBAL
        fig = plt.figure(figsize=(8.5, 6.0))
        ax = fig.add_subplot(111)
        global_rows = plot_global_page(
            ax=ax,
            metric=args.metric,
            fold_ids=fold_ids,
            fold_sizes=fold_sizes,
            fold_scores=fold_scores,
            targets=targets,
            evalue_floor=args.evalue_floor,
            title=args.title,
            y_floor=y_floor,
        )
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
        rows.extend(global_rows)

        # Remaining pages: CV per held-out fold
        rows.extend(
            plot_cv_pages(
                pdf=pdf,
                metric=args.metric,
                fold_ids=fold_ids,
                fold_sizes=fold_sizes,
                fold_scores=fold_scores,
                targets=targets,
                evalue_floor=args.evalue_floor,
                title=args.title,
                y_floor=y_floor,
            )
        )

    write_results_tsv(args.out_tsv, rows)

    print(f"[INFO] Wrote: {args.out_pdf}")
    print(f"[INFO] Wrote: {args.out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

