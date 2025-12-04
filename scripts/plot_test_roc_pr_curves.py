#!/usr/bin/env python
"""
Plot PR and ROC curves from PalmSite eval JSON files.

Each JSON file is expected to have the structure produced by eval.py:

{
  "checkpoint": "...",
  "split": "test",
  "n_samples": N,
  "metrics": {
    "pr_auc_p_vs_rest": ...,
    "pr_auc_p_vs_n": ...,
    "roc_auc_p_vs_rest": ...,
    "roc_auc_p_vs_n": ...,
    ...
  },
    "curves": {
    "pr_curve_p_vs_rest": {
      "recall": [...],
      "precision": [...]
    },
    "roc_curve_p_vs_rest": {
      "fpr": [...],
      "tpr": [...]
    },
    "pr_curve_p_vs_n": {
      "recall": [...],
      "precision": [...]
    },
    "roc_curve_p_vs_n": {
      "fpr": [...],
      "tpr": [...]
    }
  }
}

Usage examples:

# Single model
python plot_curves.py \
  --json runs/exp/palmsite_eval_test.json \
  --label PalmSite \
  --out-prefix figs/palmsite_test

# Multiple models on same axes
python plot_curves.py \
  --json runs/exp/palmsite_eval_test.json runs/exp/neordrp_eval_test.json \
  --label PalmSite NeoRdRP \
  --out-prefix figs/compare_test
"""

import argparse
import json
import os
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def load_eval_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_pr_curves(
    evals: List[Dict[str, Any]],
    labels: List[str],
    out_path_rest: str,
    out_path_n: str,
) -> None:
    """
    Plot PR curves for:
      - P vs Rest
      - P vs N

    Saves two PNGs: out_path_rest, out_path_n
    """
    # ----- P vs Rest -----
    plt.figure(figsize=(4, 4))
    for ev, lab in zip(evals, labels):
        curves = ev.get("curves", {})
        metrics = ev.get("metrics", {})
        pr = curves.get("pr_curve_p_vs_rest", {})
        recall = pr.get("recall", [])
        precision = pr.get("precision", [])
        if not recall or not precision:
            continue
        auc_val = metrics.get("pr_auc_p_vs_rest", None)
        if auc_val is not None:
            plt.plot(
                recall,
                precision,
                label=f"{lab} (AP={auc_val:.3f})",
            )
        else:
            plt.plot(recall, precision, label=lab)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve (P vs Rest)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path_rest) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path_rest, dpi=300)
    plt.close()

    # ----- P vs N -----
    plt.figure(figsize=(4, 4))
    for ev, lab in zip(evals, labels):
        curves = ev.get("curves", {})
        metrics = ev.get("metrics", {})
        pr = curves.get("pr_curve_p_vs_n", {})
        recall = pr.get("recall", [])
        precision = pr.get("precision", [])
        if not recall or not precision:
            continue
        auc_val = metrics.get("pr_auc_p_vs_n", None)
        if auc_val is not None:
            plt.plot(
                recall,
                precision,
                label=f"{lab} (AP={auc_val:.3f})",
            )
        else:
            plt.plot(recall, precision, label=lab)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR curve (P vs N)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path_n) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path_n, dpi=300)
    plt.close()


def plot_roc_curves(
    evals: List[Dict[str, Any]],
    labels: List[str],
    out_path_rest: str,
    out_path_n: str,
) -> None:
    """
    Plot ROC curves for:
      - P vs Rest
      - P vs N

    Saves two PNGs: out_path_rest, out_path_n
    """
    # ----- P vs Rest -----
    plt.figure(figsize=(4, 4))
    for ev, lab in zip(evals, labels):
        curves = ev.get("curves", {})
        metrics = ev.get("metrics", {})
        roc = curves.get("roc_curve_p_vs_rest", {})
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        if not fpr or not tpr:
            continue
        auc_val = metrics.get("roc_auc_p_vs_rest", None)
        if auc_val is not None:
            plt.plot(
                fpr,
                tpr,
                label=f"{lab} (AUC={auc_val:.3f})",
            )
        else:
            plt.plot(fpr, tpr, label=lab)

    plt.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (P vs Rest)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path_rest) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path_rest, dpi=300)
    plt.close()

    # ----- P vs N -----
    plt.figure(figsize=(4, 4))
    for ev, lab in zip(evals, labels):
        curves = ev.get("curves", {})
        metrics = ev.get("metrics", {})
        roc = curves.get("roc_curve_p_vs_n", {})
        fpr = roc.get("fpr", [])
        tpr = roc.get("tpr", [])
        if not fpr or not tpr:
            continue
        auc_val = metrics.get("roc_auc_p_vs_n", None)
        if auc_val is not None:
            plt.plot(
                fpr,
                tpr,
                label=f"{lab} (AUC={auc_val:.3f})",
            )
        else:
            plt.plot(fpr, tpr, label=lab)

    plt.plot([0, 1], [0, 1], "k--", linewidth=0.7)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (P vs N)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=8)
    os.makedirs(os.path.dirname(out_path_n) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path_n, dpi=300)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot PR and ROC curves from eval JSON files.")
    p.add_argument(
        "--json",
        nargs="+",
        required=True,
        help="One or more eval JSON files (from eval.py).",
    )
    p.add_argument(
        "--label",
        nargs="+",
        required=False,
        help="Labels for each JSON file (default: file basename).",
    )
    p.add_argument(
        "--out-prefix",
        required=True,
        help=(
            "Output prefix for figures, e.g. figs/palmsite_test.\n"
            "Will produce *_pr_p_vs_rest.png, *_pr_p_vs_n.png, *_roc_p_vs_rest.png, *_roc_p_vs_n.png"
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    json_paths = args.json
    labels = args.label or [os.path.splitext(os.path.basename(p))[0] for p in json_paths]

    if len(labels) != len(json_paths):
        raise ValueError("Number of --label entries must match number of --json files (or omit --label).")

    evals = [load_eval_json(p) for p in json_paths]

    # Prepare output filenames
    base = args.out_prefix
    pr_rest_png = f"{base}_pr_p_vs_rest.png"
    pr_n_png = f"{base}_pr_p_vs_n.png"
    roc_rest_png = f"{base}_roc_p_vs_rest.png"
    roc_n_png = f"{base}_roc_p_vs_n.png"

    print("[plot] Plotting PR curves...")
    plot_pr_curves(evals, labels, pr_rest_png, pr_n_png)

    print("[plot] Plotting ROC curves...")
    plot_roc_curves(evals, labels, roc_rest_png, roc_n_png)

    print("[plot] Done.")
    print(f"  PR P vs Rest : {pr_rest_png}")
    print(f"  PR P vs N    : {pr_n_png}")
    print(f"  ROC P vs Rest: {roc_rest_png}")
    print(f"  ROC P vs N   : {roc_n_png}")


if __name__ == "__main__":
    main()
