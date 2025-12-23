#!/usr/bin/env python3
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training curves (loss, PR-AUC, IoU) from a metrics JSON file.\n\n"
            "Example:\n"
            "  python plot_training_curves.py "
            "--metrics 600m_train_metrics.json "
            "--out training_curves_main.pdf"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--metrics",
        required=True,
        help="Path to metrics JSON (e.g. 600m_train_metrics.json).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help=(
            "Output figure path (format inferred from extension, e.g. "
            "training_curves_main.pdf / .png / .svg)."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title (default: no title or a simple default).",
    )

    return parser.parse_args()


def load_metrics(path: str):
    if not os.path.exists(path):
        print(f"ERROR: metrics file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path, "r") as f:
        data = json.load(f)

    if "log" not in data:
        print("ERROR: JSON file must contain a top-level 'log' key.", file=sys.stderr)
        sys.exit(1)

    log = data["log"]
    if not isinstance(log, list) or len(log) == 0:
        print("ERROR: 'log' must be a non-empty list.", file=sys.stderr)
        sys.exit(1)

    # Extract series
    epochs = [entry["epoch"] for entry in log]
    train_loss = [entry["train_loss"] for entry in log]
    val_pr_rest = [entry["val_pr_auc_p_vs_rest"] for entry in log]
    val_pr_pn = [entry["val_pr_auc_p_vs_n"] for entry in log]
    val_iou = [entry["val_mean_iou"] for entry in log]
    val_iou05 = [entry["val_iou_at_0_5"] for entry in log]

    # Best epochs (if not present, we compute them)
    def argmax(values):
        return max(range(len(values)), key=lambda i: values[i])

    if "best_by_pr" in data and "epoch" in data["best_by_pr"]:
        best_pr_epoch = data["best_by_pr"]["epoch"]
    else:
        best_pr_epoch = epochs[argmax(val_pr_rest)]

    if "best_by_iou" in data and "epoch" in data["best_by_iou"]:
        best_iou_epoch = data["best_by_iou"]["epoch"]
    else:
        best_iou_epoch = epochs[argmax(val_iou)]

    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_pr_rest": val_pr_rest,
        "val_pr_pn": val_pr_pn,
        "val_iou": val_iou,
        "val_iou05": val_iou05,
        "best_pr_epoch": best_pr_epoch,
        "best_iou_epoch": best_iou_epoch,
    }


def plot_curves(metrics: dict, out_path: str, title: str | None = None) -> None:
    epochs = metrics["epochs"]
    train_loss = metrics["train_loss"]
    val_pr_rest = metrics["val_pr_rest"]
    val_pr_pn = metrics["val_pr_pn"]
    val_iou = metrics["val_iou"]
    val_iou05 = metrics["val_iou05"]
    best_pr_epoch = metrics["best_pr_epoch"]
    best_iou_epoch = metrics["best_iou_epoch"]

    # Basic sanity: warn if extension is odd, but let matplotlib handle it.
    _, ext = os.path.splitext(out_path)
    if ext == "":
        print(
            f"WARNING: output path '{out_path}' has no extension; "
            "matplotlib may default to PNG.",
            file=sys.stderr,
        )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8, 8),
        sharex=True,
        constrained_layout=True,
    )

    # Panel A: train loss
    ax0 = axes[0]
    ax0.plot(epochs, train_loss, marker=".", linewidth=1)
    ax0.axvline(best_pr_epoch, linestyle=":", linewidth=1)
    ax0.set_ylabel("Train loss")
    if title:
        ax0.set_title(title)
    else:
        ax0.set_title("Training dynamics")
    ax0.grid(True, alpha=0.3)

    # Panel B: PR-AUC
    ax1 = axes[1]
    ax1.plot(
        epochs,
        val_pr_rest,
        marker=".",
        linewidth=1,
        label="val PR-AUC Positives vs rest",
    )
    ax1.plot(
        epochs,
        val_pr_pn,
        linestyle="-",
        marker=".",
        linewidth=1,
        label="val PR-AUC Positives vs Negatives",
    )

    ax1.axvline(best_pr_epoch, linestyle=":", linewidth=1)
    ymax_pr = max(max(val_pr_rest), max(val_pr_pn))
    ax1.text(
        best_pr_epoch + 0.5,
        ymax_pr,
        f"best PR epoch {best_pr_epoch}",
        va="top",
        fontsize=8,
    )

    ax1.set_ylabel("PR-AUC")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower right", fontsize=8)

    # Panel C: IoU (main metric: val_mean_iou)
    ax2 = axes[2]
    ax2.plot(
        epochs,
        val_iou,
        marker=".",
        linewidth=1,
        label="val mean IoU",
    )
    ax2.plot(
        epochs,
        val_iou05,
        linestyle="-",
        marker=".",
        linewidth=1,
        label="val IoU@0.5",
    )

    ax2.axvline(best_pr_epoch, linestyle=":", linewidth=1)
    ymax_iou = max(max(val_iou), max(val_iou05))

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8)

    # Save figure; format is inferred from out_path extension.
    print(f"Saving figure to: {out_path}", file=sys.stderr)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Also log the best metrics to stdout for convenience
    print(f"Best PR epoch: {best_pr_epoch}")
    print(f"Best IoU epoch: {best_iou_epoch}")


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics)
    plot_curves(metrics, args.out, title=args.title)


if __name__ == "__main__":
    main()
