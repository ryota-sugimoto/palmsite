#!/usr/bin/env python3
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot training curves (train/val loss, PR-AUC, IoU) from a metrics JSON file.\n\n"
            "Example:\n"
            "  python plot_training_curve_with_loss.py "
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


def argmax(values):
    return max(range(len(values)), key=lambda i: values[i])


def argmin(values):
    return min(range(len(values)), key=lambda i: values[i])


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

    required_keys = [
        "epoch",
        "train_loss",
        "val_loss_total",
        "val_pr_auc_p_vs_rest",
        "val_pr_auc_p_vs_n",
        "val_mean_iou",
        "val_iou_at_0_5",
    ]
    missing = [key for key in required_keys if key not in log[0]]
    if missing:
        print(
            "ERROR: metrics log entries are missing required keys: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract series
    epochs = [entry["epoch"] for entry in log]
    train_loss = [entry["train_loss"] for entry in log]
    val_loss_total = [entry["val_loss_total"] for entry in log]
    val_pr_rest = [entry["val_pr_auc_p_vs_rest"] for entry in log]
    val_pr_pn = [entry["val_pr_auc_p_vs_n"] for entry in log]
    val_iou = [entry["val_mean_iou"] for entry in log]
    val_iou05 = [entry["val_iou_at_0_5"] for entry in log]

    best_train_loss_idx = argmin(train_loss)
    best_val_loss_idx = argmin(val_loss_total)

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
        "val_loss_total": val_loss_total,
        "val_pr_rest": val_pr_rest,
        "val_pr_pn": val_pr_pn,
        "val_iou": val_iou,
        "val_iou05": val_iou05,
        "best_train_loss_epoch": epochs[best_train_loss_idx],
        "best_train_loss": train_loss[best_train_loss_idx],
        "best_val_loss_epoch": epochs[best_val_loss_idx],
        "best_val_loss": val_loss_total[best_val_loss_idx],
        "best_pr_epoch": best_pr_epoch,
        "best_iou_epoch": best_iou_epoch,
    }


def _epoch_to_value(epochs, values, epoch):
    idx = epochs.index(epoch)
    return values[idx]


def _annotate_epoch(ax, epoch, value, label, epochs, value_span, dy_scale=0.04):
    x_max = max(epochs)
    x_text = epoch + 1
    ha = "left"
    if epoch >= x_max - 8:
        x_text = epoch - 8
        ha = "right"

    ax.text(
        x_text,
        value + dy_scale * value_span,
        label,
        fontsize=8,
        ha=ha,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
    )


def plot_curves(metrics: dict, out_path: str, title: str | None = None) -> None:
    epochs = metrics["epochs"]
    train_loss = metrics["train_loss"]
    val_loss_total = metrics["val_loss_total"]
    val_pr_rest = metrics["val_pr_rest"]
    val_pr_pn = metrics["val_pr_pn"]
    val_iou = metrics["val_iou"]
    val_iou05 = metrics["val_iou05"]
    best_train_loss_epoch = metrics["best_train_loss_epoch"]
    best_train_loss = metrics["best_train_loss"]
    best_val_loss_epoch = metrics["best_val_loss_epoch"]
    best_val_loss = metrics["best_val_loss"]
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
        figsize=(9, 8.5),
        sharex=True,
        constrained_layout=True,
    )

    # Panel A: train/validation loss
    ax0 = axes[0]
    ax0.plot(epochs, train_loss, marker=".", linewidth=1, label="train loss")
    ax0.plot(epochs, val_loss_total, marker=".", linewidth=1, label="val loss total")

    # Mark the "best" checkpoints based on loss minima.
    for ax in axes:
        ax.axvline(best_train_loss_epoch, color="black", linestyle="--", linewidth=1, alpha=0.35)
        ax.axvline(best_val_loss_epoch, color="black", linestyle=":", linewidth=1, alpha=0.6)

    ax0.scatter([best_train_loss_epoch], [best_train_loss], s=28, zorder=3)
    ax0.scatter([best_val_loss_epoch], [best_val_loss], s=28, zorder=3)

    loss_min = min(min(train_loss), min(val_loss_total))
    loss_max = max(max(train_loss), max(val_loss_total))
    loss_span = max(loss_max - loss_min, 1e-9)

    _annotate_epoch(
        ax0,
        best_train_loss_epoch,
        best_train_loss,
        f"best train loss\n@ epoch {best_train_loss_epoch} = {best_train_loss:.4f}",
        epochs,
        loss_span,
        dy_scale=0.06,
    )
    _annotate_epoch(
        ax0,
        best_val_loss_epoch,
        best_val_loss,
        f"best val loss\n@ epoch {best_val_loss_epoch} = {best_val_loss:.4f}",
        epochs,
        loss_span,
        dy_scale=0.02,
    )

    ax0.set_ylabel("Loss")
    if title:
        ax0.set_title(title)
    else:
        ax0.set_title("Training dynamics")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="upper right", fontsize=8)

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

    best_pr_value = _epoch_to_value(epochs, val_pr_rest, best_pr_epoch)
    ax1.scatter([best_pr_epoch], [best_pr_value], s=28, zorder=3)
    ax1.text(
        best_pr_epoch + 0.5,
        best_pr_value,
        f"best PR @ epoch {best_pr_epoch}",
        va="bottom",
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

    best_iou_value = _epoch_to_value(epochs, val_iou, best_iou_epoch)
    ax2.scatter([best_iou_epoch], [best_iou_value], s=28, zorder=3)
    ax2.text(
        best_iou_epoch + 0.5,
        best_iou_value,
        f"best mean IoU @ epoch {best_iou_epoch}",
        va="bottom",
        fontsize=8,
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right", fontsize=8)

    # Save figure; format is inferred from out_path extension.
    print(f"Saving figure to: {out_path}", file=sys.stderr)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Also log the best metrics to stdout for convenience
    print(
        f"Best train loss epoch: {best_train_loss_epoch} "
        f"(train_loss={best_train_loss:.6f})"
    )
    print(
        f"Best val loss epoch: {best_val_loss_epoch} "
        f"(val_loss_total={best_val_loss:.6f})"
    )
    print(f"Best PR epoch: {best_pr_epoch}")
    print(f"Best IoU epoch: {best_iou_epoch}")


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.metrics)
    plot_curves(metrics, args.out, title=args.title)


if __name__ == "__main__":
    main()

