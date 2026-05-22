#!/usr/bin/env python3
"""
Evaluate a specified PalmSite checkpoint on a train/val/test split.

This script is intended for post-hoc evaluation of an arbitrary model .pt file,
instead of always evaluating the selection-best checkpoint produced by train.py.

Outputs:
  - JSON containing metrics and PR/ROC curve arrays
  - Optional sequence ID list
  - Optional curve TSV files
  - PDF files for PR/ROC curves, IoU violin plot, and a one-page metrics summary

Example:

python scripts/evaluate_checkpoint.py \
  --ckpt runs/exp/model_epoch_012.pt \
  --embeddings embeddings.h5 \
  --labels labels.h5 \
  --split test \
  --batch-size 32 \
  --num-workers 2 \
  --min-precision 0.90 0.95 0.99 \
  --out-dir runs/exp/eval_epoch_012
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from train import (
    TrainConfig,
    RdRPModel,
    build_datasets,
    collate_batch,
    eval_split_full,
    collect_eval_arrays,
    save_curves_tsv,
)


def _safe_stem(path: str) -> str:
    stem = Path(path).name
    for suffix in (".pt", ".pth"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)


def _filter_cfg_dict(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only fields accepted by the current TrainConfig dataclass."""
    allowed = set(TrainConfig.__dataclass_fields__.keys())
    return {k: v for k, v in raw.items() if k in allowed}


def _load_cfg_from_checkpoint(ckpt: Mapping[str, Any]) -> TrainConfig:
    if "cfg" not in ckpt or not isinstance(ckpt["cfg"], Mapping):
        raise RuntimeError(
            "Checkpoint does not contain a usable 'cfg' dictionary. "
            "Use a checkpoint saved by scripts/train.py."
        )
    return TrainConfig(**_filter_cfg_dict(ckpt["cfg"]))


def _apply_sidecar_inference_defaults(cfg: TrainConfig, ckpt_path: str) -> None:
    """Use inference_defaults.json when present to avoid train/inference drift."""
    sidecar_path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "inference_defaults.json")
    if not os.path.isfile(sidecar_path):
        return
    try:
        with open(sidecar_path, "r", encoding="utf-8", newline="") as f:
            sidecar = json.load(f)
    except Exception:
        return

    for key in (
        "dropout",
        "tau",
        "alpha_cap",
        "wmin_base",
        "wmin_floor",
        "lenfeat_scale",
        "pos_channel",
        "k_sigma",
        "coarse_stride",
        "tau_len_gamma",
        "tau_len_ref",
    ):
        if key in sidecar and hasattr(cfg, key):
            setattr(cfg, key, sidecar[key])


def _select_dataset(split: str, train_ds, val_ds, test_ds):
    split_l = split.lower()
    if split_l == "train":
        return train_ds
    if split_l in {"val", "valid", "validation"}:
        return val_ds
    if split_l == "test":
        return test_ds
    raise ValueError(f"Unknown split {split!r}; choose train, val, or test.")


def _split_id_line(line: str) -> List[str]:
    """Split one ID-list line as TSV, CSV, or whitespace-delimited text."""
    line = line.strip()
    if "\t" in line:
        return next(csv.reader([line], delimiter="\t"))
    if "," in line:
        return next(csv.reader([line], delimiter=","))
    return line.split()


def _read_id_list(path: str) -> List[str]:
    """Read chunk IDs from a plain list, TSV, CSV, or whitespace table.

    The first non-empty column is used. Blank lines and comment lines beginning
    with '#' are ignored. A common header name such as chunk_id/id/seq_id is
    skipped automatically when it appears as the first usable row.
    """
    ids: List[str] = []
    header_names = {"chunk_id", "id", "seq_id", "sequence_id", "sample_id"}
    with open(path, "r", encoding="utf-8", newline="") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = [x.strip() for x in _split_id_line(line)]
            fields = [x for x in fields if x != ""]
            if not fields:
                continue
            cid = fields[0]
            if not ids and cid.strip().lower() in header_names:
                continue
            ids.append(cid)
    if not ids:
        raise ValueError(f"No IDs were found in {path!r}.")
    return ids


def _dataset_from_id_list(template_ds, id_list: List[str]):
    """Return an eval dataset with indices ordered exactly as id_list."""
    id_to_index = {str(cid): int(i) for i, cid in enumerate(template_ds.table.ids)}
    seen = set()
    dup_ids: List[str] = []
    indices: List[int] = []
    missing: List[str] = []

    for cid in id_list:
        if cid in seen:
            dup_ids.append(cid)
            continue
        seen.add(cid)
        if cid not in id_to_index:
            missing.append(cid)
            continue
        indices.append(id_to_index[cid])

    if missing:
        preview = ", ".join(missing[:10])
        extra = "" if len(missing) <= 10 else f" ... (+{len(missing) - 10} more)"
        raise ValueError(f"{len(missing)} IDs from the ID file were not found in labels.h5: {preview}{extra}")
    if not indices:
        raise ValueError("The ID file did not match any labels.h5 rows after duplicate removal.")

    if dup_ids:
        preview = ", ".join(dup_ids[:10])
        extra = "" if len(dup_ids) <= 10 else f" ... (+{len(dup_ids) - 10} more)"
        print(f"[evaluate_checkpoint] Ignored {len(dup_ids)} duplicate IDs from ID file: {preview}{extra}")

    return template_ds.__class__(
        template_ds.emb_path,
        template_ds.labels_path,
        indices=np.asarray(indices, dtype=np.int64),
        is_train=False,
        pos_channel=template_ds.pos_channel,
        dtype=("float32" if template_ds.dtype == np.float32 else "float16"),
    )


def _state_dict_without_compile_prefix(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Handle checkpoints saved from torch.compile-wrapped models."""
    if not state_dict:
        return dict(state_dict)
    keys = list(state_dict.keys())
    if all(k.startswith("_orig_mod.") for k in keys):
        return {k[len("_orig_mod.") :]: v for k, v in state_dict.items()}
    return dict(state_dict)


def _build_model(cfg: TrainConfig, d_model: int, device: torch.device) -> RdRPModel:
    model = RdRPModel(
        d_in=int(d_model) + 1,
        tau=float(cfg.tau),
        alpha_cap=float(cfg.alpha_cap),
        p_drop=float(cfg.dropout),
    ).to(device)

    model.wmin_base = float(cfg.wmin_base)
    model.wmin_floor = float(cfg.wmin_floor)
    model.seq_pool = "gauss"
    model.lenfeat_scale = float(cfg.lenfeat_scale)
    model.coarse_stride = int(cfg.coarse_stride)
    model.tau_len_gamma = float(cfg.tau_len_gamma)
    model.tau_len_ref = float(cfg.tau_len_ref)
    model.k_sigma = float(cfg.k_sigma)
    return model


def _write_json(path: str, obj: Mapping[str, Any]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _write_ids(path: str, seq_ids: Iterable[str]) -> int:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for cid in seq_ids:
            f.write(f"{cid}\n")
            n += 1
    return n


def _as_float_array(values: Any) -> np.ndarray:
    return np.asarray(values if values is not None else [], dtype=np.float64)


def _plot_curve_pdf(
    out_pdf: str,
    x: np.ndarray,
    y: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    baseline_y: Optional[float] = None,
    log_x: bool = False,
) -> None:
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.0))

    if x.size and y.size:
        ax.plot(x, y, linewidth=1.8)
    else:
        ax.text(0.5, 0.5, "No curve data", ha="center", va="center", transform=ax.transAxes)

    if baseline_y is not None and math.isfinite(float(baseline_y)):
        ax.axhline(float(baseline_y), linestyle="--", linewidth=1.0)
        ax.text(
            0.99,
            float(baseline_y),
            f" baseline={baseline_y:.4g}",
            ha="right",
            va="bottom",
            fontsize=8,
            transform=ax.get_yaxis_transform(),
        )

    if log_x:
        positive = x[x > 0]
        if positive.size:
            ax.set_xscale("log")
            ax.set_xlim(max(float(positive.min()), 1e-8), 1.0)
    else:
        ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)



def _plot_iou_violin_pdf(out_pdf: str, iou_values: np.ndarray, title: str) -> None:
    """Write a one-panel IoU distribution violin plot as PDF."""
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    vals = np.asarray(iou_values if iou_values is not None else [], dtype=np.float64)
    vals = vals[np.isfinite(vals)]

    fig, ax = plt.subplots(figsize=(4.8, 5.2))
    if vals.size:
        parts = ax.violinplot(
            vals,
            positions=[1],
            widths=0.65,
            showmeans=False,
            showmedians=True,
            showextrema=True,
        )
        for body in parts.get("bodies", []):
            body.set_alpha(0.45)
        jitter = np.linspace(-0.08, 0.08, vals.size) if vals.size <= 200 else None
        if jitter is not None:
            ax.scatter(np.ones(vals.size) + jitter, vals, s=8, alpha=0.45)
        ax.axhline(float(vals.mean()), linestyle="--", linewidth=1.0)
        ax.text(
            1.42,
            float(vals.mean()),
            f"mean={vals.mean():.3f}",
            ha="left",
            va="center",
            fontsize=8,
        )
        ax.text(
            0.04,
            0.04,
            f"n={vals.size}\nmedian={np.median(vals):.3f}\nIoU≥0.5={(vals >= 0.5).mean():.3f}",
            ha="left",
            va="bottom",
            fontsize=9,
            transform=ax.transAxes,
        )
    else:
        ax.text(0.5, 0.5, "No span-labeled examples", ha="center", va="center", transform=ax.transAxes)

    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks([1])
    ax.set_xticklabels(["span-labeled"])
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)


def _write_iou_values_tsv(path: str, iou_values: np.ndarray) -> int:
    """Write per-span IoU values used for the violin plot."""
    vals = np.asarray(iou_values if iou_values is not None else [], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("index\tiou\n")
        for i, val in enumerate(vals):
            f.write(f"{i}\t{val:.8g}\n")
    return int(vals.size)

def _metric_lines(metrics: Mapping[str, Any]) -> List[Tuple[str, str]]:
    preferred_keys = [
        "n_pos",
        "n_unlabeled",
        "n_neg",
        "pos_rate_p_vs_rest",
        "pos_rate_p_vs_n",
        "pr_auc_p_vs_rest",
        "roc_auc_p_vs_rest",
        "pr_auc_p_vs_n",
        "roc_auc_p_vs_n",
        "mean_iou",
        "iou_at_0_5",
    ]
    recall_keys = sorted(k for k in metrics if k.startswith("recall_at_precision_"))
    keys = preferred_keys + recall_keys

    out: List[Tuple[str, str]] = []
    seen = set()
    for key in keys:
        if key in seen or key not in metrics:
            continue
        seen.add(key)
        val = metrics[key]
        if isinstance(val, float):
            if math.isnan(val):
                sval = "nan"
            else:
                sval = f"{val:.6g}"
        else:
            sval = str(val)
        out.append((key, sval))
    return out


def _write_summary_pdf(out_pdf: str, result: Mapping[str, Any]) -> None:
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    metrics = result.get("metrics", {})
    lines = _metric_lines(metrics if isinstance(metrics, Mapping) else {})

    with PdfPages(out_pdf) as pdf:
        fig = plt.figure(figsize=(8.5, 11.0))
        fig.text(0.08, 0.95, "PalmSite checkpoint evaluation", fontsize=16, weight="bold")
        fig.text(0.08, 0.915, f"Checkpoint: {result.get('checkpoint', '')}", fontsize=8)
        fig.text(0.08, 0.895, f"Split: {result.get('split', '')}", fontsize=10)
        fig.text(0.08, 0.875, f"n_samples: {result.get('n_samples', '')}", fontsize=10)

        y = 0.835
        fig.text(0.08, y, "Metric", fontsize=11, weight="bold")
        fig.text(0.58, y, "Value", fontsize=11, weight="bold")
        y -= 0.025
        for key, value in lines:
            if y < 0.08:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11.0))
                y = 0.94
                fig.text(0.08, y, "Metric", fontsize=11, weight="bold")
                fig.text(0.58, y, "Value", fontsize=11, weight="bold")
                y -= 0.025
            fig.text(0.08, y, key, fontsize=9)
            fig.text(0.58, y, value, fontsize=9)
            y -= 0.022

        pdf.savefig(fig)
        plt.close(fig)


def write_pdf_outputs(result: Mapping[str, Any], out_pdf_prefix: str) -> Dict[str, str]:
    curves = result.get("curves", {})
    metrics = result.get("metrics", {})
    split = str(result.get("split", "split"))
    prefix = out_pdf_prefix
    os.makedirs(os.path.dirname(prefix) or ".", exist_ok=True)

    if not isinstance(curves, Mapping):
        curves = {}
    if not isinstance(metrics, Mapping):
        metrics = {}

    outputs: Dict[str, str] = {}

    pr_rest = curves.get("pr_curve_p_vs_rest", {}) if isinstance(curves, Mapping) else {}
    roc_rest = curves.get("roc_curve_p_vs_rest", {}) if isinstance(curves, Mapping) else {}
    pr_n = curves.get("pr_curve_p_vs_n", {}) if isinstance(curves, Mapping) else {}
    roc_n = curves.get("roc_curve_p_vs_n", {}) if isinstance(curves, Mapping) else {}

    outputs["summary_pdf"] = f"{prefix}_{split}_summary.pdf"
    _write_summary_pdf(outputs["summary_pdf"], result)

    outputs["pr_p_vs_rest_pdf"] = f"{prefix}_{split}_pr_p_vs_rest.pdf"
    _plot_curve_pdf(
        outputs["pr_p_vs_rest_pdf"],
        _as_float_array(pr_rest.get("recall", []) if isinstance(pr_rest, Mapping) else []),
        _as_float_array(pr_rest.get("precision", []) if isinstance(pr_rest, Mapping) else []),
        xlabel="Recall",
        ylabel="Precision",
        title="PR curve: P vs rest",
        baseline_y=metrics.get("pos_rate_p_vs_rest"),
    )

    outputs["roc_p_vs_rest_pdf"] = f"{prefix}_{split}_roc_p_vs_rest.pdf"
    _plot_curve_pdf(
        outputs["roc_p_vs_rest_pdf"],
        _as_float_array(roc_rest.get("fpr", []) if isinstance(roc_rest, Mapping) else []),
        _as_float_array(roc_rest.get("tpr", []) if isinstance(roc_rest, Mapping) else []),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curve: P vs rest",
        log_x=True,
    )

    outputs["pr_p_vs_n_pdf"] = f"{prefix}_{split}_pr_p_vs_n.pdf"
    _plot_curve_pdf(
        outputs["pr_p_vs_n_pdf"],
        _as_float_array(pr_n.get("recall", []) if isinstance(pr_n, Mapping) else []),
        _as_float_array(pr_n.get("precision", []) if isinstance(pr_n, Mapping) else []),
        xlabel="Recall",
        ylabel="Precision",
        title="PR curve: P vs N",
        baseline_y=metrics.get("pos_rate_p_vs_n"),
    )

    outputs["roc_p_vs_n_pdf"] = f"{prefix}_{split}_roc_p_vs_n.pdf"
    _plot_curve_pdf(
        outputs["roc_p_vs_n_pdf"],
        _as_float_array(roc_n.get("fpr", []) if isinstance(roc_n, Mapping) else []),
        _as_float_array(roc_n.get("tpr", []) if isinstance(roc_n, Mapping) else []),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC curve: P vs N",
        log_x=True,
    )

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a specified PalmSite checkpoint and write JSON/PDF outputs."
    )
    parser.add_argument("--ckpt", required=True, help="Path to model checkpoint .pt file.")
    parser.add_argument("--embeddings", default=None, help="Path to embeddings.h5. Overrides checkpoint cfg.")
    parser.add_argument("--labels", default=None, help="Path to labels.h5. Overrides checkpoint cfg.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Split to evaluate when --id-list/--test-ids is not provided.")
    parser.add_argument(
        "--id-list",
        "--test-ids",
        dest="id_list",
        default=None,
        help=(
            "Optional file containing chunk IDs to evaluate. Accepts one ID per line, "
            "or the first column of a TSV/CSV/whitespace table. When provided, "
            "these IDs are evaluated exactly, independent of the regenerated split."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--dataset-dtype", choices=["float16", "float32"], default=None, help="Override dataset dtype.")
    parser.add_argument(
        "--min-precision",
        type=float,
        nargs="*",
        default=[0.90, 0.95],
        help="Precision thresholds for recall@precision. Example: --min-precision 0.90 0.95 0.99",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: directory containing --ckpt.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="Output JSON path. Default: <out-dir>/<ckpt_stem>_<split>_eval.json",
    )
    parser.add_argument(
        "--out-ids",
        default=None,
        help="Optional output path for evaluated chunk IDs. Default: <out-dir>/<ckpt_stem>_<split>_ids.txt",
    )
    parser.add_argument(
        "--no-out-ids",
        action="store_true",
        help="Do not write the evaluated chunk ID list.",
    )
    parser.add_argument(
        "--curves-prefix",
        default=None,
        help="Optional prefix for PR/ROC curve TSV files. Default: disabled.",
    )
    parser.add_argument(
        "--out-pdf-prefix",
        default=None,
        help="PDF output prefix. Default: <out-dir>/<ckpt_stem>. Produces summary, PR, and ROC PDFs.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable PDF outputs.",
    )
    parser.add_argument(
        "--no-iou-violin",
        action="store_true",
        help="Disable the IoU violin PDF output. This only affects the extra IoU distribution plot; PR/ROC PDFs are controlled by --no-pdf.",
    )
    parser.add_argument(
        "--iou-values-tsv",
        default=None,
        help="Optional path for per-span IoU values used in the violin plot. Default: <out-dir>/<ckpt_stem>_<split>_iou_values.tsv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ckpt_path = os.path.abspath(args.ckpt)
    ckpt_stem = _safe_stem(ckpt_path)
    out_dir = os.path.abspath(args.out_dir or os.path.dirname(ckpt_path) or ".")
    os.makedirs(out_dir, exist_ok=True)

    requested_split = args.split
    id_label = None
    if args.id_list is not None:
        id_label = "ids_" + _safe_stem(args.id_list)
    output_split_label = id_label or requested_split

    out_json = args.out_json or os.path.join(out_dir, f"{ckpt_stem}_{output_split_label}_eval.json")
    out_ids = None if args.no_out_ids else (args.out_ids or os.path.join(out_dir, f"{ckpt_stem}_{output_split_label}_ids.txt"))
    out_pdf_prefix = args.out_pdf_prefix or os.path.join(out_dir, ckpt_stem)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate_checkpoint] Using device: {device}")
    print(f"[evaluate_checkpoint] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = _load_cfg_from_checkpoint(ckpt)
    _apply_sidecar_inference_defaults(cfg, ckpt_path)

    if args.embeddings is not None:
        cfg.embeddings = args.embeddings
    if args.labels is not None:
        cfg.labels = args.labels
    if args.dataset_dtype is not None:
        cfg.dataset_dtype = args.dataset_dtype
    if device.type != "cuda" and cfg.dataset_dtype == "float16":
        print("[evaluate_checkpoint] CPU device detected; using dataset_dtype=float32.")
        cfg.dataset_dtype = "float32"

    cfg.batch_size = int(args.batch_size)
    cfg.num_workers = int(args.num_workers)

    train_ds, val_ds, test_ds, d_model = build_datasets(cfg)
    if args.id_list is not None:
        id_list = _read_id_list(args.id_list)
        eval_ds = _dataset_from_id_list(test_ds, id_list)
        print(
            f"[evaluate_checkpoint] d_model={d_model}; id_list={args.id_list}; "
            f"n={len(eval_ds)}"
        )
    else:
        eval_ds = _select_dataset(args.split, train_ds, val_ds, test_ds)
        print(f"[evaluate_checkpoint] d_model={d_model}; split={args.split}; n={len(eval_ds)}")

    loader = DataLoader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    model = _build_model(cfg, d_model=d_model, device=device)
    model.load_state_dict(_state_dict_without_compile_prefix(ckpt["model"]))
    if "temperature" in ckpt:
        model.temperature = float(ckpt["temperature"])
        print(f"[evaluate_checkpoint] Loaded checkpoint temperature: T={model.temperature:.6g}")

    result = eval_split_full(
        split_name=output_split_label,
        checkpoint_path=ckpt_path,
        model=model,
        loader=loader,
        device=device,
        min_precisions=[float(x) for x in args.min_precision],
    )

    seq_ids = result.pop("seq_ids", [])
    _write_json(out_json, result)
    print(f"[evaluate_checkpoint] Wrote JSON: {out_json}")

    if out_ids is not None:
        n_ids = _write_ids(out_ids, seq_ids)
        print(f"[evaluate_checkpoint] Wrote {n_ids} IDs: {out_ids}")

    if args.curves_prefix is not None:
        save_curves_tsv(result.get("curves", {}), args.curves_prefix, split=output_split_label)
        print(f"[evaluate_checkpoint] Wrote curve TSV files with prefix: {args.curves_prefix}")

    if not args.no_pdf:
        pdf_outputs = write_pdf_outputs(result, out_pdf_prefix)
        for label, path in pdf_outputs.items():
            print(f"[evaluate_checkpoint] Wrote {label}: {path}")

    if not args.no_iou_violin:
        # Reuse train.py's exact evaluation collector to obtain the per-span IoU
        # values that eval_split_full summarizes as mean_iou and iou_at_0_5.
        # This second forward pass keeps train.py untouched and avoids changing
        # the historical JSON schema produced by training runs.
        _, _, _, iou_values = collect_eval_arrays(model, loader, device)
        iou_pdf = f"{out_pdf_prefix}_{output_split_label}_iou_violin.pdf"
        _plot_iou_violin_pdf(
            iou_pdf,
            iou_values,
            title=f"IoU distribution: {output_split_label}",
        )
        print(f"[evaluate_checkpoint] Wrote iou_violin_pdf: {iou_pdf}")
        iou_tsv = args.iou_values_tsv or os.path.join(out_dir, f"{ckpt_stem}_{output_split_label}_iou_values.tsv")
        n_iou = _write_iou_values_tsv(iou_tsv, iou_values)
        print(f"[evaluate_checkpoint] Wrote {n_iou} IoU values: {iou_tsv}")

    metrics = result.get("metrics", {})
    print("[evaluate_checkpoint] Summary:")
    for key in (
        "pr_auc_p_vs_rest",
        "roc_auc_p_vs_rest",
        "pr_auc_p_vs_n",
        "roc_auc_p_vs_n",
        "mean_iou",
        "iou_at_0_5",
    ):
        if key in metrics:
            print(f"  {key}: {metrics[key]:.6g}")


if __name__ == "__main__":
    main()

