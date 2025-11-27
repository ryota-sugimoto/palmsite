#!/usr/bin/env python
"""
Evaluate a trained PalmSite model on validation and test splits.

This script:
  - Reconstructs the same train/val/test split as training
    (using the cfg stored in the checkpoint + labels.h5).
  - Loads a saved checkpoint (model_best.pt, model_best_iou.pt, etc.).
  - Evaluates validation and/or test sets.
  - Reports:
      * PR-AUC (P vs rest)
      * PR-AUC (P vs N)
      * Mean IoU
      * IoU >= 0.5
      * Recall at precision >= --min-precision (P vs rest, P vs N)
      * ROC-AUC (P vs rest, P vs N)
  - Optionally saves PR and ROC curve data (tsv) for plotting.

Usage example:

python eval.py \
  --ckpt runs/exp/model_best.pt \
  --embeddings embeddings.h5 \
  --labels labels.h5 \
  --min-precision 0.90 \
  --curves-prefix runs/exp/curves \
  --out runs/exp/eval_metrics.json
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from train import (
    TrainConfig,
    build_datasets,
    RdRPModel,
    collate_batch,
    evaluate,
    pr_auc,
    recall_at_precision,
)


def collect_scores_with_labels(model: RdRPModel,
                               loader: DataLoader,
                               device: torch.device):
    """
    Collect raw labels (0/1/2), binary P-vs-rest labels, and scores for a loader.

    Returns:
        y_raw:  np.ndarray of shape (N,), labels in {0,1,2}
        y_bin:  np.ndarray of shape (N,), labels in {0,1}, 1 = Positive
        scores: np.ndarray of shape (N,), sigmoid probabilities
    """
    model.eval()
    ys_raw = []
    ys_bin = []
    scores = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            L = batch["L"].to(device)
            y = batch["y"].to(device)  # 0=N, 1=U, 2=P

            out = model(x, mask, L)
            logits = out["logit"]
            T = getattr(model, "temperature", None)
            if T is not None:
                logits = logits / float(T)
            P = torch.sigmoid(logits).detach().cpu().numpy()

            y_raw = y.cpu().numpy()
            y_bin = (y_raw == 2).astype(np.int64)  # 1 = Positive, 0 = others

            ys_raw.append(y_raw)
            ys_bin.append(y_bin)
            scores.append(P)

    if not ys_raw:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    y_raw_all = np.concatenate(ys_raw)
    y_bin_all = np.concatenate(ys_bin)
    scores_all = np.concatenate(scores)
    return y_raw_all, y_bin_all, scores_all


def compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray):
    """
    Compute precision-recall curve (similar to sklearn.metrics.precision_recall_curve).

    Returns:
        precision: np.ndarray
        recall:    np.ndarray
        thresholds: np.ndarray (score thresholds)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)

    P = float((y_true == 1).sum())
    if P == 0 or y_true.size == 0:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    tp = 0.0
    fp = 0.0
    precisions = []
    recalls = []
    thresholds = []
    last_score = None

    for i in range(len(y_sorted)):
        yi = y_sorted[i]
        si = s_sorted[i]
        if last_score is None or si != last_score:
            if i > 0:
                precision = tp / (tp + fp)
                recall = tp / P
                precisions.append(precision)
                recalls.append(recall)
                thresholds.append(last_score)
            last_score = si
        if yi == 1:
            tp += 1.0
        else:
            fp += 1.0

    # final point
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / P
    precisions.append(precision)
    recalls.append(recall)
    thresholds.append(last_score)

    return np.array(precisions), np.array(recalls), np.array(thresholds)


def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    """
    Compute ROC curve (FPR, TPR) and thresholds.

    Returns:
        fpr: np.ndarray
        tpr: np.ndarray
        thresholds: np.ndarray
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)

    P = float((y_true == 1).sum())
    N = float((y_true == 0).sum())
    if P == 0 or N == 0 or y_true.size == 0:
        return np.array([]), np.array([]), np.array([])

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    tp = 0.0
    fp = 0.0
    tprs = []
    fprs = []
    thresholds = []
    last_score = None

    for i in range(len(y_sorted)):
        yi = y_sorted[i]
        si = s_sorted[i]
        if last_score is None or si != last_score:
            tprs.append(tp / P)
            fprs.append(fp / N)
            thresholds.append(si)
            last_score = si
        if yi == 1:
            tp += 1.0
        else:
            fp += 1.0

    # append final point
    tprs.append(tp / P)
    fprs.append(fp / N)
    thresholds.append(s_sorted[-1])

    # Ensure start at (0,0)
    if fprs[0] != 0.0 or tprs[0] != 0.0:
        fprs.insert(0, 0.0)
        tprs.insert(0, 0.0)
        thresholds.insert(0, thresholds[0])

    return np.array(fprs), np.array(tprs), np.array(thresholds)


def roc_auc_from_curve(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute ROC-AUC from an ROC curve (fpr, tpr).
    """
    if fpr.size == 0 or tpr.size == 0:
        return float("nan")
    return float(np.trapz(tpr, fpr))


def save_pr_curve(path: str, thresholds: np.ndarray,
                  precision: np.ndarray, recall: np.ndarray) -> None:
    """
    Save PR curve to a TSV file: threshold, precision, recall
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("threshold\tprecision\trecall\n")
        for t, p, r in zip(thresholds, precision, recall):
            f.write(f"{t:.8g}\t{p:.8g}\t{r:.8g}\n")


def save_roc_curve(path: str, thresholds: np.ndarray,
                   fpr: np.ndarray, tpr: np.ndarray) -> None:
    """
    Save ROC curve to a TSV file: threshold, fpr, tpr
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("threshold\tfpr\ttpr\n")
        for t, x, y in zip(thresholds, fpr, tpr):
            f.write(f"{t:.8g}\t{x:.8g}\t{y:.8g}\n")


def eval_split(name: str,
               model: RdRPModel,
               loader: DataLoader,
               device: torch.device,
               min_precision: float,
               curves_prefix: str | None = None):
    """
    Evaluate one split (val or test).

    Returns a dict with:
        - pr_auc_p_vs_rest
        - pr_auc_p_vs_n
        - mean_iou
        - iou_at_0_5
        - recall_at_{min_precision}_p_vs_rest
        - recall_at_{min_precision}_p_vs_n
        - roc_auc_p_vs_rest
        - roc_auc_p_vs_n

    If curves_prefix is not None, also saves PR/ROC curves as TSV files.
    """
    print(f"Evaluating on {name} ...")

    # Base summary from existing evaluate()
    base_metrics = evaluate(model, loader, device)

    # Collect labels & scores for extra metrics and curves
    y_raw, y_bin, scores = collect_scores_with_labels(model, loader, device)

    metrics = dict(base_metrics)  # copy

    # ---------- P vs rest ----------
    if y_bin.size > 0:
        pr_rest = pr_auc(y_bin, scores)
        rec_rest = recall_at_precision(y_bin, scores, min_precision=min_precision)
        prec_curve, rec_curve, thr_pr = compute_pr_curve(y_bin, scores)
        fpr_rest, tpr_rest, thr_roc_rest = compute_roc_curve(y_bin, scores)
        roc_rest = roc_auc_from_curve(fpr_rest, tpr_rest)
    else:
        pr_rest = float("nan")
        rec_rest = float("nan")
        roc_rest = float("nan")
        prec_curve = rec_curve = thr_pr = np.array([])
        fpr_rest = tpr_rest = thr_roc_rest = np.array([])

    metrics["pr_auc_p_vs_rest"] = float(pr_rest)
    metrics[f"recall_at_precision_{min_precision}_p_vs_rest"] = float(rec_rest)
    metrics["roc_auc_p_vs_rest"] = float(roc_rest)

    # ---------- P vs N (exclude Unlabeled) ----------
    mask_n_only = (y_raw != 1)
    if mask_n_only.any():
        y_bin_n = y_bin[mask_n_only]
        scores_n = scores[mask_n_only]
        pr_n = pr_auc(y_bin_n, scores_n)
        rec_n = recall_at_precision(y_bin_n, scores_n, min_precision=min_precision)
        prec_curve_n, rec_curve_n, thr_pr_n = compute_pr_curve(y_bin_n, scores_n)
        fpr_n, tpr_n, thr_roc_n = compute_roc_curve(y_bin_n, scores_n)
        roc_n = roc_auc_from_curve(fpr_n, tpr_n)
    else:
        pr_n = float("nan")
        rec_n = float("nan")
        roc_n = float("nan")
        prec_curve_n = rec_curve_n = thr_pr_n = np.array([])
        fpr_n = tpr_n = thr_roc_n = np.array([])

    metrics["pr_auc_p_vs_n"] = float(pr_n)
    metrics[f"recall_at_precision_{min_precision}_p_vs_n"] = float(rec_n)
    metrics["roc_auc_p_vs_n"] = float(roc_n)

    # ---------- Save curves (optional) ----------
    if curves_prefix is not None:
        base = f"{curves_prefix}_{name}"

        # Ensure directory exists (if prefix has a directory component)
        base_dir = os.path.dirname(base)
        if base_dir:
            os.makedirs(base_dir, exist_ok=True)

        # PR curves
        if thr_pr.size > 0:
            save_pr_curve(f"{base}_pr_p_vs_rest.tsv", thr_pr, prec_curve, rec_curve)
        if thr_pr_n.size > 0:
            save_pr_curve(f"{base}_pr_p_vs_n.tsv", thr_pr_n, prec_curve_n, rec_curve_n)

        # ROC curves
        if thr_roc_rest.size > 0:
            save_roc_curve(f"{base}_roc_p_vs_rest.tsv", thr_roc_rest, fpr_rest, tpr_rest)
        if thr_roc_n.size > 0:
            save_roc_curve(f"{base}_roc_p_vs_n.tsv", thr_roc_n, fpr_n, tpr_n)

    return metrics


def main():
    p = argparse.ArgumentParser(description="Evaluate PalmSite model on val/test splits.")
    p.add_argument("--ckpt", required=True,
                   help="Path to checkpoint (e.g., runs/exp/model_best.pt)")
    p.add_argument("--embeddings", required=True,
                   help="Path to embeddings HDF5 file used for training")
    p.add_argument("--labels", required=True,
                   help="Path to labels HDF5 file used for training")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Batch size for evaluation (default: use value from checkpoint)")
    p.add_argument("--num-workers", type=int, default=None,
                   help="DataLoader workers (default: use value from checkpoint)")
    p.add_argument("--min-precision", type=float, default=0.90,
                   help="Target precision for recall@precision metric")
    p.add_argument("--device", type=str, default=None,
                   help="Device: 'cuda', 'cpu', or empty for auto")
    p.add_argument("--no-val", action="store_true",
                   help="Skip evaluation on validation split")
    p.add_argument("--no-test", action="store_true",
                   help="Skip evaluation on test split")
    p.add_argument("--curves-prefix", type=str, default=None,
                   help=("Prefix for saving PR/ROC curves as TSV files. "
                         "Files will be named like PREFIX_val_pr_p_vs_rest.tsv, etc."))
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to JSON file to store results")

    args = p.parse_args()

    if args.device is None or args.device == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load checkpoint & config
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg_dict = ckpt.get("cfg", None)
    if cfg_dict is None:
        raise RuntimeError("Checkpoint does not contain 'cfg' field with TrainConfig.")

    cfg = TrainConfig(**cfg_dict)
    # Override paths/batch size/workers from CLI
    cfg.embeddings = args.embeddings
    cfg.labels = args.labels
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers

    # Rebuild datasets & loaders
    train_ds, val_ds, test_ds, d_model = build_datasets(cfg)
    d_in = d_model + 1  # +1 positional channel
    pin_memory = (device.type == "cuda")

    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_memory,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_memory,
        collate_fn=collate_batch,
    )

    # Build model & restore state
    model = RdRPModel(d_in=d_in, tau=cfg.tau, alpha_cap=cfg.alpha_cap,
                      p_drop=cfg.dropout).to(device)
    # Discovery knobs (must mirror train())
    model.wmin_base = cfg.wmin_base
    model.wmin_floor = cfg.wmin_floor
    model.seq_pool = 'gauss'
    model.lenfeat_scale = cfg.lenfeat_scale
    model.coarse_stride = cfg.coarse_stride
    model.tau_len_gamma = cfg.tau_len_gamma
    model.tau_len_ref = cfg.tau_len_ref
    model.k_sigma = cfg.k_sigma

    model.load_state_dict(ckpt["model"])

    # Optional temperature
    if "temperature" in ckpt:
        model.temperature = float(ckpt["temperature"])
        print(f"Loaded calibrated temperature: T = {model.temperature:.3f}")

    model.eval()

    results = {
        "checkpoint": os.path.abspath(args.ckpt),
        "min_precision": args.min_precision,
    }

    # Evaluate val
    if not args.no_val:
        metrics_val = eval_split("val", model, val_loader, device,
                                 args.min_precision, args.curves_prefix)
        results["val"] = metrics_val

    # Evaluate test
    if not args.no_test:
        metrics_test = eval_split("test", model, test_loader, device,
                                  args.min_precision, args.curves_prefix)
        results["test"] = metrics_test

    # Print to stdout
    print("\n=== Evaluation results ===")
    print(json.dumps(results, indent=2))

    # Save JSON if requested
    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()

