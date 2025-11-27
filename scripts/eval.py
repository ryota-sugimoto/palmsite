#!/usr/bin/env python
"""
Evaluate a trained PalmSite model on validation and test splits.

This script:
  - Reconstructs the same train/val/test split as training
  - Loads a saved checkpoint (model_best.pt, model_best_iou.pt, etc.)
  - Evaluates validation and test sets
  - Reports:
      * PR-AUC (P vs rest)
      * PR-AUC (P vs N)
      * Mean IoU
      * IoU >= 0.5
      * Recall at precision >= --min-precision
      * ROC-AUC (P vs rest, P vs N)
  - Optionally writes results to a JSON file.

Usage example:

python eval.py \
  --ckpt runs/exp/model_best.pt \
  --embeddings embeddings.h5 \
  --labels labels.h5 \
  --min-precision 0.90 \
  --out eval_metrics.json
"""

import argparse
import json
import os

import numpy as np
import torch

from train import (
    TrainConfig,
    build_datasets,
    RdRPModel,
    collate_batch,
    evaluate,
    pr_auc,
    recall_at_precision,
)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Simple ROC-AUC implementation (no sklearn dependency).
    y_true: 0/1 array, 1 = positive
    y_score: score/probability for the positive class
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=np.float64)

    P = float((y_true == 1).sum())
    N = float((y_true == 0).sum())
    if P == 0 or N == 0:
        return float("nan")

    # Sort by score descending
    order = np.argsort(-y_score)
    y_true_sorted = y_true[order]

    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)

    tpr = tp / P
    fpr = fp / N

    # prepend (0,0)
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    auc = float(np.trapz(tpr, fpr))
    return auc


def collect_scores_with_labels(model: RdRPModel,
                               loader: torch.utils.data.DataLoader,
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
            y_bin = (y_raw == 2).astype(np.int64)  # 1 for Positive, 0 for others

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


def eval_split(name: str,
               model: RdRPModel,
               loader: torch.utils.data.DataLoader,
               device: torch.device,
               min_precision: float):
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
    """
    print(f"Evaluating on {name} ...")

    # Use the existing evaluate() to get IoU metrics and PR-AUCs
    base_metrics = evaluate(model, loader, device)

    # Collect labels and scores for extra metrics
    y_raw, y_bin, scores = collect_scores_with_labels(model, loader, device)

    # P vs rest
    if y_bin.size > 0:
        pr_rest = pr_auc(y_bin, scores)
        rec_rest = recall_at_precision(y_bin, scores, min_precision=min_precision)
        roc_rest = roc_auc(y_bin, scores)
    else:
        pr_rest = float("nan")
        rec_rest = float("nan")
        roc_rest = float("nan")

    # P vs N (exclude Unlabeled = 1)
    mask_n_only = (y_raw != 1)
    if mask_n_only.any():
        y_bin_n = y_bin[mask_n_only]
        scores_n = scores[mask_n_only]
        pr_n = pr_auc(y_bin_n, scores_n)
        rec_n = recall_at_precision(y_bin_n, scores_n, min_precision=min_precision)
        roc_n = roc_auc(y_bin_n, scores_n)
    else:
        pr_n = float("nan")
        rec_n = float("nan")
        roc_n = float("nan")

    metrics = dict(base_metrics)  # copy
    # Overwrite PR-AUCs with recomputed ones (should match base_metrics)
    metrics["pr_auc_p_vs_rest"] = float(pr_rest)
    metrics["pr_auc_p_vs_n"] = float(pr_n)
    metrics[f"recall_at_precision_{min_precision}_p_vs_rest"] = float(rec_rest)
    metrics[f"recall_at_precision_{min_precision}_p_vs_n"] = float(rec_n)
    metrics["roc_auc_p_vs_rest"] = float(roc_rest)
    metrics["roc_auc_p_vs_n"] = float(roc_n)

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
    p.add_argument("--out", type=str, default=None,
                   help="Optional path to JSON file to store results")

    args = p.parse_args()

    if args.device is None or args.device == "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load checkpoint
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

    # Rebuild datasets and loaders
    train_ds, val_ds, test_ds, d_model = build_datasets(cfg)
    d_in = d_model + 1  # +1 positional channel

    pin_memory = (device.type == "cuda")

    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_memory,
        collate_fn=collate_batch
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_memory,
        collate_fn=collate_batch
    )

    # Build model and restore state
    model = RdRPModel(d_in=d_in, tau=cfg.tau, alpha_cap=cfg.alpha_cap,
                      p_drop=cfg.dropout).to(device)

    # Restore the "discovery knobs" to ensure exact behavior
    model.wmin_base = cfg.wmin_base
    model.wmin_floor = cfg.wmin_floor
    model.seq_pool = "gauss"
    model.lenfeat_scale = cfg.lenfeat_scale
    model.coarse_stride = cfg.coarse_stride
    model.tau_len_gamma = cfg.tau_len_gamma
    model.tau_len_ref = cfg.tau_len_ref
    model.k_sigma = cfg.k_sigma

    model.load_state_dict(ckpt["model"])

    # Optional temperature from calibration
    if "temperature" in ckpt:
        model.temperature = float(ckpt["temperature"])
        print(f"Loaded calibrated temperature: T = {model.temperature:.3f}")

    model.eval()

    results = {
        "checkpoint": os.path.abspath(args.ckpt),
        "min_precision": args.min_precision,
    }

    if not args.no_val:
        metrics_val = eval_split("validation", model, val_loader, device, args.min_precision)
        results["val"] = metrics_val

    if not args.no_test:
        metrics_test = eval_split("test", model, test_loader, device, args.min_precision)
        results["test"] = metrics_test

    # Pretty-print to stdout
    print("\n=== Evaluation results ===")
    print(json.dumps(results, indent=2))

    # Optionally save to JSON
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
