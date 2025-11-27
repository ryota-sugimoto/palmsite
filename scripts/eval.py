#!/usr/bin/env python
"""
Evaluation script for PalmSite.

Features:
  - Rebuilds train/val/test splits exactly as in training (same cfg, seed, labels.h5)
  - Loads a checkpoint (e.g. model_best.pt, model_best_iou.pt)
  - Evaluates one split: train / val / test
  - Computes:
      * PR-AUC (P vs rest, P vs N)
      * ROC-AUC (P vs rest, P vs N)
      * Mean IoU, IoU >= 0.5 (via train.evaluate)
      * Recall at given precision thresholds
      * PR curves (recall–precision points) for P vs rest and P vs N
      * ROC curves (FPR–TPR points) for P vs rest and P vs N
      * Number of samples in the evaluated split
      * Sequence IDs for the evaluated split (optionally written to a file)
  - Saves all metrics and curve points to a JSON file if --out-json is given.

Example:

python eval.py \\
  --ckpt runs/exp/model_best.pt \\
  --embeddings embeddings.h5 \\
  --labels labels.h5 \\
  --split test \\
  --batch-size 32 \\
  --num-workers 2 \\
  --min-precision 0.90 0.95 \\
  --out-json runs/exp/eval_test.json \\
  --out-ids runs/exp/test_ids.txt
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Import core components from train.py
from train import (
    TrainConfig,
    build_datasets,
    collate_batch,
    RdRPModel,
    evaluate,
    pr_auc,
    recall_at_precision,
)


def collect_scores_with_labels(
    model: RdRPModel,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect raw labels, scores, and chunk_ids for a given split.

    Returns:
        y_raw:   shape (N,), values in {0,1,2} = {N, U, P}
        y_bin:   shape (N,), values in {0,1}, 1 = Positive (P), 0 = (N or U)
        scores:  shape (N,), sigmoid probabilities P(RdRP | sequence)
        ids:     shape (N,), dtype=object, the chunk_ids from the dataset
    """
    model.eval()
    ys_raw: List[np.ndarray] = []
    ys_bin: List[np.ndarray] = []
    scores: List[np.ndarray] = []
    ids: List[str] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            mask = batch["mask"].to(device)
            L = batch["L"].to(device)
            y = batch["y"].to(device)  # 0=N, 1=U, 2=P
            chunk_ids = batch.get("chunk_ids", None)

            out = model(x, mask, L)
            logits = out["logit"]
            T = getattr(model, "temperature", None)
            if T is not None:
                logits = logits / float(T)
            P = torch.sigmoid(logits).detach().cpu().numpy()

            y_raw = y.cpu().numpy()
            y_bin = (y_raw == 2).astype(np.int64)

            ys_raw.append(y_raw)
            ys_bin.append(y_bin)
            scores.append(P)
            if chunk_ids is not None:
                ids.extend(list(chunk_ids))

    if not ys_raw:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=object),
        )

    y_raw_all = np.concatenate(ys_raw)
    y_bin_all = np.concatenate(ys_bin)
    scores_all = np.concatenate(scores)
    ids_all = np.array(ids, dtype=object) if ids else np.zeros((0,), dtype=object)
    return y_raw_all, y_bin_all, scores_all, ids_all


def compute_pr_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a simple precision-recall curve (P vs rest) without external deps.
    Returns:
        recall:    shape (K,)
        precision: shape (K,)
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    order = np.argsort(-y_score)
    y = y_true[order]

    P = float((y == 1).sum())
    if P == 0:
        # No positives: degenerate curve
        return np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64)

    tp = 0.0
    fp = 0.0
    precisions: List[float] = []
    recalls: List[float] = []

    for i in range(len(y)):
        if y[i] == 1:
            tp += 1.0
        else:
            fp += 1.0
        prec = tp / (tp + fp)
        rec = tp / P
        precisions.append(prec)
        recalls.append(rec)

    return np.asarray(recalls, dtype=np.float64), np.asarray(precisions, dtype=np.float64)


def compute_roc_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a simple ROC curve (P vs rest) without external deps.
    Returns:
        fpr: shape (K,)
        tpr: shape (K,)
    """
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    order = np.argsort(-y_score)
    y = y_true[order]

    P = float((y == 1).sum())
    N = float((y == 0).sum())
    if P == 0 or N == 0:
        # Degenerate case
        return np.array([0.0, 1.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)

    tp = 0.0
    fp = 0.0
    tps: List[float] = []
    fps: List[float] = []

    for i in range(len(y)):
        if y[i] == 1:
            tp += 1.0
        else:
            fp += 1.0
        tps.append(tp)
        fps.append(fp)

    tps_arr = np.asarray(tps, dtype=np.float64)
    fps_arr = np.asarray(fps, dtype=np.float64)

    tpr = tps_arr / P
    fpr = fps_arr / N
    return fpr, tpr


def roc_auc_from_curve(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Compute ROC-AUC from FPR/TPR curve using trapezoidal rule.
    """
    if fpr.size < 2:
        return float("nan")
    order = np.argsort(fpr)
    fpr_s = fpr[order]
    tpr_s = tpr[order]
    return float(np.trapz(tpr_s, fpr_s))


def _select_dataset(split: str, train_ds, val_ds, test_ds):
    s = split.lower()
    if s == "train":
        return train_ds
    elif s in ("val", "valid", "validation"):
        return val_ds
    elif s == "test":
        return test_ds
    else:
        raise ValueError(f"Unknown split {split!r}; use 'train', 'val', or 'test'.")


def eval_split(
    split_name: str,
    model: RdRPModel,
    loader: DataLoader,
    device: torch.device,
    min_precisions: List[float],
) -> Dict[str, Any]:
    """
    Evaluate a single split (train / val / test) and compute:
      - core metrics via evaluate()
      - recall@precision
      - PR and ROC curves for P vs rest and P vs N

    Returns a dict with:
        {
          "metrics": {...},
          "curves": {...},
          "n_samples": int,
          "seq_ids": [id1, id2, ...]
        }
    """
    print(f"[eval] Evaluating split={split_name} ...")

    # 1) Core metrics (PR-AUC and IoU metrics)
    core = evaluate(model, loader, device)

    # 2) Collect labels, scores, and ids
    y_raw, y_bin, scores, ids = collect_scores_with_labels(model, loader, device)
    n_examples = int(y_raw.size)
    print(f"[eval] Collected {n_examples} examples for split={split_name}")

    metrics: Dict[str, Any] = dict(core)

    if n_examples == 0:
        return {
            "metrics": metrics,
            "curves": {},
            "n_samples": 0,
            "seq_ids": [],
        }

    # 3) P vs rest (P vs {U+N})
    pr_rest = pr_auc(y_bin, scores)
    metrics["pr_auc_p_vs_rest"] = float(pr_rest)

    fpr_rest, tpr_rest = compute_roc_curve(y_bin, scores)
    roc_auc_rest = roc_auc_from_curve(fpr_rest, tpr_rest)
    metrics["roc_auc_p_vs_rest"] = float(roc_auc_rest)

    rec_rest_curve, prec_rest_curve = compute_pr_curve(y_bin, scores)
    pr_curve_rest = {
        "recall": rec_rest_curve.tolist(),
        "precision": prec_rest_curve.tolist(),
    }

    for mp in min_precisions:
        r_val = recall_at_precision(y_bin, scores, min_precision=float(mp))
        key = f"recall_at_precision_{mp:.2f}_p_vs_rest"
        metrics[key] = float(r_val)

    # 4) P vs N (exclude Unlabeled)
    mask_n_only = (y_raw != 1)
    if mask_n_only.any():
        y_bin_n = y_bin[mask_n_only]
        scores_n = scores[mask_n_only]

        pr_n = pr_auc(y_bin_n, scores_n)
        metrics["pr_auc_p_vs_n"] = float(pr_n)

        fpr_n, tpr_n = compute_roc_curve(y_bin_n, scores_n)
        roc_auc_n = roc_auc_from_curve(fpr_n, tpr_n)
        metrics["roc_auc_p_vs_n"] = float(roc_auc_n)

        rec_n_curve, prec_n_curve = compute_pr_curve(y_bin_n, scores_n)
        pr_curve_n = {
            "recall": rec_n_curve.tolist(),
            "precision": prec_n_curve.tolist(),
        }

        for mp in min_precisions:
            r_val_n = recall_at_precision(y_bin_n, scores_n, min_precision=float(mp))
            key_n = f"recall_at_precision_{mp:.2f}_p_vs_n"
            metrics[key_n] = float(r_val_n)
    else:
        # No negatives in this split (very unlikely for test, but be safe)
        fpr_n = np.array([], dtype=np.float64)
        tpr_n = np.array([], dtype=np.float64)
        pr_curve_n = {"recall": [], "precision": []}
        metrics["pr_auc_p_vs_n"] = float("nan")
        metrics["roc_auc_p_vs_n"] = float("nan")

    curves = {
        "pr_curve_p_vs_rest": pr_curve_rest,
        "roc_curve_p_vs_rest": {
            "fpr": fpr_rest.tolist(),
            "tpr": tpr_rest.tolist(),
        },
        "pr_curve_p_vs_n": pr_curve_n,
        "roc_curve_p_vs_n": {
            "fpr": fpr_n.tolist(),
            "tpr": tpr_n.tolist(),
        },
    }

    # Summary print (avoid printing full curves or id list)
    print(f"[eval] Split={split_name} summary:")
    print(f"  n_samples: {n_examples}")
    print(f"  PR-AUC (P vs rest): {metrics['pr_auc_p_vs_rest']:.4f}")
    print(f"  ROC-AUC (P vs rest): {metrics['roc_auc_p_vs_rest']:.4f}")
    print(f"  PR-AUC (P vs N): {metrics['pr_auc_p_vs_n']:.4f}")
    print(f"  ROC-AUC (P vs N): {metrics['roc_auc_p_vs_n']:.4f}")
    if "mean_iou" in metrics:
        print(f"  Mean IoU: {metrics['mean_iou']:.4f}")
    if "iou_at_0_5" in metrics:
        print(f"  IoU>=0.5: {metrics['iou_at_0_5']:.4f}")
    for mp in min_precisions:
        key_r = f"recall_at_precision_{mp:.2f}_p_vs_rest"
        print(f"  {key_r}: {metrics[key_r]:.4f}")

    return {
        "metrics": metrics,
        "curves": curves,
        "n_samples": n_examples,
        "seq_ids": ids.tolist(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained PalmSite model.")
    p.add_argument("--ckpt", required=True,
                   help="Path to checkpoint (e.g. runs/exp/model_best.pt)")
    p.add_argument("--embeddings", default=None,
                   help="Path to embeddings.h5 (optional; overrides cfg in checkpoint)")
    p.add_argument("--labels", default=None,
                   help="Path to labels.h5 (optional; overrides cfg in checkpoint)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"],
                   help="Dataset split to evaluate (default: test)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for evaluation")
    p.add_argument("--num-workers", type=int, default=2,
                   help="Number of DataLoader workers")
    p.add_argument(
        "--min-precision",
        type=float,
        nargs="*",
        default=[0.90],
        help=("Precision thresholds for recall@precision (default: 0.90). "
              "You can pass multiple values, e.g. --min-precision 0.90 0.95"),
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write evaluation metrics and curves as JSON "
             "(e.g. runs/exp/eval_test.json)",
    )
    p.add_argument(
        "--out-ids",
        default=None,
        help="Optional path to write the list of sequence IDs (one per line), "
             "e.g. runs/exp/test_ids.txt",
    )
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}")

    # Load checkpoint and reconstruct config
    ckpt = torch.load(args.ckpt, map_location=device)
    if "cfg" not in ckpt:
        raise RuntimeError("Checkpoint does not contain 'cfg'; ensure it was saved by train.py.")
    cfg = TrainConfig(**ckpt["cfg"])

    # Override embeddings/labels from CLI if provided
    if args.embeddings is not None:
        cfg.embeddings = args.embeddings
    if args.labels is not None:
        cfg.labels = args.labels

    # Build datasets (same split as training)
    train_ds, val_ds, test_ds, d_model = build_datasets(cfg)
    print(f"[eval] d_model={d_model}")

    ds = _select_dataset(args.split, train_ds, val_ds, test_ds)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
    )

    # Build model
    d_in = d_model + 1  # +1 for position channel
    model = RdRPModel(d_in=d_in, tau=cfg.tau, alpha_cap=cfg.alpha_cap, p_drop=cfg.dropout).to(device)

    # Inject discovery knobs for consistency with training
    model.wmin_base = cfg.wmin_base
    model.wmin_floor = cfg.wmin_floor
    model.seq_pool = "gauss"
    model.lenfeat_scale = cfg.lenfeat_scale
    model.coarse_stride = cfg.coarse_stride
    model.tau_len_gamma = cfg.tau_len_gamma
    model.tau_len_ref = cfg.tau_len_ref
    model.k_sigma = cfg.k_sigma

    # Load weights
    model.load_state_dict(ckpt["model"])

    # Optional temperature calibration
    if "temperature" in ckpt:
        model.temperature = float(ckpt["temperature"])
        print(f"[eval] Loaded calibrated temperature: T={model.temperature:.3f}")

    # Run evaluation on the requested split
    split_result = eval_split(
        split_name=args.split,
        model=model,
        loader=loader,
        device=device,
        min_precisions=args.min_precision,
    )

    seq_ids = split_result.pop("seq_ids")
    n_samples = split_result["n_samples"]

    # Assemble top-level result object
    result_out: Dict[str, Any] = {
        "checkpoint": os.path.abspath(args.ckpt),
        "split": args.split,
        "min_precisions": args.min_precision,
        "n_samples": n_samples,
        "metrics": split_result["metrics"],
        "curves": split_result["curves"],
    }

    # Save JSON if requested
    if args.out_json is not None:
        out_dir = os.path.dirname(args.out_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(result_out, f, indent=2)
        print(f"[eval] Wrote metrics+curves JSON to {args.out_json}")

    # Save sequence IDs if requested
    if args.out_ids is not None:
        out_dir_ids = os.path.dirname(args.out_ids)
        if out_dir_ids:
            os.makedirs(out_dir_ids, exist_ok=True)
        with open(args.out_ids, "w", encoding="utf-8") as f:
            for cid in seq_ids:
                f.write(f"{cid}\n")
        print(f"[eval] Wrote {n_samples} sequence IDs to {args.out_ids}")


if __name__ == "__main__":
    main()

