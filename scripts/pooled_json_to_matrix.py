#!/usr/bin/env python
"""
Convert PalmSite --pooled-json output into a matrix for clustering/classification.

Example:
  python scripts/pooled_json_to_matrix.py \
    --pooled-json pooled_panels.json \
    --panel backbone.span_attn_norm \
    --best-only \
    --out-npz pooled_span_attn.npz \
    --metadata-tsv pooled_span_attn.metadata.tsv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


def _iter_records(obj: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for key, val in obj.items():
        if key.startswith("_"):
            continue
        if isinstance(val, dict) and "pools" in val:
            yield key, val


def _get_panel(rec: Dict[str, Any], panel: str) -> List[float] | None:
    cur: Any = rec.get("pools", {})
    for part in panel.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    if cur is None:
        return None
    if not isinstance(cur, list):
        raise TypeError(f"Panel {panel!r} is not a JSON list for chunk {rec.get('chunk_id')!r}")
    return cur


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert PalmSite pooled JSON panels to matrix files.")
    ap.add_argument("--pooled-json", required=True, help="JSON produced by palmsite --pooled-json")
    ap.add_argument("--panel", default="backbone.span_attn_norm",
                    help="Panel path under pools, e.g. backbone.span_attn_norm or input.full_mean")
    ap.add_argument("--best-only", action="store_true",
                    help="Keep only records where is_best_base_chunk is true")
    ap.add_argument("--min-p", type=float, default=None, help="Optional minimum PalmSite probability")
    ap.add_argument("--id-field", choices=["base_id", "chunk_id"], default="base_id",
                    help="Identifier to store as ids in the NPZ/TSV")
    ap.add_argument("--out-npz", default=None,
                    help="Write compressed NPZ containing X, ids, chunk_ids, base_ids, P, S_idx, E_idx")
    ap.add_argument("--out-tsv", default=None,
                    help="Write dense TSV with one vector component per column")
    ap.add_argument("--metadata-tsv", default=None,
                    help="Write metadata TSV without vector columns")
    args = ap.parse_args()

    if not args.out_npz and not args.out_tsv and not args.metadata_tsv:
        raise SystemExit("Specify at least one of --out-npz, --out-tsv, or --metadata-tsv")

    with open(args.pooled_json, "r", encoding="utf-8") as fh:
        obj = json.load(fh)

    rows: List[Dict[str, Any]] = []
    vectors: List[np.ndarray] = []
    expected_dim: int | None = None

    for key, rec in _iter_records(obj):
        if args.best_only and not bool(rec.get("is_best_base_chunk", False)):
            continue
        P = float(rec.get("P", np.nan))
        if args.min_p is not None and not (P >= float(args.min_p)):
            continue
        vec_list = _get_panel(rec, args.panel)
        if vec_list is None:
            continue
        vec = np.asarray(vec_list, dtype=np.float32)
        if vec.ndim != 1:
            raise ValueError(f"Panel {args.panel!r} for {key!r} is not 1-dimensional")
        if expected_dim is None:
            expected_dim = int(vec.shape[0])
        elif int(vec.shape[0]) != expected_dim:
            raise ValueError(f"Panel dimension mismatch for {key!r}: {vec.shape[0]} vs {expected_dim}")

        chunk_id = str(rec.get("chunk_id", key))
        base_id = str(rec.get("base_id", chunk_id))
        row = {
            "id": base_id if args.id_field == "base_id" else chunk_id,
            "chunk_id": chunk_id,
            "base_id": base_id,
            "P": P,
            "S_idx": int(rec.get("S_idx", -1)),
            "E_idx": int(rec.get("E_idx", -1)),
            "L": int(rec.get("L", -1)),
            "orig_start": int(rec.get("orig_start", -1)),
            "orig_len": int(rec.get("orig_len", -1)),
            "is_best_base_chunk": bool(rec.get("is_best_base_chunk", False)),
        }
        rows.append(row)
        vectors.append(vec)

    if not vectors:
        raise SystemExit("No records matched the requested filters/panel")

    X = np.vstack(vectors).astype(np.float32, copy=False)
    ids = np.asarray([r["id"] for r in rows], dtype=object)
    chunk_ids = np.asarray([r["chunk_id"] for r in rows], dtype=object)
    base_ids = np.asarray([r["base_id"] for r in rows], dtype=object)
    P = np.asarray([r["P"] for r in rows], dtype=np.float32)
    S_idx = np.asarray([r["S_idx"] for r in rows], dtype=np.int32)
    E_idx = np.asarray([r["E_idx"] for r in rows], dtype=np.int32)

    if args.out_npz:
        os.makedirs(os.path.dirname(args.out_npz) or ".", exist_ok=True)
        np.savez_compressed(
            args.out_npz,
            X=X,
            ids=ids,
            chunk_ids=chunk_ids,
            base_ids=base_ids,
            P=P,
            S_idx=S_idx,
            E_idx=E_idx,
            panel=np.asarray(args.panel),
        )

    meta_fields = ["id", "chunk_id", "base_id", "P", "S_idx", "E_idx", "L", "orig_start", "orig_len", "is_best_base_chunk"]

    if args.metadata_tsv:
        os.makedirs(os.path.dirname(args.metadata_tsv) or ".", exist_ok=True)
        with open(args.metadata_tsv, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=meta_fields, delimiter="\t")
            wr.writeheader()
            for row in rows:
                wr.writerow(row)

    if args.out_tsv:
        os.makedirs(os.path.dirname(args.out_tsv) or ".", exist_ok=True)
        with open(args.out_tsv, "w", newline="", encoding="utf-8") as fh:
            fieldnames = meta_fields + [f"v{i}" for i in range(X.shape[1])]
            wr = csv.writer(fh, delimiter="\t")
            wr.writerow(fieldnames)
            for row, vec in zip(rows, X):
                wr.writerow([row[f] for f in meta_fields] + [float(x) for x in vec])

    print(f"Wrote {X.shape[0]} vectors with dimension {X.shape[1]} from panel {args.panel!r}.")


if __name__ == "__main__":
    main()
