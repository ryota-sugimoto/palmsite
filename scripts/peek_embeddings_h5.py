#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
from typing import Dict, Any, List, Tuple, Optional

import h5py
import numpy as np


REQUIRED_DATASETS = ["emb", "mask"]  # seq is strongly expected but some pipelines may omit it
REQUIRED_ATTRS = [
    "aa_len", "total_tokens", "orig_aa_len",
    "is_chunked", "chunk_index", "chunks_total", "orig_aa_start", "orig_aa_end",
    "bos_index", "eos_index",
    "esm_model", "esm_url", "d_model",
]


def _as_py(v):
    """Convert h5 attribute values to plain python scalars/strings where possible."""
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (bytes, bytearray)):
        try:
            return v.decode("utf-8", errors="replace")
        except Exception:
            return str(v)
    return v


def list_some(keys: List[str], n: int, seed: int) -> List[str]:
    if not keys:
        return []
    rnd = random.Random(seed)
    if len(keys) <= n:
        return keys
    return rnd.sample(keys, n)


def check_item(g: h5py.Group, key: str, strict: bool = False) -> List[str]:
    """Return list of warnings/errors for one /items/<key> group."""
    problems: List[str] = []

    # datasets
    for ds in REQUIRED_DATASETS:
        if ds not in g:
            problems.append(f"[MISSING_DATASET] {key}: missing dataset '{ds}'")

    # emb shape
    if "emb" in g:
        emb = g["emb"]
        if emb.ndim != 2:
            problems.append(f"[BAD_SHAPE] {key}: emb.ndim={emb.ndim}, expected 2")
        else:
            T, D = emb.shape
            d_model = g.attrs.get("d_model", None)
            if d_model is not None:
                d_model = _as_py(d_model)
                if isinstance(d_model, int) and d_model > 0 and D != d_model:
                    problems.append(f"[D_MODEL_MISMATCH] {key}: emb.shape[1]={D} != d_model={d_model}")

            total_tokens = g.attrs.get("total_tokens", None)
            if total_tokens is not None:
                total_tokens = _as_py(total_tokens)
                if isinstance(total_tokens, int) and total_tokens > 0 and T != total_tokens:
                    problems.append(f"[TOK_MISMATCH] {key}: emb.shape[0]={T} != total_tokens={total_tokens}")

    # mask
    if "mask" in g:
        mask = g["mask"]
        if mask.ndim != 1:
            problems.append(f"[BAD_SHAPE] {key}: mask.ndim={mask.ndim}, expected 1")
        else:
            Tm = mask.shape[0]
            if "emb" in g and g["emb"].ndim == 2:
                Te = g["emb"].shape[0]
                if Te != Tm:
                    problems.append(f"[MASK_LEN_MISMATCH] {key}: mask len={Tm} != emb T={Te}")

    # seq / aa_len checks
    aa_len = _as_py(g.attrs.get("aa_len", -1))
    seq_len = None
    if "seq" in g:
        try:
            seq = g["seq"][()].decode("utf-8") if isinstance(g["seq"][()], (bytes, bytearray)) else str(g["seq"][()])
            seq_len = len(seq)
        except Exception:
            problems.append(f"[SEQ_READ_FAIL] {key}: could not read /seq dataset")

    if isinstance(aa_len, int) and aa_len >= 0:
        if seq_len is not None and aa_len != seq_len:
            problems.append(f"[AA_LEN_MISMATCH] {key}: aa_len={aa_len} != len(seq)={seq_len}")

        # mask sum should equal aa_len if special tokens exist; or equal T if no special tokens and mask all True
        if "mask" in g:
            try:
                m = g["mask"][()]
                msum = int(np.sum(m.astype(np.int64)))
                # allowable:
                # - if special tokens: mask sum == aa_len
                # - if no special tokens: mask sum == T (and T==aa_len)
                if "emb" in g and g["emb"].ndim == 2:
                    T = g["emb"].shape[0]
                    # Either masksum==aa_len (special tokens or masked) or masksum==T (all tokens valid)
                    if not (msum == aa_len or msum == T):
                        problems.append(f"[MASK_SUM_ODD] {key}: mask.sum={msum}, aa_len={aa_len}, T={T}")
            except Exception:
                problems.append(f"[MASK_READ_FAIL] {key}: could not read /mask dataset")

    # required attrs presence
    for a in REQUIRED_ATTRS:
        if a not in g.attrs:
            problems.append(f"[MISSING_ATTR] {key}: missing attr '{a}'")

    # chunking consistency
    is_chunked = _as_py(g.attrs.get("is_chunked", False))
    chunks_total = _as_py(g.attrs.get("chunks_total", 1))
    chunk_index = _as_py(g.attrs.get("chunk_index", 1))
    orig_aa_start = _as_py(g.attrs.get("orig_aa_start", 0))
    orig_aa_end = _as_py(g.attrs.get("orig_aa_end", 0))
    if isinstance(chunks_total, int) and chunks_total > 1 and not bool(is_chunked):
        problems.append(f"[CHUNK_FLAG_ODD] {key}: chunks_total={chunks_total} but is_chunked={is_chunked}")
    if isinstance(chunk_index, int) and isinstance(chunks_total, int):
        if chunk_index < 1 or chunk_index > max(1, chunks_total):
            problems.append(f"[CHUNK_INDEX_ODD] {key}: chunk_index={chunk_index}, chunks_total={chunks_total}")
    if isinstance(orig_aa_start, int) and isinstance(orig_aa_end, int) and orig_aa_end < orig_aa_start:
        problems.append(f"[COORD_ODD] {key}: orig_aa_end({orig_aa_end}) < orig_aa_start({orig_aa_start})")

    if strict and problems:
        raise RuntimeError("Strict mode failed:\n" + "\n".join(problems))
    return problems


def print_item_summary(g: h5py.Group, key: str):
    emb = g.get("emb", None)
    mask = g.get("mask", None)

    emb_shape = tuple(emb.shape) if emb is not None else None
    emb_dtype = str(emb.dtype) if emb is not None else None
    mask_shape = tuple(mask.shape) if mask is not None else None
    mask_dtype = str(mask.dtype) if mask is not None else None

    seq_len = None
    if "seq" in g:
        try:
            s = g["seq"][()]
            if isinstance(s, (bytes, bytearray)):
                s = s.decode("utf-8", errors="replace")
            else:
                s = str(s)
            seq_len = len(s)
        except Exception:
            seq_len = None

    aa_len = _as_py(g.attrs.get("aa_len", None))
    total_tokens = _as_py(g.attrs.get("total_tokens", None))
    d_model = _as_py(g.attrs.get("d_model", None))
    is_chunked = _as_py(g.attrs.get("is_chunked", None))
    chunk_index = _as_py(g.attrs.get("chunk_index", None))
    chunks_total = _as_py(g.attrs.get("chunks_total", None))
    orig_aa_start = _as_py(g.attrs.get("orig_aa_start", None))
    orig_aa_end = _as_py(g.attrs.get("orig_aa_end", None))
    bos_index = _as_py(g.attrs.get("bos_index", None))
    eos_index = _as_py(g.attrs.get("eos_index", None))
    esm_model = _as_py(g.attrs.get("esm_model", ""))
    esm_url = _as_py(g.attrs.get("esm_url", ""))

    mask_sum = None
    if mask is not None:
        try:
            m = mask[()]
            mask_sum = int(np.sum(m.astype(np.int64)))
        except Exception:
            mask_sum = None

    print(f"--- {key}")
    print(f"  emb:  shape={emb_shape} dtype={emb_dtype}")
    print(f"  mask: shape={mask_shape} dtype={mask_dtype} sum={mask_sum}")
    print(f"  seq_len={seq_len}  aa_len={aa_len}  total_tokens={total_tokens}  d_model={d_model}")
    print(f"  chunking: is_chunked={is_chunked}  {chunk_index}/{chunks_total}  orig=[{orig_aa_start},{orig_aa_end})")
    print(f"  special: bos_index={bos_index} eos_index={eos_index}")
    print(f"  esm: model={esm_model}  url={esm_url}")


def main():
    ap = argparse.ArgumentParser(description="Peek & sanity-check embeddings H5")
    ap.add_argument("--h5", required=True, help="embeddings.h5")
    ap.add_argument("--n", type=int, default=5, help="number of random items to show")
    ap.add_argument("--seed", type=int, default=0, help="random seed for sampling items")
    ap.add_argument("--keys", default=None,
                    help="comma-separated explicit item keys to inspect instead of random sampling")
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero if any sampled item has problems")
    ap.add_argument("--scan", action="store_true",
                    help="scan ALL items and report aggregate problem counts (can be slow)")
    args = ap.parse_args()

    with h5py.File(args.h5, "r") as f:
        print(f"H5: {args.h5}")
        print(f"Top-level keys: {list(f.keys())}")

        if "items" not in f:
            raise SystemExit("ERROR: missing top-level group /items")

        items = f["items"]
        all_keys = list(items.keys())
        print(f"/items: {len(all_keys)} groups")

        # manifest info
        if "manifest" in f:
            mf = f["manifest"]
            print(f"/manifest keys: {list(mf.keys())}")
            if "rows" in mf:
                rows = mf["rows"]
                header = rows.attrs.get("header", None)
                if isinstance(header, (bytes, bytearray)):
                    header = header.decode("utf-8", errors="replace")
                print(f"/manifest/rows: n={rows.shape[0]} header={header}")

        # Choose keys
        if args.keys:
            keys = [k.strip() for k in args.keys.split(",") if k.strip()]
        else:
            keys = list_some(all_keys, args.n, args.seed)

        print(f"\nInspecting {len(keys)} items:")
        any_problem = False
        for k in keys:
            if k not in items:
                print(f"--- {k}\n  [MISSING] key not found in /items")
                any_problem = True
                continue
            g = items[k]
            print_item_summary(g, k)
            problems = check_item(g, k, strict=False)
            if problems:
                any_problem = True
                print("  Problems:")
                for p in problems:
                    print("   ", p)
            else:
                print("  OK")

        # Optional full scan
        if args.scan:
            print("\nScanning ALL items for schema/consistency problems...")
            counts: Dict[str, int] = {}
            for i, k in enumerate(all_keys, 1):
                probs = check_item(items[k], k, strict=False)
                for p in probs:
                    tag = p.split("]")[0] + "]"  # e.g. "[MISSING_ATTR]"
                    counts[tag] = counts.get(tag, 0) + 1
            if counts:
                print("Problem counts:")
                for tag, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
                    print(f"  {tag}: {c}")
            else:
                print("No problems detected in full scan.")

            any_problem = any_problem or bool(counts)

        if args.strict and any_problem:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
