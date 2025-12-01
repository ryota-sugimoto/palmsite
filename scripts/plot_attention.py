#!/usr/bin/env python
"""
Plot full-length final attention weights and Gaussian (from mu_attn, sigma_attn)
for a base_id, aggregating over all chunks.

Aggregation strategy:
  - For each chunk, we add its attention weights w[pos_local] into a global
    full_w[pos_abs] (sum over chunks).
  - Likewise, we add the chunk's Gaussian g_chunk[pos_local] into full_g[pos_abs].
  - Then we normalize full_w and full_g by their global maxima for plotting.

This emphasizes residues that are consistently attended across overlapping chunks
(e.g., a true catalytic palm) and downweights isolated spikes from a single chunk.

Expected JSON format (produced by predict.py with --attn-json):

{
  "base_chunk_0001_of_0002": {
    "L": 400,
    "orig_start": 0,
    "orig_len": 944,
    "mu_attn": ...,
    "sigma_attn": ...,
    "S_idx": ...,
    "E_idx": ...,
    "P": ...,
    "w": [...],
    "abs_pos": [...]
  },
  "base_chunk_0002_of_0002": { ... },
  ...
}

This script:
  - Groups entries by base_id (prefix before "_chunk_" if present).
  - Aggregates over all chunks for that base_id.
  - Chooses the highest-P chunk to define the predicted catalytic span
    (same logic as GFF in predict.py).
  - Plots:
      X: full-length residue index (0- or 1-based)
      Y1: aggregated final attention (w), normalized
      Y2: aggregated Gaussian, normalized
      Grey band: predicted catalytic span (from best chunk)
"""

import argparse
import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def base_id_from_chunk(cid: str) -> str:
    """Match predict.py: base_id is prefix before '_chunk_'."""
    return cid.split("_chunk_")[0] if "_chunk_" in cid else cid


def group_by_base_id(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for cid, entry in data.items():
        bid = base_id_from_chunk(cid)
        grouped.setdefault(bid, {})[cid] = entry
    return grouped


def derive_span_from_mu_sigma(
    mu_attn: float, sigma_attn: float, L: int, k_sigma: float
) -> Tuple[int, int]:
    """
    Derive S_idx/E_idx from mu_attn, sigma_attn exactly as the model does:

        S = clamp(mu_attn - k * sigma_attn, 0, 1)
        E = clamp(mu_attn + k * sigma_attn, 0, 1)
        idx = round(S * (L-1)), round(E * (L-1))   (end-inclusive)
    """
    if L <= 1:
        return 0, 0
    denom = float(max(L - 1, 1))
    S_norm = max(0.0, min(1.0, mu_attn - k_sigma * sigma_attn))
    E_norm = max(0.0, min(1.0, mu_attn + k_sigma * sigma_attn))
    S_idx = int(round(S_norm * denom))
    E_idx = int(round(E_norm * denom))
    S_idx = max(0, min(S_idx, L - 1))
    E_idx = max(0, min(E_idx, L - 1))
    if E_idx < S_idx:
        S_idx, E_idx = E_idx, S_idx
    return S_idx, E_idx


def choose_base_id(grouped: Dict[str, Dict[str, Any]], desired: str | None) -> str:
    if desired is not None:
        if desired not in grouped:
            raise KeyError(
                f"base_id '{desired}' not found. "
                f"Available examples: {list(grouped.keys())[:5]} ..."
            )
        return desired
    return next(iter(grouped.keys()))


def aggregate_full_length_for_base(
    base_id: str,
    chunks: Dict[str, Any],
    k_sigma: float,
) -> Tuple[np.ndarray, np.ndarray, int, Tuple[int, int], str]:
    """
    Aggregate all chunk entries for this base_id into full-length arrays
    by summing attention/gaussian across chunks.

    Returns:
        full_w   : float [orig_len], aggregated final attention (sum over chunks)
        full_g   : float [orig_len], aggregated Gaussian (sum over chunks)
        orig_len : int
        span_abs : (abs_S, abs_E) predicted span (0-based, full-length)
        best_cid : chunk_id that produced the best span (highest P)
    """
    # Determine orig_len as max over chunks
    orig_len = 0
    for cid, e in chunks.items():
        L = int(e["L"])
        ostart = int(e["orig_start"])
        olen = int(e.get("orig_len", ostart + L))
        orig_len = max(orig_len, olen, ostart + L)
    if orig_len <= 0:
        raise ValueError(f"Could not determine orig_len for base_id={base_id}")

    full_w = np.zeros(orig_len, dtype=float)
    full_g = np.zeros(orig_len, dtype=float)

    best_P = -1.0
    best_span: Tuple[int, int] = (0, 0)
    best_cid = ""

    for cid, e in chunks.items():
        L = int(e["L"])
        ostart = int(e["orig_start"])
        olen = int(e.get("orig_len", orig_len))

        w = np.asarray(e["w"], dtype=float)
        if w.shape[0] != L:
            raise ValueError(f"Chunk {cid}: len(w)={w.shape[0]} != L={L}")

        # final-attention parameters
        mu_attn = float(e.get("mu_attn", e.get("mu", 0.0)))
        sigma_attn = float(e.get("sigma_attn", e.get("sigma", 0.0)))

        # --- Aggregate attention: sum over chunks ---
        for idx_local in range(L):
            pos_abs = ostart + idx_local
            if 0 <= pos_abs < orig_len:
                full_w[pos_abs] += w[idx_local]

        # --- Gaussian for this chunk (normalized per chunk, then summed) ---
        if L > 1:
            idx_local_arr = np.arange(L, dtype=float)
            denom = float(max(L - 1, 1))
            pos_norm = idx_local_arr / denom
            g_chunk = np.exp(-0.5 * ((pos_norm - mu_attn) / max(sigma_attn, 1e-8)) ** 2)
            gmax = float(g_chunk.max()) if g_chunk.size > 0 else 1.0
            if gmax > 0.0:
                g_chunk = g_chunk / gmax
        else:
            g_chunk = np.zeros((L,), dtype=float)

        for idx_local in range(L):
            pos_abs = ostart + idx_local
            if 0 <= pos_abs < orig_len:
                full_g[pos_abs] += g_chunk[idx_local]

        # --- Span / P for best-chunk selection (GFF-style) ---
        P = float(e.get("P", 0.0))

        if "S_idx" in e and "E_idx" in e:
            S_idx = int(e["S_idx"])
            E_idx = int(e["E_idx"])
            S_idx = max(0, min(S_idx, L - 1))
            E_idx = max(0, min(E_idx, L - 1))
            if E_idx < S_idx:
                S_idx, E_idx = E_idx, S_idx
        else:
            S_idx, E_idx = derive_span_from_mu_sigma(mu_attn, sigma_attn, L, k_sigma)

        abs_S = ostart + S_idx
        abs_E = ostart + E_idx
        abs_S = max(0, min(abs_S, orig_len - 1))
        abs_E = max(0, min(abs_E, orig_len - 1))
        if abs_E < abs_S:
            abs_S, abs_E = abs_E, abs_S

        if P > best_P:
            best_P = P
            best_span = (abs_S, abs_E)
            best_cid = cid

    return full_w, full_g, orig_len, best_span, best_cid


def plot_full_attention(
    base_id: str,
    full_w: np.ndarray,
    full_g: np.ndarray,
    orig_len: int,
    span_abs: Tuple[int, int],
    best_cid: str,
    out_path: str,
    one_based: bool = True,
) -> None:
    """Plot full-length attention/Gaussian for a base_id."""

    if orig_len <= 0:
        raise ValueError("orig_len must be > 0")

    x = np.arange(orig_len, dtype=float)
    x_plot = x + 1.0 if one_based else x

    # Normalize attention and Gaussian
    w = full_w.copy()
    g = full_g.copy()
    w_max = float(w.max()) if w.size > 0 else 1.0
    g_max = float(g.max()) if g.size > 0 else 1.0
    if w_max > 0.0:
        w /= w_max
    if g_max > 0.0:
        g /= g_max

    S_abs, E_abs = span_abs
    if one_based:
        S_plot = S_abs + 1
        E_plot = E_abs + 1
    else:
        S_plot = S_abs
        E_plot = E_abs

    plt.figure(figsize=(10, 4))

    plt.plot(
        x_plot,
        w,
        label="Final attention (aggregated, normalized)",
        linewidth=1.5,
        color="tab:blue",
    )

    plt.plot(
        x_plot,
        g,
        linestyle="--",
        label="Gaussian (aggregated, normalized)",
        linewidth=1.5,
        color="tab:orange",
    )

    plt.axvspan(
        S_plot,
        E_plot,
        color="grey",
        alpha=0.15,
        label=f"Predicted catalytic span (best chunk: {best_cid})",
    )

    x_label = "Residue position (1-based)" if one_based else "Residue position (0-based)"
    plt.xlabel(x_label)
    plt.ylabel("Normalized weight")
    plt.title(
        f"Full-length attention for {base_id}\n"
        f"(orig_len={orig_len}, best_chunk={best_cid})"
    )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=8, loc="upper right")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[plot] Saved full-length attention plot for base_id='{base_id}' to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot full-length final attention and Gaussian from PalmSite attention JSON."
    )
    p.add_argument(
        "--json",
        required=True,
        help="JSON file produced by predict.py with --attn-json.",
    )
    p.add_argument(
        "--base-id",
        default=None,
        help=(
            "Base sequence ID to plot. If omitted, the base_id of the first JSON entry is used. "
            "base_id is defined as the prefix before '_chunk_' in chunk_id."
        ),
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output image path (e.g. figs/attention_full.png).",
    )
    p.add_argument(
        "--k-sigma",
        type=float,
        default=2.0,
        help=(
            "Number of sigmas used when deriving span from mu_attn/sigma_attn if "
            "S_idx/E_idx are not available (default: 2.0)."
        ),
    )
    p.add_argument(
        "--zero-based",
        action="store_true",
        help="Use 0-based residue positions on X axis instead of 1-based.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    data = load_json(args.json)
    grouped = group_by_base_id(data)
    base_id = choose_base_id(grouped, args.base_id)
    chunks = grouped[base_id]

    full_w, full_g, orig_len, span_abs, best_cid = aggregate_full_length_for_base(
        base_id=base_id,
        chunks=chunks,
        k_sigma=float(args.k_sigma),
    )

    plot_full_attention(
        base_id=base_id,
        full_w=full_w,
        full_g=full_g,
        orig_len=orig_len,
        span_abs=span_abs,
        best_cid=best_cid,
        out_path=args.out,
        one_based=not args.zero_based,
    )


if __name__ == "__main__":
    main()

