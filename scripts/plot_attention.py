#!/usr/bin/env python
"""
Plot residue-wise attention weights and the corresponding Gaussian
(mu, sigma) from a PalmSite attention JSON file.

Expected JSON format (one produced by predict.py with --attn-json):

{
  "chunk_id_1": {
    "L": 932,
    "orig_start": 0,
    "orig_len": 932,
    "mu": 0.4567,
    "sigma": 0.0834,
    "w": [...],          # length L, final attention weights
    "abs_pos": [...]     # length L, absolute 0-based positions
  },
  "chunk_id_2": { ... },
  ...
}

Usage:

python plot_attention_from_json.py \
    --json nsp12_attention.json \
    --chunk-id nsp12_chunk_000 \
    --out nsp12_attention.png \
    --k-sigma 2.0

If --chunk-id is omitted, the first entry in the JSON is used.
"""

import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_entry(data: Dict[str, Any], chunk_id: str | None) -> tuple[str, Dict[str, Any]]:
    if chunk_id is not None:
        if chunk_id not in data:
            raise KeyError(f"chunk_id '{chunk_id}' not found in JSON (keys: {list(data.keys())[:5]} ...)")
        return chunk_id, data[chunk_id]
    # default: first key
    first_key = next(iter(data.keys()))
    return first_key, data[first_key]


def plot_attention(
    chunk_id: str,
    entry: Dict[str, Any],
    out_path: str,
    k_sigma: float = 2.0,
    one_based: bool = True,
) -> None:
    """
    Make an X-Y plot:
      X: absolute residue index (0- or 1-based)
      Y: normalized attention weights and Gaussian(mu, sigma).

    We reconstruct the Gaussian in the *normalized* coordinate system
    used in the model (0..1 along sequence, end-inclusive), then map
    it back to discrete residue positions.

    - w_norm: w / max(w)
    - g_norm: gaussian / max(gaussian)
    """
    L = int(entry["L"])
    orig_start = int(entry["orig_start"])
    orig_len = int(entry["orig_len"])
    mu = float(entry["mu"])       # in [0,1]
    sigma = float(entry["sigma"]) # in [0,0.5]
    w = np.asarray(entry["w"], dtype=float)
    abs_pos = np.asarray(entry["abs_pos"], dtype=int)

    if len(w) != L or len(abs_pos) != L:
        raise ValueError(f"Inconsistent L ({L}) vs len(w)={len(w)} vs len(abs_pos)={len(abs_pos)}")

    if L <= 1:
        raise ValueError("Sequence length L <= 1, cannot plot meaningful curve.")

    # X-axis: actual residue positions
    # abs_pos are 0-based; for 1-based view, add 1.
    x = abs_pos + (1 if one_based else 0)

    # Normalize attention weights to [0,1] by max (for shape comparison)
    w_max = float(w.max()) if w.size > 0 else 1.0
    if w_max <= 0.0:
        w_norm = np.zeros_like(w)
    else:
        w_norm = w / w_max

    # Reconstruct Gaussian over discrete positions.
    # Model uses end-inclusive normalization: pos_norm = idx / (L-1)
    idx = np.arange(L, dtype=float)
    denom = float(max(L - 1, 1))
    pos_norm = idx / denom
    g = np.exp(-0.5 * ((pos_norm - mu) / max(sigma, 1e-8)) ** 2)
    g_max = float(g.max()) if g.size > 0 else 1.0
    if g_max <= 0.0:
        g_norm = np.zeros_like(g)
    else:
        g_norm = g / g_max

    # Estimate μ and μ ± kσ in absolute residue indices (0-based)
    mu_idx = mu * denom            # 0..(L-1)
    sigma_idx = sigma * denom      # in residues
    mu_abs = orig_start + mu_idx
    left_abs = orig_start + (mu_idx - k_sigma * sigma_idx)
    right_abs = orig_start + (mu_idx + k_sigma * sigma_idx)

    # Prepare figure
    plt.figure(figsize=(8, 4))

    # Plot attention weights
    plt.plot(x, w_norm, label="Final attention (w, normalized)", linewidth=1.5)

    # Plot Gaussian shape
    plt.plot(x, g_norm, linestyle="--", label="Gaussian from (μ, σ), normalized", linewidth=1.5)

    # Shade ±kσ region (in absolute coordinates)
    if sigma > 0:
        # convert to plotting coordinates
        if one_based:
            left_plot = left_abs + 1.0
            right_plot = right_abs + 1.0
            mu_plot = mu_abs + 1.0
        else:
            left_plot = left_abs
            right_plot = right_abs
            mu_plot = mu_abs

        plt.axvspan(left_plot, right_plot, color="grey", alpha=0.15, label=f"μ ± {k_sigma:.1f}σ (approx.)")
        plt.axvline(mu_plot, color="black", linestyle=":", linewidth=1.0, label="μ (center)")

    plt.xlabel("Residue position" + (" (1-based)" if one_based else " (0-based)"))
    plt.ylabel("Normalized weight")
    plt.title(f"Attention profile for {chunk_id}\n(L={L}, orig_start={orig_start}, orig_len={orig_len})")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=8, loc="upper right")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[plot] Saved plot for chunk_id='{chunk_id}' to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot residue-wise attention and Gaussian from PalmSite attention JSON.")
    p.add_argument("--json", required=True, help="JSON file produced by predict.py with --attn-json.")
    p.add_argument("--chunk-id", default=None,
                   help="Chunk ID to plot. If omitted, the first entry in the JSON is used.")
    p.add_argument("--out", required=True, help="Output image path (e.g. figs/nsp12_attention.png).")
    p.add_argument("--k-sigma", type=float, default=2.0,
                   help="Number of sigmas for shaded region around μ (default: 2.0).")
    p.add_argument("--zero-based", action="store_true",
                   help="Use 0-based residue positions on X axis instead of 1-based.")
    return p.parse_args()


def main():
    args = parse_args()
    data = load_json(args.json)
    chunk_id, entry = pick_entry(data, args.chunk_id)
    plot_attention(
        chunk_id=chunk_id,
        entry=entry,
        out_path=args.out,
        k_sigma=float(args.k_sigma),
        one_based=not args.zero_based,
    )


if __name__ == "__main__":
    main()

