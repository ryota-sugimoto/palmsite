#!/usr/bin/env python
"""
Plot residue-wise final attention weights and the corresponding Gaussian
(mu_attn, sigma_attn) from a PalmSite attention JSON file.

Focus: *final attention module only*.
Anchor-based (mu, sigma from w_anchor) are ignored here and can be
plotted separately for supplementary figures if needed.

Expected JSON format (produced by predict.py with --attn-json):

{
  "chunk_id_1": {
    "L": 932,
    "orig_start": 0,
    "orig_len": 932,
    "mu": 0.4567,           # anchor (not used here)
    "sigma": 0.0834,        # anchor (not used here)
    "mu_attn": 0.4721,      # final attention
    "sigma_attn": 0.0712,   # final attention
    "S_norm": 0.40,         # optional, 0–1 (final attention span)
    "E_norm": 0.65,
    "S_idx": 377,           # optional, 0-based indices
    "E_idx": 611,
    "P": 0.98,
    "w": [...],             # length L, final attention weights
    "abs_pos": [...]        # length L, absolute 0-based positions
  },
  ...
}

Usage:

python plot_attention.py \
    --json nsp12_attention.json \
    --chunk-id nsp12_chunk_000 \
    --out figs/nsp12_attention.png \
    --k-sigma 2.0

If --chunk-id is omitted, the first entry in the JSON is used.
"""

import argparse
import json
import os
from typing import Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_entry(data: Dict[str, Any], chunk_id: str | None) -> Tuple[str, Dict[str, Any]]:
    if chunk_id is not None:
        if chunk_id not in data:
            raise KeyError(
                f"chunk_id '{chunk_id}' not found in JSON "
                f"(available keys example: {list(data.keys())[:5]} ...)"
            )
        return chunk_id, data[chunk_id]
    # default: first key
    first_key = next(iter(data.keys()))
    return first_key, data[first_key]


def derive_span_from_mu_sigma(
    mu_attn: float, sigma_attn: float, L: int, k_sigma: float
) -> Tuple[int, int]:
    """
    Derive S_idx/E_idx from mu_attn, sigma_attn exactly as the model does:

        S = clamp(mu_attn - k * sigma_attn, 0, 1)
        E = clamp(mu_attn + k * sigma_attn, 0, 1)
        idx = round(S * (L-1)), round(E * (L-1))   (end-inclusive)

    Returns:
        (S_idx, E_idx) as 0-based indices in [0, L-1]
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


def plot_attention(
    chunk_id: str,
    entry: Dict[str, Any],
    out_path: str,
    k_sigma: float = 2.0,
    one_based: bool = True,
) -> None:
    """
    Make an X-Y plot using *final attention* information only:

      X: absolute residue index (0- or 1-based)
      Y1: normalized attention weights w (final attention)
      Y2: normalized Gaussian from (mu_attn, sigma_attn)

    Additionally:
      - Shaded grey band = predicted catalytic span from S_idx/E_idx
        (same span that goes into GFF).
    """

    L = int(entry["L"])
    orig_start = int(entry["orig_start"])
    orig_len = int(entry["orig_len"])

    # final-attention parameters (fallback to anchor if mu_attn/sigma_attn missing)
    mu_attn = float(entry.get("mu_attn", entry.get("mu", 0.0)))
    sigma_attn = float(entry.get("sigma_attn", entry.get("sigma", 0.0)))

    w = np.asarray(entry["w"], dtype=float)
    abs_pos = np.asarray(entry["abs_pos"], dtype=int)

    if len(w) != L or len(abs_pos) != L:
        raise ValueError(
            f"Inconsistent L ({L}) vs len(w)={len(w)} vs len(abs_pos)={len(abs_pos)}"
        )

    if L <= 1:
        raise ValueError("Sequence length L <= 1, cannot plot meaningful curve.")

    # X-axis: actual residue positions
    x = abs_pos + (1 if one_based else 0)

    # Normalize attention weights to [0,1] by max (for shape comparison)
    w_max = float(w.max()) if w.size > 0 else 1.0
    if w_max <= 0.0:
        w_norm = np.zeros_like(w)
    else:
        w_norm = w / w_max

    # Reconstruct Gaussian over discrete positions from (mu_attn, sigma_attn)
    idx = np.arange(L, dtype=float)
    denom = float(max(L - 1, 1))
    pos_norm = idx / denom
    g = np.exp(-0.5 * ((pos_norm - mu_attn) / max(sigma_attn, 1e-8)) ** 2)
    g_max = float(g.max()) if g.size > 0 else 1.0
    if g_max <= 0.0:
        g_norm = np.zeros_like(g)
    else:
        g_norm = g / g_max

    # Predicted span indices: prefer S_idx/E_idx from JSON, otherwise derive them
    if "S_idx" in entry and "E_idx" in entry:
        S_idx = int(entry["S_idx"])
        E_idx = int(entry["E_idx"])
        S_idx = max(0, min(S_idx, L - 1))
        E_idx = max(0, min(E_idx, L - 1))
        if E_idx < S_idx:
            S_idx, E_idx = E_idx, S_idx
    else:
        S_idx, E_idx = derive_span_from_mu_sigma(mu_attn, sigma_attn, L, k_sigma)

    # Convert span indices to plotting coordinates (absolute positions)
    if one_based:
        S_plot = orig_start + S_idx + 1
        E_plot = orig_start + E_idx + 1
    else:
        S_plot = orig_start + S_idx
        E_plot = orig_start + E_idx

    # Prepare figure
    plt.figure(figsize=(8, 4))

    # Plot attention weights
    plt.plot(
        x,
        w_norm,
        label="Final attention (w, normalized)",
        linewidth=1.5,
        color="tab:blue",
    )

    # Plot Gaussian shape (from final attention)
    plt.plot(
        x,
        g_norm,
        linestyle="--",
        label="Gaussian from (μ_attn, σ_attn), normalized",
        linewidth=1.5,
        color="tab:orange",
    )

    # Shade predicted span (S_idx..E_idx)
    plt.axvspan(
        S_plot,
        E_plot,
        color="grey",
        alpha=0.15,
        label="Predicted catalytic span (final attention)",
    )

    # Mark μ_attn as a vertical line
    mu_idx = mu_attn * denom
    mu_abs = orig_start + mu_idx
    if one_based:
        mu_plot = mu_abs + 1.0
    else:
        mu_plot = mu_abs
    plt.axvline(
        mu_plot,
        color="black",
        linestyle=":",
        linewidth=1.0,
        label="μ_attn (center)",
    )

    # Labels and formatting
    x_label = "Residue position (1-based)" if one_based else "Residue position (0-based)"
    plt.xlabel(x_label)
    plt.ylabel("Normalized weight")
    plt.title(
        f"Attention profile for {chunk_id}\n"
        f"(L={L}, orig_start={orig_start}, orig_len={orig_len})"
    )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend(fontsize=8, loc="upper right")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[plot] Saved plot for chunk_id='{chunk_id}' to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot residue-wise final attention and Gaussian from PalmSite attention JSON."
    )
    p.add_argument(
        "--json",
        required=True,
        help="JSON file produced by predict.py with --attn-json.",
    )
    p.add_argument(
        "--chunk-id",
        default=None,
        help="Chunk ID to plot. If omitted, the first entry in the JSON is used.",
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output image path (e.g. figs/nsp12_attention.png).",
    )
    p.add_argument(
        "--k-sigma",
        type=float,
        default=2.0,
        help="Number of sigmas used for the span when S_idx/E_idx are not present (default: 2.0).",
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

