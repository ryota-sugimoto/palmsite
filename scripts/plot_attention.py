#!/usr/bin/env python
"""
Plot full-length final attention weights and Gaussian (from mu_attn, sigma_attn)
for a base_id, aggregating over all chunks.

New features:
  - Attention curve color changed from blue to red.
  - Optional palm_annot TSV parsing to overlay catalytic motifs A/B/C.
  - Motifs are shown as translucent spans covering the motif sequence length,
    plus a vertical line and a label at the motif start position.

Aggregation strategy:
  - Attention (w): for each chunk, we add its attention weights w[pos_local]
    into a global full_w[pos_abs] (sum over chunks).
  - Gaussian (g): we compute the Gaussian only for the BEST chunk (highest P)
    and place it into full_g[pos_abs] for that chunk's residues.
  - Then we optionally smooth full_w/full_g and normalize them by their
    total sum. For visualization, we rescale the Gaussian so its maximum
    matches the maximum of the attention curve, so it is clearly visible.

Expected JSON format (produced by predict.py with --attn-json):

{
  "base|chunk_0001_of_0002|aa_000000_000400": {
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
  "base|chunk_0002_of_0002|aa_000400_000800": { ... },
  ...
}

Expected palm_annot TSV format (headerless, one sequence per line):

seq_id<TAB>key=value<TAB>key=value<TAB>...

Relevant keys (preferred / fallback):
  - posA / posB / posC                : motif start positions (typically 1-based)
  - seqA / seqB / seqC                : motif sequences
  - motif_hmm_posA/B/C, pssm_posA/B/C : fallback positions
  - motif_hmm_seqA/B/C, pssm_seqA/B/C : fallback sequences
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import transforms


@dataclass(frozen=True)
class MotifSpec:
    label: str
    start_1based: int
    seq: str

    @property
    def length(self) -> int:
        return max(len(self.seq), 1)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def base_id_from_chunk(cid: str) -> str:
    """Match predict.py: base_id is the prefix before the canonical '|chunk_' marker."""
    if "|chunk_" in cid:
        return cid.split("|chunk_", 1)[0]
    if "_chunk_" in cid:
        return cid.split("_chunk_", 1)[0]
    return cid


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


def choose_base_id(grouped: Dict[str, Dict[str, Any]], desired: Optional[str]) -> str:
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
    Aggregate all chunk entries for this base_id into full-length arrays.

    Behavior:
      - full_w: sum of attention weights over all chunks.
      - full_g: Gaussian only from the best chunk (highest P).
      - span_abs: span (abs_S, abs_E) from the best chunk, 0-based.
      - best_cid: the winner chunk.

    Returns:
        full_w   : float [orig_len]
        full_g   : float [orig_len]
        orig_len : int
        span_abs : (abs_S, abs_E) predicted span (0-based, full-length)
        best_cid : chunk_id that produced the best span (highest P)
    """
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

        w = np.asarray(e["w"], dtype=float)
        if w.shape[0] != L:
            raise ValueError(f"Chunk {cid}: len(w)={w.shape[0]} != L={L}")

        for idx_local in range(L):
            pos_abs = ostart + idx_local
            if 0 <= pos_abs < orig_len:
                full_w[pos_abs] += w[idx_local]

        mu_attn = float(e.get("mu_attn", e.get("mu", 0.0)))
        sigma_attn = float(e.get("sigma_attn", e.get("sigma", 0.0)))
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

    if best_cid:
        e = chunks[best_cid]
        L = int(e["L"])
        ostart = int(e["orig_start"])
        mu_attn = float(e.get("mu_attn", e.get("mu", 0.0)))
        sigma_attn = float(e.get("sigma_attn", e.get("sigma", 0.0)))

        if L > 1:
            idx_local_arr = np.arange(L, dtype=float)
            denom = float(max(L - 1, 1))
            pos_norm = idx_local_arr / denom
            g_chunk = np.exp(
                -0.5 * ((pos_norm - mu_attn) / max(sigma_attn, 1e-8)) ** 2
            )
        else:
            g_chunk = np.zeros((L,), dtype=float)

        for idx_local in range(L):
            pos_abs = ostart + idx_local
            if 0 <= pos_abs < orig_len:
                full_g[pos_abs] = g_chunk[idx_local]

    return full_w, full_g, orig_len, best_span, best_cid


def smooth_1d(arr: np.ndarray, k: int) -> np.ndarray:
    """
    Simple moving-average smoother with window size k.

    - If k <= 1, returns arr unchanged.
    - Uses 'edge' padding so the output has the same length as input.
    """
    if arr.size == 0 or k <= 1:
        return arr
    k = int(k)
    if k < 2:
        return arr
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    kernel = np.ones(k, dtype=float) / float(k)
    return np.convolve(padded, kernel, mode="valid")


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def load_palm_annot_motifs(path: str) -> Dict[str, Dict[str, MotifSpec]]:
    """
    Parse palm_annot TSV into:
      {seq_id: {'A': MotifSpec(...), 'B': ..., 'C': ...}}

    The file is expected to be headerless and tab-delimited, where the first field
    is seq_id and all remaining fields are key=value pairs.
    """
    motif_map: Dict[str, Dict[str, MotifSpec]] = {}

    preferred_pos_keys = {
        "A": ["posA", "motif_hmm_posA", "pssm_posA"],
        "B": ["posB", "motif_hmm_posB", "pssm_posB"],
        "C": ["posC", "motif_hmm_posC", "pssm_posC"],
    }
    preferred_seq_keys = {
        "A": ["seqA", "motif_hmm_seqA", "pssm_seqA"],
        "B": ["seqB", "motif_hmm_seqB", "pssm_seqB"],
        "C": ["seqC", "motif_hmm_seqC", "pssm_seqC"],
    }

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            seq_id = fields[0]
            kv: Dict[str, str] = {}
            for field in fields[1:]:
                if "=" not in field:
                    continue
                key, value = field.split("=", 1)
                kv[key] = value

            per_seq: Dict[str, MotifSpec] = {}
            for motif_label in ("A", "B", "C"):
                pos = None
                seq = ""
                for key in preferred_pos_keys[motif_label]:
                    pos = _safe_int(kv.get(key))
                    if pos is not None:
                        break
                for key in preferred_seq_keys[motif_label]:
                    seq = kv.get(key, "")
                    if seq:
                        break

                if pos is None:
                    continue
                per_seq[motif_label] = MotifSpec(
                    label=motif_label,
                    start_1based=pos,
                    seq=seq,
                )

            if per_seq:
                motif_map[seq_id] = per_seq
            else:
                print(
                    f"[warn] No motif A/B/C positions parsed from line {line_no} "
                    f"for seq_id='{seq_id}'"
                )

    return motif_map


def _motif_plot_coordinates(
    motif: MotifSpec,
    one_based: bool,
    orig_len: int,
) -> Tuple[float, float, float]:
    """
    Convert 1-based motif start to plotting coordinates.

    Returns:
      start_plot, end_plot, line_x
    where end_plot is inclusive span end in the current axis coordinate system.
    """
    start_0based = motif.start_1based - 1
    end_0based = start_0based + motif.length - 1

    start_0based = max(0, min(start_0based, orig_len - 1))
    end_0based = max(0, min(end_0based, orig_len - 1))

    if one_based:
        start_plot = start_0based + 1
        end_plot = end_0based + 1
        line_x = start_plot
    else:
        start_plot = start_0based
        end_plot = end_0based
        line_x = start_plot

    return float(start_plot), float(end_plot), float(line_x)


def plot_full_attention(
    base_id: str,
    full_w: np.ndarray,
    full_g: np.ndarray,
    orig_len: int,
    span_abs: Tuple[int, int],
    best_cid: str,
    out_path: str,
    motifs: Optional[Dict[str, MotifSpec]] = None,
    one_based: bool = True,
    smooth_window: int = 1,
) -> None:
    """Plot full-length attention/Gaussian for a base_id."""

    if orig_len <= 0:
        raise ValueError("orig_len must be > 0")

    x = np.arange(orig_len, dtype=float)
    x_plot = x + 1.0 if one_based else x

    w = smooth_1d(full_w.copy(), smooth_window)
    g = smooth_1d(full_g.copy(), smooth_window)

    w_sum = float(w.sum()) if w.size > 0 else 1.0
    g_sum = float(g.sum()) if g.size > 0 else 1.0
    if w_sum > 0.0:
        w /= w_sum
    if g_sum > 0.0:
        g /= g_sum

    w_max = float(w.max()) if w.size > 0 else 1.0
    g_max = float(g.max()) if g.size > 0 else 1.0
    if g_max > 0 and w_max > 0:
        g *= (w_max / g_max)

    S_abs, E_abs = span_abs
    if one_based:
        S_plot = S_abs + 1
        E_plot = E_abs + 1
    else:
        S_plot = S_abs
        E_plot = E_abs

    fig, ax = plt.subplots(figsize=(10, 4.8))

    ax.plot(
        x_plot,
        w,
        label="Final attention (aggregated, mass-normalized)",
        linewidth=1.8,
        color="tab:red",
    )

    ax.plot(
        x_plot,
        g,
        linestyle="--",
        label="Gaussian (best chunk; mass-normalized, rescaled)",
        linewidth=1.5,
        color="tab:orange",
    )

    ax.axvspan(
        S_plot,
        E_plot,
        color="grey",
        alpha=0.15,
        label=f"Predicted catalytic span (best chunk: {best_cid})",
    )

    if motifs:
        motif_colors = {
            "A": "#b2182b",
            "B": "#7b3294",
            "C": "#2166ac",
        }
        text_transform = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        label_y = {"A": 0.96, "B": 0.90, "C": 0.96}
        for motif_label in ("A", "B", "C"):
            motif = motifs.get(motif_label)
            if motif is None:
                continue
            start_plot, end_plot, line_x = _motif_plot_coordinates(
                motif=motif,
                one_based=one_based,
                orig_len=orig_len,
            )
            color = motif_colors[motif_label]
            ax.axvspan(
                start_plot,
                end_plot,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
            ax.axvline(
                line_x,
                color=color,
                linestyle=":",
                linewidth=1.4,
                label=f"Motif {motif_label} ({motif.start_1based})",
            )
            ax.text(
                line_x,
                label_y[motif_label],
                f"{motif_label}:{motif.start_1based}",
                transform=text_transform,
                color=color,
                rotation=90,
                ha="center",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.8},
            )

    x_label = "Residue position (1-based)" if one_based else "Residue position (0-based)"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Normalized weight")
    ax.set_title(
        f"Full-length attention for {base_id}\n"
        f"(orig_len={orig_len}, best_chunk={best_cid}, smooth_window={smooth_window})"
    )
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        f"[plot] Saved full-length attention plot for base_id='{base_id}' "
        f"to: {out_path} (smooth_window={smooth_window})"
    )


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
            "base_id is defined as the prefix before the canonical '|chunk_' marker in chunk_id."
        ),
    )
    p.add_argument(
        "--out",
        required=True,
        help="Output image path (e.g. figs/attention_full.png).",
    )
    p.add_argument(
        "--palm-annot",
        default=None,
        help=(
            "Optional palm_annot TSV to overlay motif A/B/C positions. "
            "The seq_id must match the selected base_id."
        ),
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
    p.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help=(
            "Window size (in residues) for moving-average smoothing of curves. "
            "1 = no smoothing (default)."
        ),
    )
    return p.parse_args()


def main() -> None:
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

    motifs = None
    if args.palm_annot:
        motif_map = load_palm_annot_motifs(args.palm_annot)
        motifs = motif_map.get(base_id)
        if motifs is None:
            available = list(motif_map.keys())[:5]
            print(
                f"[warn] base_id '{base_id}' was not found in palm_annot TSV. "
                f"Available examples: {available}"
            )
        else:
            present = ", ".join(sorted(motifs.keys()))
            print(f"[motif] Loaded motifs for {base_id}: {present}")

    plot_full_attention(
        base_id=base_id,
        full_w=full_w,
        full_g=full_g,
        orig_len=orig_len,
        span_abs=span_abs,
        best_cid=best_cid,
        out_path=args.out,
        motifs=motifs,
        one_based=not args.zero_based,
        smooth_window=int(args.smooth_window),
    )


if __name__ == "__main__":
    main()

