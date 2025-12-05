#!/usr/bin/env python3
"""
Compute per-parent-sequence IoU between PalmSite predicted spans (GFF3)
and ground-truth spans from a label TSV that uses chunk_ids, and visualize:

1) Distribution of IoU (violin / seaborn swarm / both)
2) Scatter plots:
   - IoU vs full sequence length
   - IoU vs truth span length
   - IoU vs predicted span length
   - IoU vs predicted/true span length ratio

Key assumptions (matches your labels_only_positives.tsv):
- GFF seqid = parent sequence ID (e.g. "MGYP000911387321")
- Label file has column "chunk_id" with format like:
    "MGYP000911387321_chunk_0001_of_0001_aa_000000_000260"
    "NP_041870.2_chunk_0002_of_0002_aa_001094_003094"
- span_start_aa / span_end_aa are CHUNK-LOCAL coordinates
  (for you: 1-based inclusive within the chunk).
- The "_aa_SSSSSS_EEEEEE" suffix encodes the chunk's position in the parent
  as 0-based half-open indices [S, E), so E is the parent full length if
  this chunk is last in the sequence.
- Optional ids_file contains chunk_ids or parent IDs; we map chunk_ids to
  parent IDs for filtering.

Important behavior:
- Only parents that have ground-truth spans AND are present in the GFF
  are evaluated and plotted.
- Others are skipped (counted in a summary message).

Outputs:
- <out_prefix>.per_sequence_iou.csv       (LF line endings)
- optional TSV via --per_sequence_tsv     (LF line endings)
- <out_prefix>.iou_<plot_type>.png/.pdf   (IoU distribution)
- <out_prefix>.iou_vs_full_len.png/.pdf   (IoU vs full seq len)
- <out_prefix>.iou_vs_truth_len.png/.pdf  (IoU vs truth span len)
- <out_prefix>.iou_vs_pred_len.png/.pdf   (IoU vs predicted span len)
- <out_prefix>.iou_vs_span_ratio.png/.pdf (IoU vs pred/true ratio)
"""

import argparse
import csv
import gzip
import re
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

Interval = Tuple[int, int]  # 1-based inclusive (start, end)


def smart_open(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode, encoding="utf-8")


# ---------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------

def normalize_intervals(intervals: List[Interval]) -> List[Interval]:
    if not intervals:
        return []
    ints = [(min(a, b), max(a, b)) for a, b in intervals]
    ints.sort(key=lambda x: (x[0], x[1]))
    merged = [ints[0]]
    for s, e in ints[1:]:
        ps, pe = merged[-1]
        if s <= pe + 1:  # overlap/adjacent (inclusive)
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def intervals_length(intervals: List[Interval]) -> int:
    return sum(e - s + 1 for s, e in intervals)


def intersect_intervals(a: List[Interval], b: List[Interval]) -> List[Interval]:
    a = normalize_intervals(a)
    b = normalize_intervals(b)
    i = j = 0
    out: List[Interval] = []
    while i < len(a) and j < len(b):
        s1, e1 = a[i]
        s2, e2 = b[j]
        s = max(s1, s2)
        e = min(e1, e2)
        if s <= e:
            out.append((s, e))
        if e1 < e2:
            i += 1
        else:
            j += 1
    return normalize_intervals(out)


def iou(pred: List[Interval], truth: List[Interval]) -> float:
    pred_n = normalize_intervals(pred)
    truth_n = normalize_intervals(truth)
    if not pred_n and not truth_n:
        return 1.0
    if not pred_n or not truth_n:
        return 0.0
    inter = intersect_intervals(pred_n, truth_n)
    union = normalize_intervals(pred_n + truth_n)
    il = intervals_length(inter)
    ul = intervals_length(union)
    return il / ul if ul > 0 else 0.0


# ---------------------------------------------------------------------
# ID / coordinate helpers
# ---------------------------------------------------------------------

_CHUNK_RE = re.compile(
    r'^(?P<parent>.+)_chunk_(?P<chunk>\d+)_of_(?P<total>\d+)_aa_(?P<s>\d{6})_(?P<e>\d{6})$'
)


def parse_chunk_id(chunk_id: str):
    """
    Parse a chunk_id like:
      "NP_041870.2_chunk_0002_of_0002_aa_001094_003094"

    Returns:
      parent_id, chunk_index, chunks_total, s_chunk, e_chunk
    where s_chunk/e_chunk are 0-based half-open indices in the parent.
    """
    m = _CHUNK_RE.match(chunk_id)
    if not m:
        raise ValueError(f"chunk_id does not match expected pattern: {chunk_id}")
    parent = m.group("parent")
    chunk_index = int(m.group("chunk"))
    chunks_total = int(m.group("total"))
    s_chunk = int(m.group("s"))
    e_chunk = int(m.group("e"))
    return parent, chunk_index, chunks_total, s_chunk, e_chunk


def parent_id_from_chunk(chunk_id: str) -> str:
    """
    Legacy helper: map a chunk_id to parent sequence ID without detailed parsing.
    """
    if "|chunk_" in chunk_id:
        return chunk_id.split("|chunk_", 1)[0]
    if "_chunk_" in chunk_id:
        return chunk_id.split("_chunk_", 1)[0]
    return chunk_id


def parse_id_list(ids_path: str) -> Set[str]:
    """
    Read an ID list file.

    Lines can contain chunk_ids or parent IDs.
    We convert chunk_ids to parent IDs for filtering.
    """
    out: Set[str] = set()
    with smart_open(ids_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                parent, *_ = parse_chunk_id(line)
            except ValueError:
                parent = parent_id_from_chunk(line)
            out.add(parent)
    return out


def coerce_int(x: str) -> Optional[int]:
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "nan":
        return None
    try:
        return int(float(x))
    except ValueError:
        return None


def convert_coords_local_to_1based(start: int, end: int, coord_mode: str) -> Interval:
    """
    Convert LOCAL chunk coordinates to 1-based inclusive chunk coordinates.

    coord_mode describes how span_start_aa/span_end_aa are stored:

      - 1based_inclusive: [start, end] already 1-based inclusive
      - 0based_inclusive: [start, end] 0-based inclusive      -> +1 both
      - 0based_halfopen:  [start, end) 0-based half-open      -> start+1, end
    """
    if coord_mode == "1based_inclusive":
        s, e = start, end
    elif coord_mode == "0based_inclusive":
        s, e = start + 1, end + 1
    elif coord_mode == "0based_halfopen":
        s, e = start + 1, end
    else:
        raise ValueError(f"Unknown coord_mode: {coord_mode}")

    if s > e:
        s, e = e, s
    return s, e


# ---------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------

def parse_gff_predictions(gff_path: str, feature_type: Optional[str] = None) -> Dict[str, List[Interval]]:
    """
    GFF3 -> {seqid (parent): [ (start,end), ... ] }, using 1-based inclusive coords.
    """
    pred: Dict[str, List[Interval]] = {}
    with smart_open(gff_path, "rt") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                raise ValueError(f"GFF line does not have 9 columns:\n{line}")
            seqid, source, ftype, start, end, score, strand, phase, attrs = parts
            if feature_type is not None and ftype != feature_type:
                continue
            try:
                s = int(start)
                e = int(end)
            except ValueError:
                continue
            pred.setdefault(seqid, []).append((s, e))
    for k in list(pred.keys()):
        pred[k] = normalize_intervals(pred[k])
    return pred


def parse_label_truth_spans_parent(
    label_tsv: str,
    require_use_span: bool = True,
    positive_labels: Optional[str] = None,
    coord_mode: str = "1based_inclusive",
) -> Tuple[Dict[str, List[Interval]], Dict[str, int]]:
    """
    Read label TSV (with chunk_id) and build:

      truth: { parent_seqid: [ (start,end), ... ] } (merged per parent)
      parent_len: { parent_seqid: full_sequence_length }

    Assumes columns: chunk_id, label, use_span, span_start_aa, span_end_aa.

    IMPORTANT:
      - span_start_aa/span_end_aa are CHUNK-LOCAL coordinates.
      - Chunk offsets (0-based half-open) come from chunk_id suffix "_aa_SSSSSS_EEEEEE".
      - Full parent length is max(e_chunk) over all chunks for that parent.
    """
    pos_set = None
    if positive_labels:
        pos_set = {s.strip() for s in positive_labels.split(",") if s.strip()}

    truth: Dict[str, List[Interval]] = {}
    parent_len: Dict[str, int] = {}

    with smart_open(label_tsv, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"chunk_id", "label", "use_span", "span_start_aa", "span_end_aa"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Label file missing required columns: {sorted(missing)}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            chunk_id_raw = row["chunk_id"].strip()
            parent_id, chunk_idx, chunk_total, s_chunk, e_chunk = parse_chunk_id(chunk_id_raw)

            # Track full parent length = max e_chunk (0-based half-open)
            parent_len[parent_id] = max(parent_len.get(parent_id, 0), e_chunk)

            lab = row["label"].strip()
            use_span = coerce_int(row.get("use_span", ""))

            s0 = coerce_int(row.get("span_start_aa", ""))
            e0 = coerce_int(row.get("span_end_aa", ""))

            if pos_set is not None and lab not in pos_set:
                continue
            if require_use_span and (use_span is None or use_span == 0):
                continue
            if s0 is None or e0 is None:
                continue

            # Convert chunk-local coords -> chunk 1-based inclusive
            loc_s, loc_e = convert_coords_local_to_1based(s0, e0, coord_mode)

            # Map to parent 1-based inclusive:
            # chunk covers parent positions [s_chunk, e_chunk) 0-based
            # so parent 1-based start of chunk = s_chunk + 1
            parent_s = s_chunk + loc_s
            parent_e = s_chunk + loc_e
            if parent_s > parent_e:
                parent_s, parent_e = parent_e, parent_s

            spans = truth.setdefault(parent_id, [])
            spans.append((parent_s, parent_e))

    # Normalize (union) spans per parent
    for parent_id, spans in list(truth.items()):
        truth[parent_id] = normalize_intervals(spans)

    return truth, parent_len


def intervals_to_str(intervals: List[Interval]) -> str:
    return ",".join(f"{s}-{e}" for s, e in intervals) if intervals else ""


# ---------------------------------------------------------------------
# Fallback correlation helpers (if SciPy unavailable)
# ---------------------------------------------------------------------

def pearson_corr_np(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(a: np.ndarray) -> np.ndarray:
    """
    Simple rankdata implementation (average rank for ties), 1-based.
    """
    n = a.size
    order = np.argsort(a, kind="mergesort")
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        rank = (i + j + 2) / 2.0  # average rank, 1-based
        ranks[order[i:j + 1]] = rank
        i = j + 1
    return ranks


def spearman_corr_np(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_violin(values: List[float], title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    sns.set(style="whitegrid")
    ax.violinplot([values], showmeans=False, showmedians=True, showextrema=True)
    ax.set_ylim(0, 1)
    ax.set_xticks([1])
    ax.set_xticklabels(["IoU"])
    ax.set_ylabel("Intersection over Union (IoU)")
    ax.set_title(title, fontsize=8, pad=10)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    return fig, ax


def plot_swarm(values: List[float], title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    sns.set(style="whitegrid")
    x_vals = ["IoU"] * len(values)
    sns.swarmplot(x=x_vals, y=values, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Intersection over Union (IoU)")
    ax.set_title(title, fontsize=8, pad=10)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    return fig, ax


def plot_both(values: List[float], title: str, subtitle: str):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.8), sharey=True)
    ax_v, ax_s = axes

    ax_v.violinplot([values], showmeans=False, showmedians=True, showextrema=True)
    ax_v.set_ylim(0, 1)
    ax_v.set_xticks([1])
    ax_v.set_xticklabels(["IoU"])
    ax_v.set_ylabel("Intersection over Union (IoU)")
    ax_v.set_title("Violin", fontsize=8)
    ax_v.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    x_vals = ["IoU"] * len(values)
    sns.swarmplot(x=x_vals, y=values, ax=ax_s)
    ax_s.set_ylim(0, 1)
    ax_s.set_xlabel("")
    ax_s.set_xticklabels(["IoU"])
    ax_s.set_title("Swarm", fontsize=8)
    ax_s.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

    fig.suptitle(title, fontsize=5)
    fig.text(0.5, 0.02, subtitle, ha="center", va="bottom", fontsize=5)
    fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.92])
    return fig, axes


def plot_scatter(x: np.ndarray, y: np.ndarray,
                 xlabel: str, ylabel: str,
                 title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.set(style="whitegrid")
    ax.scatter(x, y, alpha=0.7, edgecolors="none", s=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=8, pad=10)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=5)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gff", required=True, help="PalmSite prediction GFF3 (.gz ok); seqid = parent sequence ID")
    ap.add_argument("--labels", required=True, help="Label TSV with chunk_id + span_start_aa/span_end_aa (.gz ok)")
    ap.add_argument("--out_prefix", default="figB", help="Output prefix for CSV and plots")

    ap.add_argument(
        "--per_sequence_tsv",
        default=None,
        help="Optional path to write per-sequence IoU table as TSV "
             "(if not set, only CSV is written).",
    )

    ap.add_argument("--gff_feature_type", default=None,
                    help="Optional: only use GFF rows with this feature type (3rd column)")

    ap.add_argument("--ids_file", default=None,
                    help="Optional: file with one chunk_id (or parent ID) per line; "
                         "we map chunk_ids to parent IDs and evaluate only those parents")

    ap.add_argument(
        "--coord_mode",
        choices=["1based_inclusive", "0based_inclusive", "0based_halfopen"],
        default="1based_inclusive",
        help="Coordinate system of span_start_aa/span_end_aa in label file "
             "(chunk-local); converted to parent 1-based inclusive.",
    )

    ap.add_argument("--positive_labels", default=None,
                    help="Optional: only rows whose label is in this comma-separated set are used (e.g. 'P')")
    ap.add_argument("--require_use_span", action="store_true",
                    help="Only evaluate rows with use_span != 0 (recommended for IoU)")
    ap.add_argument("--no_require_use_span", dest="require_use_span", action="store_false",
                    help="Evaluate even if use_span == 0 (not recommended)")
    ap.set_defaults(require_use_span=True)

    ap.add_argument("--iou_threshold", type=float, default=0.5,
                    help="IoU threshold to report %% IoU >= X")

    ap.add_argument(
        "--plot_type",
        choices=["violin", "swarm", "both"],
        default="violin",
        help="Type of IoU distribution plot: violin, swarm (seaborn), or both.",
    )

    args = ap.parse_args()

    # Parse predictions and truth
    pred = parse_gff_predictions(args.gff, feature_type=args.gff_feature_type)
    truth, parent_len = parse_label_truth_spans_parent(
        args.labels,
        require_use_span=args.require_use_span,
        positive_labels=args.positive_labels,
        coord_mode=args.coord_mode,
    )

    pred_parents = set(pred.keys())
    truth_parents = set(truth.keys())

    # Optional filtering by ID list (ids_file contains chunk_ids or parent IDs)
    if args.ids_file:
        keep_parents = parse_id_list(args.ids_file)
        eval_ids = sorted(keep_parents & truth_parents & pred_parents)
        skipped_missing_gff = len((truth_parents & keep_parents) - pred_parents)
    else:
        eval_ids = sorted(truth_parents & pred_parents)
        skipped_missing_gff = len(truth_parents - pred_parents)

    if not eval_ids:
        raise SystemExit(
            "No parent sequences to evaluate after filtering. "
            "Check coord_mode / labels / ids_file / require_use_span / ID mapping."
        )

    rows = []
    vals: List[float] = []
    full_lens: List[int] = []
    truth_lens: List[int] = []
    pred_lens: List[int] = []
    span_ratios: List[float] = []

    n_no_pred = 0

    for pid in eval_ids:
        p_ints = pred.get(pid, [])
        t_ints = truth.get(pid, [])
        if not p_ints:
            n_no_pred += 1

        v = iou(p_ints, t_ints)
        vals.append(v)

        full_len = parent_len.get(pid, 0)
        full_lens.append(full_len)

        t_len = intervals_length(t_ints)
        truth_lens.append(t_len)

        p_len = intervals_length(p_ints)
        pred_lens.append(p_len)

        if t_len > 0:
            span_ratio = p_len / t_len
        else:
            span_ratio = float("nan")
        span_ratios.append(span_ratio)

        rows.append({
            "parent_seqid": pid,
            "full_seq_len": str(full_len),
            "truth_span_len": str(t_len),
            "pred_span_len": str(p_len),
            "span_ratio": f"{span_ratio:.6f}" if np.isfinite(span_ratio) else "",
            "pred_intervals": intervals_to_str(p_ints),
            "truth_intervals": intervals_to_str(t_ints),
            "iou": f"{v:.6f}",
        })

    n = len(vals)
    ge = sum(1 for v in vals if v >= args.iou_threshold)
    pct_ge = 100.0 * ge / n
    mean = sum(vals) / n
    median = sorted(vals)[n // 2]

    vals_arr = np.asarray(vals, dtype=float)
    full_arr = np.asarray(full_lens, dtype=float)
    truth_arr = np.asarray(truth_lens, dtype=float)
    pred_arr = np.asarray(pred_lens, dtype=float)
    ratio_arr = np.asarray(span_ratios, dtype=float)

    # Filter out non-finite ratios for correlation
    ratio_mask = np.isfinite(ratio_arr)
    ratio_arr_valid = ratio_arr[ratio_mask]
    vals_ratio_valid = vals_arr[ratio_mask]

    # Correlations with p-values
    def corr_with_p(x, y):
        if HAVE_SCIPY and x.size >= 3:
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
        else:
            r_p = pearson_corr_np(x, y)
            r_s = spearman_corr_np(x, y)
            p_p = p_s = float("nan")
        return r_p, p_p, r_s, p_s

    r_full_p, p_full_p, r_full_s, p_full_s = corr_with_p(full_arr, vals_arr)
    r_truth_p, p_truth_p, r_truth_s, p_truth_s = corr_with_p(truth_arr, vals_arr)
    r_pred_p, p_pred_p, r_pred_s, p_pred_s = corr_with_p(pred_arr, vals_arr)
    r_ratio_p, p_ratio_p, r_ratio_s, p_ratio_s = corr_with_p(ratio_arr_valid, vals_ratio_valid)

    # CSV (always) with LF endings
    out_csv = f"{args.out_prefix}.per_sequence_iou.csv"
    with open(out_csv, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "parent_seqid",
                "full_seq_len",
                "truth_span_len",
                "pred_span_len",
                "span_ratio",
                "pred_intervals",
                "truth_intervals",
                "iou",
            ],
            lineterminator="\n",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Optional TSV
    if args.per_sequence_tsv:
        with open(args.per_sequence_tsv, "w", newline="\n", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "parent_seqid",
                    "full_seq_len",
                    "truth_span_len",
                    "pred_span_len",
                    "span_ratio",
                    "pred_intervals",
                    "truth_intervals",
                    "iou",
                ],
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    # IoU distribution
    title = "PalmSite localization performance (IoU per parent sequence)"
    subtitle = (
        f"n={n} | mean={mean:.3f} | median={median:.3f} | "
        f"% IoU≥{args.iou_threshold:g} = {pct_ge:.1f}% | missing pred={n_no_pred}"
    )

    if args.plot_type == "violin":
        fig, _ = plot_violin(vals, title, subtitle)
        out_png = f"{args.out_prefix}.iou_violin.png"
        out_pdf = f"{args.out_prefix}.iou_violin.pdf"
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        plt.close(fig)
        print("Wrote:", out_png, out_pdf)

    elif args.plot_type == "swarm":
        fig, _ = plot_swarm(vals, title, subtitle)
        out_png = f"{args.out_prefix}.iou_swarm.png"
        out_pdf = f"{args.out_prefix}.iou_swarm.pdf"
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        plt.close(fig)
        print("Wrote:", out_png, out_pdf)

    else:
        fig, _ = plot_both(vals, title, subtitle)
        out_png = f"{args.out_prefix}.iou_both.png"
        out_pdf = f"{args.out_prefix}.iou_both.pdf"
        fig.savefig(out_png, dpi=300)
        fig.savefig(out_pdf)
        plt.close(fig)
        print("Wrote:", out_png, out_pdf)

    # Scatter: IoU vs full sequence length
    subtitle_full = (
        f"Pearson r={r_full_p:.3f} (p={p_full_p:.1e}), "
        f"Spearman ρ={r_full_s:.3f} (p={p_full_s:.1e}); n={n}"
    )
    fig, _ = plot_scatter(
        full_arr, vals_arr,
        xlabel="Full sequence length (aa)",
        ylabel="IoU",
        title="IoU vs full sequence length",
        subtitle=subtitle_full,
    )
    out_png_full = f"{args.out_prefix}.iou_vs_full_len.png"
    out_pdf_full = f"{args.out_prefix}.iou_vs_full_len.pdf"
    fig.savefig(out_png_full, dpi=300)
    fig.savefig(out_pdf_full)
    plt.close(fig)
    print("Wrote:", out_png_full, out_pdf_full)

    # Scatter: IoU vs truth span length
    subtitle_truth = (
        f"Pearson r={r_truth_p:.3f} (p={p_truth_p:.1e}), "
        f"Spearman ρ={r_truth_s:.3f} (p={p_truth_s:.1e}); n={n}"
    )
    fig, _ = plot_scatter(
        truth_arr, vals_arr,
        xlabel="Truth span length (aa)",
        ylabel="IoU",
        title="IoU vs truth span length",
        subtitle=subtitle_truth,
    )
    out_png_truth = f"{args.out_prefix}.iou_vs_truth_len.png"
    out_pdf_truth = f"{args.out_prefix}.iou_vs_truth_len.pdf"
    fig.savefig(out_png_truth, dpi=300)
    fig.savefig(out_pdf_truth)
    plt.close(fig)
    print("Wrote:", out_png_truth, out_pdf_truth)

    # Scatter: IoU vs predicted span length
    subtitle_pred = (
        f"Pearson r={r_pred_p:.3f} (p={p_pred_p:.1e}), "
        f"Spearman ρ={r_pred_s:.3f} (p={p_pred_s:.1e}); n={n}"
    )
    fig, _ = plot_scatter(
        pred_arr, vals_arr,
        xlabel="Predicted span length (aa)",
        ylabel="IoU",
        title="IoU vs predicted span length",
        subtitle=subtitle_pred,
    )
    out_png_pred = f"{args.out_prefix}.iou_vs_pred_len.png"
    out_pdf_pred = f"{args.out_prefix}.iou_vs_pred_len.pdf"
    fig.savefig(out_png_pred, dpi=300)
    fig.savefig(out_pdf_pred)
    plt.close(fig)
    print("Wrote:", out_png_pred, out_pdf_pred)

    # Scatter: IoU vs span_ratio (pred_len / truth_len)
    subtitle_ratio = (
        f"Pearson r={r_ratio_p:.3f} (p={p_ratio_p:.1e}), "
        f"Spearman ρ={r_ratio_s:.3f} (p={p_ratio_s:.1e}); n={ratio_arr_valid.size}"
    )
    fig, _ = plot_scatter(
        ratio_arr_valid, vals_ratio_valid,
        xlabel="Predicted / truth span length",
        ylabel="IoU",
        title="IoU vs predicted/true span length ratio",
        subtitle=subtitle_ratio,
    )
    out_png_ratio = f"{args.out_prefix}.iou_vs_span_ratio.png"
    out_pdf_ratio = f"{args.out_prefix}.iou_vs_span_ratio.pdf"
    fig.savefig(out_png_ratio, dpi=300)
    fig.savefig(out_pdf_ratio)
    plt.close(fig)
    print("Wrote:", out_png_ratio, out_pdf_ratio)

    print("Wrote:", out_csv)
    if args.per_sequence_tsv:
        print("Wrote:", args.per_sequence_tsv)
    print(f"Evaluated parent sequences (in GFF & labels): {n}")
    print(f"Parents with ground truth but missing in GFF (skipped): {skipped_missing_gff}")
    print(f"Parents present in GFF but with no predicted intervals: {n_no_pred}")
    print(f"% IoU ≥ {args.iou_threshold:g}: {pct_ge:.2f}%")
    print(f"Full length vs IoU:  Pearson r={r_full_p:.3f}, p={p_full_p:.3e}; "
          f"Spearman ρ={r_full_s:.3f}, p={p_full_s:.3e}")
    print(f"Truth span vs IoU:   Pearson r={r_truth_p:.3f}, p={p_truth_p:.3e}; "
          f"Spearman ρ={r_truth_s:.3f}, p={p_truth_s:.3e}")
    print(f"Pred span vs IoU:    Pearson r={r_pred_p:.3f}, p={p_pred_p:.3e}; "
          f"Spearman ρ={r_pred_s:.3f}, p={p_pred_s:.3e}")
    print(f"Span ratio vs IoU:   Pearson r={r_ratio_p:.3f}, p={p_ratio_p:.3e}; "
          f"Spearman ρ={r_ratio_s:.3f}, p={p_ratio_s:.3e}")


if __name__ == "__main__":
    main()

