#!/usr/bin/env python3
"""
Compute per-chunk IoU between PalmSite predicted spans (GFF3)
and ground-truth spans from a label TSV that uses chunk_ids, and visualize:

1) Distribution of IoU (violin / seaborn swarm / both)
2) Scatter plots:
   - IoU vs chunk length
   - IoU vs truth span length
   - IoU vs predicted span length
   - IoU vs predicted/true span length ratio

Key assumptions (matches your labels_only_positives.tsv):
- Label file has column "chunk_id" with format like:
    "MGYP000911387321|chunk_0001_of_0001|aa_000000_000260"
    "NP_041870.2|chunk_0002_of_0002|aa_001094_003094"
- span_start_aa / span_end_aa are CHUNK-LOCAL coordinates.
- GFF seqid is expected to be the chunk_id itself.
- Predicted spans in the GFF are treated as chunk-local coordinates, just like
  the label TSV spans.
- Optional ids_file may contain chunk_ids and/or parent IDs:
    * a chunk_id selects only that chunk
    * a parent ID selects all chunks belonging to that parent

Important behavior:
- IoU is computed PER CHUNK, not per parent.
- Only chunks that have ground-truth spans AND whose chunk_id exists in the GFF
  are evaluated by default.
- If a chunk exists in the GFF but has no predicted intervals, its IoU is 0.

Outputs:
- <out_prefix>.per_chunk_iou.csv         (LF line endings)
- optional TSV via --per_sequence_tsv    (LF line endings; kept for CLI compatibility)
- <out_prefix>.iou_<plot_type>.png/.pdf  (IoU distribution)
- <out_prefix>.iou_vs_chunk_len.png/.pdf (IoU vs chunk len)
- <out_prefix>.iou_vs_truth_len.png/.pdf (IoU vs truth span len)
- <out_prefix>.iou_vs_pred_len.png/.pdf  (IoU vs predicted span len)
- <out_prefix>.iou_vs_span_ratio.png/.pdf (IoU vs pred/true ratio)
"""

import argparse
import csv
import gzip
import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ChunkInfo:
    chunk_id: str
    parent_id: str
    chunk_index: int
    chunks_total: int
    parent_start0: int  # 0-based inclusive start in parent
    parent_end0: int    # 0-based exclusive end in parent

    @property
    def chunk_len(self) -> int:
        return self.parent_end0 - self.parent_start0

    @property
    def parent_start1(self) -> int:
        return self.parent_start0 + 1

    @property
    def parent_end1(self) -> int:
        return self.parent_end0


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

_CHUNK_PATTERNS = [
    re.compile(
        r'^(?P<parent>.+?)\|chunk_(?P<chunk>\d+)_of_(?P<total>\d+)\|aa_(?P<s>\d{6})_(?P<e>\d{6})$'
    ),
    re.compile(
        r'^(?P<parent>.+?)_chunk_(?P<chunk>\d+)_of_(?P<total>\d+)_aa_(?P<s>\d{6})_(?P<e>\d{6})$'
    ),
]


def parse_chunk_id(chunk_id: str) -> ChunkInfo:
    """
    Parse a chunk_id like:
      "NP_041870.2|chunk_0002_of_0002|aa_001094_003094"

    Returns ChunkInfo with chunk coordinates represented on the parent as
    0-based half-open [parent_start0, parent_end0).
    """
    for rx in _CHUNK_PATTERNS:
        m = rx.match(chunk_id)
        if m:
            return ChunkInfo(
                chunk_id=chunk_id,
                parent_id=m.group("parent"),
                chunk_index=int(m.group("chunk")),
                chunks_total=int(m.group("total")),
                parent_start0=int(m.group("s")),
                parent_end0=int(m.group("e")),
            )
    raise ValueError(f"chunk_id does not match expected pattern: {chunk_id}")


def parent_id_from_chunk(chunk_id: str) -> str:
    if "|chunk_" in chunk_id:
        return chunk_id.split("|chunk_", 1)[0]
    if "_chunk_" in chunk_id:
        return chunk_id.split("_chunk_", 1)[0]
    return chunk_id


def parse_id_list(ids_path: str) -> Tuple[Set[str], Set[str]]:
    """
    Read an ID list file.

    Lines can contain chunk_ids or parent IDs.
    Returns:
      (explicit_chunk_ids, explicit_parent_ids)
    """
    chunk_ids: Set[str] = set()
    parent_ids: Set[str] = set()
    with smart_open(ids_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                info = parse_chunk_id(line)
                chunk_ids.add(info.chunk_id)
            except ValueError:
                parent_ids.add(parent_id_from_chunk(line))
    return chunk_ids, parent_ids


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
    GFF3 -> {seqid: [(start, end), ...]}, using 1-based inclusive coords.

    seqid is expected to be a chunk_id, and the spans are treated as chunk-local.
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


def parse_label_truth_spans_by_chunk(
    label_tsv: str,
    require_use_span: bool = True,
    positive_labels: Optional[str] = None,
    coord_mode: str = "1based_inclusive",
) -> Tuple[Dict[str, List[Interval]], Dict[str, ChunkInfo]]:
    """
    Read label TSV (with chunk_id) and build:

      truth_by_chunk: {chunk_id: [(start, end), ...]}  # chunk-local, 1-based inclusive
      chunk_info_by_id: {chunk_id: ChunkInfo}

    Assumes columns: chunk_id, label, use_span, span_start_aa, span_end_aa.
    """
    pos_set = None
    if positive_labels:
        pos_set = {s.strip() for s in positive_labels.split(",") if s.strip()}

    truth_by_chunk: Dict[str, List[Interval]] = {}
    chunk_info_by_id: Dict[str, ChunkInfo] = {}

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
            info = parse_chunk_id(chunk_id_raw)
            chunk_info_by_id[info.chunk_id] = info

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

            loc_s, loc_e = convert_coords_local_to_1based(s0, e0, coord_mode)

            # Clamp to chunk bounds in case labels slightly exceed the chunk range.
            if info.chunk_len <= 0:
                continue
            loc_s = max(1, min(loc_s, info.chunk_len))
            loc_e = max(1, min(loc_e, info.chunk_len))
            if loc_s > loc_e:
                loc_s, loc_e = loc_e, loc_s

            truth_by_chunk.setdefault(info.chunk_id, []).append((loc_s, loc_e))

    for chunk_id, spans in list(truth_by_chunk.items()):
        truth_by_chunk[chunk_id] = normalize_intervals(spans)

    return truth_by_chunk, chunk_info_by_id



def get_chunk_prediction(pred_by_seqid: Dict[str, List[Interval]], info: ChunkInfo) -> Tuple[List[Interval], str]:
    """
    Return chunk-local predicted intervals and the source mode used.

    Only direct chunk_id matches are accepted. No parent-coordinate projection
    is performed.
    """
    if info.chunk_id in pred_by_seqid:
        return normalize_intervals(pred_by_seqid[info.chunk_id]), "chunk"
    return [], "missing"


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

    fig.suptitle(title, fontsize=10)
    fig.text(0.5, 0.02, subtitle, ha="center", va="bottom", fontsize=7)
    fig.tight_layout(rect=[0.03, 0.08, 0.97, 0.92])
    return fig, axes


def plot_scatter(x: np.ndarray, y: np.ndarray,
                 xlabel: str, ylabel: str,
                 title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    sns.set(style="whitegrid")
    ax.scatter(x, y, alpha=0.7, edgecolors="none", s=8)
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
    ap.add_argument("--gff", required=True, help="PalmSite prediction GFF3 (.gz ok); seqid may be chunk_id or parent sequence ID")
    ap.add_argument("--labels", required=True, help="Label TSV with chunk_id + span_start_aa/span_end_aa (.gz ok)")
    ap.add_argument("--out_prefix", default="figB", help="Output prefix for CSV and plots")

    ap.add_argument(
        "--per_sequence_tsv",
        default=None,
        help="Optional path to write per-chunk IoU table as TSV "
             "(argument name kept for backward CLI compatibility).",
    )

    ap.add_argument("--gff_feature_type", default=None,
                    help="Optional: only use GFF rows with this feature type (3rd column)")

    ap.add_argument("--ids_file", default=None,
                    help="Optional: file with one chunk_id or parent ID per line; "
                         "chunk_id keeps only that chunk, parent ID keeps all chunks from that parent")
    ap.add_argument(
        "--only_gff_sequences",
        dest="only_gff_sequences",
        action="store_true",
        help="Only evaluate chunks whose chunk_id exists in the GFF seqids (default).",
    )
    ap.add_argument(
        "--include_missing_gff_sequences",
        dest="only_gff_sequences",
        action="store_false",
        help="Also evaluate label chunks absent from the GFF; these receive empty predictions and typically IoU=0.",
    )

    ap.add_argument(
        "--coord_mode",
        choices=["1based_inclusive", "0based_inclusive", "0based_halfopen"],
        default="1based_inclusive",
        help="Coordinate system of span_start_aa/span_end_aa in label file "
             "(chunk-local); normalized to chunk-local 1-based inclusive.",
    )

    ap.add_argument("--positive_labels", default=None,
                    help="Optional: only rows whose label is in this comma-separated set are used (e.g. 'P')")
    ap.add_argument("--require_use_span", action="store_true",
                    help="Only evaluate rows with use_span != 0 (recommended for IoU)")
    ap.add_argument("--no_require_use_span", dest="require_use_span", action="store_false",
                    help="Evaluate even if use_span == 0 (not recommended)")
    ap.set_defaults(require_use_span=True, only_gff_sequences=True)

    ap.add_argument("--iou_threshold", type=float, default=0.5,
                    help="IoU threshold to report %% IoU >= X")

    ap.add_argument(
        "--plot_type",
        choices=["violin", "swarm", "both"],
        default="violin",
        help="Type of IoU distribution plot: violin, swarm (seaborn), or both.",
    )

    args = ap.parse_args()

    pred_by_seqid = parse_gff_predictions(args.gff, feature_type=args.gff_feature_type)
    truth_by_chunk, chunk_info_by_id = parse_label_truth_spans_by_chunk(
        args.labels,
        require_use_span=args.require_use_span,
        positive_labels=args.positive_labels,
        coord_mode=args.coord_mode,
    )

    truth_chunk_ids = set(truth_by_chunk.keys())

    if args.ids_file:
        keep_chunk_ids, keep_parent_ids = parse_id_list(args.ids_file)
        eval_chunk_ids = sorted(
            cid for cid in truth_chunk_ids
            if (cid in keep_chunk_ids) or (chunk_info_by_id[cid].parent_id in keep_parent_ids)
        )
    else:
        eval_chunk_ids = sorted(truth_chunk_ids)

    n_filtered_not_in_gff = 0
    if args.only_gff_sequences:
        gff_seqids = set(pred_by_seqid.keys())
        before = len(eval_chunk_ids)
        eval_chunk_ids = [
            cid for cid in eval_chunk_ids
            if cid in gff_seqids
        ]
        n_filtered_not_in_gff = before - len(eval_chunk_ids)

    if not eval_chunk_ids:
        raise SystemExit(
            "No chunks to evaluate after filtering. "
            "Check coord_mode / labels / ids_file / require_use_span / ID mapping."
        )

    rows = []
    vals: List[float] = []
    chunk_lens: List[int] = []
    truth_lens: List[int] = []
    pred_lens: List[int] = []
    span_ratios: List[float] = []

    n_no_prediction_source = 0
    n_no_pred_intervals = 0
    n_chunk_seqid_pred = 0

    for chunk_id in eval_chunk_ids:
        info = chunk_info_by_id[chunk_id]
        t_ints = truth_by_chunk[chunk_id]
        p_ints, pred_mode = get_chunk_prediction(pred_by_seqid, info)

        if pred_mode == "missing":
            n_no_prediction_source += 1
        elif pred_mode == "chunk":
            n_chunk_seqid_pred += 1

        if not p_ints:
            n_no_pred_intervals += 1

        v = iou(p_ints, t_ints)
        vals.append(v)

        chunk_len = info.chunk_len
        chunk_lens.append(chunk_len)

        t_len = intervals_length(t_ints)
        truth_lens.append(t_len)

        p_len = intervals_length(p_ints)
        pred_lens.append(p_len)

        span_ratio = (p_len / t_len) if t_len > 0 else float("nan")
        span_ratios.append(span_ratio)

        rows.append({
            "chunk_id": chunk_id,
            "parent_seqid": info.parent_id,
            "chunk_index": str(info.chunk_index),
            "chunks_total": str(info.chunks_total),
            "chunk_len": str(chunk_len),
            "truth_span_len": str(t_len),
            "pred_span_len": str(p_len),
            "span_ratio": f"{span_ratio:.6f}" if np.isfinite(span_ratio) else "",
            "pred_source": pred_mode,
            "pred_intervals": intervals_to_str(p_ints),
            "truth_intervals": intervals_to_str(t_ints),
            "iou": f"{v:.6f}",
        })

    n = len(vals)
    ge = sum(1 for v in vals if v >= args.iou_threshold)
    pct_ge = 100.0 * ge / n
    mean = float(np.mean(vals_arr := np.asarray(vals, dtype=float)))
    median = float(np.median(vals_arr))

    chunk_arr = np.asarray(chunk_lens, dtype=float)
    truth_arr = np.asarray(truth_lens, dtype=float)
    pred_arr = np.asarray(pred_lens, dtype=float)
    ratio_arr = np.asarray(span_ratios, dtype=float)

    ratio_mask = np.isfinite(ratio_arr)
    ratio_arr_valid = ratio_arr[ratio_mask]
    vals_ratio_valid = vals_arr[ratio_mask]

    def corr_with_p(x, y):
        if x.size < 2 or y.size < 2:
            return float("nan"), float("nan"), float("nan"), float("nan")
        if np.all(x == x[0]) or np.all(y == y[0]):
            return float("nan"), float("nan"), float("nan"), float("nan")
        if HAVE_SCIPY:
            r_p, p_p = stats.pearsonr(x, y)
            r_s, p_s = stats.spearmanr(x, y)
        else:
            r_p = pearson_corr_np(x, y)
            r_s = spearman_corr_np(x, y)
            p_p = p_s = float("nan")
        return r_p, p_p, r_s, p_s

    r_chunk_p, p_chunk_p, r_chunk_s, p_chunk_s = corr_with_p(chunk_arr, vals_arr)
    r_truth_p, p_truth_p, r_truth_s, p_truth_s = corr_with_p(truth_arr, vals_arr)
    r_pred_p, p_pred_p, r_pred_s, p_pred_s = corr_with_p(pred_arr, vals_arr)
    r_ratio_p, p_ratio_p, r_ratio_s, p_ratio_s = corr_with_p(ratio_arr_valid, vals_ratio_valid)

    out_csv = f"{args.out_prefix}.per_chunk_iou.csv"
    with open(out_csv, "w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_id",
                "parent_seqid",
                "chunk_index",
                "chunks_total",
                "chunk_len",
                "truth_span_len",
                "pred_span_len",
                "span_ratio",
                "pred_source",
                "pred_intervals",
                "truth_intervals",
                "iou",
            ],
            lineterminator="\n",
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if args.per_sequence_tsv:
        with open(args.per_sequence_tsv, "w", newline="\n", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "chunk_id",
                    "parent_seqid",
                    "chunk_index",
                    "chunks_total",
                    "chunk_len",
                    "truth_span_len",
                    "pred_span_len",
                    "span_ratio",
                    "pred_source",
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

    title = "PalmSite localization performance (IoU per chunk)"
    subtitle = (
        f"n={n} | mean={mean:.3f} | median={median:.3f} | "
        f"% IoU≥{args.iou_threshold:g} = {pct_ge:.1f}% | no pred intervals={n_no_pred_intervals}"
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

    subtitle_chunk = (
        f"Pearson r={r_chunk_p:.3f} (p={p_chunk_p:.1e}), "
        f"Spearman ρ={r_chunk_s:.3f} (p={p_chunk_s:.1e}); n={n}"
    )
    fig, _ = plot_scatter(
        chunk_arr, vals_arr,
        xlabel="Chunk length (aa)",
        ylabel="IoU",
        title="IoU vs chunk length",
        subtitle=subtitle_chunk,
    )
    out_png_chunk = f"{args.out_prefix}.iou_vs_chunk_len.png"
    out_pdf_chunk = f"{args.out_prefix}.iou_vs_chunk_len.pdf"
    fig.savefig(out_png_chunk, dpi=300)
    fig.savefig(out_pdf_chunk)
    plt.close(fig)
    print("Wrote:", out_png_chunk, out_pdf_chunk)

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
    print(f"Evaluated chunks: {n}")
    if args.only_gff_sequences:
        print(f"Chunks filtered out because chunk_id did not exist in GFF: {n_filtered_not_in_gff}")
    print(f"Chunks using direct chunk seqid predictions: {n_chunk_seqid_pred}")
    print(f"Chunks with no matching prediction source: {n_no_prediction_source}")
    print(f"Chunks with no predicted intervals after matching: {n_no_pred_intervals}")
    print(f"% IoU ≥ {args.iou_threshold:g}: {pct_ge:.2f}%")
    print(f"Chunk length vs IoU: Pearson r={r_chunk_p:.3f}, p={p_chunk_p:.3e}; "
          f"Spearman ρ={r_chunk_s:.3f}, p={p_chunk_s:.3e}")
    print(f"Truth span vs IoU:   Pearson r={r_truth_p:.3f}, p={p_truth_p:.3e}; "
          f"Spearman ρ={r_truth_s:.3f}, p={p_truth_s:.3e}")
    print(f"Pred span vs IoU:    Pearson r={r_pred_p:.3f}, p={p_pred_p:.3e}; "
          f"Spearman ρ={r_pred_s:.3f}, p={p_pred_s:.3e}")
    print(f"Span ratio vs IoU:   Pearson r={r_ratio_p:.3f}, p={p_ratio_p:.3e}; "
          f"Spearman ρ={r_ratio_s:.3f}, p={p_ratio_s:.3e}")


if __name__ == "__main__":
    main()

