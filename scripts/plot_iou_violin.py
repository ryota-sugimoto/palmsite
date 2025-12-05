#!/usr/bin/env python3
"""
Compute per-sequence IoU between PalmSite predicted spans (GFF3) and ground-truth spans from a label TSV,
then make a violin plot + % with IoU >= threshold.

Expected label TSV columns (at minimum):
  chunk_id, label, use_span, span_start_aa, span_end_aa

Expected GFF3 format: 9 tab-separated columns, seqid in column 1, start/end in columns 4/5.
"""

import argparse
import csv
import gzip
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

Interval = Tuple[int, int]  # 1-based inclusive (start, end)


def smart_open(path: str, mode: str = "rt"):
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode, encoding="utf-8")


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


def parse_gff_predictions(gff_path: str, feature_type: Optional[str] = None) -> Dict[str, List[Interval]]:
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


def parse_id_list(ids_path: str) -> set:
    ids = set()
    with smart_open(ids_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ids.add(line)
    return ids


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


def convert_coords(start: int, end: int, coord_mode: str) -> Interval:
    """
    Convert label coords to 1-based inclusive (to match GFF convention).
    coord_mode:
      - 1based_inclusive: [start, end] in 1-based inclusive
      - 0based_inclusive: [start, end] in 0-based inclusive  -> +1 both
      - 0based_halfopen:  [start, end) in 0-based half-open  -> start+1, end
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
    return (s, e)


def parse_label_truth_spans(
    label_tsv: str,
    require_use_span: bool = True,
    positive_labels: Optional[str] = None,
    coord_mode: str = "0based_halfopen",
) -> Dict[str, List[Interval]]:
    """
    Returns {chunk_id: [(start,end)]} truth spans in 1-based inclusive coords.

    - If require_use_span=True: only keep rows with use_span != 0
    - If positive_labels is provided: only keep rows where label in that comma-separated set
    """
    pos_set = None
    if positive_labels:
        pos_set = {s.strip() for s in positive_labels.split(",") if s.strip()}

    truth: Dict[str, List[Interval]] = {}
    with smart_open(label_tsv, "rt") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"chunk_id", "label", "use_span", "span_start_aa", "span_end_aa"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Label file missing required columns: {sorted(missing)}. Found: {reader.fieldnames}")

        for row in reader:
            cid = row["chunk_id"].strip()
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

            s1, e1 = convert_coords(s0, e0, coord_mode)
            if e1 <= 0:
                continue

            truth[cid] = normalize_intervals([(s1, e1)])
    return truth


def intervals_to_str(intervals: List[Interval]) -> str:
    return ",".join(f"{s}-{e}" for s, e in intervals) if intervals else ""


def make_violin_plot(values: List[float], out_png: str, out_pdf: str, title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.violinplot([values], showmeans=False, showmedians=True, showextrema=True)
    ax.set_ylim(0, 1)
    ax.set_xticks([1])
    ax.set_xticklabels(["IoU"])
    ax.set_ylabel("Intersection over Union (IoU)")
    ax.set_title(title, fontsize=12, pad=10)
    ax.text(0.5, 1.01, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=10)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_pdf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gff", required=True, help="PalmSite prediction GFF(.gz)")
    ap.add_argument("--labels", required=True, help="Label TSV with truth spans (.gz ok)")
    ap.add_argument("--out_prefix", default="figB", help="Prefix for outputs")

    ap.add_argument("--gff_feature_type", default=None, help="Optional: only use this GFF feature type (col3)")
    ap.add_argument("--ids_file", default=None, help="Optional: 1 seqid per line; only evaluate these IDs")

    ap.add_argument(
        "--coord_mode",
        choices=["1based_inclusive", "0based_inclusive", "0based_halfopen"],
        default="0based_halfopen",
        help="Coordinate system for span_start_aa/span_end_aa in label file; converted to 1-based inclusive",
    )

    ap.add_argument("--positive_labels", default=None,
                    help="Optional: evaluate only rows whose label is in this comma-separated set (e.g. 'P,Pos')")
    ap.add_argument("--require_use_span", action="store_true",
                    help="Only evaluate rows with use_span != 0 (recommended)")
    ap.add_argument("--no_require_use_span", dest="require_use_span", action="store_false",
                    help="Evaluate even if use_span==0 (not recommended for IoU)")
    ap.set_defaults(require_use_span=True)

    ap.add_argument("--iou_threshold", type=float, default=0.5, help="Threshold to report % IoU>=X")
    args = ap.parse_args()

    pred = parse_gff_predictions(args.gff, feature_type=args.gff_feature_type)
    truth = parse_label_truth_spans(
        args.labels,
        require_use_span=args.require_use_span,
        positive_labels=args.positive_labels,
        coord_mode=args.coord_mode,
    )

    # Filter by ID list (typically your test set IDs)
    if args.ids_file:
        keep = parse_id_list(args.ids_file)
        pred = {k: v for k, v in pred.items() if k in keep}
        truth = {k: v for k, v in truth.items() if k in keep}
        eval_ids = sorted(keep & set(truth.keys()))  # evaluate only those with truth
    else:
        eval_ids = sorted(truth.keys())

    if not eval_ids:
        raise SystemExit(
            "No sequences to evaluate after filtering (truth spans empty). "
            "Check coord_mode / labels / ids_file / require_use_span."
        )

    rows = []
    vals: List[float] = []
    n_no_pred = 0

    for sid in eval_ids:
        p_ints = pred.get(sid, [])
        t_ints = truth.get(sid, [])
        if not p_ints:
            n_no_pred += 1
        v = iou(p_ints, t_ints)
        vals.append(v)
        rows.append({
            "seqid": sid,
            "pred_intervals": intervals_to_str(p_ints),
            "truth_intervals": intervals_to_str(t_ints),
            "iou": f"{v:.6f}",
        })

    n = len(vals)
    ge = sum(1 for v in vals if v >= args.iou_threshold)
    pct_ge = 100.0 * ge / n
    mean = sum(vals) / n
    median = sorted(vals)[n // 2]

    out_csv = f"{args.out_prefix}.per_sequence_iou.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["seqid", "pred_intervals", "truth_intervals", "iou"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    out_png = f"{args.out_prefix}.iou_violin.png"
    out_pdf = f"{args.out_prefix}.iou_violin.pdf"
    title = "PalmSite localization performance (IoU per sequence)"
    subtitle = (
        f"n={n} | mean={mean:.3f} | median={median:.3f} | "
        f"% IoU≥{args.iou_threshold:g} = {pct_ge:.1f}% | missing pred={n_no_pred}"
    )
    make_violin_plot(vals, out_png, out_pdf, title, subtitle)

    print("Wrote:", out_csv)
    print("Wrote:", out_png)
    print("Wrote:", out_pdf)
    print(f"Evaluated sequences: {n}")
    print(f"Missing predictions: {n_no_pred}")
    print(f"% IoU ≥ {args.iou_threshold:g}: {pct_ge:.2f}%")


if __name__ == "__main__":
    main()

