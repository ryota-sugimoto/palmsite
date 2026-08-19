#!/usr/bin/env python3
"""
sample_palmsite_output_by_logit_v2.py

Stream a PalmSite GFF3 file, keep all PalmSite positives, sample non-positive
records from user-defined logit bins, and optionally plot the input and selected
sampling frequency by logit bin.

Designed for MGnify-scale PalmSite scans where the input can contain tens of
millions of records.
"""

from __future__ import annotations

import argparse
import gzip
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TextIO, Tuple


# Refined for MGnify-scale inspection:
#   - clean negatives: very low logits
#   - intermediates: broad negative-to-positive continuum
#   - near-threshold: enriched strongly, including all just below P=0.9
#   - high-logit bins: positives are kept separately; non-positive discordant
#     records in these bins are also retained by default because they are useful
#     diagnostics if P and Logit ever disagree.
DEFAULT_BIN_EDGES = "-inf,-10,-8,-6,-4,-2,0,1,1.5,2,2.197224577,3,4.59511985,inf"
DEFAULT_BIN_SIZES = "1000,1000,1500,2500,4000,5000,5000,6000,6000,all,all,all,all"


@dataclass
class Record:
    seq_id: str
    p: float
    logit: float
    calibrated_logit: Optional[float]
    length: Optional[int]
    bin_label: str
    decision: str
    gff_line: str


@dataclass
class BinSummary:
    bin_index: int
    bin_left: float
    bin_right: float
    bin_label: str
    requested_negative_sample_size: Optional[int]
    total_seen: int
    positive_seen: int
    negative_seen: int
    selected_positive: int
    selected_negative: int
    selected_total: int

    @property
    def negative_sampling_fraction(self) -> float:
        if self.negative_seen == 0:
            return float("nan")
        return self.selected_negative / self.negative_seen

    @property
    def total_selected_fraction(self) -> float:
        if self.total_seen == 0:
            return float("nan")
        return self.selected_total / self.total_seen


def smart_open(path: str, mode: str = "rt") -> TextIO:
    if path == "-":
        return sys.stdin if "r" in mode else sys.stdout
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode, encoding="utf-8")


def ensure_parent(path: str | Path) -> None:
    p = Path(path)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)


def parse_attrs(attr_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for item in attr_text.strip().split(";"):
        if not item:
            continue
        if "=" in item:
            k, v = item.split("=", 1)
            attrs[k] = v
    return attrs


def parse_float(x: Optional[str]) -> Optional[float]:
    if x is None or x == ".":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def parse_int(x: Optional[str]) -> Optional[int]:
    if x is None or x == ".":
        return None
    try:
        return int(float(x))
    except ValueError:
        return None


def parse_edges(edge_text: str) -> List[float]:
    vals: List[float] = []
    for x in edge_text.split(","):
        x = x.strip().lower()
        if x in {"-inf", "-infinity"}:
            vals.append(-math.inf)
        elif x in {"inf", "+inf", "infinity", "+infinity"}:
            vals.append(math.inf)
        else:
            vals.append(float(x))

    if len(vals) < 2:
        raise ValueError("Need at least two bin edges.")

    for a, b in zip(vals, vals[1:]):
        if not a < b:
            raise ValueError(f"Bin edges must be strictly increasing: {a}, {b}")

    return vals


def parse_bin_sizes(size_text: str, n_bins: int) -> List[Optional[int]]:
    sizes: List[Optional[int]] = []

    for x in size_text.split(","):
        x = x.strip().lower()
        if x == "all":
            sizes.append(None)
        else:
            k = int(x)
            if k < 0:
                raise ValueError("Bin sizes must be non-negative or 'all'.")
            sizes.append(k)

    if len(sizes) != n_bins:
        raise ValueError(
            f"Number of bin sizes must match number of bins. "
            f"Got {len(sizes)} sizes for {n_bins} bins."
        )

    return sizes


def bin_index(x: float, edges: Sequence[float]) -> Optional[int]:
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            return i
    if x == edges[-1]:
        return len(edges) - 2
    return None


def format_edge(v: float) -> str:
    if v == -math.inf:
        return "-inf"
    if v == math.inf:
        return "inf"
    return f"{v:g}"


def bin_label(i: int, edges: Sequence[float]) -> str:
    return f"[{format_edge(edges[i])},{format_edge(edges[i + 1])})"


def get_record_id(seqid_col: str, attrs: Dict[str, str], id_source: str) -> str:
    if id_source == "seqid":
        return seqid_col
    if id_source not in attrs:
        raise KeyError(f"Attribute '{id_source}' not found in GFF3 attributes.")
    return attrs[id_source]


def parse_gff_record(
    line: str,
    id_source: str,
    p_attr: str,
    logit_attr: str,
) -> Optional[Tuple[str, float, float, Optional[float], Optional[int]]]:
    if not line.strip() or line.startswith("#"):
        return None

    fields = line.rstrip("\n").split("\t")
    if len(fields) != 9:
        return None

    seqid_col = fields[0]
    score_col = fields[5]
    attrs = parse_attrs(fields[8])

    try:
        seq_id = get_record_id(seqid_col, attrs, id_source)
    except KeyError:
        return None

    p = parse_float(attrs.get(p_attr))
    if p is None:
        p = parse_float(score_col)

    logit = parse_float(attrs.get(logit_attr))
    calibrated_logit = parse_float(attrs.get("CalibratedLogit"))
    length = parse_int(attrs.get("len"))

    if p is None or logit is None:
        return None

    return seq_id, p, logit, calibrated_logit, length


def reservoir_add(
    reservoir: List[Record],
    item: Record,
    seen_count: int,
    k: Optional[int],
    rng: random.Random,
) -> None:
    """Reservoir-sample one item into a fixed-size bin reservoir.

    k=None means keep all records in that bin. This is useful for the immediate
    pre-threshold bin, but avoid k=None for huge low-logit bins.
    """
    if k is None:
        reservoir.append(item)
        return

    if k == 0:
        return

    if len(reservoir) < k:
        reservoir.append(item)
        return

    j = rng.randint(1, seen_count)
    if j <= k:
        reservoir[j - 1] = item


def fasta_header_id(header: str, mode: str) -> str:
    h = header[1:].rstrip("\n")
    if mode == "firstword":
        return h.split()[0]
    if mode == "full":
        return h
    raise ValueError(f"Unknown FASTA ID mode: {mode}")


def extract_fasta(
    fasta_paths: List[str],
    selected_ids: set[str],
    out_fasta: str,
    fasta_id_mode: str,
) -> int:
    n_written = 0

    out = smart_open(out_fasta, "wt")
    try:
        for fasta_path in fasta_paths:
            with smart_open(fasta_path, "rt") as fh:
                write_current = False

                for line in fh:
                    if line.startswith(">"):
                        current_id = fasta_header_id(line, fasta_id_mode)
                        write_current = current_id in selected_ids
                        if write_current:
                            n_written += 1
                            out.write(line)
                    else:
                        if write_current:
                            out.write(line)
    finally:
        if out_fasta != "-":
            out.close()

    return n_written


def write_tsv(records: Sequence[Record], out_tsv: str) -> None:
    out = smart_open(out_tsv, "wt")
    try:
        print(
            "\t".join(
                [
                    "seq_id",
                    "decision",
                    "bin",
                    "P",
                    "Logit",
                    "CalibratedLogit",
                    "len",
                    "gff_line",
                ]
            ),
            file=out,
        )

        for r in records:
            print(
                "\t".join(
                    [
                        r.seq_id,
                        r.decision,
                        r.bin_label,
                        f"{r.p:.12g}",
                        f"{r.logit:.12g}",
                        "" if r.calibrated_logit is None else f"{r.calibrated_logit:.12g}",
                        "" if r.length is None else str(r.length),
                        r.gff_line,
                    ]
                ),
                file=out,
            )
    finally:
        if out_tsv != "-":
            out.close()


def build_bin_summaries(
    edges: Sequence[float],
    bin_sizes: Sequence[Optional[int]],
    total_seen_per_bin: Sequence[int],
    positive_seen_per_bin: Sequence[int],
    negative_seen_per_bin: Sequence[int],
    selected_records: Sequence[Record],
) -> List[BinSummary]:
    n_bins = len(edges) - 1
    selected_positive = [0 for _ in range(n_bins)]
    selected_negative = [0 for _ in range(n_bins)]

    for r in selected_records:
        i = bin_index(r.logit, edges)
        if i is None:
            continue
        if r.decision == "keep_positive":
            selected_positive[i] += 1
        elif r.decision == "sampled_negative":
            selected_negative[i] += 1
        else:
            selected_negative[i] += 1

    summaries: List[BinSummary] = []
    for i in range(n_bins):
        summaries.append(
            BinSummary(
                bin_index=i,
                bin_left=edges[i],
                bin_right=edges[i + 1],
                bin_label=bin_label(i, edges),
                requested_negative_sample_size=bin_sizes[i],
                total_seen=int(total_seen_per_bin[i]),
                positive_seen=int(positive_seen_per_bin[i]),
                negative_seen=int(negative_seen_per_bin[i]),
                selected_positive=int(selected_positive[i]),
                selected_negative=int(selected_negative[i]),
                selected_total=int(selected_positive[i] + selected_negative[i]),
            )
        )
    return summaries


def write_bin_summary_tsv(summaries: Sequence[BinSummary], out_tsv: str) -> None:
    out = smart_open(out_tsv, "wt")
    try:
        print(
            "\t".join(
                [
                    "bin_index",
                    "bin_left",
                    "bin_right",
                    "bin",
                    "requested_negative_sample_size",
                    "total_seen",
                    "positive_seen",
                    "negative_seen",
                    "selected_positive",
                    "selected_negative",
                    "selected_total",
                    "negative_sampling_fraction",
                    "total_selected_fraction",
                ]
            ),
            file=out,
        )
        for s in summaries:
            requested = "all" if s.requested_negative_sample_size is None else str(s.requested_negative_sample_size)
            neg_frac = "" if math.isnan(s.negative_sampling_fraction) else f"{s.negative_sampling_fraction:.12g}"
            total_frac = "" if math.isnan(s.total_selected_fraction) else f"{s.total_selected_fraction:.12g}"
            print(
                "\t".join(
                    [
                        str(s.bin_index),
                        format_edge(s.bin_left),
                        format_edge(s.bin_right),
                        s.bin_label,
                        requested,
                        str(s.total_seen),
                        str(s.positive_seen),
                        str(s.negative_seen),
                        str(s.selected_positive),
                        str(s.selected_negative),
                        str(s.selected_total),
                        neg_frac,
                        total_frac,
                    ]
                ),
                file=out,
            )
    finally:
        if out_tsv != "-":
            out.close()


def plot_sampling_summary(
    summaries: Sequence[BinSummary],
    out_prefix: str,
    formats: Sequence[str],
    positive_p: float,
    log_counts: bool,
    title: str,
) -> None:
    """Plot input count, selected count, and sampling fraction by logit bin."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("--out-plot-prefix requires matplotlib. Install with: pip install matplotlib") from e

    x = list(range(len(summaries)))
    labels = [s.bin_label for s in summaries]
    input_counts = [s.total_seen for s in summaries]
    selected_counts = [s.selected_total for s in summaries]
    selected_negative = [s.selected_negative for s in summaries]
    selected_positive = [s.selected_positive for s in summaries]
    neg_sampling_fraction = [
        0.0 if math.isnan(s.negative_sampling_fraction) else s.negative_sampling_fraction
        for s in summaries
    ]

    width = 0.38
    fig_height = max(6.0, 0.35 * len(summaries) + 3.5)
    fig, axes = plt.subplots(2, 1, figsize=(max(10.0, 0.75 * len(summaries)), fig_height), sharex=True)

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], input_counts, width=width, label="Input records")
    ax.bar([i + width / 2 for i in x], selected_counts, width=width, label="Selected records")
    if log_counts:
        ax.set_yscale("log")
        ax.set_ylabel("Count, log scale")
    else:
        ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)

    # Also show composition of the selected set: positives kept vs sampled negatives.
    ax_comp = ax.twinx()
    ax_comp.plot(x, selected_positive, marker="o", linewidth=1.0, label="Selected positives")
    ax_comp.plot(x, selected_negative, marker="o", linewidth=1.0, label="Selected negatives")
    if log_counts:
        # Avoid log-scale warnings when some bins are zero.
        positive_nonzero = any(v > 0 for v in selected_positive + selected_negative)
        if positive_nonzero:
            ax_comp.set_yscale("log")
    ax_comp.set_ylabel("Selected composition")
    ax_comp.legend(frameon=False, loc="upper left")

    ax = axes[1]
    ax.bar(x, neg_sampling_fraction, width=0.75, label="Negative sampling fraction")
    ax.set_ylabel("Selected negatives / negative input")
    ax.set_xlabel("PalmSite logit bin")
    ax.set_ylim(0, min(1.05, max(0.05, max(neg_sampling_fraction) * 1.15 if neg_sampling_fraction else 1.0)))
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.4)
    ax.legend(frameon=False)

    # Mark the positive threshold if its equivalent logit lies on a finite bin edge.
    if 0.0 < positive_p < 1.0:
        positive_logit = math.log(positive_p / (1.0 - positive_p))
        for i, s in enumerate(summaries):
            if math.isfinite(s.bin_left) and abs(s.bin_left - positive_logit) < 1e-6:
                for a in axes:
                    a.axvline(i - 0.5, linestyle="--", linewidth=1.0, alpha=0.65)
                axes[1].text(
                    i - 0.45,
                    axes[1].get_ylim()[1] * 0.95,
                    f"P={positive_p:g}",
                    rotation=90,
                    va="top",
                    ha="left",
                    fontsize=8,
                )
                break

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()

    for fmt in formats:
        fmt_clean = fmt.lower().lstrip(".")
        out_path = f"{out_prefix}.sampling_by_logit.{fmt_clean}"
        ensure_parent(out_path)
        fig.savefig(out_path, dpi=300 if fmt_clean in {"png", "jpg", "jpeg"} else None, bbox_inches="tight")
        print(f"[summary] wrote sampling plot: {out_path}", file=sys.stderr)
    plt.close(fig)


def parse_formats(values: Optional[Sequence[str]]) -> List[str]:
    if not values:
        return ["pdf", "png"]
    out: List[str] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip().lower().lstrip(".")
            if part:
                out.append(part)
    return out or ["pdf", "png"]


def positive_call(p: float, threshold: float, inclusive: bool) -> bool:
    return p >= threshold if inclusive else p > threshold


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Keep all PalmSite positives and sample PalmSite-negative sequences "
            "from logit bins using streaming reservoir sampling. Also write/plot "
            "sampling frequency by logit bin."
        )
    )

    ap.add_argument("--gff3", required=True, help="PalmSite GFF3 file. Use '-' for stdin.")
    ap.add_argument("--out-tsv", default="-", help="Output selected metadata TSV. Default: stdout.")

    ap.add_argument(
        "--positive-p",
        type=float,
        default=0.9,
        help="Keep all records with P > this threshold by default. Default: 0.9.",
    )
    ap.add_argument(
        "--positive-inclusive",
        action="store_true",
        help="Use P >= --positive-p instead of P > --positive-p.",
    )

    ap.add_argument(
        "--bin-edges",
        default=DEFAULT_BIN_EDGES,
        help=f"Comma-separated logit bin edges. Default: {DEFAULT_BIN_EDGES}",
    )

    ap.add_argument(
        "--bin-sizes",
        default=DEFAULT_BIN_SIZES,
        help=(
            "Comma-separated non-positive sample sizes per bin. "
            "Use 'all' to keep every non-positive record in a bin. "
            f"Default: {DEFAULT_BIN_SIZES}"
        ),
    )

    ap.add_argument(
        "--id-source",
        default="seqid",
        help=(
            "Sequence ID source. Use 'seqid' for GFF3 column 1, "
            "or an attribute name such as ID. Default: seqid."
        ),
    )

    ap.add_argument("--p-attr", default="P", help="GFF3 attribute name for probability. Default: P.")
    ap.add_argument("--logit-attr", default="Logit", help="GFF3 attribute name for logit. Default: Logit.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42.")

    ap.add_argument(
        "--out-bin-summary",
        default=None,
        help=(
            "Optional output TSV summarizing total input count, selected count, and "
            "sampling fraction per logit bin. If omitted and --out-plot-prefix is set, "
            "uses <out-plot-prefix>.bin_summary.tsv."
        ),
    )
    ap.add_argument(
        "--out-plot-prefix",
        default=None,
        help="Optional output prefix for sampling-by-logit plot.",
    )
    ap.add_argument(
        "--plot-formats",
        nargs="*",
        default=["pdf", "png"],
        help="Plot formats for --out-plot-prefix. Default: pdf png.",
    )
    ap.add_argument(
        "--plot-linear-counts",
        action="store_true",
        help="Use a linear y-axis for count plot. Default is log scale.",
    )
    ap.add_argument(
        "--plot-title",
        default="PalmSite logit-bin sampling summary",
        help="Title for the sampling plot.",
    )

    ap.add_argument(
        "--fasta",
        nargs="*",
        default=[],
        help="Optional FASTA files from which to extract selected sequences.",
    )
    ap.add_argument(
        "--out-fasta",
        default=None,
        help="Optional output FASTA path for selected sequences.",
    )
    ap.add_argument(
        "--fasta-id-mode",
        choices=["firstword", "full"],
        default="firstword",
        help="How to match FASTA headers to selected IDs. Default: firstword.",
    )

    args = ap.parse_args()

    if not (0.0 < args.positive_p < 1.0):
        raise ValueError("--positive-p must be between 0 and 1.")

    edges = parse_edges(args.bin_edges)
    n_bins = len(edges) - 1
    bin_sizes = parse_bin_sizes(args.bin_sizes, n_bins)
    formats = parse_formats(args.plot_formats)

    rng = random.Random(args.seed)

    reservoirs: List[List[Record]] = [[] for _ in range(n_bins)]
    total_seen_per_bin = [0 for _ in range(n_bins)]
    positive_seen_per_bin = [0 for _ in range(n_bins)]
    negative_seen_per_bin = [0 for _ in range(n_bins)]

    total_records = 0
    parsed_records = 0
    positive_records = 0
    negative_records = 0
    skipped_records = 0
    out_of_range_records = 0

    positives: List[Record] = []

    with smart_open(args.gff3, "rt") as fh:
        for line in fh:
            total_records += 1

            parsed = parse_gff_record(
                line=line,
                id_source=args.id_source,
                p_attr=args.p_attr,
                logit_attr=args.logit_attr,
            )

            if parsed is None:
                skipped_records += 1
                continue

            parsed_records += 1
            seq_id, p, logit, calibrated_logit, length = parsed

            i = bin_index(logit, edges)
            if i is None:
                out_of_range_records += 1
                # Keep positives even if custom bin edges do not cover them, but
                # they cannot contribute to the bin-summary plot.
                bin_name = "out_of_range"
            else:
                bin_name = bin_label(i, edges)
                total_seen_per_bin[i] += 1

            if positive_call(p, args.positive_p, args.positive_inclusive):
                positive_records += 1
                if i is not None:
                    positive_seen_per_bin[i] += 1
                positives.append(
                    Record(
                        seq_id=seq_id,
                        p=p,
                        logit=logit,
                        calibrated_logit=calibrated_logit,
                        length=length,
                        bin_label=bin_name,
                        decision="keep_positive",
                        gff_line=line.rstrip("\n"),
                    )
                )
                continue

            negative_records += 1
            if i is None:
                # Non-positive records outside user-supplied bin edges are not sampled.
                skipped_records += 1
                continue

            negative_seen_per_bin[i] += 1
            rec = Record(
                seq_id=seq_id,
                p=p,
                logit=logit,
                calibrated_logit=calibrated_logit,
                length=length,
                bin_label=bin_name,
                decision="sampled_negative",
                gff_line=line.rstrip("\n"),
            )

            reservoir_add(
                reservoir=reservoirs[i],
                item=rec,
                seen_count=negative_seen_per_bin[i],
                k=bin_sizes[i],
                rng=rng,
            )

    selected: List[Record] = positives[:]
    for r in reservoirs:
        selected.extend(r)

    # Final de-duplication by sequence ID.
    # If the same sequence appears more than once, keep positives first;
    # otherwise keep the record with the highest logit.
    selected_by_id: Dict[str, Record] = {}
    for r in selected:
        old = selected_by_id.get(r.seq_id)
        if old is None:
            selected_by_id[r.seq_id] = r
        elif old.decision != "keep_positive" and r.decision == "keep_positive":
            selected_by_id[r.seq_id] = r
        elif old.decision == r.decision and r.logit > old.logit:
            selected_by_id[r.seq_id] = r

    selected = list(selected_by_id.values())
    selected.sort(key=lambda r: (r.decision != "keep_positive", r.logit), reverse=False)

    write_tsv(selected, args.out_tsv)

    summaries = build_bin_summaries(
        edges=edges,
        bin_sizes=bin_sizes,
        total_seen_per_bin=total_seen_per_bin,
        positive_seen_per_bin=positive_seen_per_bin,
        negative_seen_per_bin=negative_seen_per_bin,
        selected_records=selected,
    )

    out_bin_summary = args.out_bin_summary
    if out_bin_summary is None and args.out_plot_prefix is not None:
        out_bin_summary = f"{args.out_plot_prefix}.bin_summary.tsv"
    if out_bin_summary is not None:
        ensure_parent(out_bin_summary)
        write_bin_summary_tsv(summaries, out_bin_summary)
        print(f"[summary] wrote bin summary: {out_bin_summary}", file=sys.stderr)

    if args.out_plot_prefix is not None:
        plot_sampling_summary(
            summaries=summaries,
            out_prefix=args.out_plot_prefix,
            formats=formats,
            positive_p=args.positive_p,
            log_counts=(not args.plot_linear_counts),
            title=args.plot_title,
        )

    print(f"[summary] total input lines: {total_records}", file=sys.stderr)
    print(f"[summary] parsed GFF3 records: {parsed_records}", file=sys.stderr)
    print(f"[summary] skipped records: {skipped_records}", file=sys.stderr)
    print(f"[summary] out-of-range parsed records: {out_of_range_records}", file=sys.stderr)
    print(f"[summary] positives kept: {positive_records}", file=sys.stderr)
    print(f"[summary] non-positive records considered for sampling: {negative_records}", file=sys.stderr)
    print(f"[summary] final selected unique IDs: {len(selected)}", file=sys.stderr)

    print("[summary] logit-bin counts:", file=sys.stderr)
    for s in summaries:
        requested = "all" if s.requested_negative_sample_size is None else str(s.requested_negative_sample_size)
        neg_frac = "NA" if math.isnan(s.negative_sampling_fraction) else f"{s.negative_sampling_fraction:.6g}"
        print(
            f"  {s.bin_label}\tseen_total={s.total_seen}\tseen_positive={s.positive_seen}"
            f"\tseen_nonpositive={s.negative_seen}\trequested_nonpositive={requested}"
            f"\tselected_total={s.selected_total}\tselected_positive={s.selected_positive}"
            f"\tselected_nonpositive={s.selected_negative}\tnonpositive_sampling_fraction={neg_frac}",
            file=sys.stderr,
        )

    if args.out_fasta is not None:
        if not args.fasta:
            raise ValueError("--out-fasta was provided but no --fasta files were given.")

        selected_ids = set(selected_by_id.keys())
        n_written = extract_fasta(
            fasta_paths=args.fasta,
            selected_ids=selected_ids,
            out_fasta=args.out_fasta,
            fasta_id_mode=args.fasta_id_mode,
        )
        print(f"[summary] FASTA records written: {n_written}", file=sys.stderr)


if __name__ == "__main__":
    main()

