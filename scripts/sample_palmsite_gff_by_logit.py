#!/usr/bin/env python3

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def parse_attributes(attr_text):
    attrs = {}

    if attr_text == "." or attr_text.strip() == "":
        return attrs

    for item in attr_text.strip().split(";"):
        if not item:
            continue

        if "=" not in item:
            attrs[item] = ""
            continue

        key, value = item.split("=", 1)
        attrs[key] = value

    return attrs


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def probability_to_logit(p, eps=1e-12):
    if p is None:
        return None

    p = max(min(p, 1.0 - eps), eps)
    return math.log(p / (1.0 - p))


def parse_gff_line(line, line_no):
    line = line.rstrip("\n")

    fields = line.split("\t")

    if len(fields) != 9:
        fields = line.split(maxsplit=8)

    if len(fields) != 9:
        eprint(f"WARNING: skipping malformed GFF line {line_no}: expected 9 fields")
        return None

    seqid, source, feature_type, start, end, score, strand, phase, attr_text = fields

    try:
        start_i = int(start)
        end_i = int(end)
    except Exception:
        eprint(f"WARNING: skipping line {line_no}: invalid start/end")
        return None

    return {
        "seqid": seqid,
        "source": source,
        "type": feature_type,
        "start": start_i,
        "end": end_i,
        "score": score,
        "strand": strand,
        "phase": phase,
        "attributes": attr_text,
        "gff_line": line,
        "line_no": line_no,
    }


def choose_logit(attrs, score, logit_priority, prob_field):
    for field in logit_priority:
        if field in attrs:
            value = safe_float(attrs[field])
            if value is not None and math.isfinite(value):
                return value, field

    p = None

    if prob_field in attrs:
        p = safe_float(attrs[prob_field])
    elif "P" in attrs:
        p = safe_float(attrs["P"])
    elif "probability" in attrs:
        p = safe_float(attrs["probability"])
    elif "prob" in attrs:
        p = safe_float(attrs["prob"])
    elif score != ".":
        p = safe_float(score)

    logit = probability_to_logit(p)

    if logit is not None and math.isfinite(logit):
        return logit, "computed_from_probability"

    return None, ""


def parse_gff(
    gff_path,
    id_attr="ID",
    logit_priority=None,
    prob_field="P",
    one_record_per_id=True,
    best_by="max_logit",
):
    if logit_priority is None:
        logit_priority = ["CalibratedLogit", "Logit"]

    records = []
    best_by_id = {}

    with open(gff_path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip() or line.startswith("#"):
                continue

            parsed = parse_gff_line(line, line_no)
            if parsed is None:
                continue

            attrs = parse_attributes(parsed["attributes"])

            record_id = attrs.get(id_attr, parsed["seqid"])

            logit, logit_source = choose_logit(
                attrs=attrs,
                score=parsed["score"],
                logit_priority=logit_priority,
                prob_field=prob_field,
            )

            if logit is None:
                eprint(f"WARNING: skipping line {line_no}: no usable logit/probability")
                continue

            p = None
            if prob_field in attrs:
                p = safe_float(attrs[prob_field])
            elif "P" in attrs:
                p = safe_float(attrs["P"])
            elif parsed["score"] != ".":
                p = safe_float(parsed["score"])

            record = {
                "id": record_id,
                "seqid": parsed["seqid"],
                "source": parsed["source"],
                "type": parsed["type"],
                "start": parsed["start"],
                "end": parsed["end"],
                "score": parsed["score"],
                "strand": parsed["strand"],
                "phase": parsed["phase"],
                "attributes": parsed["attributes"],
                "gff_line": parsed["gff_line"],
                "line_no": parsed["line_no"],
                "p": p,
                "logit": logit,
                "logit_source": logit_source,
                "raw_logit": safe_float(attrs.get("Logit", "")),
                "calibrated_logit": safe_float(attrs.get("CalibratedLogit", "")),
                "temperature": safe_float(attrs.get("Temperature", "")),
                "chunk": attrs.get("Chunk", ""),
                "name": attrs.get("Name", ""),
                "mu": safe_float(attrs.get("mu", "")),
                "sigma": safe_float(attrs.get("sigma", "")),
                "length": safe_float(attrs.get("len", "")),
            }

            if not one_record_per_id:
                records.append(record)
                continue

            previous = best_by_id.get(record_id)

            if previous is None:
                best_by_id[record_id] = record
            else:
                if best_by == "max_logit":
                    if record["logit"] > previous["logit"]:
                        best_by_id[record_id] = record
                elif best_by == "max_probability":
                    old_p = previous["p"] if previous["p"] is not None else -1.0
                    new_p = record["p"] if record["p"] is not None else -1.0
                    if new_p > old_p:
                        best_by_id[record_id] = record
                else:
                    raise ValueError(f"Unsupported best_by: {best_by}")

    if one_record_per_id:
        records = list(best_by_id.values())

    return records


def parse_bin_edges(text):
    edges = []

    for value in text.split(","):
        value = value.strip()
        if not value:
            continue
        edges.append(float(value))

    edges = sorted(set(edges))

    if len(edges) < 2:
        raise ValueError("--bin-edges must contain at least two unique numeric values")

    return edges


def make_auto_edges(records, bin_width, logit_min=None, logit_max=None):
    logits = [
        r["logit"]
        for r in records
        if r["logit"] is not None and math.isfinite(r["logit"])
    ]

    if not logits:
        raise ValueError("No finite logits found")

    observed_min = min(logits)
    observed_max = max(logits)

    if logit_min is None:
        logit_min = math.floor(observed_min / bin_width) * bin_width

    if logit_max is None:
        logit_max = math.ceil(observed_max / bin_width) * bin_width

    if logit_max <= logit_min:
        logit_max = logit_min + bin_width

    edges = []
    x = logit_min

    while x < logit_max:
        edges.append(round(x, 10))
        x += bin_width

    edges.append(round(logit_max, 10))

    return edges


def assign_bin(value, edges):
    if value is None or not math.isfinite(value):
        return None

    if value < edges[0] or value > edges[-1]:
        return None

    for i in range(len(edges) - 1):
        left = edges[i]
        right = edges[i + 1]

        is_last_bin = i == len(edges) - 2

        if is_last_bin:
            if left <= value <= right:
                return i
        else:
            if left <= value < right:
                return i

    return None


def format_bin_label(bin_index, edges):
    if bin_index is None:
        return "out_of_range"

    left = edges[bin_index]
    right = edges[bin_index + 1]

    if bin_index == len(edges) - 2:
        return f"[{left:g},{right:g}]"

    return f"[{left:g},{right:g})"


def group_records_by_bin(records, edges):
    bins = defaultdict(list)
    out_of_range = []

    for record in records:
        bin_index = assign_bin(record["logit"], edges)

        if bin_index is None:
            out_of_range.append(record)
        else:
            bins[bin_index].append(record)

    return bins, out_of_range


def allocate_total_across_nonempty_bins(bins, total, n_bins):
    nonempty_bins = [
        bin_index
        for bin_index in range(n_bins)
        if len(bins.get(bin_index, [])) > 0
    ]

    if not nonempty_bins:
        return {}

    base = total // len(nonempty_bins)
    remainder = total % len(nonempty_bins)

    allocation = {}

    for i, bin_index in enumerate(nonempty_bins):
        allocation[bin_index] = base
        if i < remainder:
            allocation[bin_index] += 1

    return allocation


def sample_records(
    records,
    edges,
    per_bin=None,
    total=None,
    seed=42,
    take_all_above=None,
):
    rng = random.Random(seed)

    bins, out_of_range = group_records_by_bin(records, edges)
    n_bins = len(edges) - 1

    forced_by_id = {}

    if take_all_above is not None:
        for record in records:
            if record["logit"] >= take_all_above:
                forced_by_id[record["id"]] = record

    if per_bin is not None:
        allocation = {bin_index: per_bin for bin_index in range(n_bins)}
    elif total is not None:
        allocation = allocate_total_across_nonempty_bins(
            bins=bins,
            total=total,
            n_bins=n_bins,
        )
    else:
        raise ValueError("Either per_bin or total must be specified")

    sampled_by_id = dict(forced_by_id)

    for bin_index in range(n_bins):
        available = [
            record
            for record in bins.get(bin_index, [])
            if record["id"] not in forced_by_id
        ]

        n_to_sample = allocation.get(bin_index, 0)

        if n_to_sample <= 0 or not available:
            continue

        if len(available) <= n_to_sample:
            chosen = list(available)
        else:
            chosen = rng.sample(available, n_to_sample)

        for record in chosen:
            sampled_by_id[record["id"]] = record

    sampled = list(sampled_by_id.values())
    sampled.sort(key=lambda r: (r["logit"], r["id"]))

    sampled_ids = set(r["id"] for r in sampled)

    bin_stats = []

    for bin_index in range(n_bins):
        available_records = bins.get(bin_index, [])
        sampled_records_in_bin = [
            r for r in available_records if r["id"] in sampled_ids
        ]

        bin_stats.append({
            "bin_index": bin_index,
            "bin_left": edges[bin_index],
            "bin_right": edges[bin_index + 1],
            "bin_label": format_bin_label(bin_index, edges),
            "available": len(available_records),
            "sampled": len(sampled_records_in_bin),
        })

    return sampled, bin_stats, out_of_range


def read_fasta(fasta_path):
    seqs = {}
    current_id = None
    current_header = None
    current_lines = []

    with open(fasta_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")

            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = {
                        "header": current_header,
                        "sequence": "".join(current_lines),
                    }

                current_header = line[1:]
                current_id = current_header.split()[0]
                current_lines = []
            else:
                current_lines.append(line.strip())

        if current_id is not None:
            seqs[current_id] = {
                "header": current_header,
                "sequence": "".join(current_lines),
            }

    return seqs


def write_sampled_fasta(fasta_path, sampled_records, output_path):
    seqs = read_fasta(fasta_path)

    found = 0
    missing = 0

    with open(output_path, "w", encoding="utf-8", newline="\n") as out:
        for record in sampled_records:
            seq_id = record["id"]

            if seq_id not in seqs:
                missing += 1
                continue

            header = seqs[seq_id]["header"]
            sequence = seqs[seq_id]["sequence"]

            out.write(f">{header}\n")

            for i in range(0, len(sequence), 80):
                out.write(sequence[i:i + 80] + "\n")

            found += 1

    eprint(f"Wrote sampled FASTA: {output_path}")
    eprint(f"FASTA matched sampled IDs: {found}")
    eprint(f"FASTA missing sampled IDs: {missing}")


def write_tsv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="\n") as out:
        writer = csv.DictWriter(
            out,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_outputs(sampled, bin_stats, out_of_range, edges, out_prefix, fasta_path=None):
    out_prefix = Path(out_prefix)

    sampled_ids_path = f"{out_prefix}.sampled_ids.txt"
    sampled_tsv_path = f"{out_prefix}.sampled_records.tsv"
    bin_counts_path = f"{out_prefix}.bin_counts.tsv"
    sampled_gff_path = f"{out_prefix}.sampled.gff3"
    out_of_range_path = f"{out_prefix}.out_of_range.tsv"

    with open(sampled_ids_path, "w", encoding="utf-8", newline="\n") as out:
        for record in sampled:
            out.write(record["id"] + "\n")

    sampled_rows = []

    for record in sampled:
        bin_index = assign_bin(record["logit"], edges)

        sampled_rows.append({
            "id": record["id"],
            "seqid": record["seqid"],
            "logit": f"{record['logit']:.8g}",
            "logit_source": record["logit_source"],
            "raw_logit": "" if record["raw_logit"] is None else f"{record['raw_logit']:.8g}",
            "calibrated_logit": "" if record["calibrated_logit"] is None else f"{record['calibrated_logit']:.8g}",
            "P": "" if record["p"] is None else f"{record['p']:.8g}",
            "temperature": "" if record["temperature"] is None else f"{record['temperature']:.8g}",
            "bin_index": "" if bin_index is None else bin_index,
            "bin_label": format_bin_label(bin_index, edges),
            "start": record["start"],
            "end": record["end"],
            "span_len": record["end"] - record["start"] + 1,
            "mu": "" if record["mu"] is None else f"{record['mu']:.8g}",
            "sigma": "" if record["sigma"] is None else f"{record['sigma']:.8g}",
            "length": "" if record["length"] is None else f"{record['length']:.8g}",
            "chunk": record["chunk"],
            "name": record["name"],
            "line_no": record["line_no"],
        })

    write_tsv(
        sampled_tsv_path,
        sampled_rows,
        [
            "id",
            "seqid",
            "logit",
            "logit_source",
            "raw_logit",
            "calibrated_logit",
            "P",
            "temperature",
            "bin_index",
            "bin_label",
            "start",
            "end",
            "span_len",
            "mu",
            "sigma",
            "length",
            "chunk",
            "name",
            "line_no",
        ],
    )

    write_tsv(
        bin_counts_path,
        bin_stats,
        [
            "bin_index",
            "bin_left",
            "bin_right",
            "bin_label",
            "available",
            "sampled",
        ],
    )

    with open(sampled_gff_path, "w", encoding="utf-8", newline="\n") as out:
        out.write("##gff-version 3\n")
        for record in sampled:
            out.write(record["gff_line"] + "\n")

    out_rows = []

    for record in out_of_range:
        out_rows.append({
            "id": record["id"],
            "seqid": record["seqid"],
            "logit": f"{record['logit']:.8g}",
            "P": "" if record["p"] is None else f"{record['p']:.8g}",
            "line_no": record["line_no"],
        })

    write_tsv(
        out_of_range_path,
        out_rows,
        [
            "id",
            "seqid",
            "logit",
            "P",
            "line_no",
        ],
    )

    eprint(f"Wrote: {sampled_ids_path}")
    eprint(f"Wrote: {sampled_tsv_path}")
    eprint(f"Wrote: {bin_counts_path}")
    eprint(f"Wrote: {sampled_gff_path}")
    eprint(f"Wrote: {out_of_range_path}")

    if fasta_path is not None:
        sampled_fasta_path = f"{out_prefix}.sampled.faa"
        write_sampled_fasta(fasta_path, sampled, sampled_fasta_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample sequences from PalmSite GFF3 output across logit bins. "
            "Default behavior uses CalibratedLogit first, then Logit, then computes "
            "logit from P if needed."
        )
    )

    parser.add_argument(
        "--gff",
        required=True,
        help="PalmSite GFF3 output."
    )
    parser.add_argument(
        "--out-prefix",
        required=True,
        help="Output prefix."
    )
    parser.add_argument(
        "--fasta",
        default=None,
        help="Optional FASTA file. If provided, sampled sequences are written to FASTA."
    )
    parser.add_argument(
        "--id-attr",
        default="ID",
        help="GFF attribute used as sequence ID. Default: ID."
    )
    parser.add_argument(
        "--prob-field",
        default="P",
        help="GFF attribute used as probability field. Default: P."
    )
    parser.add_argument(
        "--logit-priority",
        default="CalibratedLogit,Logit",
        help=(
            "Comma-separated logit fields in priority order. "
            "Default: CalibratedLogit,Logit."
        )
    )
    parser.add_argument(
        "--all-records",
        action="store_true",
        help=(
            "Use all GFF records instead of keeping one best record per ID. "
            "Default: keep max-logit record per ID."
        )
    )

    sample_group = parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument(
        "--per-bin",
        type=int,
        default=None,
        help="Number of sequences to sample from each logit bin."
    )
    sample_group.add_argument(
        "--total",
        type=int,
        default=None,
        help=(
            "Total number of sequences to sample, distributed as evenly as possible "
            "across non-empty bins."
        )
    )

    bin_group = parser.add_mutually_exclusive_group(required=False)
    bin_group.add_argument(
        "--bin-edges",
        default=None,
        help=(
            "Comma-separated logit bin edges. "
            "Example: '-12,-10,-8,-6,-4,-2,0,1,2,2.24,3,5,8,12'"
        )
    )
    bin_group.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help="Uniform logit bin width. Example: 1.0"
    )

    parser.add_argument(
        "--logit-min",
        type=float,
        default=None,
        help="Minimum logit for automatic --bin-width bins."
    )
    parser.add_argument(
        "--logit-max",
        type=float,
        default=None,
        help="Maximum logit for automatic --bin-width bins."
    )
    parser.add_argument(
        "--take-all-above",
        type=float,
        default=None,
        help=(
            "Always include all records with logit >= this value, in addition to "
            "regular bin sampling. For example, 2.24 keeps all above the PalmSite "
            "positive threshold if P threshold is 0.904."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42."
    )

    args = parser.parse_args()

    logit_priority = [
        x.strip()
        for x in args.logit_priority.split(",")
        if x.strip()
    ]

    records = parse_gff(
        gff_path=args.gff,
        id_attr=args.id_attr,
        logit_priority=logit_priority,
        prob_field=args.prob_field,
        one_record_per_id=not args.all_records,
        best_by="max_logit",
    )

    if not records:
        eprint("ERROR: no usable PalmSite records found")
        sys.exit(1)

    eprint(f"Parsed usable records: {len(records)}")
    eprint(f"Logit priority: {','.join(logit_priority)}")

    if args.bin_edges is not None:
        edges = parse_bin_edges(args.bin_edges)
    else:
        bin_width = args.bin_width if args.bin_width is not None else 1.0
        edges = make_auto_edges(
            records=records,
            bin_width=bin_width,
            logit_min=args.logit_min,
            logit_max=args.logit_max,
        )

    eprint("Logit bin edges:")
    eprint(",".join(f"{x:g}" for x in edges))

    sampled, bin_stats, out_of_range = sample_records(
        records=records,
        edges=edges,
        per_bin=args.per_bin,
        total=args.total,
        seed=args.seed,
        take_all_above=args.take_all_above,
    )

    eprint(f"Sampled records: {len(sampled)}")
    eprint(f"Out-of-range records: {len(out_of_range)}")

    write_outputs(
        sampled=sampled,
        bin_stats=bin_stats,
        out_of_range=out_of_range,
        edges=edges,
        out_prefix=args.out_prefix,
        fasta_path=args.fasta,
    )


if __name__ == "__main__":
    main()
