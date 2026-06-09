#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


DIAMOND_COLUMNS = [
    "qseqid",
    "sseqid",
    "pident",
    "length",
    "qlen",
    "slen",
    "qstart",
    "qend",
    "sstart",
    "send",
    "evalue",
    "bitscore",
    "staxids",
    "sscinames",
    "stitle",
]


RDRP_TITLE_RE = re.compile(
    r"("
    r"RNA[- ]dependent RNA polymerase|"
    r"RNA[- ]directed RNA polymerase|"
    r"RdRp|"
    r"replicase|"
    r"replication[- ]associated|"
    r"polymerase|"
    r"L protein|"
    r"large protein|"
    r"polyprotein"
    r")",
    re.IGNORECASE,
)


def parse_float(value: str, default: float = float("nan")) -> float:
    try:
        return float(value)
    except ValueError:
        return default


def parse_taxids(raw: str) -> list[str]:
    taxids = []
    for part in raw.strip().split(";"):
        part = part.strip()
        if part.isdigit() and part != "0":
            taxids.append(part)
    return taxids


def is_rdrp_like_title(title: str) -> bool:
    return bool(RDRP_TITLE_RE.search(title or ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter DIAMOND RefSeq hits and generate per-query taxid lists "
            "for conservative LCA assignment."
        )
    )
    parser.add_argument(
        "--diamond",
        required=True,
        type=Path,
        help="DIAMOND outfmt 6 TSV without header.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output TSV: qseqid, taxids_for_lca, retained hit summary.",
    )
    parser.add_argument(
        "--min-query-cov",
        type=float,
        default=0.50,
        help="Minimum query coverage: alignment length / query length. Default: 0.50.",
    )
    parser.add_argument(
        "--min-subject-cov",
        type=float,
        default=0.0,
        help=(
            "Minimum subject coverage: alignment length / subject length. "
            "Default: 0.0. For span queries, query coverage is usually more useful."
        ),
    )
    parser.add_argument(
        "--min-pident",
        type=float,
        default=0.0,
        help="Minimum percent identity. Default: 0.0.",
    )
    parser.add_argument(
        "--max-evalue",
        type=float,
        default=1e-5,
        help="Maximum e-value. Default: 1e-5.",
    )
    parser.add_argument(
        "--min-relative-bitscore",
        type=float,
        default=0.90,
        help=(
            "Keep hits with bitscore >= this fraction of the best bitscore "
            "for the query. Default: 0.90."
        ),
    )
    parser.add_argument(
        "--max-hits-per-query",
        type=int,
        default=50,
        help="Maximum retained hits per query after filtering. Default: 50.",
    )
    parser.add_argument(
        "--require-rdrp-like-title",
        action="store_true",
        help=(
            "Keep only hits whose title looks polymerase/replicase/RdRP-like. "
            "Use cautiously because RefSeq titles are inconsistent."
        ),
    )
    args = parser.parse_args()

    hits_by_query: dict[str, list[dict[str, str]]] = defaultdict(list)

    with args.diamond.open("r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for fields in reader:
            if not fields:
                continue
            if len(fields) < len(DIAMOND_COLUMNS):
                raise ValueError(
                    f"Expected at least {len(DIAMOND_COLUMNS)} columns, got {len(fields)}: {fields}"
                )

            row = dict(zip(DIAMOND_COLUMNS, fields[:len(DIAMOND_COLUMNS)]))

            qlen = parse_float(row["qlen"])
            slen = parse_float(row["slen"])
            aln_len = parse_float(row["length"])
            evalue = parse_float(row["evalue"])
            bitscore = parse_float(row["bitscore"])
            pident = parse_float(row["pident"])

            if not math.isfinite(qlen) or qlen <= 0:
                continue
            if not math.isfinite(slen) or slen <= 0:
                continue
            if not math.isfinite(aln_len) or aln_len <= 0:
                continue
            if not math.isfinite(evalue):
                continue
            if not math.isfinite(bitscore):
                continue
            if not math.isfinite(pident):
                continue

            row["query_cov"] = f"{aln_len / qlen:.6f}"
            row["subject_cov"] = f"{aln_len / slen:.6f}"

            hits_by_query[row["qseqid"]].append(row)

    fieldnames = [
        "qseqid",
        "taxids_for_lca",
        "n_retained_hits",
        "n_unique_taxids",
        "best_sseqid",
        "best_taxid",
        "best_ssciname",
        "best_stitle",
        "best_pident",
        "best_query_cov",
        "best_subject_cov",
        "best_evalue",
        "best_bitscore",
        "retained_sseqids",
        "retained_taxids",
        "retained_bitscores",
        "retained_pidents",
        "retained_query_covs",
        "note",
    ]

    with args.output.open("w", newline="") as fout:
        writer = csv.DictWriter(
            fout,
            delimiter="\t",
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()

        for qseqid in sorted(hits_by_query):
            hits = hits_by_query[qseqid]
            hits.sort(key=lambda r: parse_float(r["bitscore"]), reverse=True)

            best_bitscore = parse_float(hits[0]["bitscore"])
            retained = []

            for row in hits:
                evalue = parse_float(row["evalue"])
                bitscore = parse_float(row["bitscore"])
                pident = parse_float(row["pident"])
                query_cov = parse_float(row["query_cov"])
                subject_cov = parse_float(row["subject_cov"])
                taxids = parse_taxids(row["staxids"])

                if not taxids:
                    continue
                if evalue > args.max_evalue:
                    continue
                if query_cov < args.min_query_cov:
                    continue
                if subject_cov < args.min_subject_cov:
                    continue
                if pident < args.min_pident:
                    continue
                if best_bitscore > 0 and bitscore < args.min_relative_bitscore * best_bitscore:
                    continue
                if args.require_rdrp_like_title and not is_rdrp_like_title(row["stitle"]):
                    continue

                retained.append(row)
                if len(retained) >= args.max_hits_per_query:
                    break

            if not retained:
                best = hits[0]
                writer.writerow(
                    {
                        "qseqid": qseqid,
                        "taxids_for_lca": "",
                        "n_retained_hits": "0",
                        "n_unique_taxids": "0",
                        "best_sseqid": best["sseqid"],
                        "best_taxid": parse_taxids(best["staxids"])[0] if parse_taxids(best["staxids"]) else "",
                        "best_ssciname": best["sscinames"],
                        "best_stitle": best["stitle"],
                        "best_pident": best["pident"],
                        "best_query_cov": best["query_cov"],
                        "best_subject_cov": best["subject_cov"],
                        "best_evalue": best["evalue"],
                        "best_bitscore": best["bitscore"],
                        "retained_sseqids": "",
                        "retained_taxids": "",
                        "retained_bitscores": "",
                        "retained_pidents": "",
                        "retained_query_covs": "",
                        "note": "no_retained_hits_after_filtering",
                    }
                )
                continue

            unique_taxids_ordered = []
            seen = set()
            retained_taxid_per_hit = []

            for row in retained:
                taxids = parse_taxids(row["staxids"])
                selected_taxid = taxids[0] if taxids else ""
                retained_taxid_per_hit.append(selected_taxid)
                if selected_taxid and selected_taxid not in seen:
                    seen.add(selected_taxid)
                    unique_taxids_ordered.append(selected_taxid)

            best = retained[0]
            best_taxids = parse_taxids(best["staxids"])

            writer.writerow(
                {
                    "qseqid": qseqid,
                    "taxids_for_lca": ",".join(unique_taxids_ordered),
                    "n_retained_hits": str(len(retained)),
                    "n_unique_taxids": str(len(unique_taxids_ordered)),
                    "best_sseqid": best["sseqid"],
                    "best_taxid": best_taxids[0] if best_taxids else "",
                    "best_ssciname": best["sscinames"],
                    "best_stitle": best["stitle"],
                    "best_pident": best["pident"],
                    "best_query_cov": best["query_cov"],
                    "best_subject_cov": best["subject_cov"],
                    "best_evalue": best["evalue"],
                    "best_bitscore": best["bitscore"],
                    "retained_sseqids": ",".join(row["sseqid"] for row in retained),
                    "retained_taxids": ",".join(retained_taxid_per_hit),
                    "retained_bitscores": ",".join(row["bitscore"] for row in retained),
                    "retained_pidents": ",".join(row["pident"] for row in retained),
                    "retained_query_covs": ",".join(row["query_cov"] for row in retained),
                    "note": "ok",
                }
            )


if __name__ == "__main__":
    main()
