#!/usr/bin/env python3
"""
assign_fasta_ids.py

Rewrite FASTA headers to unique sequential IDs while preserving the original header
in the description area.

Example output header:
  >MYSEQ_000000000001 original_id original description here

Usage:
  python assign_fasta_ids.py -i in.fasta -o out.fasta --prefix MYSEQ
"""

from __future__ import annotations

import argparse
import sys
from typing import Iterator, Tuple, Optional


def iter_fasta_records(handle) -> Iterator[Tuple[str, str]]:
    """
    Stream FASTA parser.
    Yields: (header_without_gt, sequence_as_one_string)
    """
    header: Optional[str] = None
    seq_chunks: list[str] = []

    for line in handle:
        line = line.rstrip("\n")
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                yield header, "".join(seq_chunks)
            header = line[1:].strip()
            seq_chunks = []
        else:
            if header is None:
                raise ValueError("Input does not look like FASTA: sequence found before any header ('>').")
            seq_chunks.append(line.strip())

    if header is not None:
        yield header, "".join(seq_chunks)


def wrap_sequence(seq: str, width: int) -> str:
    if width <= 0:
        return seq + "\n"
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width)) + "\n"


def count_fasta_records(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                n += 1
    return n


def compute_width(n_records: int, start: int, min_width: int, user_width: Optional[int]) -> int:
    if user_width is not None:
        if user_width <= 0:
            raise ValueError("--width must be a positive integer.")
        return user_width

    if n_records <= 0:
        # No records; fallback to min_width
        return max(1, min_width)

    max_id = start + n_records - 1
    needed = len(str(max_id))
    return max(min_width, needed)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Assign unique sequential IDs to all sequences in a FASTA file, preserving original headers."
    )
    ap.add_argument("-i", "--input", required=True, help="Input FASTA path")
    ap.add_argument("-o", "--output", required=True, help="Output FASTA path")
    ap.add_argument("--prefix", required=True, help="Prefix for new IDs (e.g., MYSEQ)")
    ap.add_argument("--start", type=int, default=1, help="Start index (default: 1)")
    ap.add_argument(
        "--min-width",
        type=int,
        default=12,
        help="Minimum zero-padding width if --width not given (default: 12)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=None,
        help="Exact zero-padding width (overrides --min-width and auto sizing)",
    )
    ap.add_argument(
        "--wrap",
        type=int,
        default=60,
        help="Wrap sequence lines to this width; use 0 to disable wrapping (default: 60)",
    )

    args = ap.parse_args()

    if args.start <= 0:
        ap.error("--start must be >= 1")
    if args.min_width <= 0:
        ap.error("--min-width must be >= 1")
    if args.wrap < 0:
        ap.error("--wrap must be >= 0")

    n_records = count_fasta_records(args.input)
    width = compute_width(n_records=n_records, start=args.start, min_width=args.min_width, user_width=args.width)

    if n_records == 0:
        print(f"WARNING: No FASTA records found in {args.input}", file=sys.stderr)

    written = 0
    current = args.start

    with open(args.input, "r", encoding="utf-8", errors="replace") as fin, open(
        args.output, "w", encoding="utf-8"
    ) as fout:
        for orig_header, seq in iter_fasta_records(fin):
            new_id = f"{args.prefix}_{current:0{width}d}"
            # Preserve original ID + original description exactly as it appeared (minus the leading ">")
            fout.write(f">{new_id} {orig_header}\n")
            fout.write(wrap_sequence(seq, args.wrap))
            written += 1
            current += 1

    print(
        f"Wrote {written} records to {args.output} with prefix='{args.prefix}', start={args.start}, width={width}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

