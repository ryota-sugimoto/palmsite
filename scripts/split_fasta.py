#!/usr/bin/env python3
"""
split_fasta.py

Split an input FASTA into multiple FASTA files, each containing up to N sequences.
Optionally shuffle sequence order (reproducible with --seed).

Examples:
  # Split into files of max 1000 sequences
  python split_fasta.py -i input.fasta -o out_dir --prefix chunk --max-seqs 1000

  # Shuffle before splitting (reproducible)
  python split_fasta.py -i input.fasta -o out_dir --prefix chunk --max-seqs 1000 --shuffle --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class FastaRecord:
    header: str   # header line without leading ">"
    seq: str      # sequence, no whitespace


def read_fasta(path: str) -> List[FastaRecord]:
    """
    Read a FASTA file into a list of FastaRecord.

    - Supports multi-line sequences.
    - Keeps the full header (everything after '>') unchanged.
    """
    records: List[FastaRecord] = []
    header: Optional[str] = None
    seq_parts: List[str] = []

    def flush():
        nonlocal header, seq_parts
        if header is not None:
            seq = "".join(seq_parts).replace(" ", "").replace("\t", "")
            records.append(FastaRecord(header=header, seq=seq))
        header = None
        seq_parts = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
            else:
                if header is None:
                    raise ValueError(f"FASTA parse error: sequence line encountered before header in {path}")
                seq_parts.append(line.strip())

    flush()
    return records


def write_fasta(records: Iterable[FastaRecord], path: str, wrap: int = 0) -> None:
    """
    Write records to FASTA.

    wrap:
      - 0 means no wrapping (one line per sequence).
      - otherwise wrap sequence lines at this width.
    """
    with open(path, "w", encoding="utf-8") as out:
        for r in records:
            out.write(f">{r.header}\n")
            if wrap and wrap > 0:
                s = r.seq
                for i in range(0, len(s), wrap):
                    out.write(s[i:i + wrap] + "\n")
            else:
                out.write(r.seq + "\n")


def split_records(records: List[FastaRecord], max_seqs: int) -> List[List[FastaRecord]]:
    if max_seqs <= 0:
        raise ValueError("--max-seqs must be a positive integer")
    return [records[i:i + max_seqs] for i in range(0, len(records), max_seqs)]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Split an input FASTA into multiple FASTA files with up to N sequences each. Optional shuffle."
    )
    p.add_argument("-i", "--input", required=True, help="Input FASTA file")
    p.add_argument("-o", "--out-dir", required=True, help="Output directory")
    p.add_argument("--prefix", default="split", help="Output file prefix (default: split)")
    p.add_argument("--max-seqs", type=int, required=True, help="Maximum number of sequences per output FASTA")
    p.add_argument("--shuffle", action="store_true", help="Shuffle sequence order before splitting")
    p.add_argument("--seed", type=int, default=None, help="Random seed for --shuffle (default: non-deterministic)")
    p.add_argument("--wrap", type=int, default=0, help="Wrap sequence lines to this width (0 = no wrap; default: 0)")
    p.add_argument("--dry-run", action="store_true", help="Print planned outputs but do not write files")

    args = p.parse_args(argv)

    if not os.path.isfile(args.input):
        print(f"ERROR: input FASTA not found: {args.input}", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)

    records = read_fasta(args.input)
    if not records:
        print("ERROR: no FASTA records found.", file=sys.stderr)
        return 2

    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(records)

    chunks = split_records(records, args.max_seqs)

    # Determine padding for file numbering (at least 6 digits, or enough for total chunks)
    n_chunks = len(chunks)
    pad = max(6, len(str(n_chunks)))

    planned = []
    for idx, chunk in enumerate(chunks, start=1):
        out_name = f"{args.prefix}_{idx:0{pad}d}.fasta"
        out_path = os.path.join(args.out_dir, out_name)
        planned.append((out_path, len(chunk)))

    # Report plan
    total = len(records)
    print(f"Input: {args.input}")
    print(f"Total sequences: {total}")
    print(f"Max per file: {args.max_seqs}")
    print(f"Shuffle: {bool(args.shuffle)}" + (f" (seed={args.seed})" if args.shuffle else ""))
    print(f"Output dir: {args.out_dir}")
    print(f"Chunks: {n_chunks}")
    for out_path, n in planned:
        print(f"  {out_path}  (n={n})")

    if args.dry_run:
        return 0

    # Write files
    for (out_path, _), chunk in zip(planned, chunks):
        write_fasta(chunk, out_path, wrap=args.wrap)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

