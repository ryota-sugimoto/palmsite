#!/usr/bin/env python3
"""
Split a (large) protein FASTA into K disjoint folds deterministically.

Design goals:
- Streaming / constant memory (works for 10M+ sequences)
- Deterministic fold assignment by hashing (seed + sequence ID)
- Preserves original FASTA headers and sequences
- Writes fold FASTA files and an optional TSV mapping (seq_id -> fold)

Usage example:
  python split_fasta_folds.py \
    --in neg_10M.faa.gz \
    --out-prefix neg_10M.fold \
    --folds 3 \
    --seed 42 \
    --map neg_10M.folds.tsv

Outputs:
  neg_10M.fold.0.faa
  neg_10M.fold.1.faa
  neg_10M.fold.2.faa
  neg_10M.folds.tsv  (optional)
"""

import argparse
import gzip
import hashlib
import os
import sys
from typing import Iterator, Tuple, Optional, TextIO


def open_text_auto(path: str, mode: str = "rt") -> TextIO:
    """Open plain text or .gz transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8", errors="replace")  # type: ignore[arg-type]
    return open(path, mode, encoding="utf-8", errors="replace")


def fasta_iter(path: str) -> Iterator[Tuple[str, str]]:
    """
    Yield (header_without_gt, sequence_string) from a FASTA(.gz).
    Header is the full header line without leading '>'.
    """
    with open_text_auto(path, "rt") as f:
        header: Optional[str] = None
        seq_chunks = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line)
        if header is not None:
            yield header, "".join(seq_chunks)


def wrap_seq(seq: str, width: int) -> str:
    if width <= 0:
        return seq + "\n"
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width)) + "\n"


def get_seq_id(header: str, id_mode: str) -> str:
    """
    id_mode:
      - token: first whitespace-delimited token (recommended; most tools use this)
      - full: entire header line
    """
    if id_mode == "full":
        return header
    # token
    return header.split()[0] if header else ""


def assign_fold(seq_id: str, folds: int, seed: int) -> int:
    """
    Deterministic fold assignment: fold = SHA1(f"{seed}\t{seq_id}") % folds
    """
    key = f"{seed}\t{seq_id}".encode("utf-8", errors="ignore")
    h = hashlib.sha1(key).digest()
    # Use first 8 bytes as an integer for speed
    v = int.from_bytes(h[:8], byteorder="big", signed=False)
    return v % folds


def main() -> None:
    ap = argparse.ArgumentParser(description="Split FASTA into deterministic K-folds (streaming).")
    ap.add_argument("--in", dest="inp", required=True, help="Input FASTA (.fa/.faa) or gzipped FASTA (.gz).")
    ap.add_argument("--out-prefix", required=True, help="Output prefix; writes <prefix>.<fold>.faa")
    ap.add_argument("--folds", type=int, default=3, help="Number of folds (default: 3)")
    ap.add_argument("--seed", type=int, default=42, help="Hash seed (default: 42)")
    ap.add_argument("--map", default=None, help="Optional TSV mapping output: seq_id<TAB>fold")
    ap.add_argument("--id-mode", choices=["token", "full"], default="token",
                    help="How to define sequence ID for hashing/mapping (default: token)")
    ap.add_argument("--min-len", type=int, default=0, help="Optional: skip sequences shorter than this (AA)")
    ap.add_argument("--max-len", type=int, default=0, help="Optional: skip sequences longer than this (AA); 0=off")
    ap.add_argument("--wrap", type=int, default=60, help="FASTA line wrap width (default: 60; 0 disables)")
    args = ap.parse_args()

    if args.folds < 2:
        raise SystemExit("--folds must be >= 2")

    out_paths = [f"{args.out_prefix}.{i}.faa" for i in range(args.folds)]
    outs = [open(p, "wt", encoding="utf-8") for p in out_paths]

    map_fh: Optional[TextIO] = None
    if args.map:
        map_fh = open(args.map, "wt", encoding="utf-8")
        map_fh.write("seq_id\tfold\n")

    counts = [0] * args.folds
    kept = 0
    skipped = 0

    try:
        for header, seq in fasta_iter(args.inp):
            L = len(seq)
            if L < args.min_len:
                skipped += 1
                continue
            if args.max_len and L > args.max_len:
                skipped += 1
                continue

            seq_id = get_seq_id(header, args.id_mode)
            if not seq_id:
                # Fallback to a stable identifier if header is empty/unusual
                seq_id = f"__empty_header__:{kept + skipped}"

            fold = assign_fold(seq_id, args.folds, args.seed)

            outs[fold].write(">" + header + "\n")
            outs[fold].write(wrap_seq(seq, args.wrap))

            if map_fh is not None:
                map_fh.write(f"{seq_id}\t{fold}\n")

            counts[fold] += 1
            kept += 1

            if kept % 1000000 == 0:
                msg = " ".join([f"fold{i}={counts[i]}" for i in range(args.folds)])
                print(f"[progress] kept={kept} skipped={skipped} {msg}", file=sys.stderr)

    finally:
        for fh in outs:
            fh.close()
        if map_fh is not None:
            map_fh.close()

    msg = " ".join([f"fold{i}={counts[i]}" for i in range(args.folds)])
    print(f"[done] kept={kept} skipped={skipped} {msg}", file=sys.stderr)
    print("[outputs]", file=sys.stderr)
    for p in out_paths:
        print(f"  {p}", file=sys.stderr)
    if args.map:
        print(f"  {args.map}", file=sys.stderr)


if __name__ == "__main__":
    main()
