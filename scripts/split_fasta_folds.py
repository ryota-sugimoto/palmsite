#!/usr/bin/env python3
"""
Split a (large) protein FASTA into K disjoint folds deterministically.

Design goals:
- Streaming / constant memory (works for 10M+ sequences)
- Deterministic fold assignment by hashing (seed + sequence ID)
- Preserves original FASTA headers and sequences
- Safe with large --folds values by keeping only a limited number of output
  FASTA file handles open at once
- Writes fold FASTA files and an optional TSV mapping (seq_id -> fold)

Usage example:
  python split_fasta_folds.py \
    --in neg_10M.faa.gz \
    --out-prefix neg_10M.fold \
    --folds 1024 \
    --seed 42 \
    --map neg_10M.folds.tsv

Outputs:
  neg_10M.fold.0.faa
  neg_10M.fold.1.faa
  ...
  neg_10M.fold.1023.faa
  neg_10M.folds.tsv  (optional)
"""

import argparse
from collections import OrderedDict
import gzip
import hashlib
import sys
from typing import Iterator, Optional, TextIO, Tuple


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
        seq_parts = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:].strip()
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)


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
    return header.split()[0] if header else ""


def assign_fold(seq_id: str, folds: int, seed: int) -> int:
    """Deterministic fold assignment: SHA1(seed + sequence ID) modulo folds."""
    key = f"{seed}\t{seq_id}".encode("utf-8", errors="ignore")
    h = hashlib.sha1(key).digest()
    v = int.from_bytes(h[:8], byteorder="big", signed=False)
    return v % folds


class OutputHandleCache:
    """LRU cache for output FASTA file handles."""

    def __init__(self, paths: list[str], max_open: int):
        if max_open < 1:
            raise ValueError("max_open must be >= 1")
        self.paths = paths
        self.max_open = min(max_open, len(paths))
        self.handles: OrderedDict[int, TextIO] = OrderedDict()

    def initialize_empty_files(self) -> None:
        """Truncate all output files once, then later append records as needed."""
        for path in self.paths:
            with open(path, "wt", encoding="utf-8"):
                pass

    def get(self, fold: int) -> TextIO:
        fh = self.handles.pop(fold, None)
        if fh is not None:
            self.handles[fold] = fh
            return fh

        while len(self.handles) >= self.max_open:
            _, old_fh = self.handles.popitem(last=False)
            old_fh.close()

        fh = open(self.paths[fold], "at", encoding="utf-8")
        self.handles[fold] = fh
        return fh

    def write_record(self, fold: int, header: str, seq: str, wrap: int) -> None:
        fh = self.get(fold)
        fh.write(">" + header + "\n")
        fh.write(wrap_seq(seq, wrap))

    def close_all(self) -> None:
        for fh in self.handles.values():
            fh.close()
        self.handles.clear()


def format_counts(counts: list[int], max_items: int) -> str:
    """Return a compact count summary for stderr progress messages."""
    if max_items <= 0 or len(counts) <= max_items:
        return " ".join(f"fold{i}={counts[i]}" for i in range(len(counts)))
    shown = " ".join(f"fold{i}={counts[i]}" for i in range(max_items))
    return f"{shown} ... fold{len(counts) - 1}={counts[-1]}"


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
    ap.add_argument("--max-open-outputs", type=int, default=64,
                    help="Maximum output FASTA handles kept open at once (default: 64)")
    ap.add_argument("--progress-count-folds", type=int, default=20,
                    help="How many fold counts to print in progress/done messages; 0 prints all (default: 20)")
    args = ap.parse_args()

    if args.folds < 2:
        raise SystemExit("--folds must be >= 2")
    if args.max_open_outputs < 1:
        raise SystemExit("--max-open-outputs must be >= 1")
    if args.progress_count_folds < 0:
        raise SystemExit("--progress-count-folds must be >= 0")

    out_paths = [f"{args.out_prefix}.{i}.faa" for i in range(args.folds)]
    out_cache = OutputHandleCache(out_paths, args.max_open_outputs)
    out_cache.initialize_empty_files()

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
                seq_id = f"__empty_header__:{kept + skipped}"

            fold = assign_fold(seq_id, args.folds, args.seed)
            out_cache.write_record(fold, header, seq, args.wrap)

            if map_fh is not None:
                map_fh.write(f"{seq_id}\t{fold}\n")

            counts[fold] += 1
            kept += 1

            if kept % 1000000 == 0:
                msg = format_counts(counts, args.progress_count_folds)
                print(f"[progress] kept={kept} skipped={skipped} {msg}", file=sys.stderr)

    finally:
        out_cache.close_all()
        if map_fh is not None:
            map_fh.close()

    msg = format_counts(counts, args.progress_count_folds)
    print(f"[done] kept={kept} skipped={skipped} {msg}", file=sys.stderr)
    print("[outputs]", file=sys.stderr)
    for path in out_paths:
        print(f"  {path}", file=sys.stderr)
    if args.map:
        print(f"  {args.map}", file=sys.stderr)


if __name__ == "__main__":
    main()

