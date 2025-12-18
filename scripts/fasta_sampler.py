#!/usr/bin/env python3
import argparse
import gzip
import random
import sys
from typing import Iterator, Tuple, List

def fasta_records_gz(path: str) -> Iterator[Tuple[str, str]]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        header = None
        seq_chunks: List[str] = []
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_chunks)
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(line.strip())
        if header is not None:
            yield header, "".join(seq_chunks)

def reservoir_sample_fasta(in_fa: str, out_fa: str, k: int, seed: int,
                           min_len: int, max_len: int) -> None:
    rng = random.Random(seed)
    reservoir: List[Tuple[str, str]] = []
    seen = 0

    for h, s in fasta_records_gz(in_fa):
        L = len(s)
        if L < min_len or L > max_len:
            continue

        seen += 1
        if len(reservoir) < k:
            reservoir.append((h, s))
        else:
            j = rng.randrange(seen)
            if j < k:
                reservoir[j] = (h, s)

    if len(reservoir) < k:
        print(f"[WARN] Only {len(reservoir)} sequences passed filters; requested {k}.", file=sys.stderr)

    with open(out_fa, "wt", encoding="utf-8") as w:
        for h, s in reservoir:
            w.write(f">{h}\n")
            for i in range(0, len(s), 60):
                w.write(s[i:i+60] + "\n")

def main():
    ap = argparse.ArgumentParser(description="Reservoir-sample proteins from FASTA/FASTA.GZ with length filters.")
    ap.add_argument("--in", dest="inp", required=True, help="Input FASTA (.fa/.faa) or gzipped FASTA (.gz)")
    ap.add_argument("--out", dest="out", required=True, help="Output FASTA (uncompressed)")
    ap.add_argument("-n", "--num", type=int, required=True, help="Number of sequences to sample")
    ap.add_argument("--seed", type=int, default=1, help="Random seed (default: 1)")
    ap.add_argument("--min-len", type=int, default=80, help="Minimum AA length (default: 80)")
    ap.add_argument("--max-len", type=int, default=2000, help="Maximum AA length (default: 2000)")
    args = ap.parse_args()

    reservoir_sample_fasta(args.inp, args.out, args.num, args.seed, args.min_len, args.max_len)

if __name__ == "__main__":
    main()
