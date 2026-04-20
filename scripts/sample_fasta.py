#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 <input.fasta[.gz]> [output_dir] [base_seed]" >&2
  exit 1
fi

infile="$1"
outdir="${2:-samples}"
base_seed="${3:-12345}"

if [[ ! -f "$infile" ]]; then
  echo "Error: input file not found: $infile" >&2
  exit 1
fi

mkdir -p "$outdir"

python3 - "$infile" "$outdir" "$base_seed" <<'PY'
import gzip
import os
import random
import sys
from pathlib import Path

infile = sys.argv[1]
outdir = Path(sys.argv[2])
base_seed = int(sys.argv[3])

K = 10000
REPS = 3
STEP = 1000

def open_maybe_gzip(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt")

def fasta_records(path):
    header = None
    seq_lines = []
    with open_maybe_gzip(path) as fh:
        for line in fh:
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line.rstrip("\n")
                seq_lines = []
            else:
                seq_lines.append(line.rstrip("\n"))
        if header is not None:
            yield header, "".join(seq_lines)

def write_fasta(records, path, width=80):
    with open(path, "w") as out:
        for header, seq in records:
            out.write(header + "\n")
            for i in range(0, len(seq), width):
                out.write(seq[i:i+width] + "\n")

# Independent RNG per replicate
rngs = [random.Random(base_seed + r + 1) for r in range(REPS)]
reservoirs = [[] for _ in range(REPS)]

n_seen = 0
for rec in fasta_records(infile):
    n_seen += 1
    for r in range(REPS):
        res = reservoirs[r]
        rng = rngs[r]

        if len(res) < K:
            res.append(rec)
        else:
            j = rng.randint(1, n_seen)
            if j <= K:
                res[j - 1] = rec

if n_seen == 0:
    sys.exit("Error: no FASTA records found.")

if n_seen < K:
    sys.exit(f"Error: input has only {n_seen} sequences, fewer than required {K}.")

# Shuffle each reservoir once, then emit prefixes 1000..10000
for r in range(REPS):
    order_rng = random.Random(base_seed + 100000 + r + 1)
    order_rng.shuffle(reservoirs[r])

    rep_num = r + 1
    rep_dir = outdir / f"rep{rep_num}"
    rep_dir.mkdir(parents=True, exist_ok=True)

    for n in range(STEP, K + 1, STEP):
        outfile = rep_dir / f"sample_n{n}_rep{rep_num}.fasta"
        write_fasta(reservoirs[r][:n], outfile)

# Write manifest
manifest = outdir / "manifest.tsv"
with open(manifest, "w") as mf:
    mf.write("replicate\tsample_size\toutput_file\n")
    for r in range(1, REPS + 1):
        for n in range(STEP, K + 1, STEP):
            mf.write(f"{r}\t{n}\trep{r}/sample_n{n}_rep{r}.fasta\n")

print(f"Done. Processed {n_seen} sequences.", file=sys.stderr)
print(f"Outputs written under: {outdir}", file=sys.stderr)
PY
