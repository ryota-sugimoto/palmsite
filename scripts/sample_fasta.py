#!/usr/bin/env python3

import argparse
import random
from pathlib import Path
from Bio import SeqIO


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample sequences from a FASTA using Biopython disk-backed indexing."
    )
    parser.add_argument("input_fasta", help="Input FASTA file (preferably uncompressed)")
    parser.add_argument(
        "-o", "--outdir", default="samples_by_id",
        help="Output directory [default: samples_by_id]"
    )
    parser.add_argument(
        "--index-db", default=None,
        help="SQLite index filename [default: <outdir>/<input>.idx.sqlite]"
    )
    parser.add_argument(
        "--start", type=int, default=1000,
        help="Starting sample size [default: 1000]"
    )
    parser.add_argument(
        "--end", type=int, default=10000,
        help="Ending sample size [default: 10000]"
    )
    parser.add_argument(
        "--step", type=int, default=1000,
        help="Step size [default: 1000]"
    )
    parser.add_argument(
        "--replicates", type=int, default=3,
        help="Number of independent replicates [default: 3]"
    )
    parser.add_argument(
        "--base-seed", type=int, default=12345,
        help="Base random seed [default: 12345]"
    )

    args = parser.parse_args()

    input_fasta = Path(args.input_fasta)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.index_db is None:
        index_db = outdir / f"{input_fasta.name}.idx.sqlite"
    else:
        index_db = Path(args.index_db)

    print(f"[INFO] Building/loading index: {index_db}")
    idx = SeqIO.index_db(str(index_db), str(input_fasta), "fasta")

    ids = list(idx.keys())
    total = len(ids)
    print(f"[INFO] Total sequences indexed: {total}")

    if total == 0:
        raise SystemExit("Error: no sequences found in FASTA")

    manifest_path = outdir / "manifest.tsv"
    with open(manifest_path, "w") as mf:
        mf.write("sample_size\treplicate\tseed\toutfile\n")

        for n in range(args.start, args.end + 1, args.step):
            if n > total:
                raise SystemExit(
                    f"Error: requested sample size {n}, but FASTA only has {total} sequences"
                )

            for rep in range(1, args.replicates + 1):
                seed = args.base_seed + n * 1000 + rep
                rng = random.Random(seed)

                sampled_ids = rng.sample(ids, n)

                outfile = outdir / f"sample_n{n}_rep{rep}.fasta"
                print(f"[INFO] Writing {outfile} (n={n}, rep={rep}, seed={seed})")

                with open(outfile, "w") as out_handle:
                    SeqIO.write((idx[seq_id] for seq_id in sampled_ids), out_handle, "fasta")

                mf.write(f"{n}\t{rep}\t{seed}\t{outfile.name}\n")

    idx.close()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
