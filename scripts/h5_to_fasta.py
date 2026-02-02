#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export chunk sequences from HDF5 to FASTA.

Assumes HDF5 layout like:
  /items/<chunk_id>/seq   () vlen string
"""
import os, sys, argparse, logging, h5py

def setup_logging(level="INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("h5_to_fasta")

def main():
    ap = argparse.ArgumentParser(description="Dump HDF5 sequences to FASTA")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--match", default=None, help="Only emit chunk_ids containing this substring")
    args = ap.parse_args()
    log = setup_logging(args.log_level)

    if not os.path.exists(args.h5):
        log.error("No such H5: %s", args.h5); sys.exit(1)

    n = 0; emitted = 0
    with h5py.File(args.h5, "r") as f, open(args.fasta, "w", encoding="utf-8") as out:
        if "items" not in f:
            log.error("Missing group '/items' in %s", args.h5); sys.exit(2)
        for cid in sorted(f["items"].keys()):
            n += 1
            if args.match and args.match not in cid:
                continue
            g = f["items"][cid]
            if "seq" not in g:
                log.warning("Skipping (no seq): %s", cid); continue
            try:
                seq = g["seq"].asstr()[()]
            except Exception:
                seq = g["seq"][()].decode() if isinstance(g["seq"][()], bytes) else str(g["seq"][()])
            if not seq:
                log.warning("Empty seq: %s", cid); continue
            out.write(f">{cid}\n")
            # wrap at 80 chars for readability
            for i in range(0, len(seq), 80):
                out.write(seq[i:i+80] + "\n")
            emitted += 1

    log.info("Done. Examined %d chunks, wrote %d FASTA entries → %s", n, emitted, args.fasta)

if __name__ == "__main__":
    main()

