#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Chunk a protein FASTA *exactly like* fasta_to_embed_h5.py does, but as a standalone step.

Behavior replicated from fasta_to_embed_h5.py:
- FASTA parsing: use the *first token* after '>' as seq_id
- Sequence cleaning: uppercase; strip whitespace; remove '*' ; keep printable ASCII
- Chunking:
    - If L <= chunk_len: single chunk
    - Else: sliding windows with stride = chunk_len - overlap
      starts = range(0, max(1, L - chunk_len + 1), stride)
      if last_start + chunk_len < L: append(L - chunk_len)
- Chunk IDs:
    - sanitize_id(seq_id) replaces bad chars with '_' and truncates to 240
    - chunk_id format:
        "{sid}|chunk_{i:0{w}d}_of_{total:0{w}d}|aa_{start:06d}_{end:06d}"
      where w = max(4, int(log10(total) + 1))
    - single-chunk case is always:
        "{sid}|chunk_0001_of_0001|aa_{0:06d}_{L:06d}"

Outputs:
- Chunked FASTA where each record header is the chunk_id
- Optional TSV manifest mapping chunk_id -> original coordinates
"""

import argparse
import math
from typing import Iterator, Tuple, List, Optional


# -------------------------
# FASTA utilities (match original)
# -------------------------

def _clean_seq(s: str) -> str:
    # Uppercase, strip spaces/tabs/CR/LF, remove '*' stops and non-printables
    s = s.upper().replace(" ", "").replace("\t", "").replace("\r", "").replace("\n", "")
    s = s.replace("*", "")
    return "".join(ch for ch in s if (32 <= ord(ch) <= 126) and ch.isprintable())

def read_fasta(fp) -> Iterator[Tuple[str, str]]:
    """
    Parse FASTA; use the *first token* after '>' as the sequence ID.
    Example: '>MGYP001162835132 FL=0' -> seq_id='MGYP001162835132'
    """
    seq_id, chunks = None, []
    for line in fp:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seq_id is not None:
                yield seq_id, _clean_seq("".join(chunks))
            header = line[1:].strip()
            seq_id = header.split()[0] if header else "unnamed"
            chunks = []
        else:
            chunks.append(line)
    if seq_id is not None:
        yield seq_id, _clean_seq("".join(chunks))

def sanitize_id(name: str) -> str:
    bad = '<>:"/\\|?*\t\r\n'
    sanitized = "".join(c if c not in bad else "_" for c in name.strip())
    return sanitized[:240] if sanitized else "unnamed"


# -------------------------
# Chunking (match original)
# -------------------------

def make_chunks_for_sequence(seq_id: str, seq: str, chunk_len: int, overlap: int) -> List[Tuple[str, int, int, int, int, str]]:
    """
    Returns list of tuples:
      (orig_id, orig_len, chunk_index, chunks_total, aa_start, aa_end, chunk_id, subseq)

    Note: aa_start/aa_end are 0-based half-open coordinates [start, end),
          exactly like fasta_to_embed_h5.py stores in attrs and chunk_id.
    """
    L = len(seq)

    if L <= chunk_len:
        cid = f"{sanitize_id(seq_id)}|chunk_0001_of_0001|aa_{0:06d}_{L:06d}"
        return [(seq_id, L, 1, 1, 0, L, cid, seq)]

    if overlap >= chunk_len:
        raise ValueError("--chunk-overlap must be smaller than --chunk-len")

    stride = chunk_len - overlap
    starts = list(range(0, max(1, L - chunk_len + 1), stride))
    if starts[-1] + chunk_len < L:
        starts.append(L - chunk_len)

    total = len(starts)
    w = max(4, int(math.log10(total) + 1))

    out = []
    for i, s in enumerate(starts):
        e = min(L, s + chunk_len)
        subseq = seq[s:e]
        cid = f"{sanitize_id(seq_id)}|chunk_{(i+1):0{w}d}_of_{total:0{w}d}|aa_{s:06d}_{e:06d}"
        out.append((seq_id, L, i+1, total, s, e, cid, subseq))
    return out


# -------------------------
# FASTA writer
# -------------------------

def wrap_seq(seq: str, width: int) -> str:
    if width <= 0:
        return seq + "\n"
    return "\n".join(seq[i:i+width] for i in range(0, len(seq), width)) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Chunk FASTA sequences exactly like fasta_to_embed_h5.py")
    ap.add_argument("--fasta", required=True, help="Input FASTA/FAA")
    ap.add_argument("--out", required=True, help="Output chunked FASTA")
    ap.add_argument("--chunk-len", type=int, default=2000)
    ap.add_argument("--chunk-overlap", type=int, default=128)
    ap.add_argument("--wrap", type=int, default=60, help="FASTA line wrap width (0 = no wrap)")

    ap.add_argument("--manifest", default=None, help="Optional TSV manifest output")
    args = ap.parse_args()

    # Read + chunk
    all_chunks = []
    with open(args.fasta, "r", encoding="utf-8") as f:
        for sid, seq in read_fasta(f):
            if not seq:
                continue
            chunks = make_chunks_for_sequence(sid, seq, args.chunk_len, args.chunk_overlap)
            all_chunks.extend(chunks)

    # Write chunked FASTA
    with open(args.out, "w", encoding="utf-8") as w:
        for (orig_id, orig_len, chunk_index, chunks_total, aa_start, aa_end, chunk_id, subseq) in all_chunks:
            # Header is chunk_id (embedding script uses this as seq_id in H5)
            w.write(f">{chunk_id}\n")
            w.write(wrap_seq(subseq, args.wrap))

    # Optional manifest (TSV)
    if args.manifest:
        with open(args.manifest, "w", encoding="utf-8") as m:
            m.write("\t".join([
                "chunk_id", "orig_id", "chunk_index", "chunks_total",
                "orig_aa_start", "orig_aa_end", "chunk_aa_len", "orig_aa_len"
            ]) + "\n")
            for (orig_id, orig_len, chunk_index, chunks_total, aa_start, aa_end, chunk_id, subseq) in all_chunks:
                m.write("\t".join(map(str, [
                    chunk_id, orig_id, chunk_index, chunks_total,
                    aa_start, aa_end, len(subseq), orig_len
                ])) + "\n")


if __name__ == "__main__":
    main()
