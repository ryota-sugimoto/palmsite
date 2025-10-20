from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, List

_ALLOWED = set("ACDEFGHIKLMNPQRSTVWYX")  # canonical 20 + X
# Map unusual to X (conservative); reject if too many
_MAP_TO_X = set("BJOUZ*.-?")  # B,J,O,U,Z and odd tokens

def _iter_fasta(paths: Iterable[Path]):
    seq_id, buf = None, []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s:
                    continue
                if s.startswith(">"):
                    if seq_id is not None and buf:
                        yield seq_id, "".join(buf)
                    head = s[1:].strip()
                    seq_id = head.split()[0] if head else "unnamed"
                    buf = []
                else:
                    buf.append(s)
        if seq_id is not None and buf:
            yield seq_id, "".join(buf)
            seq_id, buf = None, []
    # trailing (unlikely across files)
    if seq_id is not None and buf:
        yield seq_id, "".join(buf)

def _clean_seq(raw: str) -> Tuple[str, int, int]:
    s = raw.upper().replace(" ", "").replace("\t", "").replace("\r", "").replace("\n", "")
    fixed = 0
    cleaned = []
    for ch in s:
        if ch in _ALLOWED:
            cleaned.append(ch)
        elif ch in _MAP_TO_X:
            cleaned.append("X"); fixed += 1
        else:
            # non-printable or totally unexpected → drop (counts as fixed)
            cleaned.append("X"); fixed += 1
    return "".join(cleaned), len(s), fixed

def sanitize_and_merge_fastas(paths: List[Path], out_path: Path,
                              strict_fraction: float = 0.20,
                              min_len: int = 8,
                              quiet: bool = False) -> Tuple[int, int, int]:
    """
    Write a single sanitized FASTA at `out_path`. Replace unusual AA with 'X'.
    Drop sequences if > strict_fraction of residues were corrected OR length < min_len.
    Returns: (n_kept, n_fixed, n_dropped)
    """
    n_kept = n_fixed = n_dropped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for sid, raw in _iter_fasta(paths):
            seq, Lraw, nfix = _clean_seq(raw)
            if Lraw == 0 or len(seq) < min_len or (Lraw > 0 and (nfix / Lraw) > strict_fraction):
                n_dropped += 1
                if not quiet:
                    print(f"[sanitize] drop {sid}: len={Lraw}, fixes={nfix}", flush=True)
                continue
            n_kept += 1; n_fixed += (1 if nfix > 0 else 0)
            out.write(f">{sid}\n")
            # wrap 60 chars per line
            for i in range(0, len(seq), 60):
                out.write(seq[i:i+60] + "\n")
    return n_kept, n_fixed, n_dropped
