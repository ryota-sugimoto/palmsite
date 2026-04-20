#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create P/U/N labels from:
  - RdRpCATCH TSV (positive anchor)
  - palm_annot TSV (orthogonal local-support / span refinement)
  - Pfam hmmscan --domtblout (non-RdRP conflict filter)

Design goals
------------
This script is a conservative upgrade over label_from_rdrpcatch_and_hmmscan.py.
The main changes are:

1) A sequence is only labeled positive when RdRpCATCH and palm_annot agree.
2) The positive span prefers a local palm_annot span (pp if valid, else ext),
   rather than the broader RdRpCATCH hit span.
3) Non-RdRP Pfam overlap is checked in two places:
   - whole-sequence non-RdRP coverage (clean negative evidence)
   - overlap between the RdRpCATCH span and non-RdRP Pfam aligned regions
     (conflict evidence)
4) A strong RdRpCATCH hit that mostly overlaps a non-RdRP Pfam domain and lacks
   strong palm_annot support is labeled negative.
5) Every FASTA record gets a decision row explaining the evidence and the final
   label.

Default conservative rule
-------------------------
Let:
  RC_POS        := RdRpCATCH best-hit e-value <= --pos-ie
  PALM_POS      := palm_annot rdrp score >= --palm-score-min
  PALM_SPAN     := usable palm_annot span (prefer pp, else ext)
  PALM_CONFLICT := overlap(PALM_SPAN, non-RdRP Pfam) >=
                   --palm-ov-aa and fraction >= --palm-ov-frac
  RC_CONFLICT   := overlap(RdRpCATCH span, non-RdRP Pfam) >=
                   --rc-ov-aa and fraction >= --rc-ov-frac
  FULL_NEG      := whole-sequence non-RdRP Pfam union coverage >= --neg-qcov

Final label:
  P if RC_POS and PALM_POS and PALM_SPAN exists and not PALM_CONFLICT
  N if (RC_POS and not PALM_POS and RC_CONFLICT) or FULL_NEG
  U otherwise

Important nuance
----------------
A broad RdRpCATCH span can overlap non-RdRP domains in genuine viral polyproteins.
Therefore, RC_CONFLICT alone does *not* veto a positive when palm_annot provides
an independent clean local span. RC_CONFLICT is still reported for every record.

Outputs
-------
1) Label TSV (compatibility-oriented; same leading schema as the older script)
2) Decision TSV (one row per FASTA record with evidence + decision columns)
3) Optional JSON summary
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np


# -------------------------
# Logging / I/O
# -------------------------
def setup_logging(level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("rdrpcatch_palmannot_pfam_labels")


def open_text_maybe_gzip(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def open_write_text_maybe_gzip(path: str):
    ensure_parent_dir(path)
    if path.lower().endswith(".gz"):
        return gzip.open(path, "wt", encoding="utf-8", newline="")
    return open(path, "w", encoding="utf-8", newline="")


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)


# -------------------------
# Sequence lengths / order
# -------------------------
def parse_fasta_lengths(path: str) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    seq_id: Optional[str] = None
    cur_len = 0

    with open_text_maybe_gzip(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):   
                if seq_id is not None:
                    out.append((seq_id, cur_len))
                seq_id = line[1:].split(None, 1)[0]
                cur_len = 0
            else:
                cur_len += len(line)

    if seq_id is not None:
        out.append((seq_id, cur_len))

    if not out:
        raise RuntimeError(f"No FASTA records found in {path}")
    return out


# -------------------------
# Generic helpers
# -------------------------
def _coerce_int(s: object) -> Optional[int]:
    if s is None:
        return None
    t = str(s).strip()
    if not t or t == "-":
        return None
    try:
        return int(float(t))
    except Exception:
        return None


def _coerce_float(s: object) -> Optional[float]:
    if s is None:
        return None
    t = str(s).strip()
    if not t or t == "-":
        return None
    try:
        return float(t)
    except Exception:
        return None


def _fmt_float(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return ""
    try:
        if math.isnan(x):
            return ""
    except Exception:
        pass
    return f"{x:.{nd}f}"


def _fmt_float6(x: Optional[float]) -> str:
    return _fmt_float(x, 6)


def _fmt_evalue(ev: Optional[float]) -> str:
    if ev is None:
        return ""
    try:
        if not math.isfinite(ev):
            return ""
    except Exception:
        return ""
    return f"{ev:.3g}"


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _safe_log10_evalue(ev: Optional[float]) -> float:
    if ev is None:
        return 0.0
    if ev <= 0.0:
        ev = 1e-300
    return -math.log10(ev)


def normalize_pfam_acc(acc: str) -> str:
    a = (acc or "").strip()
    if not a or a == "-":
        return ""
    return a.split(".", 1)[0].upper()


def read_pfam_list(path: str) -> set:
    out = set()
    with open_text_maybe_gzip(path) as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            x = normalize_pfam_acc(t)
            if x:
                out.add(x)
    return out


# -------------------------
# Interval helpers
# -------------------------
def to_half_open_1based(qs: int, qe: int) -> Tuple[int, int]:
    s = min(qs, qe) - 1
    e = max(qs, qe)
    return max(0, s), max(0, e)


def merge_intervals(iv: Sequence[Tuple[int, int]], max_gap: int = 0) -> List[Tuple[int, int]]:
    if not iv:
        return []
    arr = sorted(iv, key=lambda x: (x[0], x[1]))
    out: List[Tuple[int, int]] = [arr[0]]
    for s, e in arr[1:]:
        ps, pe = out[-1]
        if s <= pe + max_gap:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def union_len(iv: Sequence[Tuple[int, int]]) -> int:
    return sum(max(0, e - s) for s, e in merge_intervals(iv, max_gap=0))


def intersect_len(iv1: Sequence[Tuple[int, int]], iv2: Sequence[Tuple[int, int]]) -> int:
    a = merge_intervals(iv1, max_gap=0)
    b = merge_intervals(iv2, max_gap=0)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            total += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


# -------------------------
# RdRpCATCH parser
# -------------------------
@dataclass(frozen=True)
class RdRPCatchHit:
    qid: str
    seq_len: Optional[int]
    best_database: str
    best_profile_name: str
    best_profile_length: Optional[int]
    best_evalue: Optional[float]
    best_bitscore: Optional[float]
    rdrp_from: Optional[int]          # 1-based inclusive
    rdrp_to: Optional[int]            # 1-based inclusive
    best_profile_coverage: Optional[float]
    best_sequence_coverage: Optional[float]


_RDRPCATCH_REQUIRED_COLUMNS = {
    "Sequence_name",
    "Sequence_length(AA)",
    "Best_hit_Database",
    "Best_hit_profile_name",
    "Best_hit_profile_length",
    "Best_hit_e-value",
    "Best_hit_bitscore",
    "RdRp_from(AA)",
    "RdRp_to(AA)",
    "Best_hit_profile_coverage",
    "Best_hit_sequence_coverage",
}


def parse_rdrpcatch_tsv(path: str, logger: logging.Logger) -> Dict[str, RdRPCatchHit]:
    out: Dict[str, RdRPCatchHit] = {}
    n_rows = 0
    n_header_repeats = 0
    n_bad = 0
    dup_counter = 0

    with open_text_maybe_gzip(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise RuntimeError(f"Could not read header from {path}")
        missing = [c for c in _RDRPCATCH_REQUIRED_COLUMNS if c not in set(reader.fieldnames)]
        if missing:
            raise RuntimeError(f"RdRPcatch TSV missing required columns: {', '.join(missing)}")

        for row in reader:
            if row.get("Sequence_name", "") == "Sequence_name":
                n_header_repeats += 1
                continue

            qid = (row.get("Sequence_name", "") or "").strip()
            if not qid:
                n_bad += 1
                continue

            rec = RdRPCatchHit(
                qid=qid,
                seq_len=_coerce_int(row.get("Sequence_length(AA)")),
                best_database=(row.get("Best_hit_Database", "") or "").strip(),
                best_profile_name=(row.get("Best_hit_profile_name", "") or "").strip(),
                best_profile_length=_coerce_int(row.get("Best_hit_profile_length")),
                best_evalue=_coerce_float(row.get("Best_hit_e-value")),
                best_bitscore=_coerce_float(row.get("Best_hit_bitscore")),
                rdrp_from=_coerce_int(row.get("RdRp_from(AA)")),
                rdrp_to=_coerce_int(row.get("RdRp_to(AA)")),
                best_profile_coverage=_coerce_float(row.get("Best_hit_profile_coverage")),
                best_sequence_coverage=_coerce_float(row.get("Best_hit_sequence_coverage")),
            )
            n_rows += 1

            prev = out.get(qid)
            if prev is None:
                out[qid] = rec
                continue

            dup_counter += 1

            def key(x: RdRPCatchHit):
                ev = x.best_evalue if x.best_evalue is not None else float("inf")
                bs = x.best_bitscore if x.best_bitscore is not None else float("-inf")
                qc = x.best_sequence_coverage if x.best_sequence_coverage is not None else float("-inf")
                return (ev, -bs, -qc)

            if key(rec) < key(prev):
                out[qid] = rec

    logger.info(
        "Loaded RdRPcatch rows=%d, unique_qids=%d, repeated_headers=%d, duplicate_qids=%d, bad=%d from %s",
        n_rows, len(out), n_header_repeats, dup_counter, n_bad, path,
    )
    return out


# -------------------------
# palm_annot parser
# -------------------------
@dataclass(frozen=True)
class PalmAnnotHit:
    qid: str
    rdrp_score: Optional[float]
    pp_lo: Optional[int]
    pp_hi: Optional[int]
    ext_lo: Optional[int]
    ext_hi: Optional[int]
    pssm_score: Optional[float]
    motif_hmm: str
    motif_hmm_evalue: Optional[float]
    hmm_rdrp_plus: str
    hmm_rdrp_plus_evalue: Optional[float]
    hmm_rdrp_minus: str
    hmm_rdrp_minus_evalue: Optional[float]
    seqA: str
    seqB: str
    seqC: str
    posA: Optional[int]
    posB: Optional[int]
    posC: Optional[int]
    raw: Dict[str, str]


def parse_palm_annot_tsv(path: str, logger: logging.Logger) -> Dict[str, PalmAnnotHit]:
    out: Dict[str, PalmAnnotHit] = {}
    n = 0
    dup = 0
    bad = 0

    with open_text_maybe_gzip(path) as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if not parts:
                continue
            qid = parts[0].strip()
            if not qid:
                bad += 1
                continue
            data: Dict[str, str] = {}
            for p in parts[1:]:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                data[k] = v

            rec = PalmAnnotHit(
                qid=qid,
                rdrp_score=_coerce_float(data.get("rdrp")),
                pp_lo=_coerce_int(data.get("pp_lo")),
                pp_hi=_coerce_int(data.get("pp_hi")),
                ext_lo=_coerce_int(data.get("ext_lo")),
                ext_hi=_coerce_int(data.get("ext_hi")),
                pssm_score=_coerce_float(data.get("pssm_score")),
                motif_hmm=(data.get("motif_hmm", "") or "").strip(),
                motif_hmm_evalue=_coerce_float(data.get("motif_hmm_evalue")),
                hmm_rdrp_plus=(data.get("hmm_rdrp_plus", "") or "").strip(),
                hmm_rdrp_plus_evalue=_coerce_float(data.get("hmm_rdrp_plus_evalue")),
                hmm_rdrp_minus=(data.get("hmm_rdrp_minus", "") or "").strip(),
                hmm_rdrp_minus_evalue=_coerce_float(data.get("hmm_rdrp_minus_evalue")),
                seqA=(data.get("seqA", "") or "").strip(),
                seqB=(data.get("seqB", "") or "").strip(),
                seqC=(data.get("seqC", "") or "").strip(),
                posA=_coerce_int(data.get("posA")),
                posB=_coerce_int(data.get("posB")),
                posC=_coerce_int(data.get("posC")),
                raw=data,
            )
            n += 1

            prev = out.get(qid)
            if prev is None:
                out[qid] = rec
                continue

            dup += 1

            def dup_key(x: PalmAnnotHit):
                score = x.rdrp_score if x.rdrp_score is not None else float("-inf")
                valid_pp = 1 if (x.pp_lo is not None and x.pp_hi is not None and x.pp_hi >= x.pp_lo) else 0
                valid_ext = 1 if (x.ext_lo is not None and x.ext_hi is not None and x.ext_hi >= x.ext_lo) else 0
                motif_count = sum(1 for z in [x.seqA, x.seqB, x.seqC] if z)
                return (score, valid_pp, valid_ext, motif_count)

            if dup_key(rec) > dup_key(prev):
                out[qid] = rec

    logger.info("Loaded palm_annot rows=%d, unique_qids=%d, duplicate_qids=%d, bad=%d from %s", n, len(out), dup, bad, path)
    return out


# -------------------------
# hmmscan domtbl parser
# -------------------------
@dataclass(frozen=True)
class HmmerDomHit:
    qid: str
    qlen: Optional[int]
    target_name: str
    target_acc: str
    target_acc_base: str
    i_evalue: float
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int
    description: str


def parse_hmmscan_domtbl(path: str, logger: logging.Logger) -> Dict[str, List[HmmerDomHit]]:
    out: Dict[str, List[HmmerDomHit]] = defaultdict(list)
    n = 0
    n_bad = 0

    with open_text_maybe_gzip(path) as f:
        for ln in f:
            if not ln.strip() or ln.lstrip().startswith("#"):
                continue
            line = ln.strip("\n").replace("\\", " ")
            parts = line.split()
            if len(parts) < 22:
                n_bad += 1
                continue
            try:
                target_name = parts[0]
                target_acc = parts[1]
                qid = parts[3]
                qlen = int(parts[5]) if parts[5] not in ("-", "") else None
                i_e = float(parts[12])
                ali_from = int(parts[17])
                ali_to = int(parts[18])
                env_from = int(parts[19])
                env_to = int(parts[20])
                desc = " ".join(parts[22:]) if len(parts) > 22 else ""
            except Exception:
                n_bad += 1
                continue
            out[qid].append(
                HmmerDomHit(
                    qid=qid,
                    qlen=qlen,
                    target_name=target_name,
                    target_acc=target_acc,
                    target_acc_base=normalize_pfam_acc(target_acc),
                    i_evalue=i_e,
                    ali_from=ali_from,
                    ali_to=ali_to,
                    env_from=env_from,
                    env_to=env_to,
                    description=desc,
                )
            )
            n += 1

    logger.info("Loaded hmmscan domtbl rows=%d (bad=%d), unique_qids=%d from %s", n, n_bad, len(out), path)
    return out


def span_from_dom(hit: HmmerDomHit, span_source: str) -> Tuple[int, int]:
    if span_source == "ali":
        lo = min(hit.ali_from, hit.ali_to)
        hi = max(hit.ali_from, hit.ali_to)
    else:
        lo = min(hit.env_from, hit.env_to)
        hi = max(hit.env_from, hit.env_to)
    return max(0, lo - 1), max(0, hi)


# -------------------------
# Evidence helpers
# -------------------------
def choose_palm_span(
    hit: Optional[PalmAnnotHit],
    pp_min_len: int,
    pp_max_len: int,
    ext_min_len: int,
    ext_max_len: int,
) -> Tuple[str, Optional[int], Optional[int], str]:
    """
    Return (source, lo1, hi1, reason).
    source is one of: pp, ext, "".
    Coordinates are 1-based inclusive.
    """
    if hit is None:
        return "", None, None, "no_palm_annot"

    if hit.pp_lo is not None and hit.pp_hi is not None and hit.pp_hi >= hit.pp_lo:
        pp_len = hit.pp_hi - hit.pp_lo + 1
        if pp_min_len <= pp_len <= pp_max_len:
            return "pp", hit.pp_lo, hit.pp_hi, "valid_pp"
        return "", None, None, f"pp_len_out_of_range:{pp_len}"

    if hit.pp_lo is not None and hit.pp_hi is not None and hit.pp_hi < hit.pp_lo:
        pp_reason = f"pp_wrapped:{hit.pp_lo}>{hit.pp_hi}"
    else:
        pp_reason = "pp_missing"

    if hit.ext_lo is not None and hit.ext_hi is not None and hit.ext_hi >= hit.ext_lo:
        ext_len = hit.ext_hi - hit.ext_lo + 1
        if ext_min_len <= ext_len <= ext_max_len:
            return "ext", hit.ext_lo, hit.ext_hi, f"fallback_ext_after_{pp_reason}"
        return "", None, None, f"ext_len_out_of_range:{ext_len}"

    return "", None, None, pp_reason


def pfam_nonrdrp_hit_stats(
    hits: Sequence[HmmerDomHit],
    palm_pfams: set,
    neg_ie: float,
    span_source: str,
) -> Tuple[List[Tuple[int, int]], Dict[str, List[Tuple[int, int]]], List[HmmerDomHit]]:
    all_intervals: List[Tuple[int, int]] = []
    model_to_intervals: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    kept_hits: List[HmmerDomHit] = []
    for h in hits:
        if h.target_acc_base in palm_pfams:
            continue
        if h.i_evalue > neg_ie:
            continue
        iv = span_from_dom(h, span_source)
        if iv[1] <= iv[0]:
            continue
        all_intervals.append(iv)
        model = f"{h.target_acc_base or h.target_name}|{h.target_name}"
        model_to_intervals[model].append(iv)
        kept_hits.append(h)
    return all_intervals, model_to_intervals, kept_hits


def summarize_model_coverage(model_to_intervals: Dict[str, List[Tuple[int, int]]], topn: int = 8) -> str:
    if not model_to_intervals:
        return ""
    pairs = []
    for model, iv in model_to_intervals.items():
        cov = union_len(iv)
        pairs.append((model, cov))
    pairs.sort(key=lambda x: (-x[1], x[0]))
    shown = pairs[:topn]
    return ",".join(f"{m}:{cov}" for m, cov in shown)


def summarize_model_overlap(
    span_iv: Sequence[Tuple[int, int]],
    model_to_intervals: Dict[str, List[Tuple[int, int]]],
    topn: int = 8,
) -> str:
    if not span_iv or not model_to_intervals:
        return ""
    pairs = []
    for model, iv in model_to_intervals.items():
        ov = intersect_len(span_iv, iv)
        if ov > 0:
            pairs.append((model, ov))
    pairs.sort(key=lambda x: (-x[1], x[0]))
    shown = pairs[:topn]
    return ",".join(f"{m}:{ov}" for m, ov in shown)


def conf_from_rdrpcatch(
    best_evalue: Optional[float],
    seq_cov: float,
    profile_cov: Optional[float],
    truncated: bool,
    pos_e_max: float,
    pos_e_sat: float,
) -> Tuple[float, float, float, float]:
    if pos_e_sat > pos_e_max:
        pos_e_sat = pos_e_max
    s_min = _safe_log10_evalue(pos_e_max)
    s_sat = _safe_log10_evalue(pos_e_sat)
    if s_sat <= s_min:
        s_sat = s_min + 1.0
    s = _safe_log10_evalue(best_evalue)
    conf_e = clip01((s - s_min) / (s_sat - s_min))
    conf_cov = clip01(seq_cov / 0.60)
    conf_hsp = clip01(profile_cov if profile_cov is not None else 0.0)
    conf = 0.6 * conf_e + 0.3 * conf_cov + 0.1 * conf_hsp
    if truncated:
        conf *= 0.5
    return conf, conf_e, conf_cov, conf_hsp


# -------------------------
# Decision logic
# -------------------------
def decide_record(
    qid: str,
    L: int,
    rc: Optional[RdRPCatchHit],
    palm: Optional[PalmAnnotHit],
    pfam_hits: Sequence[HmmerDomHit],
    palm_pfams: set,
    pos_ie: float,
    palm_score_min: float,
    neg_ie: float,
    neg_qcov: float,
    pfam_span_source: str,
    rc_ov_aa: int,
    rc_ov_frac: float,
    palm_ov_aa: int,
    palm_ov_frac: float,
    pp_min_len: int,
    pp_max_len: int,
    ext_min_len: int,
    ext_max_len: int,
    sigma_k: float,
    conf_e_max: float,
    conf_e_sat: float,
) -> Tuple[Dict[str, str], Dict[str, str], str]:
    all_intervals, model_to_intervals, kept_hits = pfam_nonrdrp_hit_stats(
        pfam_hits, palm_pfams, neg_ie, pfam_span_source
    )

    full_cov = (union_len(all_intervals) / float(L)) if (L > 0 and all_intervals) else 0.0
    full_models = summarize_model_coverage(model_to_intervals, topn=8)

    rc_pos = (
        rc is not None
        and rc.best_evalue is not None
        and rc.best_evalue <= pos_ie
        and rc.rdrp_from is not None
        and rc.rdrp_to is not None
    )
    rc_span_iv: List[Tuple[int, int]] = []
    rc_span_len = 0
    if rc_pos:
        rc_span_iv = [to_half_open_1based(int(rc.rdrp_from), int(rc.rdrp_to))]
        rc_span_len = union_len(rc_span_iv)
    rc_overlap_aa = intersect_len(rc_span_iv, all_intervals) if rc_span_iv else 0
    rc_overlap_frac = (rc_overlap_aa / float(rc_span_len)) if rc_span_len > 0 else 0.0
    rc_overlap_models = summarize_model_overlap(rc_span_iv, model_to_intervals, topn=8) if rc_span_iv else ""

    palm_score = palm.rdrp_score if palm is not None and palm.rdrp_score is not None else None
    palm_pos = palm_score is not None and palm_score >= palm_score_min
    palm_span_source, palm_span_lo, palm_span_hi, palm_span_reason = choose_palm_span(
        palm, pp_min_len, pp_max_len, ext_min_len, ext_max_len
    )
    palm_has_span = bool(palm_span_source and palm_span_lo is not None and palm_span_hi is not None)

    palm_span_iv: List[Tuple[int, int]] = []
    palm_span_len = 0
    if palm_has_span:
        palm_span_iv = [to_half_open_1based(int(palm_span_lo), int(palm_span_hi))]
        palm_span_len = union_len(palm_span_iv)
    palm_overlap_aa2 = intersect_len(palm_span_iv, all_intervals) if palm_span_iv else 0
    palm_overlap_frac2 = (palm_overlap_aa2 / float(palm_span_len)) if palm_span_len > 0 else 0.0
    palm_overlap_models = summarize_model_overlap(palm_span_iv, model_to_intervals, topn=8) if palm_span_iv else ""

    full_neg = full_cov >= neg_qcov
    rc_conflict = rc_overlap_aa >= rc_ov_aa and rc_overlap_frac >= rc_ov_frac
    palm_conflict = palm_overlap_aa2 >= palm_ov_aa and palm_overlap_frac2 >= palm_ov_frac

    decision_code = ""
    label = "U"
    if rc_pos and palm_pos and palm_has_span and not palm_conflict:
        label = "P"
        if rc_conflict:
            decision_code = "P_RESCUED_BY_CLEAN_PALM_SPAN"
        elif palm_span_source == "pp":
            decision_code = "P_CONSENSUS_CLEAN_PP"
        else:
            decision_code = "P_CONSENSUS_CLEAN_EXT"
    elif rc_pos and (not palm_pos) and rc_conflict:
        label = "N"
        decision_code = "N_RDRPCATCH_CONFLICTS_WITH_NONRDRP_PFAM"
    elif full_neg and (not palm_pos):
        label = "N"
        decision_code = "N_NONRDRP_PFAM_DOMINANT"
    else:
        label = "U"
        if rc_pos and palm_pos and palm_has_span and palm_conflict:
            decision_code = "U_PALM_SPAN_CONFLICTS_WITH_NONRDRP_PFAM"
        elif rc_pos and palm_pos and (not palm_has_span):
            decision_code = "U_NO_USABLE_PALM_SPAN"
        elif rc_pos and (not palm_pos):
            decision_code = "U_RDRPCATCH_ONLY_OR_LOW_PALM_SCORE"
        elif (not rc_pos) and palm_pos:
            decision_code = "U_PALMANNOT_ONLY"
        elif palm is not None and palm_score is not None and palm_score > 0:
            decision_code = "U_PALMANNOT_LOW_SCORE_ONLY"
        else:
            decision_code = "U_NO_HIGH_CONFIDENCE_SIGNAL"

    # Compatibility-oriented label row (leading schema same as the older script)
    use_span = 0
    S = float("nan")
    E = float("nan")
    span_start_aa = ""
    span_end_aa = ""
    truncated = 0
    conf = conf_e = conf_cov = conf_hsp = 0.0
    w_min = max(70.0 / max(1.0, float(L)), 0.02)
    sigma_star = float("nan")

    if label == "P" and palm_has_span:
        s0, e0 = palm_span_iv[0]
        s0 = max(0, min(s0, L))
        e0 = max(0, min(e0, L))
        if e0 > s0:
            use_span = 1
            span_start_aa = str(int(s0))
            span_end_aa = str(int(e0))
            S = float(s0) / float(L) if L > 0 else float("nan")
            E = float(e0) / float(L) if L > 0 else float("nan")
            truncated = 1 if (s0 == 0 or e0 == L) else 0
            seq_cov = rc.best_sequence_coverage if (rc is not None and rc.best_sequence_coverage is not None) else 0.0
            prof_cov = rc.best_profile_coverage if rc is not None else None
            best_ev = rc.best_evalue if rc is not None else None
            conf, conf_e, conf_cov, conf_hsp = conf_from_rdrpcatch(
                best_evalue=best_ev,
                seq_cov=float(seq_cov),
                profile_cov=prof_cov,
                truncated=bool(truncated),
                pos_e_max=conf_e_max,
                pos_e_sat=conf_e_sat,
            )
            w = E - S
            sigma_star = max(w / (2.0 * sigma_k), w_min / 2.0)

    rps_pos_models = ""
    rps_pos_e_min = ""
    rps_pos_qcov_max = float("nan")
    rps_pos_dcov_max = float("nan")
    if rc is not None:
        rps_pos_models = ":".join(x for x in [rc.best_database, rc.best_profile_name] if x)
        rps_pos_e_min = _fmt_evalue(rc.best_evalue)
        rps_pos_qcov_max = rc.best_sequence_coverage if rc.best_sequence_coverage is not None else float("nan")
        rps_pos_dcov_max = rc.best_profile_coverage if rc.best_profile_coverage is not None else float("nan")

    label_rec = {
        "chunk_id": qid,
        "label": label,
        "use_span": str(use_span),
        "L": str(L),
        "S": _fmt_float6(S),
        "E": _fmt_float6(E),
        "span_start_aa": span_start_aa,
        "span_end_aa": span_end_aa,
        "truncated": str(int(truncated)),
        "conf": _fmt_float(conf, 4),
        "conf_e": _fmt_float(conf_e, 4),
        "conf_cov": _fmt_float(conf_cov, 4),
        "conf_hsp": _fmt_float(conf_hsp, 4),
        "w_min": _fmt_float6(w_min),
        "sigma_star": _fmt_float6(sigma_star),
        "k": _fmt_float(sigma_k, 3),
        "rps_pos_e_min": rps_pos_e_min,
        "rps_pos_models": rps_pos_models,
        "rps_pos_qcov_max": _fmt_float(rps_pos_qcov_max, 4),
        "rps_pos_dcov_max": _fmt_float(rps_pos_dcov_max, 4),
        "rps_neg_models": full_models,
        "rps_neg_qcov_max": _fmt_float(full_cov, 4),
        "decision_code": decision_code,
        "decision_span_source": palm_span_source,
        "palm_rdrp_score": _fmt_float(palm_score, 1),
        "pfam_rdrpcatch_overlap_aa": str(int(rc_overlap_aa)) if rc_span_iv else "0",
        "pfam_rdrpcatch_overlap_frac": _fmt_float(rc_overlap_frac, 4) if rc_span_iv else "",
        "pfam_palm_overlap_aa": str(int(palm_overlap_aa2)) if palm_span_iv else "0",
        "pfam_palm_overlap_frac": _fmt_float(palm_overlap_frac2, 4) if palm_span_iv else "",
    }

    decision_rec = {
        "chunk_id": qid,
        "label": label,
        "decision_code": decision_code,
        "L": str(L),
        "rdrpcatch_positive": "1" if rc_pos else "0",
        "palm_positive": "1" if palm_pos else "0",
        "has_usable_palm_span": "1" if palm_has_span else "0",
        "full_nonrdrp_negative": "1" if full_neg else "0",
        "rdrpcatch_overlap_conflict": "1" if rc_conflict else "0",
        "palm_overlap_conflict": "1" if palm_conflict else "0",
        "rdrpcatch_best_database": rc.best_database if rc is not None else "",
        "rdrpcatch_best_profile_name": rc.best_profile_name if rc is not None else "",
        "rdrpcatch_best_profile_length": str(rc.best_profile_length) if rc is not None and rc.best_profile_length is not None else "",
        "rdrpcatch_best_evalue": _fmt_evalue(rc.best_evalue) if rc is not None else "",
        "rdrpcatch_best_bitscore": _fmt_float(rc.best_bitscore, 3) if rc is not None and rc.best_bitscore is not None else "",
        "rdrpcatch_best_seq_cov": _fmt_float(rc.best_sequence_coverage, 4) if rc is not None and rc.best_sequence_coverage is not None else "",
        "rdrpcatch_best_profile_cov": _fmt_float(rc.best_profile_coverage, 4) if rc is not None and rc.best_profile_coverage is not None else "",
        "rdrpcatch_span_from_1based": str(rc.rdrp_from) if rc is not None and rc.rdrp_from is not None else "",
        "rdrpcatch_span_to_1based": str(rc.rdrp_to) if rc is not None and rc.rdrp_to is not None else "",
        "rdrpcatch_span_len_aa": str(int(rc_span_len)) if rc_span_iv else "",
        "palm_rdrp_score": _fmt_float(palm_score, 1),
        "palm_span_source": palm_span_source,
        "palm_span_reason": palm_span_reason,
        "palm_span_from_1based": str(palm_span_lo) if palm_span_lo is not None else "",
        "palm_span_to_1based": str(palm_span_hi) if palm_span_hi is not None else "",
        "palm_span_len_aa": str(int(palm_span_len)) if palm_span_iv else "",
        "palm_pp_lo": str(palm.pp_lo) if palm is not None and palm.pp_lo is not None else "",
        "palm_pp_hi": str(palm.pp_hi) if palm is not None and palm.pp_hi is not None else "",
        "palm_ext_lo": str(palm.ext_lo) if palm is not None and palm.ext_lo is not None else "",
        "palm_ext_hi": str(palm.ext_hi) if palm is not None and palm.ext_hi is not None else "",
        "palm_has_seqA": "1" if palm is not None and bool(palm.seqA) else "0",
        "palm_has_seqB": "1" if palm is not None and bool(palm.seqB) else "0",
        "palm_has_seqC": "1" if palm is not None and bool(palm.seqC) else "0",
        "palm_posA": str(palm.posA) if palm is not None and palm.posA is not None else "",
        "palm_posB": str(palm.posB) if palm is not None and palm.posB is not None else "",
        "palm_posC": str(palm.posC) if palm is not None and palm.posC is not None else "",
        "pfam_neg_domain_count": str(len(kept_hits)),
        "pfam_neg_union_cov": _fmt_float(full_cov, 4) if kept_hits else "",
        "pfam_neg_models": full_models,
        "pfam_rdrpcatch_overlap_aa": str(int(rc_overlap_aa)) if rc_span_iv else "0",
        "pfam_rdrpcatch_overlap_frac": _fmt_float(rc_overlap_frac, 4) if rc_span_iv else "",
        "pfam_rdrpcatch_overlap_models": rc_overlap_models,
        "pfam_palm_overlap_aa": str(int(palm_overlap_aa2)) if palm_span_iv else "0",
        "pfam_palm_overlap_frac": _fmt_float(palm_overlap_frac2, 4) if palm_span_iv else "",
        "pfam_palm_overlap_models": palm_overlap_models,
    }

    return label_rec, decision_rec, decision_code


# -------------------------
# Compact H5 writer
# -------------------------
LABEL_TO_U8 = {"N": 0, "U": 1, "P": 2}


def default_out_h5(tsv_path: str) -> str:
    low = tsv_path.lower()
    for suf in (".tsv.gz", ".txt.gz", ".tsv", ".txt"):
        if low.endswith(suf):
            return tsv_path[:-len(suf)] + ".h5"
    return tsv_path + ".h5"


def init_h5_columns() -> Dict[str, List[object]]:
    return {
        "chunk_id": [],
        "label": [],
        "use_span": [],
        "L": [],
        "S": [],
        "E": [],
        "span_start_aa": [],
        "span_end_aa": [],
        "truncated": [],
        "conf": [],
        "conf_e": [],
        "conf_cov": [],
        "conf_hsp": [],
        "w_min": [],
        "sigma_star": [],
        "k": [],
        "rps_pos_e_min": [],
        "rps_pos_models": [],
        "rps_pos_qcov_max": [],
        "rps_pos_dcov_max": [],
        "rps_neg_models": [],
        "rps_neg_qcov_max": [],
        "decision_code": [],
        "decision_span_source": [],
        "palm_rdrp_score": [],
        "pfam_rdrpcatch_overlap_aa": [],
        "pfam_rdrpcatch_overlap_frac": [],
        "pfam_palm_overlap_aa": [],
        "pfam_palm_overlap_frac": [],
    }


def append_h5_record(cols: Dict[str, List[object]], label_rec: Dict[str, str]) -> None:
    cols["chunk_id"].append(label_rec.get("chunk_id", ""))
    cols["label"].append(label_rec.get("label", "U"))
    cols["use_span"].append(np.uint8(int(label_rec.get("use_span", "0") or 0)))
    cols["L"].append(np.int32(int(label_rec.get("L", "0") or 0)))
    cols["S"].append(_coerce_float(label_rec.get("S")))
    cols["E"].append(_coerce_float(label_rec.get("E")))
    cols["span_start_aa"].append(label_rec.get("span_start_aa", ""))
    cols["span_end_aa"].append(label_rec.get("span_end_aa", ""))
    cols["truncated"].append(np.uint8(int(label_rec.get("truncated", "0") or 0)))
    cols["conf"].append(_coerce_float(label_rec.get("conf")))
    cols["conf_e"].append(_coerce_float(label_rec.get("conf_e")))
    cols["conf_cov"].append(_coerce_float(label_rec.get("conf_cov")))
    cols["conf_hsp"].append(_coerce_float(label_rec.get("conf_hsp")))
    cols["w_min"].append(_coerce_float(label_rec.get("w_min")))
    cols["sigma_star"].append(_coerce_float(label_rec.get("sigma_star")))
    cols["k"].append(_coerce_float(label_rec.get("k")))
    cols["rps_pos_e_min"].append(label_rec.get("rps_pos_e_min", ""))
    cols["rps_pos_models"].append(label_rec.get("rps_pos_models", ""))
    cols["rps_pos_qcov_max"].append(_coerce_float(label_rec.get("rps_pos_qcov_max")))
    cols["rps_pos_dcov_max"].append(_coerce_float(label_rec.get("rps_pos_dcov_max")))
    cols["rps_neg_models"].append(label_rec.get("rps_neg_models", ""))
    cols["rps_neg_qcov_max"].append(_coerce_float(label_rec.get("rps_neg_qcov_max")))
    cols["decision_code"].append(label_rec.get("decision_code", ""))
    cols["decision_span_source"].append(label_rec.get("decision_span_source", ""))
    cols["palm_rdrp_score"].append(_coerce_float(label_rec.get("palm_rdrp_score")))
    cols["pfam_rdrpcatch_overlap_aa"].append(_coerce_float(label_rec.get("pfam_rdrpcatch_overlap_aa")))
    cols["pfam_rdrpcatch_overlap_frac"].append(_coerce_float(label_rec.get("pfam_rdrpcatch_overlap_frac")))
    cols["pfam_palm_overlap_aa"].append(_coerce_float(label_rec.get("pfam_palm_overlap_aa")))
    cols["pfam_palm_overlap_frac"].append(_coerce_float(label_rec.get("pfam_palm_overlap_frac")))


def write_compact_labels_h5(
    out_h5: str,
    cols: Dict[str, List[object]],
    tsv_path: str,
    schema_version: str,
    thresholds_meta: dict,
    overwrite: bool,
    logger: logging.Logger,
) -> None:
    if os.path.exists(out_h5):
        if overwrite:
            logger.warning("Overwriting %s", out_h5)
            os.remove(out_h5)
        else:
            raise FileExistsError(f"Refusing to overwrite existing H5: {out_h5} (use --overwrite-h5)")

    ensure_parent_dir(out_h5)
    vstr = h5py.string_dtype(encoding="utf-8")
    N = len(cols["chunk_id"])

    def arr_str(name: str) -> np.ndarray:
        return np.array(cols[name], dtype=vstr)

    def arr_u8(name: str) -> np.ndarray:
        if name == "label":
            return np.fromiter((LABEL_TO_U8.get(str(x), 255) for x in cols[name]), dtype=np.uint8, count=N)
        return np.fromiter((int(x) for x in cols[name]), dtype=np.uint8, count=N)

    def arr_i32(name: str) -> np.ndarray:
        return np.fromiter((int(x) for x in cols[name]), dtype=np.int32, count=N)

    def arr_f32(name: str) -> np.ndarray:
        return np.array([float("nan") if x is None else float(x) for x in cols[name]], dtype=np.float32)

    with h5py.File(out_h5, "w", libver="latest") as f:
        g = f.create_group("labels")

        for name in ("chunk_id", "rps_pos_e_min", "rps_pos_models", "rps_neg_models", "span_start_aa", "span_end_aa", "decision_code", "decision_span_source"):
            g.create_dataset(name, data=arr_str(name))

        def dset_num(name: str, arr: np.ndarray) -> None:
            g.create_dataset(name, data=arr, compression="lzf", shuffle=True, chunks=True)

        dset_num("label", arr_u8("label"))
        dset_num("use_span", arr_u8("use_span"))
        dset_num("truncated", arr_u8("truncated"))
        dset_num("L", arr_i32("L"))

        for name in (
            "S", "E", "conf", "conf_e", "conf_cov", "conf_hsp", "w_min", "sigma_star", "k",
            "rps_pos_qcov_max", "rps_pos_dcov_max", "rps_neg_qcov_max",
            "palm_rdrp_score", "pfam_rdrpcatch_overlap_aa", "pfam_rdrpcatch_overlap_frac",
            "pfam_palm_overlap_aa", "pfam_palm_overlap_frac",
        ):
            dset_num(name, arr_f32(name))

        g.attrs["label_map"] = json.dumps({"N": 0, "U": 1, "P": 2})
        f.attrs["schema_version"] = schema_version
        f.attrs["thresholds"] = json.dumps(thresholds_meta)
        f.attrs["source"] = "rdrpcatch+palm_annot+pfam"
        with open(tsv_path, "rb") as fh:
            f.attrs["tsv_sha256"] = hashlib.sha256(fh.read()).hexdigest()
        f.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    logger.info("Wrote %s with %d rows; size ~ %.2f MB", out_h5, N, os.path.getsize(out_h5) / 1e6)


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Label sequences using RdRpCATCH + palm_annot + Pfam conflict filtering")
    ap.add_argument("--fasta", required=True, help="input FASTA (.gz ok)")
    ap.add_argument("--rdrpcatch-tsv", required=True, help="RdRpCATCH results TSV(.gz)")
    ap.add_argument("--palm-annot-tsv", required=True, help="palm_annot TSV(.gz)")
    ap.add_argument("--hmmscan-domtbl", required=True, help="Pfam hmmscan --domtblout(.gz)")
    ap.add_argument("--palm-pfams", required=True, help="Pfam accessions considered RdRP Pfams")
    ap.add_argument("--out", required=True, help="output label TSV(.gz)")
    ap.add_argument("--decision-out", required=True, help="output decision TSV(.gz)")
    ap.add_argument("--summary-out", default="", help="optional summary JSON")
    ap.add_argument("--out-h5", default=None, help="compact labels H5 path (default: derive from --out)")
    ap.add_argument("--no-h5", action="store_true", help="do not write compact labels H5")
    ap.add_argument("--overwrite-h5", action="store_true", help="overwrite existing --out-h5 if it exists")
    ap.add_argument("--schema-version", default="1.1")
    ap.add_argument("--log-level", default="INFO")

    ap.add_argument("--pos-ie", type=float, default=1e-5, help="max RdRpCATCH best-hit e-value for positive anchor")
    ap.add_argument("--palm-score-min", type=float, default=75.0, help="min palm_annot rdrp score for strong support")
    ap.add_argument("--neg-ie", type=float, default=1e-5, help="max Pfam i-Evalue for non-RdRP conflict domains")
    ap.add_argument("--neg-qcov", type=float, default=0.80, help="whole-sequence non-RdRP Pfam union coverage for FULL_NEG")
    ap.add_argument("--pfam-span-source", choices=["ali", "env"], default="ali", help="Pfam span source for coverage/overlap")

    ap.add_argument("--rc-ov-aa", type=int, default=50, help="min overlap aa for RdRpCATCH-vs-Pfam conflict")
    ap.add_argument("--rc-ov-frac", type=float, default=0.50, help="min overlap fraction of RdRpCATCH span for conflict")
    ap.add_argument("--palm-ov-aa", type=int, default=20, help="min overlap aa for palm span conflict")
    ap.add_argument("--palm-ov-frac", type=float, default=0.25, help="min overlap fraction of palm span for conflict")

    ap.add_argument("--pp-min-len", type=int, default=30, help="minimum usable palm_annot pp length")
    ap.add_argument("--pp-max-len", type=int, default=400, help="maximum usable palm_annot pp length")
    ap.add_argument("--ext-min-len", type=int, default=30, help="minimum usable palm_annot ext length")
    ap.add_argument("--ext-max-len", type=int, default=600, help="maximum usable palm_annot ext length")

    ap.add_argument("--conf-e-max", type=float, default=1e-3, help="e-value where confidence rises above zero")
    ap.add_argument("--conf-e-sat", type=float, default=1e-30, help="e-value where confidence saturates")
    ap.add_argument("--sigma-k", type=float, default=2.0)
    ap.add_argument("--progress-every", type=int, default=50000)

    args = ap.parse_args()
    log = setup_logging(args.log_level)

    out_h5 = None if args.no_h5 else (args.out_h5 or default_out_h5(args.out))

    ordered_lengths = parse_fasta_lengths(args.fasta)
    qlens = dict(ordered_lengths)
    log.info("Loaded %d FASTA records from %s", len(ordered_lengths), args.fasta)

    rdrpcatch_hits = parse_rdrpcatch_tsv(args.rdrpcatch_tsv, logger=log)
    palm_hits = parse_palm_annot_tsv(args.palm_annot_tsv, logger=log)
    domtbl = parse_hmmscan_domtbl(args.hmmscan_domtbl, logger=log)
    palm_pfams = read_pfam_list(args.palm_pfams)
    if not palm_pfams:
        raise RuntimeError(f"Empty RdRP Pfam list: {args.palm_pfams}")

    extra_rdrpcatch = set(rdrpcatch_hits) - set(qlens)
    extra_palm = set(palm_hits) - set(qlens)
    extra_dom = set(domtbl) - set(qlens)
    if extra_rdrpcatch:
        log.warning("%d RdRpCATCH IDs not found in FASTA will be ignored", len(extra_rdrpcatch))
    if extra_palm:
        log.warning("%d palm_annot IDs not found in FASTA will be ignored", len(extra_palm))
    if extra_dom:
        log.warning("%d hmmscan IDs not found in FASTA will be ignored", len(extra_dom))

    label_header = [
        "chunk_id", "label", "use_span", "L", "S", "E",
        "span_start_aa", "span_end_aa", "truncated",
        "conf", "conf_e", "conf_cov", "conf_hsp",
        "w_min", "sigma_star", "k",
        "rps_pos_e_min", "rps_pos_models", "rps_pos_qcov_max", "rps_pos_dcov_max",
        "rps_neg_models", "rps_neg_qcov_max",
        "decision_code", "decision_span_source", "palm_rdrp_score",
        "pfam_rdrpcatch_overlap_aa", "pfam_rdrpcatch_overlap_frac",
        "pfam_palm_overlap_aa", "pfam_palm_overlap_frac",
    ]
    decision_header = [
        "chunk_id", "label", "decision_code", "L",
        "rdrpcatch_positive", "palm_positive", "has_usable_palm_span",
        "full_nonrdrp_negative", "rdrpcatch_overlap_conflict", "palm_overlap_conflict",
        "rdrpcatch_best_database", "rdrpcatch_best_profile_name", "rdrpcatch_best_profile_length",
        "rdrpcatch_best_evalue", "rdrpcatch_best_bitscore",
        "rdrpcatch_best_seq_cov", "rdrpcatch_best_profile_cov",
        "rdrpcatch_span_from_1based", "rdrpcatch_span_to_1based", "rdrpcatch_span_len_aa",
        "palm_rdrp_score", "palm_span_source", "palm_span_reason",
        "palm_span_from_1based", "palm_span_to_1based", "palm_span_len_aa",
        "palm_pp_lo", "palm_pp_hi", "palm_ext_lo", "palm_ext_hi",
        "palm_has_seqA", "palm_has_seqB", "palm_has_seqC",
        "palm_posA", "palm_posB", "palm_posC",
        "pfam_neg_domain_count", "pfam_neg_union_cov", "pfam_neg_models",
        "pfam_rdrpcatch_overlap_aa", "pfam_rdrpcatch_overlap_frac", "pfam_rdrpcatch_overlap_models",
        "pfam_palm_overlap_aa", "pfam_palm_overlap_frac", "pfam_palm_overlap_models",
    ]

    label_counts = Counter()
    decision_counts = Counter()
    h5_cols = init_h5_columns() if out_h5 is not None else None

    with open_write_text_maybe_gzip(args.out) as out_lab, open_write_text_maybe_gzip(args.decision_out) as out_dec:
        out_lab.write("\t".join(label_header) + "\n")
        out_dec.write("\t".join(decision_header) + "\n")

        for i, (qid, L) in enumerate(ordered_lengths, start=1):
            label_rec, decision_rec, decision_code = decide_record(
                qid=qid,
                L=L,
                rc=rdrpcatch_hits.get(qid),
                palm=palm_hits.get(qid),
                pfam_hits=domtbl.get(qid, []),
                palm_pfams=palm_pfams,
                pos_ie=args.pos_ie,
                palm_score_min=args.palm_score_min,
                neg_ie=args.neg_ie,
                neg_qcov=args.neg_qcov,
                pfam_span_source=args.pfam_span_source,
                rc_ov_aa=args.rc_ov_aa,
                rc_ov_frac=args.rc_ov_frac,
                palm_ov_aa=args.palm_ov_aa,
                palm_ov_frac=args.palm_ov_frac,
                pp_min_len=args.pp_min_len,
                pp_max_len=args.pp_max_len,
                ext_min_len=args.ext_min_len,
                ext_max_len=args.ext_max_len,
                sigma_k=args.sigma_k,
                conf_e_max=args.conf_e_max,
                conf_e_sat=args.conf_e_sat,
            )

            out_lab.write("\t".join(label_rec.get(k, "") for k in label_header) + "\n")
            out_dec.write("\t".join(decision_rec.get(k, "") for k in decision_header) + "\n")
            if h5_cols is not None:
                append_h5_record(h5_cols, label_rec)

            label_counts[label_rec["label"]] += 1
            decision_counts[decision_code] += 1

            if args.progress_every > 0 and i % args.progress_every == 0:
                log.info("Processed %d / %d records", i, len(ordered_lengths))

    summary = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fasta": args.fasta,
        "rdrpcatch_tsv": args.rdrpcatch_tsv,
        "palm_annot_tsv": args.palm_annot_tsv,
        "hmmscan_domtbl": args.hmmscan_domtbl,
        "palm_pfams": args.palm_pfams,
        "parameters": {
            "pos_ie": args.pos_ie,
            "palm_score_min": args.palm_score_min,
            "neg_ie": args.neg_ie,
            "neg_qcov": args.neg_qcov,
            "pfam_span_source": args.pfam_span_source,
            "rc_ov_aa": args.rc_ov_aa,
            "rc_ov_frac": args.rc_ov_frac,
            "palm_ov_aa": args.palm_ov_aa,
            "palm_ov_frac": args.palm_ov_frac,
            "pp_min_len": args.pp_min_len,
            "pp_max_len": args.pp_max_len,
            "ext_min_len": args.ext_min_len,
            "ext_max_len": args.ext_max_len,
            "conf_e_max": args.conf_e_max,
            "conf_e_sat": args.conf_e_sat,
            "sigma_k": args.sigma_k,
        },
        "counts": {
            "records": len(ordered_lengths),
            "labels": dict(label_counts),
            "decision_codes": dict(decision_counts),
            "rdrpcatch_ids_in_fasta": sum(1 for qid, _ in ordered_lengths if qid in rdrpcatch_hits),
            "palm_annot_ids_in_fasta": sum(1 for qid, _ in ordered_lengths if qid in palm_hits),
            "pfam_ids_in_fasta": sum(1 for qid, _ in ordered_lengths if qid in domtbl),
        },
        "outputs": {
            "label_tsv": args.out,
            "decision_tsv": args.decision_out,
            "label_h5": out_h5 or "",
        },
    }
    if args.summary_out:
        ensure_parent_dir(args.summary_out)
        with open(args.summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        log.info("Wrote summary JSON to %s", args.summary_out)

    if out_h5 is not None and h5_cols is not None:
        thresholds_meta = {
            "source": "rdrpcatch+palm_annot+pfam",
            "pos_ie": args.pos_ie,
            "palm_score_min": args.palm_score_min,
            "neg_ie": args.neg_ie,
            "neg_qcov": args.neg_qcov,
            "pfam_span_source": args.pfam_span_source,
            "rc_ov_aa": args.rc_ov_aa,
            "rc_ov_frac": args.rc_ov_frac,
            "palm_ov_aa": args.palm_ov_aa,
            "palm_ov_frac": args.palm_ov_frac,
            "pp_min_len": args.pp_min_len,
            "pp_max_len": args.pp_max_len,
            "ext_min_len": args.ext_min_len,
            "ext_max_len": args.ext_max_len,
            "conf_e_max": args.conf_e_max,
            "conf_e_sat": args.conf_e_sat,
            "sigma_k": args.sigma_k,
        }
        write_compact_labels_h5(
            out_h5=out_h5,
            cols=h5_cols,
            tsv_path=args.out,
            schema_version=args.schema_version,
            thresholds_meta=thresholds_meta,
            overwrite=args.overwrite_h5,
            logger=log,
        )

    log.info("Finished. Labels=%s", dict(label_counts))
    log.info("Top decision codes=%s", dict(decision_counts.most_common(10)))


if __name__ == "__main__":
    main()

