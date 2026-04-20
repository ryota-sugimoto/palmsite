#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P/U/N labels + span + sigma* + confidence + compact H5 output

This variant uses **merged evidence** from:
  - hmmscan --domtblout against Pfam (positive evidence via a Pfam allowlist)
  - rpsblast (positive evidence via RdRP CDD allowlist, and negative evidence via strong non-RdRP hits)

Labeling rules (strict):
  Let POS_HMM = any domtbl domain where:
      - Pfam accession (version-stripped) is in --palm-pfams
      - domain i-Evalue <= --pos-ie
  Let POS_RPS = any rpsblast subject where:
      - subject id is in --rdrp-ids
      - best evalue <= --pos-rps-e
      - query coverage (union across HSPs for that subject) >= --pos-rps-qcov
  Combine POS_HMM and POS_RPS with --pos-mode:
      - union        : POS = POS_HMM OR POS_RPS   (default)
      - intersection : POS = POS_HMM AND POS_RPS  (stricter)
  Let NEG_RPS = any rpsblast subject where:
      - subject id NOT in --rdrp-ids
      - best evalue <= --neg-e
      - query coverage (union across HSPs for that subject) >= --neg-qcov (default 0.80)
  Let NEG_HMM = any hmmscan domtbl *domain* where:
      - Pfam accession (version-stripped) is NOT in --palm-pfams
      - domain i-Evalue <= --neg-ie
      - single-domain query coverage (no merging across domains) >= --neg-qcov (default 0.80)
  Combine negatives (union): NEG = NEG_RPS OR NEG_HMM

Final label:
  - P if POS and not NEG
  - N if NEG and not POS
  - U otherwise (no significant hit OR conflict)

Span for P:
  - Prefer hmmscan span if POS_HMM (default uses env coords; --hmmscan-span ali available)
  - Else use rpsblast RdRP span built from POS_RPS HSPs.
  - Span is written as normalized S/E and as 0-based aa coords [start,end).

Confidence for P:
  - If hmmscan evidence used: from domain i-Evalue + span coverage + (optional) domtbl acc
  - Else (rps-only P): from best rps evalue + span coverage

Outputs:
  1) Readable TSV: --out labels_with_spans.tsv
  2) Compact H5:   --out-h5 labels_with_spans.h5  (default derived from --out)

The compact H5 schema matches labels_tsv_to_h5_compact.py:
  - One group: /labels
  - One dataset per column
  - label stored as uint8 (N=0, U=1, P=2)
  - numeric datasets compressed with lzf + shuffle

NOTE: This script DOES NOT modify the input embeddings HDF5. It reads it only to get sequence lengths.
"""
from __future__ import annotations

import sys
import os
import argparse
import logging
import gzip
import math
import json
import time
import hashlib
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional, Iterable

import numpy as np
import h5py


# -------------------------
# Logging
# -------------------------
def setup_logging(level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("rps_hmmscan_to_labels")


# -------------------------
# H5 lengths & sequences
# -------------------------
def load_lengths_and_seq(h5_path: str) -> Dict[str, Tuple[int, Optional[str]]]:
    """
    Read /items/<chunk_id> from an embeddings HDF5, and return {chunk_id: (L, seq_or_None)}.
    We prefer the 'seq' dataset if present, else fall back to group attr 'aa_len'.
    """
    out: Dict[str, Tuple[int, Optional[str]]] = {}
    with h5py.File(h5_path, "r") as f:
        items = f.get("items")
        if items is None:
            raise RuntimeError("H5 missing /items")
        for cid in items.keys():
            g = items[cid]
            L: Optional[int] = None
            seq: Optional[str] = None
            if "seq" in g:
                try:
                    seq = g["seq"].asstr()[()]
                except Exception:
                    val = g["seq"][()]
                    seq = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
                L = len(seq)
            if L is None or L <= 0:
                L = int(g.attrs.get("aa_len", -1))
                if L <= 0:
                    raise RuntimeError(f"length unknown for {cid}")
            out[cid] = (int(L), seq)
    return out


# -------------------------
# Read allowlists
# -------------------------
def read_id_list(path: str) -> set:
    """
    Read a simple newline-delimited ID list; ignores blank lines and comments starting with '#'.
    """
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                s.add(t)
    return s


def normalize_pfam_acc(acc: str) -> str:
    """
    Normalize a Pfam accession to the base form (no version), uppercase.
    Examples:
      PF00717.29 -> PF00717
      pf00717    -> PF00717
    """
    a = (acc or "").strip()
    if not a or a == "-":
        return ""
    a = a.split(".", 1)[0].strip()
    return a.upper()


def read_pfam_list(path: str) -> set:
    """
    Read Pfam accessions (PFxxxxx), one per line. Versions are allowed but removed.
    """
    raw = read_id_list(path)
    out = set()
    for x in raw:
        nx = normalize_pfam_acc(x)
        if nx:
            out.add(nx)
    return out


# -------------------------
# RPS parser
# -------------------------
class Hit:
    __slots__ = ("qid", "sid", "e", "bits", "qs", "qe", "ss", "se", "qlen", "slen", "title")

    def __init__(self, qid, sid, e, bits, qs, qe, ss, se, qlen, slen, title):
        self.qid = qid
        self.sid = sid
        self.e = e
        self.bits = bits
        self.qs = qs
        self.qe = qe
        self.ss = ss
        self.se = se
        self.qlen = qlen
        self.slen = slen
        self.title = title


def parse_rps_tsv(path: str) -> Dict[str, List[Hit]]:
    """
    Parse rpsblast outfmt 6 TSV. We only require the first ~12 columns that our older pipeline used:
      qseqid sacc evalue bitscore qstart qend sstart send length qlen slen stitle ...

    Extra columns (qcovs/qcovh/etc) are ignored safely.
    """
    hits: Dict[str, List[Hit]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            # pad to avoid IndexError
            while len(p) < 12:
                p.append("")
            qid = p[0]
            sid = p[1]
            try:
                e = float(p[2])
                bits = float(p[3])
                qs = int(p[4])
                qe = int(p[5])
                ss = int(p[6]) if p[6] else 0
                se = int(p[7]) if p[7] else 0
                qlen = int(p[9]) if len(p) > 9 and p[9] else None
                slen = int(p[10]) if len(p) > 10 and p[10] else None
            except Exception:
                continue
            title = p[11] if len(p) > 11 else ""
            hits[qid].append(Hit(qid, sid, e, bits, qs, qe, ss, se, qlen, slen, title))
    return hits


def sanity_check_qlen_vs_h5L(
    rps_hits: Dict[str, List[Hit]],
    qlens: Dict[str, tuple],
    logger: logging.Logger,
    abs_tol: int = 3,
    rel_tol: float = 0.10,
    max_warnings: int = 50,
) -> int:
    """
    Compare rpsblast 'qlen' against HDF5 aa_len (L).
    Warn if |qlen - L| > max(abs_tol, rel_tol*L).
    """
    n_warn = 0
    for qid, (L, _) in qlens.items():
        hits = rps_hits.get(qid)
        if not hits:
            continue
        qlen_values = [h.qlen for h in hits if isinstance(h.qlen, int) and h.qlen > 0]
        if not qlen_values:
            continue
        qlen_mode, _ = Counter(qlen_values).most_common(1)[0]
        thresh = max(abs_tol, int(rel_tol * max(1, L)))
        if abs(qlen_mode - L) > thresh:
            if max_warnings == 0 or n_warn < max_warnings:
                logger.warning("qlen mismatch for %s: rps=%d vs HDF5 L=%d (tol=%d)", qid, qlen_mode, L, thresh)
            n_warn += 1
    if max_warnings and n_warn > max_warnings:
        logger.warning("...and %d more qlen mismatches suppressed", n_warn - max_warnings)
    return n_warn


# -------------------------
# Intervals & coverage
# -------------------------
def to_half_open(qs: int, qe: int) -> Tuple[int, int]:
    """Convert 1-based inclusive BLAST qstart/qend to 0-based half-open [s,e)."""
    s = min(qs, qe) - 1
    e = max(qs, qe)
    return max(0, s), max(0, e)


def merge_intervals(iv: List[Tuple[int, int]], max_gap: int = 25) -> List[Tuple[int, int]]:
    if not iv:
        return []
    iv = sorted(iv, key=lambda x: (x[0], x[1]))
    out = [iv[0]]
    for s, e in iv[1:]:
        ps, pe = out[-1]
        if s <= pe + max_gap:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def union_len(iv: List[Tuple[int, int]]) -> int:
    return sum(max(0, e - s) for s, e in merge_intervals(iv, max_gap=0))


def qcov_of(hsps: List[Hit], L: int) -> float:
    if L <= 0:
        return 0.0
    iv = [to_half_open(h.qs, h.qe) for h in hsps]
    return min(1.0, union_len(iv) / L)


def dcov_of(hsps: List[Hit]) -> float:
    coords = []
    slen = None
    for h in hsps:
        if h.ss and h.se and h.slen:
            a, b = min(h.ss, h.se) - 1, max(h.ss, h.se)
            coords.append((max(0, a), max(0, b)))
            slen = h.slen
    if not coords or not slen:
        return 0.0
    return min(1.0, union_len(coords) / slen)


# -------------------------
# Span chooser for rps
# -------------------------
def choose_primary_span(rdrp_hsps: List[Hit], L: int, max_gap: int = 25) -> Tuple[int, int, bool]:
    """
    Choose a primary span from a list of rpsblast HSPs on the query.
    Returns (start_0based, end_0based_excl, truncated_flag).
    """
    if not rdrp_hsps:
        return (0, 0, False)

    rows = []
    for h in rdrp_hsps:
        s, e = to_half_open(h.qs, h.qe)
        rows.append((max(0, min(s, L)), max(0, min(e, L)), h.e, h.bits))
    rows.sort(key=lambda x: (x[0], x[1]))

    clusters = []  # [s,e,best_e,max_bits]
    for s, e, ev, bs in rows:
        if not clusters:
            clusters = [[s, e, ev, bs]]
        else:
            ps, pe, be, bb = clusters[-1]
            if s <= pe + max_gap:
                clusters[-1] = [ps, max(pe, e), min(be, ev), max(bb, bs)]
            else:
                clusters.append([s, e, ev, bs])

    best = None
    for s, e, be, bb in clusters:
        length = max(0, e - s)
        cand = (length, -be, bb, s, e)  # prefer longer, lower e, higher bits
        if best is None or cand > best:
            best = cand

    _, _, _, s, e = best
    s = max(0, min(s, L))
    e = max(0, min(e, L))
    truncated = (s == 0) or (e == L)
    return (s, e, truncated)


# -------------------------
# hmmscan domtbl parser
# -------------------------
@dataclass(frozen=True)
class HmmerDomHit:
    qid: str
    qlen: Optional[int]
    target_name: str
    target_acc: str       # as in file (may include version)
    target_acc_base: str  # normalized base (PFxxxxx)
    full_evalue: float
    full_score: float
    full_bias: float
    dom_idx: int
    dom_of: int
    c_evalue: float
    i_evalue: float
    dom_score: float
    dom_bias: float
    hmm_from: int
    hmm_to: int
    ali_from: int
    ali_to: int
    env_from: int
    env_to: int
    acc: Optional[float]
    description: str


def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_hmmscan_domtbl(path: str, logger: logging.Logger) -> Dict[str, List[HmmerDomHit]]:
    """
    Parse hmmscan --domtblout.

    Expected domtbl fields (space-delimited) are:
      target name, target accession, tlen,
      query name, query accession, qlen,
      E-value, score, bias,
      #, of,
      c-Evalue, i-Evalue, score, bias,
      hmm from, hmm to,
      ali from, ali to,
      env from, env to,
      acc, description...

    We keep all hits, grouped by query name (qid).
    """
    out: Dict[str, List[HmmerDomHit]] = defaultdict(list)
    n = 0
    n_bad = 0

    with _open_maybe_gzip(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            if ln.lstrip().startswith("#"):
                continue

            # Some users save domtbl snippets into RTF; skip obvious RTF control lines.
            line = ln.strip("\n")
            if line.startswith("{") or line.startswith("}") or line.startswith("\\"):
                continue

            # Remove trailing RTF line escape backslashes if present
            line = line.replace("\\", " ")

            parts = line.split()
            if len(parts) < 22:
                n_bad += 1
                continue

            try:
                target_name = parts[0]
                target_acc = parts[1]
                # tlen = int(parts[2])  # unused
                qid = parts[3]
                # qacc = parts[4]       # unused
                qlen = int(parts[5]) if parts[5] not in ("-", "") else None

                full_e = float(parts[6])
                full_score = float(parts[7])
                full_bias = float(parts[8])

                dom_idx = int(parts[9])
                dom_of = int(parts[10])

                c_e = float(parts[11])
                i_e = float(parts[12])
                dom_score = float(parts[13])
                dom_bias = float(parts[14])

                hmm_from = int(parts[15])
                hmm_to = int(parts[16])
                ali_from = int(parts[17])
                ali_to = int(parts[18])
                env_from = int(parts[19])
                env_to = int(parts[20])

                acc_raw = parts[21]
                acc = None
                try:
                    if acc_raw not in ("-", "NA", "nan", "NaN", ""):
                        acc = float(acc_raw)
                except Exception:
                    acc = None

                desc = " ".join(parts[22:]) if len(parts) > 22 else ""
            except Exception:
                n_bad += 1
                continue

            base = normalize_pfam_acc(target_acc)
            out[qid].append(
                HmmerDomHit(
                    qid=qid,
                    qlen=qlen,
                    target_name=target_name,
                    target_acc=target_acc,
                    target_acc_base=base,
                    full_evalue=full_e,
                    full_score=full_score,
                    full_bias=full_bias,
                    dom_idx=dom_idx,
                    dom_of=dom_of,
                    c_evalue=c_e,
                    i_evalue=i_e,
                    dom_score=dom_score,
                    dom_bias=dom_bias,
                    hmm_from=hmm_from,
                    hmm_to=hmm_to,
                    ali_from=ali_from,
                    ali_to=ali_to,
                    env_from=env_from,
                    env_to=env_to,
                    acc=acc,
                    description=desc,
                )
            )
            n += 1

    logger.info("Loaded hmmscan domtbl rows=%d (bad=%d), unique_qids=%d from %s", n, n_bad, len(out), path)
    return out


def sanity_check_domtbl_qlen_vs_h5L(
    dom_hits: Dict[str, List[HmmerDomHit]],
    qlens: Dict[str, tuple],
    logger: logging.Logger,
    abs_tol: int = 3,
    rel_tol: float = 0.10,
    max_warnings: int = 50,
) -> int:
    """
    Compare hmmscan domtbl 'qlen' against HDF5 aa_len (L).
    """
    n_warn = 0
    for qid, (L, _) in qlens.items():
        hits = dom_hits.get(qid)
        if not hits:
            continue
        qlen_values = [h.qlen for h in hits if isinstance(h.qlen, int) and h.qlen > 0]
        if not qlen_values:
            continue
        qlen_mode, _ = Counter(qlen_values).most_common(1)[0]
        thresh = max(abs_tol, int(rel_tol * max(1, L)))
        if abs(qlen_mode - L) > thresh:
            if max_warnings == 0 or n_warn < max_warnings:
                logger.warning("qlen mismatch for %s: hmmscan=%d vs HDF5 L=%d (tol=%d)", qid, qlen_mode, L, thresh)
            n_warn += 1
    if max_warnings and n_warn > max_warnings:
        logger.warning("...and %d more hmmscan qlen mismatches suppressed", n_warn - max_warnings)
    return n_warn


# -------------------------
# hmmscan span chooser (clustered)
# -------------------------
def _span_from_dom(hit: HmmerDomHit, span_source: str) -> Tuple[int, int]:
    """
    Return a 0-based half-open span [s,e) from a domtbl hit, using either env or ali coords.
    domtbl coords are 1-based inclusive.
    """
    if span_source == "ali":
        lo = min(hit.ali_from, hit.ali_to)
        hi = max(hit.ali_from, hit.ali_to)
    else:
        lo = min(hit.env_from, hit.env_to)
        hi = max(hit.env_from, hit.env_to)
    s = max(0, int(lo) - 1)
    e = max(0, int(hi))
    return s, e


def choose_primary_span_domtbl(
    doms: List[HmmerDomHit],
    L: int,
    max_gap: int = 25,
    span_source: str = "env",
) -> Tuple[int, int, bool, Optional[HmmerDomHit], float, float]:
    """
    Choose a primary span from domtbl hits (already filtered to relevant Pfams).

    We first convert each domain hit into a query interval (env or ali coords),
    then merge nearby intervals (<= max_gap) into clusters, and select the best cluster.

    Returns:
      (start, end, truncated, representative_hit, best_iE, best_acc)

    Where:
      - representative_hit is the *single* hit in the best cluster with the lowest i-Evalue
        (ties broken by higher dom_score).
      - best_iE is the best (minimum) i-Evalue within the best cluster.
      - best_acc is the max domtbl 'acc' within the best cluster (0 if missing).
    """
    if not doms:
        return (0, 0, False, None, float("inf"), 0.0)

    rows = []
    for h in doms:
        s, e = _span_from_dom(h, span_source=span_source)
        s = max(0, min(s, L))
        e = max(0, min(e, L))
        if e <= s:
            continue
        rows.append((s, e, h.i_evalue, h.dom_score, h.acc if h.acc is not None else -1.0, h))

    if not rows:
        return (0, 0, False, None, float("inf"), 0.0)

    rows.sort(key=lambda x: (x[0], x[1]))

    # clusters are:
    #   [s, e, best_iE, max_score, best_acc, rep_hit, rep_iE, rep_score]
    clusters = []
    for s, e, ie, sc, ac, h in rows:
        if not clusters:
            clusters.append([s, e, ie, sc, ac, h, ie, sc])
            continue

        ps, pe, bie, bsc, bac, rep, rep_ie, rep_sc = clusters[-1]
        if s <= pe + max_gap:
            ns = ps
            ne = max(pe, e)
            nie = min(bie, ie)
            nsc = max(bsc, sc)
            nac = max(bac, ac)

            # representative = lowest iE, then highest score
            nrep = rep
            nrep_ie = rep_ie
            nrep_sc = rep_sc
            if (ie < nrep_ie) or (ie == nrep_ie and sc > nrep_sc):
                nrep = h
                nrep_ie = ie
                nrep_sc = sc

            clusters[-1] = [ns, ne, nie, nsc, nac, nrep, nrep_ie, nrep_sc]
        else:
            clusters.append([s, e, ie, sc, ac, h, ie, sc])

    best = None
    best_cluster = None
    for s, e, bie, bsc, bac, rep, _, _ in clusters:
        length = max(0, e - s)
        ie_score = _safe_log10_evalue(bie)  # larger is better
        # prefer longer; then better iE; then higher score; then earlier start
        cand = (length, ie_score, bsc, -s, e)
        if best is None or cand > best:
            best = cand
            best_cluster = (s, e, bie, bac, rep)

    if best_cluster is None:
        return (0, 0, False, None, float("inf"), 0.0)

    s, e, bie, bac, rep = best_cluster
    s = max(0, min(s, L))
    e = max(0, min(e, L))
    truncated = (s == 0) or (e == L)
    best_acc = 0.0 if bac < 0 else bac
    return (s, e, truncated, rep, bie, best_acc)


# -------------------------
# Confidence utilities
# -------------------------
def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _safe_log10_evalue(ev: float) -> float:
    """
    Convert an E-value to a stable -log10 scale.
    HMMER may print 0.0 for extremely small values; treat as 1e-300.
    """
    if ev is None or (isinstance(ev, float) and math.isnan(ev)):
        return 0.0
    if ev <= 0.0:
        ev = 1e-300
    return -math.log10(ev)


def conf_from_hmmer(
    i_evalue: float,
    pos_ie_max: float,
    pos_ie_sat: float,
    cov: float,
    acc_prob: Optional[float],
    truncated: bool,
) -> Tuple[float, float, float, float]:
    """
    Confidence components from hmmscan domain i-Evalue + coverage + domtbl acc.
    """
    # Ensure sensible scale: sat should be <= max (better == smaller e-value)
    if pos_ie_sat > pos_ie_max:
        pos_ie_sat = pos_ie_max

    s_min = _safe_log10_evalue(pos_ie_max)
    s_sat = _safe_log10_evalue(pos_ie_sat)
    if s_sat <= s_min:
        s_sat = s_min + 1.0

    s = _safe_log10_evalue(i_evalue)
    conf_e = clip01((s - s_min) / (s_sat - s_min))

    conf_cov = clip01(cov / 0.60)  # saturate at 60%
    conf_hsp = clip01(acc_prob) if acc_prob is not None else 0.0

    conf = 0.6 * conf_e + 0.3 * conf_cov + 0.1 * conf_hsp
    if truncated:
        conf *= 0.5
    return conf, conf_e, conf_cov, conf_hsp


def conf_from_rps(
    evalue: float,
    pos_e_max: float,
    pos_e_sat: float,
    cov: float,
    truncated: bool,
) -> Tuple[float, float, float, float]:
    """
    Confidence components from rpsblast best e-value + coverage.
    (No HSP posterior-prob column; conf_hsp=0.)
    """
    if pos_e_sat > pos_e_max:
        pos_e_sat = pos_e_max

    s_min = _safe_log10_evalue(pos_e_max)
    s_sat = _safe_log10_evalue(pos_e_sat)
    if s_sat <= s_min:
        s_sat = s_min + 1.0

    s = _safe_log10_evalue(evalue)
    conf_e = clip01((s - s_min) / (s_sat - s_min))
    conf_cov = clip01(cov / 0.60)
    conf_hsp = 0.0

    conf = 0.6 * conf_e + 0.3 * conf_cov + 0.1 * conf_hsp
    if truncated:
        conf *= 0.5
    return conf, conf_e, conf_cov, conf_hsp


# -------------------------
# Output row (typed)
# -------------------------
LABEL_TO_U8 = {"N": 0, "U": 1, "P": 2}


@dataclass
class LabelRow:
    chunk_id: str
    label: str  # "P"/"U"/"N"
    use_span: int
    L: int
    S: float  # NaN if no span
    E: float  # NaN if no span
    span_start_aa: str
    span_end_aa: str
    truncated: int
    conf: float
    conf_e: float
    conf_cov: float
    conf_hsp: float
    w_min: float
    sigma_star: float  # NaN if no span
    k: float

    # evidence columns supported by labels_tsv_to_h5_compact.py
    rps_pos_e_min: str
    rps_pos_models: str
    rps_pos_qcov_max: float  # NaN if none
    rps_pos_dcov_max: float  # NaN if none
    rps_neg_models: str
    rps_neg_qcov_max: float  # NaN if none


def _fmt_float(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.{nd}f}"


def _fmt_float6(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.6f}"


def _fmt_evalue(ev: Optional[float]) -> str:
    if ev is None:
        return ""
    try:
        if not math.isfinite(ev):
            return ""
    except Exception:
        pass
    return f"{ev:.3g}"


def _join_models(models: Iterable[str], limit: int = 12) -> str:
    xs = [m for m in models if m]
    if not xs:
        return ""
    if len(xs) <= limit:
        return ",".join(xs)
    return ",".join(xs[:limit]) + f",...(+{len(xs)-limit})"


# -------------------------
# Labeling
# -------------------------
def decide_labels_for_query(
    qid: str,
    L: int,
    rps_hits: List[Hit],
    rdrp_ids: set,
    hmmscan_hits: List[HmmerDomHit],
    palm_pfams: set,
    pos_mode: str,
    pos_ie_max: float,
    pos_ie_sat: float,
    hmmscan_span: str,
    pos_rps_e: float,
    pos_rps_e_sat: float,
    pos_rps_qcov: float,
    thr_neg_ie: float,
    thr_neg_e: float,
    thr_neg_qcov: float,
    max_gap_for_span: int,
    sigma_k: float,
) -> Tuple[LabelRow, Optional[HmmerDomHit]]:
    """
    Returns:
      (LabelRow, representative_hmmscan_hit_or_None)
    """
    # group rps hits by subject
    by_sid: Dict[str, List[Hit]] = defaultdict(list)
    for h in rps_hits:
        by_sid[h.sid].append(h)

    subs = []
    for sid, hsps in by_sid.items():
        best_e = min(h.e for h in hsps)
        qc = qcov_of(hsps, L)
        dc = dcov_of(hsps)
        subs.append((sid, best_e, qc, dc, hsps))

    # Positive evidence from hmmscan: any *domain* hit to allowed Pfams
    hm_pos_hits = [
        h for h in hmmscan_hits
        if (h.target_acc_base in palm_pfams) and (h.i_evalue <= pos_ie_max)
    ]
    pos_hmm = len(hm_pos_hits) > 0

    # Positive evidence from rps: confident RdRP CDD hit(s)
    rps_pos_subs = [
        (sid, e, qc, dc, hsps) for sid, e, qc, dc, hsps in subs
        if (sid in rdrp_ids) and (e <= pos_rps_e) and (qc >= pos_rps_qcov)
    ]
    pos_rps = len(rps_pos_subs) > 0

    # Negative evidence from rpsblast: strong non-RdRP hits with large qcov (per-subject; no merging across subjects)
    neg_subs = [
        (sid, e, qc, dc, hsps) for sid, e, qc, dc, hsps in subs
        if (sid not in rdrp_ids) and (e <= thr_neg_e) and (qc >= thr_neg_qcov)
    ]
    neg_rps = len(neg_subs) > 0

    # Negative evidence from hmmscan: a *single* non-palm domain covering >= thr_neg_qcov (no merging across domains)
    hm_neg_candidates: List[Tuple[float, float, HmmerDomHit]] = []  # (cov, ie_score, hit)
    for h in hmmscan_hits:
        if h.target_acc_base in palm_pfams:
            continue
        if h.i_evalue > thr_neg_ie:
            continue
        s0, e0 = _span_from_dom(h, span_source=hmmscan_span)
        s0 = max(0, min(s0, L))
        e0 = max(0, min(e0, L))
        if e0 <= s0:
            continue
        cov = (e0 - s0) / float(L) if L > 0 else 0.0
        if cov >= thr_neg_qcov:
            hm_neg_candidates.append((cov, _safe_log10_evalue(h.i_evalue), h))

    neg_hmm = len(hm_neg_candidates) > 0
    rep_neg_hit: Optional[HmmerDomHit] = None
    if hm_neg_candidates:
        # prefer larger coverage; then better i-Evalue; then higher dom_score
        rep_neg_hit = max(hm_neg_candidates, key=lambda x: (x[0], x[1], x[2].dom_score))[2]

    # Union merge negatives
    neg = neg_rps or neg_hmm

    # Combine pos signals
    if pos_mode == "intersection":
        pos = pos_hmm and pos_rps
    else:
        pos = pos_hmm or pos_rps

    # Label resolution (strict)
    if pos and (not neg):
        label = "P"
    elif neg and (not pos):
        label = "N"
    else:
        label = "U"

    # Evidence summaries (for TSV/H5 compatibility)
    rps_pos_e_min = _fmt_evalue(min((e for _, e, _, _, _ in rps_pos_subs), default=None))
    rps_pos_models = _join_models([sid for sid, _, _, _, _ in sorted(rps_pos_subs, key=lambda x: x[1])])
    rps_pos_qcov_max = max((qc for _, _, qc, _, _ in rps_pos_subs), default=float("nan"))
    rps_pos_dcov_max = max((dc for _, _, _, dc, _ in rps_pos_subs), default=float("nan"))

    rps_neg_models = _join_models([sid for sid, _, _, _, _ in sorted(neg_subs, key=lambda x: x[1])])
    rps_neg_qcov_max = max((qc for _, _, qc, _, _ in neg_subs), default=float("nan"))

    # Defaults
    use_span = 0
    S = float("nan")
    E = float("nan")
    span_s = ""
    span_e = ""
    truncated = 0
    conf = conf_e = conf_cov = conf_hsp = 0.0
    w_min = max(70.0 / max(1.0, float(L)), 0.02)  # normalized
    sigma_star = float("nan")

    rep_hit: Optional[HmmerDomHit] = None

    # If we labeled N due to a hmmscan negative hit, keep a representative hit for optional debug columns
    if label == "N" and rep_neg_hit is not None:
        rep_hit = rep_neg_hit

    # For P: span + confidence
    if label == "P":
        s: Optional[int] = None
        e: Optional[int] = None
        trunc_flag = False

        # Prefer hmmscan span if available
        if pos_hmm:
            s0, e0, trunc0, rep, best_ie, best_acc = choose_primary_span_domtbl(
                doms=hm_pos_hits,
                L=L,
                max_gap=max_gap_for_span,
                span_source=hmmscan_span,
            )
            if e0 > s0:
                s, e = s0, e0
                trunc_flag = trunc0
                rep_hit = rep

            cov = ((e - s) / L) if (s is not None and e is not None and L > 0) else 0.0
            # Confidence always from hmmscan when hm evidence exists
            best_ie_use = best_ie if math.isfinite(best_ie) else min(h.i_evalue for h in hm_pos_hits)
            conf, conf_e, conf_cov, conf_hsp = conf_from_hmmer(
                i_evalue=float(best_ie_use),
                pos_ie_max=pos_ie_max,
                pos_ie_sat=pos_ie_sat,
                cov=float(cov),
                acc_prob=(rep_hit.acc if rep_hit is not None else best_acc),
                truncated=trunc_flag if (s is not None and e is not None) else False,
            )

        else:
            # rps-only span from confident RdRP hsps
            rdrp_hsps = [h for sid, _, _, _, hsps in rps_pos_subs for h in hsps]
            if rdrp_hsps:
                s0, e0, trunc0 = choose_primary_span(rdrp_hsps, L, max_gap=max_gap_for_span)
                if e0 > s0:
                    s, e = s0, e0
                    trunc_flag = trunc0

            cov = ((e - s) / L) if (s is not None and e is not None and L > 0) else 0.0
            # best rps evalue among pos subs (should exist if label==P and pos_hmm is False)
            best_rps_e = min((ev for _, ev, _, _, _ in rps_pos_subs), default=pos_rps_e)
            conf, conf_e, conf_cov, conf_hsp = conf_from_rps(
                evalue=float(best_rps_e),
                pos_e_max=pos_rps_e,
                pos_e_sat=pos_rps_e_sat,
                cov=float(cov),
                truncated=trunc_flag if (s is not None and e is not None) else False,
            )

        if s is not None and e is not None and e > s:
            use_span = 1
            truncated = 1 if trunc_flag else 0
            S = float(s) / float(L) if L > 0 else float("nan")
            E = float(e) / float(L) if L > 0 else float("nan")
            span_s = str(int(s))
            span_e = str(int(e))

            w = E - S
            sigma_star = max(w / (2.0 * sigma_k), w_min / 2.0)

    row = LabelRow(
        chunk_id=qid,
        label=label,
        use_span=int(use_span),
        L=int(L),
        S=float(S),
        E=float(E),
        span_start_aa=span_s,
        span_end_aa=span_e,
        truncated=int(truncated),
        conf=float(conf),
        conf_e=float(conf_e),
        conf_cov=float(conf_cov),
        conf_hsp=float(conf_hsp),
        w_min=float(w_min),
        sigma_star=float(sigma_star),
        k=float(sigma_k),
        rps_pos_e_min=rps_pos_e_min,
        rps_pos_models=rps_pos_models,
        rps_pos_qcov_max=float(rps_pos_qcov_max),
        rps_pos_dcov_max=float(rps_pos_dcov_max),
        rps_neg_models=rps_neg_models,
        rps_neg_qcov_max=float(rps_neg_qcov_max),
    )
    return row, rep_hit


# -------------------------
# Compact H5 writer (schema-compatible)
# -------------------------
def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)


def write_compact_labels_h5(
    out_h5: str,
    rows: List[LabelRow],
    tsv_path: str,
    schema_version: str,
    thresholds_meta: dict,
    overwrite: bool,
    logger: logging.Logger,
) -> None:
    """
    Write a compact labels.h5 under /labels, compatible with labels_tsv_to_h5_compact.py.
    """
    if os.path.exists(out_h5):
        if overwrite:
            logger.warning("Overwriting %s", out_h5)
            os.remove(out_h5)
        else:
            raise FileExistsError(f"Refusing to overwrite existing H5: {out_h5} (use --overwrite-h5)")

    _ensure_parent_dir(out_h5)

    vstr = h5py.string_dtype(encoding="utf-8")
    N = len(rows)

    # Build numpy arrays
    chunk_id = np.array([r.chunk_id for r in rows], dtype=vstr)
    label_u8 = np.fromiter((LABEL_TO_U8.get(r.label, 255) for r in rows), dtype=np.uint8, count=N)

    use_span = np.fromiter((np.uint8(1 if r.use_span else 0) for r in rows), dtype=np.uint8, count=N)
    truncated = np.fromiter((np.uint8(1 if r.truncated else 0) for r in rows), dtype=np.uint8, count=N)
    L_arr = np.fromiter((np.int32(r.L) for r in rows), dtype=np.int32, count=N)

    def f32(vals: Iterable[float]) -> np.ndarray:
        return np.array(list(vals), dtype=np.float32)

    S_arr = f32((r.S for r in rows))
    E_arr = f32((r.E for r in rows))
    conf_arr = f32((r.conf for r in rows))
    conf_e_arr = f32((r.conf_e for r in rows))
    conf_cov_arr = f32((r.conf_cov for r in rows))
    conf_hsp_arr = f32((r.conf_hsp for r in rows))
    w_min_arr = f32((r.w_min for r in rows))
    sigma_star_arr = f32((r.sigma_star for r in rows))
    k_arr = f32((r.k for r in rows))

    rps_pos_qcov_max_arr = f32((r.rps_pos_qcov_max for r in rows))
    rps_pos_dcov_max_arr = f32((r.rps_pos_dcov_max for r in rows))
    rps_neg_qcov_max_arr = f32((r.rps_neg_qcov_max for r in rows))

    # strings (optional evidence)
    rps_pos_e_min = np.array([r.rps_pos_e_min for r in rows], dtype=vstr)
    rps_pos_models = np.array([r.rps_pos_models for r in rows], dtype=vstr)
    rps_neg_models = np.array([r.rps_neg_models for r in rows], dtype=vstr)
    span_start_aa = np.array([r.span_start_aa for r in rows], dtype=vstr)
    span_end_aa = np.array([r.span_end_aa for r in rows], dtype=vstr)

    with h5py.File(out_h5, "w", libver="latest") as f:
        g = f.create_group("labels")

        # Strings: no compression/shuffle
        g.create_dataset("chunk_id", data=chunk_id)
        g.create_dataset("rps_pos_e_min", data=rps_pos_e_min)
        g.create_dataset("rps_pos_models", data=rps_pos_models)
        g.create_dataset("rps_neg_models", data=rps_neg_models)
        g.create_dataset("span_start_aa", data=span_start_aa)
        g.create_dataset("span_end_aa", data=span_end_aa)

        # Numerics: lzf + shuffle + chunked
        def dset_num(name: str, arr: np.ndarray) -> None:
            g.create_dataset(name, data=arr, compression="lzf", shuffle=True, chunks=True)

        dset_num("label", label_u8)
        dset_num("use_span", use_span)
        dset_num("truncated", truncated)
        dset_num("L", L_arr)

        dset_num("S", S_arr)
        dset_num("E", E_arr)
        dset_num("conf", conf_arr)
        dset_num("conf_e", conf_e_arr)
        dset_num("conf_cov", conf_cov_arr)
        dset_num("conf_hsp", conf_hsp_arr)
        dset_num("w_min", w_min_arr)
        dset_num("sigma_star", sigma_star_arr)
        dset_num("k", k_arr)

        dset_num("rps_pos_qcov_max", rps_pos_qcov_max_arr)
        dset_num("rps_pos_dcov_max", rps_pos_dcov_max_arr)
        dset_num("rps_neg_qcov_max", rps_neg_qcov_max_arr)

        # Group attrs
        g.attrs["label_map"] = json.dumps({"N": 0, "U": 1, "P": 2})

        # File attrs
        f.attrs["schema_version"] = schema_version
        f.attrs["thresholds"] = json.dumps(thresholds_meta)

        # record provenance (sha256 of TSV)
        with open(tsv_path, "rb") as fh:
            f.attrs["tsv_sha256"] = hashlib.sha256(fh.read()).hexdigest()
        f.attrs["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    logger.info("Wrote %s with %d rows; size ~ %.2f MB", out_h5, N, os.path.getsize(out_h5) / 1e6)


# -------------------------
# Main
# -------------------------
def default_out_h5(tsv_path: str) -> str:
    low = tsv_path.lower()
    if low.endswith(".tsv"):
        return tsv_path[:-4] + ".h5"
    if low.endswith(".txt"):
        return tsv_path[:-4] + ".h5"
    return tsv_path + ".h5"


def main() -> None:
    ap = argparse.ArgumentParser(
        description="P/U/N + span + sigma* using merged hmmscan(domtbl) + rpsblast evidence; outputs TSV + compact H5"
    )
    ap.add_argument("--h5", required=True, help="embeddings HDF5 (read-only; used only for lengths)")
    ap.add_argument("--rps", required=True, help="rpsblast outfmt 6 TSV")
    ap.add_argument("--rdrp-ids", required=True, help="file with RdRP CDD IDs (one per line)")
    ap.add_argument("--hmmscan-domtbl", required=True, help="hmmscan --domtblout (Pfam) (can be .gz)")
    ap.add_argument("--palm-pfams", required=True, help="file with Pfam IDs for RdRP palm domains (one per line), e.g. PF04197")
    ap.add_argument("--out", required=True, help="labels_with_spans.tsv")

    # H5 output
    ap.add_argument("--out-h5", default=None, help="compact labels.h5 path (default: derive from --out)")
    ap.add_argument("--no-h5", action="store_true", help="do not write H5 (TSV only)")
    ap.add_argument("--overwrite-h5", action="store_true", help="overwrite existing --out-h5 if it exists")
    ap.add_argument("--schema-version", default="1.0", help="schema_version attribute for output H5")
    ap.add_argument("--log-level", default="INFO")

    # Positive: hmmscan thresholds
    ap.add_argument("--pos-ie", type=float, default=1e-5, help="max domain i-Evalue to count as hmmscan positive")
    ap.add_argument(
        "--pos-ie-sat",
        type=float,
        default=1e-30,
        help="domain i-Evalue at which hmmscan confidence saturates (must be <= --pos-ie; default 1e-30)",
    )
    ap.add_argument(
        "--hmmscan-span",
        choices=["env", "ali"],
        default="env",
        help="Span coords to use from domtblout: env (envelope) or ali (alignment).",
    )

    # Positive: rps thresholds
    ap.add_argument("--pos-rps-e", type=float, default=1e-10, help="max rps evalue to count as RdRP positive")
    ap.add_argument("--pos-rps-e-sat", type=float, default=1e-50, help="rps evalue where confidence saturates (<= --pos-rps-e)")
    ap.add_argument("--pos-rps-qcov", type=float, default=0.35, help="min query coverage for RdRP rps hit to count as positive")

    # Combine pos signals
    ap.add_argument(
        "--pos-mode",
        choices=["union", "intersection"],
        default="union",
        help="How to combine hmmscan and rps positive evidence: union (default) or intersection (stricter).",
    )

    # Negative thresholds
    ap.add_argument("--neg-ie", type=float, default=1e-5, help="max hmmscan domain i-Evalue to count as strong negative evidence (single non-palm domain hit)")
    ap.add_argument("--neg-e", type=float, default=1e-5, help="max rps evalue to count as strong negative evidence")
    ap.add_argument("--neg-qcov", type=float, default=0.80, help="min query coverage for non-RdRP hit (rps or hmmscan) to count as negative (default 0.80)")

    # span clustering
    ap.add_argument("--span-max-gap", type=int, default=25, help="max aa gap when merging nearby HSPs/domains into one span")

    # sigma target
    ap.add_argument("--sigma-k", type=float, default=2.0, help="k in sigma* = max(w/(2k), w_min/2)")

    # qlen sanity
    ap.add_argument("--qlen-abs-tol", type=int, default=3)
    ap.add_argument("--qlen-rel-tol", type=float, default=0.10)
    ap.add_argument("--qlen-max-warn", type=int, default=50)

    # optional debug columns (TSV only)
    ap.add_argument(
        "--append-hmmscan-cols",
        action="store_true",
        help="Append representative hmmscan fields (pfam_acc, pfam_name, dom_iE, dom_score, coords, acc_prob) to output TSV.",
    )

    args = ap.parse_args()
    log = setup_logging(args.log_level)

    # Validate e-values
    for name in ("pos_ie", "pos_ie_sat", "pos_rps_e", "pos_rps_e_sat", "neg_ie", "neg_e"):
        v = getattr(args, name)
        if v <= 0:
            log.error("--%s must be > 0", name.replace("_", "-"))
            sys.exit(2)

    if args.pos_ie_sat > args.pos_ie:
        log.warning("--pos-ie-sat (%.2g) > --pos-ie (%.2g); adjusting sat to pos-ie", args.pos_ie_sat, args.pos_ie)
        args.pos_ie_sat = args.pos_ie

    if args.pos_rps_e_sat > args.pos_rps_e:
        log.warning("--pos-rps-e-sat (%.2g) > --pos-rps-e (%.2g); adjusting sat to pos-rps-e", args.pos_rps_e_sat, args.pos_rps_e)
        args.pos_rps_e_sat = args.pos_rps_e

    if not (0.0 <= args.pos_rps_qcov <= 1.0):
        log.error("--pos-rps-qcov must be in [0,1]")
        sys.exit(2)
    if not (0.0 <= args.neg_qcov <= 1.0):
        log.error("--neg-qcov must be in [0,1]")
        sys.exit(2)

    # Determine H5 output path
    out_h5 = None if args.no_h5 else (args.out_h5 or default_out_h5(args.out))

    qlens = load_lengths_and_seq(args.h5)
    rps = parse_rps_tsv(args.rps)
    domtbl = parse_hmmscan_domtbl(args.hmmscan_domtbl, logger=log)

    log.info("Sanity-checking qlen (rps) vs L (HDF5)...")
    mismatches_rps = sanity_check_qlen_vs_h5L(
        rps_hits=rps,
        qlens=qlens,
        logger=log,
        abs_tol=args.qlen_abs_tol,
        rel_tol=args.qlen_rel_tol,
        max_warnings=args.qlen_max_warn,
    )
    if mismatches_rps:
        log.warning("Found %d rps qlen≠L mismatches (see warnings above).", mismatches_rps)

    log.info("Sanity-checking qlen (hmmscan) vs L (HDF5)...")
    mismatches_hmm = sanity_check_domtbl_qlen_vs_h5L(
        dom_hits=domtbl,
        qlens=qlens,
        logger=log,
        abs_tol=args.qlen_abs_tol,
        rel_tol=args.qlen_rel_tol,
        max_warnings=args.qlen_max_warn,
    )
    if mismatches_hmm:
        log.warning("Found %d hmmscan qlen≠L mismatches (see warnings above).", mismatches_hmm)

    rdrp_ids = read_id_list(args.rdrp_ids)
    if not rdrp_ids:
        log.error("Empty RdRP ID list: %s", args.rdrp_ids)
        sys.exit(2)

    palm_pfams = read_pfam_list(args.palm_pfams)
    if not palm_pfams:
        log.error("Empty palm Pfam list: %s", args.palm_pfams)
        sys.exit(2)

    # TSV header (base columns + evidence columns)
    base_header = [
        "chunk_id", "label", "use_span", "L", "S", "E",
        "span_start_aa", "span_end_aa", "truncated",
        "conf", "conf_e", "conf_cov", "conf_hsp",
        "w_min", "sigma_star", "k",
        "rps_pos_e_min", "rps_pos_models", "rps_pos_qcov_max", "rps_pos_dcov_max",
        "rps_neg_models", "rps_neg_qcov_max",
    ]
    header = list(base_header)
    if args.append_hmmscan_cols:
        header += [
            "hm_pfam_acc",
            "hm_target_name",
            "hm_dom_iE",
            "hm_dom_score",
            "hm_span_source",
            "hm_span_from_1based",
            "hm_span_to_1based",
            "hm_acc_prob",
        ]

    # Collect rows for H5
    rows: List[LabelRow] = []

    n = nP = nU = nN = nSpan = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("\t".join(header) + "\n")

        for qid, (L, _) in qlens.items():
            row_obj, rep_hit = decide_labels_for_query(
                qid=qid,
                L=L,
                rps_hits=rps.get(qid, []),
                rdrp_ids=rdrp_ids,
                hmmscan_hits=domtbl.get(qid, []),
                palm_pfams=palm_pfams,
                pos_mode=args.pos_mode,
                pos_ie_max=args.pos_ie,
                pos_ie_sat=args.pos_ie_sat,
                hmmscan_span=args.hmmscan_span,
                pos_rps_e=args.pos_rps_e,
                pos_rps_e_sat=args.pos_rps_e_sat,
                pos_rps_qcov=args.pos_rps_qcov,
                thr_neg_ie=args.neg_ie,
                thr_neg_e=args.neg_e,
                thr_neg_qcov=args.neg_qcov,
                max_gap_for_span=args.span_max_gap,
                sigma_k=args.sigma_k,
            )
            rows.append(row_obj)

            # TSV row formatting
            rec = {
                "chunk_id": row_obj.chunk_id,
                "label": row_obj.label,
                "use_span": str(row_obj.use_span),
                "L": str(row_obj.L),
                "S": _fmt_float6(row_obj.S),
                "E": _fmt_float6(row_obj.E),
                "span_start_aa": row_obj.span_start_aa,
                "span_end_aa": row_obj.span_end_aa,
                "truncated": str(int(row_obj.truncated)),
                "conf": _fmt_float(row_obj.conf, 4),
                "conf_e": _fmt_float(row_obj.conf_e, 4),
                "conf_cov": _fmt_float(row_obj.conf_cov, 4),
                "conf_hsp": _fmt_float(row_obj.conf_hsp, 4),
                "w_min": _fmt_float6(row_obj.w_min),
                "sigma_star": _fmt_float6(row_obj.sigma_star),
                "k": _fmt_float(row_obj.k, 3),
                "rps_pos_e_min": row_obj.rps_pos_e_min,
                "rps_pos_models": row_obj.rps_pos_models,
                "rps_pos_qcov_max": _fmt_float(row_obj.rps_pos_qcov_max, 4),
                "rps_pos_dcov_max": _fmt_float(row_obj.rps_pos_dcov_max, 4),
                "rps_neg_models": row_obj.rps_neg_models,
                "rps_neg_qcov_max": _fmt_float(row_obj.rps_neg_qcov_max, 4),
            }

            row = [rec.get(k, "") for k in base_header]

            if args.append_hmmscan_cols:
                if rep_hit is None:
                    row += ["", "", "", "", "", "", "", ""]
                else:
                    if args.hmmscan_span == "ali":
                        lo = min(rep_hit.ali_from, rep_hit.ali_to)
                        hi = max(rep_hit.ali_from, rep_hit.ali_to)
                    else:
                        lo = min(rep_hit.env_from, rep_hit.env_to)
                        hi = max(rep_hit.env_from, rep_hit.env_to)
                    row += [
                        rep_hit.target_acc_base,
                        rep_hit.target_name,
                        f"{rep_hit.i_evalue:.3g}",
                        f"{rep_hit.dom_score:.2f}",
                        args.hmmscan_span,
                        str(lo),
                        str(hi),
                        "" if rep_hit.acc is None else f"{rep_hit.acc:.3f}",
                    ]

            out.write("\t".join(row) + "\n")

            n += 1
            if row_obj.label == "P":
                nP += 1
                if row_obj.use_span == 1:
                    nSpan += 1
            elif row_obj.label == "N":
                nN += 1
            else:
                nU += 1

    log.info("Wrote TSV %s (%d rows) [P=%d (span=%d), U=%d, N=%d]", args.out, n, nP, nSpan, nU, nN)

    if out_h5 is not None:
        thresholds_meta = {
            # new keys
            "pos_mode": args.pos_mode,
            "pos_ie": args.pos_ie,
            "pos_ie_sat": args.pos_ie_sat,
            "pos_rps_e": args.pos_rps_e,
            "pos_rps_e_sat": args.pos_rps_e_sat,
            "pos_rps_qcov": args.pos_rps_qcov,
            "neg_ie": args.neg_ie,
            "neg_e": args.neg_e,
            "neg_qcov": args.neg_qcov,
            # legacy-ish keys used by older converter scripts/readers
            "pos_e": args.pos_ie,
            "pos_cov": args.pos_rps_qcov,
            "loose_rdrp_e": args.pos_rps_e,
        }
        write_compact_labels_h5(
            out_h5=out_h5,
            rows=rows,
            tsv_path=args.out,
            schema_version=args.schema_version,
            thresholds_meta=thresholds_meta,
            overwrite=args.overwrite_h5,
            logger=log,
        )


if __name__ == "__main__":
    main()
