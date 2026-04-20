#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P/U/N labels + catalytic span + sigma* + compact H5 output
using:
  - RdRPcatch TSV results for positive labels and catalytic spans
  - hmmscan --domtblout against Pfam for negative labels

Labeling rules (requested behavior)
-----------------------------------
Let POS = query has any entry in --rdrpcatch-tsv.
Let NEG = query is covered by non-RdRP Pfam domains over more than --neg-qcov
          of its length, where coverage is the union of all qualifying non-RdRP
          domain spans (not necessarily one HMM).
Qualifying non-RdRP Pfam domains satisfy:
  - Pfam accession (version-stripped) NOT in --palm-pfams
  - domain i-Evalue <= --neg-ie

Positive hit criterion:
  - sequence has a RdRPcatch hit with best-hit e-value <= --pos-ie

Final label:
  - P if POS and not NEG
  - N if NEG and not POS
  - U otherwise (neither, or both)

Positive span:
  - For P rows, use the RdRPcatch aligned coordinates RdRp_from(AA)/RdRp_to(AA)
    as the catalytic-center span.
  - Span is written both as normalized S/E and as 0-based chunk-local aa coords
    [start,end), matching the older labeling TSV conventions.

Compatibility goals
-------------------
The output TSV/H5 keeps the same core schema as
scripts/label_from_rpsblast_and_hmmscan.py, including:
  chunk_id, label, use_span, L, S, E, span_start_aa, span_end_aa, truncated,
  conf, conf_e, conf_cov, conf_hsp, w_min, sigma_star, k,
  rps_pos_e_min, rps_pos_models, rps_pos_qcov_max, rps_pos_dcov_max,
  rps_neg_models, rps_neg_qcov_max

The legacy "rps_*" evidence columns are retained for downstream compatibility,
but populated with analogous RdRPcatch/Pfam information:
  - rps_pos_e_min      <- RdRPcatch best-hit e-value
  - rps_pos_models     <- RdRPcatch database:profile
  - rps_pos_qcov_max   <- RdRPcatch sequence coverage
  - rps_pos_dcov_max   <- RdRPcatch profile coverage
  - rps_neg_models     <- non-RdRP Pfam accession(s)
  - rps_neg_qcov_max   <- union non-RdRP Pfam coverage

Input lengths/order can come from either:
  - --fasta  (recommended for the current use case)
  - --h5     (embeddings HDF5, same style as older scripts)
Exactly one of these must be provided.
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
# Logging
# -------------------------
def setup_logging(level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("rdrpcatch_hmmscan_to_labels")


# -------------------------
# Generic I/O helpers
# -------------------------
def open_text_maybe_gzip(path: str):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(d, exist_ok=True)


# -------------------------
# Sequence lengths / order
# -------------------------
def parse_fasta_lengths(path: str) -> List[Tuple[str, int]]:
    """
    Read FASTA (optionally gzipped) and return [(seq_id, aa_len), ...] in file order.
    Uses the first whitespace-delimited token after '>' as the sequence ID.
    """
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


def load_lengths_and_seq_from_h5(h5_path: str) -> List[Tuple[str, int]]:
    """
    Read /items/<chunk_id> from an embeddings HDF5 and return [(chunk_id, L), ...].
    Prefer the 'seq' dataset if present, else use aa_len attr.
    """
    out: List[Tuple[str, int]] = []
    with h5py.File(h5_path, "r") as f:
        items = f.get("items")
        if items is None:
            raise RuntimeError("H5 missing /items")
        for cid in items.keys():
            g = items[cid]
            L: Optional[int] = None
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
            out.append((str(cid), int(L)))
    return out


# -------------------------
# ID / accession lists
# -------------------------
def read_id_list(path: str) -> set:
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                s.add(t)
    return s


def normalize_pfam_acc(acc: str) -> str:
    a = (acc or "").strip()
    if not a or a == "-":
        return ""
    a = a.split(".", 1)[0].strip()
    return a.upper()


def read_pfam_list(path: str) -> set:
    raw = read_id_list(path)
    out = set()
    for x in raw:
        nx = normalize_pfam_acc(x)
        if nx:
            out.add(nx)
    return out


# -------------------------
# RdRPcatch parser
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
    rdrp_from: Optional[int]    # 1-based inclusive
    rdrp_to: Optional[int]      # 1-based inclusive
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


def _coerce_int(s: str) -> Optional[int]:
    t = (s or "").strip()
    if not t or t == "-":
        return None
    try:
        return int(float(t))
    except Exception:
        return None


def _coerce_float(s: str) -> Optional[float]:
    t = (s or "").strip()
    if not t or t == "-":
        return None
    try:
        return float(t)
    except Exception:
        return None


def parse_rdrpcatch_tsv(path: str, logger: logging.Logger) -> Dict[str, RdRPCatchHit]:
    """
    Parse a RdRPcatch TSV (optionally gzipped).

    Some concatenated RdRPcatch result files may contain repeated header rows;
    those are skipped automatically.

    If duplicate sequence IDs occur, keep the best record by:
      1) smaller best-hit e-value
      2) higher best-hit bitscore
      3) higher best-hit sequence coverage
    """
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
            raise RuntimeError(
                f"RdRPcatch TSV missing required columns: {', '.join(missing)}"
            )

        for row in reader:
            # Skip repeated header rows in concatenated files
            if row.get("Sequence_name", "") == "Sequence_name":
                n_header_repeats += 1
                continue

            qid = (row.get("Sequence_name", "") or "").strip()
            if not qid:
                n_bad += 1
                continue

            rec = RdRPCatchHit(
                qid=qid,
                seq_len=_coerce_int(row.get("Sequence_length(AA)", "")),
                best_database=(row.get("Best_hit_Database", "") or "").strip(),
                best_profile_name=(row.get("Best_hit_profile_name", "") or "").strip(),
                best_profile_length=_coerce_int(row.get("Best_hit_profile_length", "")),
                best_evalue=_coerce_float(row.get("Best_hit_e-value", "")),
                best_bitscore=_coerce_float(row.get("Best_hit_bitscore", "")),
                rdrp_from=_coerce_int(row.get("RdRp_from(AA)", "")),
                rdrp_to=_coerce_int(row.get("RdRp_to(AA)", "")),
                best_profile_coverage=_coerce_float(row.get("Best_hit_profile_coverage", "")),
                best_sequence_coverage=_coerce_float(row.get("Best_hit_sequence_coverage", "")),
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
        n_rows,
        len(out),
        n_header_repeats,
        dup_counter,
        n_bad,
        path,
    )
    return out


def sanity_check_rdrpcatch_len_vs_input(
    rdrpcatch_hits: Dict[str, RdRPCatchHit],
    qlens: Dict[str, int],
    logger: logging.Logger,
    abs_tol: int = 3,
    rel_tol: float = 0.10,
    max_warnings: int = 50,
) -> int:
    n_warn = 0
    for qid, hit in rdrpcatch_hits.items():
        L = qlens.get(qid)
        qlen = hit.seq_len
        if L is None or qlen is None or qlen <= 0:
            continue
        thresh = max(abs_tol, int(rel_tol * max(1, L)))
        if abs(qlen - L) > thresh:
            if max_warnings == 0 or n_warn < max_warnings:
                logger.warning(
                    "length mismatch for %s: RdRPcatch=%d vs input L=%d (tol=%d)",
                    qid,
                    qlen,
                    L,
                    thresh,
                )
            n_warn += 1
    if max_warnings and n_warn > max_warnings:
        logger.warning("...and %d more RdRPcatch length mismatches suppressed", n_warn - max_warnings)
    return n_warn


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


def parse_hmmscan_domtbl(path: str, logger: logging.Logger) -> Dict[str, List[HmmerDomHit]]:
    out: Dict[str, List[HmmerDomHit]] = defaultdict(list)
    n = 0
    n_bad = 0

    with open_text_maybe_gzip(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            if ln.lstrip().startswith("#"):
                continue

            line = ln.strip("\n")
            if line.startswith("{") or line.startswith("}") or line.startswith("\\"):
                continue
            line = line.replace("\\", " ")

            parts = line.split()
            if len(parts) < 22:
                n_bad += 1
                continue

            try:
                target_name = parts[0]
                target_acc = parts[1]
                qid = parts[3]
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
                if acc_raw not in ("-", "NA", "nan", "NaN", ""):
                    try:
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


def sanity_check_domtbl_qlen_vs_input(
    dom_hits: Dict[str, List[HmmerDomHit]],
    qlens: Dict[str, int],
    logger: logging.Logger,
    abs_tol: int = 3,
    rel_tol: float = 0.10,
    max_warnings: int = 50,
) -> int:
    n_warn = 0
    for qid, L in qlens.items():
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
                logger.warning(
                    "qlen mismatch for %s: hmmscan=%d vs input L=%d (tol=%d)",
                    qid,
                    qlen_mode,
                    L,
                    thresh,
                )
            n_warn += 1
    if max_warnings and n_warn > max_warnings:
        logger.warning("...and %d more hmmscan qlen mismatches suppressed", n_warn - max_warnings)
    return n_warn


# -------------------------
# Coordinates / coverage
# -------------------------
def to_half_open_1based(qs: int, qe: int) -> Tuple[int, int]:
    """Convert 1-based inclusive coordinates to 0-based half-open [s,e)."""
    s = min(qs, qe) - 1
    e = max(qs, qe)
    return max(0, s), max(0, e)


def merge_intervals(iv: Sequence[Tuple[int, int]], max_gap: int = 0) -> List[Tuple[int, int]]:
    if not iv:
        return []
    arr = sorted(iv, key=lambda x: (x[0], x[1]))
    out = [arr[0]]
    for s, e in arr[1:]:
        ps, pe = out[-1]
        if s <= pe + max_gap:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out


def union_len(iv: Sequence[Tuple[int, int]]) -> int:
    return sum(max(0, e - s) for s, e in merge_intervals(iv, max_gap=0))


def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _safe_log10_evalue(ev: Optional[float]) -> float:
    if ev is None or (isinstance(ev, float) and math.isnan(ev)):
        return 0.0
    if ev <= 0.0:
        ev = 1e-300
    return -math.log10(ev)


def span_from_dom(hit: HmmerDomHit, span_source: str) -> Tuple[int, int]:
    if span_source == "ali":
        lo = min(hit.ali_from, hit.ali_to)
        hi = max(hit.ali_from, hit.ali_to)
    else:
        lo = min(hit.env_from, hit.env_to)
        hi = max(hit.env_from, hit.env_to)
    return max(0, lo - 1), max(0, hi)


# -------------------------
# Confidence
# -------------------------
def conf_from_rdrpcatch(
    best_evalue: Optional[float],
    seq_cov: float,
    profile_cov: Optional[float],
    truncated: bool,
    pos_e_max: float,
    pos_e_sat: float,
) -> Tuple[float, float, float, float]:
    """
    Confidence components from RdRPcatch best-hit e-value + sequence coverage +
    profile coverage. This does not affect labeling; it only fills the legacy
    confidence columns for downstream compatibility.
    """
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
# Output row
# -------------------------
LABEL_TO_U8 = {"N": 0, "U": 1, "P": 2}


@dataclass
class LabelRow:
    chunk_id: str
    label: str
    use_span: int
    L: int
    S: float
    E: float
    span_start_aa: str
    span_end_aa: str
    truncated: int
    conf: float
    conf_e: float
    conf_cov: float
    conf_hsp: float
    w_min: float
    sigma_star: float
    k: float

    # legacy compatibility columns
    rps_pos_e_min: str
    rps_pos_models: str
    rps_pos_qcov_max: float
    rps_pos_dcov_max: float
    rps_neg_models: str
    rps_neg_qcov_max: float


@dataclass
class OptionalSourceCols:
    rdrpcatch_best_database: str
    rdrpcatch_best_profile_name: str
    rdrpcatch_best_profile_length: str
    rdrpcatch_best_evalue: str
    rdrpcatch_best_bitscore: str
    rdrpcatch_span_from_1based: str
    rdrpcatch_span_to_1based: str
    rdrpcatch_profile_coverage: str
    rdrpcatch_sequence_coverage: str
    pfam_neg_domain_count: str
    pfam_neg_union_cov: str
    pfam_neg_models: str
    pfam_neg_span_source: str


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
        return ""
    return f"{ev:.3g}"


def _join_models(models: Iterable[str], limit: int = 12) -> str:
    xs = [m for m in models if m]
    if not xs:
        return ""
    if len(xs) <= limit:
        return ",".join(xs)
    return ",".join(xs[:limit]) + f",...(+{len(xs)-limit})"


# -------------------------
# Core labeling
# -------------------------
def decide_label_for_query(
    qid: str,
    L: int,
    rdrpcatch_hit: Optional[RdRPCatchHit],
    hmmscan_hits: Sequence[HmmerDomHit],
    palm_pfams: set,
    pos_ie: float,
    neg_ie: float,
    neg_qcov: float,
    neg_qcov_inclusive: bool,
    pfam_span_source: str,
    sigma_k: float,
    conf_e_max: float,
    conf_e_sat: float,
) -> Tuple[LabelRow, OptionalSourceCols]:
    pos = (
        rdrpcatch_hit is not None
        and rdrpcatch_hit.best_evalue is not None
        and rdrpcatch_hit.best_evalue <= pos_ie
    )

    # Negative evidence = union coverage across qualifying non-RdRP Pfam domains.
    neg_iv: List[Tuple[int, int]] = []
    neg_models: List[str] = []
    neg_domain_hits: List[HmmerDomHit] = []
    for h in hmmscan_hits:
        if h.target_acc_base in palm_pfams:
            continue
        if h.i_evalue > neg_ie:
            continue
        s0, e0 = span_from_dom(h, span_source=pfam_span_source)
        s0 = max(0, min(s0, L))
        e0 = max(0, min(e0, L))
        if e0 <= s0:
            continue
        neg_iv.append((s0, e0))
        neg_domain_hits.append(h)
        model = h.target_acc_base or h.target_name
        if model:
            neg_models.append(model)

    neg_cov = (union_len(neg_iv) / float(L)) if (L > 0 and neg_iv) else 0.0
    neg = (neg_cov >= neg_qcov) if neg_qcov_inclusive else (neg_cov > neg_qcov)

    if pos and not neg:
        label = "P"
    elif neg and not pos:
        label = "N"
    else:
        label = "U"

    # Legacy evidence-column mapping for compatibility.
    if rdrpcatch_hit is not None:
        pos_model = ":".join(
            x for x in [rdrpcatch_hit.best_database, rdrpcatch_hit.best_profile_name] if x
        )
        rps_pos_e_min = _fmt_evalue(rdrpcatch_hit.best_evalue)
        rps_pos_models = pos_model
        rps_pos_qcov_max = (
            float(rdrpcatch_hit.best_sequence_coverage)
            if rdrpcatch_hit.best_sequence_coverage is not None
            else float("nan")
        )
        rps_pos_dcov_max = (
            float(rdrpcatch_hit.best_profile_coverage)
            if rdrpcatch_hit.best_profile_coverage is not None
            else float("nan")
        )
    else:
        rps_pos_e_min = ""
        rps_pos_models = ""
        rps_pos_qcov_max = float("nan")
        rps_pos_dcov_max = float("nan")

    # Negative compatibility fields
    model_counter = Counter(neg_models)
    sorted_models = [m for m, _ in model_counter.most_common()]
    rps_neg_models = _join_models(sorted_models)
    rps_neg_qcov_max = float(neg_cov) if neg_iv else float("nan")

    # Defaults
    use_span = 0
    S = float("nan")
    E = float("nan")
    span_s = ""
    span_e = ""
    truncated = 0
    conf = conf_e = conf_cov = conf_hsp = 0.0
    w_min = max(70.0 / max(1.0, float(L)), 0.02)
    sigma_star = float("nan")

    # Positive span from RdRPcatch aligned coordinates.
    if label == "P" and rdrpcatch_hit is not None:
        rs = rdrpcatch_hit.rdrp_from
        re = rdrpcatch_hit.rdrp_to
        if rs is not None and re is not None:
            s0, e0 = to_half_open_1based(rs, re)
            s0 = max(0, min(s0, L))
            e0 = max(0, min(e0, L))
            if e0 > s0:
                use_span = 1
                span_s = str(int(s0))
                span_e = str(int(e0))
                S = float(s0) / float(L) if L > 0 else float("nan")
                E = float(e0) / float(L) if L > 0 else float("nan")
                trunc_flag = (s0 == 0) or (e0 == L)
                truncated = 1 if trunc_flag else 0

                seq_cov = (
                    float(rdrpcatch_hit.best_sequence_coverage)
                    if rdrpcatch_hit.best_sequence_coverage is not None
                    else ((e0 - s0) / float(L) if L > 0 else 0.0)
                )
                conf, conf_e, conf_cov, conf_hsp = conf_from_rdrpcatch(
                    best_evalue=rdrpcatch_hit.best_evalue,
                    seq_cov=float(seq_cov),
                    profile_cov=rdrpcatch_hit.best_profile_coverage,
                    truncated=bool(truncated),
                    pos_e_max=conf_e_max,
                    pos_e_sat=conf_e_sat,
                )

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

    source_cols = OptionalSourceCols(
        rdrpcatch_best_database=(rdrpcatch_hit.best_database if rdrpcatch_hit is not None else ""),
        rdrpcatch_best_profile_name=(rdrpcatch_hit.best_profile_name if rdrpcatch_hit is not None else ""),
        rdrpcatch_best_profile_length=(
            str(rdrpcatch_hit.best_profile_length)
            if rdrpcatch_hit is not None and rdrpcatch_hit.best_profile_length is not None
            else ""
        ),
        rdrpcatch_best_evalue=(
            _fmt_evalue(rdrpcatch_hit.best_evalue)
            if rdrpcatch_hit is not None
            else ""
        ),
        rdrpcatch_best_bitscore=(
            _fmt_float(rdrpcatch_hit.best_bitscore, 3)
            if rdrpcatch_hit is not None and rdrpcatch_hit.best_bitscore is not None
            else ""
        ),
        rdrpcatch_span_from_1based=(
            str(rdrpcatch_hit.rdrp_from)
            if rdrpcatch_hit is not None and rdrpcatch_hit.rdrp_from is not None
            else ""
        ),
        rdrpcatch_span_to_1based=(
            str(rdrpcatch_hit.rdrp_to)
            if rdrpcatch_hit is not None and rdrpcatch_hit.rdrp_to is not None
            else ""
        ),
        rdrpcatch_profile_coverage=(
            _fmt_float(rdrpcatch_hit.best_profile_coverage, 4)
            if rdrpcatch_hit is not None and rdrpcatch_hit.best_profile_coverage is not None
            else ""
        ),
        rdrpcatch_sequence_coverage=(
            _fmt_float(rdrpcatch_hit.best_sequence_coverage, 4)
            if rdrpcatch_hit is not None and rdrpcatch_hit.best_sequence_coverage is not None
            else ""
        ),
        pfam_neg_domain_count=(str(len(neg_domain_hits)) if neg_domain_hits else "0"),
        pfam_neg_union_cov=(_fmt_float(neg_cov, 4) if neg_domain_hits else ""),
        pfam_neg_models=_join_models(sorted_models),
        pfam_neg_span_source=pfam_span_source if neg_domain_hits else "",
    )
    return row, source_cols


# -------------------------
# Compact H5 writer
# -------------------------
def write_compact_labels_h5(
    out_h5: str,
    rows: List[LabelRow],
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
    N = len(rows)

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

    rps_pos_e_min = np.array([r.rps_pos_e_min for r in rows], dtype=vstr)
    rps_pos_models = np.array([r.rps_pos_models for r in rows], dtype=vstr)
    rps_neg_models = np.array([r.rps_neg_models for r in rows], dtype=vstr)
    span_start_aa = np.array([r.span_start_aa for r in rows], dtype=vstr)
    span_end_aa = np.array([r.span_end_aa for r in rows], dtype=vstr)

    with h5py.File(out_h5, "w", libver="latest") as f:
        g = f.create_group("labels")

        g.create_dataset("chunk_id", data=chunk_id)
        g.create_dataset("rps_pos_e_min", data=rps_pos_e_min)
        g.create_dataset("rps_pos_models", data=rps_pos_models)
        g.create_dataset("rps_neg_models", data=rps_neg_models)
        g.create_dataset("span_start_aa", data=span_start_aa)
        g.create_dataset("span_end_aa", data=span_end_aa)

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

        g.attrs["label_map"] = json.dumps({"N": 0, "U": 1, "P": 2})

        f.attrs["schema_version"] = schema_version
        f.attrs["thresholds"] = json.dumps(thresholds_meta)
        f.attrs["source"] = "rdrpcatch+pfam"
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
        description=(
            "Create P/U/N label TSV + compact labels H5 from RdRPcatch positives "
            "and Pfam hmmscan negatives."
        )
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fasta", help="input sequence FASTA (recommended for this workflow; .gz ok)")
    src.add_argument("--h5", help="embeddings HDF5 used only for lengths/order (legacy-compatible alternative)")

    ap.add_argument("--rdrpcatch-tsv", required=True, help="RdRPcatch results TSV or TSV.gz")
    ap.add_argument("--hmmscan-domtbl", required=True, help="hmmscan --domtblout against Pfam (.gz ok)")
    ap.add_argument("--palm-pfams", required=True, help="Pfam accessions considered RdRP Pfams (one per line)")
    ap.add_argument("--out", required=True, help="output labels TSV")

    ap.add_argument("--out-h5", default=None, help="compact labels H5 path (default: derive from --out)")
    ap.add_argument("--no-h5", action="store_true", help="do not write compact labels H5")
    ap.add_argument("--overwrite-h5", action="store_true", help="overwrite existing --out-h5 if it exists")
    ap.add_argument("--schema-version", default="1.0")
    ap.add_argument("--log-level", default="INFO")

    # Positive criterion from RdRPcatch.
    ap.add_argument(
        "--pos-ie",
        type=float,
        default=1e-5,
        help="max RdRPcatch best-hit e-value for a sequence to be considered positive",
    )

    # Negative criteria from Pfam coverage.
    ap.add_argument(
        "--neg-ie",
        type=float,
        default=1e-5,
        help="max hmmscan domain i-Evalue for a non-RdRP Pfam domain to contribute to negative coverage",
    )
    ap.add_argument(
        "--neg-qcov",
        type=float,
        default=0.80,
        help="negative if union non-RdRP Pfam coverage is > this fraction of sequence length (default 0.80)",
    )
    ap.add_argument(
        "--neg-qcov-inclusive",
        action="store_true",
        help="use >= instead of > for the negative coverage threshold",
    )
    ap.add_argument(
        "--pfam-span-source",
        choices=["env", "ali"],
        default="env",
        help="Pfam coords used when computing negative coverage (env or ali). Default: env",
    )

    # Confidence-only knobs for positive rows.
    ap.add_argument(
        "--conf-e-max",
        type=float,
        default=1e-3,
        help="RdRPcatch e-value at which confidence starts above zero (used only for conf columns)",
    )
    ap.add_argument(
        "--conf-e-sat",
        type=float,
        default=1e-30,
        help="RdRPcatch e-value at which confidence saturates (used only for conf columns)",
    )
    ap.add_argument("--sigma-k", type=float, default=2.0, help="k in sigma* = max(w/(2*k), w_min/2)")

    # Optional extra source columns (TSV only).
    ap.add_argument(
        "--append-source-cols",
        action="store_true",
        help="append RdRPcatch/Pfam source-specific columns after the legacy-compatible core columns",
    )

    # Length sanity checks.
    ap.add_argument("--qlen-abs-tol", type=int, default=3)
    ap.add_argument("--qlen-rel-tol", type=float, default=0.10)
    ap.add_argument("--qlen-max-warn", type=int, default=50)

    args = ap.parse_args()
    log = setup_logging(args.log_level)

    if args.pos_ie <= 0:
        log.error("--pos-ie must be > 0")
        sys.exit(2)
    if args.neg_ie <= 0:
        log.error("--neg-ie must be > 0")
        sys.exit(2)
    if not (0.0 <= args.neg_qcov <= 1.0):
        log.error("--neg-qcov must be in [0,1]")
        sys.exit(2)
    if args.conf_e_max <= 0 or args.conf_e_sat <= 0:
        log.error("--conf-e-max and --conf-e-sat must be > 0")
        sys.exit(2)
    if args.conf_e_sat > args.conf_e_max:
        log.warning(
            "--conf-e-sat (%.2g) > --conf-e-max (%.2g); adjusting saturation to conf-e-max",
            args.conf_e_sat,
            args.conf_e_max,
        )
        args.conf_e_sat = args.conf_e_max

    out_h5 = None if args.no_h5 else (args.out_h5 or default_out_h5(args.out))

    # Input lengths + order
    if args.fasta:
        ordered_lengths = parse_fasta_lengths(args.fasta)
        input_source = f"fasta:{args.fasta}"
    else:
        ordered_lengths = load_lengths_and_seq_from_h5(args.h5)
        input_source = f"h5:{args.h5}"
    qlens = dict(ordered_lengths)
    log.info("Loaded %d input sequences from %s", len(ordered_lengths), input_source)

    rdrpcatch_hits = parse_rdrpcatch_tsv(args.rdrpcatch_tsv, logger=log)
    domtbl = parse_hmmscan_domtbl(args.hmmscan_domtbl, logger=log)
    palm_pfams = read_pfam_list(args.palm_pfams)
    if not palm_pfams:
        log.error("Empty RdRP Pfam list: %s", args.palm_pfams)
        sys.exit(2)

    log.info("Sanity-checking RdRPcatch sequence lengths vs input lengths...")
    mismatches_rdrpcatch = sanity_check_rdrpcatch_len_vs_input(
        rdrpcatch_hits=rdrpcatch_hits,
        qlens=qlens,
        logger=log,
        abs_tol=args.qlen_abs_tol,
        rel_tol=args.qlen_rel_tol,
        max_warnings=args.qlen_max_warn,
    )
    if mismatches_rdrpcatch:
        log.warning("Found %d RdRPcatch length mismatches (see warnings above).", mismatches_rdrpcatch)

    log.info("Sanity-checking hmmscan qlen vs input lengths...")
    mismatches_hmm = sanity_check_domtbl_qlen_vs_input(
        dom_hits=domtbl,
        qlens=qlens,
        logger=log,
        abs_tol=args.qlen_abs_tol,
        rel_tol=args.qlen_rel_tol,
        max_warnings=args.qlen_max_warn,
    )
    if mismatches_hmm:
        log.warning("Found %d hmmscan qlen mismatches (see warnings above).", mismatches_hmm)

    extra_rdrpcatch = set(rdrpcatch_hits) - set(qlens)
    extra_domtbl = set(domtbl) - set(qlens)
    if extra_rdrpcatch:
        log.warning("%d RdRPcatch sequence IDs were not found in the input sequence set and will be ignored", len(extra_rdrpcatch))
    if extra_domtbl:
        log.warning("%d hmmscan query IDs were not found in the input sequence set and will be ignored", len(extra_domtbl))

    base_header = [
        "chunk_id", "label", "use_span", "L", "S", "E",
        "span_start_aa", "span_end_aa", "truncated",
        "conf", "conf_e", "conf_cov", "conf_hsp",
        "w_min", "sigma_star", "k",
        "rps_pos_e_min", "rps_pos_models", "rps_pos_qcov_max", "rps_pos_dcov_max",
        "rps_neg_models", "rps_neg_qcov_max",
    ]
    source_header = [
        "rdrpcatch_best_database",
        "rdrpcatch_best_profile_name",
        "rdrpcatch_best_profile_length",
        "rdrpcatch_best_evalue",
        "rdrpcatch_best_bitscore",
        "rdrpcatch_span_from_1based",
        "rdrpcatch_span_to_1based",
        "rdrpcatch_profile_coverage",
        "rdrpcatch_sequence_coverage",
        "pfam_neg_domain_count",
        "pfam_neg_union_cov",
        "pfam_neg_models",
        "pfam_neg_span_source",
    ]
    header = list(base_header)
    if args.append_source_cols:
        header.extend(source_header)

    rows: List[LabelRow] = []
    ensure_parent_dir(args.out)

    n = nP = nU = nN = nSpan = 0
    with open(args.out, "w", encoding="utf-8", newline="") as out:
        out.write("\t".join(header) + "\n")

        for qid, L in ordered_lengths:
            row_obj, src_obj = decide_label_for_query(
                qid=qid,
                L=L,
                rdrpcatch_hit=rdrpcatch_hits.get(qid),
                hmmscan_hits=domtbl.get(qid, []),
                palm_pfams=palm_pfams,
                pos_ie=args.pos_ie,
                neg_ie=args.neg_ie,
                neg_qcov=args.neg_qcov,
                neg_qcov_inclusive=args.neg_qcov_inclusive,
                pfam_span_source=args.pfam_span_source,
                sigma_k=args.sigma_k,
                conf_e_max=args.conf_e_max,
                conf_e_sat=args.conf_e_sat,
            )
            rows.append(row_obj)

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
                **src_obj.__dict__,
            }
            out.write("\t".join(rec.get(k, "") for k in header) + "\n")

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
            "source": "rdrpcatch+pfam",
            "input_source": input_source,
            "pos_ie": args.pos_ie,
            "neg_ie": args.neg_ie,
            "neg_qcov": args.neg_qcov,
            "neg_qcov_op": ">=" if args.neg_qcov_inclusive else ">",
            "pfam_span_source": args.pfam_span_source,
            "conf_e_max": args.conf_e_max,
            "conf_e_sat": args.conf_e_sat,
            "sigma_k": args.sigma_k,
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

