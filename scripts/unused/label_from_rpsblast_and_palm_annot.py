#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P/U/N labels + span + sigma* + confidence

This variant uses palm_annot TSV for *Positive* labeling only.

- Positive (P): palm_annot reports rdrp score >= --palm-score-min for the query.
- Negative (N) and Unlabeled (U) rules are unchanged from label_embed_h5.py:
    * neg evidence comes from rpsblast hits to NON-RdRP CDD IDs (qcov >= --neg-qcov, e <= --neg-e)
    * N is only assigned when there is neg evidence and there is no "loose RdRP" rpsblast hit
      (loose RdRP: any RdRP CDD ID hit with e <= --loose-rdrp-e)
    * if both pos and neg -> U (conflicting evidence)

Span for P:
- Prefer palm_annot ext_lo/ext_hi (1-based inclusive) if present, else pp_lo/pp_hi.
- Converted to 0-based half-open [start, end) in amino-acid coordinates.

Confidence for P:
- Derived from palm_annot rdrp score, span width, and (optional) gate_gdd_prob:
    conf_e   = normalized rdrp score
    conf_cov = normalized span width / L (saturates at 0.6)
    conf_hsp = gate_gdd_prob (if available), else 0
    conf     = 0.6*conf_e + 0.3*conf_cov + 0.1*conf_hsp
    conf *= 0.5 if truncated (span touches N- or C- terminus)

Output schema matches label_embed_h5.py by default; optionally append palm_annot fields.

NOTE: This script DOES NOT modify the input HDF5. It reads it to get L (sequence length) only.
"""
import sys
import argparse
import logging
import math
import gzip
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional

import h5py


# -------------------------
# Logging
# -------------------------
def setup_logging(level: str = "INFO") -> logging.Logger:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("palm_annot_to_labels")


# -------------------------
# H5 lengths & sequences
# -------------------------
def load_lengths_and_seq(h5_path: str) -> Dict[str, Tuple[int, Optional[str]]]:
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
# Read allowlist
# -------------------------
def read_id_list(path: str) -> set:
    s = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                s.add(t)
    return s


# -------------------------
# RPS parser (unchanged)
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
    Parse outfmt 6 with columns:
      qseqid sacc evalue bitscore qstart qend sstart send length qlen slen stitle qcovs qcovhsp
    """
    hits: Dict[str, List[Hit]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            while len(p) < 14:
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
                qlen = int(p[9]) if p[9] else None
                slen = int(p[10]) if p[10] else None
            except Exception:
                continue
            title = p[11]
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
# Intervals & coverage (unchanged)
# -------------------------
def to_half_open(qs: int, qe: int) -> Tuple[int, int]:
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
# Span fallback (unchanged, used only if palm span missing)
# -------------------------
def choose_primary_span(rdrp_hsps: List[Hit], L: int, max_gap: int = 25) -> Tuple[int, int, bool]:
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
# palm_annot parser
# -------------------------
@dataclass(frozen=True)
class PalmAnnot:
    score: float
    ext_lo: Optional[int] = None  # 1-based inclusive
    ext_hi: Optional[int] = None  # 1-based inclusive
    pp_lo: Optional[int] = None   # 1-based inclusive
    pp_hi: Optional[int] = None   # 1-based inclusive
    gate_gdd_prob: Optional[float] = None
    gate_prob: Optional[float] = None
    gdd_prob: Optional[float] = None
    gate: Optional[str] = None
    gdd: Optional[str] = None


def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_palm_annot_tsv(path: str, logger: logging.Logger) -> Dict[str, PalmAnnot]:
    """
    palm_annot output is tab-delimited:
      <qid>  key=value  key=value  ...

    We keep the best record per qid (highest rdrp score).
    """
    out: Dict[str, PalmAnnot] = {}
    n = 0
    n_bad = 0

    with _open_maybe_gzip(path) as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            parts = ln.rstrip("\n").split("\t")
            if not parts:
                continue
            qid = parts[0]

            kv = {}
            for tok in parts[1:]:
                if "=" not in tok:
                    continue
                k, v = tok.split("=", 1)
                kv[k] = v

            if "rdrp" not in kv:
                n_bad += 1
                continue

            try:
                score = float(kv["rdrp"])
            except Exception:
                n_bad += 1
                continue

            def as_int(key: str) -> Optional[int]:
                if key not in kv or kv[key] == "":
                    return None
                try:
                    return int(float(kv[key]))
                except Exception:
                    return None

            def as_float(key: str) -> Optional[float]:
                if key not in kv or kv[key] == "":
                    return None
                try:
                    return float(kv[key])
                except Exception:
                    return None

            rec = PalmAnnot(
                score=score,
                ext_lo=as_int("ext_lo"),
                ext_hi=as_int("ext_hi"),
                pp_lo=as_int("pp_lo"),
                pp_hi=as_int("pp_hi"),
                gate_gdd_prob=as_float("gate_gdd_prob"),
                gate_prob=as_float("gate_prob"),
                gdd_prob=as_float("gdd_prob"),
                gate=kv.get("gate"),
                gdd=kv.get("gdd"),
            )

            prev = out.get(qid)
            if prev is None or rec.score > prev.score:
                out[qid] = rec

            n += 1

    logger.info("Loaded palm_annot rows=%d (bad=%d), unique_qids=%d from %s", n, n_bad, len(out), path)
    return out


# -------------------------
# Confidence utilities
# -------------------------
def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def conf_from_palm(
    score: float,
    score_min: float,
    score_sat: float,
    cov: float,
    gate_gdd_prob: Optional[float],
    truncated: bool,
) -> Tuple[float, float, float, float]:
    if score_sat <= score_min:
        score_sat = score_min + 1.0
    conf_e = clip01((score - score_min) / (score_sat - score_min))
    conf_cov = clip01(cov / 0.60)  # saturate at 60% (kept consistent with original convention)
    conf_hsp = clip01(gate_gdd_prob) if gate_gdd_prob is not None else 0.0
    conf = 0.6 * conf_e + 0.3 * conf_cov + 0.1 * conf_hsp
    if truncated:
        conf *= 0.5
    return conf, conf_e, conf_cov, conf_hsp


# -------------------------
# Labeling
# -------------------------
def decide_labels_for_query(
    qid: str,
    L: int,
    hits: List[Hit],
    rdrp_ids: set,
    palm: Optional[PalmAnnot],
    palm_score_min: float,
    palm_score_sat: float,
    thr_neg_e: float,
    thr_neg_qcov: float,
    thr_loose_rdrp_e: float,
    max_gap_for_span: int,
    sigma_k: float,
) -> Dict[str, str]:
    # group rps hits by subject
    by_sid: Dict[str, List[Hit]] = defaultdict(list)
    for h in hits:
        by_sid[h.sid].append(h)

    subs = []
    for sid, hsps in by_sid.items():
        best_e = min(h.e for h in hsps)
        qc = qcov_of(hsps, L)
        dc = dcov_of(hsps)
        subs.append((sid, best_e, qc, dc, hsps))

    # Positive from palm_annot only
    pos = (palm is not None) and (palm.score >= palm_score_min)

    # Unchanged: "loose RdRP" from rps hits to RdRP IDs
    loose_rdrp = any((sid in rdrp_ids and e <= thr_loose_rdrp_e) for sid, e, _, _, _ in subs)

    # Unchanged: negative evidence from strong non-RdRP hits
    neg_subs = [(sid, e, qc, dc, hsps) for sid, e, qc, dc, hsps in subs
                if (sid not in rdrp_ids) and e <= thr_neg_e and qc >= thr_neg_qcov]
    neg = len(neg_subs) > 0

    # Unchanged label resolution
    if pos and neg:
        label = "U"
    elif pos:
        label = "P"
    elif (not loose_rdrp) and neg:
        label = "N"
    else:
        label = "U"

    # Defaults
    use_span = 0
    S = E = ""
    span_s = span_e = ""
    truncated = False
    conf = conf_e = conf_cov = conf_hsp = 0.0
    w_min = max(70.0 / max(1.0, L), 0.02)  # normalized

    # For P: span + confidence prefer palm_annot; fallback to rps RdRP span only if palm span missing
    if label == "P":
        s: Optional[int] = None
        e: Optional[int] = None

        # Palm span preference: ext_lo/ext_hi else pp_lo/pp_hi
        if palm is not None:
            lo = palm.ext_lo if (palm.ext_lo is not None and palm.ext_hi is not None) else None
            hi = palm.ext_hi if (palm.ext_lo is not None and palm.ext_hi is not None) else None
            if lo is None or hi is None:
                lo = palm.pp_lo if (palm.pp_lo is not None and palm.pp_hi is not None) else None
                hi = palm.pp_hi if (palm.pp_lo is not None and palm.pp_hi is not None) else None

            if lo is not None and hi is not None and hi > 0:
                # Convert 1-based inclusive [lo,hi] -> 0-based half-open [s0,e0)
                s0 = max(0, int(lo) - 1)
                e0 = int(hi)  # inclusive hi -> exclusive end
                s0 = min(s0, L)
                e0 = min(max(e0, 0), L)
                if e0 > s0:
                    s, e = s0, e0

        # Fallback: if palm span missing, reuse rps-based span selection at loose threshold
        if s is None or e is None:
            rdrp_hsps = [h for sid, ev, _, _, hsps in subs if (sid in rdrp_ids and ev <= thr_loose_rdrp_e) for h in hsps]
            if rdrp_hsps:
                s0, e0, _ = choose_primary_span(rdrp_hsps, L, max_gap=max_gap_for_span)
                if e0 > s0:
                    s, e = s0, e0

        if s is not None and e is not None and e > s:
            truncated = (s == 0) or (e == L)
            S = f"{s / L:.6f}"
            E = f"{e / L:.6f}"
            span_s = str(s)
            span_e = str(e)
            use_span = 1

        # confidence from palm_annot score (even if span missing)
        if palm is not None:
            cov = ((e - s) / L) if (use_span == 1 and s is not None and e is not None and L > 0) else 0.0
            conf, conf_e, conf_cov, conf_hsp = conf_from_palm(
                score=float(palm.score),
                score_min=palm_score_min,
                score_sat=palm_score_sat,
                cov=float(cov),
                gate_gdd_prob=palm.gate_gdd_prob,
                truncated=truncated if use_span else False,
            )

    sigma_star = ""
    if use_span == 1 and S and E:
        w = float(E) - float(S)
        sigma_star = f"{max(w / (2.0 * sigma_k), w_min / 2.0):.6f}"

    rec = {
        "chunk_id": qid,
        "label": label,
        "use_span": str(use_span),
        "L": str(L),
        "S": S,
        "E": E,
        "span_start_aa": span_s,
        "span_end_aa": span_e,
        "truncated": "1" if (use_span and truncated) else "0",
        "conf": f"{conf:.4f}",
        "conf_e": f"{conf_e:.4f}",
        "conf_cov": f"{conf_cov:.4f}",
        "conf_hsp": f"{conf_hsp:.4f}",
        "w_min": f"{w_min:.6f}",
        "sigma_star": sigma_star,
        "k": f"{sigma_k:.3f}",
    }
    return rec


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="P/U/N + span + sigma* using palm_annot for positives, rpsblast for neg/U rules"
    )
    ap.add_argument("--h5", required=True, help="embeddings HDF5 (read-only)")
    ap.add_argument("--rps", required=True, help="rpsblast outfmt 6 TSV (for neg/loose rules)")
    ap.add_argument("--rdrp-ids", required=True, help="file with RdRP CDD IDs (one per line)")
    ap.add_argument("--palm-annot", required=True, help="palm_annot output TSV (can be .gz)")
    ap.add_argument("--out", required=True, help="labels_with_spans.tsv")
    ap.add_argument("--log-level", default="INFO")

    # palm_annot thresholding
    ap.add_argument("--palm-score-min", type=float, default=50.0, help="minimum palm_annot rdrp score to call Positive")
    ap.add_argument("--palm-score-sat", type=float, default=None, help="score at which confidence saturates (default=min+10)")

    # unchanged thresholds (neg/U rules)
    ap.add_argument("--neg-e", type=float, default=1e-5)
    ap.add_argument("--neg-qcov", type=float, default=0.80)
    ap.add_argument("--loose-rdrp-e", type=float, default=1e-2)

    # span fallback
    ap.add_argument("--span-max-gap", type=int, default=25)

    # sigma target
    ap.add_argument("--sigma-k", type=float, default=2.0, help="k in sigma* = max(w/(2k), w_min/2)")

    # qlen sanity
    ap.add_argument("--qlen-abs-tol", type=int, default=3)
    ap.add_argument("--qlen-rel-tol", type=float, default=0.10)
    ap.add_argument("--qlen-max-warn", type=int, default=50)

    # optional debug columns
    ap.add_argument(
        "--append-palm-cols",
        action="store_true",
        help="Append palm_annot fields (palm_rdrp, palm_ext_lo, palm_ext_hi, palm_pp_lo, palm_pp_hi) to output TSV.",
    )

    args = ap.parse_args()
    log = setup_logging(args.log_level)

    if args.palm_score_sat is None:
        args.palm_score_sat = args.palm_score_min + 10.0

    qlens = load_lengths_and_seq(args.h5)
    rps = parse_rps_tsv(args.rps)

    log.info("Sanity-checking qlen (rps) vs L (HDF5)...")
    mismatches = sanity_check_qlen_vs_h5L(
        rps_hits=rps,
        qlens=qlens,
        logger=log,
        abs_tol=args.qlen_abs_tol,
        rel_tol=args.qlen_rel_tol,
        max_warnings=args.qlen_max_warn,
    )
    if mismatches:
        log.warning("Found %d qlen≠L mismatches (see warnings above).", mismatches)

    rdrp_ids = read_id_list(args.rdrp_ids)
    if not rdrp_ids:
        log.error("Empty RdRP ID list: %s", args.rdrp_ids)
        sys.exit(2)

    palm_map = parse_palm_annot_tsv(args.palm_annot, logger=log)

    base_header = [
        "chunk_id", "label", "use_span", "L", "S", "E",
        "span_start_aa", "span_end_aa", "truncated",
        "conf", "conf_e", "conf_cov", "conf_hsp", "w_min", "sigma_star", "k",
    ]
    header = list(base_header)
    if args.append_palm_cols:
        header += ["palm_rdrp", "palm_ext_lo", "palm_ext_hi", "palm_pp_lo", "palm_pp_hi"]

    n = nP = nU = nN = nSpan = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("\t".join(header) + "\n")

        for qid, (L, _) in qlens.items():
            palm = palm_map.get(qid)
            rec = decide_labels_for_query(
                qid=qid,
                L=L,
                hits=rps.get(qid, []),
                rdrp_ids=rdrp_ids,
                palm=palm,
                palm_score_min=args.palm_score_min,
                palm_score_sat=args.palm_score_sat,
                thr_neg_e=args.neg_e,
                thr_neg_qcov=args.neg_qcov,
                thr_loose_rdrp_e=args.loose_rdrp_e,
                max_gap_for_span=args.span_max_gap,
                sigma_k=args.sigma_k,
            )

            row = [rec.get(k, "") for k in base_header]

            if args.append_palm_cols:
                if palm is None:
                    row += ["", "", "", "", ""]
                else:
                    row += [
                        f"{palm.score:.3f}",
                        "" if palm.ext_lo is None else str(palm.ext_lo),
                        "" if palm.ext_hi is None else str(palm.ext_hi),
                        "" if palm.pp_lo is None else str(palm.pp_lo),
                        "" if palm.pp_hi is None else str(palm.pp_hi),
                    ]

            out.write("\t".join(row) + "\n")

            n += 1
            if rec["label"] == "P":
                nP += 1
                if rec["use_span"] == "1":
                    nSpan += 1
            elif rec["label"] == "N":
                nN += 1
            else:
                nU += 1

    log.info("Done. Wrote %d rows → %s [P=%d (span=%d), U=%d, N=%d]", n, args.out, nP, nSpan, nU, nN)


if __name__ == "__main__":
    main()
