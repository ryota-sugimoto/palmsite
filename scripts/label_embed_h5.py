#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P/U/N labels + catalytic span + sigma* + confidence (rpsblast→labels_with_spans.tsv)

Adds:
  - w_min = max(70/L_eff, 0.02)
  - sigma_star = max((E-S)/(2*k), w_min/2)    [only when use_span==1]
  - conf ∈ [0,1], from E-value, coverage, HSP count; conf *= 0.5 if truncated

TSV columns (new at end):
  ... S E span_start_aa span_end_aa truncated conf conf_e conf_cov conf_hsp w_min sigma_star k
"""
import os, sys, argparse, logging, h5py, math
from collections import defaultdict
from collections import Counter
from typing import List, Tuple, Dict, Optional

def setup_logging(level="INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=lvl, format="%(asctime)s | %(levelname)s | %(message)s")
    return logging.getLogger("rps_to_labels")

# -------------------------
# H5 lengths & sequences
# -------------------------
def load_lengths_and_seq(h5_path: str) -> Dict[str, Tuple[int, Optional[str]]]:
    out = {}
    with h5py.File(h5_path, "r") as f:
        items = f.get("items")
        if items is None:
            raise RuntimeError("H5 missing /items")
        for cid in items.keys():
            g = items[cid]
            L = None; seq = None
            if "seq" in g:
                try:    seq = g["seq"].asstr()[()]
                except Exception:
                    val = g["seq"][()]
                    seq = val.decode() if isinstance(val, bytes) else str(val)
                L = len(seq)
            if L is None or L <= 0:
                L = int(g.attrs.get("aa_len", -1))
                if L <= 0: raise RuntimeError(f"length unknown for {cid}")
            out[cid] = (L, seq)
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
# RPS parser
# -------------------------
class Hit:
    __slots__=("qid","sid","e","bits","qs","qe","ss","se","qlen","slen","title")
    def __init__(self,qid,sid,e,bits,qs,qe,ss,se,qlen,slen,title):
        self.qid=qid; self.sid=sid; self.e=e; self.bits=bits
        self.qs=qs; self.qe=qe; self.ss=ss; self.se=se
        self.qlen=qlen; self.slen=slen; self.title=title

def parse_rps_tsv(path: str) -> Dict[str, List[Hit]]:
    """
    Parse outfmt 6 with columns:
      qseqid sacc evalue bitscore qstart qend sstart send length qlen slen stitle qcovs qcovhsp
    """
    hits = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            if not ln.strip() or ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            # ensure 14 fields are safe to access
            while len(p) < 14:
                p.append("")
            qid = p[0]; sid = p[1]
            try:
                e = float(p[2]); bits = float(p[3])
                qs = int(p[4]);  qe = int(p[5])
                ss = int(p[6]) if p[6] else 0
                se = int(p[7]) if p[7] else 0
                # p[8] = alignment length (unused here)
                qlen = int(p[9])  if p[9]  else None
                slen = int(p[10]) if p[10] else None
            except Exception:
                continue
            title = p[11]  # ignore qcovs (p[12]) and qcovhsp (p[13])
            hits[qid].append(Hit(qid, sid, e, bits, qs, qe, ss, se, qlen, slen, title))
    return hits

def sanity_check_qlen_vs_h5L(
    rps_hits: Dict[str, List[Hit]],
    qlens: Dict[str, tuple],
    logger,
    abs_tol: int = 3,
    rel_tol: float = 0.10,
    max_warnings: int = 50
) -> int:
    """
    Compare rpsblast 'qlen' against HDF5 aa_len (L).
    Warn if |qlen - L| > max(abs_tol, rel_tol*L). Returns count of mismatches.
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
                logger.warning(
                    "qlen mismatch for %s: rps=%d vs HDF5 L=%d (tol=%d)",
                    qid, qlen_mode, L, thresh
                )
            n_warn += 1
    if max_warnings and n_warn > max_warnings:
        logger.warning("...and %d more qlen mismatches suppressed", n_warn - max_warnings)
    return n_warn
# -------------------------
# Intervals & coverage
# -------------------------
def to_half_open(qs:int, qe:int):
    s = min(qs,qe)-1; e = max(qs,qe)
    return max(0,s), max(0,e)

def merge_intervals(iv: List[Tuple[int,int]], max_gap:int=25):
    if not iv: return []
    iv = sorted(iv, key=lambda x:(x[0],x[1]))
    out=[iv[0]]
    for s,e in iv[1:]:
        ps,pe=out[-1]
        if s <= pe + max_gap: out[-1]=(ps, max(pe,e))
        else: out.append((s,e))
    return out

def union_len(iv: List[Tuple[int,int]]):
    return sum(max(0,e-s) for s,e in merge_intervals(iv, max_gap=0))

def qcov_of(hsps: List["Hit"], L:int)->float:
    if L<=0: return 0.0
    iv=[to_half_open(h.qs,h.qe) for h in hsps]
    return min(1.0, union_len(iv)/L)

def dcov_of(hsps: List["Hit"])->float:
    coords=[]; slen=None
    for h in hsps:
        if h.ss and h.se and h.slen:
            a,b=min(h.ss,h.se)-1, max(h.ss,h.se)
            coords.append((max(0,a), max(0,b))); slen=h.slen
    if not coords or not slen: return 0.0
    return min(1.0, union_len(coords)/slen)

# -------------------------
# Span selection
# -------------------------
def choose_primary_span(rdrp_hsps: List["Hit"], L:int, max_gap:int=25):
    if not rdrp_hsps: return (0,0,False)
    rows=[]
    for h in rdrp_hsps:
        s,e=to_half_open(h.qs,h.qe)
        rows.append((max(0,min(s,L)), max(0,min(e,L)), h.e, h.bits))
    rows.sort(key=lambda x:(x[0],x[1]))
    clusters=[]  # [s,e,best_e,max_bits]
    for s,e,ev,bs in rows:
        if not clusters: clusters=[[s,e,ev,bs]]
        else:
            ps,pe,be,bb=clusters[-1]
            if s <= pe + max_gap:
                clusters[-1]=[ps, max(pe,e), min(be,ev), max(bb,bs)]
            else:
                clusters.append([s,e,ev,bs])
    best=None
    for s,e,be,bb in clusters:
        length=max(0,e-s)
        cand=(length, -be, bb, s, e)  # prefer longer, lower e, higher bits
        if best is None or cand>best: best=cand
    _,_,_, s,e = best
    s=max(0,min(s,L)); e=max(0,min(e,L))
    truncated=(s==0) or (e==L)
    return (s,e,truncated)

# -------------------------
# Confidence from rps evidence
# -------------------------
def clip01(x: float)->float:
    return 0.0 if x<0.0 else 1.0 if x>1.0 else x

def conf_from_metrics(e_best: Optional[float], cov_best: float, hsp_count: int, truncated: bool)->Tuple[float,float,float,float]:
    # Components
    if e_best is None or e_best<=0.0:
        conf_e=1.0       # treat missing/zero as max conf (shouldn't happen for P)
    else:
        conf_e = clip01(( -math.log10(e_best) - 5.0 ) / 5.0)  # 0 at 1e-5, 1 at 1e-10
    conf_cov = clip01(cov_best / 0.60)                         # saturate ~60% coverage
    conf_hsp = clip01(math.log1p(hsp_count) / math.log(1+3))   # ≥3 HSPs ≈ 1
    conf = 0.6*conf_e + 0.3*conf_cov + 0.1*conf_hsp
    if truncated:
        conf *= 0.5
    return conf, conf_e, conf_cov, conf_hsp

# -------------------------
# Labeling
# -------------------------
def decide_labels_for_query(
    qid: str, L: int,
    hits: List["Hit"], rdrp_ids: set,
    thr_pos_e: float, thr_pos_cov: float,
    thr_neg_e: float, thr_neg_qcov: float,
    thr_loose_rdrp_e: float,
    max_gap_for_span: int,
    sigma_k: float
):
    by_sid=defaultdict(list)
    for h in hits: by_sid[h.sid].append(h)

    subs=[]
    for sid,hsps in by_sid.items():
        best_e=min(h.e for h in hsps)
        qc=qcov_of(hsps, L)
        dc=dcov_of(hsps)
        subs.append((sid,best_e,qc,dc,hsps))

    pos_subs=[(sid,e,qc,dc,hsps) for sid,e,qc,dc,hsps in subs
              if sid in rdrp_ids and e<=thr_pos_e and (qc>=thr_pos_cov or dc>=thr_pos_cov)]
    pos = len(pos_subs)>0

    loose_rdrp = any((sid in rdrp_ids and e<=thr_loose_rdrp_e) for sid,e,_,_,_ in subs)

    neg_subs=[(sid,e,qc,dc,hsps) for sid,e,qc,dc,hsps in subs
              if (sid not in rdrp_ids) and e<=thr_neg_e and qc>=thr_neg_qcov]
    neg = len(neg_subs)>0

    if pos and neg: label="U"
    elif pos:       label="P"
    elif (not loose_rdrp) and neg: label="N"
    else:           label="U"

    # Defaults
    use_span=0; S=E=""; span_s=span_e=""; truncated=False
    conf=conf_e=conf_cov=conf_hsp=0.0
    w_min = max(70.0/max(1.0,L), 0.02)  # normalized

    if label=="P":
        # Gather RdRP HSPs at pos threshold first, else fallback to loose_rdrp
        rdrp_hsps=[h for sid,e,qc,dc,hsps in subs if (sid in rdrp_ids and e<=thr_pos_e) for h in hsps]
        if not rdrp_hsps:
            rdrp_hsps=[h for sid,e,qc,dc,hsps in subs if (sid in rdrp_ids and e<=thr_loose_rdrp_e) for h in hsps]

        if rdrp_hsps:
            s,e,truncated = choose_primary_span(rdrp_hsps, L, max_gap=max_gap_for_span)
            if e> s:
                S=f"{s/L:.6f}"; E=f"{e/L:.6f}"
                span_s=str(s); span_e=str(e)
                use_span=1

            # confidence: use best RdRP e and best of qcov/dcov across RdRP subjects
            e_best = min((h.e for h in rdrp_hsps), default=None)
            # compute cov_best over RdRP subjects
            cov_best = 0.0
            # recompute per subject to count HSPs properly
            rdrp_sid_map=defaultdict(list)
            for h in rdrp_hsps: rdrp_sid_map[h.sid].append(h)
            hsp_count = 0
            for hsps in rdrp_sid_map.values():
                hsp_count += len(hsps)
                qc = qcov_of(hsps, L); dc = dcov_of(hsps)
                cov_best = max(cov_best, qc, dc)

            conf, conf_e, conf_cov, conf_hsp = conf_from_metrics(e_best, cov_best, hsp_count, truncated)

    # sigma* target (only meaningful when we have a usable span)
    sigma_star = ""
    if use_span==1:
        w = float(E) - float(S)          # normalized width
        sigma_star = f"{max(w/(2.0*sigma_k), w_min/2.0):.6f}"

    rec = {
        "chunk_id": qid, "label": label, "use_span": str(use_span), "L": str(L),
        "S": S, "E": E,
        "span_start_aa": span_s, "span_end_aa": span_e,
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
def main():
    ap = argparse.ArgumentParser(description="P/U/N + span + sigma* + confidence from rpsblast (CDD)")
    ap.add_argument("--h5", required=True)
    ap.add_argument("--rps", required=True, help="rpsblast outfmt 6 TSV")
    ap.add_argument("--rdrp-ids", required=True, help="file with RdRP CDD IDs (one per line)")
    ap.add_argument("--out", required=True, help="labels_with_spans.tsv")
    ap.add_argument("--log-level", default="INFO")

    # thresholds
    ap.add_argument("--pos-e", type=float, default=1e-5)
    ap.add_argument("--pos-cov", type=float, default=0.35)
    ap.add_argument("--neg-e", type=float, default=1e-5)
    ap.add_argument("--neg-qcov", type=float, default=0.80)
    ap.add_argument("--loose-rdrp-e", type=float, default=1e-2)

    # span building
    ap.add_argument("--span-max-gap", type=int, default=25)

    # sigma target
    ap.add_argument("--sigma-k", type=float, default=2.0, help="k in sigma* = max(w/(2k), w_min/2)")

    ap.add_argument("--qlen-abs-tol", type=int, default=3,
                    help="Absolute tolerance (aa) for qlen vs HDF5 L mismatch.")
    ap.add_argument("--qlen-rel-tol", type=float, default=0.10,
                    help="Relative tolerance for qlen vs HDF5 L (fraction of L).")
    ap.add_argument("--qlen-max-warn", type=int, default=50,
                    help="Max warnings to print (0 = no limit).")

    args = ap.parse_args()
    log = setup_logging(args.log_level)

    qlens = load_lengths_and_seq(args.h5)   # {qid: (L, seq)}
    rps = parse_rps_tsv(args.rps)

    log.info("Sanity-checking qlen (rps) vs L (HDF5)...")
    mismatches = sanity_check_qlen_vs_h5L(
        rps_hits=rps,
        qlens=qlens,
        logger=log,
        abs_tol=args.qlen_abs_tol,
        rel_tol=args.qlen_rel_tol,
        max_warnings=args.qlen_max_warn
    )
    if mismatches:
        log.warning("Found %d qlen≠L mismatches (see warnings above).", mismatches)

    rdrp_ids = read_id_list(args.rdrp_ids)
    if not rdrp_ids:
        log.error("Empty RdRP ID list: %s", args.rdrp_ids); sys.exit(2)

    n=nP=nU=nN=nSpan=0
    with open(args.out, "w", encoding="utf-8") as out:
        header = [
            "chunk_id","label","use_span","L","S","E",
            "span_start_aa","span_end_aa","truncated",
            "conf","conf_e","conf_cov","conf_hsp","w_min","sigma_star","k"
        ]
        out.write("\t".join(header) + "\n")
        for qid,(L,_) in qlens.items():
            rec = decide_labels_for_query(
                qid, L, rps.get(qid, []), rdrp_ids,
                args.pos_e, args.pos_cov,
                args.neg_e, args.neg_qcov,
                args.loose_rdrp_e, args.span_max_gap,
                args.sigma_k
            )
            out.write("\t".join(rec[k] for k in header) + "\n")
            n += 1
            if rec["label"] == "P":
                nP += 1
                if rec["use_span"] == "1": nSpan += 1
            elif rec["label"] == "N": nN += 1
            else: nU += 1

    log.info("Done. Wrote %d rows → %s [P=%d (span=%d), U=%d, N=%d]",
             n, args.out, nP, nSpan, nU, nN)

if __name__ == "__main__":
    main()

