#!/usr/bin/env python3
"""
make_palmsite_noise_mutants.py

Generate amino-acid perturbation/noise FASTA files from original PalmSite outputs.

The script uses PalmSite attention/logit outputs from the original sequence to define
target regions such as:
  - whole sequence
  - PalmSite predicted span
  - outside predicted span
  - high-attention residues
  - low-attention residues
  - user-defined regions
  - sliding-window occlusion/mutation

It does NOT run PalmSite. After generating mutant FASTA files, run PalmSite on the
mutants with --attn-json and --logits-json, then compare mutant calibrated_logit
against the original calibrated_logit recorded in the manifest.

Typical use on a directory:
  python make_palmsite_noise_mutants.py \
    --input-dir noise_analysis \
    --out-dir noise_mutants \
    --rates 0.01,0.02,0.05,0.10,0.20 \
    --replicates 50 \
    --targets whole,span,outside_span,high_attention,low_attention \
    --mutation-mode random \
    --seed 1

Sliding-window scan:
  python make_palmsite_noise_mutants.py \
    --input-dir noise_analysis \
    --out-dir noise_mutants_windows \
    --targets sliding_window \
    --window-size 15 \
    --window-stride 3 \
    --window-scope span \
    --mutation-mode random \
    --window-rates 1.0 \
    --window-replicates 1 \
    --seed 1

Sliding-window scan with replicates and partial-window mutation fractions:
  python make_palmsite_noise_mutants.py \
    --input-dir noise_analysis \
    --out-dir noise_mutants_windows \
    --targets sliding_window \
    --window-size 15 \
    --window-stride 3 \
    --window-scope span \
    --window-rates 0.25,0.50,1.0 \
    --window-replicates 20 \
    --mutation-mode random \
    --seed 1

User-defined regions TSV:
  sequence_pattern region_name start1 end1
  hsrv_rdrp        c_like      408    448
  hsrv_rdrp        ab_like     343    382

  python make_palmsite_noise_mutants.py \
    --input-dir noise_analysis \
    --out-dir noise_mutants_regions \
    --regions-tsv regions.tsv \
    --targets user_regions \
    --rates 0.25,0.50,1.0 \
    --replicates 50
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


AA = "ACDEFGHIKLMNPQRSTVWY"
AA_SET = set(AA)

# Conservative amino-acid groups. The replacement is sampled from the same group
# when possible and never equals the original residue.
CONSERVATIVE_GROUPS = [
    "AVLIM",      # hydrophobic aliphatic
    "FYW",        # aromatic
    "ST",         # small polar
    "NQ",         # amide
    "DE",         # acidic
    "KRH",        # basic
    "GP",         # special/flexible
    "C",          # cysteine separate by default
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate noisy/mutated amino-acid FASTA files using original PalmSite outputs."
    )

    src = ap.add_argument_group("Input")
    src.add_argument("--input-dir", help="Directory containing matched *.fasta and *.attention.json files.")
    src.add_argument("--fasta", action="append", help="Input FASTA. Can be supplied multiple times.")
    src.add_argument("--attention-json", action="append", help="PalmSite --attn-json output. Can be supplied multiple times.")
    src.add_argument("--logits-json", action="append", help="PalmSite --logits-json output. Optional; can be supplied multiple times.")
    src.add_argument("--regions-tsv", help="Optional TSV: sequence_pattern region_name start1 end1. Used with target=user_regions.")
    src.add_argument("--out-dir", required=True, help="Output directory.")
    src.add_argument("--prefix", default="palmsite_noise", help="Output file prefix.")

    exp = ap.add_argument_group("Experiment design")
    exp.add_argument(
        "--targets",
        default="whole,span,outside_span,high_attention,low_attention",
        help=("Comma-separated targets. Options: whole,span,outside_span,high_attention,"
              "low_attention,user_regions,sliding_window.")
    )
    exp.add_argument("--rates", default="0.01,0.02,0.05,0.10,0.20", help="Comma-separated mutation fractions for dose-response targets.")
    exp.add_argument("--replicates", type=int, default=50, help="Replicates per target/rate/sequence.")
    exp.add_argument("--attention-fraction", type=float, default=0.10, help="Fraction of residues selected for high/low attention targets.")
    exp.add_argument("--attention-scope", choices=["span", "full"], default="span", help="Where high/low attention residues are ranked.")
    exp.add_argument("--min-target-residues", type=int, default=1, help="Minimum number of mutable residues per mutant.")
    exp.add_argument("--include-original", action="store_true", help="Include original sequences in output FASTA and manifest.")

    mut = ap.add_argument_group("Mutation model")
    mut.add_argument(
        "--mutation-mode",
        choices=["random", "conservative", "nonconservative", "alanine", "glycine", "shuffle"],
        default="random",
        help=("random: random AA except itself; conservative: same biochemical group; "
              "nonconservative: outside original group; alanine/glycine: replacement; "
              "shuffle: shuffle selected residues among themselves.")
    )
    mut.add_argument("--preserve-catalytic-acidic", action="store_true",
                     help="Do not mutate D/E residues. Useful for conservative-noise controls, not for catalytic-site disruption.")
    mut.add_argument("--seed", type=int, default=1, help="Random seed.")

    win = ap.add_argument_group("Sliding-window options")
    win.add_argument("--window-size", type=int, default=15, help="Sliding-window size.")
    win.add_argument("--window-stride", type=int, default=3, help="Sliding-window stride.")
    win.add_argument("--window-scope", choices=["span", "full"], default="span", help="Scope for sliding-window starts.")
    win.add_argument("--window-rates", default=None,
                     help=("Comma-separated mutation fractions for sliding-window mutants. "
                           "If omitted, uses --window-mutate-fraction for backward compatibility."))
    win.add_argument("--window-replicates", type=int, default=1,
                     help="Replicates per sliding window per window mutation fraction.")
    win.add_argument("--window-mutate-fraction", type=float, default=1.0,
                     help=("Backward-compatible single fraction of residues inside each sliding window to mutate. "
                           "Ignored when --window-rates is supplied."))

    out = ap.add_argument_group("Output")
    out.add_argument("--one-fasta-per-sequence", action="store_true",
                     help="Write separate FASTA/manifest per input sequence prefix.")
    out.add_argument("--max-records-per-fasta", type=int, default=0,
                     help="If >0, split output FASTA into numbered shards with at most this many records.")
    out.add_argument("--id-max-len", type=int, default=180, help="Maximum generated FASTA ID length before adding hash.")

    args = ap.parse_args()

    if not args.input_dir and not args.fasta:
        ap.error("Provide --input-dir or --fasta/--attention-json.")
    if args.fasta and not args.attention_json:
        ap.error("--attention-json is required when using --fasta directly.")
    if args.window_replicates < 1:
        ap.error("--window-replicates must be >= 1.")
    if args.window_rates is not None:
        # Validate now so bad values fail before any output is opened.
        parse_csv_floats(args.window_rates)
    elif not (0 <= args.window_mutate_fraction <= 1):
        ap.error("--window-mutate-fraction must be between 0 and 1.")
    return args


def parse_csv_floats(s: str) -> List[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        if not (0 <= v <= 1):
            raise ValueError(f"Rate must be between 0 and 1: {v}")
        vals.append(v)
    return vals


def parse_csv_strings(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, List[str]] = {}
    sid: Optional[str] = None
    chunks: List[str] = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if sid is not None:
                    seqs[sid] = ["".join(chunks)]
                sid = line[1:].split()[0]
                chunks = []
            else:
                chunks.append(line)
        if sid is not None:
            seqs[sid] = ["".join(chunks)]
    return {k: v[0] for k, v in seqs.items()}


def write_fasta_record(handle, seq_id: str, seq: str, width: int = 80) -> None:
    handle.write(f">{seq_id}\n")
    for i in range(0, len(seq), width):
        handle.write(seq[i:i + width] + "\n")


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def best_attention_entry(attn: dict) -> Tuple[str, dict]:
    """Choose one PalmSite chunk per original sequence.

    Prefer is_best_base_chunk=True. If multiple are present, choose max P.
    If none are marked, choose max P among all non-meta entries.
    """
    entries = [(k, v) for k, v in attn.items() if not k.startswith("_")]
    if not entries:
        raise ValueError("No entries in attention JSON")
    marked = [(k, v) for k, v in entries if bool(v.get("is_best_base_chunk"))]
    pool = marked if marked else entries
    return max(pool, key=lambda kv: float(kv[1].get("P", -1e9)))


def best_logits_entry(logits: Optional[dict], chunk_id: str, base_id: str) -> Optional[dict]:
    if not logits:
        return None
    if chunk_id in logits:
        return logits[chunk_id]
    entries = [(k, v) for k, v in logits.items() if not k.startswith("_")]
    same_base = [(k, v) for k, v in entries if v.get("base_id") == base_id]
    pool = same_base if same_base else entries
    if not pool:
        return None
    return max(pool, key=lambda kv: float(kv[1].get("P", -1e9)))[1]


def discover_input_sets(input_dir: Path) -> List[Tuple[str, Path, Path, Optional[Path]]]:
    """Return list of (name, fasta, attention_json, logits_json)."""
    fasta_files = sorted(list(input_dir.glob("*.fasta")) + list(input_dir.glob("*.fa")) + list(input_dir.glob("*.faa")))
    out = []
    for fasta in fasta_files:
        stem = fasta.name
        for suffix in [".fasta", ".fa", ".faa"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        candidates = [
            input_dir / f"{stem}.attention.json",
            input_dir / f"{stem}.attn.json",
            input_dir / f"{stem}.weights.json",
            input_dir / f"{stem}.weight.json",
        ]
        attn = next((p for p in candidates if p.exists()), None)
        if attn is None:
            continue
        logits = input_dir / f"{stem}.logits.json"
        out.append((stem, fasta, attn, logits if logits.exists() else None))
    if not out:
        raise ValueError(f"No matched *.fasta + *.attention.json files found in {input_dir}")
    return out


def explicit_input_sets(fastas: List[str], attns: List[str], logits: Optional[List[str]]) -> List[Tuple[str, Path, Path, Optional[Path]]]:
    if len(fastas) != len(attns):
        raise ValueError("--fasta and --attention-json must have the same number of arguments")
    logits = logits or []
    if logits and len(logits) != len(fastas):
        raise ValueError("If supplied, --logits-json must have the same number of arguments as --fasta")
    out = []
    for i, (fa, at) in enumerate(zip(fastas, attns)):
        p = Path(fa)
        stem = p.name
        for suffix in [".fasta", ".fa", ".faa"]:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        out.append((stem, Path(fa), Path(at), Path(logits[i]) if logits else None))
    return out


def safe_id(text: str, max_len: int = 180) -> str:
    text = re.sub(r"[^A-Za-z0-9_.|:+-]+", "_", text)
    if len(text) <= max_len:
        return text
    h = hashlib.sha1(text.encode()).hexdigest()[:10]
    return text[:max_len - 11] + "_" + h


def aa_group(aa: str) -> Optional[str]:
    for g in CONSERVATIVE_GROUPS:
        if aa in g:
            return g
    return None


def choose_replacement(aa: str, mode: str, rng: random.Random) -> str:
    aa = aa.upper()
    if mode == "alanine":
        return "A" if aa != "A" else rng.choice([x for x in AA if x != aa])
    if mode == "glycine":
        return "G" if aa != "G" else rng.choice([x for x in AA if x != aa])
    if mode == "random":
        return rng.choice([x for x in AA if x != aa])
    group = aa_group(aa)
    if mode == "conservative":
        if group and len(group) > 1:
            return rng.choice([x for x in group if x != aa])
        return rng.choice([x for x in AA if x != aa])
    if mode == "nonconservative":
        if group:
            candidates = [x for x in AA if x not in group and x != aa]
        else:
            candidates = [x for x in AA if x != aa]
        return rng.choice(candidates)
    raise ValueError(f"Unsupported replacement mode: {mode}")


def mutate_sequence(
    seq: str,
    positions0: Sequence[int],
    rate: float,
    rng: random.Random,
    mode: str,
    min_target_residues: int = 1,
    preserve_catalytic_acidic: bool = False,
) -> Tuple[str, List[int], List[str], List[str]]:
    """Mutate a fraction of positions inside positions0.

    Returns mutant sequence, mutated 0-based positions, original aas, mutant aas.
    """
    seq_list = list(seq)
    positions = [p for p in positions0 if 0 <= p < len(seq_list) and seq_list[p].upper() in AA_SET]
    if preserve_catalytic_acidic:
        positions = [p for p in positions if seq_list[p].upper() not in {"D", "E"}]
    if not positions:
        return seq, [], [], []

    n = int(round(len(positions) * rate))
    if rate > 0:
        n = max(min_target_residues, n)
    n = min(n, len(positions))
    if n <= 0:
        return seq, [], [], []

    chosen = sorted(rng.sample(positions, n))
    original = [seq_list[p] for p in chosen]

    if mode == "shuffle":
        shuffled = original[:]
        for _ in range(20):
            rng.shuffle(shuffled)
            if any(a != b for a, b in zip(original, shuffled)):
                break
        # If all residues identical or shuffle failed, fall back to random mutations.
        if all(a == b for a, b in zip(original, shuffled)):
            shuffled = [choose_replacement(a.upper(), "random", rng) for a in original]
        for p, aa_new in zip(chosen, shuffled):
            seq_list[p] = aa_new
    else:
        for p in chosen:
            aa = seq_list[p].upper()
            if aa in AA_SET:
                seq_list[p] = choose_replacement(aa, mode, rng)

    mutated = [seq_list[p] for p in chosen]
    return "".join(seq_list), chosen, original, mutated


def region_positions_from_entry(entry: dict, seq_len: int, target: str, attention_fraction: float, attention_scope: str) -> List[int]:
    abs_pos = list(map(int, entry.get("abs_pos", list(range(seq_len)))))
    w = list(map(float, entry.get("w", [0.0] * len(abs_pos))))
    span_start = int(entry.get("orig_start", 0)) + int(entry.get("S_idx", 0))
    span_end = int(entry.get("orig_start", 0)) + int(entry.get("E_idx", seq_len - 1))

    full_positions = list(range(seq_len))
    span_positions = [p for p in full_positions if span_start <= p <= span_end]
    outside_positions = [p for p in full_positions if not (span_start <= p <= span_end)]

    if target == "whole":
        return full_positions
    if target == "span":
        return span_positions
    if target == "outside_span":
        return outside_positions

    if target in {"high_attention", "low_attention"}:
        if attention_scope == "span":
            allowed = set(span_positions)
        else:
            allowed = set(full_positions)
        pairs = [(p, score) for p, score in zip(abs_pos, w) if p in allowed and 0 <= p < seq_len]
        if not pairs:
            return []
        n = max(1, int(round(len(pairs) * attention_fraction)))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=(target == "high_attention"))
        return sorted([p for p, _ in pairs[:n]])

    raise ValueError(f"Unknown target: {target}")


def sliding_windows(entry: dict, seq_len: int, window_size: int, stride: int, scope: str) -> List[Tuple[str, List[int], int, int]]:
    span_start = int(entry.get("orig_start", 0)) + int(entry.get("S_idx", 0))
    span_end = int(entry.get("orig_start", 0)) + int(entry.get("E_idx", seq_len - 1))
    if scope == "span":
        start_min, start_max = span_start, span_end
    else:
        start_min, start_max = 0, seq_len - 1
    out = []
    if window_size <= 0 or stride <= 0:
        raise ValueError("--window-size and --window-stride must be >0")
    for s in range(start_min, start_max + 1, stride):
        e = min(s + window_size - 1, start_max)
        if e < s:
            continue
        positions = list(range(s, e + 1))
        name = f"window_{s+1}_{e+1}"
        out.append((name, positions, s + 1, e + 1))
        if e == start_max:
            break
    return out


def load_regions_tsv(path: Optional[str]) -> Dict[str, List[Tuple[str, int, int]]]:
    regions: Dict[str, List[Tuple[str, int, int]]] = defaultdict(list)
    if not path:
        return regions
    with open(path) as fh:
        for line_no, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"\t+|\s+", line)
            if parts[0].lower() in {"sequence_pattern", "pattern", "sample"}:
                continue
            if len(parts) < 4:
                raise ValueError(f"Bad regions TSV line {line_no}: {raw!r}")
            pat, name, start1, end1 = parts[:4]
            s = int(start1)
            e = int(end1)
            if s > e:
                s, e = e, s
            regions[pat].append((name, s, e))
    return regions


def matching_user_regions(regions: Dict[str, List[Tuple[str, int, int]]], sample_name: str, base_id: str, fasta_id: str) -> List[Tuple[str, List[int], int, int]]:
    out = []
    haystack = f"{sample_name} {base_id} {fasta_id}"
    for pat, regs in regions.items():
        if pat in haystack:
            for name, s1, e1 in regs:
                out.append((name, list(range(s1 - 1, e1)), s1, e1))
    return out


def select_sequence_for_entry(seqs: Dict[str, str], base_id: str) -> Tuple[str, str]:
    if base_id in seqs:
        return base_id, seqs[base_id]
    hits = [sid for sid in seqs if base_id in sid or sid in base_id]
    if len(hits) == 1:
        return hits[0], seqs[hits[0]]
    if len(seqs) == 1:
        sid = next(iter(seqs))
        return sid, seqs[sid]
    raise ValueError(f"Could not match base_id={base_id!r} to FASTA IDs {list(seqs)}")


def open_fasta_writer(out_dir: Path, prefix: str, shard_index: int) -> Tuple[Path, object]:
    if shard_index == 0:
        path = out_dir / f"{prefix}.mutants.fasta"
    else:
        path = out_dir / f"{prefix}.mutants.part{shard_index:03d}.fasta"
    return path, open(path, "w")


def run() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rates = parse_csv_floats(args.rates)
    window_rates = parse_csv_floats(args.window_rates) if args.window_rates is not None else [float(args.window_mutate_fraction)]
    targets = parse_csv_strings(args.targets)
    allowed_targets = {"whole", "span", "outside_span", "high_attention", "low_attention", "user_regions", "sliding_window"}
    bad = [t for t in targets if t not in allowed_targets]
    if bad:
        raise ValueError(f"Unknown targets: {bad}; allowed={sorted(allowed_targets)}")

    if args.input_dir:
        input_sets = discover_input_sets(Path(args.input_dir))
    else:
        input_sets = explicit_input_sets(args.fasta, args.attention_json, args.logits_json)

    user_regions = load_regions_tsv(args.regions_tsv)

    manifest_fields = [
        "mutant_id", "sample_name", "source_fasta", "source_attention_json", "source_logits_json",
        "fasta_id", "base_id", "chunk_id", "target", "region_name", "rate", "replicate",
        "n_mutated", "positions_1based", "original_aas", "mutant_aas",
        "span_start1", "span_end1", "original_P", "original_logit", "original_calibrated_logit",
        "temperature", "mutation_mode", "seed"
    ]

    # Global outputs unless one-fasta-per-sequence is requested.
    global_fasta_handle = None
    global_fasta_path = None
    global_manifest_handle = None
    global_manifest_writer = None
    global_count_in_shard = 0
    global_shard = 0

    if not args.one_fasta_per_sequence:
        global_fasta_path, global_fasta_handle = open_fasta_writer(out_dir, args.prefix, 0)
        global_manifest_path = out_dir / f"{args.prefix}.manifest.tsv"
        global_manifest_handle = open(global_manifest_path, "w", newline="")
        global_manifest_writer = csv.DictWriter(global_manifest_handle, fieldnames=manifest_fields, delimiter="\t")
        global_manifest_writer.writeheader()

    per_sample_summaries = []

    def write_record(seq_id: str, seq: str, row: dict, sample_prefix: str,
                     sample_fasta_handle=None, sample_manifest_writer=None):
        nonlocal global_fasta_handle, global_fasta_path, global_manifest_writer, global_count_in_shard, global_shard
        if args.one_fasta_per_sequence:
            write_fasta_record(sample_fasta_handle, seq_id, seq)
            sample_manifest_writer.writerow(row)
        else:
            if args.max_records_per_fasta and global_count_in_shard >= args.max_records_per_fasta:
                global_fasta_handle.close()
                global_shard += 1
                global_fasta_path, global_fasta_handle = open_fasta_writer(out_dir, args.prefix, global_shard)
                global_count_in_shard = 0
            write_fasta_record(global_fasta_handle, seq_id, seq)
            global_manifest_writer.writerow(row)
            global_count_in_shard += 1

    for sample_name, fasta_path, attn_path, logits_path in input_sets:
        seqs = parse_fasta(fasta_path)
        attn = load_json(attn_path)
        logits = load_json(logits_path) if logits_path else None

        chunk_id, entry = best_attention_entry(attn)
        base_id = entry.get("base_id", chunk_id)
        fasta_id, seq = select_sequence_for_entry(seqs, base_id)
        seq_len = len(seq)

        log_entry = best_logits_entry(logits, chunk_id, base_id)
        original_P = float(entry.get("P", log_entry.get("P") if log_entry else float("nan")))
        original_logit = float(entry.get("logit", log_entry.get("logit") if log_entry else float("nan")))
        original_calibrated_logit = float(entry.get("calibrated_logit", log_entry.get("calibrated_logit") if log_entry else float("nan")))
        temperature = float(entry.get("temperature", log_entry.get("temperature") if log_entry else float("nan")))

        span_start0 = int(entry.get("orig_start", 0)) + int(entry.get("S_idx", 0))
        span_end0 = int(entry.get("orig_start", 0)) + int(entry.get("E_idx", seq_len - 1))
        span_start1, span_end1 = span_start0 + 1, span_end0 + 1

        sample_fasta_handle = None
        sample_manifest_handle = None
        sample_manifest_writer = None
        if args.one_fasta_per_sequence:
            sample_prefix = safe_id(f"{args.prefix}.{sample_name}", args.id_max_len)
            sample_fasta_path = out_dir / f"{sample_prefix}.mutants.fasta"
            sample_manifest_path = out_dir / f"{sample_prefix}.manifest.tsv"
            sample_fasta_handle = open(sample_fasta_path, "w")
            sample_manifest_handle = open(sample_manifest_path, "w", newline="")
            sample_manifest_writer = csv.DictWriter(sample_manifest_handle, fieldnames=manifest_fields, delimiter="\t")
            sample_manifest_writer.writeheader()

        n_written = 0

        if args.include_original:
            mut_id = safe_id(f"{sample_name}|original|P={original_P:.6g}|logit={original_calibrated_logit:.6g}", args.id_max_len)
            row = {
                "mutant_id": mut_id, "sample_name": sample_name, "source_fasta": str(fasta_path),
                "source_attention_json": str(attn_path), "source_logits_json": str(logits_path) if logits_path else "",
                "fasta_id": fasta_id, "base_id": base_id, "chunk_id": chunk_id,
                "target": "original", "region_name": "original", "rate": 0, "replicate": 0,
                "n_mutated": 0, "positions_1based": "", "original_aas": "", "mutant_aas": "",
                "span_start1": span_start1, "span_end1": span_end1,
                "original_P": original_P, "original_logit": original_logit,
                "original_calibrated_logit": original_calibrated_logit,
                "temperature": temperature, "mutation_mode": args.mutation_mode, "seed": args.seed
            }
            write_record(mut_id, seq, row, sample_name, sample_fasta_handle, sample_manifest_writer)
            n_written += 1

        # Dose-response targets
        for target in targets:
            if target in {"sliding_window", "user_regions"}:
                continue
            positions = region_positions_from_entry(entry, seq_len, target, args.attention_fraction, args.attention_scope)
            if not positions:
                continue
            for rate in rates:
                for rep in range(1, args.replicates + 1):
                    mut_seq, pos0, orig_aas, mut_aas = mutate_sequence(
                        seq, positions, rate, rng, args.mutation_mode,
                        min_target_residues=args.min_target_residues,
                        preserve_catalytic_acidic=args.preserve_catalytic_acidic,
                    )
                    if not pos0:
                        continue
                    mut_id_raw = (
                        f"{sample_name}|target={target}|rate={rate:g}|rep={rep}|"
                        f"n={len(pos0)}|mode={args.mutation_mode}"
                    )
                    mut_id = safe_id(mut_id_raw, args.id_max_len)
                    row = {
                        "mutant_id": mut_id, "sample_name": sample_name, "source_fasta": str(fasta_path),
                        "source_attention_json": str(attn_path), "source_logits_json": str(logits_path) if logits_path else "",
                        "fasta_id": fasta_id, "base_id": base_id, "chunk_id": chunk_id,
                        "target": target, "region_name": target, "rate": rate, "replicate": rep,
                        "n_mutated": len(pos0),
                        "positions_1based": ",".join(str(p + 1) for p in pos0),
                        "original_aas": "".join(orig_aas), "mutant_aas": "".join(mut_aas),
                        "span_start1": span_start1, "span_end1": span_end1,
                        "original_P": original_P, "original_logit": original_logit,
                        "original_calibrated_logit": original_calibrated_logit,
                        "temperature": temperature, "mutation_mode": args.mutation_mode, "seed": args.seed
                    }
                    write_record(mut_id, mut_seq, row, sample_name, sample_fasta_handle, sample_manifest_writer)
                    n_written += 1

        # User-defined regions
        if "user_regions" in targets:
            regs = matching_user_regions(user_regions, sample_name, base_id, fasta_id)
            for region_name, positions, start1, end1 in regs:
                for rate in rates:
                    for rep in range(1, args.replicates + 1):
                        mut_seq, pos0, orig_aas, mut_aas = mutate_sequence(
                            seq, positions, rate, rng, args.mutation_mode,
                            min_target_residues=args.min_target_residues,
                            preserve_catalytic_acidic=args.preserve_catalytic_acidic,
                        )
                        if not pos0:
                            continue
                        mut_id = safe_id(
                            f"{sample_name}|target=user_region|region={region_name}|"
                            f"{start1}-{end1}|rate={rate:g}|rep={rep}|mode={args.mutation_mode}",
                            args.id_max_len
                        )
                        row = {
                            "mutant_id": mut_id, "sample_name": sample_name, "source_fasta": str(fasta_path),
                            "source_attention_json": str(attn_path), "source_logits_json": str(logits_path) if logits_path else "",
                            "fasta_id": fasta_id, "base_id": base_id, "chunk_id": chunk_id,
                            "target": "user_regions", "region_name": region_name, "rate": rate, "replicate": rep,
                            "n_mutated": len(pos0),
                            "positions_1based": ",".join(str(p + 1) for p in pos0),
                            "original_aas": "".join(orig_aas), "mutant_aas": "".join(mut_aas),
                            "span_start1": span_start1, "span_end1": span_end1,
                            "original_P": original_P, "original_logit": original_logit,
                            "original_calibrated_logit": original_calibrated_logit,
                            "temperature": temperature, "mutation_mode": args.mutation_mode, "seed": args.seed
                        }
                        write_record(mut_id, mut_seq, row, sample_name, sample_fasta_handle, sample_manifest_writer)
                        n_written += 1

        # Sliding windows
        if "sliding_window" in targets:
            for region_name, positions, start1, end1 in sliding_windows(
                entry, seq_len, args.window_size, args.window_stride, args.window_scope
            ):
                for window_rate in window_rates:
                    for rep in range(1, args.window_replicates + 1):
                        mut_seq, pos0, orig_aas, mut_aas = mutate_sequence(
                            seq, positions, window_rate, rng, args.mutation_mode,
                            min_target_residues=args.min_target_residues,
                            preserve_catalytic_acidic=args.preserve_catalytic_acidic,
                        )
                        if not pos0:
                            continue
                        mut_id = safe_id(
                            f"{sample_name}|target=sliding_window|region={region_name}|"
                            f"rate={window_rate:g}|rep={rep}|n={len(pos0)}|mode={args.mutation_mode}",
                            args.id_max_len
                        )
                        row = {
                            "mutant_id": mut_id, "sample_name": sample_name, "source_fasta": str(fasta_path),
                            "source_attention_json": str(attn_path), "source_logits_json": str(logits_path) if logits_path else "",
                            "fasta_id": fasta_id, "base_id": base_id, "chunk_id": chunk_id,
                            "target": "sliding_window", "region_name": region_name,
                            "rate": window_rate, "replicate": rep,
                            "n_mutated": len(pos0),
                            "positions_1based": ",".join(str(p + 1) for p in pos0),
                            "original_aas": "".join(orig_aas), "mutant_aas": "".join(mut_aas),
                            "span_start1": span_start1, "span_end1": span_end1,
                            "original_P": original_P, "original_logit": original_logit,
                            "original_calibrated_logit": original_calibrated_logit,
                            "temperature": temperature, "mutation_mode": args.mutation_mode, "seed": args.seed
                        }
                        write_record(mut_id, mut_seq, row, sample_name, sample_fasta_handle, sample_manifest_writer)
                        n_written += 1

        if args.one_fasta_per_sequence:
            sample_fasta_handle.close()
            sample_manifest_handle.close()

        per_sample_summaries.append({
            "sample_name": sample_name,
            "fasta": str(fasta_path),
            "attention_json": str(attn_path),
            "logits_json": str(logits_path) if logits_path else None,
            "fasta_id": fasta_id,
            "base_id": base_id,
            "chunk_id": chunk_id,
            "sequence_length": seq_len,
            "span_start1": span_start1,
            "span_end1": span_end1,
            "original_P": original_P,
            "original_logit": original_logit,
            "original_calibrated_logit": original_calibrated_logit,
            "n_mutants_written": n_written,
        })

    if global_fasta_handle:
        global_fasta_handle.close()
    if global_manifest_handle:
        global_manifest_handle.close()

    summary = {
        "parameters": vars(args),
        "n_input_sets": len(input_sets),
        "samples": per_sample_summaries,
    }
    with open(out_dir / f"{args.prefix}.summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote outputs to: {out_dir}")
    print(f"Input sets: {len(input_sets)}")
    print(f"Total mutants/originals written: {sum(x['n_mutants_written'] for x in per_sample_summaries)}")
    if not args.one_fasta_per_sequence:
        print(f"FASTA: {out_dir / (args.prefix + '.mutants.fasta')}")
        print(f"Manifest: {out_dir / (args.prefix + '.manifest.tsv')}")
    print(f"Summary: {out_dir / (args.prefix + '.summary.json')}")


if __name__ == "__main__":
    run()

