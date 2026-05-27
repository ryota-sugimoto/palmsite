#!/usr/bin/env python3
"""
generate_motif_centered_perturbations_v3_unique_control_windows.py

Generate motif-centered amino-acid substitution perturbations from:
  1. an input protein FASTA file,
  2. a palm_annot TSV file containing motif positions, and optionally
  3. a PalmSite GFF/GFF3 file containing original PalmSite-predicted spans.

Main use case:
  - mutate motif A/B/C-centered windows;
  - generate matched random_in_span and random_outside_span controls;
  - define the predicted span for controls from PalmSite GFF output;
  - sample independent random control windows per control replicate; windows are sampled without replacement and made non-overlapping when enough valid positions exist.

Coordinates:
  - palm_annot motif positions are interpreted as 1-based motif-start coordinates.
  - GFF start/end coordinates are interpreted as 1-based inclusive coordinates.
  - output manifest reports 1-based inclusive coordinates.

The script writes:
  <out_prefix>.perturbed.fasta
  <out_prefix>.manifest.tsv
  <out_prefix>.skipped.tsv
  <out_prefix>.summary.tsv

All output text uses LF line endings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote

CANONICAL_AA = "ACDEFGHIKLMNPQRSTVWY"
DEFAULT_MOTIF_SEQ_KEYS = {
    "A": ("seqA", "motif_hmm_seqA", "pssm_seqA"),
    "B": ("seqB", "motif_hmm_seqB", "pssm_seqB"),
    "C": ("seqC", "motif_hmm_seqC", "pssm_seqC"),
}
DEFAULT_MOTIF_POS_KEYS = {
    "A": ("posA", "motif_hmm_posA", "pssm_posA"),
    "B": ("posB", "motif_hmm_posB", "pssm_posB"),
    "C": ("posC", "motif_hmm_posC", "pssm_posC"),
}
DEFAULT_PALM_ANNOT_SPAN_KEY_PAIRS = (
    ("pp_lo", "pp_hi"),
    ("ext_lo", "ext_hi"),
    ("hmm_rdrp_plus_lo", "hmm_rdrp_plus_hi"),
)


@dataclass(frozen=True)
class FastaRecord:
    seq_id: str
    header: str
    seq: str


@dataclass(frozen=True)
class MotifRecord:
    seq_id: str
    fields: Dict[str, str]


@dataclass(frozen=True)
class MotifSite:
    motif: str
    start_1based: int
    end_1based: int
    center_1based: int
    seq: str
    source_pos_key: str
    source_seq_key: str
    verified: bool


@dataclass(frozen=True)
class SpanRecord:
    seq_id: str
    start_1based: int
    end_1based: int
    source: str
    feature_type: str = ""
    score: Optional[float] = None
    raw_score: str = ""
    attributes: str = ""


@dataclass(frozen=True)
class WindowTarget:
    target_name: str
    motif: str
    window_start_1based: int
    window_end_1based: int
    motif_start_1based: str
    motif_end_1based: str
    motif_center_1based: str
    motif_seq: str
    span: Optional[SpanRecord]
    control_window_replicate: str = ""


@dataclass(frozen=True)
class PerturbationResult:
    perturbed_seq: str
    original_window: str
    perturbed_window: str
    mutated_positions_1based: List[int]
    mutated_from: List[str]
    mutated_to: List[str]


class SkipWriter:
    def __init__(self) -> None:
        self.rows: List[Dict[str, str]] = []
        self.counter: Counter[str] = Counter()

    def add(self, seq_id: str, reason: str, detail: str = "") -> None:
        self.rows.append({"seq_id": seq_id, "reason": reason, "detail": detail})
        self.counter[reason] += 1


def die(message: str, exit_code: int = 2) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def parse_csv_list(value: str) -> List[str]:
    if value is None or value == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> List[float]:
    items = []
    for token in parse_csv_list(value):
        try:
            x = float(token)
        except ValueError:
            die(f"Could not parse float value in list: {token!r}")
        if x < 0.0 or x > 1.0:
            die(f"Mutation rates must be in [0, 1], got {x}")
        items.append(x)
    if not items:
        die("At least one mutation rate is required")
    return items


def parse_int_list(value: str) -> List[int]:
    items = []
    for token in parse_csv_list(value):
        try:
            x = int(token)
        except ValueError:
            die(f"Could not parse integer value in list: {token!r}")
        if x <= 0:
            die(f"Window sizes must be positive integers, got {x}")
        items.append(x)
    if not items:
        die("At least one window size is required")
    return items


def safe_token(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.:+-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "NA"


def wrap_fasta(seq: str, width: int = 60) -> str:
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def read_fasta(path: Path) -> Dict[str, FastaRecord]:
    records: Dict[str, FastaRecord] = {}
    current_header: Optional[str] = None
    current_id: Optional[str] = None
    chunks: List[str] = []

    def flush() -> None:
        nonlocal current_header, current_id, chunks
        if current_header is None or current_id is None:
            return
        seq = "".join(chunks).replace(" ", "").replace("\t", "").upper()
        if not seq:
            die(f"Empty FASTA sequence for {current_id!r}")
        if current_id in records:
            die(f"Duplicate FASTA ID found: {current_id!r}")
        records[current_id] = FastaRecord(seq_id=current_id, header=current_header, seq=seq)
        current_header = None
        current_id = None
        chunks = []

    with path.open("r", encoding="utf-8", newline=None) as handle:
        for line in handle:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if line.startswith(">"):
                flush()
                current_header = line[1:].strip()
                current_id = current_header.split()[0]
                if not current_id:
                    die(f"Malformed FASTA header: {line!r}")
            else:
                if current_header is None:
                    die("FASTA sequence line appeared before first header")
                chunks.append(line.strip())
        flush()

    return records


def parse_key_value_tokens(tokens: Sequence[str]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            fields[key] = value
    return fields


def read_palm_annot(path: Path) -> Dict[str, MotifRecord]:
    """Read palm_annot-style TSV.

    The uploaded palm_annot file is key=value TSV without a header:
      seq_id<TAB>pssm_score=...<TAB>...<TAB>seqA=...<TAB>posA=...

    This reader also accepts a conventional headered TSV if one is encountered.
    """
    records: Dict[str, MotifRecord] = {}
    with path.open("r", encoding="utf-8", newline=None) as handle:
        first_data_tokens: Optional[List[str]] = None
        pos = handle.tell()
        for raw in handle:
            if raw.startswith("#") or not raw.strip():
                pos = handle.tell()
                continue
            first_data_tokens = raw.rstrip("\n\r").split("\t")
            break
        if first_data_tokens is None:
            return records
        handle.seek(pos)

        # Headered TSV heuristic: first row contains column labels such as seq_id, posA, seqA.
        is_headered = (
            any(tok in {"seq_id", "id", "name", "query", "target"} for tok in first_data_tokens)
            or ("posA" in first_data_tokens and "seqA" in first_data_tokens)
        ) and not any("=" in tok for tok in first_data_tokens[1:])

        if is_headered:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                return records
            id_column = None
            for candidate in ("seq_id", "id", "name", "query", "target"):
                if candidate in reader.fieldnames:
                    id_column = candidate
                    break
            if id_column is None:
                id_column = reader.fieldnames[0]
            for row in reader:
                if not row:
                    continue
                seq_id = (row.get(id_column) or "").strip()
                if not seq_id:
                    continue
                fields = {k: (v or "").strip() for k, v in row.items() if k is not None}
                if seq_id in records:
                    # Keep the first by default; duplicate motif rows are uncommon for this use case.
                    continue
                records[seq_id] = MotifRecord(seq_id=seq_id, fields=fields)
        else:
            for raw in handle:
                raw = raw.rstrip("\n\r")
                if not raw or raw.startswith("#"):
                    continue
                tokens = raw.split("\t")
                if not tokens:
                    continue
                if "=" in tokens[0]:
                    fields = parse_key_value_tokens(tokens)
                    seq_id = fields.get("seq_id") or fields.get("id") or fields.get("query") or ""
                    if not seq_id:
                        continue
                else:
                    seq_id = tokens[0].strip()
                    fields = parse_key_value_tokens(tokens[1:])
                if not seq_id:
                    continue
                if seq_id in records:
                    continue
                records[seq_id] = MotifRecord(seq_id=seq_id, fields=fields)
    return records


def parse_int_field(fields: Dict[str, str], keys: Sequence[str]) -> Tuple[Optional[int], str]:
    for key in keys:
        value = fields.get(key, "")
        if value == "" or value == "." or value.lower() == "nan":
            continue
        try:
            return int(float(value)), key
        except ValueError:
            continue
    return None, ""


def parse_str_field(fields: Dict[str, str], keys: Sequence[str]) -> Tuple[str, str]:
    for key in keys:
        value = fields.get(key, "")
        if value and value != "." and value.lower() != "nan":
            return value.strip().upper(), key
    return "", ""


def choose_motif_site(
    motif: str,
    record: MotifRecord,
    fasta_record: FastaRecord,
    motif_center_mode: str,
    position_tolerance: int,
    skip_writer: SkipWriter,
) -> Optional[MotifSite]:
    pos, pos_key = parse_int_field(record.fields, DEFAULT_MOTIF_POS_KEYS[motif])
    motif_seq, seq_key = parse_str_field(record.fields, DEFAULT_MOTIF_SEQ_KEYS[motif])
    if pos is None:
        skip_writer.add(record.seq_id, f"missing_pos{motif}", ",".join(DEFAULT_MOTIF_POS_KEYS[motif]))
        return None
    if not motif_seq:
        skip_writer.add(record.seq_id, f"missing_seq{motif}", ",".join(DEFAULT_MOTIF_SEQ_KEYS[motif]))
        return None
    if pos < 1 or pos > len(fasta_record.seq):
        skip_writer.add(
            record.seq_id,
            f"pos{motif}_out_of_range",
            f"pos={pos};len={len(fasta_record.seq)}",
        )
        return None

    motif_len = len(motif_seq)
    start = pos
    end = pos + motif_len - 1
    verified = False

    if end <= len(fasta_record.seq):
        observed = fasta_record.seq[start - 1 : end]
        if observed == motif_seq:
            verified = True
        elif position_tolerance > 0:
            lo = max(1, start - position_tolerance)
            hi = min(len(fasta_record.seq) - motif_len + 1, start + position_tolerance)
            for candidate_start in range(lo, hi + 1):
                candidate_end = candidate_start + motif_len - 1
                if fasta_record.seq[candidate_start - 1 : candidate_end] == motif_seq:
                    start = candidate_start
                    end = candidate_end
                    verified = True
                    break
    elif position_tolerance > 0:
        lo = max(1, start - position_tolerance)
        hi = min(len(fasta_record.seq) - motif_len + 1, start + position_tolerance)
        for candidate_start in range(lo, hi + 1):
            candidate_end = candidate_start + motif_len - 1
            if fasta_record.seq[candidate_start - 1 : candidate_end] == motif_seq:
                start = candidate_start
                end = candidate_end
                verified = True
                break

    if not verified and not motif_seq.startswith("X"):
        # Do not skip by default; many motif annotation tools report broader or approximate motif strings.
        # The manifest records verified=false so the user can filter later if desired.
        pass

    if motif_center_mode == "start":
        center = start
    elif motif_center_mode == "center":
        center = (start + end) // 2
    else:
        die(f"Unsupported motif center mode: {motif_center_mode}")

    return MotifSite(
        motif=motif,
        start_1based=start,
        end_1based=end,
        center_1based=center,
        seq=motif_seq,
        source_pos_key=pos_key,
        source_seq_key=seq_key,
        verified=verified,
    )


def parse_gff_attributes(attr_text: str) -> Dict[str, str]:
    attrs: Dict[str, str] = {}
    for part in attr_text.split(";"):
        if not part:
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            attrs[unquote(key)] = unquote(value)
        else:
            attrs[unquote(part)] = ""
    return attrs


def parse_score(raw_score: str) -> Optional[float]:
    if raw_score == "." or raw_score == "":
        return None
    try:
        return float(raw_score)
    except ValueError:
        return None


def read_palmsite_gff(
    path: Path,
    feature_types: Sequence[str],
    mode: str,
) -> Dict[str, SpanRecord]:
    feature_type_set = set(feature_types)
    by_id: Dict[str, List[SpanRecord]] = defaultdict(list)

    with path.open("r", encoding="utf-8", newline=None) as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.rstrip("\n\r")
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split("\t")
            if len(parts) < 9:
                print(f"WARNING: skipping malformed GFF line {line_no}: expected 9 columns", file=sys.stderr)
                continue
            seq_id, source, feature_type, start_s, end_s, score_s, strand, phase, attrs_s = parts[:9]
            if feature_type_set and feature_type not in feature_type_set:
                continue
            try:
                start = int(start_s)
                end = int(end_s)
            except ValueError:
                print(f"WARNING: skipping malformed GFF line {line_no}: non-integer start/end", file=sys.stderr)
                continue
            if start > end:
                start, end = end, start
            attrs = parse_gff_attributes(attrs_s)
            score = parse_score(score_s)
            span = SpanRecord(
                seq_id=seq_id,
                start_1based=start,
                end_1based=end,
                source="palmsite_gff",
                feature_type=feature_type,
                score=score,
                raw_score=score_s,
                attributes=attrs_s,
            )
            by_id[seq_id].append(span)

            # Also index by ID and Chunk base if present, useful if seqid differs from FASTA id.
            attr_id = attrs.get("ID", "")
            if attr_id and attr_id != seq_id:
                by_id[attr_id].append(span)
            chunk = attrs.get("Chunk", "")
            if chunk:
                chunk_base = chunk.split("|", 1)[0]
                if chunk_base and chunk_base != seq_id:
                    by_id[chunk_base].append(span)

    selected: Dict[str, SpanRecord] = {}
    for seq_id, spans in by_id.items():
        if not spans:
            continue
        if mode == "first":
            selected[seq_id] = spans[0]
        elif mode == "longest":
            selected[seq_id] = max(spans, key=lambda x: (x.end_1based - x.start_1based + 1, x.score if x.score is not None else -1e300))
        elif mode == "highest_score":
            selected[seq_id] = max(spans, key=lambda x: (x.score if x.score is not None else -1e300, x.end_1based - x.start_1based + 1))
        elif mode == "union":
            start = min(x.start_1based for x in spans)
            end = max(x.end_1based for x in spans)
            best_score = max((x.score for x in spans if x.score is not None), default=None)
            feature_types_joined = ",".join(sorted(set(x.feature_type for x in spans)))
            selected[seq_id] = SpanRecord(
                seq_id=seq_id,
                start_1based=start,
                end_1based=end,
                source="palmsite_gff_union",
                feature_type=feature_types_joined,
                score=best_score,
                raw_score="" if best_score is None else str(best_score),
                attributes=f"n_features={len(spans)}",
            )
        else:
            die(f"Unsupported --gff-span-mode: {mode}")
    return selected


def get_palm_annot_span(record: MotifRecord) -> Optional[SpanRecord]:
    for lo_key, hi_key in DEFAULT_PALM_ANNOT_SPAN_KEY_PAIRS:
        lo, _ = parse_int_field(record.fields, (lo_key,))
        hi, _ = parse_int_field(record.fields, (hi_key,))
        if lo is not None and hi is not None:
            if lo > hi:
                lo, hi = hi, lo
            return SpanRecord(
                seq_id=record.seq_id,
                start_1based=lo,
                end_1based=hi,
                source=f"palm_annot:{lo_key}-{hi_key}",
            )
    return None


def clamp_window_around_center(center_1based: int, window_size: int, seq_len: int) -> Tuple[int, int]:
    if window_size > seq_len:
        return 1, seq_len
    left = (window_size - 1) // 2
    start = center_1based - left
    end = start + window_size - 1
    if start < 1:
        start = 1
        end = window_size
    if end > seq_len:
        end = seq_len
        start = seq_len - window_size + 1
    return start, end


def enumerate_starts_in_interval(interval_start: int, interval_end: int, window_size: int) -> List[int]:
    if interval_start > interval_end:
        return []
    if interval_end - interval_start + 1 < window_size:
        return []
    return list(range(interval_start, interval_end - window_size + 2))


def choose_random_in_span_window(
    span: SpanRecord,
    seq_len: int,
    window_size: int,
    rng: random.Random,
) -> Optional[Tuple[int, int]]:
    start = max(1, span.start_1based)
    end = min(seq_len, span.end_1based)
    starts = enumerate_starts_in_interval(start, end, window_size)
    if not starts:
        return None
    wstart = rng.choice(starts)
    return wstart, wstart + window_size - 1


def choose_random_outside_span_window(
    span: SpanRecord,
    seq_len: int,
    window_size: int,
    margin: int,
    rng: random.Random,
) -> Optional[Tuple[int, int]]:
    span_start = max(1, span.start_1based)
    span_end = min(seq_len, span.end_1based)

    left_end = span_start - margin - 1
    right_start = span_end + margin + 1

    starts: List[int] = []
    starts.extend(enumerate_starts_in_interval(1, left_end, window_size))
    starts.extend(enumerate_starts_in_interval(right_start, seq_len, window_size))
    if not starts:
        return None
    wstart = rng.choice(starts)
    return wstart, wstart + window_size - 1


def choose_random_anywhere_window(seq_len: int, window_size: int, rng: random.Random) -> Optional[Tuple[int, int]]:
    if seq_len < window_size:
        return None
    wstart = rng.randint(1, seq_len - window_size + 1)
    return wstart, wstart + window_size - 1


def enumerate_control_starts(
    control: str,
    span: Optional[SpanRecord],
    seq_len: int,
    window_size: int,
    margin: int,
) -> List[int]:
    """Return all valid 1-based start positions for a control window.

    The returned starts are used for sampling across --control-replicates.
    The caller samples unique starts and prefers non-overlapping windows when
    enough valid positions exist.
    """
    if control == "random_anywhere":
        return enumerate_starts_in_interval(1, seq_len, window_size)

    if span is None:
        return []

    span_start = max(1, span.start_1based)
    span_end = min(seq_len, span.end_1based)

    if control == "random_in_span":
        return enumerate_starts_in_interval(span_start, span_end, window_size)

    if control == "random_outside_span":
        left_end = span_start - margin - 1
        right_start = span_end + margin + 1
        starts: List[int] = []
        starts.extend(enumerate_starts_in_interval(1, left_end, window_size))
        starts.extend(enumerate_starts_in_interval(right_start, seq_len, window_size))
        return starts

    die(f"Unsupported control: {control}")
    return []


def mutate_window(
    seq: str,
    start_1based: int,
    end_1based: int,
    mutation_rate: float,
    mutation_mode: str,
    alphabet: str,
    rng: random.Random,
    mutate_noncanonical: bool,
) -> PerturbationResult:
    seq_chars = list(seq)
    start0 = start_1based - 1
    end0_exclusive = end_1based
    window_positions = list(range(start0, end0_exclusive))

    eligible_positions = []
    alphabet_set = set(alphabet)
    for idx in window_positions:
        aa = seq_chars[idx].upper()
        if aa in alphabet_set:
            eligible_positions.append(idx)
        elif mutate_noncanonical:
            eligible_positions.append(idx)

    if not eligible_positions or mutation_rate <= 0.0:
        original_window = seq[start0:end0_exclusive]
        return PerturbationResult(
            perturbed_seq=seq,
            original_window=original_window,
            perturbed_window=original_window,
            mutated_positions_1based=[],
            mutated_from=[],
            mutated_to=[],
        )

    if mutation_mode == "exact_fraction":
        n_mutate = int(round(len(eligible_positions) * mutation_rate))
        if mutation_rate > 0.0:
            n_mutate = max(1, n_mutate)
        n_mutate = min(n_mutate, len(eligible_positions))
        chosen = sorted(rng.sample(eligible_positions, n_mutate))
    elif mutation_mode == "bernoulli":
        chosen = sorted(idx for idx in eligible_positions if rng.random() < mutation_rate)
        if not chosen and mutation_rate > 0.0:
            chosen = [rng.choice(eligible_positions)]
    else:
        die(f"Unsupported mutation mode: {mutation_mode}")

    mutated_positions_1based: List[int] = []
    mutated_from: List[str] = []
    mutated_to: List[str] = []
    for idx in chosen:
        old = seq_chars[idx].upper()
        choices = [aa for aa in alphabet if aa != old]
        if not choices:
            continue
        new = rng.choice(choices)
        seq_chars[idx] = new
        mutated_positions_1based.append(idx + 1)
        mutated_from.append(old)
        mutated_to.append(new)

    perturbed = "".join(seq_chars)
    return PerturbationResult(
        perturbed_seq=perturbed,
        original_window=seq[start0:end0_exclusive],
        perturbed_window=perturbed[start0:end0_exclusive],
        mutated_positions_1based=mutated_positions_1based,
        mutated_from=mutated_from,
        mutated_to=mutated_to,
    )


def deterministic_child_seed(master_seed: int, *parts: object) -> int:
    text = "|".join([str(master_seed)] + [str(p) for p in parts])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def resolve_span(
    seq_id: str,
    motif_record: MotifRecord,
    palmsite_spans: Dict[str, SpanRecord],
    span_source: str,
) -> Optional[SpanRecord]:
    if span_source == "palmsite_gff":
        return palmsite_spans.get(seq_id)
    if span_source == "palm_annot":
        return get_palm_annot_span(motif_record)
    if span_source == "auto":
        return palmsite_spans.get(seq_id) or get_palm_annot_span(motif_record)
    die(f"Unsupported --span-source: {span_source}")
    return None


def make_targets_for_record(
    fasta_record: FastaRecord,
    motif_record: MotifRecord,
    motifs: Sequence[str],
    require_motifs: Sequence[str],
    window_sizes: Sequence[int],
    motif_center_mode: str,
    position_tolerance: int,
    controls: Sequence[str],
    palmsite_spans: Dict[str, SpanRecord],
    span_source: str,
    outside_span_margin: int,
    control_replicates: int,
    rng: random.Random,
    skip_writer: SkipWriter,
) -> List[Tuple[int, WindowTarget]]:
    seq_id = fasta_record.seq_id
    seq_len = len(fasta_record.seq)

    motif_sites: Dict[str, MotifSite] = {}
    for motif in sorted(set(list(motifs) + list(require_motifs))):
        site = choose_motif_site(
            motif=motif,
            record=motif_record,
            fasta_record=fasta_record,
            motif_center_mode=motif_center_mode,
            position_tolerance=position_tolerance,
            skip_writer=skip_writer,
        )
        if site is not None:
            motif_sites[motif] = site

    missing_required = [m for m in require_motifs if m not in motif_sites]
    if missing_required:
        skip_writer.add(seq_id, "missing_required_motifs", ",".join(missing_required))
        return []

    span = resolve_span(seq_id, motif_record, palmsite_spans, span_source)
    if controls and span is None:
        skip_writer.add(seq_id, "missing_span_for_controls", f"span_source={span_source}")

    targets: List[Tuple[int, WindowTarget]] = []

    for window_size in window_sizes:
        if window_size > seq_len:
            skip_writer.add(seq_id, "window_larger_than_sequence", f"window_size={window_size};len={seq_len}")
            continue
        for motif in motifs:
            site = motif_sites.get(motif)
            if site is None:
                # choose_motif_site already wrote detailed skip reason.
                continue
            wstart, wend = clamp_window_around_center(site.center_1based, window_size, seq_len)
            targets.append(
                (
                    window_size,
                    WindowTarget(
                        target_name=f"motif_{motif}",
                        motif=motif,
                        window_start_1based=wstart,
                        window_end_1based=wend,
                        motif_start_1based=str(site.start_1based),
                        motif_end_1based=str(site.end_1based),
                        motif_center_1based=str(site.center_1based),
                        motif_seq=site.seq,
                        span=span,
                    ),
                )
            )

        for control in controls:
            if span is None and control != "random_anywhere":
                continue

            starts = enumerate_control_starts(
                control=control,
                span=span,
                seq_len=seq_len,
                window_size=window_size,
                margin=outside_span_margin,
            )
            if not starts:
                span_detail = "NA" if span is None else f"{span.start_1based}-{span.end_1based}"
                skip_writer.add(
                    seq_id,
                    f"no_valid_{control}_window",
                    f"window_size={window_size};span={span_detail};len={seq_len}",
                )
                continue

            rng.shuffle(starts)

            # Prefer non-overlapping control windows. If there are not enough
            # non-overlapping windows, fill the remaining replicates with unique
            # starts, which can overlap by residues but never duplicate the exact
            # same window.
            selected_starts: List[int] = []
            for candidate_start in starts:
                candidate_end = candidate_start + window_size - 1
                overlaps_existing = any(
                    not (candidate_end < chosen_start or candidate_start > chosen_start + window_size - 1)
                    for chosen_start in selected_starts
                )
                if not overlaps_existing:
                    selected_starts.append(candidate_start)
                    if len(selected_starts) >= control_replicates:
                        break

            nonoverlap_count = len(selected_starts)
            if len(selected_starts) < control_replicates:
                selected_set = set(selected_starts)
                for candidate_start in starts:
                    if candidate_start in selected_set:
                        continue
                    selected_starts.append(candidate_start)
                    selected_set.add(candidate_start)
                    if len(selected_starts) >= control_replicates:
                        break

            span_detail = "NA" if span is None else f"{span.start_1based}-{span.end_1based}"
            if len(starts) < control_replicates:
                skip_writer.add(
                    seq_id,
                    f"insufficient_unique_{control}_windows",
                    f"window_size={window_size};requested={control_replicates};available_unique_starts={len(starts)};span={span_detail};len={seq_len}",
                )
            elif nonoverlap_count < control_replicates:
                skip_writer.add(
                    seq_id,
                    f"insufficient_nonoverlapping_{control}_windows",
                    f"window_size={window_size};requested={control_replicates};available_nonoverlap={nonoverlap_count};available_unique_starts={len(starts)};span={span_detail};len={seq_len}",
                )

            for control_window_rep, wstart in enumerate(selected_starts, start=1):
                wend = wstart + window_size - 1
                targets.append(
                    (
                        window_size,
                        WindowTarget(
                            target_name=control,
                            motif="",
                            window_start_1based=wstart,
                            window_end_1based=wend,
                            motif_start_1based="",
                            motif_end_1based="",
                            motif_center_1based="",
                            motif_seq="",
                            span=span,
                            control_window_replicate=str(control_window_rep),
                        ),
                    )
                )

    return targets


def write_tsv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate motif-centered and span-control amino-acid substitution perturbations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fasta", required=True, type=Path, help="Input protein FASTA.")
    parser.add_argument("--palm-annot", required=True, type=Path, help="palm_annot TSV with motif positions.")
    parser.add_argument("--palmsite-gff", type=Path, default=None, help="PalmSite GFF/GFF3 from original sequences, used for span controls.")
    parser.add_argument("--out-prefix", required=True, type=Path, help="Output prefix.")

    parser.add_argument("--motifs", default="A,B,C", help="Comma-separated motifs to perturb.")
    parser.add_argument("--require-motifs", default="", help="Comma-separated motifs required for a sequence to be used. Example: A,B,C")
    parser.add_argument("--window-size", default="7", help="One or more comma-separated fixed window sizes. Example: 5,7,11")
    parser.add_argument("--motif-center-mode", choices=("center", "start"), default="center", help="How to center motif windows from palm_annot motif start/sequence.")
    parser.add_argument("--position-tolerance", type=int, default=0, help="Search this many residues around motif position if motif sequence does not verify exactly.")

    parser.add_argument("--mutation-rates", default="1.0", help="Comma-separated mutation rates in [0,1].")
    parser.add_argument("--mutation-mode", choices=("exact_fraction", "bernoulli"), default="exact_fraction", help="How to choose mutated sites within the window.")
    parser.add_argument("--replicates", type=int, default=5, help="Replicates for motif-centered targets.")
    parser.add_argument("--control-replicates", type=int, default=None, help="Number of independently sampled random control windows per sequence/control/window-size. Defaults to --replicates. Windows use unique starts and are non-overlapping when enough valid positions exist.")
    parser.add_argument("--alphabet", default=CANONICAL_AA, help="Residue alphabet for substitutions.")
    parser.add_argument("--mutate-noncanonical", action="store_true", help="Allow noncanonical residues in target windows to be replaced by canonical residues.")

    parser.add_argument(
        "--controls",
        default="",
        help="Comma-separated controls: random_in_span,random_outside_span,random_anywhere.",
    )
    parser.add_argument(
        "--span-source",
        choices=("palmsite_gff", "palm_annot", "auto"),
        default="auto",
        help="Span source for random_in_span/random_outside_span controls.",
    )
    parser.add_argument("--outside-span-margin", type=int, default=0, help="Minimum residue gap between PalmSite span and random_outside_span control window.")
    parser.add_argument(
        "--gff-feature-types",
        default="",
        help="Comma-separated GFF feature types to use as PalmSite spans. Empty means all feature types.",
    )
    parser.add_argument(
        "--gff-span-mode",
        choices=("highest_score", "longest", "first", "union"),
        default="highest_score",
        help="How to choose a span if multiple GFF features exist for one sequence.",
    )

    parser.add_argument("--seed", type=int, default=1, help="Master random seed.")
    parser.add_argument("--max-records", type=int, default=0, help="Process at most this many FASTA records after filtering. 0 means no limit.")
    parser.add_argument("--include-original", action="store_true", help="Also write original sequences to the output FASTA with perturbation_class=original.")
    parser.add_argument("--strict-gff-span", action="store_true", help="Skip control generation when PalmSite GFF span is absent, even with --span-source auto.")

    args = parser.parse_args(argv)

    if args.replicates < 1:
        die("--replicates must be >= 1")
    if args.control_replicates is not None and args.control_replicates < 1:
        die("--control-replicates must be >= 1")
    if args.position_tolerance < 0:
        die("--position-tolerance must be >= 0")
    if args.outside_span_margin < 0:
        die("--outside-span-margin must be >= 0")
    if not args.fasta.exists():
        die(f"FASTA file does not exist: {args.fasta}")
    if not args.palm_annot.exists():
        die(f"palm_annot file does not exist: {args.palm_annot}")
    if args.palmsite_gff is not None and not args.palmsite_gff.exists():
        die(f"PalmSite GFF file does not exist: {args.palmsite_gff}")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    motifs = parse_csv_list(args.motifs)
    require_motifs = parse_csv_list(args.require_motifs)
    controls = parse_csv_list(args.controls)
    window_sizes = parse_int_list(args.window_size)
    mutation_rates = parse_float_list(args.mutation_rates)
    gff_feature_types = parse_csv_list(args.gff_feature_types)
    control_replicates = args.control_replicates if args.control_replicates is not None else args.replicates

    valid_motifs = {"A", "B", "C"}
    for motif in motifs + require_motifs:
        if motif not in valid_motifs:
            die(f"Unsupported motif {motif!r}; supported motifs are A,B,C")
    valid_controls = {"random_in_span", "random_outside_span", "random_anywhere"}
    for control in controls:
        if control not in valid_controls:
            die(f"Unsupported control {control!r}; supported controls are {','.join(sorted(valid_controls))}")
    if ("random_in_span" in controls or "random_outside_span" in controls) and args.span_source == "palmsite_gff" and args.palmsite_gff is None:
        die("--palmsite-gff is required when --span-source palmsite_gff is used with span controls")
    if args.strict_gff_span and args.palmsite_gff is None:
        die("--strict-gff-span requires --palmsite-gff")
    if args.strict_gff_span:
        args.span_source = "palmsite_gff"

    print(f"Reading FASTA: {args.fasta}", file=sys.stderr)
    fasta_records = read_fasta(args.fasta)
    print(f"  FASTA records: {len(fasta_records):,}", file=sys.stderr)

    print(f"Reading palm_annot TSV: {args.palm_annot}", file=sys.stderr)
    motif_records = read_palm_annot(args.palm_annot)
    print(f"  palm_annot records: {len(motif_records):,}", file=sys.stderr)

    palmsite_spans: Dict[str, SpanRecord] = {}
    if args.palmsite_gff is not None:
        print(f"Reading PalmSite GFF: {args.palmsite_gff}", file=sys.stderr)
        palmsite_spans = read_palmsite_gff(args.palmsite_gff, gff_feature_types, args.gff_span_mode)
        print(f"  GFF span keys: {len(palmsite_spans):,}", file=sys.stderr)

    out_fasta = args.out_prefix.with_suffix(".perturbed.fasta")
    out_manifest = args.out_prefix.with_suffix(".manifest.tsv")
    out_skipped = args.out_prefix.with_suffix(".skipped.tsv")
    out_summary = args.out_prefix.with_suffix(".summary.tsv")

    skip_writer = SkipWriter()
    manifest_rows: List[Dict[str, object]] = []
    summary_counter: Counter[str] = Counter()
    processed = 0
    generated = 0

    with out_fasta.open("w", encoding="utf-8", newline="\n") as fasta_out:
        for seq_id, fasta_record in fasta_records.items():
            motif_record = motif_records.get(seq_id)
            if motif_record is None:
                skip_writer.add(seq_id, "missing_palm_annot_record", "")
                continue

            record_rng = random.Random(deterministic_child_seed(args.seed, seq_id, "targets"))
            targets = make_targets_for_record(
                fasta_record=fasta_record,
                motif_record=motif_record,
                motifs=motifs,
                require_motifs=require_motifs,
                window_sizes=window_sizes,
                motif_center_mode=args.motif_center_mode,
                position_tolerance=args.position_tolerance,
                controls=controls,
                palmsite_spans=palmsite_spans,
                span_source=args.span_source,
                outside_span_margin=args.outside_span_margin,
                control_replicates=control_replicates,
                rng=record_rng,
                skip_writer=skip_writer,
            )
            if not targets:
                continue

            processed += 1
            if args.include_original:
                original_id = f"{safe_token(seq_id)}|perturb=original"
                fasta_out.write(f">{original_id} original_id={seq_id} perturbation_class=original\n")
                fasta_out.write(wrap_fasta(fasta_record.seq) + "\n")
                manifest_rows.append(
                    {
                        "perturb_id": original_id,
                        "original_id": seq_id,
                        "original_header": fasta_record.header,
                        "perturbation_class": "original",
                        "target_name": "original",
                        "motif": "",
                        "window_size": "",
                        "mutation_rate": 0.0,
                        "replicate": 0,
                        "n_mutated": 0,
                        "mutated_positions_1based_csv": "",
                        "mutated_from_csv": "",
                        "mutated_to_csv": "",
                        "seq_len": len(fasta_record.seq),
                    }
                )

            for window_size, target in targets:
                is_control = target.target_name.startswith("random_")
                # For controls, --control-replicates was already applied by sampling
                # independent window starts in make_targets_for_record(). Each control
                # target therefore gets exactly one substitution realization per rate.
                n_reps = 1 if is_control else args.replicates
                perturbation_class = "control" if is_control else "motif_centered"

                for mutation_rate in mutation_rates:
                    for rep in range(1, n_reps + 1):
                        display_rep = int(target.control_window_replicate) if is_control and target.control_window_replicate else rep
                        rng = random.Random(
                            deterministic_child_seed(
                                args.seed,
                                seq_id,
                                target.target_name,
                                window_size,
                                target.window_start_1based,
                                target.window_end_1based,
                                mutation_rate,
                                display_rep,
                            )
                        )
                        result = mutate_window(
                            seq=fasta_record.seq,
                            start_1based=target.window_start_1based,
                            end_1based=target.window_end_1based,
                            mutation_rate=mutation_rate,
                            mutation_mode=args.mutation_mode,
                            alphabet=args.alphabet,
                            rng=rng,
                            mutate_noncanonical=args.mutate_noncanonical,
                        )
                        rate_token = str(mutation_rate).replace(".", "p")
                        perturb_id = (
                            f"{safe_token(seq_id)}|target={safe_token(target.target_name)}"
                            f"|w={window_size}|rate={rate_token}|rep={display_rep}"
                        )
                        header = (
                            f">{perturb_id} original_id={seq_id} "
                            f"perturbation_class={perturbation_class} "
                            f"target_name={target.target_name} motif={target.motif or 'NA'} "
                            f"window={target.window_start_1based}-{target.window_end_1based} "
                            f"mutation_rate={mutation_rate} replicate={display_rep}"
                        )
                        fasta_out.write(header + "\n")
                        fasta_out.write(wrap_fasta(result.perturbed_seq) + "\n")

                        span = target.span
                        manifest_rows.append(
                            {
                                "perturb_id": perturb_id,
                                "original_id": seq_id,
                                "original_header": fasta_record.header,
                                "perturbation_class": perturbation_class,
                                "target_name": target.target_name,
                                "motif": target.motif,
                                "motif_start_1based": target.motif_start_1based,
                                "motif_end_1based": target.motif_end_1based,
                                "motif_center_1based": target.motif_center_1based,
                                "motif_seq": target.motif_seq,
                                "window_size": window_size,
                                "window_start_1based": target.window_start_1based,
                                "window_end_1based": target.window_end_1based,
                                "mutation_rate": mutation_rate,
                                "mutation_mode": args.mutation_mode,
                                "replicate": display_rep,
                                "control_window_replicate": target.control_window_replicate,
                                "n_mutated": len(result.mutated_positions_1based),
                                "mutated_positions_1based_csv": ",".join(map(str, result.mutated_positions_1based)),
                                "mutated_from_csv": ",".join(result.mutated_from),
                                "mutated_to_csv": ",".join(result.mutated_to),
                                "original_window": result.original_window,
                                "perturbed_window": result.perturbed_window,
                                "seq_len": len(fasta_record.seq),
                                "control_span_source": span.source if span is not None else "",
                                "control_span_start_1based": span.start_1based if span is not None else "",
                                "control_span_end_1based": span.end_1based if span is not None else "",
                                "control_span_feature_type": span.feature_type if span is not None else "",
                                "control_span_score": span.score if span is not None else "",
                                "control_span_raw_score": span.raw_score if span is not None else "",
                                "control_span_attributes": span.attributes if span is not None else "",
                            }
                        )
                        generated += 1
                        summary_counter[f"generated:{perturbation_class}:{target.target_name}:w{window_size}:rate{mutation_rate}"] += 1

            if args.max_records > 0 and processed >= args.max_records:
                break

    manifest_fields = [
        "perturb_id",
        "original_id",
        "original_header",
        "perturbation_class",
        "target_name",
        "motif",
        "motif_start_1based",
        "motif_end_1based",
        "motif_center_1based",
        "motif_seq",
        "window_size",
        "window_start_1based",
        "window_end_1based",
        "mutation_rate",
        "mutation_mode",
        "replicate",
        "control_window_replicate",
        "n_mutated",
        "mutated_positions_1based_csv",
        "mutated_from_csv",
        "mutated_to_csv",
        "original_window",
        "perturbed_window",
        "seq_len",
        "control_span_source",
        "control_span_start_1based",
        "control_span_end_1based",
        "control_span_feature_type",
        "control_span_score",
        "control_span_raw_score",
        "control_span_attributes",
    ]
    write_tsv(out_manifest, manifest_rows, manifest_fields)

    skipped_fields = ["seq_id", "reason", "detail"]
    write_tsv(out_skipped, skip_writer.rows, skipped_fields)

    summary_rows: List[Dict[str, object]] = [
        {"key": "fasta_records", "value": len(fasta_records)},
        {"key": "palm_annot_records", "value": len(motif_records)},
        {"key": "palmsite_gff_span_keys", "value": len(palmsite_spans)},
        {"key": "processed_records", "value": processed},
        {"key": "generated_perturbed_sequences", "value": generated},
        {"key": "skipped_events", "value": len(skip_writer.rows)},
        {"key": "motifs", "value": ",".join(motifs)},
        {"key": "require_motifs", "value": ",".join(require_motifs)},
        {"key": "controls", "value": ",".join(controls)},
        {"key": "window_sizes", "value": ",".join(map(str, window_sizes))},
        {"key": "mutation_rates", "value": ",".join(map(str, mutation_rates))},
        {"key": "span_source", "value": args.span_source},
        {"key": "gff_span_mode", "value": args.gff_span_mode},
        {"key": "outside_span_margin", "value": args.outside_span_margin},
        {"key": "control_replicates", "value": control_replicates},
        {"key": "control_window_sampling", "value": "unique_starts_prefer_nonoverlap"},
    ]
    for key, value in sorted(summary_counter.items()):
        summary_rows.append({"key": key, "value": value})
    for reason, value in sorted(skip_writer.counter.items()):
        summary_rows.append({"key": f"skipped:{reason}", "value": value})
    write_tsv(out_summary, summary_rows, ["key", "value"])

    print("Done.", file=sys.stderr)
    print(f"  Perturbed FASTA: {out_fasta}", file=sys.stderr)
    print(f"  Manifest:        {out_manifest}", file=sys.stderr)
    print(f"  Skipped:         {out_skipped}", file=sys.stderr)
    print(f"  Summary:         {out_summary}", file=sys.stderr)
    print(f"  Generated:       {generated:,} perturbed sequences", file=sys.stderr)
    print(f"  Processed:       {processed:,} original sequences", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

