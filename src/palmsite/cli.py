from __future__ import annotations

import os

# Silence HuggingFace tokenizers parallelism warning in forked contexts (if user has not set it).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import sys
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import click

from .sanitize import sanitize_and_merge_fastas
from .embed_shim import embed_fastas_to_h5
from .infer_simple import predict_to_gff
from . import __version__



def _ts() -> str:
    """Local timestamp for CLI progress messages."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _log(msg: str, *, quiet: bool) -> None:
    """Emit a timestamped progress/log line to stderr (unless quiet)."""
    if quiet:
        return
    click.echo(f"{_ts()} {msg}", err=True)

def _iter_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    """Minimal FASTA iterator (expects already-sanitized sequences)."""
    sid: Optional[str] = None
    buf: List[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">" ):
                if sid is not None:
                    yield sid, "".join(buf)
                hd = s[1:].strip()
                sid = hd.split()[0] if hd else "unnamed"
                buf = []
            else:
                buf.append(s)
    if sid is not None:
        yield sid, "".join(buf)


def _write_fasta(records: List[Tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as out:
        for sid, seq in records:
            out.write(f">{sid}\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i+60] + "\n")


def _iter_microbatches(
    fasta_path: Path,
    max_seqs: Optional[int],
    max_tokens: Optional[int],
) -> Iterator[Tuple[List[Tuple[str, str]], int]]:
    """
    Yield (records, approx_tokens) where approx_tokens = sum(len(seq)+2).

    Batching is sequence-aware: all chunks of a given sequence remain in the same micro-batch
    (the chunking happens later inside the embedding engine).
    """
    batch: List[Tuple[str, str]] = []
    tok_sum = 0

    for sid, seq in _iter_fasta(fasta_path):
        seq_tok = len(seq) + 2

        # Hard sequence-count cap
        if max_seqs is not None and max_seqs > 0 and len(batch) >= max_seqs:
            yield batch, tok_sum
            batch = []
            tok_sum = 0

        # Token cap (keep at least 1 seq per batch)
        if (
            max_tokens is not None
            and max_tokens > 0
            and batch
            and (tok_sum + seq_tok) > max_tokens
        ):
            yield batch, tok_sum
            batch = []
            tok_sum = 0

        batch.append((sid, seq))
        tok_sum += seq_tok

    if batch:
        yield batch, tok_sum


class _AttnJSONWriter:
    """Incremental JSON object writer for --attn-json in streaming mode."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fp = open(path, "w", encoding="utf-8")
        self._fp.write("{\n")
        self._first = True

    def write_dict(self, d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if not self._first:
                self._fp.write(",\n")
            else:
                self._first = False
            self._fp.write(json.dumps(k))
            self._fp.write(": ")
            self._fp.write(json.dumps(v))
        self._fp.flush()

    def close(self) -> None:
        if self._fp:
            self._fp.write("\n}\n")
            self._fp.close()
            self._fp = None


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(__version__, prog_name="palmsite")
@click.option(
    "-o", "--gff-out", default=None,
    help="Write GFF3 to this path; default: stdout if omitted",
)
@click.option(
    "-p", "--min-p", default=0.5, show_default=True,
    help="Minimum probability threshold for output",
)
@click.option(
    "-b", "--backbone", default="600m", show_default=True,
    type=click.Choice(["300m", "600m", "6b"], case_sensitive=False),
    help="Embedding backbone (300m/600m local; 6b via Forge)",
)
@click.option(
    "-m", "--model-id", default=None,
    help="Hugging Face model repo for PalmSite weights (default: PALMSITE_MODEL_ID or ryota-sugimoto/palmsite)",
)
@click.option(
    "--revision", default=None, hidden=True,
    help="Optional HF model revision (or set PALMSITE_MODEL_REV)",
)
@click.option(
    "-d", "--device", default="auto", show_default=True,
    type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
    help="Device for local ESM-C embedding (ignored for 6b)",
)
@click.option(
    "-k", "--token", default=None,
    help="ESM Forge token for backbone=6b (or set ESM_FORGE_TOKEN)",
)
@click.option(
    "-t", "--tmp-dir", default=None,
    help="Temporary directory (default: auto-created and deleted)",
)
@click.option(
    "-q", "--quiet", is_flag=True, help="Suppress progress logs"
)
@click.option(
    "-v", "--verbose", is_flag=True, help="Verbose logs (overrides quiet)"
)
@click.option(
    "--keep-tmp", is_flag=True,
    help="Keep temporary files (sanitized FASTA and per-micro-batch embeddings)",
)
@click.option(
    "--attn-json", default=None,
    help="Write per-residue attention + span details as JSON (can be large)",
)
@click.option(
    "--micro-batch-seqs", default=None, type=int,
    help="Process input in micro-batches of N sequences (bounds temp embedding size)",
)
@click.option(
    "--micro-batch-tokens", default=None, type=int,
    help="Process input in micro-batches capped at ~N tokens (sum(len(seq)+2)); default is backbone-dependent",
)
@click.argument("fastas", nargs=-1, type=click.Path(exists=True, dir_okay=False))
def main(
    gff_out,
    min_p,
    backbone,
    model_id,
    revision,
    device,
    token,
    tmp_dir,
    quiet,
    verbose,
    keep_tmp,
    attn_json,
    micro_batch_seqs,
    micro_batch_tokens,
    fastas,
):
    """
    PalmSite — RdRP catalytic center predictor.

    The CLI sanitizes FASTA, embeds sequences with ESM-C, runs PalmSite prediction,
    and writes GFF3. As of v0.2.0, embedding+prediction run in **micro-batches**
    to keep temporary embedding files small and to stream results earlier.
    """
    if not fastas:
        click.echo("No FASTA provided. See `palmsite --help`.", err=True)
        sys.exit(2)

    # Pick log level for embedding engine
    if verbose:
        log_level = "DEBUG"
        quiet = False  # verbose wins
    elif quiet:
        log_level = "WARNING"
    else:
        log_level = "INFO"

    tmp_base = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="palmsite_"))
    tmp_base.mkdir(parents=True, exist_ok=True)
    merged_fa = tmp_base / "sanitized_merged.fasta"
    batch_dir = tmp_base / "microbatches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # 1) Sanitize & merge
    try:
        _log("[1/3] Sanitizing & merging FASTA…", quiet=quiet)
        n_ok, n_fixed, n_dropped = sanitize_and_merge_fastas(
            [Path(p) for p in fastas], merged_fa, strict_fraction=0.20, quiet=quiet
        )
        if n_ok == 0:
            click.echo("All sequences were invalid after sanitization; aborting.", err=True)
            sys.exit(2)
        _log(f"Sanitized sequences: kept={n_ok}, fixed={n_fixed}, dropped={n_dropped}", quiet=quiet)
    except Exception as e:
        click.echo(f"Sanitization failed: {e}", err=True)
        sys.exit(2)

    # Decide streaming micro-batch caps (defaults are tuned for disk friendliness)
    if micro_batch_seqs is not None and micro_batch_seqs <= 0:
        micro_batch_seqs = None
    if micro_batch_tokens is not None and micro_batch_tokens <= 0:
        micro_batch_tokens = None
    if micro_batch_tokens is None:
        micro_batch_tokens = 120_000 if backbone == "6b" else 80_000

    if not quiet:
        caps = []
        if micro_batch_seqs:
            caps.append(f"seqs≤{micro_batch_seqs}")
        if micro_batch_tokens:
            caps.append(f"tokens≈≤{micro_batch_tokens}")
        caps_s = ", ".join(caps) if caps else "(no caps)"
        _log(f"[2/3] Embedding + predicting in micro-batches ({caps_s})…", quiet=quiet)

    out_fp = None
    attn_writer: Optional[_AttnJSONWriter] = None
    try:
        # Prepare output stream
        if gff_out:
            out_fp = open(gff_out, "w", encoding="utf-8")
            stream = out_fp
        else:
            stream = sys.stdout

        # Prepare attention writer (incremental)
        if attn_json:
            attn_writer = _AttnJSONWriter(attn_json)

        header_written = False
        n_batches = 0
        n_seqs_done = 0

        for records, approx_tokens in _iter_microbatches(merged_fa, micro_batch_seqs, micro_batch_tokens):
            n_batches += 1
            n_seqs_done += len(records)

            batch_fa = batch_dir / f"batch_{n_batches:06d}.fasta"
            batch_h5 = batch_dir / f"emb_{n_batches:06d}.h5"

            # Write micro-batch FASTA
            _write_fasta(records, batch_fa)

            _log(f"  [batch {n_batches}] seqs={len(records)} tokens≈{approx_tokens}", quiet=quiet)

            # Embed (in-process to allow caching across micro-batches)
            try:
                embed_fastas_to_h5(
                    fasta_path=str(batch_fa),
                    h5_path=str(batch_h5),
                    backbone=backbone,
                    device=device,
                    token=token,
                    quiet=quiet,
                    log_level=log_level,
                    progress_every=25,
                    # Keep inner batching defaults, but align Forge token cap with user cap when provided
                    max_tokens_per_batch=(micro_batch_tokens if backbone == "6b" else None),
                    use_subprocess=False,
                )
            except Exception as e:
                click.echo(f"Embedding failed (batch {n_batches}): {e}", err=True)
                sys.exit(2)

            # Predict and stream GFF
            try:
                attn_obj = predict_to_gff(
                    embeddings_h5=str(batch_h5),
                    backbone=backbone,
                    model_id=model_id,
                    revision=revision,
                    min_p=float(min_p),
                    out_stream=stream,
                    attn_json=None,  # handled incrementally here
                    write_header=(not header_written),
                    return_attn=bool(attn_writer),
                )
                header_written = True
                if attn_writer and attn_obj:
                    attn_writer.write_dict(attn_obj)
            except Exception as e:
                click.echo(f"Inference failed (batch {n_batches}): {e}", err=True)
                sys.exit(2)
            finally:
                # Free disk aggressively: delete per-batch embeddings as soon as possible
                if not keep_tmp:
                    try:
                        batch_h5.unlink(missing_ok=True)
                    except Exception:
                        pass
                    try:
                        batch_fa.unlink(missing_ok=True)
                    except Exception:
                        pass

        _log(f"[3/3] Done. Processed {n_seqs_done} sequences in {n_batches} batches.", quiet=quiet)

    finally:
        if attn_writer:
            attn_writer.close()
        if out_fp:
            out_fp.close()
        if (not tmp_dir) and (not keep_tmp):
            # Cleanup our own temp dir unless user asked to keep it.
            shutil.rmtree(tmp_base, ignore_errors=True)


if __name__ == "__main__":
    main()
