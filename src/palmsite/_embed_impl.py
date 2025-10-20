#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed protein FASTA sequences with ESM-C and store **token-wise embeddings** in a single HDF5.

- Local ESM-C (HF) for 300m / 600m
- Forge (remote) for 6B with batch executor + token-aware micro-batching

This module exposes both:
  1) a CLI:   python -m palmsite._embed_impl --fasta ... --h5 ...
  2) a library function: embed_fasta_to_h5(...)

Changes in this version:
- Progress lines with ETA and throughput
- DEBUG-level per-chunk traces (opt-in with --log-level DEBUG)
- Batching for both Forge and local backends
  * Forge: concurrency via esm.sdk.batch_executor
  * Local: sequential per micro-batch to avoid GPU contention
- Token-aware micro-batching with --max-tokens-per-batch
- Targeted suppression of benign ESM SDK warning (torch.tensor(tensor) copy-construct)
"""
from __future__ import annotations

import argparse
import math
import hashlib
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import h5py
import numpy as np

# Optional backends (lazy-imported inside _run)
ESMC = None
ESM3ForgeInferenceClient = None
ESMProtein = None
ESMProteinError = None
LogitsConfig = None
batch_executor = None


# ----------------------------
# Logging
# ----------------------------

def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("palmsite.embed")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def _fmt_eta(seconds: Optional[float]) -> str:
    if not seconds or not (seconds > 0) or seconds == float("inf"):
        return "--:--:--"
    s = int(seconds + 0.5)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ----------------------------
# Model name helpers
# ----------------------------

def _norm_name(name: str) -> str:
    return (name or "").strip().lower()

def resolve_model_name(name: str) -> str:
    n = _norm_name(name)
    if n in {"300m", "esmc_300m", "esmc-300m"}:
        return "esmc_300m"
    if n in {"600m", "esmc_600m", "esmc-600m"}:
        return "esmc_600m"
    if n in {"6b", "esmc_6b", "esmc-6b", "esmc-6b-2024-12"}:
        return "esmc-6b-2024-12"
    return name

def is_local_model_name(name: str) -> bool:
    n = _norm_name(name)
    return n in {"esmc_300m", "esmc_600m", "esmc-300m", "esmc-600m"}


# ----------------------------
# FASTA reader
# ----------------------------

def iter_fasta(path: Path) -> Iterator[Tuple[str, str]]:
    sid, buf = None, []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith(">"):
                if sid is not None:
                    yield sid, "".join(buf)
                hd = s[1:].strip()
                sid = hd.split()[0] if hd else "unnamed"
                buf = []
            else:
                buf.append(s)
    if sid is not None:
        yield sid, "".join(buf)


# ----------------------------
# HDF5 writer
# ----------------------------

class H5Writer:
    def __init__(self, out_path: Path, libver: str = "earliest", compress: str = "lzf", gzip_level: int = 1):
        self.out_path = out_path
        self.libver = libver
        self.compress = compress
        self.gzip_level = int(gzip_level)
        self.h5 = None

    def __enter__(self):
        self.h5 = h5py.File(self.out_path, "a", libver=self.libver)
        if "items" not in self.h5:
            self.h5.create_group("items")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.h5:
            self.h5.flush()
            self.h5.close()
            self.h5 = None

    def exists(self, key: str) -> bool:
        return f"items/{key}" in self.h5

    def create(self, key: str, emb: np.ndarray, mask: np.ndarray, seq: str, attrs: dict):
        g = self.h5.create_group(f"items/{key}")
        # Compression
        if self.compress == "gzip":
            kw = dict(compression="gzip", compression_opts=self.gzip_level, shuffle=True)
        elif self.compress == "lzf":
            kw = dict(compression="lzf", shuffle=True)
        else:
            kw = dict()
        g.create_dataset("emb", data=emb, **kw)
        g.create_dataset("mask", data=mask.astype(bool), **kw)
        dt = h5py.special_dtype(vlen=str)
        g.create_dataset("seq", data=seq, dtype=dt)
        for k, v in attrs.items():
            g.attrs[k] = v


# ----------------------------
# Chunking (matches original)
# ----------------------------

def chunk_sequence(seq_id: str, seq: str, chunk_len: int, overlap: int) -> List[Tuple[str, str, int, int, int, int]]:
    L = len(seq)
    if L == 0:
        return []
    chunks = []
    start = 0
    idx = 0
    total = math.ceil((L + chunk_len - 1) / chunk_len)
    while start < L:
        end = min(L, start + chunk_len)
        idx += 1
        key = f"{seq_id}|chunk_{idx:04d}_of_{total:04d}|aa_{start:06d}_{end:06d}"
        chunks.append((key, seq[start:end], start, end, idx, total))
        if end >= L:
            break
        start = max(end - overlap, 0)
    return chunks


# ----------------------------
# Batch helpers
# ----------------------------

def _coerce_token_embeddings(emb, aa_len: int) -> np.ndarray:
    """
    Ensure 2-D (T, D) float32 on CPU; fix common shapes:
      - torch.Tensor or np.ndarray
      - (1, T, D) -> (T, D)
      - accidental transpose if D ~ aa_len
    """
    try:
        import torch
        if isinstance(emb, torch.Tensor):
            arr = emb.detach().to(dtype=torch.float32, device="cpu").numpy()
        else:
            arr = np.asarray(emb, dtype=np.float32)
    except Exception:
        arr = np.asarray(emb, dtype=np.float32)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.ndim != 2:
        raise RuntimeError(f"Expected (T,D) embeddings, got shape {arr.shape}")

    T, D = arr.shape
    if T not in {aa_len, aa_len + 2} and D in {aa_len, aa_len + 2}:
        arr = arr.transpose(0, 1)
    return arr


def _iter_microbatches(to_embed: List[str],
                       meta_batch: List[Tuple],
                       max_tokens: int):
    """
    Yield (sub_to_embed, sub_meta, approx_total_tokens)
    where approx_total_tokens = sum(len(seq)+2).
    """
    if max_tokens is None or max_tokens <= 0:
        yield to_embed, meta_batch, sum(len(s)+2 for s in to_embed)
        return
    i, n = 0, len(to_embed)
    while i < n:
        cur_tok, j = 0, i
        while j < n:
            t = len(to_embed[j]) + 2
            if j > i and (cur_tok + t > max_tokens):
                break
            cur_tok += t
            j += 1
        yield to_embed[i:j], meta_batch[i:j], cur_tok
        i = j


def _embed_batch_resilient(run_batch_embed, client, seqs: List[str],
                           max_retries: int, retry_backoff: float, logger) -> List[Optional[np.ndarray]]:
    """
    Try full batch first with retries, then per-item salvage.
    run_batch_embed(client, seqs) must return List[np.ndarray] (2-D each) or raise.
    """
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            outs = run_batch_embed(client, seqs)
            dt = time.time() - t0
            # Normalize: some SDK versions return Exception objects for failed items
            clean: List[Optional[np.ndarray]] = []
            n_err = 0
            for o in list(outs):
                is_exc = (
                    isinstance(o, BaseException)
                    or (isinstance(o, type) and issubclass(o, BaseException))
                )
                if is_exc:
                    n_err += 1
                    logger.error("Batch item error: %s", o)
                    clean.append(None)
                else:
                    clean.append(o)
            logger.info("Batch OK attempt %d (n=%d, %.2fs, errors=%d)",
                        attempt, len(seqs), dt, n_err)
            # If no errors → done; otherwise try to salvage failed ones per‑item
            if n_err == 0:
                return clean
            # Salvage only the failed indices
            for i, s in enumerate(seqs):
                if clean[i] is not None:
                    continue
                out_i = None
                for a in range(1, max_retries + 1):
                    try:
                        out_i = run_batch_embed(client, [s])[0]
                        if isinstance(out_i, BaseException) or (isinstance(out_i, type) and issubclass(out_i, BaseException)):
                            raise out_i  # normalize
                        break
                    except Exception as e:
                        backoff = retry_backoff * (2 ** (a - 1))
                        logger.warning("  Item %d salvage attempt %d/%d failed: %s (sleep %.1fs)",
                                       i + 1, a, max_retries, e, backoff)
                        time.sleep(backoff)
                clean[i] = out_i if out_i is not None else None
            return clean
        except Exception as e:
            backoff = retry_backoff * (2 ** (attempt - 1))
            logger.warning("Batch failed attempt %d/%d: %s (sleep %.1fs)",
                           attempt, max_retries, e, backoff)
            time.sleep(backoff)
    # Per-item salvage
    logger.warning("Falling back to per-item embedding for %d sequences", len(seqs))
    results: List[Optional[np.ndarray]] = []
    for i, s in enumerate(seqs):
        out_i = None
        for attempt in range(1, max_retries + 1):
            try:
                out_i = run_batch_embed(client, [s])[0]
                break
            except Exception as e:
                backoff = retry_backoff * (2 ** (attempt - 1))
                logger.warning("  Item %d failed attempt %d/%d: %s (sleep %.1fs)",
                               i + 1, attempt, max_retries, e, backoff)
                time.sleep(backoff)
        results.append(out_i)
    return results


# ----------------------------
# Core runner (engine)
# ----------------------------

def _run(args, *, as_library: bool = False):
    logger = setup_logging(args.log_level)

    # Silence benign Forge/ESM SDK warning about torch.tensor(tensor).
    # Show it once when DEBUG is enabled; otherwise ignore.
    if logger.isEnabledFor(logging.DEBUG):
        warnings.filterwarnings(
            "once",
            message=r"To copy construct from a tensor.*",
            category=UserWarning,
            module=r"esm\.utils\.misc",
        )
    else:
        warnings.filterwarnings(
            "ignore",
            message=r"To copy construct from a tensor.*",
            category=UserWarning,
            module=r"esm\.utils\.misc",
        )

    # Mask token in debug-dumped args
    if logger.isEnabledFor(logging.DEBUG):
        dbg = vars(args).copy()
        if "token" in dbg and dbg["token"]:
            dbg["token"] = "***"
        logger.debug(f"args={dbg}")

    # Resolve model + backend
    args.model = resolve_model_name(args.model)
    want_local = is_local_model_name(args.model)

    if want_local:
        try:
            from esm.models.esmc import ESMC as _ESMC  # local ESM-C (HF)
        except Exception as e:
            msg = "Local model requested but `esm` (ESM-C) is not installed."
            if as_library:
                raise ImportError(msg) from e
            logger.error(msg)
            sys.exit(2)
        global ESMC
        ESMC = _ESMC
    else:
        try:
            from esm.sdk.forge import ESM3ForgeInferenceClient as _ForgeClient
            from esm.sdk.api import (
                ESMProtein as _ESMProtein,
                ESMProteinError as _ESMProteinError,
                LogitsConfig as _LogitsConfig,
            )
            from esm.sdk import batch_executor as _batch_executor
        except Exception as e:
            msg = "Forge backend requested but the `esm` SDK is not installed."
            if as_library:
                raise ImportError(msg) from e
            logger.error(msg)
            sys.exit(2)
        global ESM3ForgeInferenceClient, ESMProtein, ESMProteinError, LogitsConfig, batch_executor
        ESM3ForgeInferenceClient, ESMProtein, ESMProteinError, LogitsConfig = _ForgeClient, _ESMProtein, _ESMProteinError, _LogitsConfig
        batch_executor = _batch_executor

    # Default skip behavior if H5 exists
    out_path = Path(args.h5)
    if out_path.exists() and (not args.skip_existing) and (not args.overwrite):
        logger.info("--skip-existing inferred because HDF5 already exists (use --overwrite to rebuild).")
        args.skip_existing = True

    # Load FASTA
    fa = Path(args.fasta)
    if not fa.exists():
        msg = f"FASTA not found: {fa}"
        if as_library:
            raise FileNotFoundError(msg)
        logger.error(msg)
        sys.exit(2)

    items: List[Tuple[str, str]] = list(iter_fasta(fa))
    if not items:
        msg = "No sequences in FASTA."
        if as_library:
            raise RuntimeError(msg)
        logger.error(msg)
        sys.exit(2)

    # Prepare H5 + manifest
    with H5Writer(out_path, libver=args.h5_libver, compress=args.h5_compress, gzip_level=args.h5_gzip_level) as w:
        saved = 0
        skipped = 0
        failed_items: List[str] = []
        progress_every = max(1, int(getattr(args, "progress_every", 25)))

        # Build worklist: (sid, cid, cseq, s0, s1, idx, total, orig_len)
        work: List[Tuple[str, str, str, int, int, int, int, int]] = []
        total_chunks = 0
        for sid, seq in items:
            L = len(seq)
            chunks = (chunk_sequence(sid, seq, args.chunk_len, args.chunk_overlap)
                      if L > args.chunk_len else
                      [(f"{sid}|chunk_0001_of_0001|aa_{0:06d}_{L:06d}", seq, 0, L, 1, 1)])
            total_chunks += len(chunks)
            for (cid, cseq, s0, s1, idx, total) in chunks:
                work.append((sid, cid, cseq, s0, s1, idx, total, L))

        # Count planned work respecting --skip-existing
        todo_total = sum(1 for (_, cid, _, _, _, _, _, _) in work if not (args.skip_existing and w.exists(cid)))
        logger.info(f"Planning complete: {len(items)} sequences → {total_chunks} chunks "
                    f"({todo_total} to embed; skip_existing={args.skip_existing}).")
        started_at = time.time()
        processed = 0  # chunks actually embedded (saved or failed)

        # Manifest external?
        man_fp = None
        if args.manifest:
            man_dir = Path(args.manifest).parent
            man_dir.mkdir(parents=True, exist_ok=True)
            man_fp = open(args.manifest, "a", encoding="utf-8")
            if man_fp.tell() == 0:
                man_fp.write("chunk_id,seq_id,orig_start,orig_end,aa_len,total_tokens,model\n")

        def write_manifest_row(cid, sid, s0, s1, aa_len, tok, model_name):
            if man_fp is not None:
                man_fp.write(f"{cid},{sid},{s0},{s1},{aa_len},{tok},{model_name}\n")

        # Backend setup
        if want_local:
            import torch
            dev = args.device
            if dev == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using local ESM-C model: {args.model} on device={dev}")
            # ESM-C local: use client-style API (ESMProtein, encode, logits)
            from esm.sdk.api import ESMProtein as _ESMProtein, LogitsConfig as _LogitsConfig
            model = ESMC.from_pretrained(args.model).to(dev).eval()
            logits_cfg = _LogitsConfig(sequence=True, return_embeddings=True)

            def _embed_local(seq: str) -> np.ndarray:
                prot = _ESMProtein(sequence=seq)
                tensor = model.encode(prot)
                out = model.logits(tensor, logits_cfg)
                return _coerce_token_embeddings(out.embeddings, aa_len=len(seq))

            def _run_batch_embed(client, seq_batch: List[str]) -> List[np.ndarray]:
                # Sequential per micro-batch to avoid GPU contention
                return [_embed_local(s) for s in seq_batch]

        else:
            tok = args.token or os.getenv("ESM_FORGE_TOKEN")
            if not tok:
                msg = "Forge (6B) selected but no token provided (use --token or set ESM_FORGE_TOKEN)."
                if as_library:
                    raise RuntimeError(msg)
                logger.error(msg)
                sys.exit(2)

            client = ESM3ForgeInferenceClient(model=args.model, url=args.url, token=tok)
            logits_cfg = LogitsConfig(sequence=True, return_embeddings=True)

            def _user_func_for_forge(client, sequence: str) -> np.ndarray:
                prot = ESMProtein(sequence=sequence)
                tensor = client.encode(prot)
                if isinstance(tensor, ESMProteinError):
                    # let caller decide retry strategy
                    raise RuntimeError(f"Forge encode error: {tensor}")
                out = client.logits(tensor, logits_cfg)
                return _coerce_token_embeddings(out.embeddings, aa_len=len(sequence))

            def _run_batch_embed(client, seq_batch: List[str]) -> List[np.ndarray | Exception]:
                with batch_executor() as ex:
                    return ex.execute_batch(
                        user_func=_user_func_for_forge,
                        client=client,
                        sequence=seq_batch,
                    )

        # Iterate & write (batched + micro-batched)
        dtype_out = (np.float16 if args.dtype == "float16" else np.float32)

        # Default token cap for Forge if unset
        token_cap = args.max_tokens_per_batch
        if (not want_local) and (not token_cap):
            token_cap = 120_000  # safe default; tune if you see timeouts
            logger.info("Using default Forge token cap: --max-tokens-per-batch=%d", token_cap)

        for bstart in range(0, len(work), args.batch_size):
            batch = work[bstart:bstart + args.batch_size]

            # Filter skip-existing and prepare this batch
            to_embed: List[str] = []
            meta_batch: List[Tuple[str, str, str, int, int, int, int, int]] = []
            for sid, cid, cseq, s0, s1, idx, total, Lorig in batch:
                if args.skip_existing and w.exists(cid):
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"skip exists: {cid}")
                    if not args.no_manifest_on_skip:
                        write_manifest_row(cid, sid, s0, s1, len(cseq), 0, args.model)
                    skipped += 1
                    continue
                to_embed.append(cseq)
                meta_batch.append((sid, cid, cseq, s0, s1, idx, total, Lorig))

            if not to_embed:
                if args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)
                continue

            # Token-aware micro-batching inside this batch
            for sub_seqs, sub_meta, approx_tokens in _iter_microbatches(to_embed, meta_batch, token_cap):
                logger.info("Submitting micro-batch: n=%d, tokens≈%d", len(sub_seqs), approx_tokens)
                t0 = time.time()
                outputs = _embed_batch_resilient(_run_batch_embed, (model if want_local else client),
                                                 sub_seqs, args.max_retries, args.retry_backoff, logger)
                dt = time.time() - t0
                toks = sum(len(s) + 2 for s in sub_seqs)
                if dt > 0:
                    logger.info("Throughput: %.1f seq/s, %.0f tok/s", len(sub_seqs) / dt, toks / dt)

                if len(outputs) != len(sub_meta):
                    logger.warning("Output length (%d) != input length (%d); truncating to shortest.",
                                   len(outputs), len(sub_meta))

                for (sid, cid, cseq, s0, s1, idx, total, Lorig), rep in zip(sub_meta, outputs):
                    if rep is None:
                        failed_items.append(cid)
                        processed += 1
                        # progress line
                        if (processed % progress_every == 0) or (processed == todo_total):
                            elapsed = time.time() - started_at
                            rate = (processed / elapsed) if elapsed > 0 else 0.0
                            remain = max(todo_total - processed, 0)
                            eta = (remain / rate) if rate > 0 else None
                            pct = (100.0 * processed / max(todo_total, 1))
                            logger.info(f"[embed] {pct:5.1f}%  {processed}/{todo_total}  "
                                        f"| saved={saved} skipped={skipped} failed={len(failed_items)}  "
                                        f"| {rate:.2f}/s  ETA { _fmt_eta(eta) }")
                        continue

                    try:
                        rep = np.asarray(rep, dtype=np.float32, order="C")
                        T, D = rep.shape
                        # Detect BOS/EOS; build mask & trim if present
                        if T == (len(cseq) + 2):
                            emb = rep[1:-1]
                            mask = np.ones((len(cseq),), dtype=bool)
                        else:
                            emb = rep
                            mask = np.ones((T,), dtype=bool)
                        emb = emb.astype(dtype_out, copy=False)

                        attrs = {
                            "seq_id": sid,
                            "original_seq_id": sid,
                            "seq_sha256": hashlib.sha256(cseq.encode()).hexdigest(),
                            "aa_len": int(len(cseq)),
                            "total_tokens": int(emb.shape[0]),
                            "orig_aa_len": int(Lorig),
                            "is_chunked": int(total > 1),
                            "chunk_index": int(idx),
                            "chunks_total": int(total),
                            "orig_aa_start": int(s0),
                            "orig_aa_end": int(s1),
                            "bos_index": int(-1),
                            "eos_index": int(-1),
                            "d_model": int(emb.shape[1]),
                        }
                        w.create(cid, emb, mask, cseq, attrs)
                        write_manifest_row(cid, sid, s0, s1, len(cseq), int(emb.shape[0]), args.model)
                        saved += 1
                    except Exception as e:
                        failed_items.append(cid)
                        logger.exception("Failed to write %s: %s", cid, e)
                    finally:
                        processed += 1
                        if (processed % progress_every == 0) or (processed == todo_total):
                            elapsed = time.time() - started_at
                            rate = (processed / elapsed) if elapsed > 0 else 0.0
                            remain = max(todo_total - processed, 0)
                            eta = (remain / rate) if rate > 0 else None
                            pct = (100.0 * processed / max(todo_total, 1))
                            logger.info(f"[embed] {pct:5.1f}%  {processed}/{todo_total}  "
                                        f"| saved={saved} skipped={skipped} failed={len(failed_items)}  "
                                        f"| {rate:.2f}/s  ETA { _fmt_eta(eta) }")

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

        if man_fp:
            man_fp.flush()
            man_fp.close()

    stats = {"saved": saved, "skipped": skipped, "failed_items": failed_items,
             "h5": str(out_path), "manifest": (args.manifest or "(inside H5)")}
    total_elapsed = time.time() - started_at
    logger.info(f"Embedding finished in {total_elapsed:.1f}s — saved={saved}, skipped={skipped}, "
                f"failed={len(failed_items)} → {out_path}")
    if as_library:
        return stats
    return 0


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="FASTA → ESM-C token-wise embeddings into HDF5 (no annotations)")

    # I/O
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--h5", required=True, help="Output HDF5 file (.h5)")
    ap.add_argument("--h5-libver", choices=["latest", "earliest"], default="earliest",
                    help="HDF5 file libver; 'earliest' is most portable")

    # Manifest
    ap.add_argument("--manifest", default=None, help="Optional external manifest CSV (otherwise inside H5)")
    ap.add_argument("--no-manifest-on-skip", action="store_true",
                    help="Do not append manifest rows for items skipped due to --skip-existing")

    # Model + backend
    ap.add_argument("--model", default="esmc-6b-2024-12", help="esmc_300m | esmc_600m | esmc-6b-2024-12")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Local ESM-C device (ignored for Forge)")
    ap.add_argument("--url", default="https://forge.evolutionaryscale.ai", help="Forge URL")
    ap.add_argument("--token", default=None, help="Forge token (or set ESM_FORGE_TOKEN)")

    # Performance
    ap.add_argument("--batch-size", type=int, default=512, help="Group this many chunks before micro-batching")
    ap.add_argument("--max-tokens-per-batch", type=int, default=0,
                    help="If >0, split each batch so sum(len(seq))+2 ≤ this cap (Forge default: 120000 when unset)")

    # Embedding dtype
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")

    # Chunking
    ap.add_argument("--chunk-len", type=int, default=2000)
    ap.add_argument("--chunk-overlap", type=int, default=128)

    # Resume/overwrite
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")

    # HDF5 compression
    ap.add_argument("--h5-compress", choices=["gzip", "lzf", "none"], default="lzf")
    ap.add_argument("--h5-gzip-level", type=int, default=1)

    # Reliability
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-backoff", type=float, default=2.0)
    ap.add_argument("--sleep-ms", type=int, default=0)

    # Logging
    ap.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    ap.add_argument("--progress-every", type=int, default=25,
                    help="Log a progress line after every N embedded chunks")

    args = ap.parse_args()
    _run(args, as_library=False)


# ----------------------------
# Library entrypoint for PalmSite’s top-level CLI
# ----------------------------

def embed_fasta_to_h5(fasta: str, h5: str, model: str,
                      device: str = "auto", token: str | None = None,
                      batch_size: int = 512, max_tokens_per_batch: int = 0,
                      dtype: str = "float16", chunk_len: int = 2000, chunk_overlap: int = 128,
                      skip_existing: bool = True, overwrite: bool = False,
                      h5_compress: str = "lzf", h5_gzip_level: int = 1,
                      h5_libver: str = "earliest", log_level: str = "WARNING",
                      url: str = "https://forge.evolutionaryscale.ai"):
    """
    Programmatic entrypoint used by PalmSite CLI.
    Returns stats dict: {"saved","skipped","failed_items","h5","manifest"}.
    """
    import argparse as _arg
    model_res = resolve_model_name(model)
    dev = device
    if device == "auto":
        if model_res in {"esmc_300m", "esmc_600m"}:
            try:
                import torch  # only needed for local models
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                dev = "cpu"
        else:
            dev = "cpu"
    ns = _arg.Namespace(
        fasta=fasta, h5=h5, h5_libver=h5_libver,
        manifest=None, no_manifest_on_skip=False,
        model=model_res, device=dev, url=url, token=token,
        batch_size=batch_size, max_tokens_per_batch=max_tokens_per_batch,
        dtype=dtype, chunk_len=chunk_len, chunk_overlap=chunk_overlap,
        skip_existing=skip_existing, overwrite=overwrite,
        h5_compress=h5_compress, h5_gzip_level=h5_gzip_level,
        max_retries=3, retry_backoff=2.0, sleep_ms=0,
        log_level=log_level, progress_every=25,
    )
    return _run(ns, as_library=True)


if __name__ == "__main__":
    main()
