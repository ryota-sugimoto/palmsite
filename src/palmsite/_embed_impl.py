#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embed protein FASTA sequences with ESM-C and store **token-wise embeddings** in a single HDF5.

This file is the **engine** used by PalmSite’s simple CLI.
- Local ESM-C (HF) for 300m / 600m
- Forge (remote) for 6B

We expose both a CLI (`python -m palmsite._embed_impl`) and a library function
`embed_fasta_to_h5(...)` so PalmSite can call it directly.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

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


# ----------------------------
# Model name helpers
# ----------------------------

def normalize_model_name(name: str) -> str:
    return (name or "").strip().lower()

def resolve_model_name(name: str) -> str:
    n = (name or "").strip().lower()
    if n in {"300m","esmc_300m","esmc-300m"}:
        return "esmc_300m"
    if n in {"600m","esmc_600m","esmc-600m"}:
        return "esmc_600m"
    if n in {"6b","esmc_6b","esmc-6b","esmc-6b-2024-12"}:
        return "esmc-6b-2024-12"
    return name

def is_local_model_name(n: str) -> bool:
    n = normalize_model_name(n)
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
# H5 writer
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
        # datasets
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
# Chunking (match your original)
# ----------------------------

def chunk_sequence(seq_id: str, seq: str, chunk_len: int, overlap: int) -> List[Tuple[str, str, int, int, int, int]]:
    L = len(seq)
    if L == 0:
        return []
    chunks = []
    start = 0
    idx = 0
    while start < L:
        end = min(L, start + chunk_len)
        idx += 1
        key = f"{seq_id}|chunk_{idx:04d}_of_{math.ceil((L + chunk_len - 1)/chunk_len):04d}|aa_{start:06d}_{end:06d}"
        chunks.append((key, seq[start:end], start, end, idx, math.ceil((L + chunk_len - 1)/chunk_len)))
        if end >= L:
            break
        start = max(end - overlap, 0)
    return chunks


# ----------------------------
# Core runner (engine)
# ----------------------------

def _run(args, *, as_library: bool = False):
    logger = setup_logging(args.log_level)

    # Resolve model + backend
    args.model = resolve_model_name(args.model)
    want_local = is_local_model_name(args.model)

    if want_local:
        try:
            from esm.models.esmc import ESMC as _ESMC  # local ESM-C
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
            from esm.sdk.api import ESMProtein as _ESMProtein, ESMProteinError as _ESMProteinError, LogitsConfig as _LogitsConfig
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

    # Prepare H5
    with H5Writer(out_path, libver=args.h5_libver, compress=args.h5_compress, gzip_level=args.h5_gzip_level) as w:
        saved = 0
        skipped = 0
        failed_items: List[str] = []

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
            model = ESMC.from_pretrained(args.model)
            model = model.eval().to(dev)

            def embed_seq(seq: str) -> np.ndarray:
                # NOTE: users typically call tokenizer; for ESM-C’s HF variant the forward returns dict with 'representations'
                toks = model.tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(dev)
                with torch.no_grad():
                    rep = model(**toks)["representations"][model.num_layers]  # (1,T,D)
                rep = rep[0].detach().cpu().numpy()
                # drop BOS/EOS if present (mask aligned below)
                return rep
        else:
            tok = args.token or os.getenv("ESM_FORGE_TOKEN")
            if not tok:
                msg = "Forge (6B) selected but no token provided (use --token or set ESM_FORGE_TOKEN)."
                if as_library:
                    raise RuntimeError(msg)
                logger.error(msg)
                sys.exit(2)
            client = ESM3ForgeInferenceClient(url=args.url, token=tok)
            logits_cfg = LogitsConfig(layer="token_representations")

            def embed_seq(seq: str) -> np.ndarray:
                prot = ESMProtein(sequence=seq)
                out = client.infer(protein=prot, logits=logits_cfg)
                # Expect np.ndarray (T,D) from SDK
                return out.representations  # type: ignore[attr-defined]

        # Iterate & write
        for sid, seq in items:
            chunks = chunk_sequence(sid, seq, args.chunk_len, args.chunk_overlap) if len(seq) > args.chunk_len else \
                     [(f"{sid}|chunk_0001_of_0001|aa_{0:06d}_{len(seq):06d}", seq, 0, len(seq), 1, 1)]
            for cid, cseq, s0, s1, idx, total in chunks:
                if args.skip_existing and w.exists(cid):
                    if not args.no_manifest_on_skip:
                        write_manifest_row(cid, sid, s0, s1, len(cseq), 0, args.model)
                    skipped += 1
                    continue

                # Embed with retries
                attempt = 0
                while True:
                    attempt += 1
                    try:
                        rep = embed_seq(cseq)  # (T,D) incl BOS/EOS if present
                        # Build mask: try to detect BOS/EOS by length heuristic vs AA length
                        T, D = rep.shape
                        mask = np.ones((T,), dtype=bool)
                        # cheap heuristic: if T == len(cseq) + 2, drop first/last
                        if T == len(cseq) + 2:
                            rep = rep[1:-1]
                            mask = np.ones((len(cseq),), dtype=bool)
                        elif T == len(cseq):
                            mask = np.ones((T,), dtype=bool)
                        else:
                            # fallback: best-effort keep all and mark all True
                            mask = np.ones((T,), dtype=bool)

                        emb = rep.astype(np.float16 if args.dtype == "float16" else np.float32, copy=False)

                        attrs = {
                            "seq_id": sid,
                            "original_seq_id": sid,
                            "seq_sha256": hashlib.sha256(cseq.encode()).hexdigest(),
                            "aa_len": int(len(cseq)),
                            "total_tokens": int(emb.shape[0]),
                            "orig_aa_len": int(len(seq)),
                            "is_chunked": int(True if len(seq) > args.chunk_len else False),
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
                        break
                    except Exception as e:
                        if attempt >= args.max_retries:
                            logger.error(f"FAIL {cid}: {e}")
                            failed_items.append(cid)
                            break
                        sleep_s = (args.retry_backoff ** (attempt - 1))
                        logger.warning(f"Retry {attempt}/{args.max_retries} after error: {e} (sleep {sleep_s:.1f}s)")
                        time.sleep(sleep_s)

        if args.manifest and 'man_fp' in locals() and man_fp:
            man_fp.flush()
            man_fp.close()

    stats = {"saved": saved, "skipped": skipped, "failed_items": failed_items,
             "h5": str(out_path), "manifest": (args.manifest or "(inside H5)")}
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
    ap.add_argument("--h5-libver", choices=["latest","earliest"], default="earliest",
                    help="HDF5 file libver; 'earliest' is most portable")

    # Manifest
    ap.add_argument("--manifest", default=None, help="Optional external manifest CSV (otherwise inside H5)")
    ap.add_argument("--no-manifest-on-skip", action="store_true",
                    help="Do not append manifest rows for items skipped due to --skip-existing")

    # Model + backend
    ap.add_argument("--model", default="esmc-6b-2024-12", help="esmc_300m | esmc_600m | esmc-6b-2024-12")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto", help="Local ESM-C device")
    ap.add_argument("--url", default="https://forge.evolutionaryscale.ai", help="Forge URL")
    ap.add_argument("--token", default=None, help="Forge token (or set ESM_FORGE_TOKEN)")

    # Performance
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-tokens-per-batch", type=int, default=0)

    # Embedding dtype
    ap.add_argument("--dtype", choices=["float16","float32"], default="float16")

    # Chunking
    ap.add_argument("--chunk-len", type=int, default=2000)
    ap.add_argument("--chunk-overlap", type=int, default=128)

    # Resume/overwrite
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")

    # HDF5 compression
    ap.add_argument("--h5-compress", choices=["gzip","lzf","none"], default="lzf")
    ap.add_argument("--h5-gzip-level", type=int, default=1)

    # Reliability
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-backoff", type=float, default=2.0)
    ap.add_argument("--sleep-ms", type=int, default=0)

    # Logging
    ap.add_argument("--log-level", default="INFO", help="DEBUG|INFO|WARNING|ERROR")

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
    import argparse, torch
    model_res = resolve_model_name(model)
    dev = device
    if device == "auto":
        dev = "cuda" if torch.cuda.is_available() and model_res in {"esmc_300m","esmc_600m"} else "cpu"
    ns = argparse.Namespace(
        fasta=fasta, h5=h5, h5_libver=h5_libver,
        manifest=None, no_manifest_on_skip=False,
        model=model_res, device=dev, url=url, token=token,
        batch_size=batch_size, max_tokens_per_batch=max_tokens_per_batch,
        dtype=dtype, chunk_len=chunk_len, chunk_overlap=chunk_overlap,
        skip_existing=skip_existing, overwrite=overwrite,
        h5_compress=h5_compress, h5_gzip_level=h5_gzip_level,
        max_retries=3, retry_backoff=2.0, sleep_ms=0,
        log_level=log_level,
    )
    return _run(ns, as_library=True)


if __name__ == "__main__":
    main()

