#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embed *already chunked* protein FASTA records with ESM-C and store token-wise embeddings in HDF5,
while preserving the SAME positional/chunking metadata schema as fasta_to_embed_h5.py.

Inputs:
- --chunked-fasta : FASTA where each header is a chunk_id (first token after '>')
- --chunk-manifest-tsv : TSV that maps chunk_id -> (orig_id, chunk_index, chunks_total, orig_aa_start, orig_aa_end, orig_aa_len)

Output HDF5 layout (same as fasta_to_embed_h5.py):
/items/<chunk_id>/emb    (T,D) float16/float32
/items/<chunk_id>/mask   (T,)  bool
/items/<chunk_id>/seq    ()    vlen string
/items/<chunk_id> attrs:
  seq_id, original_seq_id, seq_sha256,
  aa_len, total_tokens, orig_aa_len,
  is_chunked, chunk_index, chunks_total, orig_aa_start, orig_aa_end,
  bos_index, eos_index,
  esm_model, esm_url, d_model
/manifest/rows           (M,) vlen string; CSV-like rows; header in attrs["header"]

Notes:
- We DO NOT re-chunk. The FASTA is assumed chunked already.
- We use the manifest TSV to restore orig_id and original coordinates, so downstream code works unchanged.
"""

import os, sys, io, csv, time, argparse, logging
from dataclasses import dataclass
from typing import Iterator, Tuple, List, Dict, Optional

import numpy as np
import torch
import h5py

# Forge / ESM-C SDK (same imports as your original script)
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"To copy construct from a tensor.*",
    category=UserWarning,
    module=r"esm\.utils\.misc"
)

from esm.sdk.forge import ESM3ForgeInferenceClient
from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig
from esm.sdk import batch_executor

# Local (Hugging Face) ESM-C
try:
    from esm.models.esmc import ESMC  # available when using local 300m/600m
except Exception:
    ESMC = None


# -------------------------
# Logging
# -------------------------

def setup_logging(level: str = "INFO"):
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("embed_h5")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    return logger


# -------------------------
# FASTA utilities (match original)
# -------------------------

def _clean_seq(s: str) -> str:
    s = s.upper().replace(" ", "").replace("\t", "").replace("\r", "").replace("\n", "")
    s = s.replace("*", "")
    return "".join(ch for ch in s if (32 <= ord(ch) <= 126) and ch.isprintable())

def read_fasta(fp) -> Iterator[Tuple[str, str]]:
    """
    Parse FASTA; use the *first token* after '>' as the ID.
    For chunked FASTA, this should be the chunk_id.
    """
    seq_id, chunks = None, []
    for line in fp:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seq_id is not None:
                yield seq_id, _clean_seq("".join(chunks))
            header = line[1:].strip()
            seq_id = header.split()[0] if header else "unnamed"
            chunks = []
        else:
            chunks.append(line)
    if seq_id is not None:
        yield seq_id, _clean_seq("".join(chunks))

def load_fasta(path: str) -> List[Tuple[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(read_fasta(f))

def sanitize_id(name: str) -> str:
    # HDF5 path separator is '/', so that MUST be excluded.
    # '|' is safe in HDF5 group names, so we keep it.
    bad = '<>:"/\\?*\t\r\n'
    sanitized = "".join(c if c not in bad else "_" for c in name.strip())
    return sanitized[:240] if sanitized else "unnamed"

def sha256_of_text(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# -------------------------
# Manifest TSV → ChunkInfo
# -------------------------

@dataclass
class ChunkInfo:
    orig_id: str
    orig_len: int
    chunk_index: int
    chunks_total: int
    aa_start: int
    aa_end: int
    chunk_id: str

def load_chunk_manifest_tsv(path: str) -> Dict[str, ChunkInfo]:
    """
    Expected TSV header (from the chunking script I gave you):
      chunk_id  orig_id  chunk_index  chunks_total  orig_aa_start  orig_aa_end  chunk_aa_len  orig_aa_len

    Returns dict: chunk_id -> ChunkInfo (with orig_len from orig_aa_len).
    """
    out: Dict[str, ChunkInfo] = {}
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        # Accept either exact header or any superset containing the required columns
        col = {name: i for i, name in enumerate(header)}

        required = ["chunk_id", "orig_id", "chunk_index", "chunks_total",
                    "orig_aa_start", "orig_aa_end", "orig_aa_len"]
        missing = [r for r in required if r not in col]
        if missing:
            raise ValueError(f"Manifest TSV missing required columns: {missing}. Header={header}")

        for line in f:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            cid = parts[col["chunk_id"]]
            out[cid] = ChunkInfo(
                orig_id=parts[col["orig_id"]],
                orig_len=int(parts[col["orig_aa_len"]]),
                chunk_index=int(parts[col["chunk_index"]]),
                chunks_total=int(parts[col["chunks_total"]]),
                aa_start=int(parts[col["orig_aa_start"]]),
                aa_end=int(parts[col["orig_aa_end"]]),
                chunk_id=cid,
            )
    return out


# -------------------------
# HDF5 writer (same behavior as original)
# -------------------------

class H5Writer:
    def __init__(self, path: str, compress: str = "gzip", gzip_level: int = 1,
                 overwrite: bool = False, libver: str = "latest", logger=None):
        self.path = path
        self.logger = logger or logging.getLogger("embed_h5")
        self.compress = compress
        self.gzip_level = int(gzip_level)
        self.overwrite = bool(overwrite)
        self.libver = libver

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = h5py.File(path, "a", libver=self.libver)
        self.items = self.f.require_group("items")
        self.manifest = self.f.require_group("manifest")
        self.rows = self.manifest.get("rows")
        if self.rows is None:
            dt = h5py.string_dtype(encoding="utf-8")
            self.rows = self.manifest.create_dataset("rows", shape=(0,), maxshape=(None,), dtype=dt)
            self.rows.attrs["header"] = ",".join([
                "chunk_id","orig_id","chunk_index","chunks_total",
                "orig_aa_start","orig_aa_end","chunk_aa_len",
                "total_tokens","dim","model","dtype"
            ])
        self.vstr = h5py.string_dtype("utf-8")

    def _ckw(self):
        if self.compress == "gzip":
            return dict(compression="gzip", compression_opts=self.gzip_level, shuffle=True)
        elif self.compress == "lzf":
            return dict(compression="lzf", shuffle=True)
        else:
            return dict()

    def has(self, chunk_id: str) -> bool:
        cid = sanitize_id(chunk_id)
        if cid not in self.items:
            return False
        try:
            _ = self.items[cid]["emb"]
            return True
        except Exception:
            return False

    def dims(self, chunk_id: str) -> Tuple[int, int]:
        cid = sanitize_id(chunk_id)
        try:
            emb = self.items[cid]["emb"]
            return int(emb.shape[0]), int(emb.shape[1])
        except Exception:
            return (-1, -1)

    def add_row(self, row: List):
        buf = io.StringIO()
        csv.writer(buf).writerow(row)
        s = buf.getvalue().strip("\r\n")
        n = self.rows.shape[0]
        self.rows.resize((n + 1,))
        self.rows[n] = s

    def _prepare_group(self, chunk_id: str) -> h5py.Group:
        cid = sanitize_id(chunk_id)
        if cid in self.items:
            can_read = True
            try:
                _ = self.items[cid]["emb"]
            except Exception:
                can_read = False
            if self.overwrite or (not can_read):
                del self.items[cid]
                self.f.flush()
            else:
                raise KeyError(f"HDF5 group already exists and --overwrite not set: {cid}")
        return self.items.require_group(cid)

    def add_item(self, chunk_id: str, emb: torch.Tensor, mask: torch.Tensor, meta: dict, dtype_out: torch.dtype):
        g = self._prepare_group(chunk_id)
        ds_kwargs = self._ckw()

        emb_np = emb.to(dtype_out).cpu().numpy()
        mask_np = mask.to(torch.bool).cpu().numpy().astype(np.bool_)

        T, D = emb_np.shape
        emb_chunks = (min(T, 1024), D)

        g.create_dataset("emb", data=emb_np, chunks=emb_chunks, **ds_kwargs)
        g.create_dataset("mask", data=mask_np, chunks=True, **ds_kwargs)
        if "seq" in meta:
            g.create_dataset("seq", data=np.array(meta["seq"], dtype=self.vstr))

        for k in ["seq_id","original_seq_id","seq_sha256"]:
            if k in meta:
                g.attrs[k] = meta[k]

        lengths = meta.get("lengths", {})
        g.attrs["aa_len"] = int(lengths.get("aa", -1))
        g.attrs["total_tokens"] = int(lengths.get("total_tokens", -1))
        g.attrs["orig_aa_len"] = int(lengths.get("orig_aa_len", -1))

        chunking = meta.get("chunking", {})
        g.attrs["is_chunked"] = bool(chunking.get("is_chunked", False))
        g.attrs["chunk_index"] = int(chunking.get("chunk_index", 1))
        g.attrs["chunks_total"] = int(chunking.get("chunks_total", 1))
        g.attrs["orig_aa_start"] = int(chunking.get("orig_aa_start", 0))
        g.attrs["orig_aa_end"] = int(chunking.get("orig_aa_end", 0))

        special = meta.get("special_tokens", {})
        g.attrs["bos_index"] = int(special.get("bos_index")) if special.get("bos_index") is not None else -1
        g.attrs["eos_index"] = int(special.get("eos_index")) if special.get("eos_index") is not None else -1

        esm = meta.get("esm", {})
        g.attrs["esm_model"] = str(esm.get("model", ""))
        g.attrs["esm_url"] = str(esm.get("url", ""))
        g.attrs["d_model"] = int(esm.get("d_model", -1))

    def close(self):
        try:
            self.f.flush()
        finally:
            self.f.close()


# -------------------------
# Backend helpers (same as original)
# -------------------------

def normalize_model_name(name: str) -> str:
    return (name or "").strip().lower().replace("-", "_")

def is_local_model_name(name: str) -> bool:
    n = normalize_model_name(name)
    return n in {"esmc_300m", "esmc_600m"}

def to_hf_name(name: str) -> str:
    n = normalize_model_name(name)
    if n in {"esmc_300m", "esmc_600m"}:
        return n
    return n

def is_local_client(client) -> bool:
    if ESMC is None:
        return False
    return isinstance(client, ESMC)

def embed_sequence(client, sequence: str):
    p = ESMProtein(sequence=sequence)
    t = client.encode(p)
    if isinstance(t, ESMProteinError):
        raise t
    out = client.logits(t, LogitsConfig(sequence=False, return_embeddings=True))
    return out

def run_batch_embed(client, seq_batch: List[str]):
    if is_local_client(client):
        outs = []
        for s in seq_batch:
            outs.append(embed_sequence(client, s))
        return outs
    else:
        with batch_executor() as ex:
            return ex.execute_batch(
                user_func=embed_sequence,
                client=client,
                sequence=seq_batch,
            )

def embed_batch_resilient(client, to_embed: List[str], max_retries: int, retry_backoff: float, logger) -> List[Optional[object]]:
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            outs = run_batch_embed(client, to_embed)
            dt = time.time() - t0
            logger.info("Batch OK attempt %d (n=%d, %.2fs)", attempt, len(to_embed), dt)
            return list(outs)
        except ESMProteinError as e:
            logger.error("Batch contains invalid sequence(s): %s", e)
            break
        except Exception as e:
            backoff = retry_backoff * (2 ** (attempt - 1))
            logger.warning("Batch failed attempt %d/%d: %s (sleep %.1fs)", attempt, max_retries, e, backoff)
            time.sleep(backoff)

    logger.warning("Falling back to per-item embedding for %d sequences", len(to_embed))
    results: List[Optional[object]] = []
    for i, seq in enumerate(to_embed):
        out = None
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                out = embed_sequence(client, seq)
                dt = time.time() - t0
                logger.info("OK item %d/%d (%.2fs)", i + 1, len(to_embed), dt)
                break
            except ESMProteinError as e:
                logger.error("Invalid sequence at item %d: %s (no retry)", i + 1, e)
                break
            except Exception as e:
                backoff = retry_backoff * (2 ** (attempt - 1))
                logger.warning("  Item %d failed attempt %d/%d: %s (sleep %.1fs)",
                               i + 1, attempt, max_retries, e, backoff)
                time.sleep(backoff)
        results.append(out)
    return results

def coerce_token_embeddings(emb, aa_len: int, logger: Optional[logging.Logger] = None) -> torch.Tensor:
    if not isinstance(emb, torch.Tensor):
        emb = torch.as_tensor(emb)
    if emb.ndim == 3 and emb.shape[0] == 1:
        emb = emb[0]
    elif emb.ndim != 2:
        if logger:
            logger.error("Unexpected embedding ndim=%d, shape=%s", emb.ndim, tuple(emb.shape))
        raise ValueError(f"Expected 2-D embeddings, got shape {tuple(emb.shape)}")
    T, D = emb.shape
    if T not in {aa_len, aa_len + 2} and D in {aa_len, aa_len + 2}:
        if logger:
            logger.warning("Embeddings appear transposed; auto-fixing. shape=%s aa_len=%d", tuple(emb.shape), aa_len)
        emb = emb.transpose(0, 1)
    return emb

def iter_microbatches(to_embed: List[str],
                      meta_batch: List[Tuple[ChunkInfo, str]],
                      max_tokens: int):
    if max_tokens <= 0:
        yield to_embed, meta_batch, sum(len(s) + 2 for s in to_embed)
        return
    i, n = 0, len(to_embed)
    while i < n:
        cur_tok = 0
        j = i
        while j < n:
            t = len(to_embed[j]) + 2
            if j > i and (cur_tok + t > max_tokens):
                break
            cur_tok += t
            j += 1
        yield to_embed[i:j], meta_batch[i:j], cur_tok
        i = j


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Chunked FASTA → ESM-C token-wise embeddings into HDF5 (keeps original positional metadata)")

    # Inputs
    ap.add_argument("--chunked-fasta", required=True, help="FASTA with chunk_id headers")
    ap.add_argument("--chunk-manifest-tsv", required=True, help="TSV mapping chunk_id to original coordinates/IDs")

    # Output
    ap.add_argument("--h5", required=True, help="Output HDF5 file (.h5)")
    ap.add_argument("--h5-libver", choices=["latest", "earliest"], default="latest")

    # Model + backend
    ap.add_argument("--model", default="esmc-6b-2024-12",
                    help="Model. Use esmc_300m / esmc_600m for local HF; 6B via Forge (e.g., esmc-6b-2024-12).")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"),
                    help="Device for local HF models (ignored for Forge).")
    ap.add_argument("--url", default="https://forge.evolutionaryscale.ai", help="Forge URL (remote models).")
    ap.add_argument("--token", default=None, help="Forge token (or set ESM_FORGE_TOKEN env var).")

    # Performance / IO
    ap.add_argument("--batch-size", type=int, default=512, help="Max items grouped before micro-batching")
    ap.add_argument("--max-tokens-per-batch", type=int, default=0,
                    help="If >0, split each batch so sum(len(seq))+2 <= this cap.")
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--h5-compress", choices=["gzip", "lzf", "none"], default="gzip")
    ap.add_argument("--h5-gzip-level", type=int, default=1)

    # Robustness
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-backoff", type=float, default=2.0)
    ap.add_argument("--sleep-ms", type=int, default=0)

    # Consistency checks
    ap.add_argument("--strict-manifest", action="store_true",
                    help="Error if manifest chunk_aa_len / orig_aa_end-start doesn't match FASTA sequence length.")
    ap.add_argument("--log-level", default="INFO")

    args = ap.parse_args()
    logger = setup_logging(args.log_level)

    # Load inputs
    entries = load_fasta(args.chunked_fasta)
    if not entries:
        logger.error("No sequences found in FASTA: %s", args.chunked_fasta)
        sys.exit(1)
    entries = [(cid, seq) for cid, seq in entries if len(seq) > 0]
    logger.info("Loaded %d chunk records from %s", len(entries), args.chunked_fasta)

    manifest = load_chunk_manifest_tsv(args.chunk_manifest_tsv)
    logger.info("Loaded %d manifest rows from %s", len(manifest), args.chunk_manifest_tsv)

    # Safer resume default (same idea as original)
    if os.path.exists(args.h5) and not args.skip_existing and not args.overwrite:
        logger.info("H5 exists: %s — defaulting to --skip-existing (use --overwrite to rewrite).", args.h5)
        args.skip_existing = True

    # Backend selection
    want_local = is_local_model_name(args.model)
    if want_local and ESMC is None:
        logger.error("Local model requested (%s) but esm.models.esmc is not available.", args.model)
        sys.exit(2)

    token = args.token or os.environ.get("ESM_FORGE_TOKEN")
    if not want_local and not token:
        logger.error("Provide a Forge token via --token or ESM_FORGE_TOKEN env var for remote models.")
        sys.exit(2)

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Build client
    if want_local:
        hf_name = to_hf_name(args.model)
        logger.info("Loading local ESM-C model '%s' on device '%s'...", hf_name, args.device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            client = ESMC.from_pretrained(hf_name).to(args.device)
        backend_url = "local:huggingface"
    else:
        logger.info("Using Forge backend model '%s' at %s", args.model, args.url)
        client = ESM3ForgeInferenceClient(model=args.model, url=args.url, token=token)
        backend_url = args.url

    # Open H5
    h5w = H5Writer(args.h5, compress=args.h5_compress, gzip_level=args.h5_gzip_level,
                   overwrite=args.overwrite, libver=args.h5_libver, logger=logger)

    saved = 0
    skipped = 0
    failed = 0

    # Prebuild worklist in FASTA order (ensures deterministic output ordering)
    work: List[Tuple[ChunkInfo, str]] = []
    missing = 0
    for chunk_id, seq in entries:
        if chunk_id not in manifest:
            missing += 1
            logger.error("Chunk id missing from manifest TSV: %s", chunk_id)
            continue
        ci = manifest[chunk_id]

        if args.strict_manifest:
            L = len(seq)
            # orig_aa_end - orig_aa_start should equal chunk length (for internal chunks)
            if (ci.aa_end - ci.aa_start) != L:
                raise ValueError(f"Length mismatch for {chunk_id}: FASTA len={L} vs (aa_end-aa_start)={ci.aa_end-ci.aa_start}")
        work.append((ci, seq))

    if missing:
        logger.error("Found %d chunk IDs in FASTA that are missing in manifest. Fix manifest and rerun.", missing)
        sys.exit(3)

    total = len(work)
    logger.info("Prepared %d items for embedding (no re-chunking).", total)

    try:
        for bstart in range(0, total, args.batch_size):
            batch = work[bstart:bstart + args.batch_size]

            to_embed: List[str] = []
            meta_batch: List[Tuple[ChunkInfo, str]] = []

            for ci, subseq in batch:
                if args.skip_existing and h5w.has(ci.chunk_id):
                    T, D = h5w.dims(ci.chunk_id)
                    if T > 0 and D > 0:
                        # Still append manifest row (matching original behavior)
                        row = [
                            ci.chunk_id, ci.orig_id, ci.chunk_index, ci.chunks_total,
                            ci.aa_start, ci.aa_end, len(subseq),
                            T, D, args.model, args.dtype,
                        ]
                        h5w.add_row(row)
                        skipped += 1
                        continue
                    else:
                        logger.warning("Existing group incomplete; will re-embed: %s", ci.chunk_id)

                to_embed.append(subseq)
                meta_batch.append((ci, subseq))

            if not to_embed:
                if args.sleep_ms > 0:
                    time.sleep(args.sleep_ms / 1000.0)
                continue

            for sub_seqs, sub_meta, approx_tokens in iter_microbatches(to_embed, meta_batch, args.max_tokens_per_batch):
                logger.info("Submitting micro-batch: n=%d, tokens≈%d", len(sub_seqs), approx_tokens)
                t0 = time.time()
                outputs = embed_batch_resilient(client, sub_seqs, args.max_retries, args.retry_backoff, logger)
                dt = time.time() - t0
                toks = sum(len(s) + 2 for s in sub_seqs)
                if dt > 0:
                    logger.info("Throughput: %.1f seq/s, %.0f tok/s", len(sub_seqs) / dt, toks / dt)

                for (ci, subseq), out in zip(sub_meta, outputs):
                    if out is None:
                        failed += 1
                        logger.error("Embedding failed for %s; no output object.", ci.chunk_id)
                        continue
                    try:
                        emb_raw = out.embeddings
                        emb = coerce_token_embeddings(emb_raw, aa_len=len(subseq), logger=logger)

                        T, D = emb.shape
                        L = len(subseq)
                        has_special = (T == L + 2)
                        if has_special:
                            mask = torch.zeros(T, dtype=torch.bool)
                            if L > 0:
                                mask[1:1 + L] = True
                            bos_index, eos_index = 0, T - 1
                        else:
                            mask = torch.ones(T, dtype=torch.bool)
                            bos_index, eos_index = None, None

                        # Meta schema matches fasta_to_embed_h5.py
                        meta = {
                            "seq_id": ci.chunk_id,
                            "original_seq_id": ci.orig_id,
                            "seq_sha256": sha256_of_text(subseq),
                            "seq": subseq,
                            "lengths": {"aa": int(L), "total_tokens": int(T), "orig_aa_len": int(ci.orig_len)},
                            "chunking": {"is_chunked": ci.chunks_total > 1,
                                         "chunk_index": int(ci.chunk_index),
                                         "chunks_total": int(ci.chunks_total),
                                         "orig_aa_start": int(ci.aa_start),
                                         "orig_aa_end": int(ci.aa_end)},
                            "special_tokens": {"bos_index": bos_index, "eos_index": eos_index},
                            "esm": {"model": args.model, "url": backend_url, "d_model": int(D)},
                        }

                        h5w.add_item(ci.chunk_id, emb, mask, meta, dtype_out=dtype)

                        row = [
                            ci.chunk_id, ci.orig_id, ci.chunk_index, ci.chunks_total,
                            ci.aa_start, ci.aa_end, L, T, D, args.model, args.dtype,
                        ]
                        h5w.add_row(row)
                        saved += 1

                    except Exception as e:
                        failed += 1
                        logger.exception("Failed to write %s: %s", ci.chunk_id, e)

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Flushing and closing files...")
    finally:
        h5w.close()

    logger.info("Done. Saved=%d, skipped_existing=%d, failed=%d. H5=%s", saved, skipped, failed, args.h5)


if __name__ == "__main__":
    main()
