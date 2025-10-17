from __future__ import annotations
import os, sys, subprocess

def embed_fastas_to_h5(fasta_path: str, h5_path: str,
                       backbone: str = "600m",
                       device: str = "auto",
                       token: str | None = None,
                       quiet: bool = False,
                       log_level: str = "INFO",
                       progress_every: int = 25,
                       batch_size: int = 512,
                       max_tokens_per_batch: int | None = None):
    """
    Thin wrapper over palmsite._embed_impl (runs it in a subprocess).

    - 300m/600m → local ESM-C (HF), uses --device
    - 6b       → Forge (needs token via arg or ESM_FORGE_TOKEN)

    Parameters
    ----------
    fasta_path : str
      Input FASTA filepath.
    h5_path : str
      Output HDF5 filepath for token embeddings.
    backbone : {"300m","600m","6b"}
      Embedding backbone size.
    device : {"auto","cpu","cuda"}
      Local device (ignored for 6b).
    token : str | None
      Forge token (or set ESM_FORGE_TOKEN).
    quiet : bool
      If True, suppress stdout/stderr from the embedder unless it fails.
    log_level : str
      DEBUG | INFO | WARNING | ERROR (forwarded to embedder).
    progress_every : int
      Log a progress line after every N embedded chunks.
    batch_size : int
      Number of chunks to group before micro-batching.
    max_tokens_per_batch : int | None
      Token cap for micro-batches (sum(len(seq))+2). If None and backbone=="6b", defaults to 120000.
    """
    model = {"300m": "esmc_300m", "600m": "esmc_600m", "6b": "esmc-6b-2024-12"}[backbone]

    # Resolve device if auto for local backbones
    dev = device
    if device == "auto":
        if backbone in {"300m", "600m"}:
            try:
                import torch  # lazy import
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                raise RuntimeError(
                    "Local ESM-C (300m/600m) selected but PyTorch is not installed.\n"
                    "Install with: pip install 'palmsite[cpu]' (or use conda pytorch),\n"
                    "or switch to '-b 6b' with a Forge token."
                )
        else:
            dev = "cpu"

    cmd = [
        sys.executable, "-m", "palmsite._embed_impl",
        "--fasta", fasta_path,
        "--h5", h5_path,
        "--model", model,
        "--h5-compress", "lzf",
        "--chunk-len", "2000",
        "--chunk-overlap", "128",
        "--batch-size", str(int(batch_size)),
        "--log-level", str(log_level),
        "--progress-every", str(int(progress_every)),
    ]

    if backbone in {"300m", "600m"}:
        cmd += ["--device", dev]
    else:
        tk = token or os.getenv("ESM_FORGE_TOKEN")
        if not tk:
            raise RuntimeError("6B selected but no Forge token provided. Use -k/--token or set ESM_FORGE_TOKEN.")
        cmd += ["--token", tk]
        # Provide a safe default token cap unless caller overrides
        if max_tokens_per_batch is None:
            max_tokens_per_batch = 120_000

    if max_tokens_per_batch is not None:
        cmd += ["--max-tokens-per-batch", str(int(max_tokens_per_batch))]

    if quiet:
        env = os.environ.copy()
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Embedding error: {p.stderr.strip()}")
    else:
        subprocess.check_call(cmd)
