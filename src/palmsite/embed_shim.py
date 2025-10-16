from __future__ import annotations
import os, sys, subprocess, shlex


def embed_fastas_to_h5(fasta_path: str, h5_path: str,
                       backbone: str = "600m",
                       device: str = "auto",
                       token: str | None = None,
                       quiet: bool = False):
    """
    Thin wrapper over your existing embedding script (packaged as palmsite._embed_impl).
    - 300m/600m → local ESM-C (HF), uses --device
    - 6b       → Forge (needs token via arg or ESM_FORGE_TOKEN)
    """
    model = {"300m":"esmc_300m", "600m":"esmc_600m", "6b":"esmc-6b-2024-12"}[backbone]
    dev = device
    if device == "auto":
        if backbone in {"300m","600m"}:
            try:
                import torch  # lazy import
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                raise RuntimeError(
                    "Local ESM-C (300m/600m) selected but PyTorch is not installed.\n"
                    "Install with: pip install 'palmsite[cpu]'  (or use conda pytorch),\n"
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
        "--batch-size", "512",
    ]
    if backbone in {"300m","600m"}:
        cmd += ["--device", dev]
    else:
        tk = token or os.getenv("ESM_FORGE_TOKEN")
        if not tk:
            raise RuntimeError("6B selected but no Forge token provided. Use -k/--token or set ESM_FORGE_TOKEN.")
        cmd += ["--token", tk]

    if quiet:
        env = os.environ.copy()
        p = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"Embedding error: {p.stderr.strip()}")
    else:
        subprocess.check_call(cmd)
