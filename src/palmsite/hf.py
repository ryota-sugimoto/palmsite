from __future__ import annotations
import os
from pathlib import Path
from typing import Tuple
from huggingface_hub import hf_hub_download

# Map backbone to default HF filename
FILE_BY_BACKBONE = {
    "300m": "palmsite_esmc_300m.pt",
    "600m": "palmsite_esmc_600m.pt",
    "6b":   "palmsite_esmc_6b.pt",
}

def resolve_hf_target(backbone: str, model_id: str | None, revision: str | None) -> Tuple[str, str, str | None]:
    mid = model_id or os.getenv("PALMSITE_MODEL_ID") or "ryota-sugimoto/palmsite"
    fname = FILE_BY_BACKBONE[backbone]
    rev = revision or os.getenv("PALMSITE_MODEL_REV") or None
    return mid, fname, rev

def fetch_weights(backbone: str, model_id: str | None, revision: str | None) -> str:
    mid, fname, rev = resolve_hf_target(backbone, model_id, revision)
    return hf_hub_download(repo_id=mid, filename=fname, revision=rev)


def resolve_weights_path(
    backbone: str,
    model_id: str | None,
    revision: str | None,
    model_pt: str | os.PathLike[str] | None = None,
) -> str:
    """
    Return the PalmSite checkpoint path to use for inference.

    Priority:
      1. Explicit local checkpoint passed via ``model_pt``
      2. Default Hugging Face download path for the selected backbone
    """
    if model_pt is not None:
        ckpt_path = Path(model_pt).expanduser()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"PalmSite checkpoint not found: {ckpt_path}")
        return str(ckpt_path.resolve())

    return fetch_weights(backbone, model_id, revision)
