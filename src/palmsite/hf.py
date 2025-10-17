from __future__ import annotations
import os
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
