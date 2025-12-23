from __future__ import annotations
import os, sys, csv, h5py, numpy as np, json
from typing import Dict, Tuple, IO, Any, Optional
from .hf import fetch_weights


def _strip_prefixes(sd, prefixes=('_orig_mod.', 'module.')):
    out = {}
    for k, v in sd.items():
        kk = k
        for p in prefixes:
            if kk.startswith(p):
                kk = kk[len(p):]
        out[kk] = v
    return out


def _base_id(cid: str) -> str:
    # handle both patterns from your pipelines
    if "|chunk_" in cid:   # produced by your embedding script
        return cid.split("|chunk_")[0]
    if "_chunk_" in cid:   # legacy
        return cid.split("_chunk_")[0]
    return cid


def _d_model_from_h5(emb_h5: str) -> int:
    with h5py.File(emb_h5, "r") as h5:
        any_id = next(iter(h5["items"].keys()))
        g = h5[f"items/{any_id}"]
        return int(g.attrs.get("d_model", g["emb"].shape[1]))



# Predictor cache: avoid re-loading the PalmSite checkpoint for every micro-batch.
# Key: (backbone, model_id, revision, d_model, device_str)
_PREDICTOR_CACHE: Dict[Tuple[str, Optional[str], Optional[str], int, str], Tuple[Any, str, float]] = {}

def _get_predictor(backbone: str,
                   model_id: str | None,
                   revision: str | None,
                   d_model: int,
                   device):
    """Return a cached (model, pos_channel, temperature) tuple."""
    key = (str(backbone), model_id, revision, int(d_model), str(device))
    if key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[key]

    import torch
    from ._predict_impl import RdRPModel

    ckpt_path = fetch_weights(backbone, model_id, revision)

    # Load checkpoint (safe when available)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    cfg = ckpt.get("cfg", {}) if isinstance(ckpt, dict) else {}
    p_drop     = float(cfg.get("dropout", 0.1))
    tau        = float(cfg.get("tau", 3.0))
    alpha_cap  = float(cfg.get("alpha_cap", 2.0))
    wmin_base  = float(cfg.get("wmin_base", 70.0))
    wmin_floor = float(cfg.get("wmin_floor", 0.02))
    len_scale  = float(cfg.get("lenfeat_scale", 1.0))
    k_sigma    = float(cfg.get("k_sigma", 2.0))
    coarse     = int(cfg.get("coarse_stride", 0))
    tau_gam    = float(cfg.get("tau_len_gamma", 0.0))
    tau_ref    = float(cfg.get("tau_len_ref", 1000.0))
    pos_chan   = str(cfg.get("pos_channel", "end_inclusive"))

    model = RdRPModel(d_in=int(d_model) + 1, tau=tau, alpha_cap=alpha_cap, p_drop=p_drop).to(device)
    model.wmin_base, model.wmin_floor = wmin_base, wmin_floor
    model.lenfeat_scale = len_scale
    model.k_sigma = k_sigma
    model.coarse_stride = coarse
    model.tau_len_gamma = tau_gam
    model.tau_len_ref = tau_ref

    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    model.load_state_dict(_strip_prefixes(state), strict=True)
    model.eval()

    # Temperature for calibrated P (if present)
    T = float(ckpt.get("temperature", 1.0)) if isinstance(ckpt, dict) else 1.0

    _PREDICTOR_CACHE[key] = (model, pos_chan, T)
    return _PREDICTOR_CACHE[key]


def predict_to_gff(
    embeddings_h5: str,
    backbone: str,
    model_id: str | None,
    revision: str | None,
    min_p: float,
    out_stream: IO[str],
    attn_json: str | None = None,
    *,
    write_header: bool = True,
    return_attn: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run PalmSite on an embeddings.h5 file and write GFF3 to out_stream.

    If attn_json is not None, also write a JSON file with per-chunk
    residue-wise attention details (same schema as predict.py/_predict_impl).

    If return_attn is True, return that attention dictionary to the caller
    (useful for streaming / micro-batch aggregation).
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        # Import engine only now (it imports torch at module level)
        from ._predict_impl import (
            EmbOnlyDataset, collate
        )
    except ImportError as e:
        raise RuntimeError(
            "PalmSite prediction requires PyTorch. "
            "Install with: pip install 'palmsite[cpu]' (or use conda 'pytorch')."
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model = _d_model_from_h5(embeddings_h5)
    model, pos_chan, T = _get_predictor(backbone, model_id, revision, d_model, device)

    ds = EmbOnlyDataset(embeddings_h5, ids=None, pos_channel=pos_chan)
    # Use single-process DataLoader to avoid forking after HuggingFace tokenizers initialization
    # (which can emit warnings and, in rare cases, lead to deadlocks).
    pin = bool(device.type == "cuda")
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=pin, collate_fn=collate)

    # Aggregate best per original sequence
    best: Dict[str, Tuple[float, str, int, int, int, float, int]] = {}

    # Optional per-chunk residue-wise attention JSON
    want_attn = (attn_json is not None) or bool(return_attn)
    attn_obj: Optional[Dict[str, Any]] = {} if want_attn else None

    if write_header:
        out_stream.write("##gff-version 3\n")

    with torch.no_grad():
        for b in dl:
            x = b["x"].to(device)
            m = b["mask"].to(device)
            L = b["L"].to(device)
            o = model(x, m, L)

            P = torch.sigmoid(o["logit"] / T).cpu().numpy()
            S = o["S_pred"].cpu().numpy()
            E = o["E_pred"].cpu().numpy()
            mu = o["mu"].cpu().numpy()
            sigma = o["sigma"].cpu().numpy()
            mu_attn = o["mu_attn"].cpu().numpy()
            sigma_attn = o["sigma_attn"].cpu().numpy()
            w_full = o["w"].cpu().numpy()

            Lnp = b["L"].cpu().numpy().astype(int)
            S_idx = np.round(S * (Lnp - 1)).astype(int)
            E_idx = np.round(E * (Lnp - 1)).astype(int)
            mu_idx = np.round(mu * (Lnp - 1)).astype(int)
            orig_start = b["orig_start"].cpu().numpy().astype(int)
            orig_len   = b["orig_len"].cpu().numpy().astype(int)

            for i, cid in enumerate(b["chunk_ids"]):
                # Absolute coords on original protein (1-based in GFF)
                aS = int(orig_start[i] + S_idx[i]) + 1
                aE = int(orig_start[i] + E_idx[i]) + 1
                if aE < aS:
                    aS, aE = aE, aS
                bid = _base_id(cid)
                pi = float(P[i])

                if (bid not in best) or (pi > best[bid][0]):
                    best[bid] = (
                        pi,
                        cid,
                        aS,
                        aE,
                        int(orig_start[i] + mu_idx[i]) + 1,
                        float(sigma[i]),
                        int(orig_len[i]),
                    )

                # Optional per-chunk attention JSON (same schema as predict.py/_predict_impl)
                if attn_obj is not None:
                    Li = int(Lnp[i])
                    if Li > 0:
                        wi = w_full[i, :Li].astype(float, copy=False)
                        ostart = int(orig_start[i])
                        olen_i = int(orig_len[i])
                        abs_pos = (ostart + np.arange(Li, dtype=int)).tolist()

                        attn_obj[cid] = {
                            "L": Li,
                            "orig_start": ostart,
                            "orig_len": olen_i,
                            # anchor-based parameters
                            "mu": float(mu[i]),
                            "sigma": float(sigma[i]),
                            # final-attention-based parameters
                            "mu_attn": float(mu_attn[i]),
                            "sigma_attn": float(sigma_attn[i]),
                            # span info (normalized + indices)
                            "S_norm": float(S[i]),
                            "E_norm": float(E[i]),
                            "S_idx": int(S_idx[i]),
                            "E_idx": int(E_idx[i]),
                            # probability
                            "P": float(P[i]),
                            # per-residue attention + absolute positions
                            "w": wi.tolist(),
                            "abs_pos": abs_pos,
                        }

    # Emit GFF rows with min-p filter
    n = 0
    for bid, (Pmax, cid, aS, aE, aMu, sig, Lorig) in best.items():
        if Pmax < float(min_p):
            continue
        attrs = [
            f"ID={bid}",
            "Name=RdRP_catalytic_center",
            f"Chunk={cid}",
            f"P={Pmax:.6f}",
            f"mu={aMu}",
            f"sigma={sig:.4f}",
            f"len={Lorig}",
        ]
        row = [bid, "PalmSite", "RdRP_domain", str(aS), str(aE), f"{Pmax:.6f}", ".", ".", ";".join(attrs)]
        out_stream.write("\t".join(row) + "\n")
        n += 1

    if out_stream is not sys.stdout:
        out_stream.flush()

    # Write attention JSON if requested
    if attn_obj is not None and attn_json is not None:
        os.makedirs(os.path.dirname(attn_json) or ".", exist_ok=True)
        with open(attn_json, "w", encoding="utf-8") as fjson:
            json.dump(attn_obj, fjson, indent=2)

    if return_attn:
        # Always return a dict for caller convenience
        return attn_obj or {}
    return None
