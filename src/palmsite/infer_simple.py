from __future__ import annotations
import os, sys, csv, h5py, numpy as np, json
from typing import Dict, Tuple, IO, Any, Optional
from .hf import resolve_weights_path


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
    # Prefer the canonical |chunk_ format, with legacy _chunk_ accepted as fallback
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
# Key: (checkpoint_path, d_model, device_str)
_PREDICTOR_CACHE: Dict[Tuple[str, int, str], Tuple[Any, str, float]] = {}

def _get_predictor(backbone: str,
                   model_id: str | None,
                   revision: str | None,
                   model_pt: str | None,
                   d_model: int,
                   device):
    """Return a cached (model, pos_channel, temperature) tuple."""
    ckpt_path = resolve_weights_path(backbone, model_id, revision, model_pt=model_pt)
    key = (os.path.abspath(ckpt_path), int(d_model), str(device))
    if key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[key]

    import torch
    from ._predict_impl import RdRPModel

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
    model_pt: str | None = None,
    write_header: bool = True,
    return_attn: bool = False,
    pooled_json: str | None = None,
    include_pools_in_attn_json: bool = False,
    pool_include_input: bool = False,
    pool_top_k: int = 32,
    pool_l2_normalize: bool = True,
    return_artifacts: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Run PalmSite on an embeddings.h5 file and write GFF3 to out_stream.

    If attn_json is not None, also write a JSON file with per-chunk
    residue-wise attention details. If pooled_json is provided, write compact
    pooled vectors for zero-shot taxonomy/evolution analyses. The recommended
    panel is pools.backbone.span_attn_norm.

    If return_artifacts is True, return {"attn": ..., "pooled": ...}; this is
    used by the streaming CLI to write JSON incrementally.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        from ._predict_impl import (
            EmbOnlyDataset, collate, compute_pool_panels, pooled_file_meta
        )
    except ImportError as e:
        raise RuntimeError(
            "PalmSite prediction requires PyTorch. "
            "Install with: pip install 'palmsite[cpu]' (or use conda 'pytorch')."
        ) from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    d_model = _d_model_from_h5(embeddings_h5)
    model, pos_chan, T = _get_predictor(backbone, model_id, revision, model_pt, d_model, device)

    ds = EmbOnlyDataset(embeddings_h5, ids=None, pos_channel=pos_chan)
    pin = bool(device.type == "cuda")
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=pin, collate_fn=collate)

    best: Dict[str, Tuple[float, str, int, int, int, float, int]] = {}

    want_pools = (pooled_json is not None) or bool(return_artifacts) or bool(include_pools_in_attn_json)
    want_attn = (attn_json is not None) or bool(return_attn) or (bool(include_pools_in_attn_json) and want_pools)
    attn_obj: Optional[Dict[str, Any]] = {} if want_attn else None
    pool_obj: Optional[Dict[str, Any]] = {} if want_pools else None

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
            H_full = o["H"].detach().float().cpu().numpy() if want_pools else None
            input_full = x[:, :, :-1].detach().float().cpu().numpy() if (want_pools and pool_include_input) else None

            Lnp = b["L"].cpu().numpy().astype(int)
            S_idx = np.round(S * (Lnp - 1)).astype(int)
            E_idx = np.round(E * (Lnp - 1)).astype(int)
            mu_idx = np.round(mu * (Lnp - 1)).astype(int)
            orig_start = b["orig_start"].cpu().numpy().astype(int)
            orig_len = b["orig_len"].cpu().numpy().astype(int)

            for i, cid in enumerate(b["chunk_ids"]):
                Li = int(Lnp[i])
                if Li <= 0:
                    continue

                bid = _base_id(cid)
                pi = float(P[i])

                s_local = max(0, min(int(S_idx[i]), Li - 1))
                e_local = max(0, min(int(E_idx[i]), Li - 1))
                if e_local < s_local:
                    s_local, e_local = e_local, s_local

                aS = int(orig_start[i] + s_local) + 1
                aE = int(orig_start[i] + e_local) + 1
                if aE < aS:
                    aS, aE = aE, aS

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

                wi = w_full[i, :Li].astype(float, copy=False)
                ostart = int(orig_start[i])
                olen_i = int(orig_len[i])
                abs_pos = (ostart + np.arange(Li, dtype=int)).tolist()

                pools = None
                pool_meta = None
                if pool_obj is not None:
                    inp_i = input_full[i, :Li] if input_full is not None else None
                    pools, pool_meta = compute_pool_panels(
                        H_full[i, :Li],
                        wi,
                        s_local,
                        e_local,
                        input_emb=inp_i,
                        top_k=int(pool_top_k),
                        l2_normalize=bool(pool_l2_normalize),
                    )
                    pool_obj[cid] = {
                        "chunk_id": cid,
                        "base_id": bid,
                        "L": Li,
                        "orig_start": ostart,
                        "orig_len": olen_i,
                        "P": pi,
                        "S_norm": float(S[i]),
                        "E_norm": float(E[i]),
                        "S_idx": int(s_local),
                        "E_idx": int(e_local),
                        "mu": float(mu[i]),
                        "sigma": float(sigma[i]),
                        "mu_attn": float(mu_attn[i]),
                        "sigma_attn": float(sigma_attn[i]),
                        "pools": pools,
                        "pool_meta": pool_meta,
                    }

                if attn_obj is not None:
                    attn_entry = {
                        "L": Li,
                        "base_id": bid,
                        "orig_start": ostart,
                        "orig_len": olen_i,
                        "mu": float(mu[i]),
                        "sigma": float(sigma[i]),
                        "mu_attn": float(mu_attn[i]),
                        "sigma_attn": float(sigma_attn[i]),
                        "S_norm": float(S[i]),
                        "E_norm": float(E[i]),
                        "S_idx": int(s_local),
                        "E_idx": int(e_local),
                        "P": pi,
                        "w": wi.tolist(),
                        "abs_pos": abs_pos,
                    }
                    if include_pools_in_attn_json and pools is not None:
                        attn_entry["pools"] = pools
                        attn_entry["pool_meta"] = pool_meta
                    attn_obj[cid] = attn_entry

    best_chunk_by_base = {bid: vals[1] for bid, vals in best.items()}
    for obj in (attn_obj, pool_obj):
        if obj is None:
            continue
        for cid, rec in obj.items():
            if isinstance(rec, dict):
                bid = rec.get("base_id", _base_id(cid))
                rec["is_best_base_chunk"] = bool(cid == best_chunk_by_base.get(bid))

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

    if attn_obj is not None and attn_json is not None:
        os.makedirs(os.path.dirname(attn_json) or ".", exist_ok=True)
        with open(attn_json, "w", encoding="utf-8") as fjson:
            json.dump(attn_obj, fjson, indent=2)

    if pool_obj is not None and pooled_json is not None:
        os.makedirs(os.path.dirname(pooled_json) or ".", exist_ok=True)
        payload = {
            "_meta": pooled_file_meta(
                top_k=int(pool_top_k),
                l2_normalize=bool(pool_l2_normalize),
                include_input=bool(pool_include_input),
            )
        }
        payload.update(pool_obj)
        with open(pooled_json, "w", encoding="utf-8") as fjson:
            json.dump(payload, fjson, indent=2)

    if return_artifacts:
        return {"attn": attn_obj or {}, "pooled": pool_obj or {}}

    if return_attn:
        return attn_obj or {}
    return None
