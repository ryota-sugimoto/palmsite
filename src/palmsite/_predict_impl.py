#!/usr/bin/env python
"""
Predict RdRP probability and catalytic span from embeddings.h5 using a trained checkpoint,
and (optionally) export **embedding token vectors** from the predicted catalytic region
to an **HDF5** file. Also writes the final attention weights for the exported span.

Everything here matches the training model exactly in parameter names and shapes:
- backbone.mlp.*   (Linear -> ReLU -> LayerNorm -> Dropout) x3 with Linear at indices 0/4/8
- scorer.net.*     (TokenScorer wrapper)
- heads.seq_head.* / heads.span_head.* with 128 hidden units
- heads.alpha_head.* beginning with LayerNorm(256+2)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ----------------------------
# Helpers
# ----------------------------

def base_id_from_chunk(cid: str) -> str:
    # Support both legacy "_chunk_" and current "|chunk_" patterns
    if "|chunk_" in cid:
        return cid.split("|chunk_")[0]
    if "_chunk_" in cid:
        return cid.split("_chunk_")[0]
    return cid


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Stable softmax with a boolean mask (AMP-safe)."""
    mask = mask.bool()
    very_neg = torch.finfo(logits.dtype).min / 2
    logits = logits.masked_fill(~mask, very_neg)
    z = torch.softmax(logits.float(), dim=dim).to(logits.dtype)
    z = z * mask
    denom = z.sum(dim=dim, keepdim=True).clamp(min=1e-12)
    return z / denom


# ----------------------------
# Data
# ----------------------------

class EmbOnlyDataset(Dataset):
    """
    Loads per-chunk embeddings from HDF5 and appends a positional channel as in training.
    """
    def __init__(self, emb_path: str, ids: Optional[List[str]] = None, pos_channel: str = 'end_inclusive'):
        super().__init__()
        self.emb_path = emb_path
        self.ids = ids
        self.pos_channel = pos_channel
        self._index: List[str] = []
        with h5py.File(self.emb_path, "r") as h5:
            keys = list(h5["items"].keys())
            if ids is not None:
                ids_set = set(ids)
                self._index = [k for k in keys if k in ids_set]
            else:
                self._index = keys

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cid = self._index[idx]
        with h5py.File(self.emb_path, "r") as h5:
            g = h5[f"items/{cid}"]
            emb = g["emb"][:].astype(np.float32, copy=False)  # (T, d_model)
            mask = g["mask"][:].astype(bool)                  # (T,)
            # Use only valid residue tokens
            emb = emb[mask]
            L = emb.shape[0]

            # Positional channel (end-inclusive by default, as used in training)
            if self.pos_channel == 'end_inclusive':
                denom = float(max(L - 1, 1))
                pos = (np.arange(L, dtype=np.float32) / denom)[:, None]
            else:
                pos = (np.arange(L, dtype=np.float32) / float(max(L, 1)))[:, None]

            x = np.concatenate([emb, pos], axis=-1)  # (L, d_model+1)

            # Attributes for absolute coordinate mapping
            attrs = dict(g.attrs)
            orig_start = int(attrs.get("orig_aa_start", 0))
            orig_len = int(attrs.get("orig_aa_len", L))

        return {
            "chunk_id": cid,
            "x": torch.from_numpy(x),
            "mask": torch.ones(L, dtype=torch.bool),  # after trimming, all positions are valid
            "L": torch.tensor(L, dtype=torch.int32),
            "orig_start": torch.tensor(orig_start, dtype=torch.int32),
            "orig_len": torch.tensor(orig_len, dtype=torch.int32),
        }


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Pad to max T in batch
    T = max(int(b["x"].shape[0]) for b in batch)
    D = int(batch[0]["x"].shape[1])
    B = len(batch)
    x = torch.zeros(B, T, D, dtype=torch.float32)
    mask = torch.zeros(B, T, dtype=torch.bool)
    L = torch.zeros(B, dtype=torch.int32)
    chunk_ids: List[str] = []
    orig_start = torch.zeros(B, dtype=torch.int32)
    orig_len = torch.zeros(B, dtype=torch.int32)
    for i, b in enumerate(batch):
        t = int(b["x"].shape[0])
        x[i, :t] = b["x"]
        mask[i, :t] = True
        L[i] = b["L"]
        orig_start[i] = b["orig_start"]
        orig_len[i] = b["orig_len"]
        chunk_ids.append(b["chunk_id"])
    return {"x": x, "mask": mask, "L": L,
            "chunk_ids": chunk_ids, "orig_start": orig_start, "orig_len": orig_len}


# ----------------------------
# Model (EXACT training names + shapes)
# ----------------------------

class TokenBackbone(nn.Module):
    """
    EXACT training layout:
    mlp = [Linear(d_in->1024), ReLU, LayerNorm(1024), Dropout,
           Linear(1024->512), ReLU, LayerNorm(512), Dropout,
           Linear(512->256), ReLU, LayerNorm(256)]
    → Parameterized layers at indices 0,4,8 (Linear) and 2,6,10 (LayerNorm).
    """
    def __init__(self, d_in: int, p_drop: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 1024), nn.ReLU(), nn.LayerNorm(1024), nn.Dropout(p_drop),
            nn.Linear(1024, 512), nn.ReLU(), nn.LayerNorm(512), nn.Dropout(p_drop),
            nn.Linear(512, 256), nn.ReLU(), nn.LayerNorm(256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = x.view(B * T, D)
        h = self.mlp(x)
        return h.view(B, T, 256)


class TokenScorer(nn.Module):
    """EXACT training layout: scorer.net.*"""
    def __init__(self, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(64, 1),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.net(H).squeeze(-1)


class Heads(nn.Module):
    """
    EXACT training layout:
      heads.seq_head : Linear(256+2 -> 128) → ReLU → Dropout → Linear(128 -> 1)
      heads.span_head: Linear(256+2 -> 128) → ReLU → Dropout → Linear(128 -> 2)
      heads.alpha_head: LayerNorm(256+2) → Linear(256+2 -> 64) → ReLU → Linear(64 -> 1)
    """
    def __init__(self, p_drop: float = 0.0):
        super().__init__()
        self.seq_head = nn.Sequential(
            nn.Linear(256 + 2, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 1),
        )
        self.span_head = nn.Sequential(
            nn.Linear(256 + 2, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 2),
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(256 + 2),
            nn.Linear(256 + 2, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, c: torch.Tensor, len_feat: torch.Tensor):
        x = torch.cat([c, len_feat], dim=-1)
        logit = self.seq_head(x).squeeze(-1)
        span_params = self.span_head(x)
        alpha_raw = self.alpha_head(x).squeeze(-1)
        return logit, span_params, alpha_raw


class RdRPModel(nn.Module):
    """
    Model with training‑compatible parameter names:
      - backbone.mlp.*
      - scorer.net.*
      - heads.seq_head / heads.span_head / heads.alpha_head
    Runtime knobs are injected by the driver.
    """
    def __init__(self, d_in: int, tau: float = 3.0, alpha_cap: float = 2.0, p_drop: float = 0.0):
        super().__init__()
        self.tau = tau
        self.alpha_cap = alpha_cap
        self.backbone = TokenBackbone(d_in, p_drop=p_drop)
        self.scorer = TokenScorer(p_drop=p_drop)
        self.heads = Heads(p_drop=p_drop)
        # runtime knobs (set after construction):
        #   self.wmin_base, self.wmin_floor, self.seq_pool, self.lenfeat_scale
        #   self.coarse_stride, self.tau_len_gamma, self.tau_len_ref, self.k_sigma

    def _len_feat(self, L: torch.Tensor) -> torch.Tensor:
        Lf = L.to(torch.float32)
        feat = torch.stack([torch.log(Lf + 1.0), 1.0 / Lf.clamp(min=1.0)], dim=-1)
        return getattr(self, 'lenfeat_scale', 1.0) * feat

    def forward(self, x: torch.Tensor, mask: torch.Tensor, L: torch.Tensor) -> Dict[str, torch.Tensor]:
        H = self.backbone(x)            # (B,T,256)
        z = self.scorer(H)              # (B,T)

        B, T, _ = H.shape
        device = x.device
        pos = torch.arange(T, device=device, dtype=torch.float32)[None, :].repeat(B, 1)
        denom = (L.to(torch.float32) - 1.0).clamp(min=1.0).unsqueeze(1)
        pos = (pos / denom).clamp(0.0, 1.0)

        # Per-sequence z standardization (as in training)
        valid = mask.to(z.dtype)
        z_mean = (z * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        z_var  = ((z - z_mean)**2 * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        z_std  = z_var.add(1e-6).sqrt()
        z_norm = (z - z_mean) / z_std

        # Length‑aware tau scaling (optional)
        tau = torch.as_tensor(getattr(self, 'tau', 3.0), dtype=z.dtype, device=device)
        if getattr(self, 'tau_len_gamma', 0.0) != 0.0:
            Lf = L.to(torch.float32)
            tau = tau * ((Lf / float(getattr(self, 'tau_len_ref', 1000.0))).clamp(min=0.25, max=4.0)
                         ** float(getattr(self, 'tau_len_gamma', 0.0)))
            tau = tau.view(-1, 1)   # per‑sequence τ
        else:
            tau = tau.view(1, 1)    # scalar τ

        # Anchor weights on normalized z
        w_anchor = masked_softmax(z_norm / tau, mask, dim=1)

        # Optional coarse‑to‑fine anchor (single pass “auto‑zoom”)
        coarse_stride = int(getattr(self, 'coarse_stride', 0) or 0)
        if coarse_stride >= 2 and T >= 2:
            very_neg = torch.finfo(z.dtype).min / 2
            z_masked_norm = z_norm.masked_fill(~mask, very_neg).unsqueeze(1)                      # (B,1,T)
            zc = F.max_pool1d(z_masked_norm, kernel_size=coarse_stride, stride=coarse_stride,
                              ceil_mode=True).squeeze(1)                                          # (B,Tc)
            mc = F.max_pool1d(mask.unsqueeze(1).to(z.dtype), kernel_size=coarse_stride,
                              stride=coarse_stride, ceil_mode=True).squeeze(1) > 0                # (B,Tc)
            wc = masked_softmax(zc / (tau if tau.numel() > 1 else tau), mc, dim=1)                # (B,Tc)
            Tc = zc.shape[1]
            posc = torch.arange(Tc, device=device, dtype=torch.float32)[None, :].repeat(B, 1)     # (B,Tc)
            denomc = (torch.ceil(L.to(torch.float32) / coarse_stride) - 1.0).clamp(min=1.0).unsqueeze(1)
            muc = (wc * (posc / denomc)).sum(dim=1)
            varc = (wc * ((posc / denomc - muc.unsqueeze(1)) ** 2)).sum(dim=1)
            sigc = torch.sqrt(varc + 1e-8)
            mu_init, sigma_init = muc, sigc
        else:
            mu_init = sigma_init = None

        if mu_init is not None:
            mu = mu_init
            sigma = sigma_init
        else:
            mu = (w_anchor * pos).sum(dim=1)
            var = (w_anchor * ((pos - mu.unsqueeze(1)) ** 2)).sum(dim=1)
            sigma = torch.sqrt(var + 1e-8)

        # σ minimum width and clamp
        wmin_base = torch.tensor(getattr(self, 'wmin_base', 70.0), device=device, dtype=torch.float32)
        wmin_floor = torch.tensor(getattr(self, 'wmin_floor', 0.02), device=device, dtype=torch.float32)
        w_min = torch.maximum(wmin_base / L.to(torch.float32), wmin_floor)
        sigma = torch.maximum(sigma, w_min)
        sigma = torch.clamp(sigma, max=0.5)

        # Contexts
        c_anchor = torch.einsum('bt,btc->bc', w_anchor.float(), H.float()).to(H.dtype)
        lenf = self._len_feat(L)

        # α from anchor context + length features (softplus, capped)
        alpha_raw = self.heads.alpha_head(torch.cat([c_anchor, lenf], dim=-1)).squeeze(-1)
        alpha = F.softplus(alpha_raw)
        if self.alpha_cap is not None:
            alpha = torch.clamp(alpha, max=self.alpha_cap)

        # Final attention and Gaussian-pooled context
        q = -0.5 * ((pos - mu.unsqueeze(1)) / sigma.unsqueeze(1)).pow(2)
        z_center = z - (z * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        a = z_center + alpha.unsqueeze(1) * q
        w = masked_softmax(a, mask, dim=1)

        c_gauss = torch.einsum('bt,btc->bc', w.float(), H.float()).to(H.dtype)
        seq_logit = self.heads.seq_head(torch.cat([c_gauss, lenf], dim=-1)).squeeze(-1)

        # Span from attention mass (k·sigma over w)
        mu_attn = (w * pos).sum(dim=1)
        var_attn = (w * ((pos - mu_attn.unsqueeze(1)) ** 2)).sum(dim=1)
        sigma_attn = torch.sqrt(var_attn + 1e-8)
        sigma_attn = torch.maximum(sigma_attn, w_min)
        sigma_attn = torch.clamp(sigma_attn, max=0.5)
        k = getattr(self, 'k_sigma', 2.0)
        S = torch.clamp(mu_attn - k * sigma_attn, 0.0, 1.0)
        E = torch.clamp(mu_attn + k * sigma_attn, 0.0, 1.0)

        return {
            "logit": seq_logit,
            "P": torch.sigmoid(seq_logit),
            "S_pred": S, "E_pred": E,
            "mu": mu, "sigma": sigma, "alpha": alpha,
            "mu_attn": mu_attn, "sigma_attn": sigma_attn,
            "w": w,
        }


# ----------------------------
# HPD helper
# ----------------------------

def _attn_hpd_span_from_out(out: Dict[str, torch.Tensor], mask: torch.Tensor, L: torch.Tensor,
                            mass: float = 0.90, pos_channel: str = 'end_inclusive') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Highest Posterior Density (HPD) span that covers `mass` of attention."""
    w = out["w"].detach().cpu().numpy()        # (B,T)
    mask_np = mask.cpu().numpy()
    Lnp = L.cpu().numpy().astype(int)
    B, T = w.shape
    S = np.zeros(B, dtype=np.float32)
    E = np.ones(B, dtype=np.float32)
    ent = np.zeros(B, dtype=np.float32)

    for i in range(B):
        Li = int(mask_np[i].sum())
        if Li <= 0:
            S[i], E[i], ent[i] = 0.0, 0.0, 0.0
            continue
        wi = w[i, :Li].astype(np.float64, copy=False)
        s = wi.sum()
        if s <= 0:
            wi = np.ones((Li,), dtype=np.float64) / float(Li)
        else:
            wi = wi / s
        # shortest contiguous interval [l,r] with mass >= target
        c = np.cumsum(wi)
        best_len = Li + 1
        best = (0, Li - 1)
        l = 0
        for r in range(Li):
            cur = c[r] - (c[l - 1] if l > 0 else 0.0)
            while cur - mass >= -1e-12 and l <= r:
                clen = r - l + 1
                if clen < best_len:
                    best_len = clen
                    best = (l, r)
                l += 1
                cur = c[r] - (c[l - 1] if l > 0 else 0.0)
        l_idx, r_idx = best
        denom = float(max(Lnp[i] - 1, 1)) if pos_channel == 'end_inclusive' else float(max(Lnp[i], 1))
        S[i] = float(l_idx) / denom
        E[i] = float(r_idx) / denom
        w_safe = np.clip(wi, 1e-12, 1.0)
        ent[i] = float(-(w_safe * np.log(w_safe)).sum() / np.log(max(Li, 2)))
    return S, E, ent


# ----------------------------
# Core execution (shared by CLI and library)
# ----------------------------

def _run_with_namespace(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint (prefer weights_only when available)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

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

    # Determine d_model from HDF5
    with h5py.File(args.embeddings, "r") as h5:
        any_id = next(iter(h5["items"].keys()))
        g = h5[f"items/{any_id}"]
        d_model = int(g.attrs.get("d_model", g["emb"].shape[1]))

    # Build model
    model = RdRPModel(d_in=d_model + 1, tau=tau, alpha_cap=alpha_cap, p_drop=p_drop).to(device)
    # Inject runtime knobs (no layer name changes!)
    model.wmin_base, model.wmin_floor = wmin_base, wmin_floor
    model.lenfeat_scale = len_scale
    model.k_sigma = k_sigma
    model.coarse_stride = coarse
    model.tau_len_gamma = tau_gam
    model.tau_len_ref = tau_ref

    # Strip wrappers like "module." / "_orig_mod."
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    cleaned = {}
    for k, v in state.items():
        kk = k
        for p in ("_orig_mod.", "module."):
            if kk.startswith(p):
                kk = kk[len(p):]
        cleaned[kk] = v

    # Load with strict=True (names/shapes now match training)
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    # Temperature (prob calibration)
    T = float(ckpt.get("temperature", 1.0)) if isinstance(ckpt, dict) else 1.0

    # IDs subset (optional)
    ids = None
    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as f:
            ids = [ln.strip() for ln in f if ln.strip()]

    ds = EmbOnlyDataset(args.embeddings, ids=ids, pos_channel='end_inclusive')
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False,
                    num_workers=int(args.num_workers), pin_memory=True, collate_fn=collate)

    # Open CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fcsv = open(args.out, "w", newline="", encoding="utf-8")
    wcsv = csv.writer(fcsv)
    wcsv.writerow(["chunk_id", "P", "S", "E", "S_idx", "E_idx", "mu", "mu_idx", "sigma", "L"])

    # Aggregate best per base_id
    agg: Dict[str, Tuple[float, str, int, int, int, float, int, float, float, float]] = {}

    with torch.no_grad():
        for b in dl:
            x = b["x"].to(device)
            m = b["mask"].to(device)
            L = b["L"].to(device)
            out = model(x, m, L)
            P = torch.sigmoid(out["logit"] / T).cpu().numpy()

            if args.discovery:
                S, E, ent = _attn_hpd_span_from_out(out, m, L, mass=float(args.attn_mass), pos_channel='end_inclusive')
            else:
                S = out["S_pred"].detach().cpu().numpy()
                E = out["E_pred"].detach().cpu().numpy()
                ent = np.zeros_like(S, dtype=np.float32)

            mu    = out["mu"].detach().cpu().numpy()
            sigma = out["sigma"].detach().cpu().numpy()
            Lnp   = b["L"].detach().cpu().numpy().astype(int)
            S_idx = np.round(S * (Lnp - 1)).astype(int)
            E_idx = np.round(E * (Lnp - 1)).astype(int)
            mu_idx= np.round(mu * (Lnp - 1)).astype(int)
            orig_start = b["orig_start"].detach().cpu().numpy().astype(int)
            orig_len   = b["orig_len"].detach().cpu().numpy().astype(int)

            for i, cid in enumerate(b["chunk_ids"]):
                wcsv.writerow([cid, float(P[i]), float(S[i]), float(E[i]),
                               int(S_idx[i]), int(E_idx[i]), float(mu[i]), int(mu_idx[i]),
                               float(sigma[i]), int(Lnp[i])])

                bid = base_id_from_chunk(cid)
                aS = int(orig_start[i] + S_idx[i]) + 1  # 1-based for GFF (CSV keeps chunk idx)
                aE = int(orig_start[i] + E_idx[i]) + 1
                if aE < aS:
                    aS, aE = aE, aS
                cand = (float(P[i]), cid, int(aS), int(aE),
                        int(orig_start[i] + mu_idx[i]) + 1,
                        float(sigma[i]), int(orig_len[i]),
                        float(S[i]), float(E[i]), float(ent[i]))
                cur = agg.get(bid)
                if (cur is None) or (cand[0] > cur[0]):
                    agg[bid] = cand

    fcsv.close()

    # Base-level TSV
    if args.base_out:
        os.makedirs(os.path.dirname(args.base_out) or ".", exist_ok=True)
        with open(args.base_out, "w", newline="", encoding="utf-8") as f:
            bw = csv.writer(f)
            bw.writerow(["base_id","P","best_chunk","S_abs","E_abs","mu_abs","sigma","Lorig","S","E","attn_entropy"])
            for bid, v in agg.items():
                bw.writerow([bid] + list(v))

    # GFF (support "-" for stdout)
    if args.gff_out:
        if args.gff_out == "-":
            fgff = sys.stdout
            close_needed = False
        else:
            os.makedirs(os.path.dirname(args.gff_out) or ".", exist_ok=True)
            fgff = open(args.gff_out, "w", encoding="utf-8")
            close_needed = True
        try:
            fgff.write("##gff-version 3\n")
            n_feat = 0
            for bid, (Pmax, cid, aS, aE, aMu, sig, Lorig, Srel, Erel, ent) in agg.items():
                if Pmax < float(args.gff_min_P):
                    continue
                attrs = [f"ID={bid}", "Name=RdRP_catalytic_center",
                         f"Chunk={cid}", f"P={Pmax:.6f}", f"sigma={sig:.4f}", f"len={Lorig}",
                         f"SpanSource={'HPD' if args.discovery else 'kSigma'}",
                         f"AttnMass={args.attn_mass:.2f}", f"AttnEntropy={ent:.6f}"]
                row = [bid, args.gff_source, args.gff_type, str(aS), str(aE),
                       f"{Pmax:.6f}", ".", ".", ";".join(attrs)]
                fgff.write("\t".join(row) + "\n")
                n_feat += 1
            if fgff is not sys.stdout:
                print(f"Wrote GFF3 with {n_feat} features to: {args.gff_out}")
        finally:
            if close_needed:
                fgff.close()


# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Predict RdRP probability and span from embeddings.h5 (+ optional GFF and base-level TSV)'
    )
    ap.add_argument("--embeddings", required=True, help="embeddings.h5")
    ap.add_argument("--checkpoint", required=True, help="trained .pt")
    ap.add_argument("--out", required=True, help="Per-chunk CSV output")
    ap.add_argument("--ids-file", default=None, help="Optional list of chunk IDs to process")

    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=2)

    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-calib", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)

    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--seq_pool", type=str, default=None)
    ap.add_argument("--wmin_base", type=float, default=None)
    ap.add_argument("--wmin_floor", type=float, default=None)
    ap.add_argument("--lenfeat_scale", type=float, default=None)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--dump_weights_id", default=None)

    ap.add_argument("--base-out", default=None, help="Base-level TSV (best span per base_id)")
    ap.add_argument("--gff-out", default=None, help="GFF3 output (use '-' for stdout)")
    ap.add_argument("--gff-min-P", type=float, default=0.0)
    ap.add_argument("--gff-source", default="PalmSite")
    ap.add_argument("--gff-type", default="RdRP_domain")
    ap.add_argument("--pos_channel", default=None)

    ap.add_argument("--discovery", action="store_true", help="Use HPD span instead of k·sigma")
    ap.add_argument("--attn_mass", type=float, default=0.90)

    ap.add_argument("--window-scan", type=int, default=None)
    ap.add_argument("--window-stride", type=int, default=50)
    ap.add_argument("--coarse-stride", type=int, default=None)

    ap.add_argument("--extract-h5", default=None, help="(Deprecated here) — kept for compatibility; vectors are exported by the top-level CLI")
    ap.add_argument("--min-p", type=float, default=0.90)
    ap.add_argument("--h5-coords", choices=["abs", "chunk"], default="abs")

    args = ap.parse_args()
    _run_with_namespace(args)


# ----------------------------
# Library entrypoint for PalmSite’s top-level CLI (optional)
# ----------------------------

def predict_from_h5(embeddings: str, checkpoint: str, out_csv: str,
                    gff_out: str | None = None, gff_min_p: float = 0.0,
                    ids_file: str | None = None) -> None:
    """Library-friendly wrapper that calls this module as a CLI in-process."""
    import subprocess
    cmd = [sys.executable, '-m', 'palmsite._predict_impl',
           '--embeddings', embeddings,
           '--checkpoint', checkpoint,
           '--out', out_csv,
           '--gff-min-P', str(gff_min_p)]
    if gff_out is not None:
        cmd += ['--gff-out', gff_out]
    if ids_file is not None:
        cmd += ['--ids-file', ids_file]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"predict_from_h5 failed: {proc.stderr.strip()}")
    return


if __name__ == '__main__':
    main()

