#!/usr/bin/env python
"""
Predict RdRP probability and catalytic span from embeddings.h5 using a trained checkpoint,
and (optionally) export **embedding token vectors** from the predicted catalytic region
to an **HDF5** file. Now also saves the **final attention weights** for the exported span,
and (optionally) per-residue attention weights + mu/sigma to a JSON file.

New flags:
  --extract-h5 <path>   Write HDF5 with groups under /items/<key>:
                        - pos : int32 [N]          (positions; 0-based)
                        - vec : float32 [N, d_model]  (embedding vectors; no positional channel)
                        - w   : float32 [N]        (final attention weights aligned to pos/vec)
                        <key> uses chunk_id for uniqueness; attributes also include base_id.
  --min-p <float>       Only export vectors for chunks whose predicted P ≥ threshold (default: 0.90).
  --h5-coords {abs,chunk}
                        Which coordinate system to write for 'pos':
                        - abs   : base_id + absolute 0-based AA index (default)
                        - chunk : chunk_id + chunk-local 0-based index
  --attn-json <path>    Write residue-wise attention weights and mu/sigma per chunk as JSON.

Discovery mode and window scan are supported; HDF5 export is based on the per-chunk span S..E.

Everything else mirrors training:
- Sequence score from Gaussian-pooled context
- Span derived from final attention mass (k·sigma)
- σ clamp: sigma >= max(wmin_base/L, wmin_floor), and sigma <= 0.5
- α from anchor context + length features (softplus, capped)
"""
from __future__ import annotations
import os
import csv
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# AMP (deprecation-free API with fallback)
try:
    from torch.amp import autocast, GradScaler
    _NEW_AMP = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler
    _NEW_AMP = False


# Enable TF32 on Ampere+ to match training speed/behavior
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        # PyTorch 2.x
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass

def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax over valid positions only; zeros elsewhere. Matches training stability (AMP-safe)."""
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
    def __init__(self, emb_path: str, ids: Optional[List[str]] = None, pos_channel: str = 'end_inclusive'):
        super().__init__()
        self.emb_path = emb_path
        self._h5: Optional[h5py.File] = None
        with h5py.File(emb_path, 'r') as h5e:
            all_ids = list(h5e['items'].keys())
        if ids is None:
            self.ids = all_ids
        else:
            ids_set = set(ids)
            self.ids = [cid for cid in all_ids if cid in ids_set]
        self.pos_channel = pos_channel

    def __len__(self) -> int:
        return len(self.ids)

    def _open(self) -> None:
        if self._h5 is None:
            self._h5 = h5py.File(self.emb_path, 'r')

    def _make_pos(self, L: int) -> np.ndarray:
        if L <= 1:
            return np.zeros((1,), dtype=np.float32)
        if self.pos_channel == 'end_inclusive':
            return (np.arange(L, dtype=np.float32) / float(L - 1)).astype(np.float32)
        else:
            return (np.arange(L, dtype=np.float32) / float(L)).astype(np.float32)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        self._open()
        cid = self.ids[i]
        grp = self._h5[f'items/{cid}']
        emb = grp['emb'][...].astype(np.float32, copy=False)
        mask = grp['mask'][...]
        if mask.dtype != np.bool_:
            mask = mask.astype(bool, copy=False)
        emb = emb[mask]  # only residue tokens
        L = emb.shape[0]
        pos = self._make_pos(L)[:, None]
        x = np.concatenate([emb, pos], axis=1)  # (L, d_model+1)
        # Attributes for absolute mapping
        orig_start = int(grp.attrs.get('orig_aa_start', 0))
        orig_len = int(grp.attrs.get('orig_aa_len', grp.attrs.get('aa_len', L)))
        return {
            'chunk_id': cid,
            'x': torch.from_numpy(x),
            'mask': torch.ones(L, dtype=torch.bool),
            'L': torch.tensor(L, dtype=torch.long),
            'orig_start': torch.tensor(orig_start, dtype=torch.long),
            'orig_len': torch.tensor(orig_len, dtype=torch.long),
        }


def collate(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(s['x'].shape[0] for s in samples)
    d = samples[0]['x'].shape[1]
    B = len(samples)
    x = torch.zeros(B, max_len, d, dtype=torch.float32)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    L = torch.zeros(B, dtype=torch.long)
    orig_start = torch.zeros(B, dtype=torch.long)
    orig_len = torch.zeros(B, dtype=torch.long)
    chunk_ids: List[str] = []
    for i, s in enumerate(samples):
        Li = s['x'].shape[0]
        x[i, :Li] = s['x']
        mask[i, :Li] = True
        L[i] = s['L']
        orig_start[i] = s['orig_start']
        orig_len[i] = s['orig_len']
        chunk_ids.append(s['chunk_id'])
    return {'x': x, 'mask': mask, 'L': L, 'chunk_ids': chunk_ids, 'orig_start': orig_start, 'orig_len': orig_len}

# ----------------------------
# Model (must match training layers; logic knobs are runtime attributes)
# ----------------------------
class TokenBackbone(nn.Module):
    def __init__(self, d_in: int, p_drop: float = 0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_in, 1024), nn.ReLU(), nn.LayerNorm(1024), nn.Dropout(p_drop),
            nn.Linear(1024, 512), nn.ReLU(), nn.LayerNorm(512), nn.Dropout(p_drop),
            nn.Linear(512, 256), nn.ReLU(), nn.LayerNorm(256),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x = x.view(B*T, D)
        h = self.mlp(x)
        return h.view(B, T, 256)

class TokenScorer(nn.Module):
    def __init__(self, p_drop: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(64, 1)
        )
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        return self.net(H).squeeze(-1)

class Heads(nn.Module):
    def __init__(self, p_drop: float = 0.0):
        super().__init__()
        self.seq_head = nn.Sequential(
            nn.Linear(256 + 2, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 1)
        )
        self.span_head = nn.Sequential(
            nn.Linear(256 + 2, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 2)
        )
        # Make alpha depend on sequence length features too
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(256 + 2),
            nn.Linear(256 + 2, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, c: torch.Tensor, len_feat: torch.Tensor):
        x = torch.cat([c, len_feat], dim=-1)
        logit = self.seq_head(x).squeeze(-1)
        span_params = self.span_head(x)
        alpha_raw = self.alpha_head(x).squeeze(-1)
        return logit, span_params, alpha_raw

class RdRPModel(nn.Module):
    def __init__(self, d_in: int, tau: float = 3.0, alpha_cap: float = 2.0, p_drop: float = 0.0):
        super().__init__()
        self.tau = tau
        self.alpha_cap = alpha_cap
        self.backbone = TokenBackbone(d_in, p_drop=p_drop)
        self.scorer = TokenScorer(p_drop=p_drop)
        self.heads = Heads(p_drop=p_drop)
        # runtime knobs injected after construction:
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
        # ----- Per-sequence z standardization (Patch B) -----
        valid = mask.to(z.dtype)
        z_mean = (z * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        z_var  = ((z - z_mean)**2 * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        z_std  = z_var.add(1e-6).sqrt()
        z_norm = (z - z_mean) / z_std
        # ----- Length-aware tau scaling (optional) -----
        tau = torch.as_tensor(getattr(self, 'tau', 3.0), dtype=z.dtype, device=device)
        if getattr(self, 'tau_len_gamma', 0.0) != 0.0:
            Lf = L.to(torch.float32)
            tau = tau * ((Lf / float(getattr(self, 'tau_len_ref', 1000.0))).clamp(min=0.25, max=4.0) ** float(getattr(self, 'tau_len_gamma', 0.0)))
            tau = tau.view(-1, 1)   # per-sequence τ
        else:
            tau = tau.view(1, 1)    # scalar τ
        # Anchor weights on normalized z
        w_anchor = masked_softmax(z_norm / tau, mask, dim=1)
        # ----- Optional coarse-to-fine anchor (single pass “auto-zoom”) -----
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
        # σ minimum width
        wmin_base = torch.tensor(getattr(self, 'wmin_base', 70.0), device=device, dtype=torch.float32)
        wmin_floor = torch.tensor(getattr(self, 'wmin_floor', 0.02), device=device, dtype=torch.float32)
        w_min = torch.maximum(wmin_base / L.to(torch.float32), wmin_floor)
        sigma = torch.maximum(sigma, w_min)
        sigma = torch.clamp(sigma, max=0.5)
        # Contexts
        c_anchor = torch.einsum('bt,btc->bc', w_anchor.float(), H.float()).to(H.dtype)
        # α depends on anchor context + length features (Patch A parity)
        lenf = self._len_feat(L)
        alpha_raw = self.heads.alpha_head(torch.cat([c_anchor, lenf], dim=-1)).squeeze(-1)
        alpha = F.softplus(alpha_raw)
        if self.alpha_cap is not None:
            alpha = torch.clamp(alpha, max=self.alpha_cap)
        q = -0.5 * ((pos - mu.unsqueeze(1)) / sigma.unsqueeze(1)).pow(2)  # (B,T)
        # Center z per sequence (valid positions only) to match training
        valid = mask.to(z.dtype)
        z_center = z - (z * valid).sum(dim=1, keepdim=True) / (valid.sum(dim=1, keepdim=True).clamp(min=1.0))
        a = z_center + alpha.unsqueeze(1) * q
        w = masked_softmax(a, mask, dim=1)
        c_gauss = torch.einsum('bt,btc->bc', w.float(), H.float()).to(H.dtype)
        # Always use Gaussian-pooled context for sequence score (matches training)
        seq_logit = self.heads.seq_head(torch.cat([c_gauss, lenf], dim=-1)).squeeze(-1)
        # --- Span from attention mass (matches training) ---
        mu_attn = (w * pos).sum(dim=1)
        var_attn = (w * ((pos - mu_attn.unsqueeze(1)) ** 2)).sum(dim=1)
        sigma_attn = torch.sqrt(var_attn + 1e-8)
        sigma_attn = torch.maximum(sigma_attn, w_min)
        sigma_attn = torch.clamp(sigma_attn, max=0.5)
        k = getattr(self, 'k_sigma', 2.0)
        S = torch.clamp(mu_attn - k * sigma_attn, 0.0, 1.0)
        E = torch.clamp(mu_attn + k * sigma_attn, 0.0, 1.0)
        return {
            'logit': seq_logit,
            'P': torch.sigmoid(seq_logit),
            'S_pred': S, 'E_pred': E,
            'mu': mu, 'sigma': sigma, 'alpha': alpha,
            'mu_attn': mu_attn, 'sigma_attn': sigma_attn,
            'w': w,
        }

# ----------------------------
# Helpers
# ----------------------------
def _norm_entropy(w: np.ndarray) -> float:
    """Normalized entropy in [0,1]: -Σ w log w / log L (w must be >=0 and sum to 1 over valid)."""
    w = np.clip(w, 1e-12, 1.0)
    H = float(-(w * np.log(w)).sum())
    L = max(len(w), 1)
    return float(H / np.log(L)) if L > 1 else 0.0

def _hpd_span_indices(w: np.ndarray, mass: float = 0.90) -> Tuple[int, int]:
    """
    Shortest contiguous interval [l,r] whose mass >= 'mass'. Assumes w sums to 1 over valid.
    Two-pointer sweep, O(L). Returns 0-based inclusive indices.
    """
    L = len(w)
    if L == 0:
        return 0, 0
    # prefix sums
    c = np.cumsum(w)
    best_len = L + 1
    best = (0, L - 1)
    l = 0
    for r in range(L):
        # mass in [l,r]
        cur = c[r] - (c[l-1] if l > 0 else 0.0)
        while cur - mass >= -1e-12 and l <= r:
            # update best
            clen = r - l + 1
            if clen < best_len:
                best_len = clen
                best = (l, r)
            l += 1
            cur = c[r] - (c[l-1] if l > 0 else 0.0)
    return best

def _attn_hpd_span_from_out(out: Dict[str, torch.Tensor],
                            mask: torch.Tensor,
                            L: torch.Tensor,
                            mass: float,
                            pos_channel: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute HPD spans per batch from attention 'w' in model output.
    Returns (S_norm, E_norm, ent_norm) as numpy arrays of shape (B,).
    """
    w = out['w'].detach().cpu().numpy()              # (B,T)
    mask_np = mask.detach().cpu().numpy()            # (B,T)
    Lnp = L.detach().cpu().numpy().astype(int)       # (B,)
    B, T = w.shape
    S = np.zeros((B,), dtype=np.float32)
    E = np.zeros((B,), dtype=np.float32)
    ent = np.zeros((B,), dtype=np.float32)
    for i in range(B):
        Li = int(mask_np[i].sum())
        if Li <= 0:
            S[i] = 0.0; E[i] = 0.0; ent[i] = 0.0
            continue
        wi = w[i, :Li].astype(np.float64, copy=False)
        # re-normalize over valid just in case
        s = wi.sum()
        if s <= 0:
            wi = np.ones((Li,), dtype=np.float64) / float(Li)
        else:
            wi = wi / s
        l_idx, r_idx = _hpd_span_indices(wi, mass=mass)
        # map indices to fractional coords
        denom = float(max(Lnp[i] - 1, 1)) if pos_channel == 'end_inclusive' else float(max(Lnp[i], 1))
        S[i] = float(l_idx) / denom
        E[i] = float(r_idx) / denom
        ent[i] = _norm_entropy(wi.astype(np.float64, copy=False))
    return S, E, ent

def load_ids_file(path: Optional[str]) -> Optional[List[str]]:
    if not path:
        return None
    ids: List[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if s:
                ids.append(s)
    return ids


def base_id_from_chunk(cid: str) -> str:
    return cid.split('_chunk_')[0] if '_chunk_' in cid else cid

def safe_key(s: str) -> str:
    """Safer HDF5 group key (avoid slashes)."""
    return s.replace('/', '_')

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description='Predict RdRP probability and span from embeddings.h5 (+ optional HDF5 export of catalytic vectors + attention weights)')
    ap.add_argument('--embeddings', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--ids-file', default=None)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=2)
    ap.add_argument('--no-amp', action='store_true')
    ap.add_argument('--no-calib', action='store_true')
    ap.add_argument('--temperature', type=float, default=None, help='Override temperature (if not set, use ckpt/sidecar temperature if available, else 1.0)')
    ap.add_argument('--tau', type=float, default=None, help='Override anchor temperature τ used in softmax(z/τ)')
    ap.add_argument('--seq-pool', choices=['anchor','gauss'], default=None, help='Override classifier context; default=use checkpoint cfg')
    ap.add_argument('--wmin-base', type=float, default=None, help='Override wmin_base for σ clamp')
    ap.add_argument('--wmin-floor', type=float, default=None, help='Override wmin_floor for σ clamp')
    ap.add_argument('--lenfeat-scale', type=float, default=None, help='Override scale for length features')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--dump-weights-id', default=None)
    # Aggregate & GFF options
    ap.add_argument('--base-out', default=None, help='Write one row per original sequence (best-P region mapped to absolute coords)')
    ap.add_argument('--gff-out', default=None, help='Write GFF3 with 1-based inclusive coordinates per original sequence')
    ap.add_argument('--gff-min-P', type=float, default=0.0, help='Minimum P to include in GFF output')
    ap.add_argument('--gff-source', default='RdRPModel', help='GFF3 source field')
    ap.add_argument('--gff-type', default='RdRP_domain', help='GFF3 type field')
    ap.add_argument('--pos-channel', choices=['end_inclusive','end_exclusive'], default=None,
                    help='How to normalize positional channel. Default: use value saved with training (checkpoint cfg or inference_defaults.json).')
    # Discovery mode (attention HPD)
    ap.add_argument('--discovery', action='store_true', help='Use attention HPD span (shortest interval covering --attn-mass) instead of head span')
    ap.add_argument('--attn-mass', type=float, default=0.90, help='Attention mass to cover for HPD span (e.g., 0.90)')
    # Window scan options
    ap.add_argument('--window-scan', type=int, default=None, help='If set, slide this window length (aa) and take the max‑P window')
    ap.add_argument('--window-stride', type=int, default=50, help='Stride for window scan (aa)')
    # Coarse-to-fine anchor stride; if not provided, falls back to ckpt cfg.
    ap.add_argument('--coarse-stride', type=int, default=None,
                    help='Coarse anchor stride (>=2 activates coarse-to-fine anchoring). Default: use checkpoint cfg if present.')
    # --- HDF5 export of embedding vectors + attention weights ---
    ap.add_argument('--extract-h5', default=None, help='Path to HDF5 to write embedding vectors for catalytic spans')
    ap.add_argument('--min-p', type=float, default=0.90, help='Only export vectors for rows with P >= this threshold (default: 0.90)')
    ap.add_argument('--h5-coords', choices=['abs','chunk'], default='abs', help="Coordinate system for HDF5 pos: 'abs' (base_id, absolute 0-based AA) or 'chunk' (chunk_id, chunk-local index)")
    # --- NEW: per-residue attention JSON output ---
    ap.add_argument(
        '--attn-json',
        default=None,
        help='Optional path to write residue-wise attention weights and mu/sigma per chunk as JSON.'
    )

    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint (prefer safe weights_only; fall back if unsupported)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)

    # Derive config for parity with training
    cfg = ckpt.get('cfg', {}) if isinstance(ckpt, dict) else {}
    # Optional sidecar defaults written by training
    sidecar = {}
    try:
        ckpt_dir = os.path.dirname(args.checkpoint)
        with open(os.path.join(ckpt_dir, 'inference_defaults.json'), 'r', encoding='utf-8') as jf:
            sidecar = json.load(jf)
    except Exception:
        sidecar = {}

    # Some runtime knobs can drift if training anneals them but checkpoints keep
    # only the initial values. For those keys, prefer the sidecar defaults written
    # at the end of training.
    _PREFER_SIDECAR_KEYS = {
        'wmin_base',
        'wmin_floor',
        'lenfeat_scale',
        'k_sigma',
        'pos_channel',
        'coarse_stride',
        'tau_len_gamma',
        'tau_len_ref',
    }

    def _get(key, default):
        if key in _PREFER_SIDECAR_KEYS:
            return sidecar.get(key, cfg.get(key, default))
        # prefer checkpoint cfg, then sidecar, else hard default
        return cfg.get(key, sidecar.get(key, default))

    p_drop          = float(_get('dropout', 0.1))
    _tau            = float(_get('tau', 3.0))
    _alpha_cap      = float(_get('alpha_cap', 2.0))
    wmin_base       = float(_get('wmin_base', 70.0))
    wmin_floor      = float(_get('wmin_floor', 0.02))
    seq_pool_cfg    = str(  _get('seq_pool', 'gauss'))  # default to gauss to match training
    lenfeat_scale   = float(_get('lenfeat_scale', 1.0))
    pos_channel_cfg = str(_get('pos_channel', 'end_inclusive'))
    k_sigma         = float(_get('k_sigma', 2.0))       # read from sidecar/ckpt to match training
    tau_len_gamma   = float(_get('tau_len_gamma', 0.0))
    tau_len_ref     = float(_get('tau_len_ref', 1000.0))
    coarse_stride_cfg = int(_get('coarse_stride', 0))

    # CLI overrides (if provided)
    if args.tau is not None: _tau = float(args.tau)
    if args.seq_pool is not None: seq_pool_cfg = str(args.seq_pool)
    if args.wmin_base is not None: wmin_base = float(args.wmin_base)
    if args.wmin_floor is not None: wmin_floor = float(args.wmin_floor)
    if args.lenfeat_scale is not None: lenfeat_scale = float(args.lenfeat_scale)
    # pos_channel: default to training value unless user overrides
    pos_channel = args.pos_channel if (args.pos_channel is not None) else pos_channel_cfg
    # coarse stride: use CLI if given, else cfg default
    coarse_stride = int(coarse_stride_cfg if args.coarse_stride is None else args.coarse_stride)

    # Determine d_model from first item
    with h5py.File(args.embeddings, 'r') as h5e:
        any_id = next(iter(h5e['items'].keys()))
        d_model = int(h5e[f'items/{any_id}'].attrs.get('d_model', h5e[f'items/{any_id}']['emb'].shape[1]))

    # Build dataset/loader
    dataset = EmbOnlyDataset(args.embeddings, ids=load_ids_file(args.ids_file), pos_channel=pos_channel)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate,
                        persistent_workers=(args.num_workers>0))


    # Build model to match training layers
    model = RdRPModel(d_in=d_model + 1, tau=_tau, alpha_cap=_alpha_cap, p_drop=p_drop).to(device)

    # Inject runtime knobs (no layer changes)
    model.wmin_base = wmin_base
    model.wmin_floor = wmin_floor
    model.seq_pool = seq_pool_cfg
    model.lenfeat_scale = lenfeat_scale
    model.k_sigma = k_sigma
    model.coarse_stride = int(coarse_stride)
    model.tau_len_gamma = float(tau_len_gamma)
    model.tau_len_ref = float(tau_len_ref)


    # Extract state dict (supports several save styles) and strip wrappers
    state = ckpt['state_dict'] if (isinstance(ckpt, dict) and 'state_dict' in ckpt) else (
            ckpt['model'] if (isinstance(ckpt, dict) and 'model' in ckpt) else ckpt)

    def _strip_prefixes(sd, prefixes=('_orig_mod.', 'module.')):
        if not isinstance(sd, dict):
            return sd
        out = {}
        for k, v in sd.items():
            kk = k
            for p in prefixes:
                if kk.startswith(p):
                    kk = kk[len(p):]
                    break
            out[kk] = v
        return out

    state = _strip_prefixes(state)
    model.load_state_dict(state, strict=True)

    if args.compile and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # Temperature for calibrated probabilities
    T = 1.0
    if args.temperature is not None:
        T = float(args.temperature)
    else:
        if not args.no_calib:
            if isinstance(ckpt, dict) and ('temperature' in ckpt):
                try:
                    T = float(ckpt['temperature'])
                except Exception:
                    T = 1.0
            elif 'temperature' in sidecar:
                try:
                    T = float(sidecar['temperature'])
                except Exception:
                    T = 1.0

    amp_enabled = (device.type == 'cuda' and (not args.no_amp))

    # Optional attention weights dump (legacy, not used in JSON mode)
    dump_weights_id = args.dump_weights_id
    dump_path = None
    if dump_weights_id:
        base, ext = os.path.splitext(args.out)
        dump_path = f"{base}.weights.{dump_weights_id}.npy"

    # Aggregation store for base-level (original sequence) results
    agg: Dict[str, Tuple[float, str, int, int, int, float, int, float, float, float]] = {}

    # Prepare HDF5 writer if requested
    h5o = None
    items_group = None
    if args.extract_h5:
        os.makedirs(os.path.dirname(args.extract_h5) or '.', exist_ok=True)
        h5o = h5py.File(args.extract_h5, 'w')
        h5o.attrs['version'] = '1.1'
        h5o.attrs['d_model'] = int(d_model)
        h5o.attrs['coords'] = args.h5_coords
        h5o.attrs['min_p'] = float(args.min_p)
        h5o.attrs['pos_channel'] = pos_channel
        h5o.attrs['temperature'] = float(T)
        span_src = (f"attn_hpd{int(round(float(args.attn_mass)*100))}"
                    if args.discovery else f"attn_mass_k{model.k_sigma:.2f}")
        h5o.attrs['span_source'] = span_src
        h5o.attrs['weight_type'] = 'final_attention'  # document what "w" is
        items_group = h5o.create_group('items')

    # Optional JSON for residue-wise attention
    attn_json: Optional[Dict[str, Any]] = {} if args.attn_json else None

    # Write per-chunk CSV incrementally
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='', encoding='utf-8') as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(['chunk_id','P','S','E','S_idx','E_idx','mu','mu_idx','sigma','L'])
        model.eval()
        with torch.no_grad():
            for batch in loader:
                x = batch['x'].to(device)
                mask = batch['mask'].to(device)
                L = batch['L'].to(device)
                ctx = autocast('cuda', enabled=amp_enabled) if _NEW_AMP else autocast(enabled=amp_enabled)
                with ctx:
                    out = model(x, mask, L)
                    logits = out['logit'] / float(T)
                    P = torch.sigmoid(logits)
                P = P.detach().cpu().numpy()

                if args.discovery:
                    S, E, ent = _attn_hpd_span_from_out(
                        out, mask, L, mass=float(args.attn_mass), pos_channel=pos_channel
                    )
                else:
                    S = out['S_pred'].detach().cpu().numpy()
                    E = out['E_pred'].detach().cpu().numpy()

                # anchor-based mu, sigma
                mu = out['mu'].detach().cpu().numpy()
                sigma = out['sigma'].detach().cpu().numpy()

                # final-attention-based mu_attn, sigma_attn
                mu_attn = out['mu_attn'].detach().cpu().numpy()
                sigma_attn = out['sigma_attn'].detach().cpu().numpy()

                Lcpu = batch['L'].cpu().numpy().astype(int)
                orig_start = batch['orig_start'].cpu().numpy().astype(int)
                orig_len = batch['orig_len'].cpu().numpy().astype(int)

                # final attention over valid tokens
                w_full = out['w'].detach().cpu().numpy()  # (B, T)

                # indices under end-inclusive mapping (same as GFF)
                S_idx = np.round(S * (Lcpu - 1)).astype(int)
                E_idx = np.round(E * (Lcpu - 1)).astype(int)
                mu_idx = np.round(mu * (Lcpu - 1)).astype(int)

                # --- per-residue attention JSON storage ---
                if attn_json is not None:
                    for i, cid in enumerate(batch['chunk_ids']):
                        Li = int(Lcpu[i])
                        if Li <= 0:
                            continue
                        wi = w_full[i, :Li].astype(float, copy=False)
                        ostart = int(orig_start[i])
                        olen_i = int(orig_len[i])
                        abs_pos = (ostart + np.arange(Li, dtype=int)).tolist()

                        attn_json[cid] = {
                            "L": Li,
                            "orig_start": ostart,
                            "orig_len": olen_i,

                            # anchor-based parameters
                            "mu": float(mu[i]),
                            "sigma": float(sigma[i]),

                            # final-attention-based parameters
                            "mu_attn": float(mu_attn[i]),
                            "sigma_attn": float(sigma_attn[i]),

                            # canonical span (exactly what GFF uses)
                            "S_norm": float(S[i]),
                            "E_norm": float(E[i]),
                            "S_idx": int(S_idx[i]),
                            "E_idx": int(E_idx[i]),

                            # probability for convenience
                            "P": float(P[i]),

                            # per-residue final attention weights and positions
                            "w": wi.tolist(),
                            "abs_pos": abs_pos,
                        }

                # absolute mapping to original sequence
                abs_S = orig_start + S_idx
                abs_E = orig_start + E_idx

                # ---- emit per-chunk predictions + optional HDF5 export ----
                for i, cid in enumerate(batch['chunk_ids']):
                    # CSV row
                    writer.writerow([
                        cid,
                        float(P[i]),
                        float(S[i]),
                        float(E[i]),
                        int(S_idx[i]),
                        int(E_idx[i]),
                        float(mu[i]),
                        int(mu_idx[i]),
                        float(sigma[i]),
                        int(Lcpu[i]),
                    ])

                    # aggregate best per base id
                    bid = base_id_from_chunk(cid)
                    cand = (
                        float(P[i]),
                        cid,
                        int(abs_S[i]),
                        int(abs_E[i]),
                        int(orig_start[i] + mu_idx[i]),
                        float(sigma[i]),
                        int(orig_len[i]),
                        float(S[i]),
                        float(E[i]),
                        0.0,
                    )
                    cur = agg.get(bid)
                    if (cur is None) or (cand[0] > cur[0]):
                        agg[bid] = cand

                    # HDF5 export (unchanged)
                    if (h5o is not None) and (float(P[i]) >= float(args.min_p)):
                        Li = int(Lcpu[i])
                        s_idx = max(0, min(int(S_idx[i]), Li - 1))
                        e_idx = max(0, min(int(E_idx[i]), Li - 1))
                        if e_idx < s_idx:
                            s_idx, e_idx = e_idx, s_idx
                        vecs = x[i, :Li, :-1].detach().cpu().numpy().astype(np.float32, copy=False)
                        wi = w_full[i, :Li].astype(np.float32, copy=False)
                        w_span = wi[s_idx:e_idx + 1]
                        w_sum = float(w_span.sum())
                        if args.h5_coords == 'abs':
                            pos = (orig_start[i] + np.arange(Li, dtype=np.int64))[s_idx:e_idx + 1]
                        else:
                            pos = np.arange(Li, dtype=np.int64)[s_idx:e_idx + 1]
                        sub_vecs = vecs[s_idx:e_idx + 1]

                        key = safe_key(cid)
                        if key in items_group:
                            k = 1
                            while f"{key}__{k}" in items_group:
                                k += 1
                            key = f"{key}__{k}"
                        g = items_group.create_group(key)
                        g.create_dataset('pos', data=pos.astype(np.int32), compression=None)
                        g.create_dataset('vec', data=sub_vecs, compression=None)
                        g.create_dataset('w', data=w_span.astype(np.float32, copy=False), compression=None)
                        g.attrs['chunk_id'] = cid
                        g.attrs['base_id'] = bid
                        g.attrs['coords'] = args.h5_coords
                        g.attrs['P'] = float(P[i])
                        g.attrs['S_norm'] = float(S[i])
                        g.attrs['E_norm'] = float(E[i])
                        g.attrs['S_idx'] = int(s_idx)
                        g.attrs['E_idx'] = int(e_idx)
                        g.attrs['mu_idx'] = int(np.clip(int(mu_idx[i]), 0, Li - 1))
                        g.attrs['sigma'] = float(sigma[i])
                        g.attrs['L'] = int(Lcpu[i])
                        g.attrs['orig_start'] = int(orig_start[i])
                        g.attrs['orig_len'] = int(orig_len[i])
                        g.attrs['d_model'] = int(d_model)
                        g.attrs['pos_channel'] = pos_channel
                        g.attrs['temperature'] = float(T)
                        g.attrs['w_sum'] = w_sum
                        g.attrs['weight_type'] = 'final_attention'
                        span_src = (
                            f"attn_hpd{int(round(float(args.attn_mass) * 100))}"
                            if args.discovery else f"attn_mass_k{model.k_sigma:.2f}"
                        )
                        g.attrs['span_source'] = span_src
                        if args.discovery:
                            g.attrs['attn_mass'] = float(args.attn_mass)
                        else:
                            g.attrs['k_sigma'] = float(model.k_sigma)

                # ---- optional window scan per item (no HDF5 extraction here; export sticks to chunk spans) ----
                if args.window_scan is not None and args.window_scan > 0:
                    win = int(args.window_scan)
                    stride = int(args.window_stride)
                    B = x.size(0)
                    for i in range(B):
                        Li = int(mask[i].sum().item())  # effective residues
                        if Li <= 0:
                            continue
                        if win >= Li:
                            starts = [0]
                        else:
                            starts = list(range(0, Li - win + 1, stride))
                            if starts[-1] != Li - win:
                                starts.append(Li - win)
                        # build window minibatches
                        emb_i = x[i, :Li, :-1].cpu().numpy()  # strip pos
                        bid_i = base_id_from_chunk(batch['chunk_ids'][i])
                        best = agg.get(bid_i)
                        for k in range(0, len(starts), 64):  # process in minibatches
                            sub = starts[k:k+64]
                            Xw = []
                            Mw = []
                            Lw = []
                            for s0 in sub:
                                s1 = s0 + min(win, Li)
                                emb_w = emb_i[s0:s1]
                                L_w = emb_w.shape[0]
                                if pos_channel == 'end_inclusive':
                                    denom = float(max(L_w - 1, 1))
                                    pos_w = (np.arange(L_w, dtype=np.float32) / denom)[:, None]
                                else:  # end_exclusive
                                    pos_w = (np.arange(L_w, dtype=np.float32) / float(L_w))[:, None]
                                Xw.append(np.concatenate([emb_w, pos_w], axis=1))
                                Mw.append(np.ones((L_w,), dtype=bool))
                                Lw.append(L_w)
                            Xw = [torch.from_numpy(a) for a in Xw]
                            Mw = [torch.from_numpy(m) for m in Mw]
                            # pad
                            maxL = max(t.shape[0] for t in Xw)
                            padX = torch.zeros(len(Xw), maxL, Xw[0].shape[1], dtype=torch.float32, device=device)
                            padM = torch.zeros(len(Xw), maxL, dtype=torch.bool, device=device)
                            padL = torch.tensor(Lw, dtype=torch.long, device=device)
                            for j, t in enumerate(Xw):
                                padX[j, :t.shape[0]] = t.to(device)
                                padM[j, :t.shape[0]] = Mw[j].to(device)
                            with ctx:
                                outw = model(padX, padM, padL)
                                Pw  = torch.sigmoid(outw['logit'] / float(T)).detach().cpu().numpy()
                                if args.discovery:
                                    Sw, Ew, _ = _attn_hpd_span_from_out(outw, padM, padL, mass=float(args.attn_mass), pos_channel=pos_channel)
                                else:
                                    Sw  = outw['S_pred'].detach().cpu().numpy()
                                    Ew  = outw['E_pred'].detach().cpu().numpy()
                                muw = outw['mu'].detach().cpu().numpy()
                                sigw= outw['sigma'].detach().cpu().numpy()
                            # map best window back to absolute
                            for j, s0 in enumerate(sub):
                                Pj = float(Pw[j])
                                if best is None or Pj > best[0]:
                                    L_w = Lw[j]
                                    Sidx = int(np.round(Sw[j] * max(L_w - 1, 1)))
                                    Eidx = int(np.round(Ew[j] * max(L_w - 1, 1)))
                                    muidx = int(np.round(muw[j] * max(L_w - 1, 1)))
                                    aS = int(batch['orig_start'][i].cpu().item()) + s0 + Sidx
                                    aE = int(batch['orig_start'][i].cpu().item()) + s0 + Eidx
                                    aMu = int(batch['orig_start'][i].cpu().item()) + s0 + muidx
                                    best = (Pj, f"{batch['chunk_ids'][i]}:win{s0}-{s0+L_w}",
                                            aS, aE, aMu, float(sigw[j]), int(batch['orig_len'][i].cpu().item()),
                                            float(Sw[j]), float(Ew[j]), 0.0)
                                    agg[bid_i] = best

    # Close HDF5 if open
    if h5o is not None:
        h5o.close()

    # Write attention JSON if requested
    if attn_json is not None:
        os.makedirs(os.path.dirname(args.attn_json) or '.', exist_ok=True)
        with open(args.attn_json, 'w', encoding='utf-8') as fjson:
            json.dump(attn_json, fjson, indent=2)
        print(f"Wrote residue-wise attention weights to: {args.attn_json}")

    # Base-level CSV
    if args.base_out:
        os.makedirs(os.path.dirname(args.base_out) or '.', exist_ok=True)
        with open(args.base_out, 'w', newline='', encoding='utf-8') as fbase:
            bw = csv.writer(fbase)
            bw.writerow(['base_id','P','best_chunk_or_window','abs_start_idx','abs_end_idx','abs_mu_idx','sigma','orig_len','S_norm','E_norm','attn_entropy'])
            for bid, (Pmax, cid, aS, aE, aMu, sig, olen, Sn, En, entv) in agg.items():
                aS2 = max(0, min(aS, olen-1))
                aE2 = max(0, min(aE, olen-1))
                if aE2 < aS2:
                    aS2, aE2 = aE2, aS2
                bw.writerow([bid, Pmax, cid, aS2, aE2, aMu, sig, olen, Sn, En, entv])

    # GFF3 output (1-based inclusive)
    if args.gff_out:
        os.makedirs(os.path.dirname(args.gff_out) or '.', exist_ok=True)
        with open(args.gff_out, 'w', encoding='utf-8') as fgff:
            fgff.write('##gff-version 3\n')
            n_feat = 0
            ksig = getattr(model, 'k_sigma', 2.0)
            span_src = (f"attn_hpd{int(round(float(args.attn_mass)*100))}"
                        if args.discovery else f"attn_mass_k{ksig:.2f}")
            for bid, (Pmax, cid, aS, aE, aMu, sig, olen, Sn, En, entv) in agg.items():
                if Pmax < float(args.gff_min_P):
                    continue
                start_1b = int(aS) + 1
                end_1b = int(aE) + 1
                if end_1b < start_1b:
                    start_1b, end_1b = end_1b, start_1b
                attrs = [
                    f"ID={bid}",
                    "Name=RdRP_domain",
                    f"ChunkOrWindow={cid}",
                    f"P={Pmax:.6f}",
                    f"sigma={sig:.4f}",
                    f"SpanSource={span_src}",
                    f"AttnMass={float(args.attn_mass):.2f}",
                    f"AttnEntropy={entv:.3f}",
                    # method attributes
                    f"method_seq_pool={model.seq_pool}",
                    f"method_tau={model.tau}",
                    f"method_wmin={model.wmin_base}/{model.wmin_floor}",
                    f"method_pos_channel={pos_channel}",
                    f"method_k_sigma={ksig}",
                ]
                row = [bid, args.gff_source, args.gff_type, str(start_1b), str(end_1b), f"{Pmax:.6f}", '.', '.', ';'.join(attrs)]
                fgff.write('\t'.join(row) + '\n')
                n_feat += 1
        print(f"Wrote GFF3 with {n_feat} features to: {args.gff_out}")
    print(f"Wrote per-chunk predictions to: {args.out} Temperature used: {T} seq_pool={model.seq_pool} tau={model.tau} "
          f"wmin_base={model.wmin_base} wmin_floor={model.wmin_floor} lenfeat_scale={model.lenfeat_scale} pos_channel={pos_channel} "
          f"k_sigma={model.k_sigma} coarse_stride={model.coarse_stride} tau_len_gamma={model.tau_len_gamma} tau_len_ref={model.tau_len_ref} "
          f"discovery={'on' if args.discovery else 'off'} attn_mass={args.attn_mass} "
          f"Base-level rows: {len(agg)} (use --base-out to save) Window-scan: {'on' if args.window_scan else 'off'} (len={args.window_scan}, stride={args.window_stride}) "
          f"HDF5: {'off' if not args.extract_h5 else args.extract_h5} min_p={args.min_p} coords={args.h5_coords} (weights saved = final attention)")

if __name__ == '__main__':
    main()
