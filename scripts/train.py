#!/usr/bin/env python
"""
Train the RdRP detector with Gaussian attention and span localization.

Major features:
  • Grouped split by base protein id (avoid chunk leakage)
  • nnPU + optional clean-N BCE + span loss + anchor regularizer
  • Gaussian-pooled sequence context with width clamp (σ_min = max(wmin_base/L, wmin_floor))
  • Optional token dropout / embedding noise (with warmup+decay) / span-aware random crops
  • Flexible model selection metric: AP, windowed AP (ap_win), or Recall@Precision (rap)
  • Learning-rate schedulers: cosine, cosine with warm restarts, or ReduceLROnPlateau
  • Early stopping with a minimum-epochs guard; temperature calibration on validation

Quick usage (balanced long run):
python train.py \
  --embeddings embeddings.h5 \
  --labels labels.h5 \
  --out-dir runs/exp1 \
  --epochs 120 --min-epochs 60 --patience 24 \
  --scheduler warm_restarts --t0 10 --t-mult 2 --eta-min 1e-6 \
  --pu-prior 0.30 --pu-prior-start 0.28 --pu-prior-end 0.50 --pu-prior-anneal-epochs 16 \
  --lambda-clean-neg 0.15 --lambda-span 1.0 --lambda-anchor 0.08 \
  --token-dropout 0.12 --token-dropout-protect 10 \
  --emb-noise-std 0.03 --emb-noise-warmup-epochs 10 --emb-noise-decay-epochs 20 \
  --crop-prob-pos 0.30 --crop-prob-un 0.20 \
  --crop-windows 220,270,320,380,420,480 --crop-jitter-frac 0.125 --crop-center-only \
  --wmin-base 80 --wmin-floor 0.025 --wmin-anneal-epochs 12 \
  --coarse-stride 6 --lenfeat-scale 0.0 \
  --select-metric ap \
  --calibrate
"""

from __future__ import annotations
import os
import math
import json
import time
import random
import argparse
import logging
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional, List

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ----------------------------
# Utils & metrics
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax over valid (mask==1) positions, zeros elsewhere (stable in AMP)."""
    mask = mask.bool()
    very_neg = torch.finfo(logits.dtype).min / 2
    logits = logits.masked_fill(~mask, very_neg)
    z = torch.softmax(logits.float(), dim=dim).to(logits.dtype)
    z = z * mask
    denom = z.sum(dim=dim, keepdim=True).clamp(min=1e-12)
    return z / denom


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(-y_score)
    y_true = y_true[order].astype(np.float64)
    y_score = y_score[order]
    tp = 0.0
    fp = 0.0
    precisions = []
    recalls = []
    P = float(y_true.sum())
    if P == 0:
        return 0.0
    last_score = None
    for i in range(len(y_true)):
        yi = y_true[i]
        si = y_score[i]
        if last_score is None or si != last_score:
            if i > 0:
                precisions.append(tp / (tp + fp))
                recalls.append(tp / P)
            last_score = si
        if yi > 0.5:
            tp += 1.0
        else:
            fp += 1.0
    precisions.append(tp / (tp + fp))
    recalls.append(tp / P)
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * max(r - prev_r, 0.0)
        prev_r = r
    return float(ap)


def recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, min_precision: float = 0.90) -> float:
    """
    Maximum recall achievable while maintaining precision >= min_precision.
    Single-number, threshold-free selector tuned for discovery.
    """
    order = np.argsort(-y_score)
    y = y_true[order].astype(np.float64)
    tp = 0.0
    fp = 0.0
    P = float(y.sum())
    if P == 0:
        return 0.0
    best_recall = 0.0
    for i in range(len(y)):
        if y[i] > 0.5: tp += 1.0
        else:          fp += 1.0
        prec = tp / (tp + fp)
        rec = tp / P
        if prec >= min_precision and rec > best_recall:
            best_recall = rec
    return float(best_recall)


def span_iou(pred_s: np.ndarray, pred_e: np.ndarray, tgt_s: np.ndarray, tgt_e: np.ndarray) -> np.ndarray:
    left = np.maximum(pred_s, tgt_s)
    right = np.minimum(pred_e, tgt_e)
    inter = np.clip(right - left, a_min=0.0, a_max=None)
    union = np.maximum(pred_e - pred_s, 0.0) + np.maximum(tgt_e - tgt_s, 0.0) - inter
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou


def build_soft_span_mask_like(w_attn: torch.Tensor,
                              S: torch.Tensor, E: torch.Tensor,
                              ramp: float = 0.05) -> torch.Tensor:
    """
    Create a soft mask ~1 inside [S,E] with cosine ramps on both edges.
    Shapes: w_attn (B,L); S,E (B,) in [0,1]. Returns (B,L), not row-normalized.
    """
    B, L = w_attn.shape
    device = w_attn.device
    pos = torch.linspace(0, 1, L, device=device).view(1, L).expand(B, -1)
    S = S.view(B, 1); E = E.view(B, 1)
    S, E = torch.minimum(S, E), torch.maximum(S, E)

    ramp = float(max(0.0, min(0.25, ramp)))
    left0  = (S - ramp).clamp_min(0.0)
    left1  = S
    right1 = E
    right0 = (E + ramp).clamp_max(1.0)

    m = torch.zeros_like(w_attn)

    # Left ramp: 0→1 on [left0,left1)
    mask_l = (pos >= left0) & (pos < left1)
    den_l  = (left1 - left0).clamp_min(1e-8)
    ph_l   = (pos - left0) / den_l
    val_l  = 0.5 - 0.5 * torch.cos(torch.pi * ph_l.clamp(0,1))
    m = torch.where(mask_l, val_l, m)

    # Middle: 1 on [left1,right1]
    mask_mid = (pos >= left1) & (pos <= right1)
    m = torch.where(mask_mid, torch.ones_like(m), m)

    # Right ramp: 1→0 on (right1,right0]
    mask_r = (pos > right1) & (pos <= right0)
    den_r  = (right0 - right1).clamp_min(1e-8)
    ph_r   = (right0 - pos) / den_r
    val_r  = 0.5 - 0.5 * torch.cos(torch.pi * ph_r.clamp(0,1))
    m = torch.where(mask_r, val_r, m)

    # Zero out padded tokens
    valid = (w_attn > 0)
    m = m * valid
    return m

# ----------------------------
# Data
# ----------------------------

@dataclass
class LabelTable:
    ids: np.ndarray            # object array of chunk_ids (strings)
    labels: np.ndarray         # int array 0/1/2  (0=N, 1=U, 2=P)
    L: np.ndarray              # int AA lengths
    S: np.ndarray              # float (0..1)
    E: np.ndarray              # float (0..1)
    use_span: np.ndarray       # bool
    label_map: Dict[int, str]


class RdRPDataset(Dataset):
    def __init__(self,
                 emb_path: str,
                 labels_path: str,
                 indices: np.ndarray,
                 is_train: bool = False,
                 # cropping
                 crop_prob_pos: float = 0.0,
                 crop_prob_un: float = 0.0,
                 crop_windows: Optional[List[int]] = None,
                 crop_jitter_frac: float = 0.125,
                 crop_center_only: bool = True,
                 pos_channel: str = 'end_inclusive',
                 dtype: str = 'float32'):
        super().__init__()
        self.emb_path = emb_path
        self.labels_path = labels_path
        self.indices = np.array(indices, dtype=np.int64)
        self.is_train = is_train
        self.crop_prob_pos = float(crop_prob_pos)
        self.crop_prob_un = float(crop_prob_un)
        self.crop_windows = list(crop_windows or [])
        self.crop_jitter_frac = float(crop_jitter_frac)
        self.crop_center_only = bool(crop_center_only)
        self.pos_channel = pos_channel
        self.dtype = np.float32 if dtype == 'float32' else np.float16
        # Lazy-open per worker
        self._emb_h5: Optional[h5py.File] = None
        self._lab_h5: Optional[h5py.File] = None
        # Load labels table fully (small)
        self.table = self._load_labels_table(labels_path)
        # d_model can be read from first item lazily
        self._d_model_cache: Optional[int] = None

    @staticmethod
    def _parse_label_map_attr(g: h5py.Group) -> Dict[int, str]:
        lm = g.attrs.get('label_map', None)
        if lm is None:
            return {0: 'N', 1: 'U', 2: 'P'}
        try:
            if isinstance(lm, (bytes, bytearray)):
                lm = lm.decode('utf-8')
            if isinstance(lm, str):
                obj = json.loads(lm)
            else:
                obj = dict(lm)
        except Exception:
            try:
                obj = dict(lm)
            except Exception:
                return {0: 'N', 1: 'U', 2: 'P'}
        out: Dict[int, str] = {}
        for k, v in obj.items():
            try:
                ik = int(k)
                out[ik] = str(v)
            except Exception:
                try:
                    iv = int(v)
                    out[iv] = str(k)
                except Exception:
                    pass
        if not out:
            out = {0: 'N', 1: 'U', 2: 'P'}
        return out

    def _load_labels_table(self, path: str) -> LabelTable:
        with h5py.File(path, 'r') as h5l:
            g = h5l['labels']
            try:
                chunk_ids = g['chunk_id'].asstr()[...]
            except Exception:
                tmp = g['chunk_id'][...]
                chunk_ids = np.array([t.decode('utf-8') if isinstance(t, (bytes, bytearray)) else str(t) for t in tmp], dtype=object)
            labels = g['label'][...].astype(np.int64)
            L = g['L'][...].astype(np.int64)
            S = g['S'][...].astype(np.float32)
            E = g['E'][...].astype(np.float32)
            use_span = (g['use_span'][...].astype(np.bool_) if 'use_span' in g else np.ones_like(L, dtype=bool))
            label_map = self._parse_label_map_attr(g)
        return LabelTable(ids=chunk_ids, labels=labels, L=L, S=S, E=E, use_span=use_span, label_map=label_map)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def _open(self) -> None:
        if self._emb_h5 is None:
            self._emb_h5 = h5py.File(self.emb_path, 'r')

    def _get_d_model(self) -> int:
        if self._d_model_cache is not None:
            return self._d_model_cache
        self._open()
        any_id = self.table.ids[self.indices[0]]
        grp = self._emb_h5[f'items/{any_id}']
        d_model = int(grp.attrs.get('d_model', grp['emb'].shape[1]))
        self._d_model_cache = d_model
        return d_model

    def _make_pos(self, L: int) -> np.ndarray:
        if L <= 1:
            return np.zeros((1,), dtype=np.float32)
        if self.pos_channel == 'end_inclusive':
            return (np.arange(L, dtype=np.float32) / float(L - 1)).astype(np.float32)
        else:
            return (np.arange(L, dtype=np.float32) / float(L)).astype(np.float32)

    def _span_indices(self, S: float, E: float, L: int) -> Tuple[int, int, int]:
        s_idx = int(round(S * max(L - 1, 1)))
        e_idx = int(round(E * max(L - 1, 1)))
        c_idx = (s_idx + e_idx) // 2
        return s_idx, e_idx, c_idx

    def _maybe_crop(self, x: np.ndarray, y: int, S: float, E: float, use_span: bool):
        """Return possibly cropped (x, S, E, use_span). Uses actual x length."""
        L = x.shape[0]
        if not self.is_train or not self.crop_windows:
            return x, float(S), float(E), bool(use_span)

        rng = random.random()
        # Positive: span-aware crop
        if y == 2 and self.crop_prob_pos > 0 and rng < self.crop_prob_pos:
            W = min(int(random.choice(self.crop_windows)), L)
            if W == L:
                return x, float(S), float(E), bool(use_span)
            s_idx, e_idx, c_idx = self._span_indices(S, E, L)
            jitter = int(round(random.uniform(-self.crop_jitter_frac * W, self.crop_jitter_frac * W)))
            w0 = max(0, min(c_idx - W // 2 + jitter, L - W))
            if self.crop_center_only and not (w0 <= c_idx <= w0 + W - 1):
                w0 = max(0, min(c_idx - W // 2, L - W))
            w1 = w0 + W
            s_in = np.clip(s_idx - w0, 0, W - 1)
            e_in = np.clip(e_idx - w0, 0, W - 1)
            if e_in < s_in:
                s_in, e_in = e_in, s_in
            Sp = s_in / max(W - 1, 1)
            Ep = e_in / max(W - 1, 1)
            xp = x[w0:w1].copy()
            xp[:, -1] = self._make_pos(W)
            return xp, float(Sp), float(Ep), bool(use_span)

        # Unlabeled/negatives: random crop
        if y in (0, 1) and self.crop_prob_un > 0 and rng < self.crop_prob_un:
            W = min(int(random.choice(self.crop_windows)), L)
            if W == L:
                return x, float(S), float(E), bool(use_span)
            w0 = random.randint(0, L - W)
            xp = x[w0:w0 + W].copy()
            xp[:, -1] = self._make_pos(W)
            return xp, float(S), float(E), False

        return x, float(S), float(E), bool(use_span)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        self._open()
        j = int(self.indices[i])
        cid = self.table.ids[j]
        grp = self._emb_h5[f'items/{cid}']

        emb = grp['emb'][...]
        mask = grp['mask'][...].astype(bool, copy=False)
        emb = emb[mask]  # keep only valid tokens
        L_eff = emb.shape[0]
        if L_eff == 0:
            raise RuntimeError(f"{cid} has zero valid tokens")

        pos = self._make_pos(L_eff)[:, None]
        x = np.concatenate([emb.astype(self.dtype, copy=False),
                            pos.astype(self.dtype, copy=False)], axis=1)

        y = int(self.table.labels[j])
        S = float(self.table.S[j])
        E = float(self.table.E[j])
        use_span = bool(self.table.use_span[j])

        # Maybe crop (train only)
        x, S, E, use_span = self._maybe_crop(x, y, S, E, use_span)
        L_new = x.shape[0]

        return {
            'chunk_id': cid,
            'x': torch.from_numpy(x),
            'mask': torch.ones(L_new, dtype=torch.bool),
            'L': torch.tensor(L_new, dtype=torch.long),
            'y': torch.tensor(y, dtype=torch.long),
            'S': torch.tensor(S, dtype=torch.float32),
            'E': torch.tensor(E, dtype=torch.float32),
            'use_span': torch.tensor(use_span, dtype=torch.bool),
        }

    def __del__(self):
        try:
            if self._emb_h5 is not None:
                self._emb_h5.close()
            if self._lab_h5 is not None:
                self._lab_h5.close()
        except Exception:
            pass


def collate_batch(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_len = max(s['x'].shape[0] for s in samples)
    d = samples[0]['x'].shape[1]
    B = len(samples)
    x = torch.zeros(B, max_len, d, dtype=samples[0]['x'].dtype)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    L = torch.zeros(B, dtype=torch.long)
    y = torch.zeros(B, dtype=torch.long)
    S = torch.zeros(B, dtype=torch.float32)
    E = torch.zeros(B, dtype=torch.float32)
    use_span = torch.zeros(B, dtype=torch.bool)
    chunk_ids = []
    for i, s in enumerate(samples):
        Li = s['x'].shape[0]
        x[i, :Li] = s['x']
        mask[i, :Li] = True
        L[i] = s['L']
        y[i] = s['y']
        S[i] = s['S']
        E[i] = s['E']
        use_span[i] = s['use_span']
        chunk_ids.append(s['chunk_id'])
    return {
        'chunk_ids': chunk_ids,
        'x': x, 'mask': mask, 'L': L, 'y': y,
        'S': S, 'E': E, 'use_span': use_span,
    }

# ----------------------------
# Model
# ----------------------------

class TokenBackbone(nn.Module):
    def __init__(self, d_in: int, p_drop: float = 0.1):
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
        h = h.view(B, T, 256)
        return h


class TokenScorer(nn.Module):
    def __init__(self, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(64, 1),
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        z = self.net(H).squeeze(-1)
        return z


class Heads(nn.Module):
    def __init__(self, p_drop: float = 0.1):
        super().__init__()
        self.seq_head = nn.Sequential(
            nn.Linear(256 + 2, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 1)
        )
        self.span_head = nn.Sequential(       # kept for extensibility (not used for S/E here)
            nn.Linear(256 + 2, 128), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(128, 2)
        )
        self.alpha_head = nn.Sequential(
            nn.LayerNorm(256 + 2),
            nn.Linear(256 + 2, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, c: torch.Tensor, len_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([c, len_feat], dim=-1)
        logit = self.seq_head(x).squeeze(-1)
        span_params = self.span_head(x)
        alpha_raw = self.alpha_head(torch.cat([c, len_feat], dim=-1)).squeeze(-1)
        return logit, span_params, alpha_raw


class RdRPModel(nn.Module):
    def __init__(self, d_in: int, tau: float = 3.0, alpha_cap: float = 2.0, p_drop: float = 0.1):
        super().__init__()
        self.tau = tau
        self.alpha_cap = alpha_cap
        self.backbone = TokenBackbone(d_in, p_drop=p_drop)
        self.scorer = TokenScorer(p_drop=p_drop)
        self.heads = Heads(p_drop=p_drop)

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

        # Normalize z per sequence for stable anchoring
        valid = mask.to(z.dtype)
        z_mean = (z * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        z_var  = ((z - z_mean)**2 * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        z_std  = z_var.add(1e-6).sqrt()
        z_norm = (z - z_mean) / z_std

        # τ scaling (optional, length-aware)
        tau = torch.as_tensor(getattr(self, 'tau', 3.0), dtype=z.dtype, device=device)
        if getattr(self, 'tau_len_gamma', 0.0) != 0.0:
            Lf = L.to(torch.float32)
            tau = tau * ((Lf / float(getattr(self, 'tau_len_ref', 1000.0)))
                         .clamp(min=0.25, max=4.0) ** float(getattr(self, 'tau_len_gamma', 0.0)))
            tau = tau.view(-1, 1)
        else:
            tau = tau.view(1, 1)

        # Anchor on normalized token scores
        w_anchor = masked_softmax(z_norm / tau, mask, dim=1)  # (B,T)

        # Optional coarse-to-fine anchor
        stride = int(getattr(self, 'coarse_stride', 0) or 0)
        if stride >= 2 and T >= 2:
            very_neg = torch.finfo(z.dtype).min / 2
            z_masked = z_norm.masked_fill(~mask, very_neg).unsqueeze(1)  # (B,1,T)
            zc = F.max_pool1d(z_masked, kernel_size=stride, stride=stride, ceil_mode=True).squeeze(1)  # (B,Tc)
            mc = F.max_pool1d(mask.unsqueeze(1).to(z.dtype), kernel_size=stride, stride=stride, ceil_mode=True).squeeze(1) > 0
            wc = masked_softmax(zc / (tau if tau.numel() > 1 else tau), mc, dim=1)
            Tc = zc.shape[1]
            posc = torch.arange(Tc, device=device, dtype=torch.float32)[None, :].repeat(B,1)
            denomc = (torch.ceil(L.to(torch.float32) / stride)).clamp(min=1.0).unsqueeze(1) - 1.0
            denomc = denomc.clamp(min=1.0)
            muc = (wc * (posc / denomc)).sum(dim=1)
            varc = (wc * ((posc/denomc - muc.unsqueeze(1))**2)).sum(dim=1)
            sigc = torch.sqrt(varc + 1e-8)
            mu_init, sigma_init = muc, sigc
        else:
            mu_init = sigma_init = None

        if mu_init is not None:
            mu, sigma = mu_init, sigma_init
        else:
            mu = (w_anchor * pos).sum(dim=1)
            var = (w_anchor * ((pos - mu.unsqueeze(1)) ** 2)).sum(dim=1)
            sigma = torch.sqrt(var + 1e-8)

        # Clamp σ in [w_min, 0.5]
        w_min = torch.maximum(getattr(self, 'wmin_base', 70.0) / L.to(torch.float32),
                              torch.tensor(getattr(self, 'wmin_floor', 0.02), device=device))
        sigma = torch.maximum(sigma, w_min)
        sigma = torch.clamp(sigma, max=0.5)

        # Contexts and heads
        c_anchor = torch.einsum('bt,btc->bc', w_anchor.float(), H.float()).to(H.dtype)
        lenf = self._len_feat(L)
        alpha_raw = self.heads.alpha_head(torch.cat([c_anchor, lenf], dim=-1)).squeeze(-1)
        alpha = F.softplus(alpha_raw)
        if getattr(self, 'alpha_cap', None) is not None:
            alpha = torch.clamp(alpha, max=self.alpha_cap)

        q = -0.5 * ((pos - mu.unsqueeze(1)) / sigma.unsqueeze(1)).pow(2)  # (B,T)
        z_center = z - (z * valid).sum(dim=1, keepdim=True) / valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        a = z_center + alpha.unsqueeze(1) * q
        w = masked_softmax(a, mask, dim=1)

        c_gauss = torch.einsum('bt,btc->bc', w.float(), H.float()).to(H.dtype)
        seq_logit = self.heads.seq_head(torch.cat([c_gauss, lenf], dim=-1)).squeeze(-1)

        # Span from attention mass
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
            'z': z, 'w': w, 'H': H,
        }

# ----------------------------
# Losses
# ----------------------------

class NnPULoss(nn.Module):
    """Non-negative PU risk with BCE logits.

    R(f) = π * E_p[ℓ(f,1)] + max( E_u[ℓ(f,0)] - π * E_p[ℓ(f,0)] , 0 )
    """
    def __init__(self, pos_prior: float):
        super().__init__()
        assert 0.0 < pos_prior < 1.0, "pos_prior must be in (0,1)"
        self.pi = float(pos_prior)
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y = y.long()
        mask_p = (y == 2)
        mask_u = (y == 1)
        if mask_p.any():
            loss_p_pos = self.bce(logits[mask_p], torch.ones_like(logits[mask_p]))
            loss_p_neg = self.bce(logits[mask_p], torch.zeros_like(logits[mask_p]))
        else:
            loss_p_pos = torch.tensor(0.0, device=logits.device)
            loss_p_neg = torch.tensor(0.0, device=logits.device)
        if mask_u.any():
            loss_u_neg = self.bce(logits[mask_u], torch.zeros_like(logits[mask_u]))
        else:
            loss_u_neg = torch.tensor(0.0, device=logits.device)
        risk = self.pi * loss_p_pos + torch.clamp(loss_u_neg - self.pi * loss_p_neg, min=0.0)
        return risk


class CleanNegLoss(nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.w = weight
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        mask_n = (y.long() == 0)
        if mask_n.any() and self.w > 0:
            return self.w * self.bce(logits[mask_n], torch.zeros_like(logits[mask_n]))
        return logits.new_tensor(0.0)


class SpanLoss(nn.Module):
    def __init__(self, huber_beta: float = 0.1, iou_weight: float = 1.0):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=huber_beta, reduction='none')
        self.iou_weight = iou_weight

    def forward(self, S_pred: torch.Tensor, E_pred: torch.Tensor,
                S_t: torch.Tensor, E_t: torch.Tensor, use_span: torch.Tensor) -> torch.Tensor:
        m = use_span.bool()
        if not m.any():
            return S_pred.new_tensor(0.0)
        s_loss = self.huber(S_pred[m], S_t[m]).mean()
        e_loss = self.huber(E_pred[m], E_t[m]).mean()
        left = torch.maximum(S_pred[m], S_t[m])
        right = torch.minimum(E_pred[m], E_t[m])
        inter = torch.clamp(right - left, min=0.0)
        union = torch.clamp(E_pred[m] - S_pred[m], min=0.0) + torch.clamp(E_t[m] - S_t[m], min=0.0) - inter
        iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
        iou_loss = (1.0 - iou).mean()
        return s_loss + e_loss + self.iou_weight * iou_loss


class AnchorReg(nn.Module):
    def __init__(self, lambda_mu: float = 0.1, lambda_sigma: float = 0.1, k: float = 2.0):
        super().__init__()
        self.l_mu = lambda_mu
        self.l_sigma = lambda_sigma
        self.k = k

    def forward(self, mu: torch.Tensor, sigma: torch.Tensor,
                S_t: torch.Tensor, E_t: torch.Tensor, use_span: torch.Tensor) -> torch.Tensor:
        m = use_span.bool()
        if not m.any():
            return mu.new_tensor(0.0)
        center_t = 0.5 * (S_t[m] + E_t[m])
        width_t = (E_t[m] - S_t[m]) / (2.0 * self.k)
        loss_mu = F.smooth_l1_loss(mu[m], center_t, beta=0.2)
        loss_sigma = F.smooth_l1_loss(sigma[m], width_t, beta=0.2)
        return self.l_mu * loss_mu + self.l_sigma * loss_sigma

# ----------------------------
# Training
# ----------------------------

def _seed_worker(worker_id: int):
    base_seed = torch.initial_seed() % 2**31
    random.seed(base_seed + worker_id)
    np.random.seed(base_seed + worker_id)

@dataclass
class TrainConfig:
    # Required paths
    embeddings: str
    labels: str
    out_dir: str
    # Runtime
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 0.0
    dropout: float = 0.1
    # Loss weights
    pu_prior: float = 0.2
    lambda_clean_neg: float = 0.5
    lambda_span: float = 1.0
    lambda_anchor: float = 0.2
    lambda_contrast: float = 0.0
    contrast_margin: float = 0.2
    # Gaussian attention
    tau: float = 3.0
    alpha_cap: float = 3.0
    anchor_align_kl: float = 0.10
    anchor_align_ramp: float = 0.05
    anchor_alpha_prior: float = 1e-3
    k_sigma: float = 2.0
    huber_beta: float = 0.1
    iou_weight: float = 1.0
    # Split/IO
    train_frac: float = 0.8
    val_frac: float = 0.1
    num_workers: int = 2
    seed: int = 1337
    # Early stopping & schedulers
    patience: int = 5
    min_epochs: int = 0                 # ES cannot trigger before this
    scheduler: str = 'cosine'           # {'cosine','warm_restarts','plateau'}
    eta_min: float = 1e-6               # cosine schedules
    t0: int = 10                        # warm restarts: first cycle length
    t_mult: float = 2.0                 # warm restarts: cycle multiplier
    lr_patience: int = 5                # plateau: epochs without improvement
    lr_factor: float = 0.5              # plateau: LR drop factor
    lr_min: float = 1e-6                # plateau: minimum LR
    wmin_anneal_epochs: int = 0         # 0 → default half-epochs
    amp: bool = True
    calibrate: bool = False
    # Discovery knobs
    wmin_base: float = 70.0
    wmin_floor: float = 0.02
    lenfeat_scale: float = 1.0
    coarse_stride: int = 0
    tau_len_gamma: float = 0.0
    tau_len_ref: float = 1000.0
    # PU prior annealing
    pu_prior_start: Optional[float] = None
    pu_prior_end: Optional[float] = None
    pu_prior_anneal_epochs: int = 0
    # Augmentations (train-only)
    token_dropout: float = 0.0
    token_dropout_protect: int = 10
    emb_noise_std: float = 0.0
    emb_noise_warmup_epochs: int = 0
    emb_noise_decay_epochs: int = 0   # 0 = no decay
    crop_prob_pos: float = 0.0
    crop_prob_un: float = 0.0
    crop_windows: str = ''  # CSV
    crop_jitter_frac: float = 0.125
    crop_center_only: bool = True
    pos_channel: str = 'end_inclusive'
    # Model selection
    select_metric: str = 'ap'          # {'ap','ap_win','rap'}
    select_min_precision: float = 0.90 # for 'rap'
    select_window_len: int = 0         # for 'ap_win' (0 disables)
    select_window_stride: int = 60     # for 'ap_win'
    disable_iou_best: bool = False     # optional: skip saving IoU-best


def base_id(cid: str) -> str:
    return str(cid).split("_chunk_")[0]


def grouped_split(ids: np.ndarray, train_frac: float, val_frac: float, seed: int
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    groups: Dict[str, List[int]] = {}
    for i, cid in enumerate(ids):
        groups.setdefault(base_id(cid), []).append(i)
    group_keys = np.array(list(groups.keys()))
    rng.shuffle(group_keys)
    n_train = int(len(group_keys) * train_frac)
    n_val = int(len(group_keys) * val_frac)
    g_train = set(group_keys[:n_train])
    g_val   = set(group_keys[n_train:n_train+n_val])
    g_test  = set(group_keys[n_train+n_val:])

    def idx_from(gs: set) -> np.ndarray:
        out: List[int] = []
        for g in gs:
            out.extend(groups[g])
        return np.array(out, dtype=np.int64)

    return idx_from(g_train), idx_from(g_val), idx_from(g_test)


def build_datasets(cfg: TrainConfig) -> Tuple[RdRPDataset, RdRPDataset, RdRPDataset, int]:
    # Grouped split by base protein id to avoid leakage across chunks
    with h5py.File(cfg.labels, 'r') as h5l:
        g = h5l['labels']
        try:
            ids = g['chunk_id'].asstr()[...]
        except Exception:
            tmp = g['chunk_id'][...]
            ids = np.array([t.decode('utf-8') if isinstance(t, (bytes, bytearray)) else str(t) for t in tmp], dtype=object)
    train_idx, val_idx, test_idx = grouped_split(ids, cfg.train_frac, cfg.val_frac, cfg.seed)

    # Probe d_model
    tmp_ds = RdRPDataset(cfg.embeddings, cfg.labels, indices=np.array([0]))
    d_model = tmp_ds._get_d_model()
    del tmp_ds

    # Parse crop windows
    cw = [int(x) for x in cfg.crop_windows.split(',') if x.strip().isdigit()]
    train_ds = RdRPDataset(
        cfg.embeddings, cfg.labels, indices=train_idx, is_train=True,
        crop_prob_pos=cfg.crop_prob_pos, crop_prob_un=cfg.crop_prob_un,
        crop_windows=cw, crop_jitter_frac=cfg.crop_jitter_frac, crop_center_only=cfg.crop_center_only,
        pos_channel=cfg.pos_channel)
    val_ds = RdRPDataset(cfg.embeddings, cfg.labels, indices=val_idx, is_train=False, pos_channel=cfg.pos_channel)
    test_ds = RdRPDataset(cfg.embeddings, cfg.labels, indices=test_idx, is_train=False, pos_channel=cfg.pos_channel)
    return train_ds, val_ds, test_ds, int(d_model)


def evaluate(model: RdRPModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    ys, ps, ious = [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['x'].to(device)
            mask = batch['mask'].to(device)
            L = batch['L'].to(device)
            y = batch['y'].to(device)
            out = model(x, mask, L)
            logits = out['logit']
            T = getattr(model, 'temperature', None)
            if T is not None:
                logits = logits / float(T)
            P = torch.sigmoid(logits).detach().cpu().numpy()
            ys.append(batch['y'].cpu().numpy())
            ps.append(P)

            # IoU on rows with use_span==1
            m = batch['use_span'].cpu().numpy().astype(bool)
            if m.any():
                s_pred = out['S_pred'].detach().cpu().numpy()[m]
                e_pred = out['E_pred'].detach().cpu().numpy()[m]
                s_t = batch['S'].cpu().numpy()[m]
                e_t = batch['E'].cpu().numpy()[m]
                iou = span_iou(s_pred, e_pred, s_t, e_t)
                ious.append(iou)

    if not ys:
        return {'pr_auc_p_vs_rest': 0.0, 'pr_auc_p_vs_n': 0.0, 'mean_iou': 0.0, 'iou_at_0_5': 0.0}

    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    y_p_vs_rest = (y_all == 2).astype(np.int64)
    ap_rest = pr_auc(y_p_vs_rest, p_all)
    mask_n_only = (y_all != 1)
    ap_n = pr_auc(y_p_vs_rest[mask_n_only], p_all[mask_n_only]) if mask_n_only.any() else 0.0

    if ious:
        i_all = np.concatenate(ious)
        i_mean = float(i_all.mean())
        i_at05 = float((i_all >= 0.5).mean())
    else:
        i_mean, i_at05 = 0.0, 0.0

    return {'pr_auc_p_vs_rest': float(ap_rest),
            'pr_auc_p_vs_n': float(ap_n),
            'mean_iou': i_mean,
            'iou_at_0_5': i_at05}


def _collect_scores_full(model: RdRPModel, loader: DataLoader, device: torch.device
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true_p_vs_rest, scores) on the full sequences (pre-calibration)."""
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b['x'].to(device), b['mask'].to(device), b['L'].to(device))
            logits = out['logit']
            T = getattr(model, 'temperature', None)
            if T is not None:
                logits = logits / float(T)
            P = torch.sigmoid(logits).detach().cpu().numpy()
            y = (b['y'].cpu().numpy() == 2).astype(np.int64)
            ys.append(y); ps.append(P)
    return (np.concatenate(ys) if ys else np.zeros((0,), dtype=np.int64),
            np.concatenate(ps) if ps else np.zeros((0,), dtype=np.float32))


def _collect_scores_window_max(model: RdRPModel, loader: DataLoader, device: torch.device,
                               win_len: int, win_stride: int, pos_channel: str = 'end_inclusive'
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each sequence, slide windows and take the max-P window score.
    Returns (y_true_p_vs_rest, scores_max) (pre-calibration).
    """
    assert win_len and win_len > 0
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for b in loader:
            X = b['x'].to(device)
            M = b['mask'].to(device)
            L = b['L'].to(device)
            y = (b['y'].cpu().numpy() == 2).astype(np.int64)
            B = X.shape[0]
            for i in range(B):
                Li = int(L[i].item())
                val = float('-inf')
                if Li >= win_len:
                    for s in range(0, Li - win_len + 1, win_stride):
                        xw = X[i, s:s+win_len, :].clone()
                        if xw.shape[1] > 0:
                            if pos_channel == 'end_inclusive':
                                xw[:, -1] = torch.linspace(0, 1, win_len, device=device)
                            else:
                                xw[:, -1] = torch.arange(win_len, device=device) / float(win_len)
                        mw = torch.ones(win_len, dtype=torch.bool, device=device)
                        Lw = torch.tensor([win_len], dtype=torch.long, device=device)
                        outw = model(xw[None, :, :], mw[None, :], Lw)
                        lw = outw['logit']
                        val = max(val, float(torch.sigmoid(lw).item()))
                else:
                    out = model(X[i:i+1, :, :], M[i:i+1, :], L[i:i+1])
                    val = float(torch.sigmoid(out['logit']).item())
                ys.append(y[i]); ps.append(val)
    return np.array(ys, dtype=np.int64), np.array(ps, dtype=np.float32)


def apply_embedding_noise(x: torch.Tensor, mask: torch.Tensor, sigma: float) -> None:
    if sigma <= 0:
        return
    valid = mask.unsqueeze(-1).to(x.dtype)
    noise = torch.randn_like(x[:, :, :-1], dtype=torch.float32) * float(sigma)
    x[:, :, :-1] = x[:, :, :-1] + noise.to(x.dtype) * valid


def apply_token_dropout(mask: torch.Tensor,
                        L: torch.Tensor,
                        y: torch.Tensor,
                        S: torch.Tensor,
                        E: torch.Tensor,
                        use_span: torch.Tensor,
                        p: float,
                        protect: int) -> None:
    if p <= 0.0:
        return
    B, _ = mask.shape
    device = mask.device
    for i in range(B):
        Li = int(L[i].item())
        if Li <= 1:
            continue
        keep = (torch.rand(Li, device=device) > p)
        if bool(use_span[i].item()) and int(y[i].item()) == 2 and protect > 0:
            center = int(round(((float(S[i].item()) + float(E[i].item())) * 0.5) * max(Li - 1, 1)))
            a = max(0, center - protect)
            b = min(Li - 1, center + protect)
            keep[a:b+1] = True
        if keep.sum() == 0:
            keep[random.randrange(Li)] = True
        mask[i, :Li] = mask[i, :Li] & keep


def calibrate_temperature(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    logits_list, targets_list = [], []
    with torch.no_grad():
        for b in loader:
            y = b["y"].to(device)
            m = (y != 1)
            if not m.any():
                continue
            out = model(b["x"].to(device), b["mask"].to(device), b["L"].to(device))
            logits_list.append(out["logit"][m].detach().cpu())
            targets_list.append((y[m] == 2).float().detach().cpu())
    if not logits_list:
        return 1.0

    z = torch.cat(logits_list)
    t = torch.cat(targets_list)
    log_T = torch.tensor(0.0, requires_grad=True)      # T = exp(log_T) > 0
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=50)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        T = torch.exp(log_T) + 1e-6
        loss = bce(z / T, t)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_T).clamp(0.1, 10.0))
    return T


def train(cfg: TrainConfig) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    ensure_dir(cfg.out_dir)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    set_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    train_ds, val_ds, test_ds, d_model = build_datasets(cfg)
    d_in = d_model + 1  # +1 for position channel
    logging.info(f"Detected d_model={d_model}; model input dim={d_in}")

    gen = torch.Generator()
    gen.manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=(device.type=='cuda'),
                              collate_fn=collate_batch, persistent_workers=(cfg.num_workers>0),
                              worker_init_fn=_seed_worker, generator=gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=(device.type=='cuda'),
                            collate_fn=collate_batch, persistent_workers=(cfg.num_workers>0),
                            worker_init_fn=_seed_worker, generator=gen)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=(device.type=='cuda'),
                             collate_fn=collate_batch, persistent_workers=(cfg.num_workers>0),
                             worker_init_fn=_seed_worker, generator=gen)

    model = RdRPModel(d_in=d_in, tau=cfg.tau, alpha_cap=cfg.alpha_cap, p_drop=cfg.dropout).to(device)
    # Inject discovery knobs
    model.wmin_base = cfg.wmin_base
    model.wmin_floor = cfg.wmin_floor
    model.seq_pool = 'gauss'
    model.lenfeat_scale = cfg.lenfeat_scale
    model.coarse_stride = cfg.coarse_stride
    model.tau_len_gamma = cfg.tau_len_gamma
    model.tau_len_ref = cfg.tau_len_ref
    model.k_sigma = cfg.k_sigma

    # Optimizer (alpha-head gets 2x LR and 0 WD)
    base_params, alpha_params = [], []
    for n, p in model.named_parameters():
        if 'heads.alpha_head' in n:
            alpha_params.append(p)
        else:
            base_params.append(p)
    opt = torch.optim.Adam(
        [
            {'params': base_params, 'lr': cfg.lr, 'weight_decay': cfg.weight_decay},
            {'params': alpha_params, 'lr': cfg.lr * 2.0, 'weight_decay': 0.0},
        ]
    )

    # Optional compile
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    # Scheduler
    if cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.eta_min)
    elif cfg.scheduler == 'warm_restarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=cfg.t0, T_mult=cfg.t_mult, eta_min=cfg.eta_min
        )
    elif cfg.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode='max', factor=cfg.lr_factor, patience=cfg.lr_patience,
            threshold=1e-4, min_lr=cfg.lr_min, verbose=False
        )
    else:
        scheduler = None

    MAX_NORM = 1.0
    # AMP
    try:
        from torch.amp import autocast, GradScaler  # PyTorch 2.x
        _NEW_AMP = True
    except Exception:
        from torch.cuda.amp import autocast, GradScaler  # PyTorch 1.x
        _NEW_AMP = False
    scaler = GradScaler(enabled=(device.type == "cuda" and cfg.amp))

    loss_pu = NnPULoss(pos_prior=(cfg.pu_prior_start if cfg.pu_prior_start is not None else cfg.pu_prior))
    loss_clean = CleanNegLoss(weight=cfg.lambda_clean_neg)
    loss_span = SpanLoss(huber_beta=cfg.huber_beta, iou_weight=cfg.iou_weight)
    loss_anchor = AnchorReg(lambda_mu=cfg.lambda_anchor, lambda_sigma=cfg.lambda_anchor, k=cfg.k_sigma)

    # Track winners
    best_pr   = {'pr_auc': -1.0, 'epoch': 0}
    best_iou  = {'mean_iou': -1.0, 'pr_auc': -1.0, 'epoch': 0}
    best_sel  = {'metric': cfg.select_metric, 'score': -1.0, 'epoch': 0}
    best_path = os.path.join(cfg.out_dir, 'model_best.pt')
    best_pr_path  = os.path.join(cfg.out_dir, 'model_best_pr.pt')
    best_iou_path = os.path.join(cfg.out_dir, 'model_best_iou.pt')
    last_path = os.path.join(cfg.out_dir, 'model_last.pt')
    metrics_path = os.path.join(cfg.out_dir, 'metrics.json')
    log_rows: List[Dict[str, Any]] = []
    no_improve = 0

    def _get_lr(optimizer) -> float:
        return float(optimizer.param_groups[0]['lr'])

    for epoch in range(1, cfg.epochs + 1):
        # Anneal wmin_base over a controllable horizon
        wmin_start = float(cfg.wmin_base)
        wmin_final = float(max(0.5 * cfg.wmin_base, 20.0))  # halve, but not below 20 residues
        W_E = (cfg.wmin_anneal_epochs if cfg.wmin_anneal_epochs > 0 else max(cfg.epochs // 2, 1))
        t = min(epoch - 1, W_E)
        frac = t / max(W_E, 1)
        model.wmin_base = wmin_start + (wmin_final - wmin_start) * frac

        # Optional PU prior annealing
        if cfg.pu_prior_anneal_epochs and cfg.pu_prior_end is not None:
            start = cfg.pu_prior if cfg.pu_prior_start is None else cfg.pu_prior_start
            t = min(epoch - 1, cfg.pu_prior_anneal_epochs)
            frac = t / max(cfg.pu_prior_anneal_epochs, 1)
            loss_pu.pi = float(start + (cfg.pu_prior_end - start) * frac)

        model.train()
        t0 = time.time()
        running = {k: 0.0 for k in [
            'loss_total', 'loss_cls_pu', 'loss_cls_clean', 'loss_span',
            'loss_anchor', 'loss_align', 'loss_alpha_prior'
        ]}
        running['n'] = 0

        # Embedding noise schedule: warmup → (optional) decay after min_epochs
        if cfg.emb_noise_warmup_epochs > 0:
            warm = min(epoch / max(cfg.emb_noise_warmup_epochs, 1), 1.0)
            sigma_epoch = float(cfg.emb_noise_std) * warm
        else:
            sigma_epoch = float(cfg.emb_noise_std)
        if cfg.emb_noise_decay_epochs > 0 and epoch > cfg.min_epochs:
            tdec = max(0, epoch - cfg.min_epochs)
            fracd = min(tdec / max(cfg.emb_noise_decay_epochs, 1), 1.0)
            sigma_epoch = sigma_epoch * (1.0 - fracd)

        for batch in train_loader:
            x = batch['x'].to(device)
            mask = batch['mask'].to(device)
            L = batch['L'].to(device)
            y = batch['y'].to(device)
            S_t = batch['S'].to(device)
            E_t = batch['E'].to(device)
            use_span = batch['use_span'].to(device)

            # Augs: drop tokens first, then add noise on kept tokens
            apply_token_dropout(mask, L, y, S_t, E_t, use_span, p=cfg.token_dropout, protect=cfg.token_dropout_protect)
            apply_embedding_noise(x, mask, sigma_epoch)

            opt.zero_grad(set_to_none=True)
            amp_enabled = (device.type == 'cuda' and cfg.amp)
            ctx = (torch.amp.autocast('cuda', enabled=amp_enabled) if _NEW_AMP
                   else torch.cuda.amp.autocast(enabled=amp_enabled))
            with ctx:
                out = model(x, mask, L)
                logits = out['logit']
                l_pu = loss_pu(logits, y)
                l_clean = loss_clean(logits, y)
                l_span = cfg.lambda_span * loss_span(out['S_pred'], out['E_pred'], S_t, E_t, use_span)
                l_anchor = loss_anchor(out['mu'], out['sigma'], S_t, E_t, use_span)

                # Soft alignment of attention to span (KL)
                if cfg.anchor_align_kl > 0.0:
                    m = build_soft_span_mask_like(out['w'], S_t, E_t, ramp=cfg.anchor_align_ramp)
                    m = m / (m.sum(dim=1, keepdim=True) + 1e-8)
                    w_norm = out['w'] / (out['w'].sum(dim=1, keepdim=True) + 1e-8)
                    kl = (w_norm.clamp_min(1e-8) * (w_norm.clamp_min(1e-8).log() - m.clamp_min(1e-8).log())).sum(dim=1)
                    l_align = (kl * use_span.float()).mean() * cfg.anchor_align_kl
                else:
                    l_align = logits.new_tensor(0.0)

                # Tiny prior to discourage alpha→0 collapse
                l_alpha_prior = (torch.exp(-out['alpha']).mean() * cfg.anchor_alpha_prior) if cfg.anchor_alpha_prior > 0.0 else logits.new_tensor(0.0)

                # Optional in-sequence contrast
                l_contrast = logits.new_tensor(0.0)
                if cfg.lambda_contrast > 0.0:
                    mpos = (y == 2) & use_span
                    if mpos.any():
                        logit_pos = out['logit'][mpos]
                        Spos = S_t[mpos]; Epos = E_t[mpos]
                        mu_neg = torch.where(
                            (0.5 * (Spos + Epos)) < 0.5,
                            torch.clamp(Epos + 0.2, max=1.0),
                            torch.clamp(Spos - 0.2, min=0.0)
                        )
                        sigma_neg = out['sigma'][mpos].detach()
                        zc = (out['z'][mpos] - (out['z'][mpos] * mask[mpos].to(out['z'].dtype)).sum(dim=1, keepdim=True)
                              / mask[mpos].to(out['z'].dtype).sum(dim=1, keepdim=True).clamp(min=1.0))
                        pos_grid = torch.linspace(0, 1, zc.shape[1], device=zc.device).view(1, -1).expand(zc.shape[0], -1)
                        q_neg = -0.5 * ((pos_grid - mu_neg.unsqueeze(1)) / sigma_neg.unsqueeze(1)).pow(2)
                        a_neg = zc + out['alpha'][mpos].unsqueeze(1) * q_neg
                        very_neg = torch.finfo(a_neg.dtype).min / 2
                        a_neg = a_neg.masked_fill(~mask[mpos], very_neg)
                        w_neg = torch.softmax(a_neg, dim=1)
                        c_neg = torch.einsum('bt,btc->bc', w_neg.float(), out['H'][mpos].float()).to(out['H'].dtype)
                        lenf_pos = model._len_feat(L[mpos])
                        logit_neg = model.heads.seq_head(torch.cat([c_neg, lenf_pos], dim=-1)).squeeze(-1)
                        l_contrast = torch.relu(cfg.contrast_margin - (logit_pos - logit_neg)).mean() * cfg.lambda_contrast

                loss = l_pu + l_clean + l_span + l_anchor + l_align + l_alpha_prior + l_contrast

            scaler.scale(loss).backward()
            if amp_enabled:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
            scaler.step(opt)
            scaler.update()

            bs = x.size(0)
            running['loss_total']       += float(loss.item()) * bs
            running['loss_cls_pu']      += float(l_pu.item()) * bs
            running['loss_cls_clean']   += float(l_clean.item()) * bs
            running['loss_span']        += float(l_span.item()) * bs
            running['loss_anchor']      += float(l_anchor.item()) * bs
            running['loss_align']       += float(l_align.item()) * bs
            running['loss_alpha_prior'] += float(l_alpha_prior.item()) * bs
            running['n'] += bs

        train_time = time.time() - t0
        for k in list(running.keys()):
            if k != 'n':
                running[k] = running[k] / max(running['n'], 1)

        # ---- Eval first (metrics for logging, selection, and scheduler monitor)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            'epoch': epoch,
            'time_sec': round(train_time, 2),
            'lr': _get_lr(opt),
            'train_loss': running['loss_total'],
            'train_loss_cls_pu': running['loss_cls_pu'],
            'train_loss_cls_clean': running['loss_cls_clean'],
            'train_loss_span': running['loss_span'],
            'train_loss_anchor': running['loss_anchor'],
            'train_loss_align': running['loss_align'],
            'train_loss_alpha_prior': running['loss_alpha_prior'],
            'pi': float(loss_pu.pi),
        }
        for k, v in val_metrics.items():
            row[f'val_{k}'] = v
        log_rows.append(row)

        logging.info(
            f"Epoch {epoch:03d} | lr={row['lr']:.2e} | pi={loss_pu.pi:.3f} | "
            f"train={row['train_loss']:.4f} (pu={row['train_loss_cls_pu']:.3f}, "
            f"clean={row['train_loss_cls_clean']:.3f}, span={row['train_loss_span']:.3f}, "
            f"anchor={row['train_loss_anchor']:.3f}, align={row['train_loss_align']:.3f}, "
            f"a_prior={row['train_loss_alpha_prior']:.3f}) | "
            f"val_AP(P|rest)={row['val_pr_auc_p_vs_rest']:.4f} | IoU@0.5={row['val_iou_at_0_5']:.3f} | "
            f"time={row['time_sec']:.1f}s"
        )

        # Keep PR-best for reference & IoU-best optionally
        pr   = row['val_pr_auc_p_vs_rest']
        miou = row['val_mean_iou']
        if pr > best_pr['pr_auc'] + 1e-6:
            best_pr['pr_auc'] = pr
            best_pr['epoch']  = epoch
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, best_pr_path)
            logging.info(f"Saved PR‑AUC reference checkpoint → {best_pr_path} (PR={pr:.6f})")
        if (not cfg.disable_iou_best) and (miou > best_iou['mean_iou'] + 1e-6 or
            (abs(miou - best_iou['mean_iou']) <= 1e-6 and pr > best_iou['pr_auc'] + 1e-6)):
            best_iou['mean_iou'] = miou
            best_iou['pr_auc']   = pr
            best_iou['epoch']    = epoch
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, best_iou_path)
            logging.info(f"Saved IoU reference checkpoint → {best_iou_path} (mIoU={miou:.6f}, PR={pr:.6f})")

        # Selection metric (drives early-stop and best checkpoint)
        if cfg.select_metric == 'ap':
            y_sel, p_sel = _collect_scores_full(model, val_loader, device)
            sel = pr_auc(y_sel, p_sel)
        elif cfg.select_metric == 'rap':
            y_sel, p_sel = _collect_scores_full(model, val_loader, device)
            sel = recall_at_precision(y_sel, p_sel, cfg.select_min_precision)
        elif cfg.select_metric == 'ap_win':
            if cfg.select_window_len <= 0:
                logging.warning("select_metric=ap_win but --select-window-len is 0; falling back to 'ap'.")
                y_sel, p_sel = _collect_scores_full(model, val_loader, device)
                sel = pr_auc(y_sel, p_sel)
            else:
                y_sel, p_sel = _collect_scores_window_max(
                    model, val_loader, device, cfg.select_window_len, cfg.select_window_stride, cfg.pos_channel
                )
                sel = pr_auc(y_sel, p_sel)
        else:
            sel = pr  # fallback

        improved_sel = (sel > best_sel['score'] + 1e-6)
        if improved_sel:
            best_sel['score'] = float(sel)
            best_sel['epoch'] = int(epoch)
            best_sel['metric'] = cfg.select_metric
            torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, best_path)
            logging.info(f"✔ New best by {cfg.select_metric}: {sel:.6f} (epoch {epoch}) → {best_path}")

        # ---- Step LR scheduler exactly once, AFTER metrics exist
        if scheduler is not None:
            if cfg.scheduler == 'plateau':
                monitor = float(sel)  # monitor the selection metric
                scheduler.step(monitor)
            else:
                scheduler.step()

        # ---- Early stopping (respect min_epochs)
        no_improve = 0 if improved_sel else (no_improve + 1)
        if (epoch >= cfg.min_epochs) and (no_improve >= cfg.patience):
            logging.info(f"Early stopping after {epoch} epochs (patience {cfg.patience}).")
            break

    # Save last
    torch.save({'model': model.state_dict(), 'cfg': asdict(cfg)}, last_path)

    # Optional calibration on the selection-best
    temperature = None
    if cfg.calibrate:
        try:
            best_ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(best_ckpt['model'])
            temperature = calibrate_temperature(model, val_loader, device)
            best_ckpt['temperature'] = float(temperature)
            torch.save(best_ckpt, best_path)
            torch.save(best_ckpt, os.path.join(cfg.out_dir, 'model_best_calibrated.pt'))
            logging.info(f"Calibrated temperature T={temperature:.3f} and updated best checkpoint.")
        except Exception as e:
            logging.warning(f"Calibration failed: {e}")

    # Evaluate test on the selection-best (calibrated if available)
    try:
        best_ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(best_ckpt['model'])
        if 'temperature' in best_ckpt:
            model.temperature = float(best_ckpt['temperature'])
    except Exception as e:
        logging.warning(f"Could not load best checkpoint for final test; falling back to last. Error: {e}")

    test_final = evaluate(model, test_loader, device)
    with open(os.path.join(cfg.out_dir, 'test_final.json'), 'w') as f:
        json.dump(test_final, f, indent=2)

    # Also evaluate IoU-best on test set (if available)
    if not cfg.disable_iou_best:
        try:
            ckpt_iou = torch.load(os.path.join(cfg.out_dir, 'model_best_iou.pt'), map_location=device)
            model.load_state_dict(ckpt_iou['model'])
            if hasattr(model, 'temperature'):
                delattr(model, 'temperature')
            test_final_iou = evaluate(model, test_loader, device)
            with open(os.path.join(cfg.out_dir, 'test_final_best_iou.json'), 'w') as f:
                json.dump(test_final_iou, f, indent=2)
            logging.info("Wrote test_final_best_iou.json for IoU-best checkpoint.")
        except Exception as e:
            logging.warning(f"Could not evaluate IoU-best checkpoint: {e}")

    # Write metrics log & inference defaults
    metrics_obj = {'log': log_rows, 'best_sel': best_sel, 'best_by_pr': best_pr}
    if not cfg.disable_iou_best:
        metrics_obj['best_by_iou'] = best_iou
    if temperature is not None:
        metrics_obj['temperature'] = float(temperature)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_obj, f, indent=2)
    logging.info(f"Saved best to {best_path} and last to {last_path}. Metrics -> {metrics_path}")

    try:
        infer_defaults = {
            'dropout': float(cfg.dropout),
            'tau': float(cfg.tau),
            'alpha_cap': float(cfg.alpha_cap),
            'wmin_base': float(cfg.wmin_base),
            'wmin_floor': float(cfg.wmin_floor),
            'lenfeat_scale': float(cfg.lenfeat_scale),
            'pos_channel': str(cfg.pos_channel),
            'k_sigma': float(cfg.k_sigma),
            'select_metric': str(cfg.select_metric),
            'select_window_len': int(cfg.select_window_len),
            'select_window_stride': int(cfg.select_window_stride),
        }
        T_model = getattr(model, 'temperature', None)
        if T_model is not None:
            infer_defaults['temperature'] = float(T_model)
        elif temperature is not None:
            infer_defaults['temperature'] = float(temperature)
        with open(os.path.join(cfg.out_dir, 'inference_defaults.json'), 'w', encoding='utf-8') as jf:
            json.dump(infer_defaults, jf, indent=2)
        logging.info(f"Wrote inference defaults → {os.path.join(cfg.out_dir, 'inference_defaults.json')}")
    except Exception as e:
        logging.warning(f"Could not write inference_defaults.json: {e}")

# ----------------------------
# CLI
# ----------------------------

def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description='Train RdRP detector (PU + span) with optional augmentations and schedulers.')

    # Required
    p.add_argument('--embeddings', required=True, help='Path to HDF5 with per-token embeddings under /items/<chunk_id>/emb and bool /items/<chunk_id>/mask')
    p.add_argument('--labels', required=True, help='Path to HDF5 labels with datasets: chunk_id, label{0,1,2}, L, S, E, and optional use_span')
    p.add_argument('--out-dir', required=True, help='Directory to write checkpoints and metrics')

    # Runtime & optimization
    p.add_argument('--epochs', type=int, default=30, help='Maximum training epochs (also T_max for cosine schedules)')
    p.add_argument('--batch-size', type=int, default=16, help='Batch size')
    p.add_argument('--lr', type=float, default=1e-3, help='Base learning rate for most params (alpha head gets 2x)')
    p.add_argument('--weight-decay', type=float, default=0.0, help='L2 weight decay for base params (alpha head uses 0)')
    p.add_argument('--dropout', type=float, default=0.1, help='Dropout rate used in MLPs')

    # PU / loss weights
    p.add_argument('--pu-prior', type=float, default=0.2, help='Estimated positive prior π (0<π<1) for nnPU risk')
    p.add_argument('--lambda-clean-neg', type=float, default=0.5, help='Weight for BCE on clean negatives (label 0)')
    p.add_argument('--lambda-span', type=float, default=1.0, help='Weight for span regression/IoU loss on rows with use_span=1')
    p.add_argument('--lambda-anchor', type=float, default=0.2, help='Weight for anchor regularizer tying (mu, sigma) to span stats')
    p.add_argument('--lambda-contrast', type=float, default=0.0, help='Weight for optional in-sequence contrastive loss on positives')
    p.add_argument('--contrast-margin', type=float, default=0.2, help='Margin m in max(0, m - (logit_pos - logit_neg))')

    # Gaussian attention controls
    p.add_argument('--tau', type=float, default=3.0, help='Temperature for masked_softmax over tokens (lower → peakier)')
    p.add_argument('--alpha-cap', type=float, default=3.0, help='Upper cap on alpha = softplus(alpha_raw)')
    p.add_argument('--k-sigma', type=float, default=2.0, help='Span = mu_attn ± k*sigma_attn when deriving S/E from attention')
    p.add_argument('--huber-beta', type=float, default=0.1, help='β for SmoothL1 in span endpoints loss')
    p.add_argument('--iou-weight', type=float, default=1.0, help='Weight for IoU term in span loss')
    p.add_argument('--anchor-align-kl', type=float, default=0.10, help='Weight for KL(w || soft-span-mask); 0 disables alignment')
    p.add_argument('--anchor-align-ramp', type=float, default=0.05, help='Cosine ramp width (fraction of length) at span edges')
    p.add_argument('--anchor-alpha-prior', type=float, default=1e-3, help='Tiny prior to discourage alpha→0 collapse')

    # Discovery knobs
    p.add_argument('--wmin-base', type=float, default=70.0, help='Base sigma floor in residues; σ_min = max(wmin_base/L, wmin_floor)')
    p.add_argument('--wmin-floor', type=float, default=0.02, help='Absolute lower floor for sigma in normalized [0,1] coords')
    p.add_argument('--wmin-anneal-epochs', type=int, default=0, help='Epochs to anneal wmin_base from start→final (0=use half of total epochs)')
    p.add_argument('--lenfeat-scale', type=float, default=1.0, help='Scale for 2D length features [logL, 1/L]')
    p.add_argument('--coarse-stride', type=int, default=0, help='Optional coarse stride S for anchoring (0 disables)')
    p.add_argument('--tau-len-gamma', type=float, default=0.0, help='Exponent for length-aware τ scaling (0 disables)')
    p.add_argument('--tau-len-ref', type=float, default=1000.0, help='Reference length for τ scaling')

    # PU prior annealing
    p.add_argument('--pu-prior-start', type=float, default=None, help='Start value for PU prior annealing (default uses --pu-prior)')
    p.add_argument('--pu-prior-end', type=float, default=None, help='End value for PU prior annealing (required to enable anneal)')
    p.add_argument('--pu-prior-anneal-epochs', type=int, default=0, help='Epochs to linearly anneal the PU prior π (0 disables)')

    # Augmentations (train-only)
    p.add_argument('--token-dropout', type=float, default=0.0, help='Per-token drop probability (0..1)')
    p.add_argument('--token-dropout-protect', type=int, default=10, help='Protect ±N residues around span center on positives')
    p.add_argument('--emb-noise-std', type=float, default=0.0, help='Gaussian noise std on embedding channels')
    p.add_argument('--emb-noise-warmup-epochs', type=int, default=0, help='Warmup epochs for noise from 0→std (0 disables)')
    p.add_argument('--emb-noise-decay-epochs', type=int, default=0, help='After --min-epochs, linearly decay noise to 0 in this many epochs (0 disables)')
    p.add_argument('--crop-prob-pos', type=float, default=0.0, help='Probability to crop positives with span-aware window')
    p.add_argument('--crop-prob-un', type=float, default=0.0, help='Probability to crop unlabeled/negatives with random window')
    p.add_argument('--crop-windows', type=str, default='', help='Comma-separated crop window lengths (e.g., 220,270,320,380)')
    p.add_argument('--crop-jitter-frac', type=float, default=0.125, help='Center jitter as fraction of W (e.g., 0.125 for ±W/8)')
    p.add_argument('--crop-center-only', action='store_true', help='Require crops on positives to include the span center')
    p.add_argument('--pos-channel', choices=['end_inclusive','end_exclusive'], default='end_inclusive',
                   help="How to normalize the position channel: inclusive [0..1] or exclusive [0..1)")

    # Selection strategy
    p.add_argument('--select-metric', choices=['ap','ap_win','rap'], default='ap',
                   help="Checkpoint selection metric: 'ap' (PR‑AUC on full sequences), "
                        "'ap_win' (PR‑AUC after sliding-window max), or 'rap' "
                        '(Recall at Precision >= --select-min-precision)')
    p.add_argument('--select-min-precision', type=float, default=0.90, help='Target precision for rap selection (e.g., 0.90 or 0.95)')
    p.add_argument('--select-window-len', type=int, default=0, help='Window length for ap_win selection (0 disables ap_win)')
    p.add_argument('--select-window-stride', type=int, default=60, help='Window stride for ap_win selection')
    p.add_argument('--disable-iou-best', action='store_true', help='Do not save IoU-best checkpoint (still logged)')

    # Early stopping / schedulers / misc
    p.add_argument('--patience', type=int, default=5, help='Early stopping patience in epochs (measured on the selection metric)')
    p.add_argument('--min-epochs', type=int, default=0, help='Minimum #epochs that must elapse before ES can trigger')
    p.add_argument('--scheduler', choices=['cosine','warm_restarts','plateau'], default='cosine', help='Learning-rate scheduler')
    p.add_argument('--eta-min', type=float, default=1e-6, help='eta_min for cosine / warm restarts')
    p.add_argument('--t0', type=int, default=10, help='Warm restarts: first cycle length T0 (epochs)')
    p.add_argument('--t-mult', type=float, default=2.0, help='Warm restarts: cycle length multiplier T_{i+1} = T_i * t_mult')
    p.add_argument('--lr-patience', type=int, default=5, help='ReduceLROnPlateau: epochs without improvement before LR drop')
    p.add_argument('--lr-factor', type=float, default=0.5, help='ReduceLROnPlateau: multiplicative LR drop factor')
    p.add_argument('--lr-min', type=float, default=1e-6, help='ReduceLROnPlateau: absolute minimum LR')
    p.add_argument('--train-frac', type=float, default=0.8, help='Fraction of base-protein groups for training set')
    p.add_argument('--val-frac', type=float, default=0.1, help='Fraction of base-protein groups for validation set')
    p.add_argument('--num-workers', type=int, default=2, help='Number of DataLoader workers')
    p.add_argument('--seed', type=int, default=1337, help='Random seed')
    p.add_argument('--no-amp', action='store_true', help='Disable mixed precision even on CUDA')
    p.add_argument('--calibrate', action='store_true', help='Calibrate temperature on validation set and save in best checkpoint')

    args = p.parse_args()
    return TrainConfig(
        embeddings=args.embeddings,
        labels=args.labels,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        pu_prior=args.pu_prior,
        lambda_clean_neg=args.lambda_clean_neg,
        lambda_span=args.lambda_span,
        lambda_anchor=args.lambda_anchor,
        lambda_contrast=args.lambda_contrast,
        contrast_margin=args.contrast_margin,
        tau=args.tau,
        alpha_cap=args.alpha_cap,
        anchor_align_kl=args.anchor_align_kl,
        anchor_align_ramp=args.anchor_align_ramp,
        anchor_alpha_prior=args.anchor_alpha_prior,
        k_sigma=args.k_sigma,
        huber_beta=args.huber_beta,
        iou_weight=args.iou_weight,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        num_workers=args.num_workers,
        seed=args.seed,
        patience=args.patience,
        # ES & schedulers
        min_epochs=args.min_epochs,
        scheduler=args.scheduler,
        eta_min=args.eta_min,
        t0=args.t0,
        t_mult=args.t_mult,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        wmin_anneal_epochs=args.wmin_anneal_epochs,
        amp=not args.no_amp,
        calibrate=args.calibrate,
        # discovery & PU anneal
        wmin_base=args.wmin_base,
        wmin_floor=args.wmin_floor,
        lenfeat_scale=args.lenfeat_scale,
        coarse_stride=args.coarse_stride,
        tau_len_gamma=args.tau_len_gamma,
        tau_len_ref=args.tau_len_ref,
        pu_prior_start=args.pu_prior_start,
        pu_prior_end=args.pu_prior_end,
        pu_prior_anneal_epochs=args.pu_prior_anneal_epochs,
        # augs
        token_dropout=args.token_dropout,
        token_dropout_protect=args.token_dropout_protect,
        emb_noise_std=args.emb_noise_std,
        emb_noise_warmup_epochs=args.emb_noise_warmup_epochs,
        emb_noise_decay_epochs=args.emb_noise_decay_epochs,
        crop_prob_pos=args.crop_prob_pos,
        crop_prob_un=args.crop_prob_un,
        crop_windows=args.crop_windows,
        crop_jitter_frac=args.crop_jitter_frac,
        crop_center_only=args.crop_center_only,
        pos_channel=args.pos_channel,
        # selection
        select_metric=args.select_metric,
        select_min_precision=args.select_min_precision,
        select_window_len=args.select_window_len,
        select_window_stride=args.select_window_stride,
        disable_iou_best=args.disable_iou_best,
    )

if __name__ == '__main__':
    cfg = parse_args()
    train(cfg)

