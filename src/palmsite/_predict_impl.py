#!/usr/bin/env python
"""
Predict RdRP probability and catalytic span from embeddings.h5 using a trained checkpoint,
and (optionally) export **embedding token vectors** from the predicted catalytic region
to an **HDF5** file. Now also saves the **final attention weights** for the exported span.
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
from torch import nn
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
    """Stable softmax with a boolean mask."""
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
            emb = g["emb"][:]
            mask = g["mask"][:].astype(bool)
            seq = g["seq"][()] if "seq" in g else None
            pos = np.where(mask)[0].astype(np.int32)
            L = mask.sum().astype(np.int32)

            # Attributes from your HDF5 writer
            attrs = dict(g.attrs)
            orig_start = int(attrs.get("orig_aa_start", 0))
            orig_len = int(attrs.get("orig_aa_len", L))

        # positional channel as last dim (normalized [0,1], end-inclusive)
        pos_channel = np.linspace(0.0, 1.0, num=int(L), endpoint=True, dtype=emb.dtype)
        pos_full = np.zeros_like(mask, dtype=emb.dtype)
        pos_full[mask] = pos_channel
        pos_full = pos_full[:, None]  # (T,1)

        x = np.concatenate([emb, pos_full], axis=-1)  # (T, D+1)

        return {
            "chunk_id": cid,
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "L": torch.tensor(L, dtype=torch.int32),
            "pos": torch.from_numpy(pos),
            "orig_start": torch.tensor(orig_start, dtype=torch.int32),
            "orig_len": torch.tensor(orig_len, dtype=torch.int32),
            "seq": seq,
        }


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Pad to max T in batch
    T = max(int(x["x"].shape[0]) for x in batch)
    D = int(batch[0]["x"].shape[1])
    B = len(batch)
    x = torch.zeros(B, T, D, dtype=torch.float32)
    mask = torch.zeros(B, T, dtype=torch.bool)
    L = torch.zeros(B, dtype=torch.int32)
    chunk_ids, seqs, orig_start, orig_len = [], [], [], []
    for i, b in enumerate(batch):
        t = int(b["x"].shape[0])
        x[i, :t] = b["x"]
        mask[i, :t] = b["mask"]
        L[i] = b["L"]
        chunk_ids.append(b["chunk_id"])
        seqs.append(b["seq"])
        orig_start.append(b["orig_start"])
        orig_len.append(b["orig_len"])
    return {
        "x": x,
        "mask": mask,
        "L": L,
        "chunk_ids": chunk_ids,
        "seqs": seqs,
        "orig_start": torch.stack(orig_start),
        "orig_len": torch.stack(orig_len),
    }


# ----------------------------
# Model (name-compatible with training checkpoints)
# ----------------------------

class TokenBackbone(nn.Module):
    """
    Backward-compatible backbone:
    - Registers a single Sequential at `mlp` so checkpoints with keys `backbone.mlp.*` load.
    - Provides read-only l1/l2/l3 *views* for readability (not registered).
    """
    def __init__(self, d_in: int, p_drop: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in),              # mlp.0
            nn.Linear(d_in, 1024),           # mlp.1
            nn.GELU(),                       # mlp.2
            nn.Dropout(p_drop),              # mlp.3
            nn.LayerNorm(1024),              # mlp.4
            nn.Linear(1024, 512),            # mlp.5
            nn.GELU(),                       # mlp.6
            nn.Dropout(p_drop),              # mlp.7
            nn.LayerNorm(512),               # mlp.8
            nn.Linear(512, 256),             # mlp.9
            nn.GELU(),                       # mlp.10
        )

    def forward(self, x, mask):
        return self.mlp(x)

    # Convenience views (not registered; only for code readability if you want to inspect)
    @property
    def l1(self):  # LayerNorm(d_in) → Linear → GELU → Dropout
        return nn.Sequential(*self.mlp[:4])

    @property
    def l2(self):  # LayerNorm(1024) → Linear → GELU → Dropout
        return nn.Sequential(*self.mlp[4:8])

    @property
    def l3(self):  # LayerNorm(512) → Linear → GELU
        return nn.Sequential(*self.mlp[8:11])


class RdRPModel(nn.Module):
    """
    Runtime model with names aligned to training:
      - backbone.mlp.*
      - scorer.net.*              (alias token_scorer)
      - heads.seq_head / heads.span_head / heads.alpha_head (alpha is unused stub)
    Aliases keep your code readable (token_scorer, seq_head, span_head).
    """
    def __init__(self, d_in: int, tau: float = 3.0, alpha_cap: float = 2.0, p_drop: float = 0.1):
        super().__init__()
        self.backbone = TokenBackbone(d_in, p_drop=p_drop)

        # scorer.* (name used in checkpoints)
        self.scorer = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
        # alias used in your previous runtime code
        self.token_scorer = self.scorer

        # heads.* container (names used in checkpoints)
        seq_span_in = 256 + 2
        self.heads = nn.Module()
        self.heads.seq_head = nn.Sequential(
            nn.Linear(seq_span_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        self.heads.span_head = nn.Sequential(
            nn.Linear(seq_span_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 2)
        )
        # unused in inference but present in some checkpoints; define to satisfy load_state_dict
        self.heads.alpha_head = nn.Sequential(
            nn.Linear(seq_span_in, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )
        # friendly aliases for code clarity
        self.seq_head = self.heads.seq_head
        self.span_head = self.heads.span_head

        # inference knobs
        self.tau = torch.tensor(tau)
        self.alpha_cap = float(alpha_cap)
        self.wmin_base = 70.0
        self.wmin_floor = 0.02
        self.lenfeat_scale = 1.0
        self.k_sigma = 2.0
        self.coarse_stride = 0
        self.tau_len_gamma = 0.0
        self.tau_len_ref = 1000.0

    def forward(self, x, mask, L):
        H = self.backbone(x, mask)                      # (B,T,256)
        z = self.scorer(H).squeeze(-1)                  # (B,T)

        # Gaussian attention anchor via soft-argmax
        B, T = z.shape
        pos = torch.linspace(0, 1, T, device=z.device).unsqueeze(0).expand(B, T)
        z_soft = masked_softmax(z / self.tau, mask, dim=-1)    # (B,T)
        mu = (z_soft * pos).sum(dim=-1)                        # (B,)

        # sequence & span heads
        len_feat = torch.stack(
            [torch.log(L.float() + 1.0), 1.0 / (L.float().clamp(min=1.0))], dim=-1
        )  # (B,2)
        pooled = (z_soft.unsqueeze(-1) * H).sum(dim=1)         # (B,256)
        feat = torch.cat([pooled, len_feat], dim=-1)           # (B, 256+2)

        logit = self.heads.seq_head(feat).squeeze(-1)          # (B,)
        span_raw = self.heads.span_head(feat)                  # (B,2)
        S = span_raw[:, 0].sigmoid()
        l = span_raw[:, 1].sigmoid()
        E = S + l * (1 - S)
        sigma = (E - S) / (2.0 * max(self.k_sigma, 1e-6))

        return {
            "H": H, "z": z, "z_soft": z_soft, "mu": mu, "sigma": sigma,
            "logit": logit, "S_pred": S, "E_pred": E
        }


# ----------------------------
# HPD helper
# ----------------------------

def _attn_hpd_span_from_out(out: Dict[str, torch.Tensor], mask: torch.Tensor, L: torch.Tensor,
                            mass: float = 0.90, pos_channel: str = 'end_inclusive') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Highest Posterior Density (HPD) span that covers `mass` of attention."""
    z = out["z_soft"].detach().cpu().numpy()
    mask_np = mask.cpu().numpy()
    B, T = z.shape
    S = np.zeros(B, dtype=np.float32)
    E = np.ones(B, dtype=np.float32)
    ent = np.zeros(B, dtype=np.float32)
    for i in range(B):
        w = z[i] * mask_np[i]
        w_sum = w.sum()
        if w_sum <= 0:
            S[i], E[i], ent[i] = 0.0, 1.0, 0.0
            continue
        w = w / w_sum
        ent[i] = float(-(w[w > 0] * np.log(w[w > 0] + 1e-12)).sum())
        order = np.argsort(-w)  # high → low
        csum = 0.0
        idxs = []
        for j in order:
            csum += w[j]
            idxs.append(j)
            if csum >= mass:
                break
        s, e = min(idxs), max(idxs)
        S[i] = s / max(int(L[i].item()) - 1, 1)
        E[i] = e / max(int(L[i].item()) - 1, 1)
    return S, E, ent


# ----------------------------
# Core execution (shared by CLI and library)
# ----------------------------

def _run_with_namespace(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
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

    # Build model
    with h5py.File(args.embeddings, "r") as h5:
        any_id = next(iter(h5["items"].keys()))
        g = h5[f"items/{any_id}"]
        d_model = int(g.attrs.get("d_model", g["emb"].shape[1]))

    model = RdRPModel(d_in=d_model + 1, tau=tau, alpha_cap=alpha_cap, p_drop=p_drop).to(device)
    model.wmin_base, model.wmin_floor = wmin_base, wmin_floor
    model.lenfeat_scale = len_scale
    model.k_sigma = k_sigma
    model.coarse_stride = coarse
    model.tau_len_gamma = tau_gam
    model.tau_len_ref = tau_ref

    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    # strip DDP / compile wrappers
    cleaned = {}
    for k, v in state.items():
        kk = k
        for p in ("_orig_mod.", "module."):
            if kk.startswith(p):
                kk = kk[len(p):]
        cleaned[kk] = v
    model.load_state_dict(cleaned, strict=True)
    model.eval()

    # temperature
    T = float(ckpt.get("temperature", 1.0)) if isinstance(ckpt, dict) else 1.0

    ids = None
    if args.ids_file:
        with open(args.ids_file, "r", encoding="utf-8") as f:
            ids = [ln.strip() for ln in f if ln.strip()]

    ds = EmbOnlyDataset(args.embeddings, ids=ids, pos_channel='end_inclusive')
    dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=False,
                    num_workers=int(args.num_workers), pin_memory=True, collate_fn=collate)

    # open CSV
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fcsv = open(args.out, "w", newline="", encoding="utf-8")
    w = csv.writer(fcsv)
    w.writerow(["chunk_id", "P", "S", "E", "S_idx", "E_idx", "mu", "mu_idx", "sigma", "L"])

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
                w.writerow([cid, float(P[i]), float(S[i]), float(E[i]),
                            int(S_idx[i]), int(E_idx[i]), float(mu[i]), int(mu_idx[i]),
                            float(sigma[i]), int(Lnp[i])])
                bid = base_id_from_chunk(cid)
                aS = int(orig_start[i] + S_idx[i]) + 1  # 1-based for GFF (but CSV keeps chunk idx)
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

    # base-level TSV if requested
    if args.base_out:
        os.makedirs(os.path.dirname(args.base_out) or ".", exist_ok=True)
        with open(args.base_out, "w", newline="", encoding="utf-8") as f:
            bw = csv.writer(f)
            bw.writerow(["base_id","P","best_chunk","S_abs","E_abs","mu_abs","sigma","Lorig","S","E","attn_entropy"])
            for bid, v in agg.items():
                bw.writerow([bid] + list(v))

    # GFF if requested (support "-" for stdout)
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
                         f"SpanSource={'HPD' if args.discovery else 'kSigma'}", f"AttnMass={args.attn_mass:.2f}",
                         f"AttnEntropy={ent:.6f}"]
                row = [bid, args.gff_source, args.gff_type, str(aS), str(aE), f"{Pmax:.6f}", ".", ".", ";".join(attrs)]
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
        description='Predict RdRP probability and span from embeddings.h5 '
                    '(+ optional HDF5 export of catalytic vectors + attention weights)'
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

    ap.add_argument("--extract-h5", default=None, help="Write catalytic vectors+weights to HDF5")
    ap.add_argument("--min-p", type=float, default=0.90)
    ap.add_argument("--h5-coords", choices=["abs", "chunk"], default="abs")

    args = ap.parse_args()
    _run_with_namespace(args)


# ----------------------------
# Library entrypoint for PalmSite’s top-level CLI
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
