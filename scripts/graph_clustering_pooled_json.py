#!/usr/bin/env python3
"""
graph_clustering_pooled_json.py

Scalable kNN-graph clustering for PalmSite pooled vectors, including direct support
for palmsite --pooled-json output.

Supported inputs
----------------
1. PalmSite pooled JSON from:
   palmsite --pooled-json pooled_panels.json --pool-include-input proteins.faa

2. NPZ matrix files with keys such as:
   names + X

3. Headerless TSV matrix files:
   sequence_id<TAB>v1<TAB>v2<TAB>...

4. Legacy HDF5 predictor exports with /items/<id>/vec and optional /items/<id>/w.

Main fixes vs graph_clustering_from_h5.v3.2.py
---------------------------------------------
- Added native JSON support for PalmSite pooled panels.
- Added native NPZ support.
- Fixed dedupe='none' so duplicate IDs are retained instead of silently dropping all but one.
- Deduplicated subset kNN edges inside bootstrap workers.
- Made bootstrap subset kNN respect --mutual-knn.
- Added --min-edge-observations for consensus edges.
- Added --consensus-weight-mode to optionally combine stability and original similarity.
- Passed weights into final Leiden clustering on the consensus graph.
- Added PCA component safety guards.
- Removed duplicated edge_to_idx construction.

Recommended PalmSite panels
---------------------------
--json-panel backbone.span_attn_norm   # main PalmSite clustering
--json-panel input.span_attn_norm      # matched ESM-C control using same span/weights
--json-panel backbone.span_mean        # unweighted PalmSite span control
--json-panel input.span_mean           # unweighted ESM-C span control
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import igraph as ig
import leidenalg
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
import multiprocessing as mp


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class VectorRecord:
    name: str
    vector: np.ndarray
    P: float
    meta: Dict[str, Any]
    provenance: List[Tuple[str, str]]


# ----------------------------
# Utilities
# ----------------------------

def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < eps] = eps
    return X / nrm


def _safe_mean(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        raise ValueError("Empty array passed to mean pooling")
    return X.mean(axis=0)


def _pool_item(vec: np.ndarray, w: Optional[np.ndarray], mode: str) -> np.ndarray:
    """
    Pool per-residue vectors from legacy HDF5 inputs.

    mode='mean'      : unweighted mean
    mode='sum'       : unweighted sum
    mode='attn'      : weighted sum using raw w
    mode='attn-norm' : weighted mean using w normalized over selected residues
    """
    vec = np.asarray(vec, dtype=np.float64)
    if vec.ndim != 2:
        raise ValueError(f"Expected a 2D residue-vector array, got shape {vec.shape}")

    if mode == "mean":
        return _safe_mean(vec)
    if mode == "sum":
        return vec.sum(axis=0)

    if w is None:
        return _safe_mean(vec)

    w = np.asarray(w, dtype=np.float64).reshape(-1)
    if len(w) != vec.shape[0]:
        raise ValueError(f"Weight length mismatch: len(w)={len(w)} but vec has {vec.shape[0]} rows")

    if mode == "attn-norm":
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            w = np.ones_like(w, dtype=np.float64) / max(len(w), 1)
        else:
            w = w / s

    return (vec * w[:, None]).sum(axis=0)


def _safe_key(s: str) -> str:
    return str(s).replace("/", "_")


def _as_float_or_nan(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _make_jsonable_meta(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy scalars/bytes to ordinary Python objects when possible."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, bytes):
            out[str(k)] = v.decode("utf-8", errors="replace")
        elif isinstance(v, np.generic):
            out[str(k)] = v.item()
        elif isinstance(v, np.ndarray):
            out[str(k)] = v.tolist()
        else:
            out[str(k)] = v
    return out


# ----------------------------
# Duplicate handling
# ----------------------------

def _make_names_unique(records: List[VectorRecord], logger: Optional[logging.Logger] = None) -> List[VectorRecord]:
    """
    Retain all records but make duplicate output names unambiguous.

    The first occurrence keeps the original name. Later occurrences get:
      <name>|dup_0002, <name>|dup_0003, ...
    The original name is stored in meta['original_name'].
    """
    seen: Dict[str, int] = defaultdict(int)
    n_dups = 0
    out: List[VectorRecord] = []
    for rec in records:
        seen[rec.name] += 1
        if seen[rec.name] == 1:
            out.append(rec)
            continue
        n_dups += 1
        new_meta = dict(rec.meta)
        new_meta.setdefault("original_name", rec.name)
        new_name = f"{rec.name}|dup_{seen[rec.name]:04d}"
        out.append(VectorRecord(new_name, rec.vector, rec.P, new_meta, rec.provenance))
    if logger and n_dups:
        logger.warning(
            "Retained %d duplicate IDs by appending |dup_XXXX suffixes. "
            "Use --dedupe maxP or --dedupe mean if you want one vector per ID.",
            n_dups,
        )
    return out


def _resolve_records(
    records: List[VectorRecord],
    dedupe: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]], Dict[str, List[Tuple[str, str]]]]:
    if not records:
        raise ValueError("No vectors were loaded from the input file(s).")

    if dedupe == "none":
        kept = _make_names_unique(records, logger=logger)
    else:
        grouped: Dict[str, List[VectorRecord]] = defaultdict(list)
        for rec in records:
            grouped[rec.name].append(rec)

        kept: List[VectorRecord] = []
        n_groups_with_dups = 0
        for name, group in grouped.items():
            if len(group) == 1:
                kept.append(group[0])
                continue
            n_groups_with_dups += 1
            if dedupe == "maxP":
                best = max(group, key=lambda r: (float("-inf") if math.isnan(r.P) else r.P))
                meta = dict(best.meta)
                meta["dedupe_mode"] = "maxP"
                meta["dedupe_n_records"] = len(group)
                prov: List[Tuple[str, str]] = []
                for r in group:
                    prov.extend(r.provenance)
                kept.append(VectorRecord(name, best.vector, best.P, meta, prov))
            elif dedupe == "mean":
                Xg = np.stack([r.vector for r in group], axis=0)
                pooled = Xg.mean(axis=0)
                best_meta_rec = max(group, key=lambda r: (float("-inf") if math.isnan(r.P) else r.P))
                meta = dict(best_meta_rec.meta)
                meta["dedupe_mode"] = "mean"
                meta["dedupe_n_records"] = len(group)
                P = best_meta_rec.P
                prov = []
                for r in group:
                    prov.extend(r.provenance)
                kept.append(VectorRecord(name, pooled, P, meta, prov))
            else:
                raise ValueError(f"Unsupported dedupe mode: {dedupe}")

        if logger and n_groups_with_dups:
            logger.info("Resolved %d duplicate ID groups with dedupe=%s", n_groups_with_dups, dedupe)

    names = [r.name for r in kept]
    dims = {int(np.asarray(r.vector).shape[0]) for r in kept}
    if len(dims) != 1:
        raise ValueError(f"Vector dimension mismatch among records: {sorted(dims)}")

    X = np.stack([np.asarray(r.vector, dtype=np.float64) for r in kept], axis=0)
    metas = {r.name: r.meta for r in kept}
    provenance = {r.name: r.provenance for r in kept}
    return names, X, metas, provenance


# ----------------------------
# Data loading: PalmSite pooled JSON
# ----------------------------

def _normalize_json_panel(panel: str) -> Tuple[str, str]:
    """
    Convert panel strings into (source, panel_name).

    Accepted examples:
      backbone.span_attn_norm
      input.span_mean
      pools.backbone.span_attn_norm
    """
    p = panel.strip()
    if p.startswith("pools."):
        p = p[len("pools."):]
    parts = p.split(".")
    if len(parts) != 2:
        raise ValueError(
            "--json-panel must look like 'backbone.span_attn_norm', "
            "'input.span_mean', or 'pools.backbone.span_attn_norm'."
        )
    source, panel_name = parts
    if source not in {"backbone", "input"}:
        raise ValueError("JSON panel source must be 'backbone' or 'input'.")
    return source, panel_name


def _json_id(entry_key: str, entry: Dict[str, Any], id_field: str) -> str:
    if id_field == "entry_key":
        return str(entry_key)
    if id_field in entry and entry[id_field] is not None:
        return str(entry[id_field])
    if id_field == "chunk_id" and "chunk_id" in entry:
        return str(entry["chunk_id"])
    if id_field == "base_id" and "base_id" in entry:
        return str(entry["base_id"])
    return str(entry_key)


def _read_pooled_json_multi(
    paths: List[str],
    panel: str,
    id_field: str,
    best_only: bool,
    dedupe: str,
    l2: bool,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]], Dict[str, List[Tuple[str, str]]]]:
    source, panel_name = _normalize_json_panel(panel)
    records: List[VectorRecord] = []
    skipped_meta = 0
    skipped_best = 0
    skipped_missing = 0
    start_t = time.perf_counter()

    for path in paths:
        if logger:
            logger.info("Reading pooled JSON: %s | panel=%s.%s", path, source, panel_name)
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if not isinstance(obj, dict):
            raise ValueError(f"Expected top-level JSON object in {path}")

        for entry_key, entry in obj.items():
            if entry_key == "_meta":
                skipped_meta += 1
                continue
            if not isinstance(entry, dict):
                skipped_missing += 1
                continue
            if best_only and entry.get("is_best_base_chunk") is False:
                skipped_best += 1
                continue

            try:
                pools = entry["pools"]
                vec_raw = pools[source][panel_name]
            except Exception:
                skipped_missing += 1
                continue

            vector = np.asarray(vec_raw, dtype=np.float64)
            if vector.ndim != 1:
                raise ValueError(
                    f"Panel {source}.{panel_name} in {path}:{entry_key} is not a 1D vector; "
                    f"shape={vector.shape}"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"Panel {source}.{panel_name} in {path}:{entry_key} contains NaN/Inf")

            name = _json_id(entry_key, entry, id_field=id_field)
            P = _as_float_or_nan(entry.get("P", np.nan))
            meta = {
                "entry_key": entry_key,
                "chunk_id": entry.get("chunk_id", entry_key),
                "base_id": entry.get("base_id", ""),
                "L": entry.get("L", np.nan),
                "orig_start": entry.get("orig_start", np.nan),
                "orig_len": entry.get("orig_len", np.nan),
                "P": P,
                "logit": entry.get("logit", np.nan),
                "S_idx": entry.get("S_idx", np.nan),
                "E_idx": entry.get("E_idx", np.nan),
                "S_norm": entry.get("S_norm", np.nan),
                "E_norm": entry.get("E_norm", np.nan),
                "mu": entry.get("mu", np.nan),
                "sigma": entry.get("sigma", np.nan),
                "mu_attn": entry.get("mu_attn", np.nan),
                "sigma_attn": entry.get("sigma_attn", np.nan),
                "is_best_base_chunk": entry.get("is_best_base_chunk", None),
                "json_panel": f"{source}.{panel_name}",
                "json_source": source,
                "json_panel_name": panel_name,
            }
            if isinstance(entry.get("pool_meta"), dict):
                pool_meta = entry["pool_meta"]
                for k in (
                    "l2_normalized",
                    "backbone_dim",
                    "input_dim",
                    "span_local_start",
                    "span_local_end",
                    "span_len",
                    "top_k_used",
                    "input_controls",
                ):
                    if k in pool_meta:
                        meta[f"pool_meta.{k}"] = pool_meta[k]

            records.append(VectorRecord(name, vector, P, meta, [(path, entry_key)]))

    if logger:
        logger.info(
            "Loaded JSON vectors: records=%d, skipped_meta=%d, skipped_best=%d, skipped_missing_panel=%d, elapsed=%.2fs",
            len(records),
            skipped_meta,
            skipped_best,
            skipped_missing,
            time.perf_counter() - start_t,
        )

    names, X, metas, provenance = _resolve_records(records, dedupe=dedupe, logger=logger)
    if l2:
        X = _l2_normalize_rows(X)
    return names, X, metas, provenance


# ----------------------------
# Data loading: NPZ
# ----------------------------

def _choose_npz_key(npz: Any, preferred: Sequence[str], required_ndim: Optional[int] = None) -> str:
    for key in preferred:
        if key in npz.files:
            if required_ndim is None or np.asarray(npz[key]).ndim == required_ndim:
                return key
    for key in npz.files:
        if required_ndim is None or np.asarray(npz[key]).ndim == required_ndim:
            return key
    raise ValueError(f"Could not find an NPZ key with ndim={required_ndim}; available keys={npz.files}")


def _read_npz_multi(
    paths: List[str],
    dedupe: str,
    l2: bool,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]], Dict[str, List[Tuple[str, str]]]]:
    records: List[VectorRecord] = []
    start_t = time.perf_counter()

    for path in paths:
        if logger:
            logger.info("Reading NPZ: %s", path)
        with np.load(path, allow_pickle=True) as npz:
            x_key = _choose_npz_key(npz, preferred=("X", "vectors", "embeddings", "arr_0"), required_ndim=2)
            X = np.asarray(npz[x_key], dtype=np.float64)

            name_key: Optional[str] = None
            for candidate in ("names", "ids", "identifiers", "labels", "arr_1"):
                if candidate in npz.files and np.asarray(npz[candidate]).ndim == 1:
                    if len(np.asarray(npz[candidate])) == X.shape[0]:
                        name_key = candidate
                        break

            if name_key is None:
                names = [f"{os.path.basename(path)}::row_{i:08d}" for i in range(X.shape[0])]
            else:
                names = [str(x) for x in np.asarray(npz[name_key]).tolist()]

            if not np.isfinite(X).all():
                raise ValueError(f"NPZ matrix {path}:{x_key} contains NaN/Inf")

            for i, name in enumerate(names):
                meta = {"npz_x_key": x_key, "npz_name_key": name_key or "", "row_index": i}
                records.append(VectorRecord(name, X[i], float("nan"), meta, [(path, f"row_{i}")]))

    if logger:
        logger.info("Loaded NPZ vectors: records=%d, elapsed=%.2fs", len(records), time.perf_counter() - start_t)

    names, X, metas, provenance = _resolve_records(records, dedupe=dedupe, logger=logger)
    if l2:
        X = _l2_normalize_rows(X)
    return names, X, metas, provenance


# ----------------------------
# Data loading: TSV
# ----------------------------

def _read_tsv(
    path: str,
    l2: bool,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]], Dict[str, List[Tuple[str, str]]]]:
    import pandas as pd

    t0 = time.perf_counter()
    df = pd.read_csv(path, sep="\t", header=None)
    if df.shape[1] < 2:
        raise ValueError("TSV input must have at least two columns: ID and one vector dimension.")

    names = df.iloc[:, 0].astype(str).tolist()
    X = df.iloc[:, 1:].to_numpy(dtype=np.float64)
    if not np.isfinite(X).all():
        raise ValueError(f"TSV matrix {path} contains NaN/Inf")
    if l2:
        X = _l2_normalize_rows(X)

    metas = {n: {"row_index": i} for i, n in enumerate(names)}
    provenance = {n: [(path, f"row_{i}")] for i, n in enumerate(names)}
    if logger:
        logger.info("Loaded TSV: N=%d, d=%d in %.2fs", X.shape[0], X.shape[1], time.perf_counter() - t0)
    return names, X, metas, provenance


# ----------------------------
# Data loading: legacy HDF5
# ----------------------------

def _read_h5_multi(
    paths: List[str],
    pool: str,
    id_field: str,
    dedupe: str,
    l2: bool,
    log_interval: int,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]], Dict[str, List[Tuple[str, str]]]]:
    start_t = time.perf_counter()
    records: List[VectorRecord] = []
    d_check: Optional[int] = None
    total_groups = 0
    skipped_no_vec = 0

    for path in paths:
        with h5py.File(path, "r") as h5:
            if "items" not in h5:
                raise ValueError(f"HDF5 input {path} does not contain /items")
            keys = list(h5["items"].keys())
            n_keys = len(keys)
            if logger:
                logger.info("Reading HDF5 %s: %d groups", path, n_keys)

            for i, key in enumerate(keys, 1):
                g = h5["items"][key]
                if "vec" not in g:
                    skipped_no_vec += 1
                    continue
                V = g["vec"][...].astype(np.float64, copy=False)
                if V.ndim != 2:
                    raise ValueError(f"Expected /items/{key}/vec to be 2D, got shape={V.shape}")
                d = int(V.shape[1])
                d_check = d if d_check is None else d_check
                if d_check != d:
                    raise ValueError(f"Dimension mismatch across HDF5 files: saw {d_check} and {d}")

                W = g["w"][...].astype(np.float64, copy=False) if "w" in g else None
                name = str(g.attrs.get(id_field, key))
                P = _as_float_or_nan(g.attrs.get("P", np.nan))
                meta = _make_jsonable_meta(dict(g.attrs.items()))
                meta.setdefault("h5_key", key)
                pooled = _pool_item(V, W, mode=pool).astype(np.float64, copy=False)
                records.append(VectorRecord(name, pooled, P, meta, [(path, key)]))
                total_groups += 1

                if logger and (i % log_interval == 0):
                    logger.info("  ...%d/%d groups processed in %s", i, n_keys, path)

    if logger:
        logger.info(
            "Read HDF5 records: total_groups=%d, skipped_no_vec=%d, elapsed=%.2fs",
            total_groups,
            skipped_no_vec,
            time.perf_counter() - start_t,
        )

    names, X, metas, provenance = _resolve_records(records, dedupe=dedupe, logger=logger)
    if l2:
        X = _l2_normalize_rows(X)
    return names, X, metas, provenance


# ----------------------------
# Input dispatch
# ----------------------------

def _infer_format(paths: List[str]) -> str:
    exts = {os.path.splitext(p.lower())[1] for p in paths}
    if exts and exts <= {".json"}:
        return "json"
    if exts and exts <= {".npz"}:
        return "npz"
    if any(e in (".h5", ".hdf5") for e in exts):
        return "h5"
    return "tsv"


def load_vectors(args: argparse.Namespace, logger: logging.Logger) -> Tuple[List[str], np.ndarray, Dict[str, Dict[str, Any]], Dict[str, List[Tuple[str, str]]], str]:
    fmt = args.input_format
    if fmt == "auto":
        fmt = _infer_format(args.input)
    logger.info("Format: %s", fmt.upper())

    l2 = not args.no_l2
    if fmt == "json":
        names, X, metas, provenance = _read_pooled_json_multi(
            paths=args.input,
            panel=args.json_panel,
            id_field=args.id_field,
            best_only=args.json_best_only,
            dedupe=args.dedupe,
            l2=l2,
            logger=logger,
        )
    elif fmt == "npz":
        names, X, metas, provenance = _read_npz_multi(
            paths=args.input,
            dedupe=args.dedupe,
            l2=l2,
            logger=logger,
        )
    elif fmt == "tsv":
        if len(args.input) != 1:
            raise ValueError("TSV mode supports a single input file. Use JSON/NPZ/HDF5 mode for multi-file input.")
        names, X, metas, provenance = _read_tsv(args.input[0], l2=l2, logger=logger)
    elif fmt == "h5":
        names, X, metas, provenance = _read_h5_multi(
            paths=args.input,
            pool=args.pool,
            id_field=args.id_field,
            dedupe=args.dedupe,
            l2=l2,
            log_interval=args.log_interval,
            logger=logger,
        )
    else:
        raise ValueError(f"Unsupported input format: {fmt}")

    return names, X, metas, provenance, fmt


# ----------------------------
# kNN graph
# ----------------------------

def _knn_to_edges(
    ind: np.ndarray,
    sims: np.ndarray,
    mutual: bool,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """Convert neighbor indices/similarities into deduplicated undirected edges."""
    N, k_eff = ind.shape
    best: Dict[Tuple[int, int], float] = {}

    if not mutual:
        for i in range(N):
            for j_idx in range(k_eff):
                j = int(ind[i, j_idx])
                if i == j:
                    continue
                u, v = (i, j) if i < j else (j, i)
                w = float(sims[i, j_idx])
                if (u, v) not in best or w > best[(u, v)]:
                    best[(u, v)] = w
    else:
        neighbor_sets = [set(ind[i].tolist()) for i in range(N)]
        for i in range(N):
            for j_idx in range(k_eff):
                j = int(ind[i, j_idx])
                if i == j:
                    continue
                if i not in neighbor_sets[j]:
                    continue
                u, v = (i, j) if i < j else (j, i)
                reciprocal_positions = np.where(ind[j] == i)[0]
                if len(reciprocal_positions) > 0:
                    jpos = int(reciprocal_positions[0])
                    w = (float(sims[i, j_idx]) + float(sims[j, jpos])) / 2.0
                else:
                    w = float(sims[i, j_idx])
                if (u, v) not in best or w > best[(u, v)]:
                    best[(u, v)] = w

    edges = list(best.keys())
    weights = [best[e] for e in edges]
    return edges, weights


def build_knn_graph(
    vectors: np.ndarray,
    k: int = 30,
    mutual: bool = False,
    metric: str = "cosine",
    nn_jobs: int = 1,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[Tuple[int, int]], List[float], List[List[int]]]:
    """Build an undirected deduplicated kNN graph on the full set."""
    t0 = time.perf_counter()
    N = vectors.shape[0]
    if N <= 1:
        return [], [], [[] for _ in range(N)]

    k_eff = max(1, min(k, N - 1))
    if logger:
        logger.info(
            "Building GLOBAL kNN: N=%d, d=%d, k=%d, metric=%s, mutual=%s, nn_jobs=%s",
            N,
            vectors.shape[1],
            k_eff,
            metric,
            mutual,
            nn_jobs,
        )

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=nn_jobs).fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)
    ind = indices[:, 1:]
    dist = distances[:, 1:]

    if metric == "cosine":
        sims = np.maximum(0.0, 1.0 - dist)
    else:
        sims = 1.0 / (1.0 + dist)

    edges, weights = _knn_to_edges(ind, sims, mutual=mutual)

    adj: List[List[int]] = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    if logger:
        avg_deg = (2.0 * len(edges)) / max(1, N)
        logger.info("GLOBAL kNN built: |E|=%d, avg_degree=%.2f, elapsed=%.2fs", len(edges), avg_deg, time.perf_counter() - t0)
    return edges, weights, adj


# ----------------------------
# Clustering on one graph
# ----------------------------

def _cluster_graph(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    weights: List[float],
    algorithm: str,
    resolution: float,
    seed: int,
) -> List[int]:
    if n_nodes <= 0:
        return []
    if not edges:
        return list(range(n_nodes))

    g = ig.Graph(n=n_nodes)
    g.add_edges(edges)
    g.es["weight"] = weights

    if algorithm == "leiden":
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=resolution,
            seed=seed,
        )
        return list(part.membership)

    return list(g.community_multilevel(weights="weight").membership)


# ----------------------------
# Clustering on a subset for bootstrap workers
# ----------------------------

def _cluster_subset(sub_vectors: np.ndarray, args: argparse.Namespace) -> List[int]:
    N = sub_vectors.shape[0]
    if N <= 1:
        return list(range(N))

    k_eff = max(1, min(args.k, N - 1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=args.metric, n_jobs=1).fit(sub_vectors)
    dist, ind0 = nbrs.kneighbors(sub_vectors)
    ind = ind0[:, 1:]

    if args.metric == "cosine":
        sims = np.maximum(0.0, 1.0 - dist[:, 1:])
    else:
        sims = 1.0 / (1.0 + dist[:, 1:])

    edges, weights = _knn_to_edges(ind, sims, mutual=args.mutual_knn)
    return _cluster_graph(
        n_nodes=N,
        edges=edges,
        weights=weights,
        algorithm=args.algorithm,
        resolution=args.resolution,
        seed=args.seed,
    )


# ----------------------------
# Multiprocessing globals
# ----------------------------

_global_vectors: Optional[np.ndarray] = None
_global_args: Optional[argparse.Namespace] = None
_global_N: Optional[int] = None


def _init_pool(vectors: np.ndarray, args: argparse.Namespace, N: int) -> None:
    global _global_vectors, _global_args, _global_N
    _global_vectors = vectors
    _global_args = args
    _global_N = N


def _bootstrap_worker(b: int) -> Tuple[np.ndarray, np.ndarray]:
    if _global_vectors is None or _global_args is None or _global_N is None:
        raise RuntimeError("Bootstrap worker was not initialized")

    rng = random.Random(_global_args.seed + b)
    m = max(2, int(round(_global_args.sample_fraction * _global_N)))
    m = min(m, _global_N)
    idxs = rng.sample(range(_global_N), m)
    idxs.sort()
    sub_vectors = _global_vectors[idxs]
    memb_sub = _cluster_subset(sub_vectors, _global_args)
    return np.array(idxs, dtype=np.int32), np.array(memb_sub, dtype=np.int32)


def _iter_bootstrap_results(args: argparse.Namespace, X: np.ndarray, N: int) -> Iterable[Tuple[int, Tuple[np.ndarray, np.ndarray]]]:
    if args.workers <= 1:
        _init_pool(X, args, N)
        for b in range(args.bootstrap_iterations):
            yield b + 1, _bootstrap_worker(b)
        return

    with mp.Pool(args.workers, initializer=_init_pool, initargs=(X, args, N)) as pool:
        for idx, result in enumerate(pool.imap(_bootstrap_worker, range(args.bootstrap_iterations), chunksize=1), 1):
            yield idx, result


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="kNN graph clustering for PalmSite pooled JSON/NPZ/TSV/HDF5 vectors."
    )

    p.add_argument("--input", nargs="+", required=True, help="Input JSON, NPZ, TSV, or HDF5 file(s).")
    p.add_argument("--input-format", choices=["auto", "json", "npz", "h5", "tsv"], default="auto")

    p.add_argument(
        "--json-panel",
        default="backbone.span_attn_norm",
        help=(
            "Panel to read from PalmSite pooled JSON. Examples: "
            "backbone.span_attn_norm, input.span_attn_norm, backbone.span_mean, input.span_mean."
        ),
    )
    p.add_argument(
        "--json-best-only",
        action="store_true",
        help="For PalmSite pooled JSON, keep only entries with is_best_base_chunk=true.",
    )

    p.add_argument(
        "--id-field",
        choices=["chunk_id", "base_id", "entry_key"],
        default="chunk_id",
        help="ID used as output node name for JSON/HDF5 inputs.",
    )
    p.add_argument(
        "--dedupe",
        choices=["none", "maxP", "mean"],
        default="none",
        help=(
            "How to handle duplicate node IDs. 'none' now keeps all rows and makes duplicate names unique; "
            "'maxP' keeps the highest-P row; 'mean' averages duplicate vectors."
        ),
    )

    p.add_argument(
        "--pool",
        choices=["attn", "attn-norm", "mean", "sum"],
        default="attn-norm",
        help="Pooling mode for legacy HDF5 per-residue vectors only. Ignored for JSON/NPZ/TSV.",
    )
    p.add_argument("--no-l2", action="store_true", help="Disable final row-wise L2 normalization.")

    p.add_argument("--pca", type=float, default=None, help="PCA components (>1) or variance fraction (0-1).")
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--mutual-knn", action="store_true")
    p.add_argument("--algorithm", choices=["leiden", "louvain"], default="leiden")
    p.add_argument("--resolution", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--bootstrap-iterations", type=int, default=1)
    p.add_argument("--sample-fraction", type=float, default=0.8)
    p.add_argument("--consensus-threshold", type=float, default=0.8)
    p.add_argument(
        "--min-edge-observations",
        type=int,
        default=10,
        help="Minimum number of bootstrap co-samplings required for a consensus edge. Use 0 to disable.",
    )
    p.add_argument("--min-cluster-size", type=int, default=2)
    p.add_argument(
        "--consensus-mode",
        choices=["edge", "pairwise"],
        default="edge",
        help="Use scalable edge consensus over global kNN edges, or pairwise all-pairs consensus for small N.",
    )
    p.add_argument(
        "--consensus-weight-mode",
        choices=["frequency", "frequency-times-similarity"],
        default="frequency-times-similarity",
        help="Weight final consensus edges by stability alone or by stability multiplied by original kNN similarity.",
    )

    p.add_argument("--output", required=True)
    p.add_argument("--output-header", action="store_true", help="Write a header line to the cluster assignment TSV.")
    p.add_argument("--save-emb", default=None, help="Save loaded/processed embedding matrix as compressed NPZ.")
    p.add_argument("--save-metadata-json", default=None, help="Save node metadata/provenance as JSON.")

    p.add_argument("--export-knn-edges", default=None)
    p.add_argument("--export-knn-csr", default=None)
    p.add_argument("--export-consensus-edges", default=None)
    p.add_argument("--export-consensus-csr", default=None)
    p.add_argument("--export-node-names", default=None)

    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--nn-jobs",
        type=int,
        default=None,
        help="Threads for NearestNeighbors (-1=all). Default: 1 when bootstrapping, else -1.",
    )
    p.add_argument("--n-jobs", type=int, default=None, help="[DEPRECATED] Alias for --workers.")

    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-interval", type=int, default=2000)
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.k < 1:
        raise ValueError("--k must be >= 1")
    if args.bootstrap_iterations < 1:
        raise ValueError("--bootstrap-iterations must be >= 1")
    if not (0 < args.sample_fraction <= 1):
        raise ValueError("--sample-fraction must be in (0, 1]")
    if not (0 <= args.consensus_threshold <= 1):
        raise ValueError("--consensus-threshold must be in [0, 1]")
    if args.min_edge_observations < 0:
        raise ValueError("--min-edge-observations must be >= 0")
    if args.min_cluster_size < 1:
        raise ValueError("--min-cluster-size must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")


def maybe_pca(X: np.ndarray, pca_arg: Optional[float], l2: bool, logger: logging.Logger) -> np.ndarray:
    if pca_arg is None or pca_arg <= 0:
        return X

    t0 = time.perf_counter()
    n_samples, n_features = X.shape
    max_comp = max(1, min(n_samples - 1, n_features))

    if pca_arg > 1:
        requested = int(pca_arg)
        n_comp = min(requested, max_comp)
        if n_comp < requested:
            logger.warning(
                "Requested PCA components=%d but max valid components=%d for X.shape=%s; using %d.",
                requested,
                max_comp,
                X.shape,
                n_comp,
            )
        if n_comp < 1:
            logger.warning("Skipping PCA because fewer than 2 samples are available.")
            return X
        logger.info("PCA: reducing to %d components...", n_comp)
        pca = PCA(n_components=n_comp, svd_solver="auto", random_state=0)
        Xp = pca.fit_transform(X)
    else:
        var = float(pca_arg)
        if not (0 < var < 1):
            raise ValueError("When --pca <= 1, it must be a variance fraction in (0, 1).")
        if n_samples < 2:
            logger.warning("Skipping PCA because fewer than 2 samples are available.")
            return X
        logger.info("PCA: retaining %.2f%% variance...", 100.0 * var)
        pca = PCA(n_components=var, svd_solver="auto", random_state=0)
        Xp = pca.fit_transform(X)

    evr = float(np.sum(pca.explained_variance_ratio_))
    logger.info("PCA done in %.2fs. Retained variance: %.2f%%", time.perf_counter() - t0, 100.0 * evr)
    return _l2_normalize_rows(Xp) if l2 else Xp


# ----------------------------
# Export helpers
# ----------------------------

def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _export_edges(path: str, edges: List[Tuple[int, int]], weights: List[float], logger: logging.Logger) -> None:
    t0 = time.perf_counter()
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("src_idx\tdst_idx\tweight\n")
        for (u, v), w in zip(edges, weights):
            f.write(f"{u}\t{v}\t{w:.8f}\n")
    logger.info("Wrote edge list: %s (|E|=%d) in %.2fs", path, len(edges), time.perf_counter() - t0)


def _export_csr(path: str, N: int, edges: List[Tuple[int, int]], weights: List[float], logger: logging.Logger) -> None:
    t0 = time.perf_counter()
    _ensure_parent(path)
    if not edges:
        A = sp.csr_matrix((N, N), dtype=np.float32)
    else:
        rows = [u for (u, v) in edges] + [v for (u, v) in edges]
        cols = [v for (u, v) in edges] + [u for (u, v) in edges]
        data = weights + weights
        A = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    sp.save_npz(path, A)
    logger.info("Wrote CSR: %s (N=%d, |E|=%d) in %.2fs", path, N, len(edges), time.perf_counter() - t0)


def _save_metadata_json(
    path: str,
    names: List[str],
    metas: Dict[str, Dict[str, Any]],
    provenance: Dict[str, List[Tuple[str, str]]],
    logger: logging.Logger,
) -> None:
    _ensure_parent(path)
    payload = []
    for i, name in enumerate(names):
        payload.append(
            {
                "index": i,
                "name": name,
                "meta": _make_jsonable_meta(metas.get(name, {})),
                "provenance": provenance.get(name, []),
            }
        )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logger.info("Wrote metadata JSON: %s", path)


# ----------------------------
# Cluster result helpers
# ----------------------------

def _enforce_min_cluster_size(labels: List[int], min_cluster_size: int) -> List[Any]:
    counts = Counter(labels)
    return ["Unassigned" if counts[lbl] < min_cluster_size else lbl for lbl in labels]


def _write_assignments(
    path: str,
    names: List[str],
    labels: Sequence[Any],
    support: Optional[np.ndarray],
    output_header: bool,
    logger: logging.Logger,
) -> None:
    _ensure_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if support is None:
            if output_header:
                f.write("name\tcluster\n")
            for nm, lbl in zip(names, labels):
                f.write(f"{nm}\t{lbl}\n")
        else:
            if output_header:
                f.write("name\tcluster\tmean_consensus_support\n")
            for nm, lbl, sup in zip(names, labels, support):
                f.write(f"{nm}\t{lbl}\t{sup:.4f}\n")
    logger.info("Wrote %s", path)


def _compute_metrics(X: np.ndarray, labels: Sequence[int], metric: str) -> Tuple[float, float, float]:
    unique = set(labels)
    if len(unique) <= 1 or len(unique) >= len(labels):
        return float("nan"), float("nan"), float("nan")

    try:
        sil = silhouette_score(X, labels, metric=metric)
    except Exception:
        sil = float("nan")
    try:
        db = davies_bouldin_score(X, labels)
    except Exception:
        db = float("nan")
    try:
        ch = calinski_harabasz_score(X, labels)
    except Exception:
        ch = float("nan")
    return float(sil), float(db), float(ch)


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    args = parse_args()
    validate_args(args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("cluster")
    t_all = time.perf_counter()

    if args.n_jobs is not None and args.workers == 4:
        args.workers = args.n_jobs

    if args.nn_jobs is None:
        args._nn_jobs_effective = 1 if args.bootstrap_iterations > 1 else -1
    else:
        args._nn_jobs_effective = int(args.nn_jobs)

    names, X, metas, provenance, fmt = load_vectors(args, logger=logger)
    N, d = X.shape
    logger.info("Embedding matrix: N=%d, d=%d", N, d)

    if args.save_emb:
        t0 = time.perf_counter()
        _ensure_parent(args.save_emb)
        np.savez_compressed(args.save_emb, names=np.array(names, dtype=object), X=X.astype(np.float32))
        logger.info("Saved embeddings to %s (%.2fs)", args.save_emb, time.perf_counter() - t0)

    if args.save_metadata_json:
        _save_metadata_json(args.save_metadata_json, names, metas, provenance, logger=logger)

    good = np.isfinite(X).all(axis=1)
    if not good.all():
        nbad = int((~good).sum())
        logger.warning("Dropping %d rows with NaN/Inf before PCA/kNN", nbad)
        X = X[good]
        names = [n for n, ok in zip(names, good) if ok]
        N = X.shape[0]

    X = maybe_pca(X, args.pca, l2=(not args.no_l2), logger=logger)
    logger.info("Embedding shape for kNN: %s", X.shape)

    if args.export_node_names:
        t0 = time.perf_counter()
        _ensure_parent(args.export_node_names)
        with open(args.export_node_names, "w", encoding="utf-8", newline="\n") as f:
            f.write("index\tname\n")
            for i, nm in enumerate(names):
                f.write(f"{i}\t{nm}\n")
        logger.info("Wrote node names to %s in %.2fs", args.export_node_names, time.perf_counter() - t0)

    if args.bootstrap_iterations > 1:
        run_bootstrap_consensus(args, X, names, logger)
    else:
        run_single_clustering(args, X, names, logger)

    logger.info("Total runtime: %.2fs", time.perf_counter() - t_all)


# ----------------------------
# Single-run clustering
# ----------------------------

def run_single_clustering(args: argparse.Namespace, X: np.ndarray, names: List[str], logger: logging.Logger) -> None:
    N = X.shape[0]
    edges, weights, _ = build_knn_graph(
        X,
        k=args.k,
        mutual=args.mutual_knn,
        metric=args.metric,
        nn_jobs=args._nn_jobs_effective,
        logger=logger,
    )

    if args.export_knn_edges:
        _export_edges(args.export_knn_edges, edges, weights, logger=logger)
    if args.export_knn_csr:
        _export_csr(args.export_knn_csr, N, edges, weights, logger=logger)

    logger.info("Running %s clustering (resolution=%.3f)...", args.algorithm, args.resolution)
    labels = _cluster_graph(
        n_nodes=N,
        edges=edges,
        weights=weights,
        algorithm=args.algorithm,
        resolution=args.resolution,
        seed=args.seed,
    )

    final_labels = _enforce_min_cluster_size(labels, args.min_cluster_size)
    _write_assignments(
        args.output,
        names,
        final_labels,
        support=None,
        output_header=args.output_header,
        logger=logger,
    )

    numeric_labels = [int(lbl) if lbl != "Unassigned" else -1 for lbl in final_labels]
    sil, db, ch = _compute_metrics(X, numeric_labels, metric=args.metric)
    logger.info(
        "Metrics -- Silhouette: %.4f | Davies-Bouldin: %.4f | Calinski-Harabasz: %.4f",
        sil,
        db,
        ch,
    )


# ----------------------------
# Bootstrap consensus clustering
# ----------------------------

def run_bootstrap_consensus(args: argparse.Namespace, X: np.ndarray, names: List[str], logger: logging.Logger) -> None:
    N = X.shape[0]
    if N <= 1:
        labels = list(range(N))
        _write_assignments(args.output, names, labels, support=None, output_header=args.output_header, logger=logger)
        return

    edges_global, weights_global, adj = build_knn_graph(
        X,
        k=args.k,
        mutual=args.mutual_knn,
        metric=args.metric,
        nn_jobs=args._nn_jobs_effective,
        logger=logger,
    )

    if args.export_knn_edges:
        _export_edges(args.export_knn_edges, edges_global, weights_global, logger=logger)
    if args.export_knn_csr:
        _export_csr(args.export_knn_csr, N, edges_global, weights_global, logger=logger)

    M = len(edges_global)
    logger.info("Consensus mode: %s | Global edges: %d", args.consensus_mode, M)

    if args.consensus_mode == "pairwise":
        m = int(round(args.sample_fraction * N))
        est_pairs = (m * (m - 1)) // 2
        if est_pairs > 50_000_000:
            logger.warning(
                "pairwise mode would generate ~%d pairs per bootstrap; switching to edge mode.",
                est_pairs,
            )
            args.consensus_mode = "edge"

    t0 = time.perf_counter()
    logger.info(
        "Bootstrapping %d iterations (sample_fraction=%.2f, workers=%d, nn_jobs=%d)...",
        args.bootstrap_iterations,
        args.sample_fraction,
        args.workers,
        args._nn_jobs_effective,
    )

    coassign_edge = np.zeros(M, dtype=np.int32)
    cosample_edge = np.zeros(M, dtype=np.int32)
    edge_to_idx = {e: i for i, e in enumerate(edges_global)}

    cosample_total: Optional[sp.csr_matrix] = None
    coassign_total: Optional[sp.csr_matrix] = None
    if args.consensus_mode == "pairwise":
        cosample_total = sp.csr_matrix((N, N), dtype=np.int32)
        coassign_total = sp.csr_matrix((N, N), dtype=np.int32)

    log_every = max(1, args.bootstrap_iterations // 10)

    for idx, (idxs, memb_sub) in _iter_bootstrap_results(args, X, N):
        in_sample = np.zeros(N, dtype=bool)
        in_sample[idxs] = True
        memb_full = np.full(N, -1, dtype=np.int32)
        memb_full[idxs] = memb_sub

        if args.consensus_mode == "edge":
            for u0 in idxs:
                u = int(u0)
                mu = int(memb_full[u])
                for v in adj[u]:
                    if u < v and in_sample[v]:
                        ei = edge_to_idx.get((u, v))
                        if ei is None:
                            continue
                        cosample_edge[ei] += 1
                        if int(memb_full[v]) == mu:
                            coassign_edge[ei] += 1

        else:
            assert cosample_total is not None and coassign_total is not None
            row_cos: List[int] = []
            col_cos: List[int] = []
            row_coa: List[int] = []
            col_coa: List[int] = []
            for a in range(len(idxs)):
                ii = int(idxs[a])
                li = int(memb_full[ii])
                for b in range(a + 1, len(idxs)):
                    jj = int(idxs[b])
                    row_cos.append(ii)
                    col_cos.append(jj)
                    if int(memb_full[jj]) == li:
                        row_coa.append(ii)
                        col_coa.append(jj)
            data_cos = np.ones(len(row_cos), dtype=np.int32)
            data_coa = np.ones(len(row_coa), dtype=np.int32)
            cos = sp.coo_matrix((data_cos, (row_cos, col_cos)), shape=(N, N), dtype=np.int32).tocsr()
            coa = sp.coo_matrix((data_coa, (row_coa, col_coa)), shape=(N, N), dtype=np.int32).tocsr()
            cosample_total = cosample_total + cos
            coassign_total = coassign_total + coa

        if idx % log_every == 0 or idx == args.bootstrap_iterations:
            logger.info(
                "  ...completed %d/%d bootstraps (elapsed %.2fs)",
                idx,
                args.bootstrap_iterations,
                time.perf_counter() - t0,
            )

    if args.consensus_mode == "edge":
        cons_edges, cons_weights, freq_edge = _build_edge_consensus(
            edges_global,
            weights_global,
            coassign_edge,
            cosample_edge,
            threshold=args.consensus_threshold,
            min_observations=args.min_edge_observations,
            weight_mode=args.consensus_weight_mode,
            logger=logger,
        )
    else:
        assert cosample_total is not None and coassign_total is not None
        cons_edges, cons_weights = _build_pairwise_consensus(
            cosample_total,
            coassign_total,
            threshold=args.consensus_threshold,
            min_observations=args.min_edge_observations,
            logger=logger,
        )
        freq_edge = None

    if args.export_consensus_edges:
        _export_edges(args.export_consensus_edges, cons_edges, cons_weights, logger=logger)
    if args.export_consensus_csr:
        _export_csr(args.export_consensus_csr, N, cons_edges, cons_weights, logger=logger)

    logger.info("Running %s on consensus graph (resolution=%.3f)...", args.algorithm, args.resolution)
    labels = _cluster_graph(
        n_nodes=N,
        edges=cons_edges,
        weights=cons_weights,
        algorithm=args.algorithm,
        resolution=args.resolution,
        seed=args.seed,
    )

    final_labels = _enforce_min_cluster_size(labels, args.min_cluster_size)
    node_support = _node_support_from_consensus(N, cons_edges, cons_weights, final_labels)

    _write_assignments(
        args.output,
        names,
        final_labels,
        support=node_support,
        output_header=args.output_header,
        logger=logger,
    )

    logger.info("Bootstrap + consensus pipeline finished in %.2fs", time.perf_counter() - t0)


def _build_edge_consensus(
    edges_global: List[Tuple[int, int]],
    weights_global: List[float],
    coassign_edge: np.ndarray,
    cosample_edge: np.ndarray,
    threshold: float,
    min_observations: int,
    weight_mode: str,
    logger: logging.Logger,
) -> Tuple[List[Tuple[int, int]], List[float], np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        freq_edge = np.where(cosample_edge > 0, coassign_edge / cosample_edge, 0.0).astype(np.float64)

    obs_mask = cosample_edge >= int(min_observations)
    freq_mask = freq_edge >= float(threshold)
    cons_mask = obs_mask & freq_mask

    cons_edges: List[Tuple[int, int]] = []
    cons_weights: List[float] = []
    for i, keep in enumerate(cons_mask):
        if not keep:
            continue
        freq = float(freq_edge[i])
        sim = float(weights_global[i])
        if weight_mode == "frequency-times-similarity":
            w = freq * sim
        elif weight_mode == "frequency":
            w = freq
        else:
            raise ValueError(f"Unsupported consensus weight mode: {weight_mode}")
        cons_edges.append(edges_global[i])
        cons_weights.append(w)

    logger.info(
        "Consensus graph (edge mode): |E|=%d at threshold=%.2f, min_edge_observations=%d, weight_mode=%s",
        len(cons_edges),
        threshold,
        min_observations,
        weight_mode,
    )
    if len(edges_global) > 0:
        logger.info(
            "Edge co-sampling summary: median=%s, mean=%.2f, max=%d",
            int(np.median(cosample_edge)),
            float(np.mean(cosample_edge)),
            int(np.max(cosample_edge)),
        )
    return cons_edges, cons_weights, freq_edge


def _build_pairwise_consensus(
    cosample_total: sp.csr_matrix,
    coassign_total: sp.csr_matrix,
    threshold: float,
    min_observations: int,
    logger: logging.Logger,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    logger.info("Computing pairwise consensus frequencies...")
    coo = coassign_total.tocoo()
    cos_csr = cosample_total.tocsr()

    cons_edges: List[Tuple[int, int]] = []
    cons_weights: List[float] = []
    for i, j, co_count in zip(coo.row, coo.col, coo.data):
        if i >= j:
            continue
        cos_count = int(cos_csr[i, j])
        if cos_count < min_observations or cos_count <= 0:
            continue
        freq = float(co_count) / float(cos_count)
        if freq >= threshold:
            cons_edges.append((int(i), int(j)))
            cons_weights.append(freq)

    logger.info(
        "Consensus graph (pairwise mode): |E|=%d at threshold=%.2f, min_edge_observations=%d",
        len(cons_edges),
        threshold,
        min_observations,
    )
    return cons_edges, cons_weights


def _node_support_from_consensus(
    N: int,
    cons_edges: List[Tuple[int, int]],
    cons_weights: List[float],
    final_labels: Sequence[Any],
) -> np.ndarray:
    node_support = np.zeros(N, dtype=np.float32)
    if not cons_edges:
        return node_support

    adj_w: List[List[Tuple[int, float]]] = [[] for _ in range(N)]
    for (u, v), w in zip(cons_edges, cons_weights):
        adj_w[u].append((v, float(w)))
        adj_w[v].append((u, float(w)))

    for i in range(N):
        lbl = final_labels[i]
        if lbl == "Unassigned":
            continue
        vals = [w for j, w in adj_w[i] if final_labels[j] == lbl]
        if vals:
            node_support[i] = float(np.mean(vals))
    return node_support


if __name__ == "__main__":
    main()
