#!/usr/bin/env python3
"""
graph_clustering_from_h5.v3.2.py

Scalable kNN-graph clustering with optional **bootstrapped consensus** on large datasets.

Key changes vs v3.1
-------------------
- **Edge-based consensus (default)**: compute co-assignment frequencies **only over a fixed global kNN edge set**.
  Complexity ~ O(B · N · k). Avoids the O(B · m^2) blow-up and huge NxN sparse matrices.
- Workers now return only **(sample indices, membership)** (small), not big matrices.
- Optional legacy **pairwise** mode kept via `--consensus-mode pairwise` (auto-guarded to small N).
- Same separation of parallelism: `--workers` (processes) vs `--nn-jobs` (NearestNeighbors threads).

When to use which consensus:
- Default `edge` works for N ≳ 10k+ without blowing memory.
- Use `pairwise` only for small N (e.g., N ≤ ~5k) or larger N with small `--sample-fraction` if you need
  true all-pairs conditional frequencies.

"""
from __future__ import annotations

import argparse
import logging
import os
import math
import gc
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import igraph as ig
import leidenalg
import h5py
import multiprocessing as mp
import random

# ----------------------------
# Utilities
# ----------------------------

def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < eps] = eps
    return X / nrm

def _safe_mean(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        raise ValueError("Empty array passed to mean pooling")
    return X.mean(axis=0)

def _pool_item(vec: np.ndarray, w: np.ndarray | None, mode: str) -> np.ndarray:
    if mode == 'mean':
        return _safe_mean(vec)
    if mode == 'sum':
        return vec.sum(axis=0)
    if w is None:
        return _safe_mean(vec)
    w = w.astype(np.float64, copy=False).reshape(-1)
    if mode == 'attn-norm':
        s = float(w.sum())
        if s <= 0:
            w = np.ones_like(w) / max(len(w), 1)
        else:
            w = w / s
    return (vec * w[:, None]).sum(axis=0)

def _safe_key(s: str) -> str:
    return str(s).replace('/', '_')

# ----------------------------
# Data loading & pooling
# ----------------------------

def _iter_items(h5: h5py.File):
    if 'items' not in h5:
        return []
    items = h5['items']
    for key in items:
        yield key, items[key]

def _read_h5_multi(paths: List[str],
                   pool: str = 'attn-norm',
                   id_field: str = 'chunk_id',
                   dedupe: str = 'none',
                   l2: bool = True,
                   log_interval: int = 2000,
                   logger: logging.Logger | None = None) -> Tuple[List[str], np.ndarray, Dict[str, Dict], Dict[str, List[Tuple[str,str]]]]:
    start_t = time.perf_counter()
    per_name = defaultdict(list)
    d_check = None
    total_groups = 0

    for path in paths:
        with h5py.File(path, 'r') as h5:
            keys = list(h5['items'].keys()) if 'items' in h5 else []
            n_keys = len(keys)
            if logger:
                logger.info("Reading %s: %d groups", path, n_keys)
            for i, key in enumerate(keys, 1):
                g = h5['items'][key]
                if 'vec' not in g:
                    continue
                V = g['vec'][...].astype(np.float64, copy=False)
                d = V.shape[1]
                d_check = d if d_check is None else d_check
                if d_check != d:
                    raise ValueError(f"Dimension mismatch across files: saw {d_check} and {d}")
                W = g['w'][...].astype(np.float64, copy=False) if 'w' in g else None
                name = str(g.attrs.get(id_field, key))
                P = float(g.attrs.get('P', np.nan))
                meta = dict(g.attrs.items())
                pooled = _pool_item(V, W, mode=pool).astype(np.float64, copy=False)
                per_name[name].append((pooled, P, meta, path, key))
                total_groups += 1
                if logger and (i % log_interval == 0):
                    logger.info("  ...%d/%d groups processed in %s", i, n_keys, path)

    names: List[str] = []
    vecs: List[np.ndarray] = []
    metas: Dict[str, Dict] = {}
    provenance: Dict[str, List[Tuple[str,str]]] = {}

    dup_count = 0
    for name, lst in per_name.items():
        if len(lst) == 1 or dedupe == 'none':
            pooled, P, meta, path, key = lst[0]
        else:
            dup_count += 1
            if dedupe == 'maxP':
                pooled, P, meta, path, key = max(lst, key=lambda t: (float('-inf') if math.isnan(t[1]) else t[1]))
            elif dedupe == 'mean':
                pooled = np.stack([t[0] for t in lst], axis=0).mean(axis=0)
                _, P, meta, path, key = max(lst, key=lambda t: (float('-inf') if math.isnan(t[1]) else t[1]))
            else:
                pooled, P, meta, path, key = lst[0]
        names.append(name)
        vecs.append(pooled)
        metas[name] = meta
        provenance[name] = [(p, k) for (_, _, _, p, k) in lst]

    X = np.stack(vecs, axis=0).astype(np.float64, copy=False)
    if l2:
        X = _l2_normalize_rows(X)

    if logger:
        elapsed = time.perf_counter() - start_t
        logger.info("Union complete: %d total groups, %d unique nodes, %d duplicates resolved (mode=%s). Took %.2fs",
                    total_groups, len(names), dup_count, dedupe, elapsed)
    return names, X, metas, provenance

# ----------------------------
# kNN graph (global)
# ----------------------------

def build_knn_graph(vectors: np.ndarray, k: int = 30, mutual: bool = False,
                    metric: str = 'cosine', nn_jobs: int = 1, logger: logging.Logger | None = None) -> Tuple[List[Tuple[int,int]], List[float], List[List[int]]]:
    """
    Build an undirected kNN graph on the FULL set. Also returns adjacency list per node.
    """
    t0 = time.perf_counter()
    N = vectors.shape[0]
    if N <= 1:
        return [], [], [[] for _ in range(N)]
    k_eff = max(1, min(k, N - 1))
    if logger:
        logger.info("Building GLOBAL kNN: N=%d, d=%d, k=%d, metric=%s, mutual=%s, nn_jobs=%s", N, vectors.shape[1], k_eff, metric, mutual, nn_jobs)
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=nn_jobs).fit(vectors)
    distances, indices = nbrs.kneighbors(vectors)

    ind = indices[:, 1:]
    dist = distances[:, 1:]

    if metric == 'cosine':
        sims = np.maximum(0.0, 1.0 - dist)
    else:
        sims = 1.0 / (1.0 + dist)

    if not mutual:
        best = {}
        for i in range(N):
            for j_idx in range(k_eff):
                j = int(ind[i, j_idx])
                if i == j:
                    continue
                u, v = (i, j) if i < j else (j, i)
                w = float(sims[i, j_idx])
                if (u, v) not in best or w > best[(u, v)]:
                    best[(u, v)] = w
        edges = list(best.keys())
        weights = [best[e] for e in edges]
    else:
        neighbor_sets = [set(ind[i].tolist()) for i in range(N)]
        best = {}
        for i in range(N):
            for j_idx in range(k_eff):
                j = int(ind[i, j_idx])
                if i == j:
                    continue
                if i in neighbor_sets[j]:
                    u, v = (i, j) if i < j else (j, i)
                    try:
                        jpos = int(np.where(ind[j] == i)[0][0])
                        sji = float(sims[j, jpos])
                        w = (float(sims[i, j_idx]) + sji) / 2.0
                    except Exception:
                        w = float(sims[i, j_idx])
                    if (u, v) not in best or w > best[(u, v)]:
                        best[(u, v)] = w
        edges = list(best.keys())
        weights = [best[e] for e in edges]

    # adjacency
    adj = [[] for _ in range(N)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)

    if logger:
        avg_deg = (2.0 * len(edges)) / max(1, N)
        logger.info("GLOBAL kNN built: |E|=%d, avg_degree=%.2f, elapsed=%.2fs", len(edges), avg_deg, time.perf_counter()-t0)
    return edges, weights, adj

# ----------------------------
# Clustering on a subset (worker side)
# ----------------------------

def _cluster_subset(sub_vectors: np.ndarray, args) -> List[int]:
    # local kNN for the subset
    N = sub_vectors.shape[0]
    k_eff = max(1, min(args.k, N-1))
    nbrs = NearestNeighbors(n_neighbors=k_eff + 1, metric=args.metric, n_jobs=1).fit(sub_vectors)
    dist, ind = nbrs.kneighbors(sub_vectors)
    # similarities
    if args.metric == 'cosine':
        sims = np.maximum(0.0, 1.0 - dist[:, 1:])
    else:
        sims = 1.0 / (1.0 + dist[:, 1:])
    # edges
    edges = []
    weights = []
    for i in range(N):
        for j_idx in range(k_eff):
            j = int(ind[i, j_idx+1])
            u, v = (i, j) if i < j else (j, i)
            edges.append((u, v))
            weights.append(float(sims[i, j_idx]))
    # graph + clustering
    g = ig.Graph(n=N)
    if edges:
        g.add_edges(edges); g.es['weight'] = weights
    if args.algorithm == 'leiden':
        part = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights='weight',
            resolution_parameter=args.resolution,
            seed=args.seed
        )
        return part.membership
    else:
        return g.community_multilevel(weights='weight').membership

# ----------------------------
# Multiprocessing globals
# ----------------------------

_global_vectors = None
_global_args = None
_global_N = None

def _init_pool(vectors: np.ndarray, args, N: int):
    global _global_vectors, _global_args, _global_N
    _global_vectors = vectors
    _global_args = args
    _global_N = N

def _bootstrap_worker(b: int):
    rng = random.Random(_global_args.seed + b)
    m = max(2, int(round(_global_args.sample_fraction * _global_N)))
    m = min(m, _global_N)
    idxs = rng.sample(range(_global_N), m)
    idxs.sort()
    sub_vectors = _global_vectors[idxs]
    memb_sub = _cluster_subset(sub_vectors, _global_args)
    # Return only minimal info
    return (np.array(idxs, dtype=np.int32), np.array(memb_sub, dtype=np.int32))

# ----------------------------
# CLI
# ----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="kNN graph clustering from RdRP catalytic vectors (HDF5 union + weighted pooling).")
    p.add_argument('--input', nargs='+', required=True, help="One or more HDF5 files exported by predictor (or a TSV if --input-format tsv).")
    p.add_argument('--input-format', choices=['auto', 'h5', 'tsv'], default='auto')
    p.add_argument('--id-field', choices=['chunk_id', 'base_id'], default='chunk_id')
    p.add_argument('--dedupe', choices=['none', 'maxP', 'mean'], default='none')
    p.add_argument('--pool', choices=['attn', 'attn-norm', 'mean', 'sum'], default='attn-norm')
    p.add_argument('--no-l2', action='store_true')
    p.add_argument('--pca', type=float, default=None)
    p.add_argument('--metric', choices=['cosine', 'euclidean'], default='cosine')
    p.add_argument('--k', type=int, default=30)
    p.add_argument('--mutual-knn', action='store_true')
    p.add_argument('--algorithm', choices=['leiden', 'louvain'], default='leiden')
    p.add_argument('--resolution', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bootstrap-iterations', type=int, default=1)
    p.add_argument('--sample-fraction', type=float, default=0.8)
    p.add_argument('--consensus-threshold', type=float, default=0.8)
    p.add_argument('--min-cluster-size', type=int, default=2)
    p.add_argument('--output', required=True)
    p.add_argument('--save-emb', default=None)
    # Pooled vector writing
    p.add_argument('--write-pooled-inplace', action='store_true')
    p.add_argument('--pooled-dset-name', default='vec_pooled')
    p.add_argument('--overwrite-pooled', action='store_true')
    p.add_argument('--pooled-h5-out', default=None)
    # Graph export
    p.add_argument('--export-knn-edges', default=None)
    p.add_argument('--export-knn-csr', default=None)
    p.add_argument('--export-consensus-edges', default=None)
    p.add_argument('--export-consensus-csr', default=None)
    p.add_argument('--export-node-names', default=None)
    # Consensus mode
    p.add_argument('--consensus-mode', choices=['edge','pairwise'], default='edge',
                   help="Use 'edge' consensus over a fixed global kNN (scalable) or 'pairwise' (all pairs; small N only).")
    # Parallelism controls
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--nn-jobs', type=int, default=None, help="Threads for NearestNeighbors (-1=all). Default: 1 when bootstrapping, else -1.")
    p.add_argument('--n-jobs', type=int, default=None, help="[DEPRECATED] Alias for --workers.")
    # Logging
    p.add_argument('--log-level', default='INFO', choices=['DEBUG','INFO','WARNING','ERROR'])
    p.add_argument('--log-interval', type=int, default=2000)
    return p.parse_args()

def maybe_pca(X: np.ndarray, pca_arg: float | None, l2: bool, logger) -> np.ndarray:
    if pca_arg is None or pca_arg <= 0:
        return X
    t0 = time.perf_counter()
    if pca_arg > 1:
        n_comp = int(pca_arg)
        logger.info("PCA: reducing to %d components...", n_comp)
        pca = PCA(n_components=n_comp, svd_solver='auto', random_state=0)
        Xp = pca.fit_transform(X)
        evr = float(np.sum(pca.explained_variance_ratio_))
        logger.info("PCA done in %.2fs. Retained variance: %.2f%%", time.perf_counter()-t0, 100.0*evr)
    else:
        var = float(pca_arg)
        logger.info("PCA: retaining %.2f%% variance...", 100.0 * var)
        pca = PCA(n_components=var, svd_solver='auto', random_state=0)
        Xp = pca.fit_transform(X)
        evr = float(np.sum(pca.explained_variance_ratio_))
        logger.info("PCA done in %.2fs. Retained variance: %.2f%%", time.perf_counter()-t0, 100.0*evr)
    if not l2:
        return Xp
    return _l2_normalize_rows(Xp)

# ----------------------------
# Main
# ----------------------------

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("cluster")
    t_all = time.perf_counter()

    if args.n_jobs is not None and args.workers == 4:
        args.workers = args.n_jobs

    if args.nn_jobs is None:
        args._nn_jobs_effective = 1 if args.bootstrap_iterations > 1 else -1
    else:
        args._nn_jobs_effective = int(args.nn_jobs)

    # Determine format
    fmt = args.input_format
    if fmt == 'auto':
        exts = {os.path.splitext(p.lower())[1] for p in args.input}
        if any(e in ('.h5', '.hdf5') for e in exts):
            fmt = 'h5'
        else:
            fmt = 'tsv'
    logger.info("Format: %s", fmt.upper())

    # Load / pool
    if fmt == 'tsv':
        if len(args.input) != 1:
            raise ValueError("TSV mode supports a single input file. Convert to HDF5 first for multi-file.")
        import pandas as pd
        t0 = time.perf_counter()
        df = pd.read_csv(args.input[0], sep='\t', header=None)
        names = df.iloc[:, 0].astype(str).tolist()
        X = df.iloc[:, 1:].to_numpy(dtype=np.float64)
        if not args.no_l2:
            X = _l2_normalize_rows(X)
        metas = {n: {} for n in names}
        provenance = {n: [(args.input[0], 'tsv_row')] for n in names}
        logger.info("Loaded TSV: N=%d, d=%d in %.2fs", X.shape[0], X.shape[1], time.perf_counter()-t0)
    else:
        names, X, metas, provenance = _read_h5_multi(args.input, pool=args.pool, id_field=args.id_field,
                                                     dedupe=args.dedupe, l2=(not args.no_l2),
                                                     log_interval=args.log_interval, logger=logger)

    N, d = X.shape
    logger.info("Pooled embedding matrix: N=%d, d=%d", N, d)

    # Optional: write pooled vectors back to HDF5s
    if fmt == 'h5' and args.write_pooled_inplace:
        # (omitted here for brevity; same as v3.1) -- can be added back if needed
        logger.warning("write-pooled-inplace is temporarily disabled in v3.2 snippet to focus on consensus scaling.")

    if fmt == 'h5' and args.pooled_h5_out:
        logger.warning("pooled-h5-out is temporarily disabled in v3.2 snippet to focus on consensus scaling.")

    # Save pooled embeddings if requested
    if args.save_emb:
        t0 = time.perf_counter()
        np.savez_compressed(args.save_emb, names=np.array(names, dtype=object), X=X.astype(np.float32))
        logger.info("Saved pooled embeddings to %s (%.2fs)", args.save_emb, time.perf_counter()-t0)

    # Drop NaNs/Infs
    good = np.isfinite(X).all(axis=1)
    if not good.all():
        nbad = int((~good).sum())
        logger.warning("Dropping %d rows with NaN/Inf before PCA/kNN", nbad)
        X = X[good]; names = [n for n, ok in zip(names, good) if ok]
        N = X.shape[0]

    # PCA
    X = maybe_pca(X, args.pca, l2=(not args.no_l2), logger=logger)
    logger.info("Embedding shape for kNN: %s", X.shape)

    # Export node names if requested
    if args.export_node_names:
        t0 = time.perf_counter()
        os.makedirs(os.path.dirname(args.export_node_names) or '.', exist_ok=True)
        with open(args.export_node_names, 'w', encoding='utf-8') as f:
            f.write("index\tname\n")
            for i, nm in enumerate(names):
                f.write(f"{i}\t{nm}\n")
        logger.info("Wrote node names to %s in %.2fs", args.export_node_names, time.perf_counter()-t0)

    # Bootstrapped consensus
    if args.bootstrap_iterations > 1:
        # Precompute GLOBAL kNN once
        edges_global, weights_global, adj = build_knn_graph(X, k=args.k, mutual=args.mutual_knn,
                                                            metric=args.metric, nn_jobs=args._nn_jobs_effective, logger=logger)
        M = len(edges_global)
        logger.info("Consensus mode: %s | Global edges: %d", args.consensus_mode, M)

        if args.consensus_mode == 'pairwise':
            # Guardrail against infeasible settings
            m = int(round(args.sample_fraction * N))
            est_pairs = (m * (m - 1)) // 2
            if est_pairs > 50_000_000:
                logger.warning("pairwise mode would generate ~%d pairs per bootstrap; switching to 'edge' mode.", est_pairs)
                args.consensus_mode = 'edge'

        t0 = time.perf_counter()
        logger.info("Bootstrapping %d iterations (sample_fraction=%.2f, workers=%d, nn_jobs=%d)...",
                    args.bootstrap_iterations, args.sample_fraction, args.workers, args._nn_jobs_effective)

        # edge-based accumulators
        coassign_edge = np.zeros(M, dtype=np.int32)
        cosample_edge = np.zeros(M, dtype=np.int32)

        # pairwise accumulators (only if used; keep as CSR to avoid OOM)
        cosample_total = None
        coassign_total = None
        if args.consensus_mode == 'pairwise':
            cosample_total = sp.csr_matrix((N, N), dtype=np.int32)
            coassign_total = sp.csr_matrix((N, N), dtype=np.int32)

        
        # Build fast edge index map once
        edge_to_idx = {e:i for i,e in enumerate(edges_global)}

# Rebuild with optimized edge index map
        edge_to_idx = {e:i for i,e in enumerate(edges_global)}

        with mp.Pool(args.workers, initializer=_init_pool, initargs=(X, args, N)) as pool:
            for idx, (idxs, memb_sub) in enumerate(pool.imap(_bootstrap_worker, range(args.bootstrap_iterations), chunksize=1), 1):
                in_sample = np.zeros(N, dtype=bool); in_sample[idxs] = True
                memb_full = np.full(N, -1, dtype=np.int32); memb_full[idxs] = memb_sub

                if args.consensus_mode == 'edge':
                    for u in idxs:
                        mu = memb_full[u]
                        for v in adj[u]:
                            if u < v and in_sample[v]:
                                ei = edge_to_idx.get((u, v), None)
                                if ei is None:  # should not happen
                                    continue
                                cosample_edge[ei] += 1
                                if memb_full[v] == mu:
                                    coassign_edge[ei] += 1

                else:  # pairwise (fallback; small N only)
                    # Build upper-tri co-sample and co-assign for this sample (still O(m^2))
                    row_cos, col_cos = [], []
                    row_coa, col_coa = [], []
                    for a in range(len(idxs)):
                        ii = int(idxs[a]); li = memb_full[ii]
                        for b in range(a+1, len(idxs)):
                            jj = int(idxs[b])
                            row_cos.append(ii); col_cos.append(jj)
                            if memb_full[jj] == li:
                                row_coa.append(ii); col_coa.append(jj)
                    data1 = np.ones(len(row_cos), dtype=np.int32)
                    cos = sp.coo_matrix((data1, (row_cos, col_cos)), shape=(N, N), dtype=np.int32).tocsr()
                    data2 = np.ones(len(row_coa), dtype=np.int32)
                    coa = sp.coo_matrix((data2, (row_coa, col_coa)), shape=(N, N), dtype=np.int32).tocsr()
                    cosample_total = cosample_total + cos
                    coassign_total = coassign_total + coa
                    del row_cos, col_cos, row_coa, col_coa, data1, data2, cos, coa

                if idx % max(1, args.bootstrap_iterations // 10) == 0:
                    logger.info("  ...completed %d/%d bootstraps (elapsed %.2fs)", idx, args.bootstrap_iterations, time.perf_counter()-t0)

        # Build consensus graph
        if args.consensus_mode == 'edge':
            # frequency per edge
            with np.errstate(divide='ignore', invalid='ignore'):
                freq_edge = np.where(cosample_edge > 0, coassign_edge / cosample_edge, 0.0).astype(np.float64)
            # threshold
            cons_mask = freq_edge >= float(args.consensus_threshold)
            cons_edges = [e for e, m in zip(edges_global, cons_mask) if m]
            cons_weights = [float(f) for f, m in zip(freq_edge, cons_mask) if m]
            logger.info("Consensus graph (edge mode): |E|=%d at threshold=%.2f", len(cons_edges), args.consensus_threshold)

        else:
            logger.info("Computing consensus frequency matrix (pairwise)...")
            freq = coassign_total.copy().astype(np.float64)
            cosample_nonzero = cosample_total.copy()
            nz = cosample_nonzero.data != 0
            cosample_nonzero.data[nz] = 1.0 / cosample_nonzero.data[nz]
            cosample_nonzero.data[~nz] = 0.0
            freq = freq.multiply(cosample_nonzero)
            data = freq.data
            data[~np.isfinite(data)] = 0.0
            freq.data = data
            freq_coo = freq.tocoo()
            cons_edges = [(i, j) for i, j, f in zip(freq_coo.row, freq_coo.col, freq_coo.data) if (i < j and f >= args.consensus_threshold)]
            cons_weights = [float(f) for i, j, f in zip(freq_coo.row, freq_coo.col, freq_coo.data) if (i < j and f >= args.consensus_threshold)]
            logger.info("Consensus graph (pairwise): |E|=%d at threshold=%.2f", len(cons_edges), args.consensus_threshold)

        # Export consensus graph if requested
        def _export_edges(path: str, edges: List[Tuple[int,int]], weights: List[float]):
            t0 = time.perf_counter()
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write("src_idx\tdst_idx\tweight\n")
                for (u, v), w in zip(edges, weights):
                    f.write(f"{u}\t{v}\t{w:.6f}\n")
            logger.info("Wrote edge list: %s (|E|=%d) in %.2fs", path, len(edges), time.perf_counter()-t0)

        def _export_csr(path: str, N: int, edges: List[Tuple[int,int]], weights: List[float]):
            t0 = time.perf_counter()
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            if not edges:
                A = sp.csr_matrix((N, N), dtype=np.float32)
                sp.save_npz(path, A)
                logger.info("Wrote CSR: %s (empty graph) in %.2fs", path, time.perf_counter()-t0)
                return
            rows = [u for (u, v) in edges] + [v for (u, v) in edges]
            cols = [v for (u, v) in edges] + [u for (u, v) in edges]
            data = weights + weights
            A = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
            sp.save_npz(path, A)
            logger.info("Wrote CSR: %s (N=%d, |E|=%d) in %.2fs", path, N, len(edges), time.perf_counter()-t0)

        if args.export_consensus_edges:
            _export_edges(args.export_consensus_edges, cons_edges, cons_weights)
        if args.export_consensus_csr:
            _export_csr(args.export_consensus_csr, N, cons_edges, cons_weights)

        # Leiden on consensus graph
        gcon = ig.Graph(n=N)
        if cons_edges:
            gcon.add_edges(cons_edges); gcon.es['weight'] = cons_weights
            logger.info("Running Leiden on consensus graph...")
            part = leidenalg.find_partition(
                gcon,
                leidenalg.RBConfigurationVertexPartition,
                resolution_parameter=args.resolution,
                seed=args.seed
            )
            clabels = part.membership
        else:
            logger.warning("Consensus graph is empty; assigning unique labels.")
            clabels = list(range(N))

        # Enforce min cluster size
        counts = Counter(clabels)
        final_labels = ["Unassigned" if counts[lbl] < args.min_cluster_size else lbl for lbl in clabels]

        # Edge-based per-node support: average freq of incident consensus edges to same-label neighbors
        node_support = np.zeros(N, dtype=np.float32)
        if args.consensus_mode == 'edge' and cons_edges:
            # Build adjacency for consensus graph with weights
            adj_w = [[] for _ in range(N)]
            for (u, v), w in zip(cons_edges, cons_weights):
                adj_w[u].append((v, w)); adj_w[v].append((u, w))
            for i in range(N):
                lbl = final_labels[i]
                if lbl == "Unassigned":
                    continue
                nbrs = [(j, w) for (j, w) in adj_w[i] if final_labels[j] == lbl]
                if nbrs:
                    node_support[i] = float(np.mean([w for (_, w) in nbrs]))
                else:
                    node_support[i] = 0.0

        # Write cluster assignments
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            if args.consensus_mode == 'edge':
                for nm, lbl, sup in zip(names, final_labels, node_support):
                    f.write(f"{nm}\t{lbl}\t{sup:.4f}\n")
            else:
                for nm, lbl in zip(names, final_labels):
                    f.write(f"{nm}\t{lbl}\n")
        logger.info("Wrote %s", args.output)
        logger.info("Bootstrap + consensus pipeline finished in %.2fs", time.perf_counter()-t0)

    else:
        # Single run clustering without bootstraps
        # Build global kNN and run Leiden once
        edges, weights, _ = build_knn_graph(X, k=args.k, mutual=args.mutual_knn,
                                            metric=args.metric, nn_jobs=args._nn_jobs_effective, logger=logger)
        g = ig.Graph(n=N)
        if edges:
            g.add_edges(edges); g.es['weight'] = weights
        logger.info("Running %s clustering (resolution=%.3f)...", args.algorithm, args.resolution)
        if args.algorithm == 'leiden':
            part = leidenalg.find_partition(
                g,
                leidenalg.RBConfigurationVertexPartition,
                weights='weight',
                resolution_parameter=args.resolution,
                seed=args.seed
            )
            labels = part.membership
        else:
            labels = g.community_multilevel(weights='weight').membership

        # Metrics
        try:
            sil = silhouette_score(X, labels, metric=args.metric) if len(set(labels)) > 1 else float('nan')
        except Exception:
            sil = float('nan')
        try:
            db = davies_bouldin_score(X, labels)
        except Exception:
            db = float('nan')
        try:
            ch = calinski_harabasz_score(X, labels)
        except Exception:
            ch = float('nan')

        # Exports
        if args.export_knn_edges:
            os.makedirs(os.path.dirname(args.export_knn_edges) or '.', exist_ok=True)
            with open(args.export_knn_edges, 'w', encoding='utf-8') as f:
                f.write("src_idx\tdst_idx\tweight\n")
                for (u, v), w in zip(edges, weights):
                    f.write(f"{u}\t{v}\t{w:.6f}\n")
        if args.export_knn_csr:
            rows = [u for (u, v) in edges] + [v for (u, v) in edges]
            cols = [v for (u, v) in edges] + [u for (u, v) in edges]
            data = weights + weights
            A = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
            sp.save_npz(args.export_knn_csr, A)

        with open(args.output, 'w', encoding='utf-8') as f:
            for nm, lbl in zip(names, labels):
                f.write(f"{nm}\t{lbl}\n")
        logger.info("Wrote %s", args.output)
        logger.info("Metrics -- Silhouette: %.4f | Davies-Bouldin: %.4f | Calinski-Harabasz: %.4f",
                    sil, db, ch)

    logger.info("Total runtime: %.2fs", time.perf_counter()-t_all)


if __name__ == '__main__':
    main()

