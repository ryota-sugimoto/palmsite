#!/usr/bin/env python3
"""
plot_pooled_json_embedding.py

Reduce PalmSite pooled JSON vectors to 2-D with t-SNE and/or UMAP, join taxonomy
and graph-clustering annotations, and write annotated coordinate tables plus plots.

Input expected from PalmSite pooled JSON, for example:
  pools.backbone.span_attn_norm
  pools.input.span_attn_norm
  pools.backbone.span_mean
  pools.input.span_mean

The script is designed for the pooled_panels.json produced with --pool-include-input.
It also handles taxonomy TSVs like query_to_lca_ranked_lineage.tsv and cluster TSVs
with either headerless rows:
  sequence_id<TAB>cluster<TAB>support
or headered rows from graph_clustering_pooled_json.py with --output-header.

Arbitrary/custom labels can be supplied as a two-column TSV:
  sequence_id<TAB>label
Use --label-tsv labels.tsv and color with --color-by custom_label.

Examples
--------
Main PalmSite backbone panel:
  python plot_pooled_json_embedding.py \
    --input-json pooled_panels.json \
    --json-panel backbone.span_attn_norm \
    --json-best-only \
    --id-field base_id \
    --taxonomy-tsv query_to_lca_ranked_lineage.tsv \
    --cluster-tsv span_attn_norm.graph_clustering.tsv \
    --color-by cluster_label family order \
    --reducers tsne umap \
    --output-prefix backbone_span_attn

Matched raw ESM-C control:
  python plot_pooled_json_embedding.py \
    --input-json pooled_panels.json \
    --json-panel input.span_attn_norm \
    --json-best-only \
    --id-field base_id \
    --taxonomy-tsv query_to_lca_ranked_lineage.tsv \
    --cluster-tsv span_attn_norm.graph_clustering.tsv \
    --color-by cluster_label family order \
    --reducers tsne umap \
    --output-prefix input_span_attn

PalmSite-specific labeling on backbone embeddings:
  python plot_pooled_json_embedding.py \
    --input-json pooled_panels.json \
    --json-panel backbone.span_mean \
    --json-best-only \
    --id-field base_id \
    --palm-annot-tsv palm_annot.tsv \
    --hmm-detection-source palm_annot_rdrp \
    --palm-annot-rdrp-threshold 50 \
    --palmsite-positive-threshold 0.5 \
    --color-by palmsite_logit palmannot_rdrp_score hmm_detection_status catalytic_center_order \
    --reducers tsne umap \
    --output-prefix backbone_span_mean

Arbitrary label coloring:
  python plot_pooled_json_embedding.py \
    --input-json pooled_panels.json \
    --json-panel backbone.span_mean \
    --json-best-only \
    --id-field base_id \
    --label-tsv my_labels.tsv \
    --color-by custom_label palmsite_logit \
    --reducers tsne umap \
    --output-prefix backbone_labeled
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


TAXONOMY_RANKS = [
    "acellular_root",
    "realm",
    "kingdom",
    "phylum",
    "class",
    "order",
    "suborder",
    "family",
    "subfamily",
    "genus",
    "subgenus",
    "species",
]

COLOR_BY_ALIASES = {
    "cluster": "cluster_label",
    "probability": "palmsite_probability",
    "palm_probability": "palmsite_probability",
    "palmsite_probability": "palmsite_probability",
    "palmsite_prob": "palmsite_probability",
    "p": "palmsite_probability",
    "logit": "palmsite_logit",
    "logits": "palmsite_logit",
    "palm_logit": "palmsite_logit",
    "palmsite_logit": "palmsite_logit",
    "palmsite_logits": "palmsite_logit",
    "rdrp": "palmannot_rdrp_score",
    "rdrp_score": "palmannot_rdrp_score",
    "palmannot_rdrp": "palmannot_rdrp_score",
    "palmannot_rdrp_score": "palmannot_rdrp_score",
    "hmm": "hmm_detection_status",
    "hmm_status": "hmm_detection_status",
    "hmm_detection": "hmm_detection_status",
    "hmm_detection_status": "hmm_detection_status",
    "hmm_detected": "hmm_detected_label",
    "label": "custom_label",
    "labels": "custom_label",
    "custom": "custom_label",
    "custom_label": "custom_label",
    "arbitrary_label": "custom_label",
    "manual_label": "custom_label",
    "motif_order": "catalytic_center_order",
    "center_order": "catalytic_center_order",
    "catalytic_order": "catalytic_center_order",
    "catalytic_center_order": "catalytic_center_order",
}

PREFERRED_CATEGORY_ORDER = {
    "hmm_detection_status": [
        "HMM-detected RdRP",
        "PalmSite-positive / HMM-missed",
        "Other",
        "HMM not provided",
    ],
    "hmm_detected_label": [
        "HMM-detected",
        "HMM-missed",
        "HMM not provided",
    ],
    "catalytic_center_order": [
        "ABC",
        "CAB",
        "ACB",
        "BAC",
        "BCA",
        "CBA",
        "Partial",
        "Unresolved",
        "PalmAnnot not provided",
    ],
}

# These columns are categorical even if their values look numeric, e.g. labels 0/1/2.
FORCE_CATEGORICAL_COLUMNS = {
    "cluster_label",
    "custom_label",
    "hmm_detection_status",
    "hmm_detected_label",
    "catalytic_center_order",
}


# -----------------------------
# General utilities
# -----------------------------

def setup_logger(level: str) -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger("plot_pooled_json_embedding")


def ensure_parent(path: str | Path) -> None:
    p = Path(path)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)


def parse_list_arg(values: Optional[Sequence[str]], default: List[str]) -> List[str]:
    if not values:
        return default
    out: List[str] = []
    for v in values:
        for part in str(v).split(","):
            part = part.strip()
            if part:
                out.append(part)
    return out or default


def resolve_color_by_alias(value: str) -> str:
    key = str(value).strip()
    return COLOR_BY_ALIASES.get(key.lower(), key)


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < eps] = eps
    return X / nrm


def normalize_join_id(value: Any) -> str:
    """Normalize IDs for joining JSON base/chunk IDs, taxonomy qseqid, and cluster IDs.

    Handles examples such as:
      LUCAPROT_000000000001:305-495
      LUCAPROT_000000000001|chunk_0001_of_0001|aa_000000_001118
      LUCAPROT_000000000001
    """
    s = str(value).strip()
    if not s:
        return s

    # If an ID has whitespace-separated comments, keep the first token.
    s = s.split()[0]

    # PalmSite chunk ID: base|chunk_...|aa_...
    if "|chunk_" in s:
        s = s.split("|chunk_", 1)[0]

    # Taxonomy qseqid often has a coordinate suffix: base:start-end
    # Only remove the suffix when it looks coordinate-like.
    m = re.match(r"^(.+):(\d+)-(\d+)$", s)
    if m:
        s = m.group(1)

    return s


def safe_string_series(s: pd.Series, missing: str = "Unclassified") -> pd.Series:
    out = s.astype("string").fillna(missing).astype(str)
    out = out.replace({"": missing, "nan": missing, "None": missing, "NA": missing, "NaN": missing})
    return out


def infer_numeric(series: pd.Series) -> Tuple[bool, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    ok = numeric.notna().sum()
    return ok >= max(3, int(0.8 * len(series))), numeric


# -----------------------------
# PalmSite pooled JSON loading
# -----------------------------

def get_panel_vector(record: Dict[str, Any], panel: str) -> Optional[List[float]]:
    """Extract vector from a dotted panel path.

    Accepted forms:
      backbone.span_attn_norm
      input.span_attn_norm
      pools.backbone.span_attn_norm
    """
    parts = [p for p in panel.split(".") if p]
    if parts and parts[0] == "pools":
        parts = parts[1:]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid --json-panel '{panel}'. Expected e.g. 'backbone.span_attn_norm' "
            "or 'input.span_mean'."
        )
    source, pool_name = parts
    pools = record.get("pools", {})
    source_dict = pools.get(source, {})
    vector = source_dict.get(pool_name)
    return vector


def iter_pooled_json_items(
    json_path: str,
    load_mode: str,
    logger: logging.Logger,
) -> Iterable[Tuple[str, Any]]:
    """Yield top-level key/value pairs from a pooled JSON file.

    normal mode uses json.load and is fast for small files, but requires enough RAM
    to hold the entire JSON object. stream mode uses ijson and keeps only one
    top-level record in memory at a time. For a very large pooled_panels.json,
    install ijson and use --json-load-mode stream.
    """
    mode = load_mode
    if mode == "auto":
        try:
            size_gb = os.path.getsize(json_path) / (1024 ** 3)
        except OSError:
            size_gb = 0.0
        mode = "stream" if size_gb >= 5.0 else "normal"
        logger.info("JSON load mode auto selected %s for file size %.2f GiB", mode, size_gb)

    if mode == "normal":
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        yield from data.items()
        return

    if mode != "stream":
        raise ValueError(f"Unknown JSON load mode: {load_mode}")

    try:
        import ijson  # type: ignore
    except ImportError as e:
        raise ImportError(
            "--json-load-mode stream requires the ijson package. Install it with: "
            "python -m pip install ijson  # or conda install -c conda-forge ijson"
        ) from e

    with open(json_path, "rb") as f:
        yield from ijson.kvitems(f, "")


def load_pooled_json(
    json_path: str,
    panel: str,
    id_field: str,
    best_only: bool,
    dedupe: str,
    l2: bool,
    logger: logging.Logger,
    load_mode: str = "auto",
    max_random_points: int = 0,
    always_keep_json_positive: bool = False,
    json_positive_threshold: float = 0.5,
    random_state: int = 42,
    progress_every: int = 100000,
    always_keep_ids: Optional[set[str]] = None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    t0 = pd.Timestamp.now()

    rows: List[Dict[str, Any]] = []
    vecs: List[np.ndarray] = []
    priority_rows: List[Dict[str, Any]] = []
    priority_vecs: List[np.ndarray] = []
    bg_rows: List[Dict[str, Any]] = []
    bg_vecs: List[np.ndarray] = []
    rng = np.random.default_rng(int(random_state))
    skipped_meta = 0
    skipped_not_best = 0
    skipped_no_vector = 0
    n_records = 0
    n_eligible = 0
    n_bg_seen = 0
    if always_keep_ids is None:
        always_keep_ids = set()

    for key, record in iter_pooled_json_items(json_path, load_mode=load_mode, logger=logger):
        n_records += 1
        if progress_every and n_records % int(progress_every) == 0:
            logger.info(
                "Parsed %d JSON records; eligible=%d, kept_priority=%d, kept_random=%d",
                n_records, n_eligible, len(priority_rows), len(bg_rows),
            )
        if key == "_meta":
            skipped_meta += 1
            continue
        if not isinstance(record, dict):
            continue
        if best_only and not bool(record.get("is_best_base_chunk", False)):
            skipped_not_best += 1
            continue
        vector = get_panel_vector(record, panel)
        if vector is None:
            skipped_no_vector += 1
            continue

        vec = np.asarray(vector, dtype=np.float64)
        if vec.ndim != 1 or vec.size == 0:
            skipped_no_vector += 1
            continue

        chunk_id = str(record.get("chunk_id", key))
        base_id = str(record.get("base_id", normalize_join_id(chunk_id)))
        if id_field == "base_id":
            node_id = base_id
        elif id_field == "chunk_id":
            node_id = chunk_id
        else:
            node_id = str(record.get(id_field, base_id))

        pool_meta = record.get("pool_meta", {}) if isinstance(record.get("pool_meta", {}), dict) else {}
        row: Dict[str, Any] = {
            "node_id": node_id,
            "join_id": normalize_join_id(base_id if id_field == "base_id" else node_id),
            "base_id": base_id,
            "chunk_id": chunk_id,
            "json_key": key,
            "json_panel": panel,
            "is_best_base_chunk": bool(record.get("is_best_base_chunk", False)),
            "L": record.get("L", np.nan),
            "orig_start": record.get("orig_start", np.nan),
            "orig_len": record.get("orig_len", np.nan),
            "P": record.get("P", np.nan),
            "logit": record.get("logit", np.nan),
            "S_idx": record.get("S_idx", np.nan),
            "E_idx": record.get("E_idx", np.nan),
            "S_norm": record.get("S_norm", np.nan),
            "E_norm": record.get("E_norm", np.nan),
            "mu": record.get("mu", np.nan),
            "sigma": record.get("sigma", np.nan),
            "mu_attn": record.get("mu_attn", np.nan),
            "sigma_attn": record.get("sigma_attn", np.nan),
            "span_len": pool_meta.get("span_len", np.nan),
            "top_k_used": pool_meta.get("top_k_used", np.nan),
        }
        n_eligible += 1
        p_value = pd.to_numeric(pd.Series([row.get("P", np.nan)]), errors="coerce").iloc[0]
        is_priority = bool(
            (always_keep_json_positive and pd.notna(p_value) and float(p_value) >= float(json_positive_threshold))
            or (row["join_id"] in always_keep_ids)
            or (row["base_id"] in always_keep_ids)
            or (row["chunk_id"] in always_keep_ids)
        )

        if max_random_points and max_random_points > 0:
            if is_priority:
                priority_rows.append(row)
                priority_vecs.append(vec)
            else:
                n_bg_seen += 1
                if len(bg_rows) < int(max_random_points):
                    bg_rows.append(row)
                    bg_vecs.append(vec)
                else:
                    j = int(rng.integers(0, n_bg_seen))
                    if j < int(max_random_points):
                        bg_rows[j] = row
                        bg_vecs[j] = vec
        else:
            rows.append(row)
            vecs.append(vec)

    if max_random_points and max_random_points > 0:
        rows = priority_rows + bg_rows
        vecs = priority_vecs + bg_vecs
        logger.info(
            "Streaming sample result: priority_kept=%d, random_background_kept=%d, background_seen=%d",
            len(priority_rows), len(bg_rows), n_bg_seen,
        )

    if not rows:
        raise ValueError(
            f"No vectors were loaded from {json_path} for panel '{panel}'. "
            f"Skipped no-vector entries: {skipped_no_vector}; skipped non-best: {skipped_not_best}."
        )

    dims = {v.size for v in vecs}
    if len(dims) != 1:
        raise ValueError(f"Loaded vectors have inconsistent dimensions: {sorted(dims)}")

    df = pd.DataFrame(rows)
    X = np.vstack(vecs).astype(np.float64, copy=False)

    # Dedupe after loading, if requested.
    if dedupe != "none":
        before = len(df)
        tmp = df.copy()
        if dedupe == "first":
            keep_idx = tmp.drop_duplicates("node_id", keep="first").index.to_numpy()
        elif dedupe == "maxP":
            tmp["_P_sort"] = pd.to_numeric(tmp["P"], errors="coerce").fillna(-np.inf)
            keep_idx = tmp.sort_values("_P_sort", ascending=False).drop_duplicates("node_id", keep="first").index.to_numpy()
        elif dedupe == "mean":
            # Mean vectors per node_id and keep metadata from the max-P representative.
            tmp["_P_sort"] = pd.to_numeric(tmp["P"], errors="coerce").fillna(-np.inf)
            reps = tmp.sort_values("_P_sort", ascending=False).drop_duplicates("node_id", keep="first")
            new_rows: List[Dict[str, Any]] = []
            new_vecs: List[np.ndarray] = []
            for node_id, sub in tmp.groupby("node_id", sort=False):
                idx = sub.index.to_numpy()
                rep = reps[reps["node_id"] == node_id].iloc[0].to_dict()
                rep["n_vectors_merged"] = int(len(idx))
                new_rows.append(rep)
                new_vecs.append(X[idx].mean(axis=0))
            df = pd.DataFrame(new_rows)
            X = np.vstack(new_vecs).astype(np.float64, copy=False)
            keep_idx = None
        else:
            raise ValueError(f"Unknown dedupe mode: {dedupe}")
        if dedupe in {"first", "maxP"}:
            df = df.loc[keep_idx].reset_index(drop=True)
            X = X[keep_idx]
        logger.info("Dedupe mode %s: %d -> %d vectors", dedupe, before, len(df))
    else:
        # Make node_id unique if duplicates are present, while preserving join_id/base_id.
        duplicated = df["node_id"].duplicated(keep=False)
        if duplicated.any():
            counts: Dict[str, int] = {}
            unique_ids: List[str] = []
            for node_id in df["node_id"].astype(str):
                counts[node_id] = counts.get(node_id, 0) + 1
                if counts[node_id] == 1:
                    unique_ids.append(node_id)
                else:
                    unique_ids.append(f"{node_id}__dup{counts[node_id]}")
            df["node_id_original"] = df["node_id"]
            df["node_id"] = unique_ids
            logger.warning(
                "Detected duplicated node_id values. Kept all vectors because --dedupe none, "
                "and made node_id unique with __dup suffixes."
            )

    if l2:
        X = l2_normalize_rows(X)

    bad = ~np.isfinite(X).all(axis=1)
    if bad.any():
        logger.warning("Dropping %d vectors with NaN/Inf", int(bad.sum()))
        keep = ~bad
        df = df.loc[keep].reset_index(drop=True)
        X = X[keep]

    elapsed = (pd.Timestamp.now() - t0).total_seconds()
    logger.info(
        "Loaded JSON panel %s: N=%d, d=%d, best_only=%s, records_seen=%d, skipped_not_best=%d, skipped_no_vector=%d, elapsed=%.2fs",
        panel,
        X.shape[0],
        X.shape[1],
        best_only,
        n_records,
        skipped_not_best,
        skipped_no_vector,
        elapsed,
    )
    return df.reset_index(drop=True), X


# -----------------------------
# Annotation loading
# -----------------------------

def load_taxonomy_tsv(
    path: Optional[str],
    id_column: str,
    duplicate_policy: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["join_id"])
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if df.empty:
        logger.warning("Taxonomy TSV is empty: %s", path)
        return pd.DataFrame(columns=["join_id"])

    if id_column == "auto":
        if "qseqid" in df.columns:
            id_col = "qseqid"
        else:
            id_col = df.columns[0]
    else:
        id_col = id_column
        if id_col not in df.columns:
            raise ValueError(f"Taxonomy id column '{id_col}' not found. Available columns: {list(df.columns)}")

    df = df.copy()
    df["taxonomy_query_id"] = df[id_col].astype(str)
    df["join_id"] = df["taxonomy_query_id"].map(normalize_join_id)

    # Prefix non-rank metadata columns only when collision risk is high.
    # Rank columns are kept as simple names: family, order, phylum, etc.
    for col in TAXONOMY_RANKS:
        if col not in df.columns:
            df[col] = ""

    if duplicate_policy == "most-specific":
        rank_cols = [c for c in TAXONOMY_RANKS if c in df.columns]
        df["_tax_specificity"] = df[rank_cols].apply(lambda r: sum(bool(str(x).strip()) for x in r), axis=1)
        if "lca_taxid" in df.columns:
            df["_has_lca"] = df["lca_taxid"].astype(str).str.len() > 0
        else:
            df["_has_lca"] = False
        df = (
            df.sort_values(["_tax_specificity", "_has_lca"], ascending=[False, False])
              .drop_duplicates("join_id", keep="first")
              .drop(columns=["_tax_specificity", "_has_lca"], errors="ignore")
        )
    elif duplicate_policy == "first":
        df = df.drop_duplicates("join_id", keep="first")
    else:
        raise ValueError(f"Unknown taxonomy duplicate policy: {duplicate_policy}")

    logger.info("Loaded taxonomy TSV: %d unique join IDs from %s", df["join_id"].nunique(), path)
    return df.reset_index(drop=True)


def has_header(first_line: str) -> bool:
    parts = [p.strip().lower() for p in first_line.rstrip("\n\r").split("\t")]
    header_tokens = {
        "id", "name", "node_id", "sequence_id", "base_id", "chunk_id",
        "cluster", "cluster_label", "label", "support", "node_support",
    }
    return any(p in header_tokens for p in parts) or any("cluster" in p or "support" in p for p in parts)


def first_existing(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None



def strip_fasta_marker(value: Any) -> str:
    """Strip a leading FASTA header marker while preserving normal ID parsing."""
    s = str(value).strip()
    if s.startswith(">"):
        s = s[1:].strip()
    return s


def load_label_tsv(
    path: Optional[str],
    id_column: str,
    label_column: str,
    duplicate_policy: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load arbitrary user-supplied labels from a two-column TSV.

    Supported formats:
      ID<TAB>label                    # headerless
      id<TAB>label                    # headered
      >ID<TAB>label                   # FASTA-style ID marker is tolerated

    IDs are normalized with normalize_join_id(), matching taxonomy/cluster joins.
    """
    if not path:
        return pd.DataFrame(columns=["join_id", "custom_label_query_id", "custom_label"])

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = ""
        for line in f:
            if line.strip() and not line.startswith("#"):
                first = line
                break
    if not first:
        logger.warning("Label TSV is empty: %s", path)
        return pd.DataFrame(columns=["join_id", "custom_label_query_id", "custom_label"])

    header = has_header(first)
    if header:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, comment="#")
    else:
        df = pd.read_csv(path, sep="\t", dtype=str, header=None, keep_default_na=False, comment="#")
        if df.shape[1] < 2:
            raise ValueError(
                f"Label TSV '{path}' must contain at least two tab-separated columns: ID<TAB>label."
            )
        names = ["custom_label_query_id", "custom_label"]
        extra = [f"custom_label_extra_{i}" for i in range(max(0, df.shape[1] - len(names)))]
        df.columns = names[:df.shape[1]] + extra

    if df.empty:
        logger.warning("Label TSV is empty after parsing: %s", path)
        return pd.DataFrame(columns=["join_id", "custom_label_query_id", "custom_label"])

    if id_column == "auto":
        id_col = first_existing(
            df.columns,
            ["id", "node_id", "sequence_id", "seq_id", "qseqid", "base_id", "chunk_id", "name", "custom_label_query_id"],
        )
        if id_col is None:
            id_col = df.columns[0]
    else:
        id_col = id_column
        if id_col not in df.columns:
            raise ValueError(f"Label id column '{id_col}' not found. Available columns: {list(df.columns)}")

    if label_column == "auto":
        lab_col = first_existing(
            df.columns,
            ["label", "custom_label", "group", "category", "class", "annotation", "tag"],
        )
        if lab_col is None or lab_col == id_col:
            if df.shape[1] < 2:
                raise ValueError("Could not infer label column from label TSV.")
            lab_col = df.columns[1]
    else:
        lab_col = label_column
        if lab_col not in df.columns:
            raise ValueError(f"Label column '{lab_col}' not found. Available columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["custom_label_query_id"] = df[id_col].map(strip_fasta_marker).astype(str)
    out["join_id"] = out["custom_label_query_id"].map(normalize_join_id)
    out["custom_label"] = df[lab_col].astype(str).str.strip()

    nonempty = out["join_id"].astype(str).str.len().gt(0)
    nonempty &= out["custom_label"].astype(str).str.len().gt(0)
    skipped = int((~nonempty).sum())
    if skipped:
        logger.warning("Skipped %d label rows with empty ID or empty label", skipped)
    out = out.loc[nonempty].copy()

    if duplicate_policy == "first":
        out = out.drop_duplicates("join_id", keep="first")
    elif duplicate_policy == "last":
        out = out.drop_duplicates("join_id", keep="last")
    elif duplicate_policy == "error":
        duplicated = out["join_id"].duplicated(keep=False)
        if duplicated.any():
            examples = ", ".join(out.loc[duplicated, "join_id"].astype(str).head(10).tolist())
            raise ValueError(f"Duplicate IDs found in label TSV. Examples: {examples}")
    else:
        raise ValueError(f"Unknown label duplicate policy: {duplicate_policy}")

    logger.info("Loaded arbitrary label TSV: %d unique join IDs from %s", out["join_id"].nunique(), path)
    return out.reset_index(drop=True)


def load_cluster_tsv(
    path: Optional[str],
    id_column: str,
    cluster_column: str,
    support_column: str,
    duplicate_policy: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["join_id"])

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
    header = has_header(first)

    if header:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    else:
        df = pd.read_csv(path, sep="\t", dtype=str, header=None, keep_default_na=False)
        names = ["cluster_query_id", "cluster_label", "cluster_support"]
        extra = [f"cluster_extra_{i}" for i in range(max(0, df.shape[1] - len(names)))]
        df.columns = names[:df.shape[1]] + extra

    if df.empty:
        logger.warning("Cluster TSV is empty: %s", path)
        return pd.DataFrame(columns=["join_id"])

    if id_column == "auto":
        id_col = first_existing(df.columns, ["node_id", "id", "name", "sequence_id", "base_id", "chunk_id", "cluster_query_id"])
        if id_col is None:
            id_col = df.columns[0]
    else:
        id_col = id_column
        if id_col not in df.columns:
            raise ValueError(f"Cluster id column '{id_col}' not found. Available columns: {list(df.columns)}")

    if cluster_column == "auto":
        cl_col = first_existing(df.columns, ["cluster_label", "cluster", "label"])
        if cl_col is None:
            if df.shape[1] < 2:
                raise ValueError("Could not infer cluster column from cluster TSV.")
            cl_col = df.columns[1]
    else:
        cl_col = cluster_column
        if cl_col not in df.columns:
            raise ValueError(f"Cluster column '{cl_col}' not found. Available columns: {list(df.columns)}")

    if support_column == "auto":
        sup_col = first_existing(df.columns, ["node_support", "support", "cluster_support"])
        if sup_col is None and df.shape[1] >= 3:
            sup_col = df.columns[2]
    elif support_column == "none":
        sup_col = None
    else:
        sup_col = support_column
        if sup_col not in df.columns:
            raise ValueError(f"Cluster support column '{sup_col}' not found. Available columns: {list(df.columns)}")

    out = pd.DataFrame()
    out["cluster_query_id"] = df[id_col].astype(str)
    out["join_id"] = out["cluster_query_id"].map(normalize_join_id)
    out["cluster_label"] = df[cl_col].astype(str)
    if sup_col is not None:
        out["cluster_support"] = pd.to_numeric(df[sup_col], errors="coerce")
    else:
        out["cluster_support"] = np.nan

    if duplicate_policy == "highest-support":
        out["_support_sort"] = out["cluster_support"].fillna(-np.inf)
        out = out.sort_values("_support_sort", ascending=False).drop_duplicates("join_id", keep="first")
        out = out.drop(columns=["_support_sort"], errors="ignore")
    elif duplicate_policy == "first":
        out = out.drop_duplicates("join_id", keep="first")
    else:
        raise ValueError(f"Unknown cluster duplicate policy: {duplicate_policy}")

    logger.info("Loaded cluster TSV: %d unique join IDs from %s", out["join_id"].nunique(), path)
    return out.reset_index(drop=True)


def looks_like_key_value_tsv(first_line: str) -> bool:
    parts = first_line.rstrip("\n\r").split("\t")
    if len(parts) < 2:
        return False
    return any("=" in p for p in parts[1:min(len(parts), 8)])


def parse_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        s = str(value).strip()
        if not s or s.lower() in {"na", "nan", "none", "null", "."}:
            return float("nan")
        return float(s)
    except Exception:
        return float("nan")


def normalize_motif_order_value(value: Any) -> Optional[str]:
    s = str(value).strip().upper()
    if not s or s in {"NA", "NAN", "NONE", "NULL", ".", "UNRESOLVED", "UNKNOWN"}:
        return None
    m = re.search(r"([ABC]{3})", s)
    if m:
        order = m.group(1)
        if set(order) == {"A", "B", "C"} and len(order) == 3:
            return order
    return None


def motif_order_from_positions(row: pd.Series) -> Tuple[str, str]:
    position_sets = [
        ("pos", "posA", "posB", "posC"),
        ("pssm_pos", "pssm_posA", "pssm_posB", "pssm_posC"),
        ("motif_hmm_pos", "motif_hmm_posA", "motif_hmm_posB", "motif_hmm_posC"),
        ("dmnd_pos", "dmnd_posA", "dmnd_posB", "dmnd_posC"),
    ]
    for source, a_col, b_col, c_col in position_sets:
        positions: List[Tuple[float, str]] = []
        for col, label in [(a_col, "A"), (b_col, "B"), (c_col, "C")]:
            if col in row.index:
                val = parse_float(row.get(col))
                if np.isfinite(val):
                    positions.append((val, label))
        if len(positions) == 3:
            positions.sort(key=lambda x: x[0])
            return "".join(label for _, label in positions), source
        if len(positions) == 2:
            positions.sort(key=lambda x: x[0])
            return "Partial " + "".join(label for _, label in positions), source
    return "Unresolved", "none"


def compute_catalytic_center_order(
    row: pd.Series,
    motif_order_column: str,
) -> Tuple[str, str]:
    if motif_order_column != "auto":
        if motif_order_column not in row.index:
            raise ValueError(f"PalmAnnot motif-order column '{motif_order_column}' was not found.")
        value = normalize_motif_order_value(row.get(motif_order_column))
        if value is not None:
            return value, motif_order_column
        return "Unresolved", motif_order_column

    for col in ["catalytic_center_order", "motif_order", "pssm_ABC", "ABC"]:
        if col in row.index:
            value = normalize_motif_order_value(row.get(col))
            if value is not None:
                return value, col

    order, source = motif_order_from_positions(row)
    return order, source


def read_key_value_annotation_tsv(
    path: str,
    id_column_name: str,
    keep_aaseq: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    skipped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip("\n\r")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if not parts:
                continue
            rec: Dict[str, Any] = {id_column_name: parts[0].strip()}
            for field in parts[1:]:
                if "=" not in field:
                    continue
                key, value = field.split("=", 1)
                key = key.strip()
                if not key:
                    continue
                if key == "aaseq" and not keep_aaseq:
                    continue
                # Keep the first value if a key is repeated.
                if key not in rec:
                    rec[key] = value.strip()
            if rec[id_column_name]:
                rows.append(rec)
            else:
                skipped += 1
    if skipped:
        logger.warning("Skipped %d PalmAnnot rows with empty IDs", skipped)
    return pd.DataFrame(rows)


def load_palm_annot_tsv(
    path: Optional[str],
    id_column: str,
    motif_order_column: str,
    duplicate_policy: str,
    keep_aaseq: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["join_id"])

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()

    if looks_like_key_value_tsv(first):
        df = read_key_value_annotation_tsv(
            path=path,
            id_column_name="palmannot_query_id",
            keep_aaseq=keep_aaseq,
            logger=logger,
        )
    else:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        if df.empty:
            logger.warning("PalmAnnot TSV is empty: %s", path)
            return pd.DataFrame(columns=["join_id"])
        if id_column == "auto":
            id_col = first_existing(df.columns, ["query_id", "qseqid", "id", "name", "sequence_id", "base_id", "chunk_id"])
            if id_col is None:
                id_col = df.columns[0]
        else:
            id_col = id_column
            if id_col not in df.columns:
                raise ValueError(f"PalmAnnot id column '{id_col}' not found. Available columns: {list(df.columns)}")
        df = df.copy()
        df["palmannot_query_id"] = df[id_col].astype(str)
        if "aaseq" in df.columns and not keep_aaseq:
            df = df.drop(columns=["aaseq"])

    if df.empty:
        logger.warning("PalmAnnot TSV yielded no annotation rows: %s", path)
        return pd.DataFrame(columns=["join_id"])

    if "palmannot_query_id" not in df.columns:
        if id_column != "auto" and id_column in df.columns:
            df["palmannot_query_id"] = df[id_column].astype(str)
        else:
            df["palmannot_query_id"] = df.iloc[:, 0].astype(str)

    df = df.copy()
    df["join_id"] = df["palmannot_query_id"].map(normalize_join_id)

    orders: List[str] = []
    sources: List[str] = []
    for _, row in df.iterrows():
        order, source = compute_catalytic_center_order(row, motif_order_column)
        orders.append(order)
        sources.append(source)
    df["catalytic_center_order"] = orders
    df["catalytic_center_order_source"] = sources
    df["catalytic_center_order_resolved"] = ~df["catalytic_center_order"].isin(["Unresolved", "PalmAnnot not provided"])

    for col in [
        "rdrp", "pssm_score", "gate_prob", "gdd_prob", "gate_gdd_prob",
        "posA", "posB", "posC", "pssm_posA", "pssm_posB", "pssm_posC",
        "motif_hmm_posA", "motif_hmm_posB", "motif_hmm_posC",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if duplicate_policy == "best-score":
        score = pd.Series(0.0, index=df.index)
        score += df["catalytic_center_order_resolved"].astype(float) * 1_000_000.0
        if "rdrp" in df.columns:
            score += pd.to_numeric(df["rdrp"], errors="coerce").fillna(0.0) * 1_000.0
        if "pssm_score" in df.columns:
            score += pd.to_numeric(df["pssm_score"], errors="coerce").fillna(0.0)
        df["_palmannot_sort_score"] = score
        df = df.sort_values("_palmannot_sort_score", ascending=False).drop_duplicates("join_id", keep="first")
        df = df.drop(columns=["_palmannot_sort_score"], errors="ignore")
    elif duplicate_policy == "first":
        df = df.drop_duplicates("join_id", keep="first")
    else:
        raise ValueError(f"Unknown PalmAnnot duplicate policy: {duplicate_policy}")

    logger.info(
        "Loaded PalmAnnot TSV: %d unique join IDs from %s; resolved catalytic-center order for %d rows",
        df["join_id"].nunique(),
        path,
        int((~df["catalytic_center_order"].isin(["Unresolved", "PalmAnnot not provided"])).sum()),
    )
    return df.reset_index(drop=True)


def load_hmm_domtblout(
    path: Optional[str],
    sequence_column: str,
    evalue_field: str,
    evalue_threshold: float,
    score_threshold: Optional[float],
    duplicate_policy: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=["join_id"])

    domtblout_cols = [
        "target_name", "target_accession", "tlen", "query_name", "query_accession", "qlen",
        "full_Evalue", "full_score", "full_bias", "domain_num", "domain_of",
        "c_Evalue", "i_Evalue", "domain_score", "domain_bias", "hmm_from", "hmm_to",
        "ali_from", "ali_to", "env_from", "env_to", "acc", "description",
    ]
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n\r").split(maxsplit=22)
            if len(parts) < 22:
                continue
            if len(parts) == 22:
                parts.append("")
            rec = dict(zip(domtblout_cols, parts[:23]))
            rows.append(rec)

    if not rows:
        logger.warning("HMMER domtblout yielded no hits: %s", path)
        return pd.DataFrame(columns=["join_id"])

    df = pd.DataFrame(rows)
    numeric_cols = [
        "tlen", "qlen", "full_Evalue", "full_score", "full_bias", "domain_num", "domain_of",
        "c_Evalue", "i_Evalue", "domain_score", "domain_bias", "hmm_from", "hmm_to",
        "ali_from", "ali_to", "env_from", "env_to", "acc",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if evalue_field not in df.columns:
        raise ValueError(f"--hmm-evalue-field '{evalue_field}' is not available. Use one of: full_Evalue, c_Evalue, i_Evalue")

    before = len(df)
    keep = df[evalue_field].le(float(evalue_threshold)).fillna(False)
    if score_threshold is not None:
        keep &= df["domain_score"].ge(float(score_threshold)).fillna(False)
    df = df.loc[keep].copy()
    logger.info("HMMER domtblout thresholding retained %d/%d domain hits", len(df), before)

    if df.empty:
        return pd.DataFrame(columns=["join_id"])

    if sequence_column == "target":
        seq_col = "target_name"
        model_col = "query_name"
    elif sequence_column == "query":
        seq_col = "query_name"
        model_col = "target_name"
    elif sequence_column == "auto":
        # For hmmsearch, the sequence is normally target_name. For hmmscan, it is query_name.
        # A simple heuristic: choose the side with more unique IDs, which is usually the sequence side.
        if df["target_name"].nunique() >= df["query_name"].nunique():
            seq_col = "target_name"
            model_col = "query_name"
        else:
            seq_col = "query_name"
            model_col = "target_name"
        logger.info("Auto-selected HMM sequence column: %s", seq_col)
    else:
        raise ValueError(f"Unknown HMM sequence column mode: {sequence_column}")

    out = pd.DataFrame()
    out["hmm_query_id"] = df[seq_col].astype(str)
    out["join_id"] = out["hmm_query_id"].map(normalize_join_id)
    out["hmm_model"] = df[model_col].astype(str)
    out["hmm_best_full_Evalue"] = df["full_Evalue"]
    out["hmm_best_c_Evalue"] = df["c_Evalue"]
    out["hmm_best_i_Evalue"] = df["i_Evalue"]
    out["hmm_best_domain_score"] = df["domain_score"]
    out["hmm_best_full_score"] = df["full_score"]
    out["hmm_domtblout_detected"] = True

    if duplicate_policy == "best-evalue":
        out["_sort_e"] = pd.to_numeric(out[f"hmm_best_{evalue_field}"], errors="coerce").fillna(np.inf)
        out["_sort_score"] = pd.to_numeric(out["hmm_best_domain_score"], errors="coerce").fillna(-np.inf)
        out = out.sort_values(["_sort_e", "_sort_score"], ascending=[True, False]).drop_duplicates("join_id", keep="first")
        out = out.drop(columns=["_sort_e", "_sort_score"], errors="ignore")
    elif duplicate_policy == "first":
        out = out.drop_duplicates("join_id", keep="first")
    else:
        raise ValueError(f"Unknown HMM duplicate policy: {duplicate_policy}")

    logger.info("Loaded HMM detections: %d unique join IDs from %s", out["join_id"].nunique(), path)
    return out.reset_index(drop=True)


def merge_annotations(
    meta_df: pd.DataFrame,
    tax_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    label_df: pd.DataFrame,
    hmm_df: pd.DataFrame,
    palmannot_df: pd.DataFrame,
    palmsite_positive_threshold: float,
    positive_score_source: str,
    palm_annot_rdrp_threshold: float,
    hmm_detection_source: str,
    custom_label_missing: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    out = meta_df.copy()

    # Model scores stored in the pooled JSON.
    out["palmsite_probability"] = pd.to_numeric(out.get("P", np.nan), errors="coerce")
    out["palmsite_logit"] = pd.to_numeric(out.get("logit", np.nan), errors="coerce")

    if not tax_df.empty:
        suffix_cols = [c for c in tax_df.columns if c != "join_id"]
        out = out.merge(tax_df[["join_id"] + suffix_cols], on="join_id", how="left")
        matched = out["taxonomy_query_id"].notna().sum() if "taxonomy_query_id" in out.columns else 0
        logger.info("Taxonomy matched for %d/%d vectors", int(matched), len(out))
    else:
        logger.info("No taxonomy annotations provided")

    if not cluster_df.empty:
        out = out.merge(cluster_df, on="join_id", how="left")
        matched = out["cluster_label"].notna().sum() if "cluster_label" in out.columns else 0
        logger.info("Cluster annotations matched for %d/%d vectors", int(matched), len(out))
    else:
        logger.info("No cluster annotations provided")

    if not label_df.empty:
        out = out.merge(label_df, on="join_id", how="left")
        matched = out["custom_label"].notna().sum() if "custom_label" in out.columns else 0
        logger.info("Arbitrary labels matched for %d/%d vectors", int(matched), len(out))
    else:
        logger.info("No arbitrary label TSV provided")

    domtblout_provided = not hmm_df.empty
    if domtblout_provided:
        out = out.merge(hmm_df, on="join_id", how="left")
        out["hmm_domtblout_detected"] = out["hmm_domtblout_detected"].astype("boolean").fillna(False).astype(bool)
        matched = int(out["hmm_domtblout_detected"].sum())
        logger.info("HMMER domtblout detections matched for %d/%d vectors", matched, len(out))
    else:
        logger.info("No HMMER domtblout provided")
        out["hmm_domtblout_detected"] = False

    if not palmannot_df.empty:
        suffix_cols = [c for c in palmannot_df.columns if c != "join_id"]
        out = out.merge(palmannot_df[["join_id"] + suffix_cols], on="join_id", how="left")
        matched = out["palmannot_query_id"].notna().sum() if "palmannot_query_id" in out.columns else 0
        logger.info("PalmAnnot matched for %d/%d vectors", int(matched), len(out))
    else:
        logger.info("No PalmAnnot TSV provided")

    if "catalytic_center_order" not in out.columns:
        out["catalytic_center_order"] = "PalmAnnot not provided"
        out["catalytic_center_order_source"] = "none"
        out["catalytic_center_order_resolved"] = False
    else:
        out["catalytic_center_order"] = safe_string_series(out["catalytic_center_order"], missing="Unresolved")
        out["catalytic_center_order_source"] = safe_string_series(out.get("catalytic_center_order_source", pd.Series("none", index=out.index)), missing="none")
        out["catalytic_center_order_resolved"] = out["catalytic_center_order"].isin(["ABC", "CAB", "ACB", "BAC", "BCA", "CBA"])

    # PalmAnnot's key=value rows contain rdrp=<score>. Keep a stable, descriptive alias.
    if "rdrp" in out.columns:
        out["palmannot_rdrp_score"] = pd.to_numeric(out["rdrp"], errors="coerce")
    else:
        out["palmannot_rdrp_score"] = np.nan

    out["hmm_palmannot_rdrp_detected"] = out["palmannot_rdrp_score"].ge(float(palm_annot_rdrp_threshold)).fillna(False)
    out["hmm_palmannot_rdrp_threshold_used"] = float(palm_annot_rdrp_threshold)

    # Define the binary PalmSite-positive status used in the HMM-detected vs HMM-missed plot.
    # The default remains the model probability from pooled JSON. The older
    # --positive-score-source palm_annot_rdrp mode is kept for compatibility, but for the
    # intended figure use json_probability here and use --hmm-detection-source palm_annot_rdrp
    # to define HMM-positive status from PalmAnnot rdrp=<score>.
    if positive_score_source == "json_probability":
        out["palmsite_positive_score"] = out["palmsite_probability"]
        out["palmsite_positive_threshold_used"] = float(palmsite_positive_threshold)
        out["palmsite_positive_source"] = "json_probability"
        out["palmsite_positive"] = out["palmsite_probability"].ge(float(palmsite_positive_threshold)).fillna(False)
    elif positive_score_source == "palm_annot_rdrp":
        out["palmsite_positive_score"] = out["palmannot_rdrp_score"]
        out["palmsite_positive_threshold_used"] = float(palm_annot_rdrp_threshold)
        out["palmsite_positive_source"] = "palm_annot_rdrp"
        out["palmsite_positive"] = out["palmannot_rdrp_score"].ge(float(palm_annot_rdrp_threshold)).fillna(False)
    else:
        raise ValueError(f"Unknown positive score source: {positive_score_source}")

    if positive_score_source == "palm_annot_rdrp" and out["palmannot_rdrp_score"].notna().sum() == 0:
        logger.warning(
            "--positive-score-source palm_annot_rdrp was requested, but no PalmAnnot rdrp= scores matched the vectors. "
            "All PalmSite-positive calls will be false unless IDs are fixed."
        )

    # Define HMM-positive status. This can now be based on PalmAnnot rdrp=<score>, while
    # the global PalmSite-positive threshold remains based on model probability.
    rdrp_scores_available = int(out["palmannot_rdrp_score"].notna().sum()) > 0
    if hmm_detection_source == "auto":
        if rdrp_scores_available:
            hmm_source_resolved = "palm_annot_rdrp"
        elif domtblout_provided:
            hmm_source_resolved = "domtblout"
        else:
            hmm_source_resolved = "none"
    else:
        hmm_source_resolved = hmm_detection_source

    if hmm_source_resolved == "palm_annot_rdrp":
        out["hmm_detected"] = out["hmm_palmannot_rdrp_detected"].astype(bool)
        hmm_status_available = rdrp_scores_available
        if not rdrp_scores_available:
            logger.warning(
                "--hmm-detection-source palm_annot_rdrp was selected, but no PalmAnnot rdrp= scores matched the vectors. "
                "All HMM-detected calls will be false unless IDs or input files are fixed."
            )
    elif hmm_source_resolved == "domtblout":
        out["hmm_detected"] = out["hmm_domtblout_detected"].astype(bool)
        hmm_status_available = domtblout_provided
        if not domtblout_provided:
            logger.warning(
                "--hmm-detection-source domtblout was selected, but no HMMER domtblout hits were provided or retained."
            )
    elif hmm_source_resolved == "domtblout_or_palm_annot_rdrp":
        out["hmm_detected"] = (
            out["hmm_domtblout_detected"].astype(bool) |
            out["hmm_palmannot_rdrp_detected"].astype(bool)
        )
        hmm_status_available = domtblout_provided or rdrp_scores_available
        if not hmm_status_available:
            logger.warning(
                "--hmm-detection-source domtblout_or_palm_annot_rdrp was selected, but neither domtblout hits nor PalmAnnot rdrp= scores were available."
            )
    elif hmm_source_resolved == "none":
        out["hmm_detected"] = False
        hmm_status_available = False
    else:
        raise ValueError(f"Unknown HMM detection source: {hmm_detection_source}")

    out["hmm_detection_source"] = hmm_source_resolved

    if hmm_status_available:
        out["hmm_detection_status"] = np.where(
            out["hmm_detected"],
            "HMM-detected RdRP",
            np.where(out["palmsite_positive"], "PalmSite-positive / HMM-missed", "Other"),
        )
        out["hmm_detected_label"] = np.where(out["hmm_detected"], "HMM-detected", "HMM-missed")
    else:
        out["hmm_detection_status"] = "HMM not provided"
        out["hmm_detected_label"] = "HMM not provided"

    if "cluster_label" not in out.columns:
        out["cluster_label"] = "Unassigned"
    else:
        out["cluster_label"] = safe_string_series(out["cluster_label"], missing="Unassigned")

    if "custom_label" not in out.columns:
        out["custom_label"] = custom_label_missing
        out["custom_label_query_id"] = ""
        out["custom_label_matched"] = False
    else:
        out["custom_label_matched"] = out["custom_label"].notna() & out["custom_label"].astype(str).str.len().gt(0)
        out["custom_label"] = safe_string_series(out["custom_label"], missing=custom_label_missing)
        out["custom_label_query_id"] = safe_string_series(
            out.get("custom_label_query_id", pd.Series("", index=out.index)),
            missing="",
        )

    if "cluster_support" not in out.columns:
        out["cluster_support"] = np.nan

    for col in TAXONOMY_RANKS:
        if col not in out.columns:
            out[col] = "Unclassified"
        else:
            out[col] = safe_string_series(out[col], missing="Unclassified")

    return out

# -----------------------------
# Dimensionality reduction
# -----------------------------

def pre_pca_if_needed(
    X: np.ndarray,
    n_components: int,
    random_state: int,
    logger: logging.Logger,
) -> np.ndarray:
    if n_components <= 0:
        return X
    max_comp = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
    if X.shape[1] <= max_comp or X.shape[0] <= 3:
        logger.info("Skipping pre-PCA: X shape=%s, requested components=%d", X.shape, n_components)
        return X
    logger.info("Pre-PCA: %d -> %d dimensions", X.shape[1], max_comp)
    pca = PCA(n_components=max_comp, random_state=random_state, svd_solver="auto")
    Xp = pca.fit_transform(X)
    logger.info("Pre-PCA retained variance: %.2f%%", 100.0 * float(pca.explained_variance_ratio_.sum()))
    return Xp.astype(np.float64, copy=False)


def run_tsne(
    X: np.ndarray,
    metric: str,
    perplexity_arg: str,
    random_state: int,
    max_iter: int,
    method_arg: str,
    logger: logging.Logger,
) -> np.ndarray:
    N = X.shape[0]
    if N < 3:
        raise ValueError("t-SNE requires at least 3 points.")
    if perplexity_arg == "auto":
        perplexity = min(30.0, max(2.0, math.floor((N - 1) / 3)))
    else:
        perplexity = float(perplexity_arg)
    if perplexity >= N:
        adjusted = max(1.0, float(N - 1) / 3.0)
        logger.warning("t-SNE perplexity %.3f must be < N=%d; using %.3f", perplexity, N, adjusted)
        perplexity = adjusted

    init = "pca" if metric == "euclidean" and X.shape[1] >= 2 else "random"
    if method_arg == "auto":
        method = "exact" if N <= 5000 else "barnes_hut"
    else:
        method = method_arg
    logger.info(
        "Running t-SNE: N=%d, d=%d, metric=%s, perplexity=%.3f, max_iter=%d, method=%s",
        N, X.shape[1], metric, perplexity, max_iter, method,
    )
    tsne_kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        metric=metric,
        init=init,
        learning_rate="auto",
        random_state=random_state,
        method=method,
    )
    # scikit-learn renamed n_iter to max_iter; support both APIs.
    if "max_iter" in inspect.signature(TSNE).parameters:
        tsne_kwargs["max_iter"] = int(max_iter)
    else:
        tsne_kwargs["n_iter"] = int(max_iter)
    tsne = TSNE(**tsne_kwargs)
    return tsne.fit_transform(X).astype(np.float64, copy=False)


def run_umap(
    X: np.ndarray,
    metric: str,
    n_neighbors: int,
    min_dist: float,
    random_state: int,
    logger: logging.Logger,
) -> np.ndarray:
    try:
        import umap  # type: ignore
    except ImportError as e:
        raise ImportError(
            "UMAP requested but umap-learn is not installed. Install with: pip install umap-learn"
        ) from e
    N = X.shape[0]
    if N < 3:
        raise ValueError("UMAP requires at least 3 points.")
    n_neighbors_eff = max(2, min(int(n_neighbors), N - 1))
    logger.info(
        "Running UMAP: N=%d, d=%d, metric=%s, n_neighbors=%d, min_dist=%.3f",
        N,
        X.shape[1],
        metric,
        n_neighbors_eff,
        min_dist,
    )
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors_eff,
        min_dist=float(min_dist),
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X).astype(np.float64, copy=False)


# -----------------------------
# Plotting
# -----------------------------

def shorten_label(x: Any, max_len: int = 45) -> str:
    s = str(x)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def categorical_values_for_plot(
    series: pd.Series,
    max_categories: int,
    missing_label: str,
    color_by: str,
) -> Tuple[pd.Series, List[str]]:
    s = safe_string_series(series, missing=missing_label)
    counts = s.value_counts(dropna=False)
    preferred = [c for c in PREFERRED_CATEGORY_ORDER.get(color_by, []) if c in counts.index]
    remaining = [c for c in counts.index.tolist() if c not in preferred]
    ordered = preferred + remaining
    keep = ordered[:max_categories]
    if len(ordered) > max_categories:
        s = s.where(s.isin(keep), other="Other")
        cats = keep + ["Other"] if "Other" not in keep else keep
    else:
        cats = keep
    return s, cats


def plot_embedding(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_by: str,
    out_prefix: str,
    formats: Sequence[str],
    title: str,
    point_size: float,
    alpha: float,
    max_categories: int,
    label_centroids: bool,
    label_points: str,
    max_point_labels: int,
    point_label_field: str,
    width: float,
    height: float,
    dpi: int,
    continuous_clip_quantiles: Tuple[float, float],
    logger: logging.Logger,
) -> None:
    if color_by not in df.columns:
        logger.warning("Skipping plot for missing color column: %s", color_by)
        return

    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)

    is_num, numeric = infer_numeric(df[color_by])
    if is_num and color_by not in FORCE_CATEGORICAL_COLUMNS:
        values = numeric.to_numpy(dtype=float)
        finite = np.isfinite(values)
        vmin = vmax = None
        q_low, q_high = continuous_clip_quantiles
        if finite.any() and 0.0 <= q_low < q_high <= 1.0 and (q_low > 0.0 or q_high < 1.0):
            vmin = float(np.nanquantile(values[finite], q_low))
            vmax = float(np.nanquantile(values[finite], q_high))
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin = vmax = None
        if (~finite).any():
            ax.scatter(
                df.loc[~finite, x_col].to_numpy(),
                df.loc[~finite, y_col].to_numpy(),
                s=point_size,
                alpha=0.25,
                linewidths=0,
                label="NaN",
            )
        sc = ax.scatter(
            df.loc[finite, x_col].to_numpy(),
            df.loc[finite, y_col].to_numpy(),
            c=values[finite],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_by)
    else:
        values, categories = categorical_values_for_plot(df[color_by], max_categories, missing_label="Unclassified", color_by=color_by)
        cmap = plt.get_cmap("tab20", max(1, min(20, len(categories))))
        for idx, cat in enumerate(categories):
            mask = values == cat
            if not mask.any():
                continue
            ax.scatter(
                df.loc[mask, x_col].to_numpy(),
                df.loc[mask, y_col].to_numpy(),
                s=point_size,
                alpha=alpha,
                linewidths=0,
                color=cmap(idx % 20),
                label=f"{shorten_label(cat)} (n={int(mask.sum())})",
            )
        if categories:
            ax.legend(
                title=color_by,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize=8,
                title_fontsize=9,
                markerscale=1.2,
            )

        if label_centroids:
            for cat in categories:
                if cat == "Other":
                    continue
                mask = values == cat
                if mask.sum() < 2:
                    continue
                cx = float(df.loc[mask, x_col].mean())
                cy = float(df.loc[mask, y_col].mean())
                ax.text(cx, cy, shorten_label(cat, 30), fontsize=8, ha="center", va="center")

    if label_points != "none":
        if point_label_field not in df.columns:
            logger.warning("Point label field '%s' not found; skipping point labels", point_label_field)
        elif len(df) > max_point_labels:
            logger.warning(
                "Skipping point labels because N=%d > --max-point-labels=%d",
                len(df),
                max_point_labels,
            )
        else:
            for _, row in df.iterrows():
                ax.text(
                    float(row[x_col]),
                    float(row[y_col]),
                    shorten_label(row[point_label_field], 30),
                    fontsize=6,
                    alpha=0.85,
                )

    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    for fmt in formats:
        fmt_clean = fmt.lower().lstrip(".")
        out_path = f"{out_prefix}.{color_by}.{fmt_clean}"
        ensure_parent(out_path)
        fig.savefig(out_path, dpi=dpi if fmt_clean in {"png", "jpg", "jpeg"} else None, bbox_inches="tight")
        logger.info("Wrote plot: %s", out_path)
    plt.close(fig)


# -----------------------------
# CLI / main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot 2-D t-SNE/UMAP maps from PalmSite pooled JSON vectors with taxonomy and cluster annotations."
    )

    # Input vectors
    p.add_argument("--input-json", required=True, help="PalmSite pooled_panels.json file.")
    p.add_argument(
        "--json-panel",
        default="backbone.span_attn_norm",
        help="Panel to use, e.g. backbone.span_attn_norm, input.span_attn_norm, backbone.span_mean, input.span_mean.",
    )
    p.add_argument("--json-best-only", action="store_true", help="Keep only entries where is_best_base_chunk is true.")
    p.add_argument("--id-field", default="base_id", help="ID to use for node_id: base_id, chunk_id, or another JSON field.")
    p.add_argument("--dedupe", choices=["none", "first", "maxP", "mean"], default="none", help="How to handle duplicated node_id values after loading JSON.")
    p.add_argument("--no-l2", action="store_true", help="Do not L2-normalize vectors after loading.")
    p.add_argument(
        "--json-load-mode",
        choices=["auto", "normal", "stream"],
        default="auto",
        help=(
            "How to read --input-json. normal uses json.load and requires RAM for the whole file. "
            "stream uses ijson and is required for very large pooled_panels.json files. "
            "auto uses stream for files >=5 GiB."
        ),
    )
    p.add_argument(
        "--max-random-points",
        type=int,
        default=0,
        help=(
            "Reservoir-sample this many non-priority vectors while streaming. "
            "0 keeps all loaded vectors. For huge JSON, set this, e.g. 50000."
        ),
    )
    p.add_argument(
        "--always-keep-json-positive",
        action="store_true",
        help=(
            "When --max-random-points is used, keep all records with pooled JSON P >= "
            "--palmsite-positive-threshold in addition to the random background sample."
        ),
    )
    p.add_argument(
        "--always-keep-hmm-positive",
        action="store_true",
        help=(
            "When --max-random-points is used, force-keep sequences that are HMM-positive according to "
            "--hmm-detection-source, --hmm-domtblout, and/or palm_annot rdrp= threshold. "
            "This is useful for HMM-detected vs HMM-missed visualization."
        ),
    )
    p.add_argument(
        "--json-progress-every",
        type=int,
        default=100000,
        help="Log streaming JSON progress every N top-level records. Set 0 to disable.",
    )

    # Annotation files
    p.add_argument("--taxonomy-tsv", default=None, help="Ranked lineage TSV, e.g. query_to_lca_ranked_lineage.tsv.")
    p.add_argument("--taxonomy-id-column", default="auto", help="Taxonomy query ID column. Default: qseqid if present, else first column.")
    p.add_argument("--taxonomy-duplicate-policy", choices=["most-specific", "first"], default="most-specific")
    p.add_argument("--cluster-tsv", default=None, help="Graph clustering TSV. Headered or headerless sequence_id, cluster, support format is supported.")
    p.add_argument("--cluster-id-column", default="auto", help="Cluster TSV ID column. Default: infer automatically.")
    p.add_argument("--cluster-column", default="auto", help="Cluster label column. Default: infer automatically.")
    p.add_argument("--cluster-support-column", default="auto", help="Cluster support column. Use 'none' to ignore.")
    p.add_argument("--cluster-duplicate-policy", choices=["highest-support", "first"], default="highest-support")

    p.add_argument("--label-tsv", default=None, help="Arbitrary label TSV with ID<TAB>label rows. Headered TSVs are also supported.")
    p.add_argument("--label-id-column", default="auto", help="Arbitrary label TSV ID column. Default: infer automatically.")
    p.add_argument("--label-column", default="auto", help="Arbitrary label TSV label column. Default: infer automatically.")
    p.add_argument("--label-duplicate-policy", choices=["first", "last", "error"], default="first", help="How to handle duplicated IDs in --label-tsv.")
    p.add_argument("--custom-label-missing", default="Unlabeled", help="Value assigned to vectors not found in --label-tsv.")

    p.add_argument("--hmm-domtblout", default=None, help="HMMER domtblout from NeORdRp or another RdRP HMM search.")
    p.add_argument("--hmm-sequence-column", choices=["target", "query", "auto"], default="target", help="Which domtblout side contains sequence IDs. hmmsearch usually uses target; hmmscan usually uses query.")
    p.add_argument("--hmm-evalue-field", choices=["full_Evalue", "c_Evalue", "i_Evalue"], default="i_Evalue", help="HMMER e-value field used for detection thresholding.")
    p.add_argument("--hmm-evalue-threshold", type=float, default=1e-5, help="Maximum HMMER e-value for HMM-detected status.")
    p.add_argument("--hmm-score-threshold", type=float, default=None, help="Optional minimum HMMER domain score for HMM-detected status.")
    p.add_argument("--hmm-duplicate-policy", choices=["best-evalue", "first"], default="best-evalue")
    p.add_argument(
        "--hmm-detection-source",
        choices=["auto", "domtblout", "palm_annot_rdrp", "domtblout_or_palm_annot_rdrp"],
        default="auto",
        help=(
            "Source used for HMM-positive/RdRP-detected status in the HMM-detected vs HMM-missed plot. "
            "auto uses PalmAnnot rdrp=<score> when available, otherwise domtblout. "
            "Use palm_annot_rdrp to force rdrp=<score> thresholding."
        ),
    )

    p.add_argument("--palm-annot-tsv", default=None, help="palm_annot output TSV. Headered TSV and key=value rows are both supported.")
    p.add_argument("--palm-annot-id-column", default="auto", help="PalmAnnot ID column for headered TSV. Ignored for key=value rows.")
    p.add_argument("--palm-annot-motif-order-column", default="auto", help="Column used as catalytic center order. Default: pssm_ABC if available, otherwise inferred from A/B/C positions.")
    p.add_argument("--palm-annot-duplicate-policy", choices=["best-score", "first"], default="best-score")
    p.add_argument("--palm-annot-keep-aaseq", action="store_true", help="Keep the aaseq column from PalmAnnot in output TSVs. Default drops it to avoid huge files.")
    p.add_argument(
        "--positive-score-source",
        choices=["json_probability", "palm_annot_rdrp"],
        default="json_probability",
        help=(
            "Score used to define PalmSite-positive / HMM-missed. Default json_probability uses P from pooled JSON. "
            "For your current figure, keep this as json_probability and use --hmm-detection-source palm_annot_rdrp "
            "to define HMM-positive status from rdrp=<score>."
        ),
    )
    p.add_argument("--palmsite-positive-threshold", type=float, default=0.5, help="Probability threshold used when --positive-score-source json_probability.")
    p.add_argument("--palm-annot-rdrp-threshold", type=float, default=50.0, help="Minimum PalmAnnot rdrp=<score> used for HMM-positive status when --hmm-detection-source uses palm_annot_rdrp.")

    # Dimensionality reduction
    p.add_argument("--reducers", nargs="+", default=["tsne", "umap"], choices=["tsne", "umap"], help="Reducers to run.")
    p.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    p.add_argument("--pre-pca-components", type=int, default=50, help="PCA dimensions before t-SNE/UMAP. Set 0 to disable.")
    p.add_argument("--tsne-perplexity", default="auto", help="t-SNE perplexity or 'auto'.")
    p.add_argument("--tsne-max-iter", type=int, default=1000, help="Maximum t-SNE optimization iterations.")
    p.add_argument("--tsne-method", choices=["auto", "exact", "barnes_hut"], default="auto", help="t-SNE method. auto uses exact for N<=5000, otherwise barnes_hut.")
    p.add_argument("--umap-neighbors", type=int, default=30)
    p.add_argument("--umap-min-dist", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)

    # Output / plotting
    p.add_argument("--output-prefix", required=True, help="Output prefix for coordinate TSVs and plots.")
    p.add_argument(
        "--color-by",
        nargs="+",
        default=["palmsite_logit", "hmm_detection_status", "catalytic_center_order"],
        help="Columns to color by. Examples: custom_label palmsite_logit palmsite_probability hmm_detection_status catalytic_center_order cluster_label family order.",
    )
    p.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Plot formats, e.g. pdf png svg.")
    p.add_argument("--point-size", type=float, default=8.0)
    p.add_argument("--alpha", type=float, default=0.85)
    p.add_argument("--max-categories", type=int, default=20, help="Show top N categories; the rest become Other.")
    p.add_argument("--label-centroids", action="store_true", help="Label category centroids for categorical plots.")
    p.add_argument("--label-points", choices=["none", "all"], default="none", help="Label individual points. Use only for small N.")
    p.add_argument("--max-point-labels", type=int, default=200)
    p.add_argument("--point-label-field", default="node_id")
    p.add_argument("--plot-width", type=float, default=7.5)
    p.add_argument("--plot-height", type=float, default=6.2)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--continuous-clip-quantiles", nargs=2, type=float, default=[0.0, 1.0], metavar=("QLOW", "QHIGH"), help="Optional color clipping quantiles for continuous labels, e.g. 0.01 0.99.")

    p.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return p.parse_args()



def build_hmm_positive_keep_ids(
    hmm_df: pd.DataFrame,
    palmannot_df: pd.DataFrame,
    hmm_detection_source: str,
    palm_annot_rdrp_threshold: float,
    logger: logging.Logger,
) -> set[str]:
    """Return join IDs that should be forced into a streaming sample as HMM positives."""
    ids: set[str] = set()
    has_domtblout = not hmm_df.empty and "join_id" in hmm_df.columns
    has_rdrp = (
        not palmannot_df.empty
        and "join_id" in palmannot_df.columns
        and "rdrp" in palmannot_df.columns
        and pd.to_numeric(palmannot_df["rdrp"], errors="coerce").notna().any()
    )

    if hmm_detection_source == "auto":
        source = "palm_annot_rdrp" if has_rdrp else ("domtblout" if has_domtblout else "none")
    else:
        source = hmm_detection_source

    if source in {"domtblout", "domtblout_or_palm_annot_rdrp"} and has_domtblout:
        ids.update(hmm_df["join_id"].dropna().astype(str).tolist())

    if source in {"palm_annot_rdrp", "domtblout_or_palm_annot_rdrp"} and has_rdrp:
        rdrp = pd.to_numeric(palmannot_df["rdrp"], errors="coerce")
        keep = rdrp.ge(float(palm_annot_rdrp_threshold)).fillna(False)
        ids.update(palmannot_df.loc[keep, "join_id"].dropna().astype(str).tolist())

    logger.info("Built HMM-positive force-keep ID set from %s: %d IDs", source, len(ids))
    return ids

def main() -> None:
    args = parse_args()
    logger = setup_logger(args.log_level)

    color_by = [resolve_color_by_alias(c) for c in parse_list_arg(args.color_by, default=["palmsite_logit", "hmm_detection_status", "catalytic_center_order"])]
    if args.label_tsv and "custom_label" not in color_by:
        color_by.append("custom_label")
    formats = parse_list_arg(args.formats, default=["pdf", "png"])
    continuous_clip_quantiles = (float(args.continuous_clip_quantiles[0]), float(args.continuous_clip_quantiles[1]))
    if not (0.0 <= continuous_clip_quantiles[0] < continuous_clip_quantiles[1] <= 1.0):
        raise ValueError("--continuous-clip-quantiles must satisfy 0 <= QLOW < QHIGH <= 1")

    # Load HMM/PalmAnnot first when streaming so HMM-positive IDs can be force-kept.
    hmm_df = load_hmm_domtblout(
        args.hmm_domtblout,
        sequence_column=args.hmm_sequence_column,
        evalue_field=args.hmm_evalue_field,
        evalue_threshold=args.hmm_evalue_threshold,
        score_threshold=args.hmm_score_threshold,
        duplicate_policy=args.hmm_duplicate_policy,
        logger=logger,
    )
    palmannot_df = load_palm_annot_tsv(
        args.palm_annot_tsv,
        id_column=args.palm_annot_id_column,
        motif_order_column=args.palm_annot_motif_order_column,
        duplicate_policy=args.palm_annot_duplicate_policy,
        keep_aaseq=args.palm_annot_keep_aaseq,
        logger=logger,
    )
    force_keep_ids: set[str] = set()
    if args.always_keep_hmm_positive:
        force_keep_ids = build_hmm_positive_keep_ids(
            hmm_df=hmm_df,
            palmannot_df=palmannot_df,
            hmm_detection_source=args.hmm_detection_source,
            palm_annot_rdrp_threshold=args.palm_annot_rdrp_threshold,
            logger=logger,
        )

    meta_df, X = load_pooled_json(
        json_path=args.input_json,
        panel=args.json_panel,
        id_field=args.id_field,
        best_only=args.json_best_only,
        dedupe=args.dedupe,
        l2=(not args.no_l2),
        logger=logger,
        load_mode=args.json_load_mode,
        max_random_points=args.max_random_points,
        always_keep_json_positive=args.always_keep_json_positive,
        json_positive_threshold=args.palmsite_positive_threshold,
        random_state=args.random_state,
        progress_every=args.json_progress_every,
        always_keep_ids=force_keep_ids,
    )

    tax_df = load_taxonomy_tsv(
        args.taxonomy_tsv,
        id_column=args.taxonomy_id_column,
        duplicate_policy=args.taxonomy_duplicate_policy,
        logger=logger,
    )
    cluster_df = load_cluster_tsv(
        args.cluster_tsv,
        id_column=args.cluster_id_column,
        cluster_column=args.cluster_column,
        support_column=args.cluster_support_column,
        duplicate_policy=args.cluster_duplicate_policy,
        logger=logger,
    )
    label_df = load_label_tsv(
        args.label_tsv,
        id_column=args.label_id_column,
        label_column=args.label_column,
        duplicate_policy=args.label_duplicate_policy,
        logger=logger,
    )
    annot_df = merge_annotations(
        meta_df,
        tax_df,
        cluster_df,
        label_df,
        hmm_df,
        palmannot_df,
        palmsite_positive_threshold=args.palmsite_positive_threshold,
        positive_score_source=args.positive_score_source,
        palm_annot_rdrp_threshold=args.palm_annot_rdrp_threshold,
        hmm_detection_source=args.hmm_detection_source,
        custom_label_missing=args.custom_label_missing,
        logger=logger,
    )

    X_work = X
    if args.pre_pca_components and args.pre_pca_components > 0:
        X_work = pre_pca_if_needed(X_work, args.pre_pca_components, args.random_state, logger=logger)
        if args.metric == "cosine":
            X_work = l2_normalize_rows(X_work)

    # Save the loaded matrix and annotation before reduction for reproducibility.
    matrix_path = f"{args.output_prefix}.loaded_vectors.npz"
    ensure_parent(matrix_path)
    np.savez_compressed(
        matrix_path,
        X=X.astype(np.float32),
        node_id=annot_df["node_id"].astype(str).to_numpy(),
        join_id=annot_df["join_id"].astype(str).to_numpy(),
        base_id=annot_df["base_id"].astype(str).to_numpy(),
        chunk_id=annot_df["chunk_id"].astype(str).to_numpy(),
        json_panel=np.array([args.json_panel], dtype=object),
    )
    logger.info("Wrote loaded vector matrix: %s", matrix_path)

    base_annot_path = f"{args.output_prefix}.annotations.tsv"
    ensure_parent(base_annot_path)
    annot_df.to_csv(base_annot_path, sep="\t", index=False, lineterminator="\n")
    logger.info("Wrote annotation table: %s", base_annot_path)

    for reducer_name in args.reducers:
        if reducer_name == "tsne":
            coords = run_tsne(
                X_work,
                metric=args.metric,
                perplexity_arg=args.tsne_perplexity,
                random_state=args.random_state,
                max_iter=args.tsne_max_iter,
                method_arg=args.tsne_method,
                logger=logger,
            )
            x_col, y_col = "tsne_1", "tsne_2"
        elif reducer_name == "umap":
            coords = run_umap(
                X_work,
                metric=args.metric,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                random_state=args.random_state,
                logger=logger,
            )
            x_col, y_col = "umap_1", "umap_2"
        else:
            raise ValueError(f"Unknown reducer: {reducer_name}")

        out_df = annot_df.copy()
        out_df[x_col] = coords[:, 0]
        out_df[y_col] = coords[:, 1]
        coord_path = f"{args.output_prefix}.{reducer_name}.coordinates.tsv"
        ensure_parent(coord_path)
        out_df.to_csv(coord_path, sep="\t", index=False, lineterminator="\n")
        logger.info("Wrote coordinates: %s", coord_path)

        for col in color_by:
            col = resolve_color_by_alias(col)
            title = f"{reducer_name.upper()} of {args.json_panel} colored by {col}"
            plot_prefix = f"{args.output_prefix}.{reducer_name}"
            plot_embedding(
                out_df,
                x_col=x_col,
                y_col=y_col,
                color_by=col,
                out_prefix=plot_prefix,
                formats=formats,
                title=title,
                point_size=args.point_size,
                alpha=args.alpha,
                max_categories=args.max_categories,
                label_centroids=args.label_centroids,
                label_points=args.label_points,
                max_point_labels=args.max_point_labels,
                point_label_field=args.point_label_field,
                width=args.plot_width,
                height=args.plot_height,
                dpi=args.dpi,
                continuous_clip_quantiles=continuous_clip_quantiles,
                logger=logger,
            )

    logger.info("Finished")


if __name__ == "__main__":
    main()

