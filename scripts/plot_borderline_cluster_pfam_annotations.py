#!/usr/bin/env python3
"""
replicate_palmsite_borderline_figures.py

Reproduce the main/supplementary PalmSite-borderline figures from:
  1. PalmSite GFF3 output
  2. graph clustering assignment file
  3. Pfam hmmscan/hmmsearch domtblout
  4. optional UMAP/TSNE coordinate TSV produced by plot_pooled_json_embedding.py

The script makes:
  - UMAP colored by PalmSite logit
  - UMAP colored by graph cluster label
  - cluster-wise PalmSite logit boxplot with p=0.5..0.95 shaded
  - borderline cluster selection tables
  - Pfam domain enrichment tables
  - Pfam enrichment dotplots
  - cluster size vs Pfam annotated fraction
  - median logit vs Pfam annotated fraction, with RT-enriched clusters highlighted

Expected graph clustering format:
  sequence_id<TAB>cluster_label<TAB>cluster_support

Expected PalmSite GFF3 attributes include:
  P=...
  Logit=...
  len=...

Expected Pfam domtblout is standard HMMER domtblout. The script handles query IDs
like MGYP...:start-end by normalizing to the base ID.

Example
-------
python replicate_palmsite_borderline_figures.py \
  --gff3 palmsite.gff \
  --cluster-file span_mean.k10.graph_clustering \
  --pfam-domtblout pfam.domtblout \
  --coordinates-tsv backbone_span_mean.umap.coordinates.tsv \
  --out-prefix results/span_mean.k10 \
  --lower-p 0.5 \
  --upper-p 0.95 \
  --min-frac-borderline 0.4 \
  --min-cluster-size 100

If you do not provide --coordinates-tsv, the UMAP panels are skipped.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import fisher_exact


# -----------------------------
# General utilities
# -----------------------------

def ensure_parent(path: str | Path) -> None:
    p = Path(path)
    if p.parent and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)


def normalize_id(value: object) -> str:
    """
    Normalize sequence IDs for joining.

    Handles:
      MGYP000000000001
      MGYP000000000001:123-456
      MGYP000000000001|chunk_0001_of_0001|aa_000000_000300
    """
    s = str(value).strip()
    if not s:
        return s
    s = s.split()[0]
    if "|chunk_" in s:
        s = s.split("|chunk_", 1)[0]
    m = re.match(r"^(.+):\d+-\d+$", s)
    if m:
        s = m.group(1)
    return s


def logit_from_p(p: float) -> float:
    return math.log(p / (1.0 - p))


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    n = len(pvals)
    if n == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adj, 1.0)
    return out


def parse_gff_attributes(attr_str: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    for item in str(attr_str).split(";"):
        if "=" in item:
            k, v = item.split("=", 1)
            d[k] = v
    return d


# -----------------------------
# Load core data
# -----------------------------

def load_palmsite_gff(gff3: str) -> pd.DataFrame:
    rows = []
    with open(gff3, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9:
                continue
            seq_id = normalize_id(parts[0])
            attrs = parse_gff_attributes(parts[8])
            try:
                p = float(attrs.get("P", parts[5]))
                logit = float(attrs.get("Logit", attrs.get("CalibratedLogit", "nan")))
            except ValueError:
                continue
            try:
                seq_len = int(float(attrs["len"])) if "len" in attrs else np.nan
            except ValueError:
                seq_len = np.nan
            rows.append({
                "seq_id": seq_id,
                "P": p,
                "logit": logit,
                "orig_len": seq_len,
                "gff_line": line.rstrip("\n"),
            })
    if not rows:
        raise ValueError(f"No PalmSite records parsed from {gff3}")
    return pd.DataFrame(rows)


def load_graph_clusters(cluster_file: str) -> pd.DataFrame:
    df = pd.read_csv(
        cluster_file,
        sep="\t",
        header=None,
        names=["seq_id", "cluster_label", "cluster_support"],
    )
    df["seq_id"] = df["seq_id"].map(normalize_id)
    df["cluster_label"] = df["cluster_label"].astype(str)
    df["cluster_support"] = pd.to_numeric(df["cluster_support"], errors="coerce")
    return df


def merge_gff_clusters(gff_df: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
    out = gff_df.merge(cluster_df[["seq_id", "cluster_label", "cluster_support"]], on="seq_id", how="left")
    out["cluster_label"] = out["cluster_label"].fillna("Unassigned").astype(str)
    out["cluster_support"] = pd.to_numeric(out["cluster_support"], errors="coerce")
    return out


# -----------------------------
# Cluster summaries and borderline selection
# -----------------------------

def add_borderline_flags(
    df: pd.DataFrame,
    lower_p: float,
    upper_p: float,
    clean_positive_p: float,
    clean_negative_logit: float,
) -> pd.DataFrame:
    out = df.copy()
    out["is_borderline"] = out["P"].between(lower_p, upper_p, inclusive="both")
    out["is_near_threshold"] = out["P"].between(max(lower_p, 0.8), min(upper_p, 0.95), inclusive="both")
    out["is_clean_positive"] = out["P"] > clean_positive_p
    out["is_clean_negative"] = out["logit"] <= clean_negative_logit
    return out


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("cluster_label", sort=True)
        .agg(
            n=("seq_id", "size"),
            mean_P=("P", "mean"),
            median_P=("P", "median"),
            sd_P=("P", "std"),
            var_P=("P", "var"),
            q05_P=("P", lambda s: s.quantile(0.05)),
            q25_P=("P", lambda s: s.quantile(0.25)),
            q75_P=("P", lambda s: s.quantile(0.75)),
            q95_P=("P", lambda s: s.quantile(0.95)),
            mean_logit=("logit", "mean"),
            median_logit=("logit", "median"),
            sd_logit=("logit", "std"),
            var_logit=("logit", "var"),
            q05_logit=("logit", lambda s: s.quantile(0.05)),
            q25_logit=("logit", lambda s: s.quantile(0.25)),
            q75_logit=("logit", lambda s: s.quantile(0.75)),
            q95_logit=("logit", lambda s: s.quantile(0.95)),
            frac_borderline=("is_borderline", "mean"),
            frac_near_threshold=("is_near_threshold", "mean"),
            frac_clean_positive=("is_clean_positive", "mean"),
            frac_clean_negative=("is_clean_negative", "mean"),
            mean_len=("orig_len", "mean"),
            median_len=("orig_len", "median"),
            mean_cluster_support=("cluster_support", "mean"),
            sd_cluster_support=("cluster_support", "std"),
        )
        .reset_index()
    )
    summary["iqr_logit"] = summary["q75_logit"] - summary["q25_logit"]
    summary["iqr_P"] = summary["q75_P"] - summary["q25_P"]
    return summary


def select_borderline_clusters(
    summary: pd.DataFrame,
    lower_p: float,
    upper_p: float,
    min_frac_borderline: float,
    min_cluster_size: int,
    require_median_in_window: bool,
) -> pd.DataFrame:
    keep = (summary["n"] >= min_cluster_size) & (summary["frac_borderline"] >= min_frac_borderline)
    if require_median_in_window:
        keep &= summary["median_P"].between(lower_p, upper_p, inclusive="both")
    selected = summary.loc[keep].copy()
    selected = selected.sort_values(["frac_borderline", "median_logit", "n"], ascending=[False, True, False])
    return selected


# -----------------------------
# Pfam domtblout parsing and enrichment
# -----------------------------

DOMTBLOUT_COLS = [
    "target_name", "target_accession", "tlen", "query_name", "query_accession", "qlen",
    "full_Evalue", "full_score", "full_bias", "domain_num", "domain_of",
    "c_Evalue", "i_Evalue", "domain_score", "domain_bias", "hmm_from", "hmm_to",
    "ali_from", "ali_to", "env_from", "env_to", "acc", "description",
]


def load_domtblout(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n\r").split(maxsplit=22)
            if len(parts) < 22:
                continue
            if len(parts) == 22:
                parts.append("")
            rows.append(parts[:23])
    df = pd.DataFrame(rows, columns=DOMTBLOUT_COLS)
    if df.empty:
        return df
    numeric_cols = [
        "tlen", "qlen", "full_Evalue", "full_score", "full_bias", "domain_num", "domain_of",
        "c_Evalue", "i_Evalue", "domain_score", "domain_bias", "hmm_from", "hmm_to",
        "ali_from", "ali_to", "env_from", "env_to", "acc",
    ]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def prepare_pfam_hits(
    pfam: pd.DataFrame,
    selected_ids: set[str],
    i_evalue_cutoff: float,
    hmm_coverage_cutoff: float,
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Infer sequence side and domain side, normalize sequence IDs, and filter hits.
    Returns filtered seq-domain hits plus a note.
    """
    if pfam.empty:
        return pd.DataFrame(), "NA", "NA", "empty domtblout"

    pfam = pfam.copy()
    pfam["query_norm"] = pfam["query_name"].map(normalize_id)
    pfam["target_norm"] = pfam["target_name"].map(normalize_id)

    query_intersection = len(set(pfam["query_norm"]) & selected_ids)
    target_intersection = len(set(pfam["target_norm"]) & selected_ids)

    if query_intersection >= target_intersection:
        seq_norm_col = "query_norm"
        seq_raw_col = "query_name"
        domain_col = "target_name"
        domain_acc_col = "target_accession"
        domain_len_col = "tlen"
        seq_len_col = "qlen"
    else:
        seq_norm_col = "target_norm"
        seq_raw_col = "target_name"
        domain_col = "query_name"
        domain_acc_col = "query_accession"
        domain_len_col = "qlen"
        seq_len_col = "tlen"

    pfam["seq_id"] = pfam[seq_norm_col].astype(str)
    pfam["seq_id_raw"] = pfam[seq_raw_col].astype(str)
    pfam["domain_id"] = pfam[domain_col].astype(str)
    pfam["domain_acc"] = pfam[domain_acc_col].astype(str)
    pfam["domain_desc"] = pfam["description"].astype(str)
    pfam["domain_hmm_coverage"] = (pfam["hmm_to"] - pfam["hmm_from"] + 1) / pfam[domain_len_col]
    pfam["query_alignment_coverage"] = (pfam["ali_to"] - pfam["ali_from"] + 1) / pfam[seq_len_col]

    raw = pfam[pfam["seq_id"].isin(selected_ids)].copy()
    filt = raw[
        raw["i_Evalue"].le(i_evalue_cutoff).fillna(False)
        & raw["domain_hmm_coverage"].ge(hmm_coverage_cutoff).fillna(False)
    ].copy()

    if filt.empty and not raw.empty:
        filt = raw.copy()
        note = "No hits survived i_Evalue/coverage filter; used all raw selected-sequence hits."
    else:
        note = f"Filter: i_Evalue <= {i_evalue_cutoff:g}, domain HMM coverage >= {hmm_coverage_cutoff:g}"

    seq_domain = (
        filt.sort_values(["seq_id", "domain_id", "i_Evalue", "domain_score"], ascending=[True, True, True, False])
        .drop_duplicates(["seq_id", "domain_id"], keep="first")
        .copy()
    )
    return seq_domain, seq_raw_col, domain_col, note


def run_domain_enrichment(
    selected_seq_df: pd.DataFrame,
    pfam_seq_domain: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Foreground = one selected borderline cluster.
    Background = all other selected borderline clusters.
    """
    seq_cluster = selected_seq_df[["seq_id", "cluster_label"]].drop_duplicates().copy()
    seq_cluster["cluster_label"] = seq_cluster["cluster_label"].astype(str)
    all_seq_ids = set(seq_cluster["seq_id"])

    if pfam_seq_domain.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    domain_to_seqs = pfam_seq_domain.groupby("domain_id")["seq_id"].apply(lambda s: set(s)).to_dict()
    domain_meta = (
        pfam_seq_domain.sort_values(["domain_id", "i_Evalue", "domain_score"], ascending=[True, True, False])
        .drop_duplicates("domain_id")[["domain_id", "domain_acc", "domain_desc"]]
    )

    tests = []
    selected_clusters = sorted(seq_cluster["cluster_label"].unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    for cl in selected_clusters:
        fg = set(seq_cluster.loc[seq_cluster["cluster_label"] == cl, "seq_id"])
        bg = all_seq_ids - fg
        n_fg, n_bg = len(fg), len(bg)
        if n_fg == 0 or n_bg == 0:
            continue

        for dom, dom_seqs in domain_to_seqs.items():
            a = len(fg & dom_seqs)
            if a < 2:
                continue
            c = len(bg & dom_seqs)
            b = n_fg - a
            d = n_bg - c
            odds, pval = fisher_exact([[a, b], [c, d]], alternative="greater")
            fg_frac = a / n_fg
            bg_frac = c / n_bg
            tests.append({
                "cluster_label": cl,
                "domain_id": dom,
                "fg_with_domain": a,
                "fg_without_domain": b,
                "bg_with_domain": c,
                "bg_without_domain": d,
                "fg_n": n_fg,
                "bg_n": n_bg,
                "fg_frac": fg_frac,
                "bg_frac": bg_frac,
                "odds_ratio": odds,
                "pvalue": pval,
                "log2_enrichment": math.log2((fg_frac + 1e-9) / (bg_frac + 1e-9)),
            })

    enrich = pd.DataFrame(tests)
    if enrich.empty:
        return enrich, pd.DataFrame(), pd.DataFrame()

    enrich = enrich.merge(domain_meta, on="domain_id", how="left")
    enrich["fdr_global"] = bh_fdr(enrich["pvalue"])
    enrich["fdr_within_cluster"] = np.nan
    for cl, idx in enrich.groupby("cluster_label").groups.items():
        enrich.loc[idx, "fdr_within_cluster"] = bh_fdr(enrich.loc[idx, "pvalue"])

    enrich = enrich.sort_values(["fdr_global", "pvalue", "cluster_label", "domain_id"])

    top_by_cluster = []
    for cl, sub in enrich.groupby("cluster_label"):
        top_by_cluster.append(
            sub.sort_values(["fdr_within_cluster", "pvalue", "fg_with_domain"], ascending=[True, True, False]).head(10)
        )
    top_by_cluster_df = pd.concat(top_by_cluster, ignore_index=True) if top_by_cluster else pd.DataFrame()

    domain_counts = (
        pfam_seq_domain.groupby(["cluster_label", "domain_id"])["seq_id"]
        .nunique()
        .reset_index(name="n_seq_with_domain")
    )
    cluster_n = seq_cluster.groupby("cluster_label")["seq_id"].nunique().reset_index(name="cluster_n")
    domain_counts = domain_counts.merge(cluster_n, on="cluster_label", how="left")
    domain_counts["frac_seq_with_domain"] = domain_counts["n_seq_with_domain"] / domain_counts["cluster_n"]
    domain_counts = domain_counts.merge(domain_meta, on="domain_id", how="left")
    domain_counts = domain_counts.sort_values(["cluster_label", "n_seq_with_domain"], ascending=[True, False])

    return enrich, top_by_cluster_df, domain_counts


# -----------------------------
# Plot functions
# -----------------------------

def plot_umap_logit(
    coord_df: pd.DataFrame,
    out_prefix: str,
    x_col: str = "umap_1",
    y_col: str = "umap_2",
    logit_col: str = "palmsite_logit",
    point_size: float = 8,
    alpha: float = 0.85,
) -> None:
    if x_col not in coord_df.columns or y_col not in coord_df.columns:
        return
    if logit_col not in coord_df.columns:
        if "logit" in coord_df.columns:
            logit_col = "logit"
        else:
            return

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    values = pd.to_numeric(coord_df[logit_col], errors="coerce")
    finite = values.notna()
    sc = ax.scatter(
        coord_df.loc[finite, x_col],
        coord_df.loc[finite, y_col],
        c=values.loc[finite],
        s=point_size,
        alpha=alpha,
        linewidths=0,
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("palmsite_logit")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("UMAP of backbone.span_mean colored by palmsite_logit")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        out = f"{out_prefix}.umap.palmsite_logit.{ext}"
        ensure_parent(out)
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)



def stable_cluster_style(cluster_label: str) -> Tuple[str, str]:
    """
    Return a deterministic color/marker pair for a cluster label.

    The mapping depends on the cluster label itself, not on the subset/order of
    clusters plotted. Therefore cluster "6" keeps the same style across reruns.
    """
    color_cycle = (
        list(plt.get_cmap("tab20").colors)
        + list(plt.get_cmap("tab20b").colors)
        + list(plt.get_cmap("tab20c").colors)
    )
    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h", "*", "p"]

    s = str(cluster_label)
    if s.isdigit():
        idx = int(s)
    else:
        # Stable simple hash independent of Python's randomized hash seed.
        idx = sum((i + 1) * ord(ch) for i, ch in enumerate(s))

    color = color_cycle[idx % len(color_cycle)]
    marker = markers[(idx // len(color_cycle)) % len(markers)]
    return color, marker


def write_cluster_style_map(cluster_labels: Sequence[str], out_path: str | Path) -> None:
    rows = []
    seen = set()
    for c in sorted([str(x) for x in cluster_labels], key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        if c in seen:
            continue
        seen.add(c)
        color, marker = stable_cluster_style(c)
        # Convert RGB tuple to hex for reproducibility/readability.
        rgb = tuple(int(round(float(v) * 255)) for v in color[:3])
        hex_color = "#{:02x}{:02x}{:02x}".format(*rgb)
        rows.append({
            "cluster_label": c,
            "color_hex": hex_color,
            "marker": marker,
        })
    ensure_parent(out_path)
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def plot_umap_clusters(
    coord_df: pd.DataFrame,
    out_prefix: str,
    x_col: str = "umap_1",
    y_col: str = "umap_2",
    cluster_col: str = "cluster_label",
    point_size: float = 8,
    alpha: float = 0.85,
    max_legend_clusters: int = 20,
    selected_clusters: Optional[Sequence[str]] = None,
) -> None:
    if x_col not in coord_df.columns or y_col not in coord_df.columns or cluster_col not in coord_df.columns:
        return

    def _sort_key(v: str):
        s = str(v)
        return (0, int(s)) if s.isdigit() else (1, s)

    plot_df = coord_df.copy()
    plot_df[cluster_col] = plot_df[cluster_col].fillna("Unassigned").astype(str)

    fig, ax = plt.subplots(figsize=(7.8, 6.4))

    # If borderline clusters are provided, always highlight all of them explicitly,
    # keep Unassigned as a pale background layer, and collapse the rest into "Other".
    if selected_clusters is not None:
        selected_order = []
        seen = set()
        for c in sorted([str(x) for x in selected_clusters], key=_sort_key):
            if c not in seen:
                selected_order.append(c)
                seen.add(c)
        selected_set = set(selected_order)

        plot_df["_plot_group"] = "Other"
        plot_df.loc[plot_df[cluster_col] == "Unassigned", "_plot_group"] = "Unassigned"
        plot_df.loc[plot_df[cluster_col].isin(selected_set), "_plot_group"] = plot_df.loc[
            plot_df[cluster_col].isin(selected_set), cluster_col
        ]

        other_mask = plot_df["_plot_group"] == "Other"
        if other_mask.any():
            ax.scatter(
                plot_df.loc[other_mask, x_col],
                plot_df.loc[other_mask, y_col],
                s=max(1.0, point_size * 0.70),
                alpha=0.18,
                linewidths=0,
                color="#d9d9d9",
                label=f"Other (n={int(other_mask.sum())})",
                zorder=1,
            )

        unassigned_mask = plot_df["_plot_group"] == "Unassigned"
        if unassigned_mask.any():
            ax.scatter(
                plot_df.loc[unassigned_mask, x_col],
                plot_df.loc[unassigned_mask, y_col],
                s=max(1.0, point_size * 0.85),
                alpha=0.28,
                linewidths=0,
                color="#9ecae1",
                label=f"Unassigned (n={int(unassigned_mask.sum())})",
                zorder=2,
            )
            ax.text(
                float(plot_df.loc[unassigned_mask, x_col].mean()),
                float(plot_df.loc[unassigned_mask, y_col].mean()),
                "Unassigned",
                fontsize=10,
                ha="center",
                va="center",
                zorder=5,
            )

        # Deterministic style mapping: cluster 6, for example, always gets
        # the same color/marker regardless of which other clusters are selected.
        write_cluster_style_map(selected_order, f"{out_prefix}.cluster_label.style_map.tsv")

        for cat in selected_order:
            mask = plot_df["_plot_group"] == cat
            if not mask.any():
                continue
            color, marker = stable_cluster_style(cat)
            ax.scatter(
                plot_df.loc[mask, x_col],
                plot_df.loc[mask, y_col],
                s=point_size * 1.15,
                alpha=0.92,
                linewidths=0.15,
                edgecolors="none",
                color=color,
                marker=marker,
                label=f"{cat} (n={int(mask.sum())})",
                zorder=3,
            )
            ax.text(
                float(plot_df.loc[mask, x_col].mean()),
                float(plot_df.loc[mask, y_col].mean()),
                str(cat),
                fontsize=9,
                ha="center",
                va="center",
                zorder=6,
            )
    else:
        counts = plot_df[cluster_col].value_counts()
        top = counts.head(max_legend_clusters).index.tolist()
        plot_df["_cluster_plot"] = plot_df[cluster_col].where(plot_df[cluster_col].isin(top), "Other")
        categories = top + (["Other"] if "Other" in set(plot_df["_cluster_plot"]) else [])
        cmap = plt.get_cmap("tab20", max(1, min(20, len(categories))))

        for i, cat in enumerate(categories):
            mask = plot_df["_cluster_plot"] == cat
            ax.scatter(
                plot_df.loc[mask, x_col],
                plot_df.loc[mask, y_col],
                s=point_size,
                alpha=alpha,
                linewidths=0,
                color=cmap(i % 20),
                label=f"{cat} (n={int(mask.sum())})",
            )
            if cat != "Other" and mask.sum() > 0:
                ax.text(
                    float(plot_df.loc[mask, x_col].mean()),
                    float(plot_df.loc[mask, y_col].mean()),
                    str(cat),
                    fontsize=9,
                    ha="center",
                    va="center",
                )

    ax.legend(
        title=cluster_col,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        markerscale=1.2,
    )
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("UMAP of backbone.span_mean colored by cluster_label")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        out = f"{out_prefix}.umap.cluster_label.{ext}"
        ensure_parent(out)
        fig.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_cluster_logit_boxplot(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    out_prefix: str,
    shade_borderline: bool = True,
) -> None:
    order = summary.sort_values(["median_logit", "cluster_label"], ascending=[True, True])["cluster_label"].astype(str).tolist()
    data = [df.loc[df["cluster_label"].astype(str) == cl, "logit"].dropna().values for cl in order]
    labels = [f"{cl} (n={len(vals)})" for cl, vals in zip(order, data)]

    fig_h = max(10, 0.32 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.boxplot(data, vert=False, tick_labels=labels, showfliers=False, widths=0.7)

    p50 = logit_from_p(0.5)
    p90 = logit_from_p(0.9)
    p95 = logit_from_p(0.95)
    p99 = logit_from_p(0.99)

    if shade_borderline:
        ax.axvspan(p50, p95, alpha=0.12)

    for p, x in [(0.50, p50), (0.90, p90), (0.95, p95), (0.99, p99)]:
        ax.axvline(x, linestyle="--", linewidth=1)
        ax.text(x, 1.01, f"p={p:g}", rotation=90, ha="center", va="bottom", transform=ax.get_xaxis_transform())

    ax.set_xlabel("PalmSite logit")
    ax.set_ylabel("Cluster label")
    ax.set_title("PalmSite logit by graph cluster (sorted by median logit)")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        out = f"{out_prefix}.cluster_logit_boxplot_sorted_by_median_shaded.{ext}"
        ensure_parent(out)
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_borderline_fraction(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    out_prefix: str,
    lower_p: float,
    upper_p: float,
    min_frac_borderline: float,
    min_cluster_size: int,
) -> None:
    s = summary.sort_values(["frac_borderline", "median_logit"], ascending=[False, True]).copy()
    selected_set = set(selected["cluster_label"].astype(str))
    colors = ["tab:red" if str(c) in selected_set else "lightgray" for c in s["cluster_label"]]

    fig_h = max(6, 0.22 * len(s) + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.barh(range(len(s)), s["frac_borderline"], color=colors)
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s["cluster_label"].astype(str).tolist())
    ax.invert_yaxis()
    ax.axvline(min_frac_borderline, linestyle="--", linewidth=1)
    ax.set_xlabel(f"Fraction with {lower_p:.2f} ≤ P ≤ {upper_p:.2f}")
    ax.set_ylabel("Cluster label")
    ax.set_title(f"Borderline fraction by cluster (selected in red; n ≥ {min_cluster_size})")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        out = f"{out_prefix}.cluster_borderline_fraction.{ext}"
        ensure_parent(out)
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_borderline_bubble(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    out_prefix: str,
    lower_p: float,
    upper_p: float,
    min_frac_borderline: float,
) -> None:
    s = summary.copy()
    selected_set = set(selected["cluster_label"].astype(str))
    s["is_selected"] = s["cluster_label"].astype(str).isin(selected_set)

    var = s["var_logit"].fillna(0)
    sizes = 80 + 420 * (var / var.max() if var.max() > 0 else 1)

    fig, ax = plt.subplots(figsize=(10, 8))
    bg = s[~s["is_selected"]]
    fg = s[s["is_selected"]]

    ax.scatter(bg["median_logit"], bg["frac_borderline"], s=sizes[~s["is_selected"]],
               c="lightgray", alpha=0.6, edgecolors="none", label="Other clusters")
    ax.scatter(fg["median_logit"], fg["frac_borderline"], s=sizes[s["is_selected"]],
               c="tab:red", alpha=0.75, edgecolors="black", linewidths=0.4,
               label="Selected borderline clusters")

    for _, r in fg.iterrows():
        ax.text(r["median_logit"], r["frac_borderline"], str(r["cluster_label"]),
                fontsize=9, ha="center", va="center")

    ax.axhline(min_frac_borderline, linestyle="--", linewidth=1, color="black")
    for x in [logit_from_p(0.5), logit_from_p(0.9), logit_from_p(0.95)]:
        ax.axvline(x, linestyle="--", linewidth=1, color="gray")

    ax.set_xlabel("Median PalmSite logit")
    ax.set_ylabel(f"Fraction with {lower_p:.2f} ≤ P ≤ {upper_p:.2f}")
    ax.set_title("Cluster selection bubble plot\n(circle size = variance of logit)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        out = f"{out_prefix}.cluster_borderline_bubble.{ext}"
        ensure_parent(out)
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_pfam_annotation_coverage(cluster_annotation_summary: pd.DataFrame, out_prefix: str) -> None:
    cov = cluster_annotation_summary.sort_values("frac_pfam_annotated", ascending=True)
    fig_h = max(6, 0.35 * len(cov) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(cov["cluster_label"].astype(str), cov["frac_pfam_annotated"])
    ax.set_xlabel("Fraction with at least one Pfam hit")
    ax.set_ylabel("Borderline cluster")
    ax.set_title("Pfam annotation coverage by selected borderline cluster")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        out = f"{out_prefix}.pfam_annotation_coverage_by_cluster.{ext}"
        ensure_parent(out)
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_top_domain_dotplot(enrich: pd.DataFrame, out_prefix: str, max_pairs: int = 60) -> None:
    if enrich.empty:
        return

    sig = enrich[(enrich["fdr_within_cluster"] <= 0.1) & (enrich["fg_with_domain"] >= 3)].copy()
    if sig.empty:
        sig = enrich.sort_values(["pvalue", "fg_with_domain"], ascending=[True, False]).head(40).copy()
    else:
        sig = sig.sort_values(["fdr_within_cluster", "fg_with_domain"], ascending=[True, False]).head(max_pairs)

    sig["domain_label"] = sig["domain_id"].astype(str)
    cluster_order = sorted(sig["cluster_label"].astype(str).unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    domain_order = sig.groupby("domain_label")["fg_with_domain"].sum().sort_values(ascending=True).index.tolist()

    xmap = {c: i for i, c in enumerate(cluster_order)}
    ymap = {d: i for i, d in enumerate(domain_order)}
    sig["x"] = sig["cluster_label"].astype(str).map(xmap)
    sig["y"] = sig["domain_label"].map(ymap)

    sizes = 30 + 220 * (sig["fg_frac"] / max(sig["fg_frac"].max(), 1e-9))
    colors = -np.log10(sig["fdr_within_cluster"].clip(lower=1e-300))

    fig_w = max(8, 0.35 * len(cluster_order) + 4)
    fig_h = max(6, 0.28 * len(domain_order) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sc = ax.scatter(sig["x"], sig["y"], s=sizes, c=colors, alpha=0.8)
    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels(cluster_order, rotation=90)
    ax.set_yticks(range(len(domain_order)))
    ax.set_yticklabels(domain_order)
    ax.set_xlabel("Borderline cluster")
    ax.set_ylabel("Pfam domain")
    ax.set_title("Enriched Pfam domains in selected borderline clusters")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("-log10 within-cluster FDR")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        out = f"{out_prefix}.top_domain_enrichment_dotplot.{ext}"
        ensure_parent(out)
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def plot_all_selected_domain_dotplot(
    enrich: pd.DataFrame,
    selected: pd.DataFrame,
    out_prefix: str,
) -> None:
    if enrich.empty:
        return

    cluster_order = sorted(selected["cluster_label"].astype(str).unique(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    sig = enrich[(enrich["fdr_within_cluster"] <= 0.1) & (enrich["fg_with_domain"] >= 3)].copy()
    if sig.empty:
        return

    domain_summary = (
        sig.groupby("domain_id")
        .agg(n_clusters=("cluster_label", "nunique"), total_fg=("fg_with_domain", "sum"), best_fdr=("fdr_within_cluster", "min"))
        .reset_index()
        .sort_values(["n_clusters", "total_fg", "best_fdr", "domain_id"], ascending=[False, False, True, True])
    )
    domain_order = domain_summary["domain_id"].tolist()
    y_display = domain_order[::-1]

    xmap = {c: i for i, c in enumerate(cluster_order)}
    ymap = {d: i for i, d in enumerate(y_display)}
    sig["x"] = sig["cluster_label"].astype(str).map(xmap)
    sig["y"] = sig["domain_id"].astype(str).map(ymap)

    sizes = 20 + 280 * (sig["fg_frac"] / max(sig["fg_frac"].max(), 1e-9))
    colors = -np.log10(sig["fdr_within_cluster"].clip(lower=1e-300))

    fig_w = max(12, 0.42 * len(cluster_order) + 6)
    fig_h = max(8, 0.28 * len(domain_order) + 3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sc = ax.scatter(sig["x"], sig["y"], s=sizes, c=colors, alpha=0.8)

    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels(cluster_order, rotation=90)
    ax.set_yticks(range(len(y_display)))
    ax.set_yticklabels(y_display)
    ax.set_xlabel("Selected borderline cluster")
    ax.set_ylabel("Pfam domain")
    ax.set_title("Enriched Pfam domains across all selected borderline clusters")
    ax.grid(True, alpha=0.2)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("-log10 within-cluster FDR")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        out = f"{out_prefix}.all_selected_clusters_domain_dotplot.{ext}"
        ensure_parent(out)
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)

    presence = pd.DataFrame({"cluster_label": cluster_order})
    sig_counts = sig.groupby(sig["cluster_label"].astype(str)).size().rename("n_enriched_domains").reset_index()
    sig_counts.columns = ["cluster_label", "n_enriched_domains"]
    presence = presence.merge(sig_counts, on="cluster_label", how="left")
    presence["n_enriched_domains"] = presence["n_enriched_domains"].fillna(0).astype(int)
    presence.to_csv(f"{out_prefix}.all_selected_clusters_domain_presence.tsv", sep="\t", index=False)


def plot_cluster_size_vs_annotation(cluster_annotation_summary: pd.DataFrame, out_prefix: str) -> None:
    df = cluster_annotation_summary.copy()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(df["n_sequences"], df["frac_pfam_annotated"], alpha=0.8)
    for _, r in df.iterrows():
        ax.text(r["n_sequences"], r["frac_pfam_annotated"], str(r["cluster_label"]), fontsize=9, ha="left", va="bottom")
    ax.set_xlabel("Cluster size (number of sequences)")
    ax.set_ylabel("Pfam annotated fraction")
    ax.set_title("Borderline cluster size vs Pfam annotated fraction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        out = f"{out_prefix}.cluster_size_vs_pfam_annotated_fraction.{ext}"
        ensure_parent(out)
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def classify_rt_enriched_clusters(enrich: pd.DataFrame) -> set[str]:
    if enrich.empty:
        return set()
    rt_regex = re.compile(
        r"RVT|RT_RNaseH|RNaseH|\brve\b|gag_pre-integrs|Pao_retrotransp|SH3_retrovirus",
        flags=re.IGNORECASE,
    )
    e = enrich.copy()
    e["domain_id"] = e["domain_id"].astype(str)
    e["domain_desc"] = e["domain_desc"].fillna("").astype(str)
    rt_hits = e[e["domain_id"].str.contains(rt_regex, na=False) | e["domain_desc"].str.contains(rt_regex, na=False)]
    rt_sig = rt_hits[(rt_hits["fdr_within_cluster"] <= 0.1) & (rt_hits["fg_with_domain"] >= 3)]
    return set(rt_sig["cluster_label"].astype(str))


def plot_median_logit_vs_annotation(
    cluster_annotation_summary: pd.DataFrame,
    enrich: pd.DataFrame,
    out_prefix: str,
    highlight_rt: bool = True,
) -> None:
    df = cluster_annotation_summary.copy()
    df["cluster_label"] = df["cluster_label"].astype(str)

    nmin, nmax = df["n_sequences"].min(), df["n_sequences"].max()
    if nmax > nmin:
        df["marker_size"] = 50 + 350 * (df["n_sequences"] - nmin) / (nmax - nmin)
    else:
        df["marker_size"] = 150.0

    fig, ax = plt.subplots(figsize=(9, 7))

    if highlight_rt:
        rt_clusters = classify_rt_enriched_clusters(enrich)
        df["is_rt_enriched"] = df["cluster_label"].isin(rt_clusters)
        other = df[~df["is_rt_enriched"]]
        rt = df[df["is_rt_enriched"]]
        ax.scatter(other["median_logit"], other["frac_pfam_annotated"], s=other["marker_size"], alpha=0.7, label="Other borderline clusters")
        ax.scatter(rt["median_logit"], rt["frac_pfam_annotated"], s=rt["marker_size"], alpha=0.9, marker="s", label="RT-enriched clusters")
    else:
        ax.scatter(df["median_logit"], df["frac_pfam_annotated"], s=df["marker_size"], alpha=0.7)

    for _, r in df.iterrows():
        ax.text(r["median_logit"], r["frac_pfam_annotated"], str(r["cluster_label"]), fontsize=9, ha="center", va="bottom")

    ax.set_xlabel("Median PalmSite logit")
    ax.set_ylabel("Pfam annotated fraction")
    title = "Borderline clusters: median logit vs Pfam annotated fraction\n"
    title += "RT-enriched clusters highlighted; point size = cluster size" if highlight_rt else "point size = cluster size"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if highlight_rt:
        ax.legend()
    fig.tight_layout()

    suffix = "median_logit_vs_pfam_annotated_fraction_highlight_rt" if highlight_rt else "median_logit_vs_pfam_annotated_fraction"
    for ext in ["png", "pdf"]:
        out = f"{out_prefix}.{suffix}.{ext}"
        ensure_parent(out)
        fig.savefig(out, dpi=200 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gff3", required=True)
    ap.add_argument("--cluster-file", required=True)
    ap.add_argument("--pfam-domtblout", required=True)
    ap.add_argument("--coordinates-tsv", default=None, help="Optional coordinate table with umap_1/umap_2 and labels.")
    ap.add_argument("--out-prefix", required=True)

    ap.add_argument("--lower-p", type=float, default=0.5)
    ap.add_argument("--upper-p", type=float, default=0.95)
    ap.add_argument("--clean-positive-p", type=float, default=0.99)
    ap.add_argument("--clean-negative-logit", type=float, default=-6.0)

    ap.add_argument("--min-frac-borderline", type=float, default=0.4)
    ap.add_argument("--min-cluster-size", type=int, default=100)
    ap.add_argument("--no-require-median-in-window", action="store_true")

    ap.add_argument("--pfam-i-evalue-cutoff", type=float, default=1e-3)
    ap.add_argument("--pfam-hmm-coverage-cutoff", type=float, default=0.20)

    args = ap.parse_args()

    out_prefix = args.out_prefix
    ensure_parent(f"{out_prefix}.dummy")

    # Core data
    gff_df = load_palmsite_gff(args.gff3)
    cluster_df = load_graph_clusters(args.cluster_file)
    df = merge_gff_clusters(gff_df, cluster_df)
    df = add_borderline_flags(
        df,
        lower_p=args.lower_p,
        upper_p=args.upper_p,
        clean_positive_p=args.clean_positive_p,
        clean_negative_logit=args.clean_negative_logit,
    )

    # Cluster summary and selected borderline clusters
    summary = summarize_clusters(df)
    selected = select_borderline_clusters(
        summary,
        lower_p=args.lower_p,
        upper_p=args.upper_p,
        min_frac_borderline=args.min_frac_borderline,
        min_cluster_size=args.min_cluster_size,
        require_median_in_window=(not args.no_require_median_in_window),
    )

    summary.to_csv(f"{out_prefix}.cluster_summary.tsv", sep="\t", index=False)
    selected.to_csv(f"{out_prefix}.selected_borderline_clusters.tsv", sep="\t", index=False)

    selected_seq = df[df["cluster_label"].astype(str).isin(set(selected["cluster_label"].astype(str)))].copy()
    selected_seq.to_csv(f"{out_prefix}.selected_borderline_cluster_sequences.tsv", sep="\t", index=False)

    # Core cluster plots
    plot_cluster_logit_boxplot(df, summary, out_prefix=out_prefix, shade_borderline=True)
    plot_borderline_fraction(summary, selected, out_prefix=out_prefix, lower_p=args.lower_p, upper_p=args.upper_p,
                             min_frac_borderline=args.min_frac_borderline, min_cluster_size=args.min_cluster_size)
    plot_borderline_bubble(summary, selected, out_prefix=out_prefix, lower_p=args.lower_p, upper_p=args.upper_p,
                           min_frac_borderline=args.min_frac_borderline)

    # Optional UMAP panels
    if args.coordinates_tsv:
        coords = pd.read_csv(args.coordinates_tsv, sep="\t")
        # Ensure cluster labels if coordinates table does not have them
        if "cluster_label" not in coords.columns and "seq_id" in coords.columns:
            coords["seq_id"] = coords["seq_id"].map(normalize_id)
            coords = coords.merge(cluster_df[["seq_id", "cluster_label"]], on="seq_id", how="left")
        if "palmsite_logit" not in coords.columns and "logit" not in coords.columns and "seq_id" in coords.columns:
            coords = coords.merge(gff_df[["seq_id", "logit"]], on="seq_id", how="left")
            coords = coords.rename(columns={"logit": "palmsite_logit"})
        plot_umap_logit(coords, out_prefix=out_prefix)
        plot_umap_clusters(
            coords,
            out_prefix=out_prefix,
            selected_clusters=selected["cluster_label"].astype(str).tolist(),
        )

    # Pfam
    selected_ids = set(selected_seq["seq_id"].map(normalize_id))
    pfam = load_domtblout(args.pfam_domtblout)
    pfam_seq_domain, seq_side, domain_side, filter_note = prepare_pfam_hits(
        pfam,
        selected_ids=selected_ids,
        i_evalue_cutoff=args.pfam_i_evalue_cutoff,
        hmm_coverage_cutoff=args.pfam_hmm_coverage_cutoff,
    )
    if not pfam_seq_domain.empty:
        pfam_seq_domain = pfam_seq_domain.merge(
            selected_seq[["seq_id", "cluster_label"]].drop_duplicates(),
            on="seq_id",
            how="left",
        )

    pfam_seq_domain.to_csv(f"{out_prefix}.pfam_filtered_seq_domain_hits.tsv", sep="\t", index=False)

    # Annotation coverage
    annot_status = selected_seq[["seq_id", "cluster_label"]].drop_duplicates().copy()
    annotated_ids = set(pfam_seq_domain["seq_id"]) if not pfam_seq_domain.empty else set()
    annot_status["pfam_annotated"] = annot_status["seq_id"].isin(annotated_ids)
    cluster_annotation_summary = (
        annot_status.groupby("cluster_label")
        .agg(
            n_sequences=("seq_id", "size"),
            n_pfam_annotated=("pfam_annotated", "sum"),
            frac_pfam_annotated=("pfam_annotated", "mean"),
        )
        .reset_index()
    )
    cluster_annotation_summary = cluster_annotation_summary.merge(
        selected[["cluster_label", "median_P", "median_logit", "frac_borderline", "var_logit"]],
        on="cluster_label",
        how="left",
    )
    cluster_annotation_summary.to_csv(f"{out_prefix}.cluster_pfam_annotation_summary.tsv", sep="\t", index=False)

    # Enrichment
    enrich, top_by_cluster, domain_counts = run_domain_enrichment(selected_seq, pfam_seq_domain)
    enrich.to_csv(f"{out_prefix}.domain_enrichment_all_tests.tsv", sep="\t", index=False)
    top_by_cluster.to_csv(f"{out_prefix}.top_enriched_domains_by_cluster.tsv", sep="\t", index=False)
    domain_counts.to_csv(f"{out_prefix}.domain_counts_by_cluster.tsv", sep="\t", index=False)

    # Pfam plots
    plot_pfam_annotation_coverage(cluster_annotation_summary, out_prefix=out_prefix)
    plot_top_domain_dotplot(enrich, out_prefix=out_prefix)
    plot_all_selected_domain_dotplot(enrich, selected, out_prefix=out_prefix)
    plot_cluster_size_vs_annotation(cluster_annotation_summary, out_prefix=out_prefix)
    plot_median_logit_vs_annotation(cluster_annotation_summary, enrich, out_prefix=out_prefix, highlight_rt=True)

    # Report
    with open(f"{out_prefix}.report.txt", "w", encoding="utf-8") as out:
        print("PalmSite borderline cluster figure replication report", file=out)
        print(f"Input PalmSite records: {len(gff_df):,}", file=out)
        print(f"Input clusters: {summary.shape[0]:,}", file=out)
        print(f"Selected borderline clusters: {selected.shape[0]:,}", file=out)
        print(f"Borderline window: {args.lower_p} <= P <= {args.upper_p}", file=out)
        print(f"Min fraction borderline: {args.min_frac_borderline}", file=out)
        print(f"Min cluster size: {args.min_cluster_size}", file=out)
        print(f"Require median in window: {not args.no_require_median_in_window}", file=out)
        print("", file=out)
        print(f"Pfam sequence side inferred as: {seq_side}", file=out)
        print(f"Pfam domain side inferred as: {domain_side}", file=out)
        print(filter_note, file=out)
        print(f"Selected cluster sequences: {len(selected_ids):,}", file=out)
        print(f"Pfam seq-domain hits to selected cluster sequences: {len(pfam_seq_domain):,}", file=out)
        print(f"Pfam annotated sequences: {len(annotated_ids):,} / {len(selected_ids):,}", file=out)
        if not enrich.empty:
            print(f"Enrichment tests: {len(enrich):,}", file=out)
            print(f"Global FDR <= 0.1: {(enrich['fdr_global'] <= 0.1).sum():,}", file=out)
            print(f"Within-cluster FDR <= 0.1: {(enrich['fdr_within_cluster'] <= 0.1).sum():,}", file=out)

    print(f"Finished. Outputs written with prefix: {out_prefix}")


if __name__ == "__main__":
    main()

