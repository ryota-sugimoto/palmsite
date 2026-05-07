#!/usr/bin/env python3
"""
Visualize PalmSite sliding-window mutation results with optional annotation tracks.

Layout note: this version fixes heatmap annotation-track alignment by using an
explicit colorbar column and constrained_layout rather than tight_layout.

Adds:
- PalmSite predicted span (from original GFF or manifest)
- palm_annot motif positions A/B/C
- InterProScan domain annotations
- PDF output
- coordinate origin control (0-based or 1-based plotting)

Example:
  python visualize_palmsite_window_noise_annotated.py \
    --manifest noised_windows/palmsite_window_noise.manifest.tsv \
    --logits-json noised_windows/palmsite_window_noise.mutants.logits.json \
    --inputs-dir inputs \
    --iprscan-tsv iprscan1.tsv --iprscan-tsv iprscan2.tsv \
    --out-dir noised_windows_viz_annot \
    --prefix palmsite_window_noise \
    --coord-origin 1
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


IPR_COLS = [
    'acc', 'md5', 'len', 'db', 'sig', 'sig_desc', 'start', 'end', 'score',
    'status', 'date', 'ipr', 'ipr_desc', 'go', 'path'
]


def parse_args():
    ap = argparse.ArgumentParser(description="Visualize PalmSite sliding-window noise results with annotations.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--logits-json", required=True)
    ap.add_argument("--inputs-dir", default=None, help="Directory containing per-sample original files such as *.gff and *.palm_annot.tsv")
    ap.add_argument("--iprscan-tsv", action="append", default=[], help="InterProScan TSV file(s). Can be supplied multiple times.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="palmsite_window_noise")
    ap.add_argument("--prob-threshold", type=float, default=0.5)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--coord-origin", type=int, choices=[0, 1], default=1,
                    help="Plot x coordinates as 0-based or 1-based positions.")
    ap.add_argument("--ipr-pad", type=int, default=300,
                    help="Show InterPro domains overlapping PalmSite span +/- this many aa. Use -1 to disable pad-based filtering and show all matched domains.")
    ap.add_argument("--ipr-keep-sources", default="Pfam,CDD,ProSiteProfiles,Gene3D,SUPERFAMILY,SMART,PANTHER",
                    help="Comma-separated InterProScan member DB sources to keep.")
    ap.add_argument("--samples", default=None, help="Optional comma-separated subset of sample_name values to plot.")
    return ap.parse_args()


def choose_best(records):
    marked = [r for r in records if bool(r.get("is_best_base_chunk"))]
    pool = marked if marked else records
    return max(pool, key=lambda r: float(r.get("P", -1e9)))


def load_best_by_base(path: str | Path) -> Dict[str, dict]:
    obj = json.load(open(path))
    grouped = {}
    for key, rec in obj.items():
        if key.startswith("_"):
            continue
        base_id = rec.get("base_id", key)
        grouped.setdefault(base_id, []).append(rec)
    return {base_id: choose_best(recs) for base_id, recs in grouped.items()}


def sem(x):
    x = pd.Series(x).dropna()
    if len(x) <= 1:
        return np.nan
    return float(x.std(ddof=1) / math.sqrt(len(x)))


def safe(s):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(s))


def parse_window(region_name: str) -> Tuple[float, float, float]:
    m = re.fullmatch(r"window_(\d+)_(\d+)", str(region_name))
    if not m:
        return np.nan, np.nan, np.nan
    s = int(m.group(1))
    e = int(m.group(2))
    mid = (s + e) / 2.0
    return s, e, mid


def parse_gff_span(gff_path: Path) -> Optional[dict]:
    if not gff_path or not gff_path.exists():
        return None
    with open(gff_path) as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 9:
                continue
            seqid, source, feature, start, end, score, strand, phase, attrs = parts
            out = {
                'seqid': seqid,
                'start': int(start),
                'end': int(end),
                'score': score,
                'feature': feature,
                'attrs': attrs,
            }
            return out
    return None


def parse_keyval_tsv_row(path: Path) -> dict:
    if not path or not path.exists() or path.stat().st_size == 0:
        return {}
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split('\t')
            out = {'_id': fields[0]}
            for item in fields[1:]:
                if '=' in item:
                    k, v = item.split('=', 1)
                    out[k] = v
            return out
    return {}


def parse_palm_annot(path: Path) -> List[dict]:
    row = parse_keyval_tsv_row(path)
    motifs = []
    for name in ['A', 'B', 'C']:
        key = f'pos{name}'
        if key in row and row[key] not in ['', 'NA', 'nan']:
            try:
                motifs.append({'name': name, 'pos': int(float(row[key]))})
            except Exception:
                pass
    return motifs


def normalize_domain_label(row: pd.Series) -> str:
    label = row['ipr_desc'] if str(row['ipr_desc']) not in {'-', 'nan', 'None'} else row['sig_desc']
    label = str(label)
    label = re.sub(r'\s+', ' ', label).strip()
    return label


def load_iprscan(paths: List[str], keep_sources: List[str]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path, sep='\t', header=None, names=IPR_COLS, dtype=str)
        df['start'] = pd.to_numeric(df['start'], errors='coerce')
        df['end'] = pd.to_numeric(df['end'], errors='coerce')
        df['len'] = pd.to_numeric(df['len'], errors='coerce')
        if keep_sources:
            df = df[df['db'].isin(keep_sources)].copy()
        # Prefer domain-like annotations with InterPro accession and description.
        df = df[df['ipr'].fillna('-') != '-'].copy()
        df['label'] = df.apply(normalize_domain_label, axis=1)
        df = df[['acc', 'db', 'sig', 'label', 'start', 'end', 'ipr']].drop_duplicates().reset_index(drop=True)
        frames.append(df)
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame(columns=['acc', 'db', 'sig', 'label', 'start', 'end', 'ipr'])


def pack_intervals(intervals: List[Tuple[float, float]]) -> List[int]:
    """Assign lane indices greedily so overlapping intervals are stacked."""
    lanes_end = []
    lane_idx = []
    for start, end in intervals:
        placed = False
        for i, last_end in enumerate(lanes_end):
            if start > last_end:
                lanes_end[i] = end
                lane_idx.append(i)
                placed = True
                break
        if not placed:
            lanes_end.append(end)
            lane_idx.append(len(lanes_end) - 1)
    return lane_idx


def draw_annotation_track(ax, xmin: float, xmax: float, span: Optional[Tuple[float, float]], motifs: List[dict], domains: List[dict], coord_shift: int):
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(axis='x', which='both', length=0, labelbottom=False)

    lane_h = 0.8
    y0 = 0.1

    # PalmSite span strip at the top
    current_y = y0
    if span is not None:
        s, e = span
        s -= coord_shift
        e -= coord_shift
        ax.add_patch(Rectangle((s, current_y), max(1e-9, e - s), lane_h,
                               facecolor='lightgray', edgecolor='black', linewidth=0.8, alpha=0.7))
        ax.text((s + e) / 2, current_y + lane_h / 2, 'PalmSite span', ha='center', va='center', fontsize=7)
    current_y += 1.0

    # Domain rectangles in stacked lanes
    filtered_domains = []
    for d in domains:
        ds = d['start'] - coord_shift
        de = d['end'] - coord_shift
        if de < xmin or ds > xmax:
            continue
        filtered_domains.append((ds, de, d))

    filtered_domains.sort(key=lambda x: (x[0], x[1]))
    lane_assign = pack_intervals([(x[0], x[1]) for x in filtered_domains]) if filtered_domains else []
    nlanes = (max(lane_assign) + 1) if lane_assign else 0

    for (ds, de, d), lane in zip(filtered_domains, lane_assign):
        y = current_y + lane * 1.0
        # Clip the rectangle to the visible x-range. This avoids apparently
        # shifted or truncated domain boxes when the domain extends beyond the
        # plotted window.
        draw_s = max(ds, xmin)
        draw_e = min(de, xmax)
        if draw_e <= draw_s:
            continue
        ax.add_patch(Rectangle((draw_s, y), max(1e-9, draw_e - draw_s), lane_h,
                               facecolor='white', edgecolor='black', linewidth=0.8,
                               clip_on=True, zorder=2))
        # Only place label if the visible interval is reasonably wide.
        if (draw_e - draw_s) >= max(20, 0.05 * (xmax - xmin)):
            ax.text((draw_s + draw_e) / 2, y + lane_h / 2, d['label'],
                    ha='center', va='center', fontsize=6, clip_on=True, zorder=3)

    current_y += max(1, nlanes) * 1.0

    # Motif vertical lines and labels
    motif_colors = {'A': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:red'}
    motif_y0 = 0.0
    motif_y1 = current_y + 0.4
    for m in motifs:
        x = m['pos'] - coord_shift
        if xmin <= x <= xmax:
            ax.axvline(x, color=motif_colors.get(m['name'], 'black'), linewidth=1.2, linestyle='--')
            ax.text(x, motif_y1 + 0.1, m['name'], color=motif_colors.get(m['name'], 'black'),
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_ylim(-0.2, motif_y1 + 0.8)


def build_heatmap_edges(vals: List[float]) -> np.ndarray:
    vals = np.asarray(sorted(vals), dtype=float)
    if len(vals) == 1:
        step = 1.0
        return np.array([vals[0] - step / 2, vals[0] + step / 2], dtype=float)
    mids = (vals[:-1] + vals[1:]) / 2.0
    first = vals[0] - (vals[1] - vals[0]) / 2.0
    last = vals[-1] + (vals[-1] - vals[-2]) / 2.0
    return np.concatenate([[first], mids, [last]])


def get_sample_annotations(sample: str, sub: pd.DataFrame, inputs_dir: Optional[Path], ipr_df: pd.DataFrame, ipr_pad: int, coord_shift: int) -> dict:
    meta = {
        'span': None,
        'motifs': [],
        'domains': [],
        'seq_length': int(sub['window_end1'].max()),
    }

    # Span from original GFF if possible; otherwise from manifest first.
    if inputs_dir is not None:
        gff_path = inputs_dir / f'{sample}.gff'
        gff = parse_gff_span(gff_path)
        if gff is not None:
            meta['span'] = (gff['start'], gff['end'])
            seq_ids = {gff['seqid']}
        else:
            seq_ids = set()
        pa_path = inputs_dir / f'{sample}.palm_annot.tsv'
        meta['motifs'] = parse_palm_annot(pa_path)
    else:
        seq_ids = set()

    if meta['span'] is None and {'span_start1', 'span_end1'}.issubset(sub.columns):
        meta['span'] = (int(sub['span_start1'].iloc[0]), int(sub['span_end1'].iloc[0]))

    # Candidate accessions from manifest and gff.
    for col in ['fasta_id', 'base_id', 'chunk_id']:
        if col in sub.columns:
            seq_ids.update(map(str, sub[col].dropna().unique().tolist()))
    # simplify chunk ids to base ids
    extra_ids = set()
    for sid in seq_ids:
        if '|chunk_' in sid:
            extra_ids.add(sid.split('|chunk_')[0])
    seq_ids.update(extra_ids)

    if not ipr_df.empty and seq_ids:
        matched = ipr_df[ipr_df['acc'].isin(seq_ids)].copy()
        # If no exact match, try containment match both ways.
        if matched.empty:
            sid_list = list(seq_ids)
            matched = ipr_df[ipr_df['acc'].apply(lambda x: any((x in sid) or (sid in x) for sid in sid_list))].copy()
        if not matched.empty:
            if meta['span'] is not None and ipr_pad >= 0:
                s, e = meta['span']
                lo = max(1, s - ipr_pad)
                hi = e + ipr_pad
                matched = matched[(matched['end'] >= lo) & (matched['start'] <= hi)].copy()
            # Deduplicate by label + interval, preferring one source.
            matched = matched.sort_values(['start', 'end', 'db', 'label']).drop_duplicates(subset=['label', 'start', 'end'])
            meta['domains'] = matched[['label', 'start', 'end', 'db', 'ipr']].to_dict(orient='records')

    return meta


def plot_sample_line(sub: pd.DataFrame, sample: str, meta: dict, out_path: Path, coord_shift: int):
    xmin = float(sub['window_start1'].min() - coord_shift)
    xmax = float(sub['window_end1'].max() - coord_shift)
    rates = sorted(sub['rate'].dropna().unique())

    fig = plt.figure(figsize=(11.5, 6.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 4.75], hspace=0.04)
    ax_anno = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_anno)

    draw_annotation_track(ax_anno, xmin, xmax, meta['span'], meta['motifs'], meta['domains'], coord_shift)

    for rate in rates:
        t = sub[sub['rate'] == rate].sort_values('window_mid1')
        x = t['window_mid1'] - coord_shift
        ax.errorbar(x, t['mean_delta_calibrated_logit'], yerr=t['se_delta_calibrated_logit'],
                    marker='o', markersize=2.2, linewidth=1.0, capsize=1.8, label=f'rate={rate:g}')
    if meta['span'] is not None:
        s, e = meta['span']
        ax.axvspan(s - coord_shift, e - coord_shift, alpha=0.12)
    for m in meta['motifs']:
        ax.axvline(m['pos'] - coord_shift, color={'A':'tab:blue','B':'tab:orange','C':'tab:red'}.get(m['name'],'black'),
                   linestyle='--', linewidth=0.9, alpha=0.8)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(f'Amino-acid position ({"0-based" if coord_shift == 1 else "1-based"})')
    ax.set_ylabel('Mean Δ calibrated logit')
    ax.set_title(f'{sample}: window mutation effect on PalmSite calibrated logit')
    ax.legend(ncol=min(len(rates), 4), fontsize=8)
    # constrained_layout handles the annotation track cleanly; do not call tight_layout.
    fig.savefig(out_path)
    plt.close(fig)


def plot_sample_positive(sub: pd.DataFrame, sample: str, meta: dict, out_path: Path, coord_shift: int):
    xmin = float(sub['window_start1'].min() - coord_shift)
    xmax = float(sub['window_end1'].max() - coord_shift)
    rates = sorted(sub['rate'].dropna().unique())

    fig = plt.figure(figsize=(11.5, 6.3), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 4.75], hspace=0.04)
    ax_anno = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_anno)

    draw_annotation_track(ax_anno, xmin, xmax, meta['span'], meta['motifs'], meta['domains'], coord_shift)

    for rate in rates:
        t = sub[sub['rate'] == rate].sort_values('window_mid1')
        x = t['window_mid1'] - coord_shift
        ax.errorbar(x, t['positive_fraction'], yerr=t['se_positive_fraction'],
                    marker='o', markersize=2.2, linewidth=1.0, capsize=1.8, label=f'rate={rate:g}')
    if meta['span'] is not None:
        s, e = meta['span']
        ax.axvspan(s - coord_shift, e - coord_shift, alpha=0.12)
    for m in meta['motifs']:
        ax.axvline(m['pos'] - coord_shift, color={'A':'tab:blue','B':'tab:orange','C':'tab:red'}.get(m['name'],'black'),
                   linestyle='--', linewidth=0.9, alpha=0.8)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(f'Amino-acid position ({"0-based" if coord_shift == 1 else "1-based"})')
    ax.set_ylabel('Positive fraction')
    ax.set_title(f'{sample}: positive fraction after window mutation')
    ax.legend(ncol=min(len(rates), 4), fontsize=8)
    # constrained_layout handles the annotation track cleanly; do not call tight_layout.
    fig.savefig(out_path)
    plt.close(fig)


def plot_sample_heatmap(sub: pd.DataFrame, sample: str, meta: dict, out_path: Path, value_col: str, cbar_label: str, title: str, coord_shift: int):
    rates = sorted(sub['rate'].dropna().unique())
    mids = sorted(sub['window_mid1'].dropna().unique())
    if len(mids) == 0 or len(rates) == 0:
        return
    pivot = sub.pivot(index='rate', columns='window_mid1', values=value_col).reindex(index=rates, columns=mids)
    arr = pivot.values.astype(float)

    x_vals = np.array(mids, dtype=float) - coord_shift
    x_edges = build_heatmap_edges(x_vals.tolist())
    y_edges = build_heatmap_edges(np.array(rates, dtype=float).tolist())

    xmin = float(sub['window_start1'].min() - coord_shift)
    xmax = float(sub['window_end1'].max() - coord_shift)

    # Use an explicit colorbar column so the annotation axis and heatmap axis
    # keep the exact same horizontal span. Attaching a colorbar only to the
    # heatmap axis shrinks that axis and makes the annotation boxes appear
    # horizontally misregistered.
    fig = plt.figure(figsize=(11.5, 5.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 4.35], width_ratios=[1.0, 0.035],
                          hspace=0.04, wspace=0.03)
    ax_anno = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[1, 0], sharex=ax_anno)
    cax = fig.add_subplot(gs[1, 1])
    ax_cbtop = fig.add_subplot(gs[0, 1])
    ax_cbtop.axis('off')

    draw_annotation_track(ax_anno, xmin, xmax, meta['span'], meta['motifs'], meta['domains'], coord_shift)

    mesh = ax.pcolormesh(x_edges, y_edges, arr, shading='auto')
    if meta['span'] is not None:
        s, e = meta['span']
        ax.axvspan(s - coord_shift, e - coord_shift, alpha=0.08)
    for m in meta['motifs']:
        ax.axvline(m['pos'] - coord_shift, color={'A':'tab:blue','B':'tab:orange','C':'tab:red'}.get(m['name'],'black'),
                   linestyle='--', linewidth=0.9, alpha=0.8)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(cbar_label)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(f'Amino-acid position ({"0-based" if coord_shift == 1 else "1-based"})')
    ax.set_ylabel('Mutation rate')
    ax.set_title(title)
    # constrained_layout handles the annotation track + explicit colorbar column.
    fig.savefig(out_path)
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir = Path(args.inputs_dir) if args.inputs_dir else None
    keep_sources = [x.strip() for x in args.ipr_keep_sources.split(',') if x.strip()]
    ipr_df = load_iprscan(args.iprscan_tsv, keep_sources)

    manifest = pd.read_csv(args.manifest, sep='\t', dtype=str)
    manifest = manifest[manifest['target'] == 'sliding_window'].copy()
    num_cols = ['rate', 'replicate', 'n_mutated', 'span_start1', 'span_end1', 'original_P', 'original_logit', 'original_calibrated_logit', 'temperature', 'seed']
    for col in num_cols:
        if col in manifest.columns:
            manifest[col] = pd.to_numeric(manifest[col], errors='coerce')

    starts, ends, mids = zip(*[parse_window(x) for x in manifest['region_name']])
    manifest['window_start1'] = starts
    manifest['window_end1'] = ends
    manifest['window_mid1'] = mids

    best_log = load_best_by_base(args.logits_json)
    log_df = pd.DataFrame([
        {
            'mutant_id': mutant_id,
            'P': float(rec.get('P', np.nan)),
            'logit': float(rec.get('logit', np.nan)),
            'calibrated_logit': float(rec.get('calibrated_logit', np.nan)),
        }
        for mutant_id, rec in best_log.items()
    ])

    df = manifest.merge(log_df, on='mutant_id', how='left')
    if args.samples:
        keep = {x.strip() for x in args.samples.split(',') if x.strip()}
        df = df[df['sample_name'].isin(keep)].copy()
    df['delta_calibrated_logit'] = df['calibrated_logit'] - df['original_calibrated_logit']
    df['delta_logit'] = df['logit'] - df['original_logit']
    df['delta_P'] = df['P'] - df['original_P']
    df['positive'] = df['P'] >= args.prob_threshold
    df['drop_calibrated_logit'] = -df['delta_calibrated_logit']

    merged_path = out_dir / f'{args.prefix}.merged_metrics.tsv'
    df.to_csv(merged_path, sep='\t', index=False)

    group_cols = ['sample_name', 'rate', 'window_start1', 'window_end1', 'window_mid1']
    summary = (
        df.groupby(group_cols, as_index=False)
          .agg(
              n=('mutant_id', 'size'),
              mean_delta_calibrated_logit=('delta_calibrated_logit', 'mean'),
              sd_delta_calibrated_logit=('delta_calibrated_logit', 'std'),
              mean_drop_calibrated_logit=('drop_calibrated_logit', 'mean'),
              mean_delta_P=('delta_P', 'mean'),
              sd_delta_P=('delta_P', 'std'),
              positive_fraction=('positive', 'mean'),
              mean_P=('P', 'mean'),
              original_P=('original_P', 'first'),
              span_start1=('span_start1', 'first'),
              span_end1=('span_end1', 'first'),
          )
    )
    se_rows = []
    for keys, sub in df.groupby(group_cols):
        sample_name, rate, ws, we, wm = keys
        se_rows.append({
            'sample_name': sample_name,
            'rate': rate,
            'window_start1': ws,
            'window_end1': we,
            'window_mid1': wm,
            'se_delta_calibrated_logit': sem(sub['delta_calibrated_logit']),
            'se_delta_P': sem(sub['delta_P']),
            'se_positive_fraction': sem(sub['positive']),
        })
    summary = summary.merge(pd.DataFrame(se_rows), on=group_cols, how='left')
    summary['window_label'] = summary['window_start1'].astype(int).astype(str) + '-' + summary['window_end1'].astype(int).astype(str)
    summary_path = out_dir / f'{args.prefix}.window_summary.tsv'
    summary.to_csv(summary_path, sep='\t', index=False)

    top_rows = []
    for (sample, rate), sub in summary.groupby(['sample_name', 'rate']):
        t = sub.sort_values('mean_delta_calibrated_logit', ascending=True).head(args.top_n).copy()
        t.insert(0, 'rank', np.arange(1, len(t) + 1))
        top_rows.append(t)
    top_df = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    top_path = out_dir / f'{args.prefix}.top_windows.tsv'
    top_df.to_csv(top_path, sep='\t', index=False)

    # sample-specific plots
    samples = sorted(summary['sample_name'].dropna().unique())
    coord_shift = 1 if args.coord_origin == 0 else 0
    annotation_index = []
    for sample in samples:
        sub = summary[summary['sample_name'] == sample].copy()
        meta = get_sample_annotations(sample, df[df['sample_name'] == sample], inputs_dir, ipr_df, args.ipr_pad, coord_shift)
        annotation_index.append({
            'sample_name': sample,
            'span_start1': meta['span'][0] if meta['span'] is not None else np.nan,
            'span_end1': meta['span'][1] if meta['span'] is not None else np.nan,
            'motif_A': next((m['pos'] for m in meta['motifs'] if m['name'] == 'A'), np.nan),
            'motif_B': next((m['pos'] for m in meta['motifs'] if m['name'] == 'B'), np.nan),
            'motif_C': next((m['pos'] for m in meta['motifs'] if m['name'] == 'C'), np.nan),
            'n_domains_shown': len(meta['domains']),
        })
        plot_sample_line(sub, sample, meta, out_dir / f'{args.prefix}.{safe(sample)}.line.delta_calibrated_logit.pdf', coord_shift)
        plot_sample_positive(sub, sample, meta, out_dir / f'{args.prefix}.{safe(sample)}.line.positive_fraction.pdf', coord_shift)
        plot_sample_heatmap(sub, sample, meta, out_dir / f'{args.prefix}.{safe(sample)}.heatmap.delta_calibrated_logit.pdf',
                            value_col='mean_delta_calibrated_logit', cbar_label='Mean Δ calibrated logit',
                            title=f'{sample}: mean Δ calibrated logit heatmap', coord_shift=coord_shift)
        plot_sample_heatmap(sub, sample, meta, out_dir / f'{args.prefix}.{safe(sample)}.heatmap.positive_fraction.pdf',
                            value_col='positive_fraction', cbar_label='Positive fraction',
                            title=f'{sample}: positive fraction heatmap', coord_shift=coord_shift)

    ann_path = out_dir / f'{args.prefix}.annotation_index.tsv'
    pd.DataFrame(annotation_index).to_csv(ann_path, sep='\t', index=False)

    meta = {
        'manifest': args.manifest,
        'logits_json': args.logits_json,
        'inputs_dir': str(inputs_dir) if inputs_dir else None,
        'iprscan_tsv': args.iprscan_tsv,
        'coord_origin': args.coord_origin,
        'n_rows_merged': int(len(df)),
        'n_rows_summary': int(len(summary)),
        'samples': samples,
        'rates': sorted([float(x) for x in summary['rate'].dropna().unique().tolist()]),
        'prob_threshold': args.prob_threshold,
        'ipr_keep_sources': keep_sources,
        'ipr_pad': args.ipr_pad,
    }
    with open(out_dir / f'{args.prefix}.meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2)

    print('Wrote:', merged_path)
    print('Wrote:', summary_path)
    print('Wrote:', top_path)
    print('Wrote:', ann_path)
    print('Output directory:', out_dir)


if __name__ == '__main__':
    main()

