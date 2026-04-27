#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy.signal import find_peaks
from scipy.stats import norm, wilcoxon


MOTIF_ORDER = ['A', 'B', 'C']
MOTIF_COLORS = {'A': 'tab:blue', 'B': 'tab:orange', 'C': 'tab:green'}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Analyze zero-shot motif localization from PalmSite attention weights. "
            "Fits a Gaussian baseline to each attention profile, detects positive anomaly spikes, "
            "compares them to palm_annot motif A/B/C positions, and exports tables/figures."
        )
    )
    p.add_argument('--fasta', required=True, help='Input protein FASTA')
    p.add_argument('--attn-json', required=True, help='PalmSite attention JSON')
    p.add_argument('--gff', required=True, help='PalmSite GFF output')
    p.add_argument('--palm-annot', required=True, help='palm_annot TSV output')
    p.add_argument('--outdir', required=True, help='Output directory')
    p.add_argument('--program-name', default='PalmSite', help='Program name shown in figure titles')
    p.add_argument('--topk', default='3,4,5,6,7,8,9,10', help='Comma-separated top-K peak counts to summarize')
    p.add_argument('--peak-prominence', type=float, default=5.0, help='Peak prominence threshold on anomaly z-score')
    p.add_argument('--peak-distance', type=int, default=12, help='Minimum distance between detected peaks')
    p.add_argument('--motif-match-radius', type=int, default=10, help='Distance threshold (aa) for motif/spike matching')
    p.add_argument('--motif-score-radius', type=int, default=4, help='Half-window used to score anomaly around a motif')
    p.add_argument('--smooth-sigma', type=float, default=1.2, help='Gaussian smoothing sigma for anomaly residual')
    p.add_argument('--spike-window', type=int, default=6, help='Half-window size for extracted spike-centered sequences')
    p.add_argument('--n-examples', type=int, default=10, help='Number of example sequences to plot')
    p.add_argument('--example-min-length', type=int, default=500, help='Minimum protein length for selected example sequences')
    return p.parse_args()


def read_fasta(path: str) -> Dict[str, str]:
    seqs = {}
    for rec in SeqIO.parse(path, 'fasta'):
        seqs[rec.id] = str(rec.seq)
    return seqs


def read_attn(path: str) -> Dict[str, dict]:
    with open(path) as fh:
        return json.load(fh)


def read_gff(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            attrs = {}
            for item in fields[8].split(';'):
                if '=' in item:
                    k, v = item.split('=', 1)
                    attrs[k] = v
            rows.append(
                {
                    'seqid': fields[0],
                    'source': fields[1],
                    'feature': fields[2],
                    'start': int(fields[3]),
                    'end': int(fields[4]),
                    'score': float(fields[5]) if fields[5] != '.' else np.nan,
                    'chunk': attrs.get('Chunk'),
                    'P': float(attrs.get('P')) if attrs.get('P') is not None else np.nan,
                    'mu_attr': float(attrs.get('mu')) if attrs.get('mu') is not None else np.nan,
                    'sigma_attr': float(attrs.get('sigma')) if attrs.get('sigma') is not None else np.nan,
                    'len_attr': int(attrs.get('len')) if attrs.get('len') is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def read_palm_annot(path: str) -> Dict[str, dict]:
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            seqid = parts[0]
            d = {}
            for token in parts[1:]:
                if '=' in token:
                    k, v = token.split('=', 1)
                    d[k] = v
            out[seqid] = d
    return out


def gaussian_kernel1d(sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(4 * sigma)))
    x = np.arange(-radius, radius + 1)
    ker = np.exp(-0.5 * (x / sigma) ** 2)
    ker /= ker.sum()
    return ker


def rolling_max(arr: np.ndarray, radius: int) -> np.ndarray:
    out = np.empty_like(arr)
    n = len(arr)
    for i in range(n):
        lo = max(0, i - radius)
        hi = min(n, i + radius + 1)
        out[i] = np.max(arr[lo:hi])
    return out


def fit_gaussian_baseline(x: np.ndarray, w: np.ndarray) -> Tuple[float, float, np.ndarray]:
    sw = float(np.sum(w))
    mu = float(np.sum(x * w) / sw)
    var = float(np.sum(((x - mu) ** 2) * w) / sw)
    sigma = math.sqrt(max(var, 1e-12))
    base = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    base = base / np.sum(base) * sw
    return mu, sigma, base


def analyze_profile(
    x: np.ndarray,
    w: np.ndarray,
    domain_start: int,
    domain_end: int,
    smooth_sigma: float,
    score_radius: int,
    peak_prominence: float,
    peak_distance: int,
):
    mu, sigma, base = fit_gaussian_baseline(x, w)
    residual = w - base
    ker = gaussian_kernel1d(smooth_sigma)
    residual_smooth = np.convolve(residual, ker, mode='same')

    med = float(np.median(residual_smooth))
    mad = float(np.median(np.abs(residual_smooth - med)))
    robust_scale = max(1.4826 * mad, float(np.std(residual_smooth)) * 0.1, 1e-12)
    z = (residual_smooth - med) / robust_scale

    window_score = rolling_max(z, radius=score_radius)

    domain_mask = (x >= domain_start) & (x <= domain_end)
    domain_idx = np.where(domain_mask)[0]
    domain_window_score = window_score[domain_mask]

    peak_idx, props = find_peaks(z, prominence=peak_prominence, distance=peak_distance)
    if len(peak_idx) > 0:
        order = np.argsort(props['prominences'])[::-1]
        peak_idx = peak_idx[order]
        prominences = props['prominences'][order]
    else:
        prominences = np.array([], dtype=float)

    return {
        'mu': mu,
        'sigma': sigma,
        'baseline': base,
        'residual': residual,
        'residual_smooth': residual_smooth,
        'z': z,
        'window_score': window_score,
        'domain_mask': domain_mask,
        'domain_idx': domain_idx,
        'domain_window_score': domain_window_score,
        'peak_idx': peak_idx,
        'peak_prominence': prominences,
    }


def motif_center(annot: dict, motif: str):
    pos_key = f'pos{motif}'
    if pos_key not in annot:
        return None, None, None
    seq = annot.get(f'seq{motif}') or annot.get(f'pssm_seq{motif}') or annot.get(f'motif_hmm_seq{motif}') or ''
    pos = int(annot[pos_key])
    center = pos + (len(seq) - 1) / 2.0 if seq else float(pos)
    return pos, center, seq




def has_all_abc_motifs(annot: dict) -> bool:
    return all(f'pos{motif}' in annot for motif in ['A', 'B', 'C'])


def is_noncanonical_motif_order(order: str) -> bool:
    if order is None:
        return False
    order = str(order).strip().upper()
    return order not in {'', '.', 'ABC'}


def choose_examples(
    motif_df: pd.DataFrame,
    fasta: Dict[str, str],
    palm: Dict[str, dict],
    min_length: int = 500,
    n_examples: int = 10,
) -> List[str]:
    examples = []
    if motif_df.empty or n_examples <= 0:
        return examples

    by_seq = motif_df.groupby('seqid').agg(
        mean_percentile=('percentile', 'mean'),
        min_dist=('best_peak_distance', 'min'),
        motif_order=('motif_order', 'first'),
        gdd=('gdd', 'first'),
    ).reset_index()

    by_seq['seq_length'] = by_seq['seqid'].map(lambda s: len(fasta.get(s, '')))
    by_seq['has_all_abc'] = by_seq['seqid'].map(lambda s: has_all_abc_motifs(palm.get(s, {})))
    eligible = by_seq[(by_seq['seq_length'] > min_length) & (by_seq['has_all_abc'])].copy()

    def add_best(df: pd.DataFrame):
        if df.empty:
            return
        seqid = df.sort_values(['mean_percentile', 'min_dist'], ascending=[False, True]).iloc[0]['seqid']
        if seqid not in examples:
            examples.append(seqid)

    add_best(eligible)
    add_best(eligible[eligible['motif_order'].map(is_noncanonical_motif_order)])
    add_best(eligible[eligible['gdd'].fillna('') != 'GDD'])

    for seqid in eligible.sort_values(['mean_percentile', 'min_dist'], ascending=[False, True])['seqid']:
        if seqid not in examples:
            examples.append(seqid)
        if len(examples) >= n_examples:
            break

    return examples[:n_examples]

def save_fasta(records: List[Tuple[str, str]], path: Path):
    with open(path, 'w') as fh:
        for header, seq in records:
            fh.write(f'>{header}\n')
            for i in range(0, len(seq), 80):
                fh.write(seq[i:i+80] + '\n')


def make_fig_topk(summary_df: pd.DataFrame, program_name: str, outpath: Path):
    plt.figure(figsize=(7, 5))
    plt.plot(summary_df['top_k'], summary_df['observed_fraction'], marker='o', label='Observed')
    plt.plot(summary_df['top_k'], summary_df['expected_random_fraction'], marker='o', label='Random expectation')
    plt.xlabel('Top-K anomaly spikes considered')
    plt.ylabel(f'Fraction of motifs within ±{int(summary_df["match_radius"].iloc[0])} aa')
    plt.title(f'{program_name}: motif proximity to anomaly spikes')
    plt.xticks(summary_df['top_k'])
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def make_fig_topk_by_motif(by_motif_df: pd.DataFrame, topk_values: List[int], program_name: str, outpath: Path):
    plt.figure(figsize=(8, 5))
    for motif in MOTIF_ORDER:
        sub = by_motif_df[by_motif_df['group'] == motif]
        if sub.empty:
            continue
        row = sub.iloc[0]
        obs = [row[f'top{k}_observed_fraction'] for k in topk_values]
        exp = [row[f'top{k}_expected_random_fraction'] for k in topk_values]
        plt.plot(topk_values, obs, marker='o', label=f'Motif {motif} observed', color=MOTIF_COLORS[motif])
        plt.plot(topk_values, exp, marker='o', linestyle='--', label=f'Motif {motif} random', color=MOTIF_COLORS[motif], alpha=0.55)
    plt.xlabel('Top-K anomaly spikes considered')
    plt.ylabel('Fraction of motifs within matching radius')
    plt.title(f'{program_name}: motif-specific top-K spike proximity')
    plt.xticks(topk_values)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def make_fig_percentile_box(motif_df: pd.DataFrame, program_name: str, outpath: Path):
    labels = [m for m in MOTIF_ORDER if m in set(motif_df['motif'])]
    data = [motif_df[motif_df['motif'] == m]['percentile'].values for m in labels]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, motif in zip(bp['boxes'], labels):
        patch.set_facecolor(MOTIF_COLORS[motif])
        patch.set_alpha(0.45)
        patch.set_edgecolor(MOTIF_COLORS[motif])
    for median in bp['medians']:
        median.set_color('black')
    ax.axhline(0.5, linestyle='--', linewidth=1, color='0.5')
    ax.set_ylabel('Within-domain percentile of motif anomaly score')
    ax.set_xlabel('Catalytic motif')
    ax.set_title(f'{program_name}: motif-centered anomaly enrichment')
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def make_fig_distance_hist(motif_df: pd.DataFrame, program_name: str, outpath: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.arange(-0.5, 60.5, 2)
    for motif in MOTIF_ORDER:
        vals = motif_df.loc[motif_df['motif'] == motif, 'best_peak_distance'].dropna().values
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, histtype='step', linewidth=1.8, label=f'Motif {motif}', color=MOTIF_COLORS[motif])
    ax.axvline(10, linestyle='--', linewidth=1, color='0.5')
    ax.set_xlabel('Distance from motif center to nearest anomaly spike (aa)')
    ax.set_ylabel('Number of motif occurrences')
    ax.set_title(f'{program_name}: nearest anomaly spike distances by motif')
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def make_example_plot(seqid: str, seqrow: dict, program_name: str, outpath: Path):
    x = seqrow['x']
    w = seqrow['w']
    base = seqrow['baseline']
    z = seqrow['z']
    motif_info = seqrow['motifs']
    peak_positions = seqrow['peak_positions'][:10]

    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(x, w, linewidth=1, label='Observed weight', color='black')
    ax1.plot(x, base, linewidth=1, label='Gaussian baseline', color='0.5')
    used_labels_ax1 = set()
    for motif, pos, center, motif_seq in motif_info:
        color = MOTIF_COLORS.get(motif, 'tab:red')
        label = f'Motif {motif}' if motif not in used_labels_ax1 else None
        ax1.axvline(center, linestyle='--', linewidth=1.5, color=color, label=label)
        used_labels_ax1.add(motif)
    ax1.set_ylabel('Attention weight')
    ax1.set_title(f'{program_name}: {seqid}')
    ax1.legend(loc='best', fontsize=8)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(x, z, linewidth=1, label='Anomaly z-score', color='black')
    if len(peak_positions) > 0:
        peak_idx = [int(np.argmin(np.abs(x - p))) for p in peak_positions]
        ax2.scatter(peak_positions, z[peak_idx], s=18, label='Top anomaly spikes', color='tab:red')
    used_labels_ax2 = set()
    for motif, pos, center, motif_seq in motif_info:
        color = MOTIF_COLORS.get(motif, 'tab:red')
        label = f'Motif {motif}' if motif not in used_labels_ax2 else None
        ax2.axvline(center, linestyle='--', linewidth=1.5, color=color, label=label)
        used_labels_ax2.add(motif)
    ax2.set_xlabel('Protein position (aa)')
    ax2.set_ylabel('Anomaly z-score')
    ax2.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    topk_values = [int(x) for x in args.topk.split(',') if x.strip()]

    fasta = read_fasta(args.fasta)
    attn = read_attn(args.attn_json)
    gff = read_gff(args.gff)
    palm = read_palm_annot(args.palm_annot)

    gff = gff[gff['chunk'].isin(attn.keys())].copy()
    gff_with_palm = gff[gff['seqid'].isin(palm.keys())].copy()

    motif_rows = []
    peak_rows = []
    spike_windows_by_motif = {'A': [], 'B': [], 'C': []}
    seq_plot_data = {}

    for _, row in gff_with_palm.iterrows():
        seqid = row['seqid']
        chunk = row['chunk']
        chunk_data = attn[chunk]
        x = np.asarray(chunk_data['abs_pos'], dtype=float) + 1.0  # 1-based coordinates
        w = np.asarray(chunk_data['w'], dtype=float)

        analysis = analyze_profile(
            x=x,
            w=w,
            domain_start=int(row['start']),
            domain_end=int(row['end']),
            smooth_sigma=args.smooth_sigma,
            score_radius=args.motif_score_radius,
            peak_prominence=args.peak_prominence,
            peak_distance=args.peak_distance,
        )

        peak_idx = analysis['peak_idx']
        peak_positions = x[peak_idx] if len(peak_idx) else np.array([], dtype=float)
        peak_prom = analysis['peak_prominence']

        for rank, (pidx, ppos, pprom) in enumerate(zip(peak_idx, peak_positions, peak_prom), start=1):
            peak_rows.append(
                {
                    'seqid': seqid,
                    'chunk': chunk,
                    'domain_start': int(row['start']),
                    'domain_end': int(row['end']),
                    'peak_rank': rank,
                    'peak_position': int(round(ppos)),
                    'peak_prominence': float(pprom),
                    'peak_zscore': float(analysis['z'][pidx]),
                }
            )

        motif_info_for_plot = []
        annot = palm[seqid]
        seq = fasta.get(seqid, '')

        for motif in ['A', 'B', 'C']:
            pos, center, motif_seq = motif_center(annot, motif)
            if pos is None:
                continue
            motif_info_for_plot.append((motif, pos, center, motif_seq))

            closest_idx = int(np.argmin(np.abs(x - center)))
            motif_window_score = float(analysis['window_score'][closest_idx])
            domain_scores = analysis['domain_window_score']
            percentile = float((np.sum(domain_scores < motif_window_score) + 0.5 * np.sum(domain_scores == motif_window_score)) / len(domain_scores))

            if len(peak_positions):
                distances = np.abs(peak_positions - center)
                nearest_order = int(np.argmin(distances))
                best_peak_distance = float(distances[nearest_order])
                best_peak_position = int(round(peak_positions[nearest_order]))
                best_peak_rank = int(nearest_order + 1)
                best_peak_prominence = float(peak_prom[nearest_order])
            else:
                best_peak_distance = np.nan
                best_peak_position = np.nan
                best_peak_rank = np.nan
                best_peak_prominence = np.nan

            row_out = {
                'seqid': seqid,
                'chunk': chunk,
                'domain_start': int(row['start']),
                'domain_end': int(row['end']),
                'motif': motif,
                'motif_start': int(pos),
                'motif_center': float(center),
                'motif_sequence': motif_seq,
                'motif_order': annot.get('pssm_ABC', ''),
                'gdd': annot.get('gdd', ''),
                'motif_window_score': motif_window_score,
                'percentile': percentile,
                'best_peak_position': best_peak_position,
                'best_peak_distance': best_peak_distance,
                'best_peak_rank': best_peak_rank,
                'best_peak_prominence': best_peak_prominence,
            }

            for top_k in topk_values:
                if len(peak_positions):
                    row_out[f'top{top_k}_within_radius'] = int(np.any(np.abs(peak_positions[:top_k] - center) <= args.motif_match_radius))
                else:
                    row_out[f'top{top_k}_within_radius'] = 0

            motif_rows.append(row_out)

            # Extract spike-centered windows for sequence-logo downstream use.
            if seq and len(peak_positions):
                candidate = np.where(np.abs(peak_positions - center) <= args.motif_match_radius)[0]
                if len(candidate) > 0:
                    # Use the closest matching peak; break ties by prominence (already ordered by prominence).
                    best_local = candidate[np.argmin(np.abs(peak_positions[candidate] - center))]
                    spike_pos = int(round(peak_positions[best_local]))
                    lo = max(1, spike_pos - args.spike_window)
                    hi = min(len(seq), spike_pos + args.spike_window)
                    subseq = seq[lo - 1:hi]
                    header = (
                        f'{seqid}|motif={motif}|peak={spike_pos}|motif_center={center:.1f}'
                        f'|distance={abs(spike_pos-center):.1f}|domain={int(row["start"])}-{int(row["end"])}'
                    )
                    spike_windows_by_motif[motif].append((header, subseq))

        seq_plot_data[seqid] = {
            'x': x,
            'w': w,
            'baseline': analysis['baseline'],
            'z': analysis['z'],
            'peak_positions': peak_positions,
            'motifs': motif_info_for_plot,
        }

    motif_df = pd.DataFrame(motif_rows)
    peak_df = pd.DataFrame(peak_rows)

    # Save raw result tables.
    motif_tsv = outdir / 'motif_level_results.tsv'
    peak_tsv = outdir / 'peak_level_results.tsv'
    motif_df.to_csv(motif_tsv, sep='\t', index=False)
    peak_df.to_csv(peak_tsv, sep='\t', index=False)

    # Sequence windows for downstream logo generation.
    for motif, records in spike_windows_by_motif.items():
        save_fasta(records, outdir / f'spike_windows_motif_{motif}.fasta')

    # Summary tables.
    summary_rows = []
    by_motif_rows = []

    overall_w = wilcoxon(motif_df['percentile'] - 0.5, alternative='greater', zero_method='wilcox') if len(motif_df) else None

    for top_k in topk_values:
        obs = float(motif_df[f'top{top_k}_within_radius'].mean())

        # Exact random expectation per motif under K random positions placed uniformly inside the PalmSite domain.
        expected_vals = []
        for _, r in motif_df.iterrows():
            domain_len = int(r['domain_end'] - r['domain_start'] + 1)
            left = max(int(r['domain_start']), int(math.floor(r['motif_center'] - args.motif_match_radius)))
            right = min(int(r['domain_end']), int(math.ceil(r['motif_center'] + args.motif_match_radius)))
            window_len = max(0, right - left + 1)
            K = min(top_k, domain_len)
            if K >= domain_len or (domain_len - window_len) < K:
                expected = 1.0
            else:
                expected = 1.0 - math.comb(domain_len - window_len, K) / math.comb(domain_len, K)
            expected_vals.append(expected)
        expected_vals = np.asarray(expected_vals, dtype=float)
        exp_mean = float(np.mean(expected_vals))
        var_sum = float(np.sum(expected_vals * (1.0 - expected_vals)))
        zstat = float((motif_df[f'top{top_k}_within_radius'].sum() - np.sum(expected_vals)) / math.sqrt(var_sum)) if var_sum > 0 else np.nan
        pval = float(norm.sf(zstat)) if np.isfinite(zstat) else np.nan

        summary_rows.append(
            {
                'metric': f'top{top_k}_coverage',
                'top_k': top_k,
                'match_radius': args.motif_match_radius,
                'observed_fraction': obs,
                'expected_random_fraction': exp_mean,
                'z_statistic_vs_random': zstat,
                'p_value_vs_random': pval,
            }
        )

    # Percentile-based enrichment overall and by motif.
    overall_summary = {
        'n_input_sequences': len(fasta),
        'n_gff_domains': int(len(gff)),
        'n_domains_with_palm_annot': int(len(gff_with_palm)),
        'n_motif_occurrences': int(len(motif_df)),
        'n_detected_peaks': int(len(peak_df)),
        'mean_percentile_overall': float(motif_df['percentile'].mean()) if len(motif_df) else np.nan,
        'median_percentile_overall': float(motif_df['percentile'].median()) if len(motif_df) else np.nan,
        'wilcoxon_statistic_overall': float(overall_w.statistic) if overall_w is not None else np.nan,
        'wilcoxon_pvalue_overall': float(overall_w.pvalue) if overall_w is not None else np.nan,
        'mean_nearest_peak_distance': float(motif_df['best_peak_distance'].mean()) if len(motif_df) else np.nan,
        'median_nearest_peak_distance': float(motif_df['best_peak_distance'].median()) if len(motif_df) else np.nan,
    }
    overall_df = pd.DataFrame([overall_summary])
    overall_df.to_csv(outdir / 'summary_overall.tsv', sep='\t', index=False)

    for label, sub in [('overall', motif_df)] + [(m, motif_df[motif_df['motif'] == m].copy()) for m in ['A', 'B', 'C']]:
        if len(sub) == 0:
            continue
        wres = wilcoxon(sub['percentile'] - 0.5, alternative='greater', zero_method='wilcox')
        entry = {
            'group': label,
            'n_motifs': int(len(sub)),
            'mean_percentile': float(sub['percentile'].mean()),
            'median_percentile': float(sub['percentile'].median()),
            'mean_nearest_peak_distance': float(sub['best_peak_distance'].mean()),
            'median_nearest_peak_distance': float(sub['best_peak_distance'].median()),
            'wilcoxon_statistic': float(wres.statistic),
            'wilcoxon_pvalue': float(wres.pvalue),
            'n_spike_windows_extracted': int(len(spike_windows_by_motif[label])) if label in spike_windows_by_motif else int(sum(sub['best_peak_distance'] <= args.motif_match_radius)),
        }
        for top_k in topk_values:
            obs = float(sub[f'top{top_k}_within_radius'].mean())
            expected_vals = []
            for _, r in sub.iterrows():
                domain_len = int(r['domain_end'] - r['domain_start'] + 1)
                left = max(int(r['domain_start']), int(math.floor(r['motif_center'] - args.motif_match_radius)))
                right = min(int(r['domain_end']), int(math.ceil(r['motif_center'] + args.motif_match_radius)))
                window_len = max(0, right - left + 1)
                K = min(top_k, domain_len)
                if K >= domain_len or (domain_len - window_len) < K:
                    expected = 1.0
                else:
                    expected = 1.0 - math.comb(domain_len - window_len, K) / math.comb(domain_len, K)
                expected_vals.append(expected)
            entry[f'top{top_k}_observed_fraction'] = obs
            entry[f'top{top_k}_expected_random_fraction'] = float(np.mean(expected_vals))
        by_motif_rows.append(entry)

    topk_df = pd.DataFrame(summary_rows)
    by_motif_df = pd.DataFrame(by_motif_rows)
    topk_df.to_csv(outdir / 'summary_topk.tsv', sep='\t', index=False)
    by_motif_df.to_csv(outdir / 'summary_by_motif.tsv', sep='\t', index=False)

    # Figures.
    make_fig_topk(topk_df, args.program_name, outdir / 'fig_topk_coverage_vs_random.pdf')
    make_fig_topk_by_motif(by_motif_df, topk_values, args.program_name, outdir / 'fig_topk_coverage_by_motif.pdf')
    make_fig_percentile_box(motif_df, args.program_name, outdir / 'fig_motif_percentile_boxplot.pdf')
    make_fig_distance_hist(motif_df, args.program_name, outdir / 'fig_nearest_peak_distance_histogram.pdf')

    examples = choose_examples(
        motif_df,
        fasta=fasta,
        palm=palm,
        min_length=args.example_min_length,
        n_examples=args.n_examples,
    )
    for seqid in examples:
        if seqid in seq_plot_data:
            make_example_plot(seqid, seq_plot_data[seqid], args.program_name, outdir / f'fig_example_{seqid}.pdf')

    # Write a small readme.
    with open(outdir / 'README.txt', 'w') as fh:
        fh.write(
            'Zero-shot motif spike analysis outputs\n'
            '================================\n\n'
            'Files:\n'
            '  - summary_overall.tsv: overall dataset summary\n'
            '  - summary_topk.tsv: observed vs random motif/spike proximity by top-K peaks\n'
            '  - summary_by_motif.tsv: summary statistics for motifs A/B/C and overall\n'
            '  - motif_level_results.tsv: one row per motif occurrence\n'
            '  - peak_level_results.tsv: one row per detected anomaly peak\n'
            '  - spike_windows_motif_[A|B|C].fasta: extracted spike-centered sequence windows for logo building\n'
            '  - fig_*.pdf: summary and example figures\n\n'
            'Method summary:\n'
            '  1) Use the PalmSite GFF Chunk attribute to select the relevant attention profile for each predicted domain.\n'
            '  2) Fit a Gaussian baseline to the attention weights by moment matching.\n'
            '  3) Define anomaly spikes as positive local maxima in the residual (observed - Gaussian baseline), after light Gaussian smoothing and robust z-score scaling.\n'
            '  4) Compare these spikes with catalytic motif positions predicted by palm_annot.\n'
            '  5) Extract spike-centered sequence windows for downstream sequence-logo analysis.\n'
        )


if __name__ == '__main__':
    main()

