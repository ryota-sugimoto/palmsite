#!/usr/bin/env python3
"""Analyze program benchmark resource logs.

Usage:
    python analyze_palmsite_benchmark.py \
        /path/to/benchmark.tar.gz \
        /path/to/output_dir \
        --program-name PalmSite \
        --figure-format pdf

Expected archive contents:
    JSON files named like sample_n<number_of_sequences>_rep<replication_id>.json

Outputs:
    - run_summary.csv
    - summary_by_size.csv
    - overall_summary.json
    - overall_summary.md
    - cpu_utilization_over_time.<figure-format>
    - gpu_utilization_over_time.<figure-format>
    - cpu_memory_rss_over_time.<figure-format>
    - gpu_memory_over_time.<figure-format>
    - runtime_vs_sequences.<figure-format>
    - throughput_vs_sequences.<figure-format>
    - peak_memory_vs_sequences.<figure-format>
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PATTERN = re.compile(r"sample_n(?P<n>\d+)_rep(?P<rep>\d+)\.json$")


def bytes_to_gib(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value) / (1024 ** 3)


def extract_archive(tar_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists():
        for p in sorted(extract_dir.rglob("*"), reverse=True):
            if p.is_file() or p.is_symlink():
                p.unlink()
            elif p.is_dir():
                p.rmdir()
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(extract_dir)


def parse_runs(extract_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    json_files = sorted(extract_dir.rglob("sample_n*_rep*.json"))
    if not json_files:
        raise FileNotFoundError("No sample_n*_rep*.json files found after extraction.")

    run_rows = []
    sample_rows = []
    env = None

    for path in json_files:
        match = PATTERN.match(path.name)
        if not match:
            continue
        n_sequences = int(match.group("n"))
        rep = int(match.group("rep"))

        with path.open() as f:
            data = json.load(f)

        meta = data["meta"]
        result = data["result"]
        summary = data["summary"]
        gpu_summary = summary.get("gpu") or {}
        gpu_summary_device = (gpu_summary.get("devices") or [{}])[0] if gpu_summary.get("devices") else {}

        if env is None:
            env = {
                "logical_cpu_count": meta.get("system_resources", {}).get("logical_cpu_count"),
                "physical_cpu_count": meta.get("system_resources", {}).get("physical_cpu_count"),
                "system_memory_gib": bytes_to_gib(meta.get("system_resources", {}).get("total_memory_bytes")),
                "gpu_device_count": meta.get("gpu_monitoring", {}).get("device_count"),
                "gpu_name": ((meta.get("gpu_monitoring", {}).get("devices") or [{}])[0]).get("name"),
                "gpu_memory_gib": bytes_to_gib(((meta.get("gpu_monitoring", {}).get("devices") or [{}])[0]).get("memory_total_bytes")),
                "interval_seconds": meta.get("interval_seconds"),
            }

        run_rows.append({
            "file": path.name,
            "n_sequences": n_sequences,
            "rep": rep,
            "interval_s": meta.get("interval_seconds"),
            "wall_clock_s": summary.get("wall_clock_seconds"),
            "sample_count": summary.get("sample_count"),
            "avg_cpu_percent": summary.get("avg_cpu_percent"),
            "max_cpu_percent": summary.get("max_cpu_percent"),
            "avg_cpu_machine_percent": summary.get("avg_cpu_percent_of_total_machine"),
            "max_cpu_machine_percent": summary.get("max_cpu_percent_of_total_machine"),
            "peak_rss_bytes": summary.get("peak_rss_bytes"),
            "peak_vms_bytes": summary.get("peak_vms_bytes"),
            "peak_memory_percent_of_system": summary.get("peak_memory_percent_of_system"),
            "peak_threads": summary.get("peak_num_threads"),
            "cpu_time_total_s": summary.get("cpu_time_total_s"),
            "io_write_bytes": (summary.get("total_io") or {}).get("write_bytes"),
            "io_read_chars": (summary.get("total_io") or {}).get("read_chars"),
            "voluntary_ctx": (summary.get("total_context_switches") or {}).get("voluntary"),
            "involuntary_ctx": (summary.get("total_context_switches") or {}).get("involuntary"),
            "gpu_available": gpu_summary.get("available"),
            "peak_process_tree_gpu_mem_bytes": gpu_summary_device.get("peak_process_tree_gpu_memory_bytes"),
            "peak_device_memory_used_bytes": gpu_summary_device.get("peak_device_memory_used_bytes"),
            "peak_gpu_util_percent": gpu_summary_device.get("peak_device_utilization_gpu_percent"),
            "peak_power_watts": gpu_summary_device.get("peak_power_watts"),
            "returncode": result.get("returncode"),
            "timed_out": result.get("timed_out"),
            "started_at_local": meta.get("started_at_local"),
            "finished_at_local": meta.get("finished_at_local"),
        })

        for sample in data.get("samples", []):
            process = sample.get("process") or {}
            gpu = sample.get("gpu") or {}
            gpu_device = (gpu.get("devices") or [{}])[0] if gpu.get("devices") else {}
            sample_rows.append({
                "file": path.name,
                "n_sequences": n_sequences,
                "rep": rep,
                "t_rel_s": sample.get("t_rel_s"),
                "cpu_percent": process.get("cpu_percent"),
                "cpu_machine_percent": process.get("cpu_percent_of_total_machine"),
                "cpu_time_total_s": process.get("cpu_time_total_s"),
                "rss_bytes": process.get("rss_bytes"),
                "vms_bytes": process.get("vms_bytes"),
                "memory_percent_of_system": process.get("memory_percent_of_system"),
                "num_threads": process.get("num_threads"),
                "live_process_count": process.get("live_process_count"),
                "read_chars": (process.get("io") or {}).get("read_chars"),
                "write_bytes": (process.get("io") or {}).get("write_bytes"),
                "gpu_device_mem_used_bytes": gpu_device.get("device_memory_used_bytes"),
                "gpu_process_mem_bytes": gpu_device.get("process_tree_gpu_memory_bytes"),
                "gpu_util_percent": gpu_device.get("device_utilization_gpu_percent"),
                "gpu_mem_util_percent": gpu_device.get("device_utilization_memory_percent"),
                "gpu_power_watts": gpu_device.get("power_watts"),
                "gpu_temp_c": gpu_device.get("temperature_c"),
            })

    runs = pd.DataFrame(run_rows).sort_values(["n_sequences", "rep"]).reset_index(drop=True)
    samples = pd.DataFrame(sample_rows).sort_values(["n_sequences", "rep", "t_rel_s"]).reset_index(drop=True)

    runs["throughput_seq_s"] = runs["n_sequences"] / runs["wall_clock_s"]

    live_samples = samples.copy()
    if "live_process_count" in live_samples.columns:
        live_samples = live_samples[live_samples["live_process_count"] > 0].copy()

    run_sample_agg = live_samples.groupby("file").agg(
        mean_cpu_percent_obs=("cpu_percent", "mean"),
        mean_cpu_machine_percent_obs=("cpu_machine_percent", "mean"),
        mean_rss_bytes=("rss_bytes", "mean"),
        mean_gpu_util_percent=("gpu_util_percent", "mean"),
        mean_gpu_process_mem_bytes=("gpu_process_mem_bytes", "mean"),
        mean_gpu_power_watts=("gpu_power_watts", "mean"),
        peak_gpu_temp_c=("gpu_temp_c", "max"),
    )
    runs = runs.merge(run_sample_agg, on="file", how="left")

    return runs, live_samples, env


def add_human_columns(runs: pd.DataFrame, samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    runs = runs.copy()
    samples = samples.copy()

    for col_in, col_out in [
        ("peak_rss_bytes", "peak_rss_gib"),
        ("peak_vms_bytes", "peak_vms_gib"),
        ("peak_process_tree_gpu_mem_bytes", "peak_process_tree_gpu_mem_gib"),
        ("peak_device_memory_used_bytes", "peak_device_memory_used_gib"),
        ("mean_rss_bytes", "mean_rss_gib"),
        ("mean_gpu_process_mem_bytes", "mean_gpu_process_mem_gib"),
        ("io_write_bytes", "io_write_gib"),
    ]:
        if col_in in runs.columns:
            runs[col_out] = runs[col_in].astype(float) / (1024 ** 3)

    for col_in, col_out in [
        ("rss_bytes", "rss_gib"),
        ("vms_bytes", "vms_gib"),
        ("gpu_process_mem_bytes", "gpu_process_mem_gib"),
        ("gpu_device_mem_used_bytes", "gpu_device_mem_used_gib"),
        ("write_bytes", "write_gib"),
    ]:
        if col_in in samples.columns:
            samples[col_out] = samples[col_in].astype(float) / (1024 ** 3)

    return runs, samples


def summarize_by_size(runs: pd.DataFrame) -> pd.DataFrame:
    grouped = runs.groupby("n_sequences").agg(
        reps=("rep", "count"),
        wall_clock_mean_s=("wall_clock_s", "mean"),
        wall_clock_sd_s=("wall_clock_s", "std"),
        throughput_mean_seq_s=("throughput_seq_s", "mean"),
        throughput_sd_seq_s=("throughput_seq_s", "std"),
        avg_cpu_percent_mean=("avg_cpu_percent", "mean"),
        peak_cpu_percent_mean=("max_cpu_percent", "mean"),
        peak_rss_mean_gib=("peak_rss_gib", "mean"),
        peak_rss_sd_gib=("peak_rss_gib", "std"),
        peak_gpu_mem_mean_gib=("peak_process_tree_gpu_mem_gib", "mean"),
        peak_gpu_mem_sd_gib=("peak_process_tree_gpu_mem_gib", "std"),
        peak_gpu_util_mean=("peak_gpu_util_percent", "mean"),
        peak_power_mean_watts=("peak_power_watts", "mean"),
        mean_gpu_util_mean=("mean_gpu_util_percent", "mean"),
        mean_gpu_power_mean_watts=("mean_gpu_power_watts", "mean"),
    ).reset_index()

    grouped["wall_clock_cv_percent"] = 100 * grouped["wall_clock_sd_s"] / grouped["wall_clock_mean_s"]
    grouped["throughput_cv_percent"] = 100 * grouped["throughput_sd_seq_s"] / grouped["throughput_mean_seq_s"]
    grouped["peak_rss_cv_percent"] = 100 * grouped["peak_rss_sd_gib"] / grouped["peak_rss_mean_gib"]
    grouped["peak_gpu_mem_cv_percent"] = 100 * grouped["peak_gpu_mem_sd_gib"] / grouped["peak_gpu_mem_mean_gib"]
    return grouped


def overall_summary(runs: pd.DataFrame, env: dict, program_name: str) -> dict:
    coeffs = np.polyfit(runs["n_sequences"], runs["wall_clock_s"], 1)
    predicted = np.polyval(coeffs, runs["n_sequences"])
    ss_res = float(((runs["wall_clock_s"] - predicted) ** 2).sum())
    ss_tot = float(((runs["wall_clock_s"] - runs["wall_clock_s"].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0

    peak_rss_idx = runs["peak_rss_bytes"].idxmax()
    peak_gpu_idx = runs["peak_process_tree_gpu_mem_bytes"].idxmax()
    peak_cpu_idx = runs["max_cpu_percent"].idxmax()
    peak_power_idx = runs["peak_power_watts"].idxmax()

    return {
        "program_name": program_name,
        "environment": env,
        "benchmark": {
            "run_count": int(len(runs)),
            "unique_input_sizes": int(runs["n_sequences"].nunique()),
            "replicates_per_input_size": int(runs.groupby("n_sequences")["rep"].count().mode().iat[0]),
            "total_sequences_processed": int(runs["n_sequences"].sum()),
            "average_sequences_per_run": float(runs["n_sequences"].mean()),
            "median_sequences_per_run": float(runs["n_sequences"].median()),
            "all_runs_succeeded": bool((runs["returncode"].eq(0) & ~runs["timed_out"]).all()),
        },
        "runtime": {
            "mean_wall_clock_s": float(runs["wall_clock_s"].mean()),
            "median_wall_clock_s": float(runs["wall_clock_s"].median()),
            "min_wall_clock_s": float(runs["wall_clock_s"].min()),
            "max_wall_clock_s": float(runs["wall_clock_s"].max()),
            "mean_throughput_seq_s": float(runs["throughput_seq_s"].mean()),
            "median_throughput_seq_s": float(runs["throughput_seq_s"].median()),
            "min_throughput_seq_s": float(runs["throughput_seq_s"].min()),
            "max_throughput_seq_s": float(runs["throughput_seq_s"].max()),
            "runtime_vs_sequences_linear_fit": {
                "seconds_per_sequence": float(coeffs[0]),
                "intercept_seconds": float(coeffs[1]),
                "r_squared": float(r2),
            },
        },
        "cpu": {
            "mean_avg_cpu_percent": float(runs["avg_cpu_percent"].mean()),
            "max_peak_cpu_percent": float(runs["max_cpu_percent"].max()),
            "max_peak_cpu_run": runs.loc[peak_cpu_idx, "file"],
            "mean_peak_rss_gib": float(runs["peak_rss_gib"].mean()),
            "max_peak_rss_gib": float(runs["peak_rss_gib"].max()),
            "max_peak_rss_run": runs.loc[peak_rss_idx, "file"],
            "mean_peak_threads": float(runs["peak_threads"].mean()),
            "max_peak_threads": int(runs["peak_threads"].max()),
        },
        "gpu": {
            "gpu_available": bool(runs["gpu_available"].fillna(False).all()),
            "mean_peak_gpu_memory_gib": float(runs["peak_process_tree_gpu_mem_gib"].mean()),
            "max_peak_gpu_memory_gib": float(runs["peak_process_tree_gpu_mem_gib"].max()),
            "max_peak_gpu_memory_run": runs.loc[peak_gpu_idx, "file"],
            "max_peak_device_memory_gib": float(runs["peak_device_memory_used_gib"].max()),
            "mean_peak_gpu_util_percent": float(runs["peak_gpu_util_percent"].mean()),
            "max_peak_gpu_util_percent": float(runs["peak_gpu_util_percent"].max()),
            "mean_peak_power_watts": float(runs["peak_power_watts"].mean()),
            "max_peak_power_watts": float(runs["peak_power_watts"].max()),
            "max_peak_power_run": runs.loc[peak_power_idx, "file"],
            "mean_observed_gpu_util_percent": float(runs["mean_gpu_util_percent"].mean()),
            "mean_observed_gpu_memory_gib": float(runs["mean_gpu_process_mem_gib"].mean()),
            "mean_observed_gpu_power_watts": float(runs["mean_gpu_power_watts"].mean()),
        },
    }


def save_figure(out_path: Path) -> None:
    suffix = out_path.suffix.lower()
    if suffix == ".pdf":
        plt.savefig(out_path, bbox_inches="tight")
    else:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_time_series_by_size(
    samples: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_path: Path,
    y_transform=None,
) -> None:
    plot_df = samples[["n_sequences", "rep", "t_rel_s", value_col]].dropna().copy()
    plot_df["t_bin_s"] = (plot_df["t_rel_s"] / 0.5).round() * 0.5
    grouped = plot_df.groupby(["n_sequences", "t_bin_s"])[value_col].mean().reset_index()

    plt.figure(figsize=(11, 6.5))
    for n_sequences in sorted(grouped["n_sequences"].unique()):
        sub = grouped[grouped["n_sequences"] == n_sequences]
        y = sub[value_col].to_numpy(dtype=float)
        if y_transform is not None:
            y = y_transform(y)
        plt.plot(sub["t_bin_s"], y, linewidth=1.8, label=f"n={n_sequences}")
    plt.xlabel("Time since start (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    save_figure(out_path)


def plot_runtime_vs_sequences(runs: pd.DataFrame, out_path: Path, program_name: str) -> None:
    means = runs.groupby("n_sequences", as_index=False).agg(
        wall_clock_mean_s=("wall_clock_s", "mean"),
        wall_clock_sd_s=("wall_clock_s", "std"),
    )

    plt.figure(figsize=(9, 6))
    plt.scatter(runs["n_sequences"], runs["wall_clock_s"], alpha=0.7, label="individual runs")
    plt.plot(means["n_sequences"], means["wall_clock_mean_s"], marker="o", linewidth=2, label="mean across reps")
    plt.xlabel("Number of sequences")
    plt.ylabel("Wall clock time (s)")
    plt.title(f"{program_name} runtime scaling")
    plt.legend()
    plt.tight_layout()
    save_figure(out_path)


def plot_throughput_vs_sequences(runs: pd.DataFrame, out_path: Path, program_name: str) -> None:
    means = runs.groupby("n_sequences", as_index=False).agg(
        throughput_mean_seq_s=("throughput_seq_s", "mean"),
        throughput_sd_seq_s=("throughput_seq_s", "std"),
    )

    plt.figure(figsize=(9, 6))
    plt.scatter(runs["n_sequences"], runs["throughput_seq_s"], alpha=0.7, label="individual runs")
    plt.plot(means["n_sequences"], means["throughput_mean_seq_s"], marker="o", linewidth=2, label="mean across reps")
    plt.xlabel("Number of sequences")
    plt.ylabel("Throughput (sequences/s)")
    plt.title(f"{program_name} throughput by input size")
    plt.legend()
    plt.tight_layout()
    save_figure(out_path)


def plot_peak_memory_vs_sequences(runs: pd.DataFrame, out_path: Path, program_name: str) -> None:
    means = runs.groupby("n_sequences", as_index=False).agg(
        peak_rss_mean_gib=("peak_rss_gib", "mean"),
        peak_gpu_mem_mean_gib=("peak_process_tree_gpu_mem_gib", "mean"),
    )

    plt.figure(figsize=(9, 6))
    plt.plot(means["n_sequences"], means["peak_rss_mean_gib"], marker="o", linewidth=2, label="CPU RSS peak")
    plt.plot(means["n_sequences"], means["peak_gpu_mem_mean_gib"], marker="o", linewidth=2, label="GPU process-memory peak")
    plt.xlabel("Number of sequences")
    plt.ylabel("Peak memory (GiB)")
    plt.title(f"{program_name} peak CPU/GPU memory by input size")
    plt.legend()
    plt.tight_layout()
    save_figure(out_path)


def write_markdown_summary(summary: dict, out_path: Path) -> None:
    bench = summary["benchmark"]
    runtime = summary["runtime"]
    cpu = summary["cpu"]
    gpu = summary["gpu"]
    env = summary["environment"]
    program_name = summary["program_name"]

    md = f"""# {program_name} benchmark summary

## Environment
- CPUs: {env['physical_cpu_count']} physical / {env['logical_cpu_count']} logical
- System memory: {env['system_memory_gib']:.2f} GiB
- GPU: {env['gpu_name']} ({env['gpu_memory_gib']:.2f} GiB)
- Sampling interval: {env['interval_seconds']} s

## Coverage
- Runs: {bench['run_count']}
- Unique input sizes: {bench['unique_input_sizes']}
- Replicates per size: {bench['replicates_per_input_size']}
- Total sequences processed: {bench['total_sequences_processed']}
- Average sequences per run: {bench['average_sequences_per_run']:.1f}
- All runs succeeded: {bench['all_runs_succeeded']}

## Runtime
- Mean wall clock: {runtime['mean_wall_clock_s']:.2f} s
- Median wall clock: {runtime['median_wall_clock_s']:.2f} s
- Range: {runtime['min_wall_clock_s']:.2f}–{runtime['max_wall_clock_s']:.2f} s
- Mean throughput: {runtime['mean_throughput_seq_s']:.2f} sequences/s
- Median throughput: {runtime['median_throughput_seq_s']:.2f} sequences/s
- Runtime linear fit: {runtime['runtime_vs_sequences_linear_fit']['seconds_per_sequence']:.6f} s/sequence, intercept {runtime['runtime_vs_sequences_linear_fit']['intercept_seconds']:.2f} s, R²={runtime['runtime_vs_sequences_linear_fit']['r_squared']:.4f}

## CPU
- Mean average CPU: {cpu['mean_avg_cpu_percent']:.2f}%
- Maximum peak CPU: {cpu['max_peak_cpu_percent']:.2f}% ({cpu['max_peak_cpu_run']})
- Mean peak RSS: {cpu['mean_peak_rss_gib']:.2f} GiB
- Maximum peak RSS: {cpu['max_peak_rss_gib']:.2f} GiB ({cpu['max_peak_rss_run']})
- Maximum peak threads: {cpu['max_peak_threads']}

## GPU
- Mean observed GPU utilization: {gpu['mean_observed_gpu_util_percent']:.2f}%
- Mean peak GPU memory: {gpu['mean_peak_gpu_memory_gib']:.2f} GiB
- Maximum peak GPU memory: {gpu['max_peak_gpu_memory_gib']:.2f} GiB ({gpu['max_peak_gpu_memory_run']})
- Maximum device memory used: {gpu['max_peak_device_memory_gib']:.2f} GiB
- Mean peak GPU utilization: {gpu['mean_peak_gpu_util_percent']:.2f}%
- Maximum peak GPU utilization: {gpu['max_peak_gpu_util_percent']:.2f}%
- Mean observed GPU power: {gpu['mean_observed_gpu_power_watts']:.2f} W
- Maximum peak GPU power: {gpu['max_peak_power_watts']:.2f} W ({gpu['max_peak_power_run']})
"""
    out_path.write_text(md)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze resource benchmark JSON logs from a tar.gz archive.")
    parser.add_argument("tarball", type=Path, help="Path to benchmark tar.gz archive")
    parser.add_argument("output_dir", type=Path, help="Directory for plots and summary tables")
    parser.add_argument("--program-name", default="Program", help="Program name to show in charts and tables")
    parser.add_argument(
        "--figure-format",
        default="pdf",
        choices=["pdf", "png"],
        help="Output format for figures (default: pdf)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = output_dir / "_extracted"
    figure_suffix = args.figure_format.lower()

    extract_archive(args.tarball, extract_dir)
    runs, samples, env = parse_runs(extract_dir)
    runs, samples = add_human_columns(runs, samples)

    runs.insert(0, "program_name", args.program_name)
    by_size = summarize_by_size(runs)
    by_size.insert(0, "program_name", args.program_name)
    summary = overall_summary(runs, env, args.program_name)
    summary["figure_format"] = figure_suffix

    runs.to_csv(output_dir / "run_summary.csv", index=False)
    by_size.to_csv(output_dir / "summary_by_size.csv", index=False)
    with (output_dir / "overall_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    write_markdown_summary(summary, output_dir / "overall_summary.md")

    plot_time_series_by_size(
        samples=samples,
        value_col="cpu_percent",
        ylabel="CPU utilization (% of one core)",
        title=f"{args.program_name} CPU utilization over time",
        out_path=output_dir / f"cpu_utilization_over_time.{figure_suffix}",
    )
    plot_time_series_by_size(
        samples=samples,
        value_col="gpu_util_percent",
        ylabel="GPU utilization (%)",
        title=f"{args.program_name} GPU utilization over time",
        out_path=output_dir / f"gpu_utilization_over_time.{figure_suffix}",
    )
    plot_time_series_by_size(
        samples=samples,
        value_col="rss_gib",
        ylabel="RSS memory (GiB)",
        title=f"{args.program_name} CPU memory usage over time",
        out_path=output_dir / f"cpu_memory_rss_over_time.{figure_suffix}",
    )
    plot_time_series_by_size(
        samples=samples,
        value_col="gpu_process_mem_gib",
        ylabel="GPU process memory (GiB)",
        title=f"{args.program_name} GPU memory usage over time",
        out_path=output_dir / f"gpu_memory_over_time.{figure_suffix}",
    )
    plot_runtime_vs_sequences(runs, output_dir / f"runtime_vs_sequences.{figure_suffix}", args.program_name)
    plot_throughput_vs_sequences(runs, output_dir / f"throughput_vs_sequences.{figure_suffix}", args.program_name)
    plot_peak_memory_vs_sequences(runs, output_dir / f"peak_memory_vs_sequences.{figure_suffix}", args.program_name)


if __name__ == "__main__":
    main()

