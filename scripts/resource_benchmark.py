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
from scipy.optimize import curve_fit


PATTERN = re.compile(r"sample_n(?P<n>\d+)_rep(?P<rep>\d+)\.json$")


def bytes_to_gib(value: float | int | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value) / (1024 ** 3)


def finite_float_or_none(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def finite_int_or_none(value) -> int | None:
    if value is None or pd.isna(value):
        return None
    value = float(value)
    return int(value) if np.isfinite(value) else None


def idxmax_or_none(series: pd.Series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return values.idxmax()


def run_name_or_none(runs: pd.DataFrame, idx) -> str | None:
    if idx is None or pd.isna(idx):
        return None
    return str(runs.loc[idx, "file"])


def fmt_float(value, digits: int = 2, suffix: str = "") -> str:
    value = finite_float_or_none(value)
    if value is None:
        return "not available"
    return f"{value:.{digits}f}{suffix}"


def fmt_run(value) -> str:
    return str(value) if value is not None and not pd.isna(value) else "not available"


def fit_runtime_scaling(runs: pd.DataFrame) -> dict:
    """Fit wall-clock runtime as a linear function of input sequence count.

    Model:
        wall_clock_s = intercept_seconds + seconds_per_sequence * n_sequences

    The saturated throughput is the reciprocal of the fitted slope. This is
    intended to estimate the large-input processing rate after fixed startup
    overhead becomes negligible.
    """
    fit_df = runs[["n_sequences", "wall_clock_s"]].dropna().copy()
    fit_df = fit_df[np.isfinite(fit_df["n_sequences"]) & np.isfinite(fit_df["wall_clock_s"])]

    if len(fit_df) < 2 or fit_df["n_sequences"].nunique() < 2:
        return {
            "seconds_per_sequence": None,
            "intercept_seconds": None,
            "r_squared": None,
            "saturated_throughput_seq_s": None,
            "fit_n_runs": int(len(fit_df)),
            "fit_n_input_sizes": int(fit_df["n_sequences"].nunique()),
        }

    coeffs = np.polyfit(fit_df["n_sequences"], fit_df["wall_clock_s"], 1)
    seconds_per_sequence = float(coeffs[0])
    intercept_seconds = float(coeffs[1])
    predicted = np.polyval(coeffs, fit_df["n_sequences"])
    ss_res = float(((fit_df["wall_clock_s"] - predicted) ** 2).sum())
    ss_tot = float(((fit_df["wall_clock_s"] - fit_df["wall_clock_s"].mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot else 1.0
    saturated_throughput_seq_s = (
        1.0 / seconds_per_sequence
        if seconds_per_sequence > 0
        else None
    )

    return {
        "seconds_per_sequence": seconds_per_sequence,
        "intercept_seconds": intercept_seconds,
        "r_squared": float(r_squared),
        "saturated_throughput_seq_s": saturated_throughput_seq_s,
        "fit_n_runs": int(len(fit_df)),
        "fit_n_input_sizes": int(fit_df["n_sequences"].nunique()),
    }



def safe_aic_bic(ss_res: float, n_obs: int, n_params: int) -> tuple[float | None, float | None]:
    """Return AIC/BIC for least-squares fits using residual sum of squares."""
    if n_obs <= 0 or n_params <= 0 or ss_res < 0:
        return None, None
    if ss_res == 0:
        return float("-inf"), float("-inf")
    aic = n_obs * np.log(ss_res / n_obs) + 2 * n_params
    bic = n_obs * np.log(ss_res / n_obs) + n_params * np.log(n_obs)
    return float(aic), float(bic)


def r_squared_from_observed_predicted(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    """Compute ordinary R² for observed and predicted vectors."""
    if observed.size == 0 or predicted.size == 0 or observed.size != predicted.size:
        return None
    ss_res = float(np.sum((observed - predicted) ** 2))
    ss_tot = float(np.sum((observed - np.mean(observed)) ** 2))
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else None
    return float(1.0 - ss_res / ss_tot)


def exponential_saturation_model(n, saturated_throughput, k_sequences):
    """Empirical saturation model: y = T_sat * (1 - exp(-n / k))."""
    return saturated_throughput * (1.0 - np.exp(-n / k_sequences))


def hyperbolic_saturation_model(n, saturated_throughput, k_sequences):
    """Fixed-overhead-like saturation model: y = T_sat * n / (K + n)."""
    return saturated_throughput * n / (k_sequences + n)


def fit_one_throughput_saturation_model(
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    model_func,
) -> dict:
    """Fit one two-parameter throughput saturation model."""
    if x.size < 4 or np.unique(x).size < 3:
        return {
            "available": False,
            "model": model_name,
            "reason": "Need at least 4 valid runs and 3 unique input sizes.",
            "saturated_throughput_seq_s": None,
            "k_sequences": None,
            "r_squared": None,
            "aic": None,
            "bic": None,
            "fit_n_runs": int(x.size),
            "fit_n_input_sizes": int(np.unique(x).size),
        }

    max_y = float(np.max(y))
    median_x = float(np.median(x))
    if max_y <= 0 or median_x <= 0:
        return {
            "available": False,
            "model": model_name,
            "reason": "Input sizes and observed throughputs must be positive.",
            "saturated_throughput_seq_s": None,
            "k_sequences": None,
            "r_squared": None,
            "aic": None,
            "bic": None,
            "fit_n_runs": int(x.size),
            "fit_n_input_sizes": int(np.unique(x).size),
        }

    # T_sat should usually be above the largest observed throughput. Keeping the
    # lower bound slightly below max_y helps the optimizer when data are noisy.
    p0 = [max_y * 1.25, median_x]
    lower = [max_y * 0.5, 1e-9]
    upper = [np.inf, np.inf]

    try:
        popt, pcov = curve_fit(
            model_func,
            x,
            y,
            p0=p0,
            bounds=(lower, upper),
            maxfev=10000,
        )
        predicted = model_func(x, *popt)
        ss_res = float(np.sum((y - predicted) ** 2))
        r_squared = r_squared_from_observed_predicted(y, predicted)
        aic, bic = safe_aic_bic(ss_res, int(y.size), int(len(popt)))
        standard_errors = []
        if pcov is not None and np.all(np.isfinite(pcov)):
            standard_errors = np.sqrt(np.diag(pcov)).tolist()

        return {
            "available": True,
            "model": model_name,
            "saturated_throughput_seq_s": float(popt[0]),
            "k_sequences": float(popt[1]),
            "r_squared": r_squared,
            "rss": ss_res,
            "aic": aic,
            "bic": bic,
            "parameter_standard_errors": {
                "saturated_throughput_seq_s": finite_float_or_none(standard_errors[0]) if len(standard_errors) > 0 else None,
                "k_sequences": finite_float_or_none(standard_errors[1]) if len(standard_errors) > 1 else None,
            },
            "fit_n_runs": int(x.size),
            "fit_n_input_sizes": int(np.unique(x).size),
        }
    except Exception as exc:
        return {
            "available": False,
            "model": model_name,
            "reason": str(exc),
            "saturated_throughput_seq_s": None,
            "k_sequences": None,
            "r_squared": None,
            "aic": None,
            "bic": None,
            "fit_n_runs": int(x.size),
            "fit_n_input_sizes": int(np.unique(x).size),
        }


def fit_throughput_saturation_models(runs: pd.DataFrame) -> dict:
    """Fit nonlinear saturation models directly to observed throughput.

    Models:
        exponential: y = T_sat * (1 - exp(-n / k))
        hyperbolic:  y = T_sat * n / (k + n)

    The hyperbolic model is mathematically consistent with a fixed-overhead
    runtime model. The exponential model is an empirical saturation curve.
    """
    fit_df = runs[["n_sequences", "observed_throughput_seq_s"]].dropna().copy()
    fit_df = fit_df[
        np.isfinite(fit_df["n_sequences"])
        & np.isfinite(fit_df["observed_throughput_seq_s"])
        & (fit_df["n_sequences"] > 0)
        & (fit_df["observed_throughput_seq_s"] > 0)
    ]
    x = fit_df["n_sequences"].to_numpy(dtype=float)
    y = fit_df["observed_throughput_seq_s"].to_numpy(dtype=float)

    fits = {
        "hyperbolic": fit_one_throughput_saturation_model(
            x,
            y,
            "hyperbolic",
            hyperbolic_saturation_model,
        ),
        "exponential": fit_one_throughput_saturation_model(
            x,
            y,
            "exponential",
            exponential_saturation_model,
        ),
    }

    available = {name: fit for name, fit in fits.items() if fit.get("available")}
    best_by_aic = None
    if available:
        best_by_aic = min(
            available,
            key=lambda name: (
                float("inf") if available[name].get("aic") is None else available[name]["aic"]
            ),
        )

    return {
        "fit_target": "observed_throughput_seq_s",
        "fit_n_runs": int(x.size),
        "fit_n_input_sizes": int(np.unique(x).size) if x.size else 0,
        "best_model_by_aic": best_by_aic,
        "models": fits,
    }

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

    runs["observed_throughput_seq_s"] = runs["n_sequences"] / runs["wall_clock_s"]
    # Backward-compatible alias for older downstream code and plots.
    runs["throughput_seq_s"] = runs["observed_throughput_seq_s"]

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
        observed_throughput_mean_seq_s=("observed_throughput_seq_s", "mean"),
        observed_throughput_sd_seq_s=("observed_throughput_seq_s", "std"),
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
    grouped["observed_throughput_cv_percent"] = 100 * grouped["observed_throughput_sd_seq_s"] / grouped["observed_throughput_mean_seq_s"]
    # Backward-compatible aliases for older plotting/table code.
    grouped["throughput_mean_seq_s"] = grouped["observed_throughput_mean_seq_s"]
    grouped["throughput_sd_seq_s"] = grouped["observed_throughput_sd_seq_s"]
    grouped["throughput_cv_percent"] = grouped["observed_throughput_cv_percent"]
    grouped["peak_rss_cv_percent"] = 100 * grouped["peak_rss_sd_gib"] / grouped["peak_rss_mean_gib"]
    grouped["peak_gpu_mem_cv_percent"] = 100 * grouped["peak_gpu_mem_sd_gib"] / grouped["peak_gpu_mem_mean_gib"]
    return grouped


def overall_summary(runs: pd.DataFrame, env: dict, program_name: str) -> dict:
    throughput_saturation_fits = fit_throughput_saturation_models(runs)

    peak_rss_idx = idxmax_or_none(runs["peak_rss_bytes"])
    peak_gpu_idx = idxmax_or_none(runs["peak_process_tree_gpu_mem_bytes"])
    peak_cpu_idx = idxmax_or_none(runs["max_cpu_percent"])
    peak_power_idx = idxmax_or_none(runs["peak_power_watts"])

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
            "mean_observed_throughput_seq_s": float(runs["observed_throughput_seq_s"].mean()),
            "median_observed_throughput_seq_s": float(runs["observed_throughput_seq_s"].median()),
            "min_observed_throughput_seq_s": float(runs["observed_throughput_seq_s"].min()),
            "max_observed_throughput_seq_s": float(runs["observed_throughput_seq_s"].max()),
            # Backward-compatible aliases for older report readers.
            "mean_throughput_seq_s": float(runs["observed_throughput_seq_s"].mean()),
            "median_throughput_seq_s": float(runs["observed_throughput_seq_s"].median()),
            "min_throughput_seq_s": float(runs["observed_throughput_seq_s"].min()),
            "max_throughput_seq_s": float(runs["observed_throughput_seq_s"].max()),
            "throughput_saturation_fits": throughput_saturation_fits,
        },
        "cpu": {
            "mean_avg_cpu_percent": finite_float_or_none(runs["avg_cpu_percent"].mean()),
            "max_peak_cpu_percent": finite_float_or_none(runs["max_cpu_percent"].max()),
            "max_peak_cpu_run": run_name_or_none(runs, peak_cpu_idx),
            "mean_peak_rss_gib": finite_float_or_none(runs["peak_rss_gib"].mean()),
            "max_peak_rss_gib": finite_float_or_none(runs["peak_rss_gib"].max()),
            "max_peak_rss_run": run_name_or_none(runs, peak_rss_idx),
            "mean_peak_threads": finite_float_or_none(runs["peak_threads"].mean()),
            "max_peak_threads": finite_int_or_none(runs["peak_threads"].max()),
        },
        "gpu": {
            "gpu_available": bool(runs["gpu_available"].fillna(False).all()),
            "mean_peak_gpu_memory_gib": finite_float_or_none(runs["peak_process_tree_gpu_mem_gib"].mean()),
            "max_peak_gpu_memory_gib": finite_float_or_none(runs["peak_process_tree_gpu_mem_gib"].max()),
            "max_peak_gpu_memory_run": run_name_or_none(runs, peak_gpu_idx),
            "max_peak_device_memory_gib": finite_float_or_none(runs["peak_device_memory_used_gib"].max()),
            "mean_peak_gpu_util_percent": finite_float_or_none(runs["peak_gpu_util_percent"].mean()),
            "max_peak_gpu_util_percent": finite_float_or_none(runs["peak_gpu_util_percent"].max()),
            "mean_peak_power_watts": finite_float_or_none(runs["peak_power_watts"].mean()),
            "max_peak_power_watts": finite_float_or_none(runs["peak_power_watts"].max()),
            "max_peak_power_run": run_name_or_none(runs, peak_power_idx),
            "mean_observed_gpu_util_percent": finite_float_or_none(runs["mean_gpu_util_percent"].mean()),
            "mean_observed_gpu_memory_gib": finite_float_or_none(runs["mean_gpu_process_mem_gib"].mean()),
            "mean_observed_gpu_power_watts": finite_float_or_none(runs["mean_gpu_power_watts"].mean()),
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
        observed_throughput_mean_seq_s=("observed_throughput_seq_s", "mean"),
        observed_throughput_sd_seq_s=("observed_throughput_seq_s", "std"),
    )
    throughput_fits = fit_throughput_saturation_models(runs)

    plt.figure(figsize=(9, 6))
    plt.scatter(
        runs["n_sequences"],
        runs["observed_throughput_seq_s"],
        alpha=0.7,
        label="individual runs",
    )
    plt.plot(
        means["n_sequences"],
        means["observed_throughput_mean_seq_s"],
        marker="o",
        linewidth=2,
        label="mean observed throughput",
    )

    x_min = float(runs["n_sequences"].min())
    x_max = float(runs["n_sequences"].max())
    x_grid = np.linspace(0.0, x_max * 1.05, 300)

    hyperbolic = throughput_fits["models"].get("hyperbolic", {})
    if hyperbolic.get("available"):
        t_sat = hyperbolic["saturated_throughput_seq_s"]
        k_seq = hyperbolic["k_sequences"]
        plt.plot(
            x_grid,
            hyperbolic_saturation_model(x_grid, t_sat, k_seq),
            linewidth=1.8,
            label=f"hyperbolic fit T_sat={t_sat:.1f} seq/s",
        )
        plt.axhline(
            t_sat,
            linestyle="--",
            linewidth=1.4,
            label=f"hyperbolic saturation={t_sat:.1f} seq/s",
        )

    exponential = throughput_fits["models"].get("exponential", {})
    if exponential.get("available"):
        t_sat = exponential["saturated_throughput_seq_s"]
        k_seq = exponential["k_sequences"]
        plt.plot(
            x_grid,
            exponential_saturation_model(x_grid, t_sat, k_seq),
            linewidth=1.8,
            linestyle=":",
            label=f"exponential fit T_sat={t_sat:.1f} seq/s",
        )
        plt.axhline(
            t_sat,
            linestyle="--",
            linewidth=1.4,
            label=f"exponential saturation={t_sat:.1f} seq/s",
        )

    plt.xlim(left=max(0.0, x_min * 0.8), right=x_max * 1.05)
    plt.xlabel("Number of sequences")
    plt.ylabel("Observed throughput (sequences/s)")
    plt.title(f"{program_name} throughput by input size")
    plt.legend(fontsize=8)
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
    throughput_fits = runtime.get("throughput_saturation_fits") or {}
    saturation_models = throughput_fits.get("models") or {}
    hyperbolic_fit = saturation_models.get("hyperbolic") or {}
    exponential_fit = saturation_models.get("exponential") or {}

    def fit_summary_text(fit: dict) -> str:
        if not fit.get("available"):
            return f"not available ({fit.get('reason', 'fit failed')})"
        return (
            f"T_sat={fit['saturated_throughput_seq_s']:.2f} sequences/s, "
            f"K={fit['k_sequences']:.2f} sequences, "
            f"R²={fit['r_squared']:.4f}, "
            f"AIC={fit['aic']:.2f}"
        )

    md = f"""# {program_name} benchmark summary

## Environment
- CPUs: {env['physical_cpu_count']} physical / {env['logical_cpu_count']} logical
- System memory: {env['system_memory_gib']:.2f} GiB
- GPU: {env.get('gpu_name') or 'not available'} ({fmt_float(env.get('gpu_memory_gib'), 2, ' GiB')})
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
- Mean observed throughput: {runtime['mean_observed_throughput_seq_s']:.2f} sequences/s
- Median observed throughput: {runtime['median_observed_throughput_seq_s']:.2f} sequences/s
- Hyperbolic throughput saturation fit: {fit_summary_text(hyperbolic_fit)}
- Exponential throughput saturation fit: {fit_summary_text(exponential_fit)}
- Best nonlinear throughput model by AIC: {throughput_fits.get('best_model_by_aic') or 'not available'}

## CPU
- Mean average CPU: {fmt_float(cpu['mean_avg_cpu_percent'], 2, '%')}
- Maximum peak CPU: {fmt_float(cpu['max_peak_cpu_percent'], 2, '%')} ({fmt_run(cpu['max_peak_cpu_run'])})
- Mean peak RSS: {fmt_float(cpu['mean_peak_rss_gib'], 2, ' GiB')}
- Maximum peak RSS: {fmt_float(cpu['max_peak_rss_gib'], 2, ' GiB')} ({fmt_run(cpu['max_peak_rss_run'])})
- Maximum peak threads: {fmt_run(cpu['max_peak_threads'])}

## GPU
- GPU metrics available: {gpu['gpu_available']}
- Mean observed GPU utilization: {fmt_float(gpu['mean_observed_gpu_util_percent'], 2, '%')}
- Mean peak GPU memory: {fmt_float(gpu['mean_peak_gpu_memory_gib'], 2, ' GiB')}
- Maximum peak GPU memory: {fmt_float(gpu['max_peak_gpu_memory_gib'], 2, ' GiB')} ({fmt_run(gpu['max_peak_gpu_memory_run'])})
- Maximum device memory used: {fmt_float(gpu['max_peak_device_memory_gib'], 2, ' GiB')}
- Mean peak GPU utilization: {fmt_float(gpu['mean_peak_gpu_util_percent'], 2, '%')}
- Maximum peak GPU utilization: {fmt_float(gpu['max_peak_gpu_util_percent'], 2, '%')}
- Mean observed GPU power: {fmt_float(gpu['mean_observed_gpu_power_watts'], 2, ' W')}
- Maximum peak GPU power: {fmt_float(gpu['max_peak_power_watts'], 2, ' W')} ({fmt_run(gpu['max_peak_power_run'])})
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


