#!/usr/bin/env python3
"""
Wrap a command, sample resource usage at a fixed interval, and write a JSON report.

Features
--------
- Tracks the launched process *and its descendants*.
- Samples wall time, CPU usage, memory usage, thread count, file descriptors / handles,
  context switches, and disk I/O.
- Optionally tracks NVIDIA GPU metrics through NVML (via `nvidia-ml-py`, imported as
  the `pynvml` module).
- Writes both raw samples and a run summary to JSON after the command finishes.

Example
-------
  python command_resource_monitor.py \
      --interval 0.2 \
      --output palmsite_run.json \
      -- python palmsite.py --input data.txt

For shell commands:
  python command_resource_monitor.py \
      --shell \
      --output run.json \
      -- "python palmsite.py --input data.txt | tee log.txt"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psutil

try:
    import pynvml as nvml  # provided by the official `nvidia-ml-py` package
except Exception as exc:  # pragma: no cover - depends on optional package
    nvml = None
    _NVML_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
else:
    _NVML_IMPORT_ERROR = None

MONITOR_VERSION = "1.0.0"
PSUTIL_EXCEPTIONS = (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied)


@dataclass(frozen=True)
class ProcKey:
    pid: int
    create_time: float


@dataclass
class TrackedProc:
    key: ProcKey
    proc: psutil.Process
    last_cpu_user_s: float = 0.0
    last_cpu_system_s: float = 0.0
    last_io: Dict[str, int] = field(default_factory=dict)
    last_ctx_voluntary: int = 0
    last_ctx_involuntary: int = 0
    finalized: bool = False


@dataclass
class DeadTotals:
    cpu_user_s: float = 0.0
    cpu_system_s: float = 0.0
    io_read_count: int = 0
    io_write_count: int = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    io_read_chars: int = 0
    io_write_chars: int = 0
    io_other_count: int = 0
    io_other_bytes: int = 0
    ctx_voluntary: int = 0
    ctx_involuntary: int = 0


@dataclass
class SummaryAccumulator:
    peak_rss_bytes: int = 0
    peak_vms_bytes: int = 0
    peak_uss_bytes: Optional[int] = None
    peak_num_threads: int = 0
    peak_num_fds: Optional[int] = None
    peak_num_handles: Optional[int] = None
    peak_live_processes: int = 0
    peak_memory_percent_of_system: float = 0.0
    peak_gpu_process_tree_memory_by_index: Dict[int, int] = field(default_factory=dict)
    peak_gpu_device_memory_used_by_index: Dict[int, int] = field(default_factory=dict)
    peak_gpu_util_by_index: Dict[int, float] = field(default_factory=dict)
    peak_gpu_power_watts_by_index: Dict[int, float] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def decode_if_bytes(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return repr(value)
    return value


def round_float(value: Optional[float], digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return round(float(value), digits)


def safe_call(func, default=None):
    try:
        return func()
    except Exception:
        return default


def make_proc_key(proc: psutil.Process) -> Optional[ProcKey]:
    try:
        return ProcKey(pid=proc.pid, create_time=proc.create_time())
    except PSUTIL_EXCEPTIONS:
        return None


def same_process_identity(proc: psutil.Process, key: ProcKey) -> bool:
    try:
        return proc.pid == key.pid and abs(proc.create_time() - key.create_time) < 1e-9
    except PSUTIL_EXCEPTIONS:
        return False


def is_effectively_alive(proc: psutil.Process, key: ProcKey) -> bool:
    try:
        if not proc.is_running():
            return False
        if not same_process_identity(proc, key):
            return False
        status = proc.status()
        if status == psutil.STATUS_ZOMBIE:
            return False
        return True
    except PSUTIL_EXCEPTIONS:
        return False


def command_to_string(command: Sequence[str], shell: bool) -> str:
    if shell:
        return command[0] if command else ""
    if os.name == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(list(command))


def safe_swap_total_bytes() -> Optional[int]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        swap = safe_call(psutil.swap_memory, default=None)
    if swap is None:
        return None
    try:
        return int(swap.total)
    except Exception:
        return None


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command and write sampled resource usage to JSON."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Sampling interval in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the JSON report to write.",
    )
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Run the command through the shell. Use with care.",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for the wrapped command.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional timeout in seconds. If exceeded, the process tree is terminated.",
    )
    parser.add_argument(
        "--grace-period",
        type=float,
        default=5.0,
        help="Seconds to wait between terminate and kill when stopping the process tree (default: 5).",
    )
    parser.add_argument(
        "--include-uss",
        action="store_true",
        help="Sample USS (unique set size) when available. This can be slower than RSS/VMS.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run. Prefer using `--` before the command.",
    )
    args = parser.parse_args(argv)

    if args.interval <= 0:
        parser.error("--interval must be > 0")
    if args.timeout is not None and args.timeout <= 0:
        parser.error("--timeout must be > 0")
    if args.grace_period < 0:
        parser.error("--grace-period must be >= 0")

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("No command specified. Example: -- python palmsite.py")
    if args.shell and len(command) != 1:
        parser.error("With --shell, provide the shell command as a single string after --")
    args.command = command
    return args


class GPUMonitor:
    def __init__(self) -> None:
        self.available = False
        self.reason: Optional[str] = None
        self.device_count = 0
        self.static_devices: List[Dict[str, Any]] = []
        self._initialized = False

        if nvml is None:
            self.reason = (
                "NVML Python bindings not importable. Install `nvidia-ml-py` to enable GPU metrics"
                + (f" ({_NVML_IMPORT_ERROR})" if _NVML_IMPORT_ERROR else "")
            )
            return

        try:
            nvml.nvmlInit()
            self._initialized = True
            self.device_count = int(nvml.nvmlDeviceGetCount())
            for index in range(self.device_count):
                try:
                    handle = nvml.nvmlDeviceGetHandleByIndex(index)
                    mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                    self.static_devices.append(
                        {
                            "index": index,
                            "name": decode_if_bytes(nvml.nvmlDeviceGetName(handle)),
                            "uuid": decode_if_bytes(nvml.nvmlDeviceGetUUID(handle)),
                            "memory_total_bytes": int(getattr(mem, "total", 0)),
                        }
                    )
                except Exception as exc:  # pragma: no cover - hardware dependent
                    self.static_devices.append(
                        {
                            "index": index,
                            "name": None,
                            "uuid": None,
                            "memory_total_bytes": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
            self.available = True
        except Exception as exc:  # pragma: no cover - hardware dependent
            self.reason = f"Failed to initialize NVML: {type(exc).__name__}: {exc}"
            self.available = False

    def close(self) -> None:
        if self._initialized:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass
            self._initialized = False

    def _get_process_lists(self, handle) -> Dict[int, Optional[int]]:
        if nvml is None:
            return {}

        pid_to_used_mem: Dict[int, Optional[int]] = {}
        function_names = [
            "nvmlDeviceGetComputeRunningProcesses_v3",
            "nvmlDeviceGetComputeRunningProcesses_v2",
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses_v3",
            "nvmlDeviceGetGraphicsRunningProcesses_v2",
            "nvmlDeviceGetGraphicsRunningProcesses",
            "nvmlDeviceGetMPSComputeRunningProcesses_v3",
            "nvmlDeviceGetMPSComputeRunningProcesses_v2",
            "nvmlDeviceGetMPSComputeRunningProcesses",
        ]
        not_available = getattr(nvml, "NVML_VALUE_NOT_AVAILABLE", None)

        for func_name in function_names:
            func = getattr(nvml, func_name, None)
            if func is None:
                continue
            try:
                entries = func(handle)
            except Exception:
                continue
            for entry in entries or []:
                pid = int(getattr(entry, "pid", -1))
                if pid < 0:
                    continue
                used = getattr(entry, "usedGpuMemory", None)
                if used == not_available:
                    used = None
                elif used is not None:
                    try:
                        used = int(used)
                    except Exception:
                        used = None
                if pid not in pid_to_used_mem:
                    pid_to_used_mem[pid] = used
                else:
                    prev = pid_to_used_mem[pid]
                    if prev is None:
                        pid_to_used_mem[pid] = used
                    elif used is None:
                        pid_to_used_mem[pid] = prev
                    else:
                        pid_to_used_mem[pid] = max(prev, used)
        return pid_to_used_mem

    def _try_metric(self, func, transform=None):
        try:
            value = func()
        except Exception:
            return None
        if transform is not None:
            try:
                value = transform(value)
            except Exception:
                return None
        return value

    def sample(self, process_tree_pids: Iterable[int]) -> Dict[str, Any]:
        pids = set(int(pid) for pid in process_tree_pids)
        if not self.available or nvml is None:
            return {
                "available": False,
                "reason": self.reason,
                "devices": [],
            }

        devices: List[Dict[str, Any]] = []
        for index in range(self.device_count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
            except Exception as exc:  # pragma: no cover - hardware dependent
                devices.append(
                    {
                        "index": index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            static_info = self.static_devices[index] if index < len(self.static_devices) else {}
            mem = self._try_metric(lambda: nvml.nvmlDeviceGetMemoryInfo(handle))
            util = self._try_metric(lambda: nvml.nvmlDeviceGetUtilizationRates(handle))
            process_mem_by_pid = self._get_process_lists(handle)
            matching_pids = sorted(pid for pid in process_mem_by_pid if pid in pids)
            process_tree_mem = 0
            process_tree_mem_unknown = False
            for pid in matching_pids:
                used = process_mem_by_pid[pid]
                if used is None:
                    process_tree_mem_unknown = True
                else:
                    process_tree_mem += int(used)

            device_record: Dict[str, Any] = {
                "index": index,
                "name": static_info.get("name"),
                "uuid": static_info.get("uuid"),
                "device_memory_total_bytes": int(getattr(mem, "total", 0)) if mem is not None else static_info.get("memory_total_bytes"),
                "device_memory_used_bytes": int(getattr(mem, "used", 0)) if mem is not None else None,
                "device_memory_free_bytes": int(getattr(mem, "free", 0)) if mem is not None else None,
                "device_utilization_gpu_percent": int(getattr(util, "gpu", 0)) if util is not None else None,
                "device_utilization_memory_percent": int(getattr(util, "memory", 0)) if util is not None else None,
                "process_tree_gpu_memory_bytes": process_tree_mem,
                "process_tree_gpu_memory_unknown_for_some_pids": process_tree_mem_unknown,
                "process_tree_gpu_pids": matching_pids,
                "power_watts": self._try_metric(lambda: nvml.nvmlDeviceGetPowerUsage(handle), lambda x: round(float(x) / 1000.0, 3)),
                "temperature_c": self._try_metric(lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU), int),
                "performance_state": self._try_metric(lambda: nvml.nvmlDeviceGetPerformanceState(handle), int),
                "sm_clock_mhz": self._try_metric(lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM), int),
                "mem_clock_mhz": self._try_metric(lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM), int),
                "pcie_tx_kb_per_s": self._try_metric(lambda: nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES), int),
                "pcie_rx_kb_per_s": self._try_metric(lambda: nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES), int),
            }
            devices.append(device_record)

        return {
            "available": True,
            "reason": None,
            "devices": devices,
        }


def discover_live_process_tree(
    root_proc: psutil.Process,
    root_key: ProcKey,
    tracked: Dict[ProcKey, TrackedProc],
) -> Dict[ProcKey, psutil.Process]:
    discovered: Dict[ProcKey, psutil.Process] = {}
    queue: List[psutil.Process] = []

    if is_effectively_alive(root_proc, root_key):
        queue.append(root_proc)
    else:
        for item in tracked.values():
            if is_effectively_alive(item.proc, item.key):
                queue.append(item.proc)

    while queue:
        proc = queue.pop()
        key = make_proc_key(proc)
        if key is None or key in discovered:
            continue
        if not is_effectively_alive(proc, key):
            continue
        discovered[key] = proc
        try:
            children = proc.children()
        except PSUTIL_EXCEPTIONS:
            children = []
        for child in children:
            child_key = make_proc_key(child)
            if child_key is None or child_key in discovered:
                continue
            queue.append(child)

    return discovered


class MonitorEngine:
    def __init__(self, args: argparse.Namespace, popen: subprocess.Popen) -> None:
        self.args = args
        self.popen = popen
        self.start_monotonic = time.monotonic()
        self.start_wall_utc = utc_now_iso()
        self.start_wall_local = local_now_iso()
        self.dead_totals = DeadTotals()
        self.summary = SummaryAccumulator()
        self.samples: List[Dict[str, Any]] = []
        self.root_proc = psutil.Process(popen.pid)
        root_key = make_proc_key(self.root_proc)
        if root_key is None:
            raise RuntimeError("Failed to identify launched process")
        self.root_key = root_key
        self.tracked: Dict[ProcKey, TrackedProc] = {
            root_key: TrackedProc(key=root_key, proc=self.root_proc)
        }
        self.prev_sample_t: Optional[float] = None
        self.prev_cpu_total_s: Optional[float] = None
        self.total_system_memory_bytes = int(psutil.virtual_memory().total)
        self.logical_cpu_count = psutil.cpu_count(logical=True) or 1
        self.gpu_monitor = GPUMonitor()
        self.stop_reason: Optional[str] = None
        self.interrupted = False
        self.timed_out = False
        self._signal_received: Optional[str] = None
        self._terminate_requested = False
        self._kill_deadline_monotonic: Optional[float] = None

    def register_signal_handlers(self) -> None:
        def handler(signum, _frame):
            signame = signal.Signals(signum).name
            self._signal_received = signame
            self.interrupted = True
            self.stop_reason = f"Interrupted by {signame}"
            self.request_termination()

        for signame in ("SIGINT", "SIGTERM"):
            sig = getattr(signal, signame, None)
            if sig is not None:
                signal.signal(sig, handler)

    def request_termination(self) -> None:
        if self._terminate_requested:
            return
        self._terminate_requested = True
        self._kill_deadline_monotonic = time.monotonic() + float(self.args.grace_period)
        self._terminate_live_processes(force=False)

    def maybe_escalate_kill(self) -> None:
        if not self._terminate_requested:
            return
        if self._kill_deadline_monotonic is None:
            return
        if time.monotonic() >= self._kill_deadline_monotonic:
            self._terminate_live_processes(force=True)
            self._kill_deadline_monotonic = None

    def _terminate_live_processes(self, force: bool) -> None:
        live_items = [item for item in self.tracked.values() if is_effectively_alive(item.proc, item.key)]
        live_items.sort(key=lambda item: item.key.pid, reverse=True)
        for item in live_items:
            try:
                if force:
                    item.proc.kill()
                else:
                    item.proc.terminate()
            except PSUTIL_EXCEPTIONS:
                continue

    def _finalize_dead_processes(self, live_keys: Iterable[ProcKey]) -> None:
        live_key_set = set(live_keys)
        for item in self.tracked.values():
            if item.finalized:
                continue
            if item.key in live_key_set:
                continue
            if is_effectively_alive(item.proc, item.key):
                continue
            self.dead_totals.cpu_user_s += item.last_cpu_user_s
            self.dead_totals.cpu_system_s += item.last_cpu_system_s
            self.dead_totals.io_read_count += int(item.last_io.get("read_count", 0))
            self.dead_totals.io_write_count += int(item.last_io.get("write_count", 0))
            self.dead_totals.io_read_bytes += int(item.last_io.get("read_bytes", 0))
            self.dead_totals.io_write_bytes += int(item.last_io.get("write_bytes", 0))
            self.dead_totals.io_read_chars += int(item.last_io.get("read_chars", 0))
            self.dead_totals.io_write_chars += int(item.last_io.get("write_chars", 0))
            self.dead_totals.io_other_count += int(item.last_io.get("other_count", 0))
            self.dead_totals.io_other_bytes += int(item.last_io.get("other_bytes", 0))
            self.dead_totals.ctx_voluntary += item.last_ctx_voluntary
            self.dead_totals.ctx_involuntary += item.last_ctx_involuntary
            item.finalized = True

    def collect_sample(self) -> None:
        live_tree = discover_live_process_tree(self.root_proc, self.root_key, self.tracked)
        for key, proc in live_tree.items():
            if key not in self.tracked:
                self.tracked[key] = TrackedProc(key=key, proc=proc)
            else:
                self.tracked[key].proc = proc

        self._finalize_dead_processes(live_tree.keys())

        live_proc_records: List[Dict[str, Any]] = []
        current_cpu_user_s = self.dead_totals.cpu_user_s
        current_cpu_system_s = self.dead_totals.cpu_system_s
        current_io_read_count = self.dead_totals.io_read_count
        current_io_write_count = self.dead_totals.io_write_count
        current_io_read_bytes = self.dead_totals.io_read_bytes
        current_io_write_bytes = self.dead_totals.io_write_bytes
        current_io_read_chars = self.dead_totals.io_read_chars
        current_io_write_chars = self.dead_totals.io_write_chars
        current_io_other_count = self.dead_totals.io_other_count
        current_io_other_bytes = self.dead_totals.io_other_bytes
        current_ctx_voluntary = self.dead_totals.ctx_voluntary
        current_ctx_involuntary = self.dead_totals.ctx_involuntary

        current_rss_bytes = 0
        current_vms_bytes = 0
        current_uss_bytes: Optional[int] = 0 if self.args.include_uss else None
        current_num_threads = 0
        current_num_fds: Optional[int] = 0 if os.name != "nt" else None
        current_num_handles: Optional[int] = 0 if os.name == "nt" else None

        for key, proc in live_tree.items():
            tracked_item = self.tracked[key]
            try:
                with proc.oneshot():
                    cpu_times = proc.cpu_times()
                    mem = proc.memory_info()
                    io = safe_call(proc.io_counters, default=None)
                    ctx = safe_call(proc.num_ctx_switches, default=None)
                    num_threads = safe_call(proc.num_threads, default=0) or 0
                    num_fds = safe_call(proc.num_fds, default=None) if os.name != "nt" else None
                    num_handles = safe_call(proc.num_handles, default=None) if os.name == "nt" else None
                    full_mem = safe_call(proc.memory_full_info, default=None) if self.args.include_uss else None
            except PSUTIL_EXCEPTIONS:
                continue

            user_s = float(getattr(cpu_times, "user", 0.0))
            system_s = float(getattr(cpu_times, "system", 0.0))
            rss = int(getattr(mem, "rss", 0))
            vms = int(getattr(mem, "vms", 0))
            uss = None
            if full_mem is not None:
                uss = getattr(full_mem, "uss", None)
                if uss is not None:
                    uss = int(uss)

            io_dict = {
                "read_count": int(getattr(io, "read_count", 0)) if io is not None else 0,
                "write_count": int(getattr(io, "write_count", 0)) if io is not None else 0,
                "read_bytes": int(getattr(io, "read_bytes", 0)) if io is not None else 0,
                "write_bytes": int(getattr(io, "write_bytes", 0)) if io is not None else 0,
                "read_chars": int(getattr(io, "read_chars", 0)) if io is not None else 0,
                "write_chars": int(getattr(io, "write_chars", 0)) if io is not None else 0,
                "other_count": int(getattr(io, "other_count", 0)) if io is not None else 0,
                "other_bytes": int(getattr(io, "other_bytes", 0)) if io is not None else 0,
            }
            voluntary = int(getattr(ctx, "voluntary", 0)) if ctx is not None else 0
            involuntary = int(getattr(ctx, "involuntary", 0)) if ctx is not None else 0

            tracked_item.last_cpu_user_s = user_s
            tracked_item.last_cpu_system_s = system_s
            tracked_item.last_io = io_dict
            tracked_item.last_ctx_voluntary = voluntary
            tracked_item.last_ctx_involuntary = involuntary

            current_cpu_user_s += user_s
            current_cpu_system_s += system_s
            current_io_read_count += io_dict["read_count"]
            current_io_write_count += io_dict["write_count"]
            current_io_read_bytes += io_dict["read_bytes"]
            current_io_write_bytes += io_dict["write_bytes"]
            current_io_read_chars += io_dict["read_chars"]
            current_io_write_chars += io_dict["write_chars"]
            current_io_other_count += io_dict["other_count"]
            current_io_other_bytes += io_dict["other_bytes"]
            current_ctx_voluntary += voluntary
            current_ctx_involuntary += involuntary

            current_rss_bytes += rss
            current_vms_bytes += vms
            if current_uss_bytes is not None:
                current_uss_bytes += int(uss or 0)
            current_num_threads += int(num_threads)
            if current_num_fds is not None:
                current_num_fds += int(num_fds or 0)
            if current_num_handles is not None:
                current_num_handles += int(num_handles or 0)

            live_proc_records.append(
                {
                    "pid": key.pid,
                    "create_time_epoch_s": round_float(key.create_time),
                    "cpu_user_s": round_float(user_s),
                    "cpu_system_s": round_float(system_s),
                    "rss_bytes": rss,
                    "vms_bytes": vms,
                    "uss_bytes": uss,
                    "num_threads": int(num_threads),
                    "num_fds": int(num_fds) if num_fds is not None else None,
                    "num_handles": int(num_handles) if num_handles is not None else None,
                    "io": io_dict,
                    "context_switches": {
                        "voluntary": voluntary,
                        "involuntary": involuntary,
                    },
                }
            )

        t_now = time.monotonic()
        t_rel = t_now - self.start_monotonic
        cpu_total_s = current_cpu_user_s + current_cpu_system_s
        cpu_percent = None
        cpu_percent_of_total_machine = None
        if self.prev_sample_t is not None and self.prev_cpu_total_s is not None:
            delta_t = t_now - self.prev_sample_t
            delta_cpu = cpu_total_s - self.prev_cpu_total_s
            if delta_t > 0:
                cpu_percent = max(0.0, (delta_cpu / delta_t) * 100.0)
                cpu_percent_of_total_machine = cpu_percent / float(self.logical_cpu_count)

        self.prev_sample_t = t_now
        self.prev_cpu_total_s = cpu_total_s

        memory_percent_of_system = (
            (current_rss_bytes / self.total_system_memory_bytes) * 100.0
            if self.total_system_memory_bytes > 0
            else None
        )

        gpu_snapshot = self.gpu_monitor.sample([key.pid for key in live_tree.keys()])

        self.summary.peak_rss_bytes = max(self.summary.peak_rss_bytes, current_rss_bytes)
        self.summary.peak_vms_bytes = max(self.summary.peak_vms_bytes, current_vms_bytes)
        if current_uss_bytes is not None:
            if self.summary.peak_uss_bytes is None:
                self.summary.peak_uss_bytes = current_uss_bytes
            else:
                self.summary.peak_uss_bytes = max(self.summary.peak_uss_bytes, current_uss_bytes)
        self.summary.peak_num_threads = max(self.summary.peak_num_threads, current_num_threads)
        if current_num_fds is not None:
            self.summary.peak_num_fds = (
                current_num_fds
                if self.summary.peak_num_fds is None
                else max(self.summary.peak_num_fds, current_num_fds)
            )
        if current_num_handles is not None:
            self.summary.peak_num_handles = (
                current_num_handles
                if self.summary.peak_num_handles is None
                else max(self.summary.peak_num_handles, current_num_handles)
            )
        self.summary.peak_live_processes = max(self.summary.peak_live_processes, len(live_tree))
        if memory_percent_of_system is not None:
            self.summary.peak_memory_percent_of_system = max(
                self.summary.peak_memory_percent_of_system,
                memory_percent_of_system,
            )

        if gpu_snapshot.get("available"):
            for device in gpu_snapshot.get("devices", []):
                index = int(device.get("index", -1))
                if index < 0:
                    continue
                proc_mem = device.get("process_tree_gpu_memory_bytes")
                dev_used = device.get("device_memory_used_bytes")
                util_gpu = device.get("device_utilization_gpu_percent")
                power = device.get("power_watts")
                if proc_mem is not None:
                    self.summary.peak_gpu_process_tree_memory_by_index[index] = max(
                        self.summary.peak_gpu_process_tree_memory_by_index.get(index, 0),
                        int(proc_mem),
                    )
                if dev_used is not None:
                    self.summary.peak_gpu_device_memory_used_by_index[index] = max(
                        self.summary.peak_gpu_device_memory_used_by_index.get(index, 0),
                        int(dev_used),
                    )
                if util_gpu is not None:
                    self.summary.peak_gpu_util_by_index[index] = max(
                        self.summary.peak_gpu_util_by_index.get(index, 0.0),
                        float(util_gpu),
                    )
                if power is not None:
                    self.summary.peak_gpu_power_watts_by_index[index] = max(
                        self.summary.peak_gpu_power_watts_by_index.get(index, 0.0),
                        float(power),
                    )

        sample_record = {
            "timestamp_utc": utc_now_iso(),
            "t_rel_s": round_float(t_rel),
            "process": {
                "root_pid": self.root_key.pid,
                "live_process_count": len(live_tree),
                "live_pids": sorted(key.pid for key in live_tree.keys()),
                "cpu_percent": round_float(cpu_percent),
                "cpu_percent_of_total_machine": round_float(cpu_percent_of_total_machine),
                "cpu_time_user_s": round_float(current_cpu_user_s),
                "cpu_time_system_s": round_float(current_cpu_system_s),
                "cpu_time_total_s": round_float(cpu_total_s),
                "rss_bytes": current_rss_bytes,
                "vms_bytes": current_vms_bytes,
                "uss_bytes": current_uss_bytes,
                "memory_percent_of_system": round_float(memory_percent_of_system),
                "num_threads": current_num_threads,
                "num_fds": current_num_fds,
                "num_handles": current_num_handles,
                "io": {
                    "read_count": current_io_read_count,
                    "write_count": current_io_write_count,
                    "read_bytes": current_io_read_bytes,
                    "write_bytes": current_io_write_bytes,
                    "read_chars": current_io_read_chars,
                    "write_chars": current_io_write_chars,
                    "other_count": current_io_other_count,
                    "other_bytes": current_io_other_bytes,
                },
                "context_switches": {
                    "voluntary": current_ctx_voluntary,
                    "involuntary": current_ctx_involuntary,
                },
                "live_processes": live_proc_records,
            },
            "gpu": gpu_snapshot,
        }
        self.samples.append(sample_record)

    def run(self) -> Dict[str, Any]:
        self.register_signal_handlers()

        while True:
            loop_started = time.monotonic()
            if self.args.timeout is not None and (loop_started - self.start_monotonic) > self.args.timeout:
                if not self.timed_out:
                    self.timed_out = True
                    self.stop_reason = f"Timeout exceeded ({self.args.timeout} s)"
                    self.request_termination()

            self.maybe_escalate_kill()
            self.collect_sample()

            root_finished = self.popen.poll() is not None
            live_any = any(is_effectively_alive(item.proc, item.key) for item in self.tracked.values() if not item.finalized)
            if root_finished and not live_any:
                break

            elapsed = time.monotonic() - loop_started
            sleep_for = max(0.0, float(self.args.interval) - elapsed)
            time.sleep(sleep_for)

        # Finalize any processes that ended after the last sampling step.
        self._finalize_dead_processes([])

        end_monotonic = time.monotonic()
        wall_clock_seconds = end_monotonic - self.start_monotonic
        finished_utc = utc_now_iso()
        finished_local = local_now_iso()
        returncode = self.popen.returncode
        exit_signal = None
        if isinstance(returncode, int) and returncode < 0:
            try:
                exit_signal = signal.Signals(-returncode).name
            except Exception:
                exit_signal = f"SIG{-returncode}"

        cpu_series = [
            sample["process"]["cpu_percent"]
            for sample in self.samples
            if sample["process"]["cpu_percent"] is not None
        ]
        cpu_series_normalized = [
            sample["process"]["cpu_percent_of_total_machine"]
            for sample in self.samples
            if sample["process"]["cpu_percent_of_total_machine"] is not None
        ]
        final_process = self.samples[-1]["process"] if self.samples else {}
        gpu_summary_devices: List[Dict[str, Any]] = []
        if self.gpu_monitor.available:
            static_by_index = {d.get("index"): d for d in self.gpu_monitor.static_devices}
            indices = sorted({
                *self.summary.peak_gpu_process_tree_memory_by_index.keys(),
                *self.summary.peak_gpu_device_memory_used_by_index.keys(),
                *self.summary.peak_gpu_util_by_index.keys(),
                *self.summary.peak_gpu_power_watts_by_index.keys(),
            })
            for index in indices:
                static_info = static_by_index.get(index, {})
                gpu_summary_devices.append(
                    {
                        "index": index,
                        "name": static_info.get("name"),
                        "uuid": static_info.get("uuid"),
                        "peak_process_tree_gpu_memory_bytes": self.summary.peak_gpu_process_tree_memory_by_index.get(index),
                        "peak_device_memory_used_bytes": self.summary.peak_gpu_device_memory_used_by_index.get(index),
                        "peak_device_utilization_gpu_percent": round_float(self.summary.peak_gpu_util_by_index.get(index)),
                        "peak_power_watts": round_float(self.summary.peak_gpu_power_watts_by_index.get(index), 3),
                    }
                )

        report = {
            "meta": {
                "monitor_version": MONITOR_VERSION,
                "command": list(self.args.command),
                "command_str": command_to_string(self.args.command, self.args.shell),
                "shell": bool(self.args.shell),
                "cwd": self.args.cwd if self.args.cwd is not None else os.getcwd(),
                "interval_seconds": float(self.args.interval),
                "timeout_seconds": self.args.timeout,
                "include_uss": bool(self.args.include_uss),
                "started_at_utc": self.start_wall_utc,
                "started_at_local": self.start_wall_local,
                "finished_at_utc": finished_utc,
                "finished_at_local": finished_local,
                "hostname": socket.gethostname(),
                "platform": {
                    "system": platform.system(),
                    "release": platform.release(),
                    "version": platform.version(),
                    "machine": platform.machine(),
                    "processor": platform.processor(),
                    "python": sys.version,
                },
                "system_resources": {
                    "logical_cpu_count": self.logical_cpu_count,
                    "physical_cpu_count": psutil.cpu_count(logical=False),
                    "total_memory_bytes": self.total_system_memory_bytes,
                    "swap_total_bytes": safe_swap_total_bytes(),
                },
                "gpu_monitoring": {
                    "available": self.gpu_monitor.available,
                    "reason": self.gpu_monitor.reason,
                    "device_count": self.gpu_monitor.device_count,
                    "devices": self.gpu_monitor.static_devices,
                },
            },
            "result": {
                "root_pid": self.root_key.pid,
                "returncode": returncode,
                "exit_signal": exit_signal,
                "timed_out": self.timed_out,
                "interrupted": self.interrupted,
                "signal_received": self._signal_received,
                "stop_reason": self.stop_reason,
            },
            "summary": {
                "wall_clock_seconds": round_float(wall_clock_seconds),
                "sample_count": len(self.samples),
                "peak_live_processes": self.summary.peak_live_processes,
                "cpu_time_user_s": round_float(final_process.get("cpu_time_user_s")),
                "cpu_time_system_s": round_float(final_process.get("cpu_time_system_s")),
                "cpu_time_total_s": round_float(final_process.get("cpu_time_total_s")),
                "avg_cpu_percent": round_float(sum(cpu_series) / len(cpu_series)) if cpu_series else None,
                "max_cpu_percent": round_float(max(cpu_series)) if cpu_series else None,
                "avg_cpu_percent_of_total_machine": round_float(sum(cpu_series_normalized) / len(cpu_series_normalized)) if cpu_series_normalized else None,
                "max_cpu_percent_of_total_machine": round_float(max(cpu_series_normalized)) if cpu_series_normalized else None,
                "peak_rss_bytes": self.summary.peak_rss_bytes,
                "peak_vms_bytes": self.summary.peak_vms_bytes,
                "peak_uss_bytes": self.summary.peak_uss_bytes,
                "peak_memory_percent_of_system": round_float(self.summary.peak_memory_percent_of_system),
                "peak_num_threads": self.summary.peak_num_threads,
                "peak_num_fds": self.summary.peak_num_fds,
                "peak_num_handles": self.summary.peak_num_handles,
                "total_io": final_process.get("io"),
                "total_context_switches": final_process.get("context_switches"),
                "gpu": {
                    "available": self.gpu_monitor.available,
                    "reason": self.gpu_monitor.reason,
                    "devices": gpu_summary_devices,
                },
            },
            "samples": self.samples,
        }
        return report

    def close(self) -> None:
        self.gpu_monitor.close()


def launch_command(args: argparse.Namespace) -> subprocess.Popen:
    kwargs: Dict[str, Any] = {
        "cwd": args.cwd,
        "shell": bool(args.shell),
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True

    return subprocess.Popen(args.command if not args.shell else args.command[0], **kwargs)


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    popen = launch_command(args)
    engine = MonitorEngine(args, popen)

    try:
        report = engine.run()
    finally:
        engine.close()

    write_json(args.output, report)

    summary = report.get("summary", {})
    result = report.get("result", {})
    print(
        json.dumps(
            {
                "output": os.path.abspath(args.output),
                "returncode": result.get("returncode"),
                "wall_clock_seconds": summary.get("wall_clock_seconds"),
                "peak_rss_bytes": summary.get("peak_rss_bytes"),
                "peak_gpu_process_tree_memory": {
                    str(dev.get("index")): dev.get("peak_process_tree_gpu_memory_bytes")
                    for dev in summary.get("gpu", {}).get("devices", [])
                },
            },
            indent=2,
        )
    )
    return int(result.get("returncode") or 0)


if __name__ == "__main__":
    sys.exit(main())

