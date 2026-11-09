#!/usr/bin/env python
"""
Telemetry Helpers

Backend-agnostic (mps/cuda/rocm/cpu) GPU memory/compute stats and host resource
stats. On AMD systems running PyTorch ROCm
builds, memory stats are read through the same torch.cuda API as NVIDIA
(torch implements HIP behind it); compute utilization uses a best-effort
rocm-smi subprocess call. All functions are synchronous and perform blocking
calls (subprocess, psutil sampling) - async callers should run them via
loop.run_in_executor().

Host and MPS stats need psutil: install with `pip install metastable[telemetry]`.
"""

import json
import logging
import os
import shutil
import subprocess

import torch

logger = logging.getLogger(__name__)


def _psutil():
    """Import psutil lazily (optional 'telemetry' extra)."""
    try:
        import psutil
    except ImportError as e:
        raise ImportError(
            "psutil is required for host/MPS telemetry: "
            "pip install metastable[telemetry]"
        ) from e
    return psutil

# Backends addressed through the CUDA API: NVIDIA CUDA builds and AMD ROCm
# builds alike (torch.version.hip set). Memory stats come from torch.cuda.
_CUDA_API_BACKENDS = ("cuda", "rocm")


def get_gpu_memory_stats(device: str) -> dict:
    """Get GPU memory usage for the active compute backend label
    ("cuda", "rocm", "mps", "cpu")."""
    if device in _CUDA_API_BACKENDS:
        free, total = torch.cuda.mem_get_info()
        return {
            "available": True,
            "used_bytes": total - free,
            "total_bytes": total,
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "total_is_approximate": False,
        }

    if device == "mps":
        allocated = torch.mps.current_allocated_memory()
        driver_allocated = torch.mps.driver_allocated_memory()
        total = _psutil().virtual_memory().total
        return {
            "available": True,
            "used_bytes": driver_allocated,
            "total_bytes": total,
            "allocated_bytes": allocated,
            "reserved_bytes": driver_allocated,
            "total_is_approximate": True,
        }

    return {"available": False, "reason": "no GPU detected (running on CPU)"}


def get_gpu_compute_stats(device: str) -> dict:
    """Get GPU compute utilization percentage, when the platform exposes one."""
    if device == "cuda":
        return _nvidia_compute_stats()
    if device == "rocm":
        return _amd_compute_stats()

    return {
        "available": False,
        "reason": f"GPU compute utilization is not exposed by PyTorch on '{device}'",
    }


def _nvidia_compute_stats() -> dict:
    """Best-effort GPU utilization via the nvidia-smi CLI."""
    if not shutil.which("nvidia-smi"):
        return {"available": False, "reason": "nvidia-smi not found on PATH"}

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        utilization = float(result.stdout.strip().splitlines()[0])
        return {"available": True, "utilization_percent": utilization}
    except Exception as e:
        logger.warning("Failed to query nvidia-smi: %s", e)
        return {"available": False, "reason": f"nvidia-smi query failed: {e}"}


def _amd_compute_stats() -> dict:
    """Best-effort GPU utilization via the rocm-smi CLI (ROCm builds)."""
    if not shutil.which("rocm-smi"):
        return {"available": False, "reason": "rocm-smi not found on PATH"}

    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--json"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        data = json.loads(result.stdout)
        # rocm-smi JSON layout varies across ROCm releases: {"0": {"GPU use (%)":
        # "12"}} or {"card0": {...}}. Scan for the first utilization-looking key.
        for card in data.values():
            if not isinstance(card, dict):
                continue
            for key, value in card.items():
                if "use" in key.lower() and "%" in key:
                    utilization = float(str(value).strip().rstrip("%"))
                    return {"available": True, "utilization_percent": utilization}
        return {"available": False, "reason": "could not parse rocm-smi output"}
    except Exception as e:
        logger.warning("Failed to query rocm-smi: %s", e)
        return {"available": False, "reason": f"rocm-smi query failed: {e}"}


def get_host_stats() -> dict:
    """Get host (node) CPU/RAM usage."""
    psutil = _psutil()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()

    try:
        load_avg = os.getloadavg()
    except (AttributeError, OSError):
        load_avg = None

    return {
        "cpu_percent": cpu_percent,
        "cpu_count": psutil.cpu_count(),
        "memory_used_bytes": mem.used,
        "memory_total_bytes": mem.total,
        "memory_percent": mem.percent,
        "load_average": load_avg,
    }
