import builtins
import json
import subprocess
from types import SimpleNamespace

import pytest

from metastable import telemetry


def test_cpu_memory_stats():
    assert telemetry.get_gpu_memory_stats("cpu") == {
        "available": False, "reason": "no GPU detected (running on CPU)"}


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_cuda_api_memory_stats(backend, monkeypatch):
    monkeypatch.setattr(telemetry.torch.cuda, "mem_get_info", lambda: (2, 10))
    monkeypatch.setattr(telemetry.torch.cuda, "memory_allocated", lambda: 3)
    monkeypatch.setattr(telemetry.torch.cuda, "memory_reserved", lambda: 4)
    assert telemetry.get_gpu_memory_stats(backend) == {
        "available": True, "used_bytes": 8, "total_bytes": 10, "allocated_bytes": 3,
        "reserved_bytes": 4, "total_is_approximate": False}


def test_mps_memory_stats(monkeypatch):
    monkeypatch.setattr(telemetry.torch.mps, "current_allocated_memory", lambda: 5)
    monkeypatch.setattr(telemetry.torch.mps, "driver_allocated_memory", lambda: 7)
    stats = telemetry.get_gpu_memory_stats("mps")
    assert stats["used_bytes"] == 7 and stats["allocated_bytes"] == 5
    assert stats["total_is_approximate"] is True
    assert stats["total_bytes"] > 0


def test_compute_stats_unavailable_on_cpu_and_mps():
    for backend in ("cpu", "mps"):
        stats = telemetry.get_gpu_compute_stats(backend)
        assert stats["available"] is False and backend in stats["reason"]


@pytest.mark.parametrize("backend,tool", [("cuda", "nvidia-smi"), ("rocm", "rocm-smi")])
def test_compute_stats_tool_missing(backend, tool, monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: None)
    assert telemetry.get_gpu_compute_stats(backend) == {"available": False, "reason": f"{tool} not found on PATH"}


def _fake_run(stdout):
    return lambda *a, **k: SimpleNamespace(stdout=stdout)


def test_nvidia_compute_stats(monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(telemetry.subprocess, "run", _fake_run("42\n17\n"))
    assert telemetry.get_gpu_compute_stats("cuda") == {"available": True, "utilization_percent": 42.0}


@pytest.mark.parametrize("payload", [
    {"card0": {"GPU use (%)": "12"}},
    {"0": {"Name": "x", "GPU use (%)": "12%"}},
    {"system": "info", "card1": {"GPU use (%)": 12}},
])
def test_amd_compute_stats(payload, monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(telemetry.subprocess, "run", _fake_run(json.dumps(payload)))
    assert telemetry.get_gpu_compute_stats("rocm") == {"available": True, "utilization_percent": 12.0}


def test_amd_unparseable(monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(telemetry.subprocess, "run", _fake_run(json.dumps({"card0": {"Temp": "40"}})))
    assert telemetry.get_gpu_compute_stats("rocm") == {"available": False, "reason": "could not parse rocm-smi output"}


@pytest.mark.parametrize("backend", ["cuda", "rocm"])
def test_tool_failure(backend, monkeypatch):
    monkeypatch.setattr(telemetry.shutil, "which", lambda name: "/usr/bin/" + name)

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("smi", 2)

    monkeypatch.setattr(telemetry.subprocess, "run", boom)
    stats = telemetry.get_gpu_compute_stats(backend)
    assert stats["available"] is False and "query failed" in stats["reason"]


def test_host_stats():
    stats = telemetry.get_host_stats()
    assert set(stats) == {"cpu_percent", "cpu_count", "memory_used_bytes", "memory_total_bytes",
                          "memory_percent", "load_average"}
    assert stats["memory_total_bytes"] > 0


def test_host_stats_without_loadavg(monkeypatch):
    def no_loadavg():
        raise OSError

    monkeypatch.setattr(telemetry.os, "getloadavg", no_loadavg)
    assert telemetry.get_host_stats()["load_average"] is None


def test_missing_psutil_has_helpful_error(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"metastable\[telemetry\]"):
        telemetry.get_host_stats()
