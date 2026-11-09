import os

import pytest
import torch

from metastable import device as dm
from metastable.device import DeviceSettings
from metastable.exceptions import ConfigurationError


@pytest.fixture
def no_gpu(monkeypatch):
    monkeypatch.setattr(dm.apple_mps, "is_available", lambda: False)
    monkeypatch.setattr(dm.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dm.torch.version, "hip", None)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Pretend a CUDA-API GPU is present (NVIDIA unless hip is patched)."""
    monkeypatch.setattr(dm.apple_mps, "is_available", lambda: False)
    monkeypatch.setattr(dm.cuda, "is_available", lambda: True)
    monkeypatch.setattr(dm.cuda, "get_device_name", lambda i: f"GPU{i}")
    monkeypatch.setattr(dm.cuda, "get_device_capability", lambda i: (8, 6))
    monkeypatch.setattr(dm.cuda, "mem_get_info", lambda i: (4 * 1024**3, 8 * 1024**3))
    monkeypatch.setattr(dm.torch.version, "hip", None)


@pytest.fixture
def fake_mps(monkeypatch):
    monkeypatch.setattr(dm.apple_mps, "is_available", lambda: True)
    monkeypatch.setattr(dm.torch.version, "hip", None)


class AttrConfig:
    """Mimics an attribute-style config container exposing to_dict()."""

    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return dict(self._data)


# torch_device_string ---------------------------------------------------------

@pytest.mark.parametrize("backend,index,expected", [
    ("cuda", 0, "cuda:0"),
    ("rocm", 1, "cuda:1"),
    (" CUDA ", 2, "cuda:2"),
    ("mps", 3, "mps"),
    ("cpu", 3, "cpu"),
])
def test_torch_device_string(backend, index, expected):
    assert dm.torch_device_string(backend, index) == expected


def test_torch_device_string_unknown():
    with pytest.raises(ConfigurationError):
        dm.torch_device_string("tpu")


# dtype resolution ---------------------------------------------------------------

@pytest.mark.parametrize("alias,expected", [
    ("float16", torch.float16), ("fp16", torch.float16), ("half", torch.float16),
    ("bfloat16", torch.bfloat16), ("bf16", torch.bfloat16),
    ("float32", torch.float32), ("fp32", torch.float32), ("float", torch.float32),
])
def test_explicit_dtype(alias, expected):
    assert dm._resolve_dtype(alias, "cuda") == (expected, "config")


@pytest.mark.parametrize("backend,expected", [
    ("cpu", torch.float32), ("cuda", torch.float16), ("rocm", torch.float16), ("mps", torch.float16),
])
def test_auto_dtype(backend, expected):
    assert dm._resolve_dtype("auto", backend) == (expected, "auto")


def test_unknown_dtype():
    with pytest.raises(ConfigurationError, match="dtype"):
        dm._resolve_dtype("int8", "cpu")


# config normalization -------------------------------------------------------------

def test_normalize_config_variants():
    assert dm._normalize_config(None) == {}
    assert dm._normalize_config({"backend": "cpu"}) == {"backend": "cpu"}
    assert dm._normalize_config(AttrConfig({"index": 1})) == {"index": 1}


@pytest.mark.parametrize("bad", ["cpu", 3, ["backend"]])
def test_normalize_config_rejects_non_mappings(bad):
    with pytest.raises(ConfigurationError):
        dm._normalize_config(bad)


# resolve_device_settings ------------------------------------------------------------

def test_auto_falls_back_to_cpu(no_gpu):
    s = dm.resolve_device_settings()
    assert (s.backend, s.torch_device, s.vendor, s.dtype, s.dtype_source) == (
        "cpu", "cpu", "none", torch.float32, "auto")


def test_auto_prefers_mps(fake_mps, monkeypatch):
    monkeypatch.setattr(dm.cuda, "is_available", lambda: True)
    s = dm.resolve_device_settings({})
    assert (s.backend, s.torch_device, s.vendor, s.dtype) == ("mps", "mps", "apple", torch.float16)


def test_auto_detects_nvidia(fake_cuda):
    s = dm.resolve_device_settings({"index": 1})
    assert (s.backend, s.torch_device, s.vendor, s.gpu_name, s.index) == (
        "cuda", "cuda:1", "nvidia", "GPU1", 1)


def test_auto_detects_rocm(fake_cuda, monkeypatch):
    monkeypatch.setattr(dm.torch.version, "hip", "7.2")
    s = dm.resolve_device_settings(None)
    assert (s.backend, s.vendor, s.torch_device, s.hip_version) == ("rocm", "amd", "cuda:0", "7.2")


def test_forced_cpu_ignores_gpu(fake_cuda):
    s = dm.resolve_device_settings({"backend": "cpu", "dtype": "bf16"})
    assert (s.backend, s.dtype, s.dtype_source) == ("cpu", torch.bfloat16, "config")


def test_forced_cuda(fake_cuda):
    s = dm.resolve_device_settings({"backend": "CUDA"})
    assert (s.backend, s.vendor, s.torch_device) == ("cuda", "nvidia", "cuda:0")


def test_forced_cuda_without_gpu(no_gpu):
    with pytest.raises(ConfigurationError, match="no accessible GPU"):
        dm.resolve_device_settings({"backend": "cuda"})


def test_forced_rocm_without_hip(fake_cuda):
    with pytest.raises(ConfigurationError, match="HIP/ROCm"):
        dm.resolve_device_settings({"backend": "rocm"})


def test_forced_rocm(fake_cuda, monkeypatch):
    monkeypatch.setattr(dm.torch.version, "hip", "7.2")
    s = dm.resolve_device_settings({"backend": "rocm", "index": 1})
    assert (s.backend, s.vendor, s.torch_device) == ("rocm", "amd", "cuda:1")


def test_forced_mps(fake_mps):
    assert dm.resolve_device_settings({"backend": "mps"}).backend == "mps"


def test_forced_mps_unavailable(no_gpu):
    with pytest.raises(ConfigurationError, match="Metal"):
        dm.resolve_device_settings({"backend": "mps"})


def test_unknown_backend(no_gpu):
    with pytest.raises(ConfigurationError, match="not one of"):
        dm.resolve_device_settings({"backend": "tpu"})


def test_none_values_fall_back_to_defaults(no_gpu):
    s = dm.resolve_device_settings({"backend": None, "index": None, "dtype": None})
    assert (s.backend, s.index, s.dtype_source) == ("cpu", 0, "auto")


def test_accepts_attribute_style_config(no_gpu):
    assert dm.resolve_device_settings(AttrConfig({"backend": "cpu"})).backend == "cpu"


# environment overrides ----------------------------------------------------------------

def test_apply_env(monkeypatch):
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    monkeypatch.delenv("METASTABLE_TEST_VAR", raising=False)
    dm.apply_device_env({"gfx_override": "10.3.0", "env": {"METASTABLE_TEST_VAR": 5}})
    assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "10.3.0"
    assert os.environ["METASTABLE_TEST_VAR"] == "5"


def test_apply_env_accepts_attribute_style_env(monkeypatch):
    monkeypatch.delenv("METASTABLE_TEST_VAR", raising=False)
    dm.apply_device_env({"env": AttrConfig({"METASTABLE_TEST_VAR": "x"})})
    assert os.environ["METASTABLE_TEST_VAR"] == "x"


def test_apply_env_runs_once(monkeypatch):
    monkeypatch.delenv("METASTABLE_TEST_VAR", raising=False)
    dm.apply_device_env({"env": {"METASTABLE_TEST_VAR": "first"}})
    dm.apply_device_env({"env": {"METASTABLE_TEST_VAR": "second"}})
    assert os.environ["METASTABLE_TEST_VAR"] == "first"


def test_apply_env_noop_without_config(monkeypatch):
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    dm.apply_device_env(None)
    assert "HSA_OVERRIDE_GFX_VERSION" not in os.environ


def test_resolve_applies_env(no_gpu, monkeypatch):
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    dm.resolve_device_settings({"backend": "cpu", "gfx_override": "10.3.0"})
    assert os.environ["HSA_OVERRIDE_GFX_VERSION"] == "10.3.0"


# process-wide settings ------------------------------------------------------------------

def test_get_device_settings_auto_detects_and_caches(no_gpu, monkeypatch):
    calls = []
    real = dm.resolve_device_settings
    monkeypatch.setattr(dm, "resolve_device_settings", lambda c=None: calls.append(c) or real(c))
    first = dm.get_device_settings()
    second = dm.get_device_settings()
    assert first is second
    assert first.backend == "cpu"
    assert calls == [{}]


def test_configure_device_overrides_cache(fake_cuda):
    assert dm.get_device_settings().backend == "cuda"
    configured = dm.configure_device({"backend": "cpu"})
    assert configured.backend == "cpu"
    assert dm.get_device_settings() is configured
    assert dm.get_torch_device() == "cpu"


def test_configure_device_invalid_config_raises(no_gpu):
    with pytest.raises(ConfigurationError):
        dm.configure_device({"backend": "tpu"})


def test_reset_device_settings(fake_cuda):
    dm.configure_device({"backend": "cpu"})
    dm.reset_device_settings()
    assert dm.get_device_settings().backend == "cuda"


def test_device_settings_cpu_constructor():
    s = DeviceSettings.cpu()
    assert (s.backend, s.torch_device, s.dtype, s.dtype_source) == ("cpu", "cpu", torch.float32, "config")
    assert DeviceSettings.cpu(torch.bfloat16).dtype == torch.bfloat16


def test_device_settings_is_frozen():
    with pytest.raises(Exception):
        DeviceSettings.cpu().backend = "cuda"
