#!/usr/bin/env python
"""
Compute Device Selection

Centralizes compute-device detection and override handling for every
supported backend:

- "cuda"  - NVIDIA GPUs (PyTorch CUDA builds)
- "rocm"  - AMD GPUs (PyTorch ROCm builds; PyTorch exposes HIP through the
            CUDA API, so torch-level device strings stay "cuda[:N]")
- "mps"   - Apple Silicon (Metal Performance Shaders)
- "cpu"   - CPU fallback

Configuration is a plain mapping (every key optional):

    {
        "backend": "auto",        # auto | cuda | rocm | mps | cpu
        "index": 0,               # GPU ordinal for multi-GPU hosts
        "dtype": "auto",          # auto | float16 | bfloat16 | float32
        "gfx_override": "10.3.0", # sets HSA_OVERRIDE_GFX_VERSION (see below)
        "env": {},                # extra env vars, applied before HIP/CUDA init
    }

Applications register it once at startup with configure_device(); every
pipeline generator and merge pipeline then picks it up through
get_device_settings(). Without configure_device() the library falls back to
pure auto-detection. Generators also accept an explicit DeviceSettings
instance for per-object overrides.

`gfx_override` exists for AMD cards that are absent from the official ROCm
support matrix (e.g. RX 6700 XT / Navi 14 is gfx1031): setting it to "10.3.0"
makes the HSA runtime present the card as gfx1030. The variable must be set
before the first HIP initialization, so configure_device() must run before
any other torch.cuda call in the process.

Environment overrides are applied exactly once per process and are idempotent;
resolved settings are cached until configure_device() or
reset_device_settings() is called.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
import torch.cuda as cuda
import torch.backends.mps as apple_mps

from metastable.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Supported dtype aliases for the "dtype" config key
_DTYPE_ALIASES = {
    "auto": None,  # sentinel: resolve from backend
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}

# Backends addressable through the CUDA API (NVIDIA CUDA builds and AMD ROCm
# builds alike). torch.device("cuda:N") is the correct torch-level string for
# both; "rocm" is only a backend/vendor label, never a torch device string.
_CUDA_API_BACKENDS = ("cuda", "rocm")

# Module-level guard: environment overrides must be applied before the first
# HIP/HSA initialization, and only once per process.
_ENV_APPLIED = False

# Process-wide configuration registered via configure_device(), and the
# settings resolved from it (lazily, on first get_device_settings()).
_DEVICE_CONFIG: Dict[str, Any] = {}
_DEVICE_SETTINGS: Optional["DeviceSettings"] = None


@dataclass(frozen=True)
class DeviceSettings:
    """
    Resolved compute device settings.

    Attributes:
        backend:      Backend label: "cuda", "rocm", "mps", or "cpu".
        torch_device: Torch-level device string ("cuda:N", "mps", "cpu") safe
                      to pass to torch APIs, Generator(), and safetensors.
        index:        GPU ordinal (multi-GPU hosts); 0 for mps/cpu.
        dtype:        Recommended torch dtype for weights/inference.
        dtype_source: "auto" (backend default) or "config" (explicitly set).
        vendor:       "nvidia", "amd", "apple", or "none".
        gpu_name:     Human-readable GPU name, or None on CPU hosts.
        hip_version:  torch.version.hip on ROCm builds, else None.
    """

    backend: str
    torch_device: str
    index: int
    dtype: torch.dtype
    dtype_source: str
    vendor: str
    gpu_name: Optional[str] = None
    hip_version: Optional[str] = None

    @classmethod
    def cpu(cls, dtype: torch.dtype = torch.float32) -> "DeviceSettings":
        """Convenience constructor for an explicit CPU device."""
        return cls(
            backend="cpu",
            torch_device="cpu",
            index=0,
            dtype=dtype,
            dtype_source="config",
            vendor="none",
        )


def _normalize_config(device_config: Any) -> Dict[str, Any]:
    """
    Normalize a device config to a plain dict.

    Accepts None, any Mapping, or an object exposing to_dict() (e.g. the
    attribute-style config containers used by applications).
    """
    if device_config is None:
        return {}
    if isinstance(device_config, Mapping):
        return dict(device_config)
    to_dict = getattr(device_config, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise ConfigurationError(
        f"device config: expected a mapping or None, got {type(device_config).__name__}"
    )


def apply_device_env(device_config: Any = None) -> None:
    """
    Apply device-related environment variables (idempotent, once per process).

    Must run before the first HIP/CUDA initialization - the HSA runtime reads
    HSA_OVERRIDE_GFX_VERSION when it enumerates GPUs, so setting it later has
    no effect. resolve_device_settings() calls this automatically.

    Args:
        device_config: device configuration mapping (see module docstring)
    """
    global _ENV_APPLIED
    if _ENV_APPLIED:
        return
    _ENV_APPLIED = True

    params = _normalize_config(device_config)

    gfx_override = params.get("gfx_override")
    if gfx_override:
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = str(gfx_override)
        logger.info(
            "gfx_override set: HSA_OVERRIDE_GFX_VERSION=%s",
            gfx_override,
        )

    env_map = params.get("env")
    if env_map:
        for key, value in _normalize_config(env_map).items():
            os.environ[str(key)] = str(value)
            logger.info("device env applied: %s=%s", key, value)


def torch_device_string(backend: str, index: int = 0) -> str:
    """
    Map a backend label to the torch-level device string.

    Args:
        backend: one of "cuda", "rocm", "mps", "cpu"
        index:   GPU ordinal (ignored for mps/cpu)

    Returns:
        Device string accepted by torch, Generator(), and safetensors
        (e.g. "cuda:0" for both NVIDIA CUDA and AMD ROCm backends).

    Raises:
        ConfigurationError: if backend is not a recognized label.
    """
    backend = str(backend).strip().lower()
    if backend in _CUDA_API_BACKENDS:
        return f"cuda:{index}"
    if backend == "mps":
        return "mps"
    if backend == "cpu":
        return "cpu"
    raise ConfigurationError(
        f"Unknown device backend '{backend}'. Expected one of: cuda, rocm, mps, cpu"
    )


def resolve_device_settings(device_config: Any = None) -> DeviceSettings:
    """
    Resolve the compute device from configuration and hardware detection.

    Uncached: use get_device_settings() for the process-wide cached value.

    Args:
        device_config: device configuration mapping (None or missing keys
                       fall back to defaults: backend=auto, index=0,
                       dtype=auto, no env overrides)

    Returns:
        DeviceSettings with the resolved backend, torch device string, dtype,
        vendor, and GPU identification.

    Raises:
        ConfigurationError: if an explicitly requested backend/dtype is not
                            available in the running PyTorch build.
    """
    apply_device_env(device_config)

    params = _normalize_config(device_config)
    backend_cfg = str(params.get("backend") or "auto").strip().lower()
    index = int(params.get("index") or 0)
    dtype_cfg = str(params.get("dtype") or "auto").strip().lower()

    hip_version = torch.version.hip
    vendor = "none"
    gpu_name: Optional[str] = None

    if backend_cfg == "auto":
        if apple_mps.is_available():
            logger.info("Apple Metal Performance Shaders Available!")
            backend, vendor, torch_device = "mps", "apple", "mps"
        elif cuda.is_available():
            gpu_name = cuda.get_device_name(index)
            device_capabilities = cuda.get_device_capability(index)
            device_available_mem, device_total_mem = [
                x / 1024**3 for x in cuda.mem_get_info(index)
            ]
            if hip_version:
                backend, vendor = "rocm", "amd"
                logger.info(
                    "AMD ROCm GPU available: %s - HIP %s - %s - %.1f/%.1f GB VRAM",
                    gpu_name,
                    hip_version,
                    device_capabilities,
                    device_available_mem,
                    device_total_mem,
                )
            else:
                backend, vendor = "cuda", "nvidia"
                logger.info(
                    "GPU available: %s - %s - %.1f/%.1f GB VRAM",
                    gpu_name,
                    device_capabilities,
                    device_available_mem,
                    device_total_mem,
                )
            torch_device = f"cuda:{index}"
        else:
            backend, torch_device = "cpu", "cpu"
            logger.warning("No GPU found. Using CPU (this will be slow).")
    elif backend_cfg in _CUDA_API_BACKENDS:
        if backend_cfg == "rocm" and not hip_version:
            raise ConfigurationError(
                "device backend is 'rocm' but the installed PyTorch build has no "
                "HIP/ROCm support (torch.version.hip is None). Install a ROCm "
                "build of torch from the PyTorch ROCm wheel index."
            )
        if not cuda.is_available():
            raise ConfigurationError(
                f"device backend is '{backend_cfg}' but torch reports no accessible "
                "GPU. If this is an AMD card: check that the amdgpu driver is "
                "loaded, that your user is in the 'video' and 'render' groups, "
                "and - for cards missing from the ROCm support matrix such as "
                "the RX 5500 XT (gfx1034) - that gfx_override is set "
                "(e.g. gfx_override: '10.3.0')."
            )
        backend = backend_cfg
        vendor = "amd" if hip_version else "nvidia"
        gpu_name = cuda.get_device_name(index)
        torch_device = f"cuda:{index}"
        logger.info(
            "Device backend forced via config: %s (%s%s)",
            backend,
            gpu_name,
            f" - HIP {hip_version}" if hip_version else "",
        )
    elif backend_cfg == "mps":
        if not apple_mps.is_available():
            raise ConfigurationError(
                "device backend is 'mps' but Metal Performance Shaders are not "
                "available on this host."
            )
        backend, vendor, torch_device = "mps", "apple", "mps"
        logger.info("Apple Metal Performance Shaders selected via config.")
    elif backend_cfg == "cpu":
        backend, vendor, torch_device = "cpu", "none", "cpu"
        logger.info("Device backend forced via config: cpu")
    else:
        raise ConfigurationError(
            f"device backend '{backend_cfg}' is not one of: auto, cuda, rocm, mps, cpu"
        )

    dtype, dtype_source = _resolve_dtype(dtype_cfg, backend)
    return DeviceSettings(
        backend=backend,
        torch_device=torch_device,
        index=index,
        dtype=dtype,
        dtype_source=dtype_source,
        vendor=vendor,
        gpu_name=gpu_name,
        hip_version=hip_version,
    )


def _resolve_dtype(dtype_cfg: str, backend: str) -> tuple:
    """
    Resolve the inference dtype from the config value and backend.

    Returns:
        Tuple of (torch.dtype, source) where source is "config" for an
        explicit dtype or "auto" for the backend default.
    """
    if dtype_cfg not in _DTYPE_ALIASES:
        raise ConfigurationError(
            f"device dtype '{dtype_cfg}' is not one of: auto, float16, bfloat16, float32"
        )

    if dtype_cfg != "auto":
        return _DTYPE_ALIASES[dtype_cfg], "config"

    # Backend defaults: fp32 on CPU, fp16 on every accelerator. ROCm keeps
    # fp16 by default because bf16 support is unreliable on older RDNA GPUs
    # (e.g. RX 5500 XT / gfx1034); opt in explicitly via dtype on RDNA2+
    # cards where bf16 kernels are usable.
    if backend == "cpu":
        return torch.float32, "auto"
    if backend == "rocm":
        logger.info(
            "ROCm: defaulting to float16 (bfloat16 is not reliably supported on "
            "older RDNA GPUs); set dtype: bfloat16 to override."
        )
    return torch.float16, "auto"


def configure_device(device_config: Any = None) -> DeviceSettings:
    """
    Register the process-wide device configuration and resolve it.

    Call once at application startup, before any other torch.cuda usage,
    so environment overrides (gfx_override, env) take effect.

    Args:
        device_config: device configuration mapping (see module docstring)

    Returns:
        The resolved DeviceSettings (also cached for get_device_settings()).
    """
    global _DEVICE_CONFIG, _DEVICE_SETTINGS
    _DEVICE_CONFIG = _normalize_config(device_config)
    _DEVICE_SETTINGS = resolve_device_settings(_DEVICE_CONFIG)
    return _DEVICE_SETTINGS


def get_device_settings() -> DeviceSettings:
    """
    Return the process-wide device settings (resolved lazily and cached).

    Uses the configuration registered via configure_device(), or pure
    auto-detection when none was registered.
    """
    global _DEVICE_SETTINGS
    if _DEVICE_SETTINGS is None:
        _DEVICE_SETTINGS = resolve_device_settings(_DEVICE_CONFIG)
    return _DEVICE_SETTINGS


def reset_device_settings() -> None:
    """
    Forget the registered configuration and cached settings.

    Environment overrides already applied to the process are not undone.
    """
    global _DEVICE_CONFIG, _DEVICE_SETTINGS
    _DEVICE_CONFIG = {}
    _DEVICE_SETTINGS = None


def get_torch_device() -> str:
    """
    Convenience accessor for the torch-level device string of the resolved
    device (e.g. "cuda:0" on NVIDIA and AMD ROCm alike, "mps", "cpu").
    """
    return get_device_settings().torch_device
