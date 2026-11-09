"""Shared fixtures: no test downloads models or needs an accelerator."""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
import torch
from diffusers import EulerDiscreteScheduler
from PIL import Image
from safetensors.torch import save_file

from metastable import device as device_module
from metastable.device import DeviceSettings


@pytest.fixture(autouse=True)
def reset_device_state(monkeypatch):
    """Isolate the process-wide device configuration between tests."""
    device_module.reset_device_settings()
    monkeypatch.setattr(device_module, "_ENV_APPLIED", False)
    yield
    device_module.reset_device_settings()


@pytest.fixture
def cpu_device() -> DeviceSettings:
    return DeviceSettings.cpu()


@pytest.fixture
def gpu_device() -> DeviceSettings:
    """A fake CUDA device; only used where nothing touches real CUDA."""
    return DeviceSettings(
        backend="cuda",
        torch_device="cuda:0",
        index=0,
        dtype=torch.float16,
        dtype_source="auto",
        vendor="nvidia",
        gpu_name="Fake GPU",
    )


class FakePipeline:
    """
    Stand-in for a diffusers pipeline: records calls and returns a solid
    image of the requested (or input-image) size.
    """

    def __init__(self, width: int = 64, height: int = 32):
        self.scheduler = EulerDiscreteScheduler()
        self.vae = "original-vae"
        self.default_size = (width, height)
        self.calls: List[Dict[str, Any]] = []
        self.to_calls: List[str] = []
        self.offload_calls = 0
        self.active_adapters: List[str] = []
        self.loaded_loras: List[Any] = []
        self.fused: List[Dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if "image" in kwargs:
            size = kwargs["image"].size
        else:
            size = (kwargs.get("width", self.default_size[0]),
                    kwargs.get("height", self.default_size[1]))
        return SimpleNamespace(images=[Image.new("RGB", size, (10, 20, 30))])

    def to(self, device):
        self.to_calls.append(device)
        return self

    def enable_model_cpu_offload(self):
        self.offload_calls += 1

    def get_active_adapters(self):
        return list(self.active_adapters)

    def load_lora_weights(self, path, adapter_name):
        self.loaded_loras.append((path, adapter_name))

    def fuse_lora(self, lora_scale, adapter_name):
        self.fused.append({"lora_scale": lora_scale, "adapter_name": adapter_name})
        self.active_adapters.append(adapter_name)


@pytest.fixture
def fake_pipeline_cls():
    return FakePipeline


def make_state_dict(seed: int, dtype=torch.float32) -> Dict[str, torch.Tensor]:
    """A tiny UNet-shaped state dict."""
    g = torch.Generator().manual_seed(seed)
    return {
        "model.conv.weight": torch.randn(4, 3, 3, 3, generator=g).to(dtype),
        "model.conv.bias": torch.randn(4, generator=g).to(dtype),
        "model.linear.weight": torch.randn(8, 4, generator=g).to(dtype),
    }


@pytest.fixture
def state_dict_factory():
    return make_state_dict


@pytest.fixture
def checkpoint_factory(tmp_path):
    """Write a tiny safetensors checkpoint and return its path."""

    def _make(name: str, seed: int = 0, weights=None, metadata=None):
        path = tmp_path / f"{name}.safetensors"
        save_file(weights if weights is not None else make_state_dict(seed),
                  str(path), metadata=metadata)
        return path

    return _make
