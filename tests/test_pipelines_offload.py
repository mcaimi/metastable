"""Offload hook ownership between pipelines sharing modules."""

from types import SimpleNamespace

import torch
from accelerate.hooks import ModelHook, add_hook_to_module
from diffusers import FlowMatchEulerDiscreteScheduler

from metastable.device import DeviceSettings
from metastable.pipelines import flux
from metastable.pipelines.flux import FluxPipelineGenerator


def _gpu(index=0):
    return DeviceSettings(backend="cuda", torch_device=f"cuda:{index}", index=index,
                          dtype=torch.float16, dtype_source="auto", vendor="nvidia")


def _flux_gen(fake_pipeline_cls, device, components_dir="/m/components"):
    gen = FluxPipelineGenerator("t", components_dir, device=device)
    gen._init_device()
    gen.pipeline, gen.inpaint_pipeline = fake_pipeline_cls(), fake_pipeline_cls()
    for p in gen.pipelines:
        p.scheduler = FlowMatchEulerDiscreteScheduler()
    return gen


# offload hook ownership -------------------------------------------------------------------------

def test_inference_reinstalls_offload_hooks_first(fake_pipeline_cls, cpu_device):
    gen = _flux_gen(fake_pipeline_cls, cpu_device)
    gen._placement = "offload"
    order = []
    gen.pipeline.enable_model_cpu_offload = lambda device=None: order.append(("offload", device))
    original_call = gen.pipeline.__call__
    gen.pipeline.__class__ = type("Recording", (fake_pipeline_cls,), {
        "__call__": lambda self, **kw: order.append(("call",)) or original_call(**kw)})

    gen.forward("p", "", None, 4, 64, 32, 3.5, 1)
    assert order == [("offload", "cpu"), ("call",)]
    assert gen.inpaint_pipeline.offload_calls == 0  # only the pipeline about to run


def test_inference_with_device_placement_installs_no_hooks(fake_pipeline_cls, cpu_device):
    gen = _flux_gen(fake_pipeline_cls, cpu_device)
    gen.to_device()
    gen.forward("p", "", None, 4, 64, 32, 3.5, 1)
    assert gen.pipeline.offload_calls == 0


def test_offload_targets_configured_gpu_index(fake_pipeline_cls):
    gen = _flux_gen(fake_pipeline_cls, _gpu(index=1))
    gen.to_device()
    assert [p.offload_devices for p in gen.pipelines] == [["cuda:1"], ["cuda:1"]]


def test_remove_offload_hooks_strips_modules_only():
    module = torch.nn.Linear(2, 2)
    add_hook_to_module(module, ModelHook())
    flux._remove_offload_hooks({"vae": module, "tokenizer": "not-a-module"})
    assert not hasattr(module, "_hf_hook")


def test_build_strips_stale_hooks_before_loading_transformer(monkeypatch, cpu_device):
    order = []
    shared = {"vae": "vae", "scheduler": FlowMatchEulerDiscreteScheduler()}
    monkeypatch.setattr(flux, "load_flux_shared_components", lambda d, dtype: shared)
    monkeypatch.setattr(flux, "_remove_offload_hooks", lambda comps: order.append("strip"))
    monkeypatch.setattr(flux, "FluxTransformer2DModel", SimpleNamespace(
        from_single_file=lambda path, **kw: order.append("transformer") or "transformer"))
    monkeypatch.setattr(FluxPipelineGenerator, "PIPELINE_CLASS",
                        staticmethod(lambda transformer, **components: SimpleNamespace(**components)))

    gen = FluxPipelineGenerator("t", "/m/components", device=cpu_device)
    gen.load_pipeline()
    assert order == ["strip", "transformer"]
    assert gen.pipeline.scheduler is not shared["scheduler"]

