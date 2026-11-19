"""Locking between pipelines sharing modules."""

import threading

from diffusers import FlowMatchEulerDiscreteScheduler

from metastable.pipelines import flux
from metastable.pipelines.flux import FluxPipelineGenerator, components_lock
from metastable.pipelines.sd15 import SD15PipelineGenerator


def _flux_gen(fake_pipeline_cls, device, components_dir="/m/components"):
    gen = FluxPipelineGenerator("t", components_dir, device=device)
    gen._init_device()
    gen.pipeline, gen.inpaint_pipeline = fake_pipeline_cls(), fake_pipeline_cls()
    for p in gen.pipelines:
        p.scheduler = FlowMatchEulerDiscreteScheduler()
    return gen


# locks ------------------------------------------------------------------------------------------

def test_flux_generators_share_lock_per_components_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "c").mkdir()
    a = FluxPipelineGenerator("a", "c")
    b = FluxPipelineGenerator("b", tmp_path / "c")
    other = FluxPipelineGenerator("a", tmp_path / "other")
    assert a.lock is b.lock is components_lock(tmp_path / "c")
    assert other.lock is not a.lock


def test_single_file_generators_have_own_lock():
    assert SD15PipelineGenerator("a").lock is not SD15PipelineGenerator("a").lock


def test_lock_is_reentrant(fake_pipeline_cls, cpu_device, monkeypatch):
    monkeypatch.setattr(flux, "load_flux_shared_components", lambda d, dtype: {})
    gen = _flux_gen(fake_pipeline_cls, cpu_device)
    with gen.lock:
        gen.to_device()
        gen.add_loras([{"path": "/l/a.safetensors"}])
        gen.forward("p", "", None, 4, 64, 32, 3.5, 1)
    assert gen.pipeline.fused


def test_generators_sharing_components_never_run_concurrently(fake_pipeline_cls, cpu_device):
    gen_a = _flux_gen(fake_pipeline_cls, cpu_device)
    gen_b = _flux_gen(fake_pipeline_cls, cpu_device)
    b_thread = threading.Thread(target=gen_b.forward, args=("p", "", None, 4, 64, 32, 3.5, 2))
    seen_during_a = {}

    original_call = gen_a.pipeline.__call__

    def a_call(self, **kwargs):
        b_thread.start()
        b_thread.join(timeout=0.2)
        seen_during_a["b_alive"] = b_thread.is_alive()
        seen_during_a["b_calls"] = len(gen_b.pipeline.calls)
        return original_call(**kwargs)

    gen_a.pipeline.__class__ = type("Blocking", (fake_pipeline_cls,), {"__call__": a_call})
    gen_a.forward("p", "", None, 4, 64, 32, 3.5, 1)
    b_thread.join(timeout=5)

    assert seen_during_a == {"b_alive": True, "b_calls": 0}
    assert not b_thread.is_alive()
    assert len(gen_b.pipeline.calls) == 1


def test_generators_with_separate_components_run_concurrently(fake_pipeline_cls, cpu_device):
    gen_a = _flux_gen(fake_pipeline_cls, cpu_device, "/m/components-a")
    gen_b = _flux_gen(fake_pipeline_cls, cpu_device, "/m/components-b")
    b_thread = threading.Thread(target=gen_b.forward, args=("p", "", None, 4, 64, 32, 3.5, 2))
    seen = {}
    original_call = gen_a.pipeline.__call__

    def a_call(self, **kwargs):
        b_thread.start()
        b_thread.join(timeout=5)
        seen["b_calls"] = len(gen_b.pipeline.calls)
        return original_call(**kwargs)

    gen_a.pipeline.__class__ = type("Blocking", (fake_pipeline_cls,), {"__call__": a_call})
    gen_a.forward("p", "", None, 4, 64, 32, 3.5, 1)
    assert seen == {"b_calls": 1}
