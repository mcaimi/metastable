from types import SimpleNamespace

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from metastable.device import DeviceSettings
from metastable.exceptions import PipelineError
from metastable.pipelines import zimage
from metastable.pipelines.zimage import ZImagePipelineGenerator
from metastable.schedulers import FLOW_MATCH_SCHEDULERS

SHARED_SCHEDULER = FlowMatchEulerDiscreteScheduler(shift=3.0)


class FakeZImagePipelineClass:
    def __init__(self, transformer, **components):
        self.transformer = transformer
        self.components = components


@pytest.fixture
def patched_zimage(monkeypatch):
    calls = {"transformer": [], "shared": []}
    monkeypatch.setattr(zimage, "ZImageTransformer2DModel", SimpleNamespace(
        from_single_file=lambda path, **kw: calls["transformer"].append((path, kw)) or f"transformer:{path}"))

    def _shared(components_dir, dtype):
        calls["shared"].append((components_dir, dtype))
        return {"vae": "vae", "scheduler": SHARED_SCHEDULER, "text_encoder": "te", "tokenizer": "tok"}

    monkeypatch.setattr(zimage, "load_zimage_shared_components", _shared)
    monkeypatch.setattr(ZImagePipelineGenerator, "PIPELINE_CLASS", type("Txt", (FakeZImagePipelineClass,), {}))
    monkeypatch.setattr(ZImagePipelineGenerator, "INPAINT_PIPELINE_CLASS", type("Inp", (FakeZImagePipelineClass,), {}))
    return calls


def _device(backend="cuda", dtype=torch.float16, source="auto"):
    return DeviceSettings(backend=backend, torch_device="cpu" if backend == "cpu" else "cuda:0",
                          index=0, dtype=dtype, dtype_source=source, vendor="x")


def test_class_attributes():
    from diffusers import ZImageInpaintPipeline, ZImagePipeline
    assert ZImagePipelineGenerator.MODEL_TYPE == "zimage"
    assert ZImagePipelineGenerator.SCHEDULERS is FLOW_MATCH_SCHEDULERS
    assert ZImagePipelineGenerator.PIPELINE_CLASS is ZImagePipeline
    assert ZImagePipelineGenerator.INPAINT_PIPELINE_CLASS is ZImageInpaintPipeline


def test_resolutions_are_16px_aligned():
    for w, h in ZImagePipelineGenerator.get_resolutions().values():
        assert w % 16 == 0 and h % 16 == 0


def test_constructor_stores_components_dir(tmp_path):
    gen = ZImagePipelineGenerator(tmp_path / "z.safetensors", tmp_path / "components")
    assert gen.components_dir == str(tmp_path / "components")
    assert gen.model_checkpoint == str(tmp_path / "z.safetensors")


# dtype selection --------------------------------------------------------------------------------

def test_prefers_bf16_on_accelerator_with_auto_dtype():
    gen = ZImagePipelineGenerator("z", "c", device=_device("cuda", torch.float16, "auto"))
    gen._init_device()
    assert gen.dtype == torch.bfloat16


def test_explicit_dtype_wins():
    gen = ZImagePipelineGenerator("z", "c", device=_device("rocm", torch.float16, "config"))
    gen._init_device()
    assert gen.dtype == torch.float16


def test_cpu_keeps_float32():
    gen = ZImagePipelineGenerator("z", "c", device=_device("cpu", torch.float32, "auto"))
    gen._init_device()
    assert gen.dtype == torch.float32


# loading ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("loader,attr,other", [
    ("load_pipeline", "pipeline", "inpaint_pipeline"),
    ("load_inpaint_pipeline", "inpaint_pipeline", "pipeline"),
])
def test_load_composes_transformer_and_shared_components(patched_zimage, loader, attr, other):
    gen = ZImagePipelineGenerator("/m/z.safetensors", "/m/components", device=_device())
    getattr(gen, loader)()
    pipe = getattr(gen, attr)
    assert getattr(gen, other) is None
    assert pipe.transformer == "transformer:/m/z.safetensors"
    assert {k: pipe.components[k] for k in ("vae", "text_encoder", "tokenizer")} == {
        "vae": "vae", "text_encoder": "te", "tokenizer": "tok"}
    scheduler = pipe.components["scheduler"]
    assert isinstance(scheduler, FlowMatchEulerDiscreteScheduler)
    assert scheduler is not SHARED_SCHEDULER
    assert scheduler.config.shift == 3.0  # private copy keeps the shared config
    assert patched_zimage["transformer"] == [("/m/z.safetensors", {"torch_dtype": torch.bfloat16})]
    assert patched_zimage["shared"] == [("/m/components", torch.bfloat16)]


def test_shared_components_loader_is_cached(monkeypatch, tmp_path):
    loads = []

    def fake_cls(name):
        return SimpleNamespace(from_pretrained=lambda d, subfolder, **kw: loads.append((name, subfolder, kw)) or name)

    for name in ["Qwen3Model", "AutoTokenizer", "AutoencoderKL", "FlowMatchEulerDiscreteScheduler"]:
        monkeypatch.setattr(zimage, name, fake_cls(name))
    zimage.load_zimage_shared_components.cache_clear()
    try:
        first = zimage.load_zimage_shared_components(str(tmp_path), torch.bfloat16)
        second = zimage.load_zimage_shared_components(str(tmp_path), torch.bfloat16)
        assert first is second
        assert set(first) == {"text_encoder", "tokenizer", "vae", "scheduler"}
        assert len(loads) == 4
        subfolders = {name: (sub, kw) for name, sub, kw in loads}
        assert subfolders["Qwen3Model"] == ("text_encoder", {"torch_dtype": torch.bfloat16})
        assert subfolders["AutoTokenizer"] == ("tokenizer", {})
        assert subfolders["FlowMatchEulerDiscreteScheduler"] == ("scheduler", {})
        zimage.load_zimage_shared_components(str(tmp_path), torch.float16)
        assert len(loads) == 8  # dtype is part of the cache key
    finally:
        zimage.load_zimage_shared_components.cache_clear()


def test_load_zimage_component_is_uncached(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(zimage, "Qwen3Model", SimpleNamespace(
        from_pretrained=lambda d, subfolder, **kw: loads.append((subfolder, kw)) or object()))
    first = zimage.load_zimage_component(str(tmp_path), "text_encoder", torch.bfloat16)
    second = zimage.load_zimage_component(str(tmp_path), "text_encoder", torch.bfloat16)
    assert first is not second
    assert loads == [("text_encoder", {"torch_dtype": torch.bfloat16})] * 2


# placement --------------------------------------------------------------------------------------

def test_to_device_uses_cpu_offload_on_accelerator(fake_pipeline_cls):
    gen = ZImagePipelineGenerator("z", "c", device=_device())
    gen._init_device()
    gen.pipeline, gen.inpaint_pipeline = fake_pipeline_cls(), fake_pipeline_cls()
    gen.to_device()
    assert [p.offload_devices for p in gen.pipelines] == [["cuda:0"], ["cuda:0"]]
    assert [p.to_calls for p in gen.pipelines] == [[], []]
    assert gen._placement == "offload"


def test_to_device_on_cpu_moves_pipeline(fake_pipeline_cls, cpu_device):
    gen = ZImagePipelineGenerator("z", "c", device=cpu_device)
    gen._init_device()
    gen.pipeline = fake_pipeline_cls()
    gen.to_device()
    assert gen.pipeline.offload_calls == 0
    assert gen.pipeline.to_calls == ["cpu"]


# inference --------------------------------------------------------------------------------------

@pytest.fixture
def zimage_gen(fake_pipeline_cls, cpu_device):
    gen = ZImagePipelineGenerator("z", "c", device=cpu_device)
    gen._init_device()
    gen.pipeline, gen.inpaint_pipeline = fake_pipeline_cls(), fake_pipeline_cls()
    for p in gen.pipelines:
        p.scheduler = FlowMatchEulerDiscreteScheduler(shift=6.0)
    return gen


def test_forward_turbo_settings(zimage_gen):
    _, meta = zimage_gen.forward("p", "", None, 8, 64, 32, 0.0, 1)
    call = zimage_gen.pipeline.calls[0]
    assert call["guidance_scale"] == 0.0
    assert call["num_inference_steps"] == 8
    assert "true_cfg_scale" not in call
    assert meta["instances"][0]["guidance_scale"] == 0.0


def test_forward_base_settings_pass_negative_prompt(zimage_gen):
    zimage_gen.forward("p", "blurry", "FlowMatch Euler", 30, 64, 32, 4.0, 1)
    call = zimage_gen.pipeline.calls[0]
    assert call["guidance_scale"] == 4.0
    assert call["negative_prompt"] == "blurry"
    # Swapping the scheduler keeps the components directory's shift
    assert zimage_gen.pipeline.scheduler.config.shift == 6.0


def test_forward_rejects_heun(zimage_gen):
    with pytest.raises(PipelineError, match="FlowMatch Euler"):
        zimage_gen.forward("p", "", "FlowMatch Heun", 8, 64, 32, 0.0, 1)


def test_inpaint_strength(zimage_gen):
    img, mask = Image.new("RGB", (32, 32)), Image.new("L", (32, 32))
    _, meta = zimage_gen.forward_inpaint("p", "", img, mask, None, 8, 0.0, 1, strength=0.8)
    assert zimage_gen.inpaint_pipeline.calls[0]["strength"] == 0.8
    assert meta["inpainting"] is True


def test_inpaint_without_strength_uses_pipeline_default(zimage_gen):
    img, mask = Image.new("RGB", (32, 32)), Image.new("L", (32, 32))
    zimage_gen.forward_inpaint("p", "", img, mask, None, 8, 0.0, 1)
    assert "strength" not in zimage_gen.inpaint_pipeline.calls[0]


# LoRA -------------------------------------------------------------------------------------------

def test_lora_isolation_reloads_nothing(monkeypatch, zimage_gen):
    shared = {"text_encoder": object(), "vae": object(), "scheduler": SHARED_SCHEDULER}
    loads = []
    monkeypatch.setattr(zimage, "load_zimage_shared_components", lambda d, dtype: shared)
    monkeypatch.setattr(zimage, "load_zimage_component", lambda *a: loads.append(a))
    for p in zimage_gen.pipelines:
        p._lora_loadable_modules = ["transformer"]  # ZImageLoraLoaderMixin
        p.transformer = object()
        p.text_encoder = shared["text_encoder"]

    zimage_gen.add_loras([{"path": "/l/style.safetensors"}])
    assert loads == []
    assert all(p.text_encoder is shared["text_encoder"] for p in zimage_gen.pipelines)
    assert all(p.fused for p in zimage_gen.pipelines)
