from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from metastable.device import DeviceSettings
from metastable.pipelines import flux
from metastable.pipelines.flux import FluxPipelineGenerator
from metastable.schedulers import FLUX_SCHEDULERS
from diffusers import FlowMatchEulerDiscreteScheduler


SHARED_SCHEDULER = FlowMatchEulerDiscreteScheduler()


class FakeFluxPipelineClass:
    def __init__(self, transformer, **components):
        self.transformer = transformer
        self.components = components


@pytest.fixture
def patched_flux(monkeypatch):
    calls = {"transformer": [], "shared": []}
    monkeypatch.setattr(flux, "FluxTransformer2DModel", SimpleNamespace(
        from_single_file=lambda path, **kw: calls["transformer"].append((path, kw)) or f"transformer:{path}"))

    def _shared(components_dir, dtype):
        calls["shared"].append((components_dir, dtype))
        return {"vae": "vae", "scheduler": SHARED_SCHEDULER, "text_encoder": "te"}

    monkeypatch.setattr(flux, "load_flux_shared_components", _shared)
    monkeypatch.setattr(FluxPipelineGenerator, "PIPELINE_CLASS", type("Txt", (FakeFluxPipelineClass,), {}))
    monkeypatch.setattr(FluxPipelineGenerator, "INPAINT_PIPELINE_CLASS", type("Inp", (FakeFluxPipelineClass,), {}))
    return calls


def _device(backend="cuda", dtype=torch.float16, source="auto"):
    return DeviceSettings(backend=backend, torch_device="cpu" if backend == "cpu" else "cuda:0",
                          index=0, dtype=dtype, dtype_source=source, vendor="x")


def test_class_attributes():
    assert FluxPipelineGenerator.MODEL_TYPE == "flux"
    assert FluxPipelineGenerator.SCHEDULERS is FLUX_SCHEDULERS
    from diffusers import FluxInpaintPipeline, FluxPipeline
    assert FluxPipelineGenerator.PIPELINE_CLASS is FluxPipeline
    assert FluxPipelineGenerator.INPAINT_PIPELINE_CLASS is FluxInpaintPipeline


def test_resolutions_are_16px_aligned():
    for w, h in FluxPipelineGenerator.get_resolutions().values():
        assert w % 16 == 0 and h % 16 == 0


def test_constructor_stores_components_dir(tmp_path):
    gen = FluxPipelineGenerator(tmp_path / "t.safetensors", tmp_path / "components")
    assert gen.components_dir == str(tmp_path / "components")
    assert gen.model_checkpoint == str(tmp_path / "t.safetensors")


# dtype selection --------------------------------------------------------------------------------

def test_prefers_bf16_on_accelerator_with_auto_dtype():
    gen = FluxPipelineGenerator("t", "c", device=_device("cuda", torch.float16, "auto"))
    gen._init_device()
    assert gen.dtype == torch.bfloat16


def test_explicit_dtype_wins():
    gen = FluxPipelineGenerator("t", "c", device=_device("rocm", torch.float16, "config"))
    gen._init_device()
    assert gen.dtype == torch.float16


def test_cpu_keeps_float32():
    gen = FluxPipelineGenerator("t", "c", device=_device("cpu", torch.float32, "auto"))
    gen._init_device()
    assert gen.dtype == torch.float32


# loading ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("loader,attr,other", [
    ("load_pipeline", "pipeline", "inpaint_pipeline"),
    ("load_inpaint_pipeline", "inpaint_pipeline", "pipeline"),
])
def test_load_composes_transformer_and_shared_components(patched_flux, loader, attr, other):
    gen = FluxPipelineGenerator("/m/flux.safetensors", "/m/components", device=_device())
    getattr(gen, loader)()
    pipe = getattr(gen, attr)
    assert getattr(gen, other) is None
    assert pipe.transformer == "transformer:/m/flux.safetensors"
    assert pipe.components["vae"] == "vae" and pipe.components["text_encoder"] == "te"
    assert isinstance(pipe.components["scheduler"], FlowMatchEulerDiscreteScheduler)
    assert pipe.components["scheduler"] is not SHARED_SCHEDULER
    assert patched_flux["transformer"] == [("/m/flux.safetensors", {"torch_dtype": torch.bfloat16})]
    assert patched_flux["shared"] == [("/m/components", torch.bfloat16)]


def test_shared_components_loader_is_cached(monkeypatch, tmp_path):
    loads = []

    def fake_cls(name):
        return SimpleNamespace(from_pretrained=lambda d, subfolder, **kw: loads.append((name, subfolder, kw)) or name)

    for name in ["CLIPTextModel", "CLIPTokenizer", "T5EncoderModel", "T5TokenizerFast",
                 "AutoencoderKL", "FlowMatchEulerDiscreteScheduler"]:
        monkeypatch.setattr(flux, name, fake_cls(name))
    flux.load_flux_shared_components.cache_clear()
    try:
        first = flux.load_flux_shared_components(str(tmp_path), torch.bfloat16)
        second = flux.load_flux_shared_components(str(tmp_path), torch.bfloat16)
        assert first is second
        assert set(first) == {"text_encoder", "tokenizer", "text_encoder_2", "tokenizer_2", "vae", "scheduler"}
        assert len(loads) == 6
        subfolders = {name: (sub, kw) for name, sub, kw in loads}
        assert subfolders["T5EncoderModel"] == ("text_encoder_2", {"torch_dtype": torch.bfloat16})
        assert subfolders["CLIPTokenizer"] == ("tokenizer", {})
        flux.load_flux_shared_components(str(tmp_path), torch.float16)
        assert len(loads) == 12  # dtype is part of the cache key
    finally:
        flux.load_flux_shared_components.cache_clear()


# placement --------------------------------------------------------------------------------------

def test_to_device_uses_cpu_offload_on_accelerator(fake_pipeline_cls):
    gen = FluxPipelineGenerator("t", "c", device=_device())
    gen._init_device()
    gen.pipeline, gen.inpaint_pipeline = fake_pipeline_cls(), fake_pipeline_cls()
    gen.to_device()
    assert [p.offload_calls for p in gen.pipelines] == [1, 1]
    assert [p.to_calls for p in gen.pipelines] == [[], []]
    assert gen._placement == "offload"


def test_to_device_on_cpu_moves_pipeline(fake_pipeline_cls, cpu_device):
    gen = FluxPipelineGenerator("t", "c", device=cpu_device)
    gen._init_device()
    gen.pipeline = fake_pipeline_cls()
    gen.to_device()
    assert gen.pipeline.offload_calls == 0
    assert gen.pipeline.to_calls == ["cpu"]
    assert gen._placement == "device"


# inference --------------------------------------------------------------------------------------

@pytest.fixture
def flux_gen(fake_pipeline_cls, cpu_device):
    from diffusers import FlowMatchEulerDiscreteScheduler
    gen = FluxPipelineGenerator("t", "c", device=cpu_device)
    gen._init_device()
    gen.pipeline, gen.inpaint_pipeline = fake_pipeline_cls(), fake_pipeline_cls()
    for p in gen.pipelines:
        p.scheduler = FlowMatchEulerDiscreteScheduler()
    return gen


def test_forward_passes_true_cfg_scale(flux_gen):
    flux_gen.forward("p", "n", "FlowMatch Heun", 4, 64, 32, 3.5, 1, true_cfg_scale=2.0)
    call = flux_gen.pipeline.calls[0]
    assert call["true_cfg_scale"] == 2.0
    assert call["guidance_scale"] == 3.5
    assert type(flux_gen.pipeline.scheduler).__name__ == "FlowMatchHeunDiscreteScheduler"


def test_forward_default_true_cfg_scale(flux_gen):
    flux_gen.forward("p", "n", None, 4, 64, 32, 3.5, 1)
    assert flux_gen.pipeline.calls[0]["true_cfg_scale"] == 1.0


def test_forward_rejects_discrete_schedulers(flux_gen):
    from metastable.exceptions import PipelineError
    with pytest.raises(PipelineError, match="FlowMatch Euler"):
        flux_gen.forward("p", "n", "Euler", 4, 64, 32, 3.5, 1)


def test_inpaint_strength_and_true_cfg(flux_gen):
    img, mask = Image.new("RGB", (32, 32)), Image.new("L", (32, 32))
    _, meta = flux_gen.forward_inpaint("p", "n", img, mask, None, 4, 3.5, 1, strength=0.8, true_cfg_scale=1.5)
    call = flux_gen.inpaint_pipeline.calls[0]
    assert call["strength"] == 0.8
    assert call["true_cfg_scale"] == 1.5
    assert meta["inpainting"] is True


def test_inpaint_without_strength_uses_pipeline_default(flux_gen):
    img, mask = Image.new("RGB", (32, 32)), Image.new("L", (32, 32))
    flux_gen.forward_inpaint("p", "n", img, mask, None, 4, 3.5, 1)
    assert "strength" not in flux_gen.inpaint_pipeline.calls[0]


# LoRA isolation from shared components ----------------------------------------------------------

class FakeModule:
    def __init__(self, name):
        self.name = name
        self.to_calls = []

    def to(self, device):
        self.to_calls.append(device)
        return self


@pytest.fixture
def lora_flux(monkeypatch, fake_pipeline_cls, cpu_device):
    shared = {name: FakeModule(f"shared:{name}") for name in ("text_encoder", "text_encoder_2", "vae")}
    loads = []

    def _component(components_dir, name, dtype):
        loads.append((components_dir, name, dtype))
        return FakeModule(f"private:{name}")

    monkeypatch.setattr(flux, "load_flux_shared_components", lambda d, dtype: shared)
    monkeypatch.setattr(flux, "load_flux_component", _component)

    gen = FluxPipelineGenerator("t", "/m/components", device=cpu_device)
    gen._init_device()
    gen.pipeline, gen.inpaint_pipeline = fake_pipeline_cls(), fake_pipeline_cls()
    for p in gen.pipelines:
        p._lora_loadable_modules = ["transformer", "text_encoder"]
        p.transformer = FakeModule("transformer")
        for name, module in shared.items():
            setattr(p, name, module)
    return SimpleNamespace(gen=gen, shared=shared, loads=loads)


def test_lora_fusion_never_touches_shared_components(lora_flux):
    gen, shared = lora_flux.gen, lora_flux.shared
    gen.add_loras([{"path": "/l/style.safetensors", "strength": 0.7}])

    for p in gen.pipelines:
        assert p.text_encoder is not shared["text_encoder"]
        assert p.text_encoder.name == "private:text_encoder"
        assert p.text_encoder_2 is shared["text_encoder_2"]
        assert p.vae is shared["vae"]
        assert p.transformer.name == "transformer"
    assert gen.pipeline.text_encoder is not gen.inpaint_pipeline.text_encoder
    assert lora_flux.loads == [("/m/components", "text_encoder", torch.float32)] * 2


def test_isolation_happens_once_per_pipeline(lora_flux):
    gen = lora_flux.gen
    gen.add_loras([{"path": "/l/a.safetensors"}])
    private = gen.pipeline.text_encoder
    gen.add_loras([{"path": "/l/b.safetensors"}])
    assert gen.pipeline.text_encoder is private
    assert len(lora_flux.loads) == 2


def test_already_active_adapter_skips_isolation(lora_flux):
    gen = lora_flux.gen
    for p in gen.pipelines:
        p.active_adapters.append("name_a")
    assert gen.add_loras([{"path": "/l/a.safetensors"}]) == []
    assert lora_flux.loads == []
    assert gen.pipeline.text_encoder is lora_flux.shared["text_encoder"]


def test_isolated_component_follows_device_placement(lora_flux):
    gen = lora_flux.gen
    gen.to_device()  # cpu host: plain .to()
    gen.add_loras([{"path": "/l/a.safetensors"}])
    assert gen.pipeline.text_encoder.to_calls == ["cpu"]
    assert gen.pipeline.offload_calls == 0


def test_isolated_component_reinstalls_offload_hooks(lora_flux):
    gen = lora_flux.gen
    gen._placement = "offload"
    gen.add_loras([{"path": "/l/a.safetensors"}])
    assert [p.offload_calls for p in gen.pipelines] == [1, 1]
    assert gen.pipeline.text_encoder.to_calls == []


def test_load_flux_component_is_uncached(monkeypatch, tmp_path):
    loads = []
    monkeypatch.setattr(flux, "CLIPTextModel", SimpleNamespace(
        from_pretrained=lambda d, subfolder, **kw: loads.append((subfolder, kw)) or object()))
    first = flux.load_flux_component(str(tmp_path), "text_encoder", torch.bfloat16)
    second = flux.load_flux_component(str(tmp_path), "text_encoder", torch.bfloat16)
    assert first is not second
    assert loads == [("text_encoder", {"torch_dtype": torch.bfloat16})] * 2
