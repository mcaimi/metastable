from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler
from PIL import Image

from metastable import device as dm
from metastable.exceptions import PipelineError
from metastable.pipelines import base
from metastable.pipelines.base import (
    BasePipelineGenerator,
    LoraSpec,
    format_metadata,
    normalize_path,
)


class DummyGenerator(BasePipelineGenerator):
    MODEL_TYPE = "dummy"

    def __init__(self, *args, pipeline_cls=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pipeline_cls = pipeline_cls

    @staticmethod
    def get_resolutions():
        return {"64x32": (64, 32)}

    def load_pipeline(self):
        self._init_device()
        self.pipeline = self._pipeline_cls()

    def load_inpaint_pipeline(self):
        self._init_device()
        self.inpaint_pipeline = self._pipeline_cls()


@pytest.fixture
def gen(fake_pipeline_cls, cpu_device):
    return DummyGenerator("model.safetensors", device=cpu_device, pipeline_cls=fake_pipeline_cls)


@pytest.fixture
def loaded(gen):
    gen.load_pipeline()
    return gen


@pytest.fixture
def loaded_inpaint(gen):
    gen.load_inpaint_pipeline()
    return gen


# helpers ------------------------------------------------------------------------------------

def test_normalize_path():
    assert normalize_path("a/b") == Path("a/b")
    p = Path("x")
    assert normalize_path(p) is p
    with pytest.raises(TypeError):
        normalize_path(3)


def test_format_metadata():
    meta = format_metadata("cat", "dog", steps=20, width=64, height=32, cfg=5.5, seed=42, scheduler="Euler")
    assert meta == {"instances": [{
        "prompt": "cat", "negative_prompt": "dog", "num_inference_steps": 20, "width": 64,
        "height": 32, "guidance_scale": 5.5, "seed": 42, "scheduler": "Euler"}]}


def test_format_metadata_random_seed(monkeypatch):
    monkeypatch.setattr("metastable.seed.get_random_seed", lambda bits: 99)
    assert format_metadata("p", seed=-1)["instances"][0]["seed"] == 99


def test_load_custom_vae_function(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(base, "AutoencoderKL", SimpleNamespace(
        from_single_file=lambda path, **kw: calls.append((path, kw)) or "vae"))
    assert base.load_custom_vae(tmp_path / "vae.safetensors", dtype=torch.float16) == "vae"
    path, kwargs = calls[0]
    assert path == str((tmp_path / "vae.safetensors").absolute())
    assert kwargs == {"subfolder": "vae", "use_safetensors": True, "torch_dtype": torch.float16}


def test_load_custom_unet_function(monkeypatch):
    calls = []
    monkeypatch.setattr(base, "UNet2DConditionModel", SimpleNamespace(
        from_single_file=lambda path, **kw: calls.append((path, kw)) or "unet"))
    assert base.load_custom_unet("unet.safetensors") == "unet"
    assert calls[0][1]["subfolder"] == "unet"
    assert calls[0][1]["torch_dtype"] is None


# LoraSpec -------------------------------------------------------------------------------------

def test_lora_spec_adapter_name():
    assert LoraSpec("/loras/style.safetensors").adapter_name == "name_style"
    assert LoraSpec("/loras/style.safetensors", name="custom").adapter_name == "custom"
    assert LoraSpec("x.safetensors").strength == 0.5


@pytest.mark.parametrize("entry,expected", [
    ({"path": "a.safetensors", "strength": 0.8}, LoraSpec("a.safetensors", 0.8)),
    ({"lora_path": Path("b.safetensors"), "merge_strength": "0.3"}, LoraSpec(Path("b.safetensors"), 0.3)),
    ({"path": "c.safetensors"}, LoraSpec("c.safetensors", 0.5)),
    ({"path": "d.safetensors", "adapter_name": "dd"}, LoraSpec("d.safetensors", 0.5, "dd")),
])
def test_lora_spec_from_mapping(entry, expected):
    assert LoraSpec.from_mapping(entry) == expected


def test_lora_spec_from_mapping_requires_path():
    with pytest.raises(PipelineError, match="no 'path'"):
        LoraSpec.from_mapping({"strength": 1.0})


# construction / device ------------------------------------------------------------------------

def test_initial_state(gen):
    assert gen.model_checkpoint == "model.safetensors"
    assert gen.pipeline is None and gen.inpaint_pipeline is None
    assert gen.pipelines == []
    assert gen.get_scheduler_config() is None


def test_checkpoint_path_normalized_to_str(fake_pipeline_cls):
    assert DummyGenerator(Path("/m/x.safetensors"), pipeline_cls=fake_pipeline_cls).model_checkpoint == "/m/x.safetensors"


def test_explicit_device_injection(gen, cpu_device):
    gen.load_pipeline()
    assert (gen.accelerator, gen.torch_device, gen.dtype, gen.dtype_source) == (
        "cpu", "cpu", torch.float32, "config")


def test_falls_back_to_process_device(fake_pipeline_cls):
    dm.configure_device({"backend": "cpu", "dtype": "bfloat16"})
    g = DummyGenerator("m", pipeline_cls=fake_pipeline_cls)
    g.load_pipeline()
    assert g.torch_device == "cpu"
    assert g.dtype == torch.bfloat16


def test_abstract_methods_enforced():
    with pytest.raises(TypeError):
        BasePipelineGenerator("m")


# forward ----------------------------------------------------------------------------------------

def test_forward_requires_pipeline(gen):
    with pytest.raises(PipelineError, match="No model loaded"):
        gen.forward("p", "n", "Euler", 10, 64, 32, 7.0, 1)


def test_forward(loaded):
    image, meta = loaded.forward("a cat", "blurry", "Euler", 12, 64, 32, 6.5, 1234, clip_skip=2)
    assert isinstance(image, np.ndarray)
    assert image.shape == (32, 64, 3)
    assert image.dtype == np.uint8
    call = loaded.pipeline.calls[0]
    assert call["prompt"] == "a cat"
    assert call["negative_prompt"] == "blurry"
    assert call["num_inference_steps"] == 12
    assert (call["width"], call["height"]) == (64, 32)
    assert call["guidance_scale"] == 6.5
    assert call["clip_skip"] == 2
    assert isinstance(call["generator"], torch.Generator)
    assert call["generator"].initial_seed() == 1234
    assert meta == {"instances": [{
        "prompt": "a cat", "negative_prompt": "blurry", "num_inference_steps": 12, "width": 64,
        "height": 32, "guidance_scale": 6.5, "seed": 1234, "scheduler": "Euler"}]}


def test_forward_random_seed_matches_metadata(loaded, monkeypatch):
    seeds = iter([111, 222])
    monkeypatch.setattr("metastable.seed.get_random_seed", lambda bits: next(seeds))
    _, meta = loaded.forward("p", "n", "Euler", 1, 64, 32, 7, -1)
    assert meta["instances"][0]["seed"] == 111
    assert loaded.pipeline.calls[0]["generator"].initial_seed() == 111


def test_forward_is_deterministic_per_seed(loaded):
    loaded.forward("p", "n", None, 1, 64, 32, 7, 5)
    loaded.forward("p", "n", None, 1, 64, 32, 7, 5)
    g1, g2 = (c["generator"] for c in loaded.pipeline.calls)
    assert torch.equal(torch.rand(3, generator=g1), torch.rand(3, generator=g2))


def test_metadata_reports_actual_output_size(loaded, fake_pipeline_cls):
    class FixedSize(fake_pipeline_cls):
        def __call__(self, **kwargs):
            super().__call__(**kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (16, 8))])

    loaded.pipeline = FixedSize()
    _, meta = loaded.forward("p", "n", None, 1, 1024, 1024, 7, 1)
    assert (meta["instances"][0]["width"], meta["instances"][0]["height"]) == (16, 8)


def test_scheduler_swapped_from_config(loaded):
    loaded.pipeline.scheduler = EulerDiscreteScheduler(num_train_timesteps=500)
    loaded.forward("p", "n", "DPM++ 2M", 1, 64, 32, 7, 1)
    assert isinstance(loaded.pipeline.scheduler, DPMSolverMultistepScheduler)
    assert loaded.pipeline.scheduler.config.num_train_timesteps == 500


def test_none_scheduler_keeps_current(loaded):
    current = loaded.pipeline.scheduler
    _, meta = loaded.forward("p", "n", None, 1, 64, 32, 7, 1)
    assert loaded.pipeline.scheduler is current
    assert meta["instances"][0]["scheduler"] is None


def test_unknown_scheduler(loaded):
    with pytest.raises(PipelineError, match="Unknown scheduler 'Bogus'.*DPM\\+\\+ 2M"):
        loaded.forward("p", "n", "Bogus", 1, 64, 32, 7, 1)
    assert loaded.pipeline.calls == []


# forward_inpaint ----------------------------------------------------------------------------------

def test_inpaint_requires_pipeline(loaded):
    with pytest.raises(PipelineError, match="No inpainting model"):
        loaded.forward_inpaint("p", "n", Image.new("RGB", (8, 8)), Image.new("L", (8, 8)), "Euler", 1, 7, 1)


def test_inpaint(loaded_inpaint):
    image = Image.new("RGB", (48, 24))
    mask = np.ones((24, 48, 3), dtype=np.float64)
    result, meta = loaded_inpaint.forward_inpaint("p", "n", image, mask, "Euler a", 8, 4.0, 3, strength=0.7)
    call = loaded_inpaint.inpaint_pipeline.calls[0]
    assert call["image"] is image
    assert call["mask_image"].mode == "L"
    assert call["mask_image"].getpixel((0, 0)) == 255
    assert call["strength"] == 0.7
    assert "width" not in call and "height" not in call
    assert result.shape == (24, 48, 3)
    assert meta["inpainting"] is True
    assert meta["instances"][0]["width"] == 48
    assert meta["instances"][0]["num_inference_steps"] == 8


def test_inpaint_accepts_numpy_image(loaded_inpaint):
    loaded_inpaint.forward_inpaint("p", "n", np.zeros((8, 8, 3), np.uint8), Image.new("L", (8, 8)), None, 1, 7, 1)
    assert isinstance(loaded_inpaint.inpaint_pipeline.calls[0]["image"], Image.Image)


def test_scheduler_config_prefers_inpaint(loaded):
    loaded.load_inpaint_pipeline()
    loaded.inpaint_pipeline.scheduler = EulerDiscreteScheduler(num_train_timesteps=321)
    assert loaded.get_scheduler_config().num_train_timesteps == 321
    loaded.inpaint_pipeline = None
    assert loaded.get_scheduler_config().num_train_timesteps == 1000


# LoRAs --------------------------------------------------------------------------------------------

def test_add_loras_noop(gen):
    assert gen.add_loras(None) == []
    assert gen.add_loras([]) == []


def test_add_loras_without_pipeline(gen):
    with pytest.raises(PipelineError, match="No pipeline loaded"):
        gen.add_loras([LoraSpec("a.safetensors")])


def test_add_loras_on_all_pipelines(loaded):
    loaded.load_inpaint_pipeline()
    fused = loaded.add_loras([
        LoraSpec("/l/style.safetensors", 0.8),
        {"lora_path": "/l/detail.safetensors", "merge_strength": 0.4},
    ])
    assert fused == ["name_style", "name_detail"]
    for pipe in loaded.pipelines:
        assert pipe.loaded_loras == [(Path("/l/style.safetensors"), "name_style"),
                                     (Path("/l/detail.safetensors"), "name_detail")]
        assert pipe.fused == [{"lora_scale": 0.8, "adapter_name": "name_style"},
                              {"lora_scale": 0.4, "adapter_name": "name_detail"}]


def test_add_loras_skips_active_adapters(loaded):
    loaded.add_loras([LoraSpec("/l/style.safetensors")])
    assert loaded.add_loras([LoraSpec("/l/style.safetensors")]) == []
    assert len(loaded.pipeline.loaded_loras) == 1


def test_add_loras_invalid_entry(loaded):
    with pytest.raises(PipelineError):
        loaded.add_loras([{"strength": 1}])


# placement ------------------------------------------------------------------------------------------

def test_to_device(loaded):
    loaded.load_inpaint_pipeline()
    loaded.to_device()
    assert [p.to_calls for p in loaded.pipelines] == [["cpu"], ["cpu"]]
    assert loaded._placement == "device"


# custom VAE ---------------------------------------------------------------------------------------------

class FakeVAE:
    def __init__(self):
        self.moved_to = None

    def to(self, device):
        self.moved_to = device
        return self


@pytest.fixture
def fake_vae_loader(monkeypatch):
    calls = []

    def _load(path, dtype=None):
        calls.append((path, dtype))
        return FakeVAE()

    monkeypatch.setattr(base, "load_custom_vae", _load)
    return calls


def test_custom_vae_requires_pipeline(gen, fake_vae_loader):
    with pytest.raises(PipelineError, match="No pipeline loaded"):
        gen.load_custom_vae("vae.safetensors")
    assert fake_vae_loader == []


def test_custom_vae_before_placement(loaded, fake_vae_loader):
    loaded.load_inpaint_pipeline()
    loaded.load_custom_vae("vae.safetensors")
    assert fake_vae_loader == [("vae.safetensors", torch.float32)]
    vae = loaded.pipeline.vae
    assert isinstance(vae, FakeVAE)
    assert loaded.inpaint_pipeline.vae is vae  # both pipelines get it
    assert vae.moved_to is None


def test_custom_vae_after_device_placement(loaded, fake_vae_loader):
    loaded.to_device()
    loaded.load_custom_vae("vae.safetensors")
    assert loaded.pipeline.vae.moved_to == "cpu"


def test_custom_vae_after_offload(loaded, fake_vae_loader):
    loaded._placement = "offload"
    loaded.load_custom_vae("vae.safetensors")
    assert loaded.pipeline.vae.moved_to is None
    assert loaded.pipeline.offload_calls == 1
