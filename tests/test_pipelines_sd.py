import pytest
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLPipeline,
)

from metastable.pipelines import (
    IllustriousPipelineGenerator,
    SD15PipelineGenerator,
    SDXLPipelineGenerator,
)
from metastable.schedulers import SCHEDULERS

GENERATORS = [SD15PipelineGenerator, SDXLPipelineGenerator, IllustriousPipelineGenerator]


class RecordingPipelineClass:
    calls = []

    @classmethod
    def from_single_file(cls, path, **kwargs):
        cls.calls.append((path, kwargs))
        return cls()


@pytest.fixture
def recording(monkeypatch):
    def _patch(generator_cls):
        txt = type("Txt", (RecordingPipelineClass,), {"calls": []})
        inp = type("Inp", (RecordingPipelineClass,), {"calls": []})
        monkeypatch.setattr(generator_cls, "PIPELINE_CLASS", txt)
        monkeypatch.setattr(generator_cls, "INPAINT_PIPELINE_CLASS", inp)
        return txt, inp
    return _patch


@pytest.mark.parametrize("cls,txt,inp,model_type,default", [
    (SD15PipelineGenerator, StableDiffusionPipeline, StableDiffusionInpaintPipeline, "sd15", 512),
    (SDXLPipelineGenerator, StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, "sdxl", 1024),
    (IllustriousPipelineGenerator, StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline, "illustrious", 1024),
])
def test_class_attributes(cls, txt, inp, model_type, default):
    assert cls.PIPELINE_CLASS is txt
    assert cls.INPAINT_PIPELINE_CLASS is inp
    assert cls.MODEL_TYPE == model_type
    assert cls.DEFAULT_WIDTH == cls.DEFAULT_HEIGHT == default
    assert cls.SCHEDULERS is SCHEDULERS


def test_sd15_resolutions():
    assert SD15PipelineGenerator.get_resolutions() == {
        "512x512": (512, 512), "512x768": (512, 768), "768x512": (768, 512)}


def test_sdxl_resolutions():
    res = SDXLPipelineGenerator.get_resolutions()
    assert len(res) == 9
    assert res["1024x1024"] == (1024, 1024)
    assert res["640x1536"] == (640, 1536)


def test_illustrious_extends_sdxl_resolutions():
    res = IllustriousPipelineGenerator.get_resolutions()
    assert SDXLPipelineGenerator.get_resolutions().items() <= res.items()
    assert res["1280x1280"] == (1280, 1280)
    assert res["1536x1536"] == (1536, 1536)
    assert len(res) == 11


@pytest.mark.parametrize("name,size", [(n, s) for cls in GENERATORS for n, s in cls.get_resolutions().items()])
def test_resolution_keys_match_values(name, size):
    assert name == f"{size[0]}x{size[1]}"


@pytest.mark.parametrize("cls", GENERATORS)
def test_load_pipeline(cls, recording, cpu_device):
    txt, inp = recording(cls)
    gen = cls("/m/model.safetensors", device=cpu_device)
    gen.load_pipeline()
    assert isinstance(gen.pipeline, txt)
    assert gen.inpaint_pipeline is None
    assert txt.calls == [("/m/model.safetensors", {"torch_dtype": torch.float32, "use_safetensors": True})]
    assert inp.calls == []


@pytest.mark.parametrize("cls", GENERATORS)
def test_load_inpaint_pipeline(cls, recording, gpu_device):
    txt, inp = recording(cls)
    gen = cls("/m/model.safetensors", device=gpu_device)
    gen.load_inpaint_pipeline()
    assert isinstance(gen.inpaint_pipeline, inp)
    assert gen.pipeline is None
    assert inp.calls == [("/m/model.safetensors", {"torch_dtype": torch.float16, "use_safetensors": True})]
    assert gen.torch_device == "cuda:0"
