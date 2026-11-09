import pytest

from metastable.device import DeviceSettings
from metastable.exceptions import PipelineError
from metastable.pipelines import (
    PIPELINES,
    FluxPipelineGenerator,
    IllustriousPipelineGenerator,
    SD15PipelineGenerator,
    SDXLPipelineGenerator,
    available_model_types,
    create_generator,
    get_generator_class,
)


def test_registry_contents():
    assert PIPELINES == {
        "sd15": SD15PipelineGenerator,
        "sdxl": SDXLPipelineGenerator,
        "illustrious": IllustriousPipelineGenerator,
        "flux": FluxPipelineGenerator,
    }
    assert available_model_types() == ["sd15", "sdxl", "illustrious", "flux"]


def test_keys_match_model_type():
    for key, cls in PIPELINES.items():
        assert cls.MODEL_TYPE == key


def test_get_generator_class_unknown():
    with pytest.raises(PipelineError, match="Unknown model type 'sd3'.*sd15"):
        get_generator_class("sd3")


def test_create_generator_does_not_load():
    gen = create_generator("sdxl", "/m/x.safetensors")
    assert isinstance(gen, SDXLPipelineGenerator)
    assert gen.pipeline is None


def test_create_generator_forwards_kwargs():
    device = DeviceSettings.cpu()
    gen = create_generator("flux", "/m/f.safetensors", components_dir="/m/c", device=device)
    assert isinstance(gen, FluxPipelineGenerator)
    assert gen.components_dir == "/m/c"
    assert gen._device_override is device


def test_create_flux_requires_components_dir():
    with pytest.raises(TypeError):
        create_generator("flux", "/m/f.safetensors")
