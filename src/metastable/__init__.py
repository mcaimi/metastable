"""
metastable - Stable Diffusion middleware.

Pipeline generators (SD1.5, SDXL, Illustrious XL, Flux), compute device
selection and checkpoint merging, usable from API backends and standalone
applications alike.

Public names are resolved lazily, so `import metastable` stays cheap and
heavy dependencies (diffusers, transformers) load on first use.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from metastable.exceptions import (
    CheckpointLoadError,
    ConfigurationError,
    MergeError,
    MetastableError,
    PipelineError,
)

__version__ = "0.1.0"

# public name -> defining module
_LAZY_EXPORTS = {
    # device
    "DeviceSettings": "metastable.device",
    "configure_device": "metastable.device",
    "get_device_settings": "metastable.device",
    "get_torch_device": "metastable.device",
    "reset_device_settings": "metastable.device",
    "resolve_device_settings": "metastable.device",
    # schedulers
    "SCHEDULERS": "metastable.schedulers",
    "FLUX_SCHEDULERS": "metastable.schedulers",
    # seed
    "get_random_seed": "metastable.seed",
    "resolve_seed": "metastable.seed",
    # model files
    "enumerate_models": "metastable.models",
    "read_safetensors_header": "metastable.models",
    "read_safetensors_metadata": "metastable.models",
    # pipelines
    "BasePipelineGenerator": "metastable.pipelines",
    "FluxPipelineGenerator": "metastable.pipelines",
    "GenerationMetadata": "metastable.pipelines",
    "IllustriousPipelineGenerator": "metastable.pipelines",
    "LoraSpec": "metastable.pipelines",
    "PIPELINES": "metastable.pipelines",
    "SD15PipelineGenerator": "metastable.pipelines",
    "SDXLPipelineGenerator": "metastable.pipelines",
    "available_model_types": "metastable.pipelines",
    "create_generator": "metastable.pipelines",
    "get_generator_class": "metastable.pipelines",
    # merge
    "MergeConfig": "metastable.merge",
    "MergeMethod": "metastable.merge",
    "MergePipeline": "metastable.merge",
}

__all__ = [
    "CheckpointLoadError",
    "ConfigurationError",
    "MergeError",
    "MetastableError",
    "PipelineError",
    "__version__",
    *_LAZY_EXPORTS,
]


def __getattr__(name: str):
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'metastable' has no attribute '{name}'")
    value = getattr(import_module(module_name), name)
    globals()[name] = value  # cache for subsequent lookups
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))


if TYPE_CHECKING:  # pragma: no cover - static analyzers only
    from metastable.device import (  # noqa: F401
        DeviceSettings,
        configure_device,
        get_device_settings,
        get_torch_device,
        reset_device_settings,
        resolve_device_settings,
    )
    from metastable.merge import MergeConfig, MergeMethod, MergePipeline  # noqa: F401
    from metastable.models import (  # noqa: F401
        enumerate_models,
        read_safetensors_header,
        read_safetensors_metadata,
    )
    from metastable.pipelines import (  # noqa: F401
        PIPELINES,
        BasePipelineGenerator,
        FluxPipelineGenerator,
        GenerationMetadata,
        IllustriousPipelineGenerator,
        LoraSpec,
        SD15PipelineGenerator,
        SDXLPipelineGenerator,
        available_model_types,
        create_generator,
        get_generator_class,
    )
    from metastable.schedulers import FLUX_SCHEDULERS, SCHEDULERS  # noqa: F401
    from metastable.seed import get_random_seed, resolve_seed  # noqa: F401
