"""Stable Diffusion pipeline generators."""

from metastable.pipelines.base import (
    BasePipelineGenerator,
    LoraSpec,
    format_metadata,
    load_custom_unet,
    load_custom_vae,
)
from metastable.pipelines.flux import FluxPipelineGenerator, load_flux_shared_components
from metastable.pipelines.illustrious import IllustriousPipelineGenerator
from metastable.pipelines.metadata import GenerationMetadata
from metastable.pipelines.registry import (
    PIPELINES,
    available_model_types,
    create_generator,
    get_generator_class,
)
from metastable.pipelines.sd15 import SD15PipelineGenerator
from metastable.pipelines.sdxl import SDXLPipelineGenerator

__all__ = [
    "BasePipelineGenerator",
    "FluxPipelineGenerator",
    "GenerationMetadata",
    "IllustriousPipelineGenerator",
    "LoraSpec",
    "PIPELINES",
    "SD15PipelineGenerator",
    "SDXLPipelineGenerator",
    "available_model_types",
    "create_generator",
    "format_metadata",
    "get_generator_class",
    "load_custom_unet",
    "load_custom_vae",
    "load_flux_shared_components",
]
