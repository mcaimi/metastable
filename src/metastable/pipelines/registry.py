#!/usr/bin/env python
"""
Pipeline Generator Registry

Maps model type identifiers to their generator classes.
"""

from pathlib import Path
from typing import Dict, List, Type, Union

from metastable.exceptions import PipelineError
from metastable.pipelines.base import BasePipelineGenerator
from metastable.pipelines.flux import FluxPipelineGenerator
from metastable.pipelines.illustrious import IllustriousPipelineGenerator
from metastable.pipelines.sd15 import SD15PipelineGenerator
from metastable.pipelines.sdxl import SDXLPipelineGenerator

PIPELINES: Dict[str, Type[BasePipelineGenerator]] = {
    cls.MODEL_TYPE: cls
    for cls in (
        SD15PipelineGenerator,
        SDXLPipelineGenerator,
        IllustriousPipelineGenerator,
        FluxPipelineGenerator,
    )
}


def available_model_types() -> List[str]:
    """Return the registered model type identifiers."""
    return list(PIPELINES)


def get_generator_class(model_type: str) -> Type[BasePipelineGenerator]:
    """
    Look up the generator class for a model type.

    Raises:
        PipelineError: If the model type is unknown
    """
    try:
        return PIPELINES[model_type]
    except KeyError:
        raise PipelineError(
            f"Unknown model type '{model_type}'. "
            f"Available: {', '.join(PIPELINES)}"
        ) from None


def create_generator(
    model_type: str, model_checkpoint: Union[str, Path], **kwargs
) -> BasePipelineGenerator:
    """
    Instantiate (without loading) the generator for a model type.

    Args:
        model_type: One of available_model_types()
        model_checkpoint: Path to the checkpoint
        **kwargs: Extra constructor arguments (e.g. components_dir for flux,
            device for an explicit DeviceSettings)
    """
    return get_generator_class(model_type)(model_checkpoint, **kwargs)
