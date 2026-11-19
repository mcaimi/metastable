#!/usr/bin/env python
"""
Flux Pipeline Generator

Unlike SD1.5/SDXL/Illustrious, a Flux checkpoint is not a complete
single-file pipeline: it only ships the transformer. The CLIP-L + T5-XXL
text encoders/tokenizers, VAE, and scheduler are loaded once from a shared
local diffusers-format directory and reused across every Flux checkpoint
(see TransformerPipelineGenerator for sharing, locking and offload).
"""

import logging
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
from diffusers import (
    FluxPipeline,
    FluxInpaintPipeline,
    FluxTransformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from metastable.pipelines.transformer import (  # noqa: F401 - re-exported
    TransformerPipelineGenerator,
    _remove_offload_hooks,
    components_lock,
)
from metastable.schedulers import FLOW_MATCH_SCHEDULERS

logger = logging.getLogger(__name__)


# component name -> (class, subfolder, loaded with torch_dtype)
_FLUX_COMPONENTS = {
    "text_encoder": ("CLIPTextModel", "text_encoder", True),
    "tokenizer": ("CLIPTokenizer", "tokenizer", False),
    "text_encoder_2": ("T5EncoderModel", "text_encoder_2", True),
    "tokenizer_2": ("T5TokenizerFast", "tokenizer_2", False),
    "vae": ("AutoencoderKL", "vae", True),
    "scheduler": ("FlowMatchEulerDiscreteScheduler", "scheduler", False),
}


def load_flux_component(components_dir: str, name: str, dtype: torch.dtype):
    """Load a single Flux component from the shared directory (uncached).

    Used to give a pipeline a private copy of a component it is about to
    modify (e.g. a text encoder receiving fused LoRA weights).
    """
    class_name, subfolder, takes_dtype = _FLUX_COMPONENTS[name]
    # Resolved through module globals so tests can patch the classes
    component_class = globals()[class_name]
    kwargs = {"torch_dtype": dtype} if takes_dtype else {}
    return component_class.from_pretrained(components_dir, subfolder=subfolder, **kwargs)


@lru_cache(maxsize=2)
def load_flux_shared_components(components_dir: str, dtype: torch.dtype) -> dict:
    """Load Flux's shared text encoders/tokenizers/VAE/scheduler once.

    Cached so the (large) T5-XXL/CLIP/VAE weights are loaded a single time
    per (components_dir, dtype) and reused in-memory across every Flux
    checkpoint swap. The returned modules must never be modified in place:
    see TransformerPipelineGenerator._isolate_lora_targets().
    """
    logger.info("Loading shared Flux components from: %s", components_dir)
    return {
        name: load_flux_component(components_dir, name, dtype)
        for name in _FLUX_COMPONENTS
    }


class FluxPipelineGenerator(TransformerPipelineGenerator):
    """Flux Pipeline Generator for text-to-image and inpainting."""

    PIPELINE_CLASS = FluxPipeline
    INPAINT_PIPELINE_CLASS = FluxInpaintPipeline
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    MODEL_TYPE = "flux"
    DISPLAY_NAME = "Flux"
    SCHEDULERS = FLOW_MATCH_SCHEDULERS

    @staticmethod
    def get_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported Flux resolutions (16px-aligned)."""
        return {
            "1024x1024": (1024, 1024),
            "1360x768": (1360, 768),
            "768x1360": (768, 1360),
            "1536x1024": (1536, 1024),
            "1024x1536": (1024, 1536),
            "896x1152": (896, 1152),
            "1152x896": (1152, 896),
        }

    # Loader hooks resolve the module-level names at call time, so tests
    # can patch them on this module.
    def _load_shared_components(self) -> dict:
        return load_flux_shared_components(self.components_dir, self.dtype)

    def _load_component(self, name: str):
        return load_flux_component(self.components_dir, name, self.dtype)

    def _load_transformer(self, **kwargs):
        return FluxTransformer2DModel.from_single_file(self.model_checkpoint, **kwargs)

    def forward(
        self,
        positive_prompt: str,
        negative_prompt: str,
        scheduler_type: Optional[str],
        steps: int,
        width: int,
        height: int,
        cfg: float,
        seed: int,
        true_cfg_scale: float = 1.0,
        **extra_kwargs,
    ) -> Tuple:
        """Generate an image from text prompts.

        true_cfg_scale: Flux's negative_prompt is only honored when this is
        greater than 1.0 (diffusers FluxPipeline semantics).
        """
        return super().forward(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            scheduler_type=scheduler_type,
            steps=steps,
            width=width,
            height=height,
            cfg=cfg,
            seed=seed,
            true_cfg_scale=true_cfg_scale,
            **extra_kwargs,
        )

    def forward_inpaint(
        self,
        positive_prompt: str,
        negative_prompt: str,
        image,
        mask_image,
        scheduler_type: Optional[str],
        steps: int,
        cfg: float,
        seed: int,
        strength: Optional[float] = None,
        true_cfg_scale: float = 1.0,
        **extra_kwargs,
    ) -> Tuple:
        """Inpaint an image region.

        strength: Flux's inpaint pipeline re-noises the masked region rather
        than denoising from pure noise; None lets the pipeline use its own
        default (0.6).
        """
        extra_kwargs["true_cfg_scale"] = true_cfg_scale
        if strength is not None:
            extra_kwargs["strength"] = strength

        return super().forward_inpaint(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            scheduler_type=scheduler_type,
            steps=steps,
            cfg=cfg,
            seed=seed,
            **extra_kwargs,
        )
