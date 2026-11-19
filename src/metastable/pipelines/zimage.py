#!/usr/bin/env python
"""
Z-Image Pipeline Generator

Z-Image (Tongyi-MAI) checkpoints only ship the transformer. The Qwen3 text
encoder, tokenizer, VAE and scheduler are loaded once from a shared local
diffusers-format directory and reused across every Z-Image checkpoint (see
TransformerPipelineGenerator for sharing, locking and offload).

The same generator serves both Z-Image Base and Z-Image Turbo: they share the
architecture and differ in weights and recommended settings.

    checkpoint  steps   cfg (guidance_scale)  negative prompt
    Turbo       ~8      0.0                   ignored (CFG off)
    Base        28-50   ~3-5                  used

CFG is enabled whenever cfg > 0. Pair each checkpoint with the components
directory from its own repository (Tongyi-MAI/Z-Image-Turbo or
Tongyi-MAI/Z-Image): the text encoder and VAE are identical, but the
scheduler shift differs (3.0 for Turbo, 6.0 for Base).
"""

import logging
from functools import lru_cache
from typing import Dict, Optional, Tuple

import torch
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImageInpaintPipeline,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from transformers import AutoTokenizer, Qwen3Model

from metastable.pipelines.transformer import TransformerPipelineGenerator
from metastable.schedulers import FLOW_MATCH_SCHEDULERS

logger = logging.getLogger(__name__)


# component name -> (class, subfolder, loaded with torch_dtype)
_ZIMAGE_COMPONENTS = {
    "text_encoder": ("Qwen3Model", "text_encoder", True),
    "tokenizer": ("AutoTokenizer", "tokenizer", False),
    "vae": ("AutoencoderKL", "vae", True),
    "scheduler": ("FlowMatchEulerDiscreteScheduler", "scheduler", False),
}


def load_zimage_component(components_dir: str, name: str, dtype: torch.dtype):
    """Load a single Z-Image component from the shared directory (uncached)."""
    class_name, subfolder, takes_dtype = _ZIMAGE_COMPONENTS[name]
    # Resolved through module globals so tests can patch the classes
    component_class = globals()[class_name]
    kwargs = {"torch_dtype": dtype} if takes_dtype else {}
    return component_class.from_pretrained(components_dir, subfolder=subfolder, **kwargs)


@lru_cache(maxsize=2)
def load_zimage_shared_components(components_dir: str, dtype: torch.dtype) -> dict:
    """Load Z-Image's shared text encoder/tokenizer/VAE/scheduler once.

    Cached so the (large) Qwen3 text encoder and the VAE are loaded a single
    time per (components_dir, dtype) and reused in-memory across every
    Z-Image checkpoint swap. The returned modules must never be modified in
    place: see TransformerPipelineGenerator._isolate_lora_targets().
    """
    logger.info("Loading shared Z-Image components from: %s", components_dir)
    return {
        name: load_zimage_component(components_dir, name, dtype)
        for name in _ZIMAGE_COMPONENTS
    }


class ZImagePipelineGenerator(TransformerPipelineGenerator):
    """Z-Image (Base and Turbo) Pipeline Generator for text-to-image and
    inpainting."""

    PIPELINE_CLASS = ZImagePipeline
    INPAINT_PIPELINE_CLASS = ZImageInpaintPipeline
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    MODEL_TYPE = "zimage"
    DISPLAY_NAME = "Z-Image"
    SCHEDULERS = FLOW_MATCH_SCHEDULERS

    @staticmethod
    def get_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported Z-Image resolutions (16px-aligned)."""
        return {
            "1024x1024": (1024, 1024),
            "1152x896": (1152, 896),
            "896x1152": (896, 1152),
            "1216x832": (1216, 832),
            "832x1216": (832, 1216),
            "1344x768": (1344, 768),
            "768x1344": (768, 1344),
            "1280x720": (1280, 720),
            "720x1280": (720, 1280),
            "1536x1536": (1536, 1536),
        }

    # Loader hooks resolve the module-level names at call time, so tests
    # can patch them on this module.
    def _load_shared_components(self) -> dict:
        return load_zimage_shared_components(self.components_dir, self.dtype)

    def _load_component(self, name: str):
        return load_zimage_component(self.components_dir, name, self.dtype)

    def _load_transformer(self, **kwargs):
        return ZImageTransformer2DModel.from_single_file(self.model_checkpoint, **kwargs)

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
        **extra_kwargs,
    ) -> Tuple:
        """Inpaint an image region.

        strength: how much the masked region is re-noised; None lets the
        pipeline use its own default (1.0, i.e. denoise from pure noise).
        """
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
