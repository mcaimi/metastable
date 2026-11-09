#!/usr/bin/env python
"""
Flux Pipeline Generator

Unlike SD1.5/SDXL/Illustrious, a Flux checkpoint is not a complete
single-file pipeline: it only ships the transformer. The CLIP-L + T5-XXL
text encoders/tokenizers, VAE, and scheduler are loaded once from a shared
local diffusers-format directory and reused across every Flux checkpoint.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from diffusers import (
    FluxPipeline,
    FluxInpaintPipeline,
    FluxTransformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
)
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from metastable.device import DeviceSettings
from metastable.pipelines.base import BasePipelineGenerator
from metastable.schedulers import FLUX_SCHEDULERS

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def load_flux_shared_components(components_dir: str, dtype: torch.dtype) -> dict:
    """Load Flux's shared text encoders/tokenizers/VAE/scheduler once.

    Cached so the (large) T5-XXL/CLIP/VAE weights are loaded a single time
    per (components_dir, dtype) and reused in-memory across every Flux
    checkpoint swap.
    """
    logger.info("Loading shared Flux components from: %s", components_dir)
    return {
        "text_encoder": CLIPTextModel.from_pretrained(
            components_dir, subfolder="text_encoder", torch_dtype=dtype
        ),
        "tokenizer": CLIPTokenizer.from_pretrained(
            components_dir, subfolder="tokenizer"
        ),
        "text_encoder_2": T5EncoderModel.from_pretrained(
            components_dir, subfolder="text_encoder_2", torch_dtype=dtype
        ),
        "tokenizer_2": T5TokenizerFast.from_pretrained(
            components_dir, subfolder="tokenizer_2"
        ),
        "vae": AutoencoderKL.from_pretrained(
            components_dir, subfolder="vae", torch_dtype=dtype
        ),
        "scheduler": FlowMatchEulerDiscreteScheduler.from_pretrained(
            components_dir, subfolder="scheduler"
        ),
    }


class FluxPipelineGenerator(BasePipelineGenerator):
    """Flux Pipeline Generator for text-to-image and inpainting."""

    PIPELINE_CLASS = FluxPipeline
    INPAINT_PIPELINE_CLASS = FluxInpaintPipeline
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    MODEL_TYPE = "flux"
    SCHEDULERS = FLUX_SCHEDULERS

    def __init__(
        self,
        model_checkpoint: Union[str, Path],
        components_dir: Union[str, Path],
        device: Optional[DeviceSettings] = None,
    ):
        """
        Args:
            model_checkpoint: Path to the transformer-only safetensors file
            components_dir: Shared diffusers-format directory holding the
                text encoders, tokenizers, VAE and scheduler
            device: Explicit device settings (see BasePipelineGenerator)
        """
        super().__init__(model_checkpoint, device=device)
        self.components_dir: str = str(components_dir)

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

    def _init_device(self) -> None:
        """Initialize GPU/device settings, preferring bf16 for Flux/T5.

        bf16 is only auto-selected when the user hasn't pinned a dtype in the
        device config: ROCm hosts (especially older RDNA GPUs like the RX
        5500 XT) default to float16, and an explicit dtype always wins.
        """
        super()._init_device()
        if self.accelerator != "cpu" and self.dtype_source == "auto":
            self.dtype = torch.bfloat16

    def _build(self, pipeline_class, label: str):
        """Assemble a Flux pipeline from the checkpoint transformer and the
        shared components."""
        self._init_device()
        logger.info("Loading Flux %stransformer checkpoint: %s", label, self.model_checkpoint)
        transformer = FluxTransformer2DModel.from_single_file(
            self.model_checkpoint, torch_dtype=self.dtype
        )
        shared = load_flux_shared_components(self.components_dir, self.dtype)
        return pipeline_class(transformer=transformer, **shared)

    def load_pipeline(self) -> None:
        """Load the Flux generation pipeline."""
        self.pipeline = self._build(self.PIPELINE_CLASS, "")

    def load_inpaint_pipeline(self) -> None:
        """Load the Flux inpainting pipeline."""
        self.inpaint_pipeline = self._build(self.INPAINT_PIPELINE_CLASS, "inpaint ")

    def to_device(self) -> None:
        """Move pipeline(s) to the configured compute device.

        Flux is large enough that a naive .to(device) often won't fit in
        consumer VRAM, so model CPU offload is used instead whenever an
        accelerator is available. enable_model_cpu_offload() raises on a
        CPU-only host, so fall back to the base behavior there.
        """
        if self.accelerator == "cpu":
            super().to_device()
            return

        for pipeline in self.pipelines:
            pipeline.enable_model_cpu_offload()
        self._placement = "offload"

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
