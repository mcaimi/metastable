#!/usr/bin/env python
"""
Flux Pipeline Generator

Unlike SD1.5/SDXL/Illustrious, a Flux checkpoint is not a complete
single-file pipeline: it only ships the transformer. The CLIP-L + T5-XXL
text encoders/tokenizers, VAE, and scheduler are loaded once from a shared
local diffusers-format directory and reused across every Flux checkpoint.

Because those modules are shared, every generator using the same components
directory also shares one lock: inference, LoRA fusion and placement changes
on any of them run one at a time (model CPU offload moves shared modules
between devices, which would break a concurrent run).
"""

import logging
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from accelerate.hooks import remove_hook_from_module
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


# resolved components directory -> lock shared by its generators
_COMPONENT_LOCKS: Dict[str, threading.RLock] = {}
_COMPONENT_LOCKS_GUARD = threading.Lock()


def components_lock(components_dir: Union[str, Path]) -> threading.RLock:
    """Return the lock shared by every generator using components_dir.

    Keyed by the resolved path only (not dtype): a coarser key can only
    serialize more than needed, never race.
    """
    key = str(Path(components_dir).resolve())
    with _COMPONENT_LOCKS_GUARD:
        return _COMPONENT_LOCKS.setdefault(key, threading.RLock())


def _remove_offload_hooks(components: dict) -> None:
    """Strip accelerate hooks from shared modules.

    A shared module's offload hook references the previous model in the
    offload chain of the last pipeline that ran (e.g. the VAE's hook holds
    that pipeline's transformer), keeping it alive after its generator is
    dropped. Pipelines still using the modules reinstall their hooks before
    their next run.
    """
    for module in components.values():
        if isinstance(module, torch.nn.Module) and hasattr(module, "_hf_hook"):
            remove_hook_from_module(module, recurse=True)


@lru_cache(maxsize=2)
def load_flux_shared_components(components_dir: str, dtype: torch.dtype) -> dict:
    """Load Flux's shared text encoders/tokenizers/VAE/scheduler once.

    Cached so the (large) T5-XXL/CLIP/VAE weights are loaded a single time
    per (components_dir, dtype) and reused in-memory across every Flux
    checkpoint swap. The returned modules must never be modified in place:
    see FluxPipelineGenerator._isolate_lora_targets().
    """
    logger.info("Loading shared Flux components from: %s", components_dir)
    return {
        name: load_flux_component(components_dir, name, dtype)
        for name in _FLUX_COMPONENTS
    }


class FluxPipelineGenerator(BasePipelineGenerator):
    """Flux Pipeline Generator for text-to-image and inpainting."""

    PIPELINE_CLASS = FluxPipeline
    INPAINT_PIPELINE_CLASS = FluxInpaintPipeline
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    MODEL_TYPE = "flux"
    SCHEDULERS = FLOW_MATCH_SCHEDULERS

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
        self._lock = components_lock(self.components_dir)

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
        shared components.

        Stale offload hooks are stripped from the shared modules before the
        transformer is loaded, so a previous checkpoint's transformer is
        released first rather than coexisting with the new one in memory.
        Each pipeline gets its own scheduler: schedulers carry per-run state.
        """
        self._init_device()
        with self._lock:
            shared = dict(load_flux_shared_components(self.components_dir, self.dtype))
            _remove_offload_hooks(shared)
        scheduler = shared["scheduler"]
        shared["scheduler"] = type(scheduler).from_config(scheduler.config)

        logger.info("Loading Flux %stransformer checkpoint: %s", label, self.model_checkpoint)
        transformer = FluxTransformer2DModel.from_single_file(
            self.model_checkpoint, torch_dtype=self.dtype
        )
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

        with self._lock:
            for pipeline in self.pipelines:
                self._enable_offload(pipeline)
            self._placement = "offload"

    def _isolate_lora_targets(self, pipeline) -> None:
        """Give the pipeline private copies of shared LoRA-loadable components.

        Fusing a LoRA rewrites weights in place. Components that are still
        the cached shared instances (e.g. the CLIP text encoder) are
        reloaded from disk first, so the fused weights never leak into the
        cache or into other pipelines. Components a LoRA can't touch (T5,
        VAE) stay shared.
        """
        shared = load_flux_shared_components(self.components_dir, self.dtype)
        isolated = False
        for name in getattr(pipeline, "_lora_loadable_modules", ()):
            if name not in shared or getattr(pipeline, name, None) is not shared[name]:
                continue
            logger.info("Loading private Flux %s for LoRA fusion", name)
            private = load_flux_component(self.components_dir, name, self.dtype)
            if self._placement == "device":
                private = private.to(self.torch_device)
            setattr(pipeline, name, private)
            isolated = True

        if isolated and self._placement == "offload":
            # New modules carry no offload hooks; reinstall them
            self._enable_offload(pipeline)

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
