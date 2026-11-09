#!/usr/bin/env python
"""
Base classes for Stable Diffusion Pipelines

Provides the abstract base class every architecture-specific pipeline
generator inherits from, plus shared loaders and metadata helpers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import logging

import numpy as np
import torch
from torch import Generator
from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel, DiffusionPipeline

from metastable.device import DeviceSettings, get_device_settings
from metastable.exceptions import PipelineError
from metastable.images import convert_to_pil, convert_mask_to_grayscale
from metastable.schedulers import SCHEDULERS
from metastable.seed import resolve_seed

logger = logging.getLogger(__name__)


def normalize_path(path: Union[str, Path]) -> Path:
    """Convert string or Path to Path object."""
    if isinstance(path, str):
        return Path(path)
    elif isinstance(path, Path):
        return path
    raise TypeError(f"Expected str or Path, got {type(path)}")


def load_custom_vae(
    checkpoint: Union[str, Path], dtype: Optional[torch.dtype] = None
) -> AutoencoderKL:
    """Load a custom Variational Autoencoder from a single-file checkpoint."""
    path = normalize_path(checkpoint)
    logger.info("Loading custom VAE: %s", path)
    return AutoencoderKL.from_single_file(
        str(path.absolute()), subfolder="vae", use_safetensors=True, torch_dtype=dtype
    )


def load_custom_unet(
    checkpoint: Union[str, Path], dtype: Optional[torch.dtype] = None
) -> UNet2DConditionModel:
    """Load custom UNET weights from a single-file checkpoint."""
    path = normalize_path(checkpoint)
    logger.info("Loading custom UNet: %s", path)
    return UNet2DConditionModel.from_single_file(
        str(path.absolute()), subfolder="unet", use_safetensors=True, torch_dtype=dtype
    )


def format_metadata(
    prompt: str,
    negative_prompt: str = "",
    steps: int = 10,
    width: int = 512,
    height: int = 512,
    cfg: float = 7,
    seed: int = -1,
    scheduler: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare generation metadata payload.

    A negative seed is resolved to a random one; pass the seed actually used
    for inference to keep the metadata reproducible.
    """
    return {
        "instances": [
            {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_inference_steps": steps,
                "width": width,
                "height": height,
                "guidance_scale": cfg,
                "seed": resolve_seed(seed),
                "scheduler": scheduler,
            }
        ]
    }


@dataclass(frozen=True)
class LoraSpec:
    """
    A LoRA adapter to fuse into a pipeline.

    Attributes:
        path:     Path to the LoRA safetensors file
        strength: Fusion scale (lora_scale)
        name:     Adapter name; defaults to one derived from the file stem
    """

    path: Union[str, Path]
    strength: float = 0.5
    name: Optional[str] = None

    @property
    def adapter_name(self) -> str:
        return self.name or f"name_{Path(self.path).stem}"

    @classmethod
    def from_mapping(cls, entry: Mapping[str, Any]) -> "LoraSpec":
        """
        Build a LoraSpec from a mapping.

        Accepts "path" or "lora_path" for the file and "strength" or
        "merge_strength" for the scale.
        """
        path = entry.get("path", entry.get("lora_path"))
        if path is None:
            raise PipelineError(f"LoRA entry has no 'path': {dict(entry)}")
        strength = entry.get("strength", entry.get("merge_strength", 0.5))
        return cls(path=path, strength=float(strength), name=entry.get("adapter_name"))


class BasePipelineGenerator(ABC):
    """Abstract base class for Stable Diffusion pipeline generators."""

    # Subclasses must define these
    PIPELINE_CLASS = None
    INPAINT_PIPELINE_CLASS = None
    DEFAULT_WIDTH = 512
    DEFAULT_HEIGHT = 512
    MODEL_TYPE = "base"
    # Scheduler registry used by _run_inference(); override for architectures
    # (e.g. Flux) whose noise schedule isn't compatible with the default set.
    SCHEDULERS: Dict[str, Any] = SCHEDULERS

    def __init__(
        self,
        model_checkpoint: Union[str, Path],
        device: Optional[DeviceSettings] = None,
    ):
        """
        Args:
            model_checkpoint: Path to the single-file checkpoint
            device: Explicit device settings; defaults to the process-wide
                settings from metastable.device.get_device_settings()
        """
        self.model_checkpoint: str = str(model_checkpoint)
        self.pipeline: Optional[DiffusionPipeline] = None
        self.inpaint_pipeline: Optional[DiffusionPipeline] = None
        self.accelerator: Optional[str] = None
        self.torch_device: Optional[str] = None
        self.dtype: Optional[torch.dtype] = None
        self.dtype_source: str = "auto"
        self._device_override: Optional[DeviceSettings] = device
        # Placement state: None (not placed yet), "device" (moved with
        # .to()), or "offload" (model CPU offload hooks installed)
        self._placement: Optional[str] = None

    @staticmethod
    @abstractmethod
    def get_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported resolutions for this model type."""
        pass

    @property
    def pipelines(self) -> List[DiffusionPipeline]:
        """The currently loaded pipelines (txt2img and/or inpaint)."""
        return [p for p in (self.pipeline, self.inpaint_pipeline) if p is not None]

    def _init_device(self) -> None:
        """Initialize GPU/device settings.

        self.accelerator holds the backend label ("cuda", "rocm", "mps",
        "cpu"); self.torch_device holds the torch-level device string to use
        with torch APIs (on AMD ROCm builds that is still "cuda:N").
        """
        settings = self._device_override or get_device_settings()
        self.accelerator = settings.backend
        self.torch_device = settings.torch_device
        self.dtype = settings.dtype
        self.dtype_source = settings.dtype_source

    def _create_generator(self, seed: int) -> Generator:
        """Create a torch Generator with the given (non-negative) seed."""
        return Generator(self.torch_device).manual_seed(seed)

    def _set_scheduler(self, pipeline, scheduler_type: Optional[str]) -> None:
        """Swap the pipeline scheduler; None keeps the current one."""
        if scheduler_type is None:
            return
        scheduler_class = self.SCHEDULERS.get(scheduler_type)
        if scheduler_class is None:
            raise PipelineError(
                f"Unknown scheduler '{scheduler_type}' for {self.MODEL_TYPE}. "
                f"Available: {', '.join(self.SCHEDULERS)}"
            )
        logger.debug("Using scheduler: %s", scheduler_type)
        pipeline.scheduler = scheduler_class.from_config(pipeline.scheduler.config)

    def _run_inference(
        self,
        pipeline,
        prompt: str,
        negative_prompt: str,
        scheduler_type: Optional[str],
        seed: int,
        **kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run inference on the given pipeline."""
        self._set_scheduler(pipeline, scheduler_type)

        # Resolve the seed once so the generator and the metadata agree
        actual_seed = resolve_seed(seed)
        gen = self._create_generator(actual_seed)

        prediction = pipeline(
            prompt=prompt, negative_prompt=negative_prompt, generator=gen, **kwargs
        )
        image = np.array(prediction.images[0])

        # Report the real output size (inpaint pipelines infer it from the
        # input image rather than taking width/height arguments)
        height, width = image.shape[:2]
        metadata = format_metadata(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=kwargs.get("num_inference_steps", 10),
            width=width,
            height=height,
            cfg=kwargs.get("guidance_scale", 7),
            seed=actual_seed,
            scheduler=scheduler_type,
        )

        return image, metadata

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
        **extra_kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate an image from text prompts.

        Args:
            scheduler_type: Name from SCHEDULERS, or None to keep the
                pipeline's current scheduler
            seed: Generation seed; negative for random
            **extra_kwargs: Forwarded verbatim to the diffusers pipeline call

        Returns:
            Tuple of (HxWxC uint8 image array, metadata dict)

        Raises:
            PipelineError: If no pipeline is loaded or the scheduler is unknown
        """
        if self.pipeline is None:
            logger.error("Cannot generate image: no pipeline loaded")
            raise PipelineError("No model loaded. Please load a pipeline first.")

        return self._run_inference(
            self.pipeline,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            scheduler_type=scheduler_type,
            seed=seed,
            num_inference_steps=steps,
            width=width,
            height=height,
            guidance_scale=cfg,
            **extra_kwargs,
        )

    def forward_inpaint(
        self,
        positive_prompt: str,
        negative_prompt: str,
        image: Union[np.ndarray, Image.Image],
        mask_image: Union[np.ndarray, Image.Image],
        scheduler_type: Optional[str],
        steps: int,
        cfg: float,
        seed: int,
        **extra_kwargs,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Inpaint an image region.

        Args:
            image: Base image (PIL or numpy array)
            mask_image: Mask, white = repaint (converted to grayscale)
            **extra_kwargs: Forwarded verbatim to the diffusers pipeline call

        Raises:
            PipelineError: If no inpainting pipeline is loaded
        """
        if self.inpaint_pipeline is None:
            logger.error("Cannot inpaint: no inpainting pipeline loaded")
            raise PipelineError(
                "No inpainting model loaded. Please load an inpainting pipeline first."
            )

        # Ensure proper image formats
        pil_image = convert_to_pil(image)
        pil_mask = convert_mask_to_grayscale(convert_to_pil(mask_image))

        result, metadata = self._run_inference(
            self.inpaint_pipeline,
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            scheduler_type=scheduler_type,
            seed=seed,
            image=pil_image,
            mask_image=pil_mask,
            num_inference_steps=steps,
            guidance_scale=cfg,
            **extra_kwargs,
        )

        metadata["inpainting"] = True
        return result, metadata

    def get_scheduler_config(self) -> Optional[Dict]:
        """Return the current scheduler configuration."""
        if self.inpaint_pipeline is not None:
            return self.inpaint_pipeline.scheduler.config
        elif self.pipeline is not None:
            return self.pipeline.scheduler.config
        return None

    def _load_single_file(self, pipeline_class, label: str):
        """Load a complete single-file checkpoint with the given pipeline class."""
        self._init_device()
        logger.info("Loading %s checkpoint: %s", label, self.model_checkpoint)
        return pipeline_class.from_single_file(
            self.model_checkpoint, torch_dtype=self.dtype, use_safetensors=True
        )

    @abstractmethod
    def load_pipeline(self) -> None:
        """Load the main generation pipeline."""
        pass

    @abstractmethod
    def load_inpaint_pipeline(self) -> None:
        """Load the inpainting pipeline."""
        pass

    def add_loras(
        self, loras: Optional[Iterable[Union[LoraSpec, Mapping[str, Any]]]] = None
    ) -> List[str]:
        """
        Load and fuse LoRA adapters into the loaded pipeline(s).

        Args:
            loras: LoraSpec instances or mappings accepted by
                LoraSpec.from_mapping(); None or empty is a no-op

        Returns:
            Adapter names fused by this call (already-active ones excluded)

        Raises:
            PipelineError: If no pipeline is loaded
        """
        specs = [
            lora if isinstance(lora, LoraSpec) else LoraSpec.from_mapping(lora)
            for lora in (loras or [])
        ]
        if not specs:
            return []
        if not self.pipelines:
            raise PipelineError("No pipeline loaded to add LoRAs to")

        fused: List[str] = []
        for spec in specs:
            adapter_name = spec.adapter_name
            logger.info("Loading LoRA: %s (strength: %.2f)", spec.path, spec.strength)

            for pipeline in self.pipelines:
                if adapter_name in pipeline.get_active_adapters():
                    logger.info(
                        "LoRA Loader: %s: adapter already loaded in the pipeline",
                        adapter_name,
                    )
                    continue
                pipeline.load_lora_weights(Path(spec.path), adapter_name=adapter_name)
                pipeline.fuse_lora(lora_scale=spec.strength, adapter_name=adapter_name)
                if adapter_name not in fused:
                    fused.append(adapter_name)

        return fused

    def to_device(self) -> None:
        """Move pipeline(s) to the configured compute device."""
        for pipeline in self.pipelines:
            pipeline.to(self.torch_device)
        self._placement = "device"

    def load_custom_vae(self, vae_checkpoint: Union[str, Path]) -> None:
        """
        Replace the VAE of every loaded pipeline with a custom one.

        The VAE is loaded in the generator's dtype and placed consistently
        with the pipeline (moved to the device, or re-hooked for CPU offload).

        Raises:
            PipelineError: If no pipeline is loaded
        """
        if not self.pipelines:
            raise PipelineError("No pipeline loaded to set custom VAE")

        custom_vae = load_custom_vae(vae_checkpoint, dtype=self.dtype)
        if self._placement == "device":
            custom_vae = custom_vae.to(self.torch_device)

        for pipeline in self.pipelines:
            pipeline.vae = custom_vae
            if self._placement == "offload":
                # New modules carry no offload hooks; reinstall them
                pipeline.enable_model_cpu_offload()
