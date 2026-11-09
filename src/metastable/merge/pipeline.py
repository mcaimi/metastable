#!/usr/bin/env python
"""
Model Merging Pipeline for Stable Diffusion

Supports SD1.5, SDXL and Illustrious XL checkpoints with:
    - Simple 2-model merging
    - Multi-step recipe merging
    - Checkpoint caching across recipe steps
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging
import time

import torch

from metastable.device import get_device_settings, torch_device_string
from metastable.exceptions import MergeError
from metastable.merge.methods import MergeMethod
from metastable.merge.ops import (
    MergeConfig,
    load_checkpoint_dict,
    merge_checkpoints,
    save_checkpoint,
)
from metastable.pipelines.illustrious import IllustriousPipelineGenerator
from metastable.pipelines.sd15 import SD15PipelineGenerator
from metastable.pipelines.sdxl import SDXLPipelineGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelArchitectureInfo:
    """Information about a model architecture."""

    name: str
    base_resolution: int
    supported_resolutions: List[str] = field(default_factory=list)


SD15_ARCHITECTURE = ModelArchitectureInfo(
    name="sd15",
    base_resolution=512,
    supported_resolutions=list(SD15PipelineGenerator.get_resolutions()),
)

SDXL_ARCHITECTURE = ModelArchitectureInfo(
    name="sdxl",
    base_resolution=1024,
    supported_resolutions=list(SDXLPipelineGenerator.get_resolutions()),
)

ILLUSTRIOUS_ARCHITECTURE = ModelArchitectureInfo(
    name="illustrious",
    base_resolution=1024,
    supported_resolutions=list(IllustriousPipelineGenerator.get_resolutions()),
)

# Architectures that support key-by-key checkpoint merging. Flux is absent:
# its transformer-only checkpoints need a dedicated merge path.
ARCHITECTURES: Dict[str, ModelArchitectureInfo] = {
    arch.name: arch
    for arch in (SD15_ARCHITECTURE, SDXL_ARCHITECTURE, ILLUSTRIOUS_ARCHITECTURE)
}


def _resolve_merge_device(device: str) -> str:
    if device == "auto":
        return get_device_settings().torch_device
    if device == "rocm":
        return torch_device_string("rocm", get_device_settings().index)
    return device


class MergePipeline:
    """
    Merging pipeline for Stable Diffusion checkpoints of one architecture.

    Recipe steps cache loaded checkpoints; call clear_cache() when done.
    """

    def __init__(
        self,
        architecture: ModelArchitectureInfo,
        device: str = "auto",
    ):
        """
        Initialize the merge pipeline.

        Args:
            architecture: Model architecture information
            device: Device used to load checkpoints for recipes ("auto",
                "cpu", "cuda[:N]", "mps", "rocm"). "rocm" (AMD GPUs under
                PyTorch ROCm builds) is mapped to its torch-level device
                string, since both torch and safetensors address ROCm
                devices through the CUDA API ("cuda:N").
        """
        self.architecture = architecture
        self.device = _resolve_merge_device(device)
        self.loaded_checkpoints: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def for_model_type(cls, model_type: str, device: str = "auto") -> "MergePipeline":
        """
        Create a merge pipeline for a model type identifier.

        Raises:
            MergeError: If the model type does not support merging
        """
        try:
            architecture = ARCHITECTURES[model_type]
        except KeyError:
            raise MergeError(
                f"Merging is not supported for model type '{model_type}'. "
                f"Supported: {', '.join(ARCHITECTURES)}"
            ) from None
        return cls(architecture=architecture, device=device)

    def load_and_cache_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        cache_key: Optional[str] = None,
    ) -> str:
        """
        Load and cache a checkpoint for future merging operations.

        Args:
            checkpoint_path: Path to the checkpoint file
            cache_key: Optional custom key; defaults to the absolute path

        Returns:
            The cache key used to store the checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        cache_key = cache_key or str(checkpoint_path.absolute())

        if cache_key not in self.loaded_checkpoints:
            weights, metadata = load_checkpoint_dict(checkpoint_path, self.device)
            self.loaded_checkpoints[cache_key] = {
                "weights": weights,
                "metadata": metadata,
                "path": checkpoint_path,
            }
            logger.info("Cached checkpoint: %s", cache_key)

        return cache_key

    def create_merge_recipe(
        self,
        base_model: Union[str, Path],
        merge_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create a merge recipe for multi-step merging.

        Args:
            base_model: Path to the base model checkpoint
            merge_steps: Step dicts with keys "target_model" (required),
                "method" (default "linear"), "alpha" (default 0.5) and
                "show_progress" (default False)

        Returns:
            Recipe dictionary

        Raises:
            MergeError: If a step is malformed
        """
        for i, step in enumerate(merge_steps):
            if "target_model" not in step:
                raise MergeError(f"Recipe step {i} has no 'target_model'")
            MergeMethod.from_string(step.get("method", "linear"))

        return {
            "base_model": str(base_model),
            "steps": [dict(step) for step in merge_steps],
            "created_at": time.time(),
            "architecture": self.architecture.name,
        }

    def execute_merge_recipe(
        self,
        recipe: Dict[str, Any],
        output_path: Union[str, Path],
    ) -> bool:
        """
        Execute a merge recipe: each step merges the running result with the
        step's target. ADDITIVE/SUBTRACT steps use the original base model as
        the difference reference.

        Args:
            recipe: Recipe dictionary from create_merge_recipe
            output_path: Path to save the final merged model

        Returns:
            True if the result was saved

        Raises:
            MergeError: If any step fails
        """
        try:
            base_key = self.load_and_cache_checkpoint(recipe["base_model"])
            base_weights = self.loaded_checkpoints[base_key]["weights"]
            current_weights = dict(base_weights)

            recipe_metadata = {
                "recipe": recipe,
                "execution_timestamp": time.time(),
                "steps_executed": [],
                "architecture": self.architecture.name,
            }

            for i, step in enumerate(recipe["steps"]):
                logger.info("Executing merge step %d/%d", i + 1, len(recipe["steps"]))

                target_key = self.load_and_cache_checkpoint(step["target_model"])
                target_weights = self.loaded_checkpoints[target_key]["weights"]

                progress_callback = None
                if step.get("show_progress", False):
                    progress_callback = lambda p: logger.debug(
                        "Progress: %.1f%%", p * 100
                    )

                config = MergeConfig(
                    method=MergeMethod.from_string(step.get("method", "linear")),
                    alpha=step.get("alpha", 0.5),
                    device=self.device,
                    progress_callback=progress_callback,
                )

                base_reference = None
                if config.method in (MergeMethod.ADDITIVE, MergeMethod.SUBTRACT):
                    base_reference = base_weights

                current_weights = merge_checkpoints(
                    current_weights, target_weights, config, base_reference
                )

                recipe_metadata["steps_executed"].append(
                    {
                        "step_index": i,
                        "target_model": str(step["target_model"]),
                        "method": config.method.value,
                        "alpha": config.alpha,
                        "completed_at": time.time(),
                    }
                )

            if not save_checkpoint(current_weights, output_path, recipe_metadata):
                raise MergeError(f"Failed to save merged model to {output_path}")
            return True

        except MergeError:
            raise
        except Exception as e:
            logger.error("Failed to execute merge recipe: %s", e)
            raise MergeError(f"Recipe execution failed: {e}") from e

    def merge(
        self,
        base_model: Union[str, Path],
        target_model: Union[str, Path],
        config: MergeConfig,
        output_path: Union[str, Path],
    ) -> str:
        """
        Merge two checkpoints and save the result.

        Checkpoints are loaded to config.device.

        Args:
            base_model: Path to base model
            target_model: Path to target model
            config: Merge configuration
            output_path: Path to save merged model

        Returns:
            Path to the saved model

        Raises:
            MergeError: If loading, merging or saving fails
        """
        try:
            base_weights, base_metadata = load_checkpoint_dict(base_model, config.device)
            target_weights, target_metadata = load_checkpoint_dict(
                target_model, config.device
            )

            logger.info(
                "Merging %s with %s using %s",
                base_model,
                target_model,
                config.method.value,
            )
            merged_weights = merge_checkpoints(base_weights, target_weights, config)

            merged_metadata = None
            if config.preserve_metadata:
                merged_metadata = {
                    "merge_info": {
                        "base_model": str(base_model),
                        "target_model": str(target_model),
                        "method": config.method.value,
                        "alpha": config.alpha,
                        "timestamp": time.time(),
                    },
                    "model_info": {
                        "architecture": self.architecture.name,
                        "base_resolution": self.architecture.base_resolution,
                        "supported_resolutions": self.architecture.supported_resolutions,
                    },
                    "base_metadata": base_metadata,
                    "target_metadata": target_metadata,
                }

            if not save_checkpoint(merged_weights, output_path, merged_metadata):
                raise MergeError(f"Failed to save merged model to {output_path}")

            logger.info("Merged model saved: %s", output_path)
            return str(output_path)

        except MergeError:
            raise
        except Exception as e:
            logger.error("Merge failed: %s", e)
            raise MergeError(f"Model merge operation failed: {e}") from e

    def clear_cache(self) -> None:
        """Clear cached checkpoints to free memory."""
        self.loaded_checkpoints.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Checkpoint cache cleared")

    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached checkpoints."""
        return {
            "cached_models": list(self.loaded_checkpoints.keys()),
            "cache_size": len(self.loaded_checkpoints),
            "device": self.device,
            "architecture": self.architecture.name,
        }


class SD15MergePipeline(MergePipeline):
    """SD1.5 merge pipeline."""

    def __init__(self, device: str = "auto"):
        super().__init__(architecture=SD15_ARCHITECTURE, device=device)


class SDXLMergePipeline(MergePipeline):
    """SDXL merge pipeline."""

    def __init__(self, device: str = "auto"):
        super().__init__(architecture=SDXL_ARCHITECTURE, device=device)


class IllustriousMergePipeline(MergePipeline):
    """Illustrious XL merge pipeline."""

    def __init__(self, device: str = "auto"):
        super().__init__(architecture=ILLUSTRIOUS_ARCHITECTURE, device=device)
