#!/usr/bin/env python
"""
Model Merging Pipeline

Supports SD1.5, SDXL and Illustrious XL checkpoints with:
    - Simple 2-model merging
    - Multi-step recipe merging
    - Batch merging of one base with many targets

File-based merges stream tensors (see metastable.merge.stream), so memory
use stays constant regardless of model size.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time

import torch

from metastable.device import get_device_settings, torch_device_string
from metastable.exceptions import MergeError
from metastable.merge.files import read_sidecar, write_sidecar
from metastable.merge.inspect import check_compatible, inspect_checkpoint
from metastable.merge.methods import MergeMethod
from metastable.merge.ops import (
    MergeConfig,
    _parse_output_dtype,
    batch_merge_checkpoints,
    dtype_name,
    load_checkpoint_dict,
)
from metastable.merge.stream import MergeStep, stream_merge
from metastable.pipelines.illustrious import IllustriousPipelineGenerator
from metastable.pipelines.sd15 import SD15PipelineGenerator
from metastable.pipelines.sdxl import SDXLPipelineGenerator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelArchitectureInfo:
    """
    Information about a model architecture.

    Attributes:
        name: Model type identifier
        base_resolution: Native training resolution
        supported_resolutions: Resolution names from the generator
        layouts: Known key layouts as (name, marker keys); a checkpoint has
            a layout when all its marker keys are present (prefix
            stripped). Empty for architectures without layout checks.
        transformer_only: Checkpoints hold a diffusion transformer only; in
            all-in-one files (transformer under "model.diffusion_model."
            next to other components) only the transformer is merged
    """

    name: str
    base_resolution: int
    supported_resolutions: List[str] = field(default_factory=list)
    layouts: Tuple[Tuple[str, Tuple[str, ...]], ...] = ()
    transformer_only: bool = False


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

# Architectures that support key-by-key checkpoint merging
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
        output_dtype: Optional[Union[str, torch.dtype]] = None,
    ) -> bool:
        """
        Execute a merge recipe: each step merges the running result with the
        step's target. ADDITIVE/SUBTRACT steps use the original base model as
        the difference reference.

        All checkpoints are validated from their headers first; the steps are
        then applied tensor by tensor and streamed to output_path.

        Args:
            recipe: Recipe dictionary from create_merge_recipe
            output_path: Path to save the final merged model
            output_dtype: dtype of the merged tensors (see MergeConfig)

        Returns:
            True if the result was saved

        Raises:
            MergeError: If any step fails
        """
        try:
            base_info = inspect_checkpoint(recipe["base_model"], self.architecture)
            targets = [
                inspect_checkpoint(step["target_model"], self.architecture)
                for step in recipe["steps"]
            ]
            check_compatible(base_info, *targets)

            steps = []
            for step, target in zip(recipe["steps"], targets):
                config = MergeConfig(
                    method=step.get("method", "linear"), alpha=step.get("alpha", 0.5)
                )
                steps.append(MergeStep(target, config.method, config.alpha))

            progress_callback = None
            if any(step.get("show_progress", False) for step in recipe["steps"]):
                progress_callback = lambda p: logger.debug("Progress: %.1f%%", p * 100)

            logger.info("Executing %d-step merge recipe", len(steps))
            dtype = _parse_output_dtype(output_dtype)
            result = stream_merge(
                base_info,
                steps,
                output_path,
                reference=base_info,
                device=self.device,
                output_dtype=dtype,
                progress_callback=progress_callback,
            )

            completed_at = time.time()
            recipe_metadata = {
                "recipe": recipe,
                "execution_timestamp": completed_at,
                "steps_executed": [
                    {
                        "step_index": i,
                        "target_model": str(step["target_model"]),
                        "method": merge_step.method.value,
                        "alpha": merge_step.alpha,
                        "completed_at": completed_at,
                    }
                    for i, (step, merge_step) in enumerate(zip(recipe["steps"], steps))
                ],
                "architecture": self.architecture.name,
                "layout": base_info.layout,
                "output_dtype": dtype_name(dtype),
                "copied_keys": result.copied_keys,
            }
            write_sidecar(output_path, recipe_metadata)
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

        Both checkpoints are validated from their headers, then merged tensor
        by tensor on config.device and streamed to output_path.

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
            base_info = inspect_checkpoint(base_model, self.architecture)
            target_info = inspect_checkpoint(target_model, self.architecture)
            check_compatible(base_info, target_info)

            logger.info(
                "Merging %s with %s using %s",
                base_model,
                target_model,
                config.method.value,
            )
            result = stream_merge(
                base_info,
                [MergeStep(target_info, config.method, config.alpha)],
                output_path,
                device=config.device,
                output_dtype=config.output_dtype,
                progress_callback=config.progress_callback,
                chunk_size=config.chunk_size,
            )

            if config.preserve_metadata:
                write_sidecar(output_path, {
                    "merge_info": {
                        "base_model": str(base_model),
                        "target_model": str(target_model),
                        "method": config.method.value,
                        "alpha": config.alpha,
                        "output_dtype": dtype_name(config.output_dtype),
                        "copied_keys": result.copied_keys,
                        "timestamp": time.time(),
                    },
                    "model_info": {
                        "architecture": self.architecture.name,
                        "layout": base_info.layout,
                        "base_resolution": self.architecture.base_resolution,
                        "supported_resolutions": self.architecture.supported_resolutions,
                    },
                    "base_metadata": read_sidecar(base_model),
                    "target_metadata": read_sidecar(target_model),
                })

            logger.info("Merged model saved: %s", output_path)
            return str(output_path)

        except MergeError:
            raise
        except Exception as e:
            logger.error("Merge failed: %s", e)
            raise MergeError(f"Model merge operation failed: {e}") from e

    def batch_merge(
        self,
        base_model: Union[str, Path],
        target_models: List[Union[str, Path]],
        output_dir: Union[str, Path],
        config: MergeConfig,
    ) -> List[str]:
        """
        Merge a base checkpoint with each target (see batch_merge_checkpoints).

        Returns:
            Paths of the merged files that were created
        """
        return batch_merge_checkpoints(
            base_model, target_models, output_dir, config, architecture=self.architecture
        )

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

