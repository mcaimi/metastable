#!/usr/bin/env python
"""
Merge Operations for Stable Diffusion Checkpoints

Provides low-level functions for:
- Loading checkpoints
- Validating checkpoint compatibility
- Merging checkpoint dictionaries
- Saving merged results
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from safetensors.torch import load_file, save_file

from metastable.exceptions import CheckpointLoadError, MergeError
from metastable.merge.methods import MergeMethod, merge_tensors

logger = logging.getLogger(__name__)

StateDict = Dict[str, torch.Tensor]


@dataclass
class MergeConfig:
    """
    Configuration for merge operations.

    Attributes:
        method: Merge method
        alpha: Weight of the target model (see metastable.merge.methods)
        device: Device checkpoints are loaded to
        preserve_metadata: Write a JSON sidecar with merge provenance
        chunk_size: Merge this many tensors at a time (None = all at once)
        progress_callback: Called with progress in [0, 1]
    """
    method: MergeMethod = MergeMethod.LINEAR
    alpha: float = 0.5
    device: str = "cpu"
    preserve_metadata: bool = True
    chunk_size: Optional[int] = None
    progress_callback: Optional[Callable[[float], None]] = None

    def __post_init__(self) -> None:
        self.method = MergeMethod.from_string(self.method)
        try:
            self.alpha = float(self.alpha)
        except (TypeError, ValueError):
            raise MergeError(f"alpha must be a number, got {self.alpha!r}") from None
        if self.chunk_size is not None and self.chunk_size < 1:
            raise MergeError(f"chunk_size must be >= 1, got {self.chunk_size}")


def load_checkpoint_dict(
    checkpoint_file: Union[str, Path],
    device: str = "cpu",
) -> Tuple[StateDict, Dict]:
    """
    Load a checkpoint file and return weights and metadata.

    Metadata comes from a JSON sidecar with the same stem, if present.

    Args:
        checkpoint_file: Path to the safetensors checkpoint
        device: Device to load tensors to

    Returns:
        Tuple of (weights_dict, metadata_dict)

    Raises:
        CheckpointLoadError: If the checkpoint cannot be loaded
    """
    checkpoint_path = Path(checkpoint_file)

    if not checkpoint_path.exists():
        raise CheckpointLoadError(f"Checkpoint file not found: {checkpoint_path}")

    if checkpoint_path.suffix != ".safetensors":
        raise CheckpointLoadError(
            f"Only safetensors format supported, got: {checkpoint_path.suffix}"
        )

    try:
        logger.info("Loading checkpoint: %s", checkpoint_path)
        weights = load_file(checkpoint_path, device=device)

        metadata = {}
        metadata_file = checkpoint_path.with_suffix(".json")
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        return weights, metadata

    except Exception as e:
        logger.error("Failed to load checkpoint %s: %s", checkpoint_path, e)
        raise CheckpointLoadError(
            f"Failed to load checkpoint {checkpoint_path}: {e}"
        ) from e


def validate_checkpoints(
    base_ckpt: StateDict,
    *other_ckpts: StateDict,
) -> Tuple[bool, List[str]]:
    """
    Validate that checkpoints are compatible for merging.

    Args:
        base_ckpt: Base checkpoint dictionary
        *other_ckpts: Additional checkpoints to validate against base

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    base_keys = set(base_ckpt.keys())

    for i, ckpt in enumerate(other_ckpts):
        ckpt_keys = set(ckpt.keys())

        extra_keys = ckpt_keys - base_keys
        missing_keys = base_keys - ckpt_keys

        if extra_keys:
            errors.append(f"Checkpoint {i + 1} has extra keys: {sorted(extra_keys)[:5]}...")
        if missing_keys:
            errors.append(f"Checkpoint {i + 1} missing keys: {sorted(missing_keys)[:5]}...")

        for key in sorted(base_keys & ckpt_keys):
            if base_ckpt[key].shape != ckpt[key].shape:
                errors.append(
                    f"Shape mismatch for key '{key}': "
                    f"{tuple(base_ckpt[key].shape)} vs {tuple(ckpt[key].shape)}"
                )

    for error in errors:
        logger.warning(error)

    is_valid = len(errors) == 0
    if is_valid:
        logger.debug("Checkpoint validation passed")
    else:
        logger.error("Checkpoint validation failed with %d errors", len(errors))

    return is_valid, errors


def merge_checkpoints(
    base_ckpt: StateDict,
    target_ckpt: StateDict,
    config: MergeConfig,
    base_reference: Optional[StateDict] = None,
) -> StateDict:
    """
    Merge two checkpoints using the specified configuration.

    Honors config.chunk_size (releasing accelerator memory between chunks)
    and config.progress_callback.

    Args:
        base_ckpt: Base checkpoint dictionary
        target_ckpt: Target checkpoint dictionary
        config: Merge configuration
        base_reference: Optional reference for additive/subtractive merges

    Returns:
        Merged checkpoint dictionary

    Raises:
        MergeError: If checkpoints are incompatible or merge fails
    """
    references = (base_reference,) if base_reference is not None else ()
    is_valid, errors = validate_checkpoints(base_ckpt, target_ckpt, *references)
    if not is_valid:
        raise MergeError(f"Checkpoint validation failed: {errors}")

    use_reference = (
        base_reference is not None
        and config.method in (MergeMethod.ADDITIVE, MergeMethod.SUBTRACT)
    )

    keys = list(base_ckpt.keys())
    total_keys = len(keys)
    chunk_size = config.chunk_size or max(total_keys, 1)
    merged_dict: StateDict = {}

    for i, key in enumerate(keys):
        if config.progress_callback:
            config.progress_callback(i / total_keys)

        merged_dict[key] = merge_tensors(
            base_ckpt[key],
            target_ckpt[key],
            config.method,
            config.alpha,
            base_tensor=base_reference[key] if use_reference else None,
        )

        if config.chunk_size and (i + 1) % chunk_size == 0:
            logger.debug("Merged chunk %d/%d", (i + 1) // chunk_size,
                         (total_keys + chunk_size - 1) // chunk_size)
            _empty_accelerator_cache()

    if config.progress_callback:
        config.progress_callback(1.0)

    return merged_dict


def _empty_accelerator_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_checkpoint(
    weights: StateDict,
    filename: Union[str, Path],
    metadata: Optional[Dict] = None,
    backup_existing: bool = True,
) -> bool:
    """
    Save a checkpoint dictionary to disk.

    Args:
        weights: Checkpoint weights dictionary
        filename: Output path
        metadata: Optional metadata, written as a JSON sidecar
        backup_existing: Rename an existing output file instead of overwriting

    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filename)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if backup_existing and filepath.exists():
            backup_path = filepath.with_suffix(
                f".backup_{time.time_ns()}.safetensors"
            )
            filepath.rename(backup_path)
            logger.info("Created backup: %s", backup_path)

        logger.info("Saving checkpoint: %s", filepath)
        # safetensors refuses non-contiguous tensors
        save_file({k: v.contiguous() for k, v in weights.items()}, filepath)

        if metadata:
            metadata_file = filepath.with_suffix(".json")
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.debug("Saved metadata: %s", metadata_file)

        return True

    except Exception as e:
        logger.error("Failed to save checkpoint: %s", e)
        return False


def batch_merge_checkpoints(
    base_checkpoint: Union[str, Path],
    checkpoint_list: List[Union[str, Path]],
    output_dir: Union[str, Path],
    config: MergeConfig,
) -> List[str]:
    """
    Merge a base checkpoint with multiple other checkpoints.

    Failures on individual targets are logged and skipped.

    Args:
        base_checkpoint: Path to base checkpoint
        checkpoint_list: List of checkpoint paths to merge with base
        output_dir: Directory for output files (merged_<target stem>.safetensors)
        config: Merge configuration

    Returns:
        List of successfully created output file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_weights, base_metadata = load_checkpoint_dict(base_checkpoint, config.device)

    results = []

    for i, target_checkpoint in enumerate(checkpoint_list):
        try:
            logger.info("Merging %d/%d: %s", i + 1, len(checkpoint_list), target_checkpoint)

            target_weights, target_metadata = load_checkpoint_dict(
                target_checkpoint, config.device
            )

            merged_weights = merge_checkpoints(base_weights, target_weights, config)

            target_path = Path(target_checkpoint)
            output_filename = output_path / f"merged_{target_path.stem}.safetensors"

            merged_metadata = None
            if config.preserve_metadata:
                merged_metadata = {
                    "base_model": str(base_checkpoint),
                    "target_model": str(target_checkpoint),
                    "merge_method": config.method.value,
                    "merge_alpha": config.alpha,
                    "timestamp": time.time(),
                    "base_metadata": base_metadata,
                    "target_metadata": target_metadata,
                }

            if save_checkpoint(merged_weights, output_filename, merged_metadata):
                results.append(str(output_filename))
                logger.info("Successfully saved: %s", output_filename)
            else:
                logger.error("Failed to save: %s", output_filename)

        except Exception as e:
            logger.error("Error merging %s: %s", target_checkpoint, e)

    return results
