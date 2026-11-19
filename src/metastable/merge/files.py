#!/usr/bin/env python
"""
Checkpoint File Helpers

Small helpers shared by the in-memory and streaming merge code: JSON
sidecars, output backups, key/shape comparison and accelerator cache
release.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

logger = logging.getLogger(__name__)

Shapes = Mapping[str, Tuple[int, ...]]


def read_sidecar(checkpoint_file: Union[str, Path]) -> Dict[str, Any]:
    """Return the JSON sidecar (same stem, .json) of a checkpoint, or {}."""
    metadata_file = Path(checkpoint_file).with_suffix(".json")
    if not metadata_file.exists():
        return {}
    with open(metadata_file, "r") as f:
        return json.load(f)


def write_sidecar(checkpoint_file: Union[str, Path], metadata: Dict[str, Any]) -> None:
    """Write metadata as the JSON sidecar of a checkpoint."""
    metadata_file = Path(checkpoint_file).with_suffix(".json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.debug("Saved metadata: %s", metadata_file)


def backup_existing(filepath: Union[str, Path]) -> Optional[Path]:
    """Rename an existing file to a timestamped backup; return its new path."""
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    backup_path = filepath.with_suffix(f".backup_{time.time_ns()}.safetensors")
    filepath.rename(backup_path)
    logger.info("Created backup: %s", backup_path)
    return backup_path


def compare_key_shapes(base: Shapes, others: Sequence[Shapes]) -> List[str]:
    """
    Compare key sets and tensor shapes of checkpoints against a base.

    Returns:
        Human-readable error messages; empty if all checkpoints match
    """
    errors = []
    base_keys = set(base)

    for i, other in enumerate(others):
        other_keys = set(other)

        extra_keys = other_keys - base_keys
        missing_keys = base_keys - other_keys

        if extra_keys:
            errors.append(f"Checkpoint {i + 1} has extra keys: {sorted(extra_keys)[:5]}...")
        if missing_keys:
            errors.append(f"Checkpoint {i + 1} missing keys: {sorted(missing_keys)[:5]}...")

        for key in sorted(base_keys & other_keys):
            if tuple(base[key]) != tuple(other[key]):
                errors.append(
                    f"Shape mismatch for key '{key}': "
                    f"{tuple(base[key])} vs {tuple(other[key])}"
                )

    return errors


def empty_accelerator_cache() -> None:
    """Release cached accelerator memory (CUDA/ROCm)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
