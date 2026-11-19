#!/usr/bin/env python
"""
Checkpoint Inspection

Describes a safetensors checkpoint from its header alone (no tensor data is
read) and checks that checkpoints can be merged:

- which keys are merged and which are copied unchanged (all-in-one files
  bundling a transformer with a VAE and text encoders),
- the key prefix and key layout (original/ComfyUI vs diffusers),
- fp8 weights, which are rejected because merging them loses precision,
- matching key sets, shapes and layouts across checkpoints.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from metastable.exceptions import CheckpointLoadError, MergeError
from metastable.merge.files import compare_key_shapes
from metastable.models import read_safetensors_header

if TYPE_CHECKING:  # pragma: no cover
    from metastable.merge.pipeline import ModelArchitectureInfo

logger = logging.getLogger(__name__)

# Prefix of the diffusion model weights in single-file checkpoints
TRANSFORMER_PREFIX = "model.diffusion_model."

_FP8_DTYPE_PREFIX = "F8_"
_SCALED_FP8_MARKERS = (".scale_weight", ".scale_input")


@dataclass(frozen=True)
class TensorEntry:
    """A tensor as described by a safetensors header."""

    file_key: str
    dtype: str  # safetensors dtype name, e.g. "BF16"
    shape: Tuple[int, ...]


@dataclass(frozen=True)
class CheckpointInfo:
    """
    Header-level description of a checkpoint.

    Attributes:
        path: Checkpoint file
        merge_keys: Normalized key (prefix stripped) -> tensor to merge,
            in file order
        copy_keys: File key -> tensor copied unchanged (components of an
            all-in-one file other than the transformer)
        prefix: Prefix stripped from merge keys ("" if none)
        layout: Detected key layout name, or None for architectures that
            don't define layouts
        metadata: The header's "__metadata__" entry
    """

    path: Path
    merge_keys: Dict[str, TensorEntry]
    copy_keys: Dict[str, TensorEntry] = field(default_factory=dict)
    prefix: str = ""
    layout: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def shapes(self) -> Dict[str, Tuple[int, ...]]:
        """Normalized merge key -> shape."""
        return {key: entry.shape for key, entry in self.merge_keys.items()}


def _read_header(path: Path) -> dict:
    if not path.exists():
        raise CheckpointLoadError(f"Checkpoint file not found: {path}")
    if path.suffix != ".safetensors":
        raise CheckpointLoadError(f"Only safetensors format supported, got: {path.suffix}")
    try:
        return read_safetensors_header(path)
    except ValueError as e:
        raise CheckpointLoadError(f"Failed to load checkpoint {path}: {e}") from e


def _detect_layout(merge_keys: Dict[str, TensorEntry], architecture) -> Optional[str]:
    for name, markers in architecture.layouts:
        if all(marker in merge_keys for marker in markers):
            return name
    return None


def _check_not_fp8(path: Path, merge_keys: Dict[str, TensorEntry]) -> None:
    for key, entry in merge_keys.items():
        if entry.dtype.startswith(_FP8_DTYPE_PREFIX):
            raise MergeError(
                f"{path.name}: tensor '{entry.file_key}' is {entry.dtype}; fp8 "
                "checkpoints can't be merged without losing precision. Use a "
                "bf16/fp16/fp32 version of the model."
            )
        if key == "scaled_fp8" or key.endswith(_SCALED_FP8_MARKERS):
            raise MergeError(
                f"{path.name}: contains scaled-fp8 tensors ('{entry.file_key}'); "
                "fp8 checkpoints can't be merged without losing precision. Use a "
                "bf16/fp16/fp32 version of the model."
            )


def inspect_checkpoint(
    checkpoint_file: Union[str, Path],
    architecture: Optional["ModelArchitectureInfo"] = None,
) -> CheckpointInfo:
    """
    Describe a checkpoint from its safetensors header.

    Without an architecture, or for architectures that are not
    transformer-only, every tensor is merged under its own name. For
    transformer-only architectures, tensors under "model.diffusion_model."
    are merged (matched with the prefix stripped) and any other tensor (VAE,
    text encoders of an all-in-one file) is copied unchanged; the key layout
    must match one of the architecture's layouts.

    Raises:
        CheckpointLoadError: If the file is missing or not a safetensors file
        MergeError: If merged tensors are fp8, or the file doesn't match the
            architecture
    """
    path = Path(checkpoint_file)
    header = _read_header(path)
    metadata = dict(header.pop("__metadata__", None) or {})
    entries = {
        key: TensorEntry(key, value["dtype"], tuple(value["shape"]))
        for key, value in header.items()
    }

    transformer_only = getattr(architecture, "transformer_only", False)
    prefix = ""
    copy_keys: Dict[str, TensorEntry] = {}
    if transformer_only and any(key.startswith(TRANSFORMER_PREFIX) for key in entries):
        prefix = TRANSFORMER_PREFIX
        merge_keys = {
            key[len(prefix):]: entry
            for key, entry in entries.items()
            if key.startswith(prefix)
        }
        copy_keys = {key: entry for key, entry in entries.items() if not key.startswith(prefix)}
    else:
        merge_keys = entries

    layout = None
    if architecture is not None and architecture.layouts:
        layout = _detect_layout(merge_keys, architecture)
        if layout is None:
            raise MergeError(
                f"{path.name} does not look like a {architecture.name} checkpoint "
                f"(expected one of the layouts: "
                f"{', '.join(name for name, _ in architecture.layouts)})"
            )

    _check_not_fp8(path, merge_keys)

    if copy_keys:
        logger.info(
            "%s: all-in-one checkpoint, %d transformer tensors merged, %d copied",
            path.name, len(merge_keys), len(copy_keys),
        )
    return CheckpointInfo(
        path=path,
        merge_keys=merge_keys,
        copy_keys=copy_keys,
        prefix=prefix,
        layout=layout,
        metadata=metadata,
    )


def check_compatible(base: CheckpointInfo, *others: CheckpointInfo) -> None:
    """
    Check that checkpoints can be merged with the base.

    Raises:
        MergeError: If key layouts differ, or merge keys or shapes don't match
    """
    for other in others:
        if other.layout != base.layout:
            raise MergeError(
                f"Checkpoint layouts differ: {base.path.name} uses the "
                f"{base.layout} layout, {other.path.name} uses the {other.layout} "
                "layout. Convert one of them so both use the same layout."
            )

    errors = compare_key_shapes(base.shapes(), [other.shapes() for other in others])
    if errors:
        for error in errors:
            logger.warning(error)
        logger.error("Checkpoint validation failed with %d errors", len(errors))
        raise MergeError(f"Checkpoint validation failed: {errors}")
