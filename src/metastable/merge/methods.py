#!/usr/bin/env python
"""
Tensor Merge Methods

Every method follows the same alpha convention: alpha is the weight of the
*target* model. alpha=0 keeps the base model, alpha=1 moves fully towards
(LINEAR/SLERP) or applies the full difference of (ADDITIVE/SUBTRACT) the
target model.
"""

import logging
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

from metastable.exceptions import MergeError

logger = logging.getLogger(__name__)


class MergeMethod(Enum):
    """
    Enumeration of available model merge methods.

    LINEAR:   (1 - alpha) * base + alpha * target
    SLERP:    Spherical linear interpolation from base (alpha=0) to target
              (alpha=1); better preserves weight magnitudes
    ADDITIVE: base + alpha * (target - reference)
    SUBTRACT: base - alpha * (target - reference)

    For ADDITIVE/SUBTRACT the reference defaults to the base itself.
    """
    LINEAR = "linear"
    SLERP = "slerp"
    ADDITIVE = "additive"
    SUBTRACT = "subtract"

    @classmethod
    def from_string(cls, value: "str | MergeMethod") -> "MergeMethod":
        """Create MergeMethod from a (case-insensitive) string value."""
        if isinstance(value, cls):
            return value
        for method in cls:
            if method.value == str(value).strip().lower():
                return method
        raise MergeError(
            f"Unknown merge method: {value}. "
            f"Available: {', '.join(m.value for m in cls)}"
        )


def _check_shapes(t0: torch.Tensor, t1: torch.Tensor) -> None:
    if t0.shape != t1.shape:
        raise MergeError(
            f"Tensors must have the same shape, got {tuple(t0.shape)} vs {tuple(t1.shape)}"
        )


def slerp(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Spherical linear interpolation between two tensors.

    SLERP interpolates along the surface of a hypersphere, which can
    preserve the magnitude and structure of neural network weights better
    than linear interpolation for some merge operations.

    Args:
        t0: Base tensor (returned at alpha=0)
        t1: Target tensor (returned at alpha=1)
        alpha: Weight of the target tensor

    Returns:
        Interpolated tensor with same shape as inputs

    Raises:
        MergeError: If tensors have different shapes
    """
    _check_shapes(t0, t1)
    original_shape = t0.shape

    # Flatten tensors for calculation
    t0_flat = t0.flatten()
    t1_flat = t1.flatten()

    # Normalize vectors (L2 norm)
    t0_norm = F.normalize(t0_flat, dim=0)
    t1_norm = F.normalize(t1_flat, dim=0)

    # Calculate angle between vectors
    dot_product = torch.clamp(torch.dot(t0_norm, t1_norm), -1.0, 1.0)
    omega = torch.acos(dot_product)

    # Handle (anti)parallel vectors, where sin(omega) ~ 0 makes the
    # spherical formula unstable: fall back to linear interpolation
    sin_omega = torch.sin(omega)
    if sin_omega.abs() < 1e-6:
        return linear(t0, t1, alpha)

    result = (
        (torch.sin((1 - alpha) * omega) / sin_omega) * t0_flat +
        (torch.sin(alpha * omega) / sin_omega) * t1_flat
    )

    return result.reshape(original_shape)


def linear(t0: torch.Tensor, t1: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Linear interpolation between two tensors.

    Args:
        t0: Base tensor (returned at alpha=0)
        t1: Target tensor (returned at alpha=1)
        alpha: Weight of the target tensor

    Returns:
        (1 - alpha) * t0 + alpha * t1

    Raises:
        MergeError: If tensors have different shapes
    """
    _check_shapes(t0, t1)
    return (t0 * (1 - alpha)) + (t1 * alpha)


def merge_tensors(
    tensor_a: torch.Tensor,
    tensor_b: torch.Tensor,
    method: MergeMethod,
    alpha: float,
    base_tensor: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Merge two tensors using the specified method.

    Computation happens in float32; the result is cast back to the dtype
    of tensor_a.

    Args:
        tensor_a: Base tensor
        tensor_b: Target tensor
        method: Merge method to use
        alpha: Weight of the target tensor (see module docstring)
        base_tensor: Reference tensor for ADDITIVE/SUBTRACT (defaults to
            tensor_a)

    Returns:
        Merged tensor with original dtype

    Raises:
        MergeError: If an unsupported merge method is specified or shapes differ
    """
    _check_shapes(tensor_a, tensor_b)

    original_dtype = tensor_a.dtype
    tensor_a = tensor_a.to(torch.float32)
    tensor_b = tensor_b.to(torch.float32)

    if method == MergeMethod.LINEAR:
        result = linear(tensor_a, tensor_b, alpha)

    elif method == MergeMethod.SLERP:
        result = slerp(tensor_a, tensor_b, alpha)

    elif method in (MergeMethod.ADDITIVE, MergeMethod.SUBTRACT):
        reference = tensor_a if base_tensor is None else base_tensor.to(torch.float32)
        _check_shapes(tensor_a, reference)
        diff = tensor_b - reference
        if method == MergeMethod.ADDITIVE:
            result = tensor_a + diff * alpha
        else:
            result = tensor_a - diff * alpha

    else:
        raise MergeError(f"Unsupported merge method: {method}")

    return result.to(original_dtype)
