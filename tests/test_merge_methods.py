import math

import pytest
import torch

from metastable.exceptions import MergeError
from metastable.merge.methods import MergeMethod, linear, merge_tensors, slerp


@pytest.fixture
def base():
    return torch.tensor([[1.0, 0.0], [2.0, -1.0]])


@pytest.fixture
def target():
    return torch.tensor([[0.0, 1.0], [-1.0, 3.0]])


# MergeMethod -----------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("linear", MergeMethod.LINEAR), ("SLERP", MergeMethod.SLERP),
    (" Additive ", MergeMethod.ADDITIVE), ("subtract", MergeMethod.SUBTRACT),
    (MergeMethod.SLERP, MergeMethod.SLERP),
])
def test_from_string(value, expected):
    assert MergeMethod.from_string(value) is expected


def test_from_string_unknown():
    with pytest.raises(MergeError, match="Unknown merge method"):
        MergeMethod.from_string("average")


# alpha convention: alpha is the weight of the target --------------------------------

@pytest.mark.parametrize("method", [MergeMethod.LINEAR, MergeMethod.SLERP])
def test_interpolation_endpoints(method, base, target):
    assert torch.allclose(merge_tensors(base, target, method, 0.0), base, atol=1e-6)
    assert torch.allclose(merge_tensors(base, target, method, 1.0), target, atol=1e-6)


@pytest.mark.parametrize("method", [MergeMethod.LINEAR, MergeMethod.SLERP])
def test_low_alpha_stays_closer_to_base(method, base, target):
    merged = merge_tensors(base, target, method, 0.2)
    assert torch.dist(merged, base) < torch.dist(merged, target)


def test_linear_formula(base, target):
    assert torch.allclose(linear(base, target, 0.25), 0.75 * base + 0.25 * target)


def test_linear_and_slerp_agree_on_direction(base, target):
    """Same alpha means the same thing for both interpolation methods."""
    for alpha in (0.1, 0.3, 0.7, 0.9):
        lin = linear(base, target, alpha)
        sph = slerp(base, target, alpha)
        assert (torch.dist(lin, base) < torch.dist(lin, target)) == (
            torch.dist(sph, base) < torch.dist(sph, target))


# slerp ------------------------------------------------------------------------------

def test_slerp_orthogonal_unit_vectors():
    a, b = torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])
    mid = slerp(a, b, 0.5)
    assert torch.allclose(mid, torch.tensor([math.sqrt(0.5), math.sqrt(0.5)]), atol=1e-6)
    assert torch.isclose(mid.norm(), torch.tensor(1.0))


def test_slerp_preserves_shape(base, target):
    assert slerp(base, target, 0.3).shape == base.shape


def test_slerp_parallel_falls_back_to_linear():
    a = torch.tensor([1.0, 2.0, 3.0])
    b = 2 * a
    assert torch.allclose(slerp(a, b, 0.5), linear(a, b, 0.5))


def test_slerp_antiparallel_is_finite():
    a = torch.tensor([1.0, 0.0])
    result = slerp(a, -a, 0.5)
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("fn", [linear, slerp])
def test_shape_mismatch(fn):
    with pytest.raises(MergeError, match="same shape"):
        fn(torch.zeros(2), torch.zeros(3), 0.5)


# additive / subtract ------------------------------------------------------------------

def test_additive_with_reference(base, target):
    ref = torch.full_like(base, 0.5)
    result = merge_tensors(base, target, MergeMethod.ADDITIVE, 0.5, base_tensor=ref)
    assert torch.allclose(result, base + 0.5 * (target - ref))


def test_subtract_with_reference(base, target):
    ref = torch.full_like(base, 0.5)
    result = merge_tensors(base, target, MergeMethod.SUBTRACT, 0.5, base_tensor=ref)
    assert torch.allclose(result, base - 0.5 * (target - ref))


def test_additive_without_reference_uses_base(base, target):
    result = merge_tensors(base, target, MergeMethod.ADDITIVE, 0.3)
    assert torch.allclose(result, base + 0.3 * (target - base))


def test_reference_shape_mismatch(base, target):
    with pytest.raises(MergeError):
        merge_tensors(base, target, MergeMethod.ADDITIVE, 0.5, base_tensor=torch.zeros(3))


# merge_tensors general -------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("method", list(MergeMethod))
def test_dtype_preserved(dtype, method, base, target):
    result = merge_tensors(base.to(dtype), target.to(dtype), method, 0.5)
    assert result.dtype == dtype


def test_computes_in_float32():
    """The intermediate difference (120000) overflows fp16 but the result fits."""
    a = torch.tensor([0.0], dtype=torch.float16)
    b = torch.tensor([60000.0], dtype=torch.float16)
    ref = torch.tensor([-60000.0], dtype=torch.float16)
    assert torch.isinf(b - ref).all()  # what fp16 arithmetic would produce
    result = merge_tensors(a, b, MergeMethod.ADDITIVE, 0.25, base_tensor=ref)
    assert result.dtype == torch.float16
    assert torch.allclose(result.float(), torch.tensor([30000.0]))


def test_unsupported_method(base, target):
    with pytest.raises(MergeError, match="Unsupported"):
        merge_tensors(base, target, "linear", 0.5)


def test_merge_tensors_shape_mismatch():
    with pytest.raises(MergeError):
        merge_tensors(torch.zeros(2), torch.zeros(3), MergeMethod.ADDITIVE, 0.5)
