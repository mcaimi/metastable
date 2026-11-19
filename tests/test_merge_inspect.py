"""Header-level checkpoint inspection and compatibility checks."""

import pytest
import torch
from safetensors.torch import save_file

from metastable.exceptions import CheckpointLoadError, MergeError
from metastable.merge.inspect import check_compatible, inspect_checkpoint
from metastable.merge.pipeline import SDXL_ARCHITECTURE

@pytest.fixture
def ckpt(tmp_path):
    def _make(name, tensors, metadata=None):
        path = tmp_path / f"{name}.safetensors"
        save_file(tensors, str(path), metadata=metadata)
        return path
    return _make


# generic files ----------------------------------------------------------------------------------

def test_generic_checkpoint_merges_every_key(ckpt, state_dict_factory):
    sd = state_dict_factory(0)
    info = inspect_checkpoint(ckpt("sd", sd, metadata={"title": "x"}))
    assert set(info.merge_keys) == set(sd)
    assert info.copy_keys == {} and info.prefix == "" and info.layout is None
    assert info.metadata == {"title": "x"}
    entry = info.merge_keys["model.conv.weight"]
    assert (entry.file_key, entry.dtype, entry.shape) == ("model.conv.weight", "F32", (4, 3, 3, 3))


def test_sd_architecture_keeps_diffusion_model_prefix(ckpt):
    tensors = {"model.diffusion_model.out.weight": torch.zeros(2), "first_stage_model.w": torch.zeros(2)}
    info = inspect_checkpoint(ckpt("sdxl", tensors), SDXL_ARCHITECTURE)
    assert set(info.merge_keys) == set(tensors) and info.copy_keys == {}


@pytest.mark.parametrize("name,content", [
    ("missing.safetensors", None),
    ("model.ckpt", b"x"),
    ("bad.safetensors", b"\x01"),
])
def test_unreadable_files_raise_checkpoint_load_error(tmp_path, name, content):
    path = tmp_path / name
    if content is not None:
        path.write_bytes(content)
    with pytest.raises(CheckpointLoadError):
        inspect_checkpoint(path)


# fp8 --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_weights_rejected_for_sd(ckpt, state_dict_factory, dtype):
    sd = state_dict_factory(0)
    sd["model.linear.weight"] = sd["model.linear.weight"].to(dtype)
    with pytest.raises(MergeError, match="fp8"):
        inspect_checkpoint(ckpt("sd", sd), SDXL_ARCHITECTURE)
    with pytest.raises(MergeError, match="fp8"):
        inspect_checkpoint(ckpt("sd", sd))


# keys and shapes --------------------------------------------------------------------------------

def test_key_and_shape_mismatches_reported(ckpt, state_dict_factory):
    base = inspect_checkpoint(ckpt("base", state_dict_factory(0)))
    other_sd = state_dict_factory(1)
    other_sd.pop("model.conv.bias")
    other_sd["model.extra"] = torch.zeros(1)
    other_sd["model.linear.weight"] = torch.zeros(2, 2)
    other = inspect_checkpoint(ckpt("other", other_sd))
    with pytest.raises(MergeError, match="validation failed") as exc:
        check_compatible(base, other)
    message = str(exc.value)
    assert "extra keys: ['model.extra']" in message
    assert "missing keys: ['model.conv.bias']" in message
    assert "Shape mismatch for key 'model.linear.weight': (8, 4) vs (2, 2)" in message


def test_compatible_checkpoints_pass(ckpt, state_dict_factory):
    a = inspect_checkpoint(ckpt("a", state_dict_factory(0)))
    b = inspect_checkpoint(ckpt("b", state_dict_factory(1)))
    check_compatible(a, b, b)
