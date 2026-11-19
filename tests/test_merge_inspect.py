"""Header-level checkpoint inspection and compatibility checks."""

import pytest
import torch
from safetensors.torch import save_file

from metastable.exceptions import CheckpointLoadError, MergeError
from metastable.merge.inspect import (
    TRANSFORMER_PREFIX,
    check_compatible,
    inspect_checkpoint,
)
from metastable.merge.pipeline import FLUX_ARCHITECTURE, SDXL_ARCHITECTURE, ZIMAGE_ARCHITECTURE

ZIMAGE_DIFFUSERS = {
    "layers.0.attention.to_q.weight": torch.zeros(4, 4),
    "all_x_embedder.2-1.weight": torch.zeros(4, 2),
}
ZIMAGE_ORIGINAL = {
    "layers.0.attention.qkv.weight": torch.zeros(12, 4),
    "x_embedder.weight": torch.zeros(4, 2),
}
FLUX_ORIGINAL = {
    "double_blocks.0.img_attn.qkv.weight": torch.zeros(6, 2),
    "single_blocks.0.linear1.weight": torch.zeros(8, 2),
}
FLUX_DIFFUSERS = {
    "transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 2),
    "single_transformer_blocks.0.attn.to_q.weight": torch.zeros(2, 2),
}
BUNDLED_EXTRAS = {
    "vae.decoder.conv_in.weight": torch.zeros(2, 2),
    "text_encoders.t5xxl.weight": torch.zeros(3).to(torch.float8_e4m3fn),
}


def _prefixed(tensors):
    return {TRANSFORMER_PREFIX + k: v for k, v in tensors.items()}


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


# layouts ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("arch,tensors,layout", [
    (ZIMAGE_ARCHITECTURE, ZIMAGE_DIFFUSERS, "diffusers"),
    (ZIMAGE_ARCHITECTURE, ZIMAGE_ORIGINAL, "original"),
    (FLUX_ARCHITECTURE, FLUX_DIFFUSERS, "diffusers"),
    (FLUX_ARCHITECTURE, FLUX_ORIGINAL, "original"),
])
def test_layout_detection(ckpt, arch, tensors, layout):
    assert inspect_checkpoint(ckpt("plain", tensors), arch).layout == layout
    assert inspect_checkpoint(ckpt("prefixed", _prefixed(tensors)), arch).layout == layout


def test_wrong_architecture_rejected(ckpt, state_dict_factory):
    with pytest.raises(MergeError, match="does not look like a zimage checkpoint"):
        inspect_checkpoint(ckpt("sd", state_dict_factory(0)), ZIMAGE_ARCHITECTURE)
    with pytest.raises(MergeError, match="does not look like a flux checkpoint"):
        inspect_checkpoint(ckpt("z", ZIMAGE_DIFFUSERS), FLUX_ARCHITECTURE)


def test_layout_mismatch_rejected(ckpt):
    a = inspect_checkpoint(ckpt("a", ZIMAGE_DIFFUSERS), ZIMAGE_ARCHITECTURE)
    b = inspect_checkpoint(ckpt("b", ZIMAGE_ORIGINAL), ZIMAGE_ARCHITECTURE)
    with pytest.raises(MergeError, match="a.safetensors uses the diffusers layout, b.safetensors uses the original"):
        check_compatible(a, b)


# prefixes and all-in-one files ------------------------------------------------------------------

def test_prefix_is_stripped_for_matching(ckpt):
    plain = inspect_checkpoint(ckpt("plain", ZIMAGE_DIFFUSERS), ZIMAGE_ARCHITECTURE)
    prefixed = inspect_checkpoint(ckpt("prefixed", _prefixed(ZIMAGE_DIFFUSERS)), ZIMAGE_ARCHITECTURE)
    assert prefixed.prefix == TRANSFORMER_PREFIX and plain.prefix == ""
    assert set(prefixed.merge_keys) == set(plain.merge_keys)
    key = "all_x_embedder.2-1.weight"
    assert prefixed.merge_keys[key].file_key == TRANSFORMER_PREFIX + key
    check_compatible(plain, prefixed)
    check_compatible(prefixed, plain)


def test_all_in_one_checkpoint_split(ckpt):
    info = inspect_checkpoint(ckpt("aio", {**_prefixed(ZIMAGE_DIFFUSERS), **BUNDLED_EXTRAS}),
                              ZIMAGE_ARCHITECTURE)
    assert set(info.merge_keys) == set(ZIMAGE_DIFFUSERS)
    assert set(info.copy_keys) == set(BUNDLED_EXTRAS)
    assert info.copy_keys["text_encoders.t5xxl.weight"].dtype == "F8_E4M3"  # copied, so allowed


def test_all_in_one_matches_transformer_only_file(ckpt):
    aio = inspect_checkpoint(ckpt("aio", {**_prefixed(FLUX_ORIGINAL), **BUNDLED_EXTRAS}), FLUX_ARCHITECTURE)
    plain = inspect_checkpoint(ckpt("plain", FLUX_ORIGINAL), FLUX_ARCHITECTURE)
    check_compatible(aio, plain)
    check_compatible(plain, aio)


# fp8 --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_fp8_weights_rejected_for_sd(ckpt, state_dict_factory, dtype):
    sd = state_dict_factory(0)
    sd["model.linear.weight"] = sd["model.linear.weight"].to(dtype)
    with pytest.raises(MergeError, match="fp8"):
        inspect_checkpoint(ckpt("sd", sd), SDXL_ARCHITECTURE)
    with pytest.raises(MergeError, match="fp8"):
        inspect_checkpoint(ckpt("sd", sd))


def test_fp8_transformer_rejected(ckpt):
    tensors = {k: v.to(torch.float8_e4m3fn) for k, v in ZIMAGE_DIFFUSERS.items()}
    with pytest.raises(MergeError, match="fp8"):
        inspect_checkpoint(ckpt("z", tensors), ZIMAGE_ARCHITECTURE)


@pytest.mark.parametrize("key", ["scaled_fp8", "layers.0.attention.to_q.scale_weight",
                                 "layers.0.attention.to_q.scale_input"])
def test_scaled_fp8_markers_rejected(ckpt, key):
    with pytest.raises(MergeError, match="scaled-fp8"):
        inspect_checkpoint(ckpt("z", {**ZIMAGE_DIFFUSERS, key: torch.zeros(1)}), ZIMAGE_ARCHITECTURE)


def test_scaled_fp8_marker_in_copied_component_allowed(ckpt):
    tensors = {**_prefixed(ZIMAGE_DIFFUSERS), "text_encoders.t5xxl.scale_weight": torch.zeros(1)}
    info = inspect_checkpoint(ckpt("aio", tensors), ZIMAGE_ARCHITECTURE)
    assert "text_encoders.t5xxl.scale_weight" in info.copy_keys


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
