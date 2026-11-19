"""
Merge tiny randomly initialised Flux / Z-Image transformers and reload the
result with diffusers' from_single_file().
"""

import pytest
import torch
from diffusers import FluxTransformer2DModel, ZImageTransformer2DModel
from safetensors.torch import load_file, save_file

from metastable.exceptions import MergeError
from metastable.merge.methods import MergeMethod, merge_tensors
from metastable.merge.ops import MergeConfig
from metastable.merge.pipeline import FluxMergePipeline, ZImageMergePipeline

PREFIX = "model.diffusion_model."


def _tiny_zimage(seed):
    torch.manual_seed(seed)
    return ZImageTransformer2DModel(
        dim=64, n_layers=1, n_refiner_layers=1, n_heads=2, n_kv_heads=2,
        cap_feat_dim=32, axes_dims=[8, 12, 12], axes_lens=[64, 32, 32],
    )


def _tiny_flux(seed):
    torch.manual_seed(seed)
    return FluxTransformer2DModel(
        patch_size=1, in_channels=4, num_layers=1, num_single_layers=1,
        attention_head_dim=8, num_attention_heads=2, joint_attention_dim=16,
        pooled_projection_dim=16, guidance_embeds=True, axes_dims_rope=(2, 2, 4),
    )


def _state_dict(model):
    return {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}


def _zimage_to_original(sd, prefix=""):
    """Inverse of diffusers' Z-Image single-file conversion (test helper)."""
    renames = {
        "all_final_layer.2-1.": "final_layer.",
        "all_x_embedder.2-1.": "x_embedder.",
        ".attention.to_out.0.": ".attention.out.",
        ".attention.norm_k.": ".attention.k_norm.",
        ".attention.norm_q.": ".attention.q_norm.",
    }
    out = {}
    for key, tensor in sd.items():
        if ".attention.to_k." in key or ".attention.to_v." in key:
            continue
        if ".attention.to_q." in key:
            base = key.replace(".attention.to_q.", ".attention.qkv.")
            out[base] = torch.cat([
                tensor, sd[key.replace("to_q", "to_k")], sd[key.replace("to_q", "to_v")],
            ])
            continue
        for old, new in renames.items():
            key = key.replace(old, new)
        out[key] = tensor
    out["norm_final.weight"] = torch.ones(1)  # present in original files, dropped by diffusers
    return {prefix + k: v for k, v in out.items()}


def _expected(sd_a, sd_b, alpha):
    return {k: merge_tensors(sd_a[k], sd_b[k], MergeMethod.LINEAR, alpha) for k in sd_a}


def _assert_model_equals(model, expected):
    loaded = model.state_dict()
    assert set(loaded) == set(expected)
    for key, tensor in expected.items():
        assert torch.equal(loaded[key], tensor), key


@pytest.fixture
def zimage_pair(tmp_path):
    a, b = _tiny_zimage(0), _tiny_zimage(1)
    components = tmp_path / "components"
    a.save_config(components / "transformer")
    return _state_dict(a), _state_dict(b), components


def _from_single_file(cls, path, components):
    return cls.from_single_file(str(path), config=str(components), subfolder="transformer")


def test_zimage_diffusers_layout_round_trip(tmp_path, zimage_pair):
    sd_a, sd_b, components = zimage_pair
    save_file(sd_a, str(tmp_path / "a.safetensors"))
    save_file(sd_b, str(tmp_path / "b.safetensors"))
    out = tmp_path / "out.safetensors"

    ZImageMergePipeline(device="cpu").merge(
        tmp_path / "a.safetensors", tmp_path / "b.safetensors", MergeConfig(alpha=0.25), out)

    assert set(load_file(str(out))) == set(sd_a)
    _assert_model_equals(_from_single_file(ZImageTransformer2DModel, out, components),
                         _expected(sd_a, sd_b, 0.25))


@pytest.mark.parametrize("prefix_a,prefix_b", [("", ""), (PREFIX, ""), ("", PREFIX)])
def test_zimage_original_layout_round_trip(tmp_path, zimage_pair, prefix_a, prefix_b):
    sd_a, sd_b, components = zimage_pair
    save_file(_zimage_to_original(sd_a, prefix_a), str(tmp_path / "a.safetensors"))
    save_file(_zimage_to_original(sd_b, prefix_b), str(tmp_path / "b.safetensors"))
    out = tmp_path / "out.safetensors"

    ZImageMergePipeline(device="cpu").merge(
        tmp_path / "a.safetensors", tmp_path / "b.safetensors", MergeConfig(alpha=0.25), out)

    # Output keeps the base file's key names (layout and prefix)
    assert set(load_file(str(out))) == set(_zimage_to_original(sd_a, prefix_a))
    _assert_model_equals(_from_single_file(ZImageTransformer2DModel, out, components),
                         _expected(sd_a, sd_b, 0.25))


def test_zimage_mixed_layouts_rejected(tmp_path, zimage_pair):
    sd_a, sd_b, _ = zimage_pair
    save_file(sd_a, str(tmp_path / "a.safetensors"))
    save_file(_zimage_to_original(sd_b), str(tmp_path / "b.safetensors"))
    with pytest.raises(MergeError, match="layouts differ"):
        ZImageMergePipeline(device="cpu").merge(
            tmp_path / "a.safetensors", tmp_path / "b.safetensors", MergeConfig(),
            tmp_path / "out.safetensors")


def test_zimage_all_in_one_keeps_base_components(tmp_path, zimage_pair):
    sd_a, sd_b, _ = zimage_pair
    extras = {
        "vae.decoder.conv_in.weight": torch.randn(4, 4).to(torch.bfloat16),
        "text_encoders.qwen3.embed.weight": torch.randn(8).to(torch.float8_e4m3fn),
    }
    save_file({**_zimage_to_original(sd_a, PREFIX), **extras}, str(tmp_path / "a.safetensors"))
    save_file(_zimage_to_original(sd_b), str(tmp_path / "b.safetensors"))
    out = tmp_path / "out.safetensors"

    ZImageMergePipeline(device="cpu").merge(
        tmp_path / "a.safetensors", tmp_path / "b.safetensors",
        MergeConfig(alpha=0.5, output_dtype="float16"), out)

    merged = load_file(str(out))
    for key, tensor in extras.items():
        assert merged[key].dtype == tensor.dtype
        assert torch.equal(merged[key].view(torch.uint8), tensor.view(torch.uint8))
    original_a = _zimage_to_original(sd_a)
    original_b = _zimage_to_original(sd_b)
    key = "layers.0.attention.qkv.weight"
    expected = merge_tensors(original_a[key], original_b[key], MergeMethod.LINEAR, 0.5)
    assert torch.equal(merged[PREFIX + key], expected.to(torch.float16))


def test_zimage_transformer_only_base_with_all_in_one_target(tmp_path, zimage_pair):
    sd_a, sd_b, _ = zimage_pair
    save_file(_zimage_to_original(sd_a), str(tmp_path / "a.safetensors"))
    save_file({**_zimage_to_original(sd_b, PREFIX), "vae.w": torch.zeros(2)},
              str(tmp_path / "b.safetensors"))
    out = tmp_path / "out.safetensors"
    ZImageMergePipeline(device="cpu").merge(
        tmp_path / "a.safetensors", tmp_path / "b.safetensors", MergeConfig(), out)
    assert set(load_file(str(out))) == set(_zimage_to_original(sd_a))


def test_zimage_recipe_round_trip(tmp_path, zimage_pair):
    sd_a, sd_b, components = zimage_pair
    sd_c = _state_dict(_tiny_zimage(2))
    for name, sd in zip("abc", (sd_a, sd_b, sd_c)):
        save_file(sd, str(tmp_path / f"{name}.safetensors"))
    pipe = ZImageMergePipeline(device="cpu")
    recipe = pipe.create_merge_recipe(tmp_path / "a.safetensors", [
        {"target_model": tmp_path / "b.safetensors", "method": "slerp", "alpha": 0.3},
        {"target_model": tmp_path / "c.safetensors", "method": "additive", "alpha": 0.2},
    ])
    out = tmp_path / "out.safetensors"
    pipe.execute_merge_recipe(recipe, out)

    expected = {}
    for k in sd_a:
        step = merge_tensors(sd_a[k], sd_b[k], MergeMethod.SLERP, 0.3)
        expected[k] = merge_tensors(step, sd_c[k], MergeMethod.ADDITIVE, 0.2, base_tensor=sd_a[k])
    _assert_model_equals(_from_single_file(ZImageTransformer2DModel, out, components), expected)


def test_flux_diffusers_layout_round_trip(tmp_path):
    a, b = _tiny_flux(0), _tiny_flux(1)
    components = tmp_path / "components"
    a.save_config(components / "transformer")
    sd_a, sd_b = _state_dict(a), _state_dict(b)
    save_file(sd_a, str(tmp_path / "a.safetensors"))
    save_file(sd_b, str(tmp_path / "b.safetensors"))
    out = tmp_path / "out.safetensors"

    FluxMergePipeline(device="cpu").merge(
        tmp_path / "a.safetensors", tmp_path / "b.safetensors", MergeConfig(alpha=0.6), out)

    _assert_model_equals(_from_single_file(FluxTransformer2DModel, out, components),
                         _expected(sd_a, sd_b, 0.6))
