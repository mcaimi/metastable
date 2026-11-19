import json

import pytest
import torch
from safetensors.torch import load_file

from metastable import device as dm
from metastable.device import DeviceSettings
from metastable.exceptions import MergeError
from metastable.merge import pipeline as mp
from metastable.merge.ops import MergeConfig
from metastable.merge.pipeline import (
    ARCHITECTURES,
    IllustriousMergePipeline,
    MergePipeline,
    SD15MergePipeline,
    SDXLMergePipeline,
)
from metastable.pipelines import (
    IllustriousPipelineGenerator,
    SD15PipelineGenerator,
    SDXLPipelineGenerator,
)


@pytest.fixture
def cpu_configured():
    dm.configure_device({"backend": "cpu"})


# architectures ---------------------------------------------------------------------------

def test_architectures_registry():
    assert list(ARCHITECTURES) == ["sd15", "sdxl", "illustrious"]
    assert "flux" not in ARCHITECTURES
    assert not any(arch.transformer_only or arch.layouts for arch in ARCHITECTURES.values())


@pytest.mark.parametrize("name,gen,base_res", [
    ("sd15", SD15PipelineGenerator, 512),
    ("sdxl", SDXLPipelineGenerator, 1024),
    ("illustrious", IllustriousPipelineGenerator, 1024),
])
def test_architecture_resolutions_match_generators(name, gen, base_res):
    arch = ARCHITECTURES[name]
    assert arch.supported_resolutions == list(gen.get_resolutions())
    assert arch.base_resolution == base_res


# device resolution ----------------------------------------------------------------------------

def test_auto_device_uses_process_settings(cpu_configured):
    assert MergePipeline.for_model_type("sd15").device == "cpu"


def test_explicit_device():
    assert SD15MergePipeline(device="cpu").device == "cpu"
    assert SDXLMergePipeline(device="cuda:1").device == "cuda:1"


def test_rocm_device_maps_to_cuda_api(monkeypatch):
    monkeypatch.setattr(mp, "get_device_settings", lambda: DeviceSettings(
        backend="rocm", torch_device="cuda:2", index=2, dtype=torch.float16,
        dtype_source="auto", vendor="amd"))
    assert MergePipeline.for_model_type("sdxl", device="rocm").device == "cuda:2"


# constructors -------------------------------------------------------------------------------------

@pytest.mark.parametrize("cls,name", [
    (SD15MergePipeline, "sd15"), (SDXLMergePipeline, "sdxl"), (IllustriousMergePipeline, "illustrious"),
])
def test_convenience_classes(cls, name):
    assert cls(device="cpu").architecture.name == name


def test_for_model_type_rejects_flux():
    with pytest.raises(MergeError, match="not supported.*flux"):
        MergePipeline.for_model_type("flux")


# caching ----------------------------------------------------------------------------------------------

def test_load_and_cache_checkpoint(checkpoint_factory, monkeypatch):
    path = checkpoint_factory("a")
    pipe = SD15MergePipeline(device="cpu")
    loads = []
    real = mp.load_checkpoint_dict
    monkeypatch.setattr(mp, "load_checkpoint_dict", lambda p, d: loads.append(p) or real(p, d))
    key = pipe.load_and_cache_checkpoint(path)
    assert key == str(path.absolute())
    assert pipe.load_and_cache_checkpoint(str(path)) == key
    assert len(loads) == 1
    assert pipe.loaded_checkpoints[key]["path"] == path


def test_cache_key_distinguishes_same_stem(tmp_path, checkpoint_factory):
    a = checkpoint_factory("model", seed=0)
    (tmp_path / "other").mkdir()
    b = tmp_path / "other" / "model.safetensors"
    b.write_bytes(a.read_bytes())
    pipe = SD15MergePipeline(device="cpu")
    assert pipe.load_and_cache_checkpoint(a) != pipe.load_and_cache_checkpoint(b)


def test_custom_cache_key(checkpoint_factory):
    pipe = SD15MergePipeline(device="cpu")
    assert pipe.load_and_cache_checkpoint(checkpoint_factory("a"), cache_key="mine") == "mine"
    assert pipe.get_cache_info() == {
        "cached_models": ["mine"], "cache_size": 1, "device": "cpu", "architecture": "sd15"}


def test_clear_cache(checkpoint_factory):
    pipe = SD15MergePipeline(device="cpu")
    pipe.load_and_cache_checkpoint(checkpoint_factory("a"))
    pipe.clear_cache()
    assert pipe.get_cache_info()["cache_size"] == 0


# merge -----------------------------------------------------------------------------------------------------

def test_merge(tmp_path, checkpoint_factory, state_dict_factory):
    base, target = checkpoint_factory("base", seed=0), checkpoint_factory("target", seed=1)
    out = tmp_path / "out.safetensors"
    pipe = SDXLMergePipeline(device="cpu")
    result = pipe.merge(base, target, MergeConfig(alpha=0.25), out)
    assert result == str(out)
    a, b = state_dict_factory(0), state_dict_factory(1)
    merged = load_file(str(out))
    for k in a:
        assert torch.allclose(merged[k], 0.75 * a[k] + 0.25 * b[k])
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["merge_info"]["method"] == "linear"
    assert meta["merge_info"]["alpha"] == 0.25
    assert meta["model_info"]["architecture"] == "sdxl"


def test_merge_without_metadata(tmp_path, checkpoint_factory):
    out = tmp_path / "out.safetensors"
    SD15MergePipeline(device="cpu").merge(
        checkpoint_factory("a"), checkpoint_factory("b", seed=1),
        MergeConfig(preserve_metadata=False), out)
    assert out.exists() and not out.with_suffix(".json").exists()


def test_merge_missing_checkpoint_wrapped(tmp_path, checkpoint_factory):
    with pytest.raises(MergeError, match="merge operation failed"):
        SD15MergePipeline(device="cpu").merge(
            checkpoint_factory("a"), tmp_path / "missing.safetensors", MergeConfig(),
            tmp_path / "out.safetensors")


def test_merge_incompatible_raises_merge_error(tmp_path, checkpoint_factory):
    bad = checkpoint_factory("bad", weights={"x": torch.zeros(1)})
    with pytest.raises(MergeError, match="validation failed"):
        SD15MergePipeline(device="cpu").merge(
            checkpoint_factory("a"), bad, MergeConfig(), tmp_path / "out.safetensors")


def _failing_stream_merge(*args, **kwargs):
    raise OSError("disk full")


def test_merge_write_failure_wrapped(tmp_path, checkpoint_factory, monkeypatch):
    monkeypatch.setattr(mp, "stream_merge", _failing_stream_merge)
    out = tmp_path / "out.safetensors"
    with pytest.raises(MergeError, match="merge operation failed: disk full"):
        SD15MergePipeline(device="cpu").merge(
            checkpoint_factory("a"), checkpoint_factory("b", seed=1), MergeConfig(), out)
    assert not out.exists() and not out.with_suffix(".json").exists()


# recipes -----------------------------------------------------------------------------------------------------

def test_create_merge_recipe(tmp_path):
    pipe = SD15MergePipeline(device="cpu")
    steps = [{"target_model": "t.safetensors", "method": "slerp", "alpha": 0.3}]
    recipe = pipe.create_merge_recipe(tmp_path / "base.safetensors", steps)
    assert recipe["base_model"] == str(tmp_path / "base.safetensors")
    assert recipe["steps"] == steps and recipe["steps"][0] is not steps[0]
    assert recipe["architecture"] == "sd15"
    assert isinstance(recipe["created_at"], float)


@pytest.mark.parametrize("step,match", [({"method": "linear"}, "target_model"),
                                        ({"target_model": "t", "method": "bogus"}, "Unknown merge method")])
def test_create_merge_recipe_validates_steps(step, match):
    with pytest.raises(MergeError, match=match):
        SD15MergePipeline(device="cpu").create_merge_recipe("base", [step])


def test_execute_merge_recipe(tmp_path, checkpoint_factory, state_dict_factory):
    base = checkpoint_factory("base", seed=0)
    t1, t2 = checkpoint_factory("t1", seed=1), checkpoint_factory("t2", seed=2)
    pipe = SD15MergePipeline(device="cpu")
    recipe = pipe.create_merge_recipe(base, [
        {"target_model": t1, "method": "LINEAR", "alpha": 0.5, "show_progress": True},
        {"target_model": t2, "method": "additive", "alpha": 0.25},
    ])
    out = tmp_path / "recipe.safetensors"
    assert pipe.execute_merge_recipe(recipe, out) is True

    a, b, c = state_dict_factory(0), state_dict_factory(1), state_dict_factory(2)
    merged = load_file(str(out))
    for k in a:
        step1 = 0.5 * a[k] + 0.5 * b[k]
        expected = step1 + 0.25 * (c[k] - a[k])  # additive uses the original base as reference
        assert torch.allclose(merged[k], expected, atol=1e-6)

    meta = json.loads(out.with_suffix(".json").read_text())
    assert [s["method"] for s in meta["steps_executed"]] == ["linear", "additive"]
    assert [s["alpha"] for s in meta["steps_executed"]] == [0.5, 0.25]
    assert len({s["completed_at"] for s in meta["steps_executed"]}) == 1
    # streamed recipes don't load checkpoints into the cache
    assert pipe.get_cache_info()["cache_size"] == 0


def test_execute_recipe_defaults(tmp_path, checkpoint_factory, state_dict_factory):
    pipe = SD15MergePipeline(device="cpu")
    recipe = pipe.create_merge_recipe(checkpoint_factory("base"), [{"target_model": checkpoint_factory("t", seed=1)}])
    out = tmp_path / "out.safetensors"
    pipe.execute_merge_recipe(recipe, out)
    a, b = state_dict_factory(0), state_dict_factory(1)
    assert torch.allclose(load_file(str(out))["model.conv.bias"], 0.5 * a["model.conv.bias"] + 0.5 * b["model.conv.bias"])


def test_execute_recipe_missing_target_wrapped(tmp_path, checkpoint_factory):
    pipe = SD15MergePipeline(device="cpu")
    recipe = pipe.create_merge_recipe(checkpoint_factory("base"), [{"target_model": tmp_path / "missing.safetensors"}])
    with pytest.raises(MergeError, match="Recipe execution failed"):
        pipe.execute_merge_recipe(recipe, tmp_path / "out.safetensors")


def test_execute_recipe_write_failure_wrapped(tmp_path, checkpoint_factory, monkeypatch):
    monkeypatch.setattr(mp, "stream_merge", _failing_stream_merge)
    pipe = SD15MergePipeline(device="cpu")
    recipe = pipe.create_merge_recipe(checkpoint_factory("base"), [])
    with pytest.raises(MergeError, match="Recipe execution failed: disk full"):
        pipe.execute_merge_recipe(recipe, tmp_path / "out.safetensors")


# streaming specifics ----------------------------------------------------------------------------

def test_merge_metadata_records_layout_dtype_and_copies(tmp_path, checkpoint_factory):
    out = tmp_path / "out.safetensors"
    SDXLMergePipeline(device="cpu").merge(
        checkpoint_factory("a"), checkpoint_factory("b", seed=1),
        MergeConfig(output_dtype="float16"), out)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["merge_info"]["output_dtype"] == "float16"
    assert meta["merge_info"]["copied_keys"] == 0
    assert meta["model_info"]["layout"] is None
    assert all(t.dtype == torch.float16 for t in load_file(str(out)).values())


def test_merge_reads_input_sidecars(tmp_path, checkpoint_factory):
    a, b = checkpoint_factory("a"), checkpoint_factory("b", seed=1)
    a.with_suffix(".json").write_text(json.dumps({"origin": "a"}))
    out = tmp_path / "out.safetensors"
    SD15MergePipeline(device="cpu").merge(a, b, MergeConfig(), out)
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["base_metadata"] == {"origin": "a"} and meta["target_metadata"] == {}


def test_merge_rejects_fp8_checkpoints(tmp_path, checkpoint_factory, state_dict_factory):
    sd = state_dict_factory(1)
    sd["model.linear.weight"] = sd["model.linear.weight"].to(torch.float8_e4m3fn)
    with pytest.raises(MergeError, match="fp8"):
        SDXLMergePipeline(device="cpu").merge(
            checkpoint_factory("a"), checkpoint_factory("b", weights=sd), MergeConfig(),
            tmp_path / "out.safetensors")


def test_recipe_output_dtype(tmp_path, checkpoint_factory):
    pipe = SD15MergePipeline(device="cpu")
    recipe = pipe.create_merge_recipe(checkpoint_factory("base"), [{"target_model": checkpoint_factory("t", seed=1)}])
    out = tmp_path / "out.safetensors"
    pipe.execute_merge_recipe(recipe, out, output_dtype="bf16")
    assert all(t.dtype == torch.bfloat16 for t in load_file(str(out)).values())
    assert json.loads(out.with_suffix(".json").read_text())["output_dtype"] == "bfloat16"


def test_batch_merge_method_end_to_end(tmp_path, checkpoint_factory):
    results = SD15MergePipeline(device="cpu").batch_merge(
        checkpoint_factory("base"), [checkpoint_factory("t", seed=1)], tmp_path / "out", MergeConfig())
    assert results == [str(tmp_path / "out" / "merged_t.safetensors")]
