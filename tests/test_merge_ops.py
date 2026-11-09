import json

import pytest
import torch
from safetensors.torch import load_file

from metastable.exceptions import CheckpointLoadError, MergeError
from metastable.merge import ops
from metastable.merge.methods import MergeMethod
from metastable.merge.ops import (
    MergeConfig,
    batch_merge_checkpoints,
    load_checkpoint_dict,
    merge_checkpoints,
    save_checkpoint,
    validate_checkpoints,
)


# MergeConfig ---------------------------------------------------------------------------

def test_config_defaults():
    c = MergeConfig()
    assert (c.method, c.alpha, c.device, c.preserve_metadata, c.chunk_size) == (
        MergeMethod.LINEAR, 0.5, "cpu", True, None)


def test_config_coerces_method_and_alpha():
    c = MergeConfig(method="slerp", alpha="0.25")
    assert c.method is MergeMethod.SLERP
    assert c.alpha == 0.25


@pytest.mark.parametrize("kwargs", [{"alpha": "abc"}, {"alpha": None}, {"method": "bogus"}, {"chunk_size": 0}])
def test_config_validation(kwargs):
    with pytest.raises(MergeError):
        MergeConfig(**kwargs)


# load_checkpoint_dict --------------------------------------------------------------------

def test_load_checkpoint(checkpoint_factory, state_dict_factory):
    path = checkpoint_factory("a", seed=1)
    weights, metadata = load_checkpoint_dict(path)
    expected = state_dict_factory(1)
    assert weights.keys() == expected.keys()
    assert all(torch.equal(weights[k], expected[k]) for k in expected)
    assert metadata == {}


def test_load_checkpoint_reads_json_sidecar(checkpoint_factory):
    path = checkpoint_factory("a")
    path.with_suffix(".json").write_text(json.dumps({"origin": "test"}))
    assert load_checkpoint_dict(str(path))[1] == {"origin": "test"}


def test_load_missing_checkpoint(tmp_path):
    with pytest.raises(CheckpointLoadError, match="not found"):
        load_checkpoint_dict(tmp_path / "missing.safetensors")


def test_load_wrong_suffix(tmp_path):
    path = tmp_path / "model.ckpt"
    path.write_bytes(b"")
    with pytest.raises(CheckpointLoadError, match="Only safetensors"):
        load_checkpoint_dict(path)


def test_load_corrupt_checkpoint(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"garbage")
    with pytest.raises(CheckpointLoadError, match="Failed to load"):
        load_checkpoint_dict(path)


# validate_checkpoints ----------------------------------------------------------------------

def test_validate_compatible(state_dict_factory):
    assert validate_checkpoints(state_dict_factory(0), state_dict_factory(1)) == (True, [])


def test_validate_no_others(state_dict_factory):
    assert validate_checkpoints(state_dict_factory(0)) == (True, [])


def test_validate_extra_and_missing_keys(state_dict_factory):
    base = state_dict_factory(0)
    other = state_dict_factory(1)
    other["extra"] = torch.zeros(1)
    del other["model.conv.bias"]
    ok, errors = validate_checkpoints(base, other)
    assert not ok
    assert any("extra keys" in e and "extra" in e for e in errors)
    assert any("missing keys" in e and "model.conv.bias" in e for e in errors)


def test_validate_shape_mismatch(state_dict_factory):
    other = state_dict_factory(1)
    other["model.conv.bias"] = torch.zeros(5)
    ok, errors = validate_checkpoints(state_dict_factory(0), other)
    assert not ok
    assert errors == ["Shape mismatch for key 'model.conv.bias': (4,) vs (5,)"]


def test_validate_reports_checkpoint_index(state_dict_factory):
    bad = state_dict_factory(2)
    bad["x"] = torch.zeros(1)
    _, errors = validate_checkpoints(state_dict_factory(0), state_dict_factory(1), bad)
    assert errors and all(e.startswith("Checkpoint 2") for e in errors)


# merge_checkpoints ----------------------------------------------------------------------------

def test_merge_linear(state_dict_factory):
    a, b = state_dict_factory(0), state_dict_factory(1)
    merged = merge_checkpoints(a, b, MergeConfig(alpha=0.25))
    for k in a:
        assert torch.allclose(merged[k], 0.75 * a[k] + 0.25 * b[k])


def test_merge_does_not_mutate_inputs(state_dict_factory):
    a, b = state_dict_factory(0), state_dict_factory(1)
    a_copy = {k: v.clone() for k, v in a.items()}
    merge_checkpoints(a, b, MergeConfig(method="slerp"))
    assert all(torch.equal(a[k], a_copy[k]) for k in a)


def test_merge_incompatible_raises(state_dict_factory):
    b = state_dict_factory(1)
    b["model.conv.bias"] = torch.zeros(9)
    with pytest.raises(MergeError, match="validation failed"):
        merge_checkpoints(state_dict_factory(0), b, MergeConfig())


def test_merge_with_reference(state_dict_factory):
    a, b, ref = state_dict_factory(0), state_dict_factory(1), state_dict_factory(2)
    merged = merge_checkpoints(a, b, MergeConfig(method="additive", alpha=0.5), base_reference=ref)
    for k in a:
        assert torch.allclose(merged[k], a[k] + 0.5 * (b[k] - ref[k]))


def test_reference_ignored_for_interpolation(state_dict_factory):
    a, b, ref = state_dict_factory(0), state_dict_factory(1), state_dict_factory(2)
    with_ref = merge_checkpoints(a, b, MergeConfig(), base_reference=ref)
    without = merge_checkpoints(a, b, MergeConfig())
    assert all(torch.equal(with_ref[k], without[k]) for k in a)


def test_merge_validates_reference(state_dict_factory):
    ref = state_dict_factory(2)
    del ref["model.conv.bias"]
    with pytest.raises(MergeError):
        merge_checkpoints(state_dict_factory(0), state_dict_factory(1),
                          MergeConfig(method="additive"), base_reference=ref)


def test_merge_progress_callback(state_dict_factory):
    progress = []
    merge_checkpoints(state_dict_factory(0), state_dict_factory(1),
                      MergeConfig(progress_callback=progress.append))
    assert progress == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]


def test_merge_empty_checkpoints():
    progress = []
    assert merge_checkpoints({}, {}, MergeConfig(progress_callback=progress.append)) == {}
    assert progress == [1.0]


def test_chunked_merge_matches_unchunked(state_dict_factory, monkeypatch):
    emptied = []
    monkeypatch.setattr(ops, "_empty_accelerator_cache", lambda: emptied.append(1))
    a, b = state_dict_factory(0), state_dict_factory(1)
    chunked = merge_checkpoints(a, b, MergeConfig(method="slerp", alpha=0.3, chunk_size=2))
    full = merge_checkpoints(a, b, MergeConfig(method="slerp", alpha=0.3))
    assert all(torch.equal(chunked[k], full[k]) for k in a)
    assert len(emptied) == 1  # 3 keys, chunk of 2 -> one full chunk boundary


# save_checkpoint ------------------------------------------------------------------------------

def test_save_and_reload(tmp_path, state_dict_factory):
    weights = state_dict_factory(0)
    out = tmp_path / "nested" / "out.safetensors"
    assert save_checkpoint(weights, out, metadata={"a": 1})
    loaded = load_file(str(out))
    assert all(torch.equal(loaded[k], weights[k]) for k in weights)
    assert json.loads(out.with_suffix(".json").read_text()) == {"a": 1}


def test_save_without_metadata_writes_no_sidecar(tmp_path, state_dict_factory):
    out = tmp_path / "out.safetensors"
    assert save_checkpoint(state_dict_factory(0), out)
    assert not out.with_suffix(".json").exists()


def test_save_non_contiguous_tensor(tmp_path):
    t = torch.randn(4, 3).t()
    assert not t.is_contiguous()
    assert save_checkpoint({"t": t}, tmp_path / "out.safetensors")


def test_save_backs_up_existing(tmp_path, state_dict_factory):
    out = tmp_path / "out.safetensors"
    save_checkpoint(state_dict_factory(0), out)
    save_checkpoint(state_dict_factory(1), out)
    backups = list(tmp_path.glob("out.backup_*.safetensors"))
    assert len(backups) == 1
    assert torch.equal(load_file(str(backups[0]))["model.conv.bias"], state_dict_factory(0)["model.conv.bias"])
    assert torch.equal(load_file(str(out))["model.conv.bias"], state_dict_factory(1)["model.conv.bias"])


def test_save_overwrite_without_backup(tmp_path, state_dict_factory):
    out = tmp_path / "out.safetensors"
    save_checkpoint(state_dict_factory(0), out)
    save_checkpoint(state_dict_factory(1), out, backup_existing=False)
    assert not list(tmp_path.glob("*.backup_*"))


def test_save_failure_returns_false(tmp_path):
    assert save_checkpoint({"bad": "not a tensor"}, tmp_path / "out.safetensors") is False


def test_save_metadata_with_non_json_values(tmp_path, state_dict_factory):
    out = tmp_path / "out.safetensors"
    assert save_checkpoint(state_dict_factory(0), out, metadata={"path": tmp_path})
    assert json.loads(out.with_suffix(".json").read_text()) == {"path": str(tmp_path)}


# batch_merge_checkpoints ------------------------------------------------------------------------

def test_batch_merge(tmp_path, checkpoint_factory, state_dict_factory):
    base = checkpoint_factory("base", seed=0)
    targets = [checkpoint_factory("t1", seed=1), checkpoint_factory("t2", seed=2)]
    out_dir = tmp_path / "out"
    results = batch_merge_checkpoints(base, targets, out_dir, MergeConfig(alpha=1.0))
    assert results == [str(out_dir / "merged_t1.safetensors"), str(out_dir / "merged_t2.safetensors")]
    merged = load_file(results[1])
    assert torch.allclose(merged["model.conv.bias"], state_dict_factory(2)["model.conv.bias"])
    meta = json.loads((out_dir / "merged_t1.json").read_text())
    assert meta["merge_method"] == "linear"
    assert meta["merge_alpha"] == 1.0
    assert meta["target_model"] == str(targets[0])


def test_batch_merge_skips_failures(tmp_path, checkpoint_factory):
    base = checkpoint_factory("base", seed=0)
    bad = checkpoint_factory("bad", weights={"other": torch.zeros(1)})
    good = checkpoint_factory("good", seed=1)
    results = batch_merge_checkpoints(base, [bad, tmp_path / "missing.safetensors", good],
                                      tmp_path / "out", MergeConfig())
    assert results == [str(tmp_path / "out" / "merged_good.safetensors")]


def test_batch_merge_without_metadata(tmp_path, checkpoint_factory):
    results = batch_merge_checkpoints(checkpoint_factory("base"), [checkpoint_factory("t", seed=1)],
                                      tmp_path / "out", MergeConfig(preserve_metadata=False))
    assert len(results) == 1
    assert not list((tmp_path / "out").glob("*.json"))


def test_batch_merge_missing_base_raises(tmp_path):
    with pytest.raises(CheckpointLoadError):
        batch_merge_checkpoints(tmp_path / "missing.safetensors", [], tmp_path / "out", MergeConfig())
