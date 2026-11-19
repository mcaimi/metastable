"""Streaming safetensors writer and tensor-by-tensor merge engine."""

import json
import struct

import pytest
import torch
from safetensors.torch import load_file, save_file

from metastable.exceptions import MergeError
from metastable.merge import stream
from metastable.merge.inspect import inspect_checkpoint
from metastable.merge.methods import MergeMethod, merge_tensors
from metastable.merge.ops import MergeConfig, merge_checkpoints
from metastable.merge.stream import (
    MergeStep,
    SafetensorsStreamWriter,
    safetensors_dtype,
    stream_merge,
    torch_dtype,
)


def _write_all(path, tensors, **kwargs):
    entries = [(name, t.dtype, t.shape) for name, t in tensors.items()]
    with SafetensorsStreamWriter(path, entries, **kwargs) as writer:
        for name in writer.names:
            writer.write(name, tensors[name])
    return writer


def _as_bytes(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8)


# writer -----------------------------------------------------------------------------------------

def test_writer_round_trip_all_dtypes(tmp_path):
    tensors = {
        "f32": torch.randn(3, 2),
        "f16": torch.randn(4).half(),
        "bf16": torch.randn(2, 2).to(torch.bfloat16),
        "f8": torch.randn(5).to(torch.float8_e4m3fn),
        "f8b": torch.randn(5).to(torch.float8_e5m2),
        "i64": torch.arange(3),
        "u8": torch.arange(4, dtype=torch.uint8),
        "flag": torch.tensor([True, False]),
        "scalar": torch.tensor(2.5),
        "empty": torch.zeros(0, 3),
    }
    out = tmp_path / "w.safetensors"
    _write_all(out, tensors, metadata={"source": "test", "n": 1})

    loaded = load_file(str(out))
    assert set(loaded) == set(tensors)
    for name, tensor in tensors.items():
        assert loaded[name].dtype == tensor.dtype and loaded[name].shape == tensor.shape
        assert torch.equal(_as_bytes(loaded[name]), _as_bytes(tensor))

    raw = out.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    assert (8 + header_len) % 8 == 0
    assert json.loads(raw[8:8 + header_len])["__metadata__"] == {"source": "test", "n": "1"}


def test_writer_orders_largest_dtype_first(tmp_path):
    writer = SafetensorsStreamWriter(tmp_path / "w.safetensors", [
        ("a", torch.float16, (1,)), ("b", torch.float32, (1,)),
        ("c", torch.uint8, (1,)), ("d", torch.float32, (2,)),
    ])
    assert writer.names == ["b", "d", "a", "c"]


@pytest.mark.parametrize("name,tensor,match", [
    ("b", torch.zeros(2), "out of order"),
    ("a", torch.zeros(2, dtype=torch.float16), "declared"),
    ("a", torch.zeros(3), "declared"),
])
def test_writer_rejects_mismatched_writes(tmp_path, name, tensor, match):
    out = tmp_path / "w.safetensors"
    with pytest.raises(MergeError, match=match):
        with SafetensorsStreamWriter(out, [("a", torch.float32, (2,)), ("b", torch.float32, (2,))]) as w:
            w.write(name, tensor)
    assert list(tmp_path.iterdir()) == []


def test_writer_failure_leaves_no_files(tmp_path):
    out = tmp_path / "w.safetensors"
    with pytest.raises(RuntimeError):
        with SafetensorsStreamWriter(out, [("a", torch.float32, (2,)), ("b", torch.float32, (2,))]) as w:
            w.write("a", torch.zeros(2))
            raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []


def test_writer_incomplete_raises_and_cleans_up(tmp_path):
    out = tmp_path / "w.safetensors"
    with pytest.raises(MergeError, match="Only 1 of 2"):
        with SafetensorsStreamWriter(out, [("a", torch.float32, (2,)), ("b", torch.float32, (2,))]) as w:
            w.write("a", torch.zeros(2))
    assert list(tmp_path.iterdir()) == []


def test_writer_keeps_existing_output_on_failure(tmp_path):
    out = tmp_path / "w.safetensors"
    out.write_bytes(b"original")
    with pytest.raises(RuntimeError):
        with SafetensorsStreamWriter(out, [("a", torch.float32, (1,))]):
            raise RuntimeError("boom")
    assert out.read_bytes() == b"original"


def test_writer_backs_up_existing_output(tmp_path):
    out = tmp_path / "w.safetensors"
    out.write_bytes(b"original")
    _write_all(out, {"a": torch.ones(1)})
    backups = list(tmp_path.glob("w.backup_*.safetensors"))
    assert len(backups) == 1 and backups[0].read_bytes() == b"original"
    assert torch.equal(load_file(str(out))["a"], torch.ones(1))


def test_writer_overwrites_without_backup(tmp_path):
    out = tmp_path / "w.safetensors"
    out.write_bytes(b"original")
    _write_all(out, {"a": torch.ones(1)}, backup_existing=False)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["w.safetensors"]


def test_writer_rejects_duplicate_names(tmp_path):
    with pytest.raises(MergeError, match="Duplicate"):
        SafetensorsStreamWriter(tmp_path / "w.safetensors",
                                [("a", torch.float32, (1,)), ("a", torch.float32, (1,))])


def test_dtype_tables():
    assert torch_dtype("BF16") is torch.bfloat16
    assert safetensors_dtype(torch.float8_e4m3fn) == "F8_E4M3"
    with pytest.raises(MergeError):
        torch_dtype("F4")
    with pytest.raises(MergeError):
        safetensors_dtype(torch.complex64)


# engine -----------------------------------------------------------------------------------------

def _save(path, tensors):
    save_file(tensors, str(path))
    return inspect_checkpoint(path)


@pytest.fixture
def pair(tmp_path, state_dict_factory):
    a, b = state_dict_factory(0), state_dict_factory(1)
    a["model.half"] = torch.randn(4, 4).to(torch.bfloat16)
    b["model.half"] = torch.randn(4, 4).to(torch.bfloat16)
    return a, b, _save(tmp_path / "a.safetensors", a), _save(tmp_path / "b.safetensors", b)


@pytest.mark.parametrize("method", list(MergeMethod))
def test_stream_merge_matches_in_memory_merge(tmp_path, pair, method):
    a, b, info_a, info_b = pair
    out = tmp_path / "out.safetensors"
    result = stream_merge(info_a, [MergeStep(info_b, method, 0.3)], out)

    expected = merge_checkpoints(a, b, MergeConfig(method=method, alpha=0.3))
    merged = load_file(str(out))
    assert result.merged_keys == len(a) and result.copied_keys == 0
    assert set(merged) == set(expected)
    assert all(torch.equal(merged[k], expected[k]) for k in expected)


def test_stream_recipe_matches_in_memory_recipe(tmp_path, state_dict_factory):
    a, b, c = (state_dict_factory(i) for i in range(3))
    infos = [_save(tmp_path / f"{n}.safetensors", sd) for n, sd in zip("abc", (a, b, c))]
    steps = [(MergeMethod.LINEAR, 0.5), (MergeMethod.ADDITIVE, 0.25), (MergeMethod.SUBTRACT, 0.1)]
    targets = [infos[1], infos[2], infos[1]]

    out = tmp_path / "out.safetensors"
    stream_merge(infos[0], [MergeStep(t, m, al) for t, (m, al) in zip(targets, steps)], out,
                 reference=infos[0])

    # In-memory recipe semantics: ADDITIVE/SUBTRACT use the original base
    current = dict(a)
    for target, (method, alpha) in zip((b, c, b), steps):
        ref = a if method in (MergeMethod.ADDITIVE, MergeMethod.SUBTRACT) else None
        current = merge_checkpoints(current, target, MergeConfig(method=method, alpha=alpha), ref)
    merged = load_file(str(out))
    assert all(torch.equal(merged[k], current[k]) for k in a)


def test_stream_merge_output_dtype(tmp_path, pair):
    a, b, info_a, info_b = pair
    out = tmp_path / "out.safetensors"
    stream_merge(info_a, [MergeStep(info_b, MergeMethod.LINEAR, 0.5)], out,
                 output_dtype=torch.float16)
    merged = load_file(str(out))
    assert all(t.dtype == torch.float16 for t in merged.values())
    key = "model.linear.weight"
    expected = merge_tensors(a[key], b[key], MergeMethod.LINEAR, 0.5).to(torch.float16)
    assert torch.equal(merged[key], expected)


def test_stream_merge_progress_and_chunks(tmp_path, pair, monkeypatch):
    emptied = []
    monkeypatch.setattr(stream, "empty_accelerator_cache", lambda: emptied.append(1))
    a, b, info_a, info_b = pair
    progress = []
    stream_merge(info_a, [MergeStep(info_b, MergeMethod.LINEAR, 0.5)], tmp_path / "o.safetensors",
                 progress_callback=progress.append, chunk_size=2)
    assert progress == [i / len(a) for i in range(len(a))] + [1.0]
    assert len(emptied) == len(a) // 2


def test_stream_merge_without_steps_copies_base(tmp_path, pair):
    a, _, info_a, _ = pair
    out = tmp_path / "out.safetensors"
    stream_merge(info_a, [], out)
    merged = load_file(str(out))
    assert all(torch.equal(merged[k], a[k]) for k in a)


def test_stream_merge_failure_keeps_previous_output(tmp_path, pair, monkeypatch):
    _, _, info_a, info_b = pair
    out = tmp_path / "out.safetensors"
    out.write_bytes(b"previous")

    def boom(*args, **kwargs):
        raise RuntimeError("merge failed")

    monkeypatch.setattr(stream, "merge_tensors", boom)
    with pytest.raises(RuntimeError):
        stream_merge(info_a, [MergeStep(info_b, MergeMethod.LINEAR, 0.5)], out)
    assert out.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.safetensors", "b.safetensors", "out.safetensors"]
