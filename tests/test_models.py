import os
import struct

import pytest
import torch
from safetensors.torch import save_file

from metastable.models import (
    enumerate_models,
    read_safetensors_header,
    read_safetensors_metadata,
)


def test_enumerate_models_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    for p in ["a.safetensors", "sub/b.safetensors", "c.ckpt", "notes.txt"]:
        (tmp_path / p).write_bytes(b"")
    found = enumerate_models(tmp_path)
    assert set(found) == {"a.safetensors", "b.safetensors"}
    assert found["b.safetensors"] == tmp_path / "sub" / "b.safetensors"


def test_enumerate_models_accepts_str_and_sees_new_files(tmp_path):
    assert enumerate_models(str(tmp_path)) == {}
    (tmp_path / "new.safetensors").write_bytes(b"")
    assert "new.safetensors" in enumerate_models(str(tmp_path))


def test_enumerate_models_missing_dir_is_empty(tmp_path):
    assert enumerate_models(tmp_path / "missing") == {}


def test_enumerate_models_rejects_bad_type():
    with pytest.raises(TypeError):
        enumerate_models(123)


def test_read_header(checkpoint_factory):
    path = checkpoint_factory("model", metadata={"author": "me"})
    header = read_safetensors_header(path)
    assert header["__metadata__"] == {"author": "me"}
    assert header["model.conv.weight"]["shape"] == [4, 3, 3, 3]
    assert header["model.conv.weight"]["dtype"] == "F32"
    assert read_safetensors_header(str(path)) == header


def test_read_metadata(checkpoint_factory):
    assert read_safetensors_metadata(checkpoint_factory("m", metadata={"k": "v"})) == {"k": "v"}
    assert read_safetensors_metadata(checkpoint_factory("n")) == {}


def test_header_cache_invalidated_on_rewrite(checkpoint_factory):
    path = checkpoint_factory("model", metadata={"version": "1"})
    assert read_safetensors_metadata(path) == {"version": "1"}
    save_file({"x": torch.zeros(2)}, str(path), metadata={"version": "2", "extra": "yes"})
    # force a distinct mtime even on coarse-grained filesystems
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert read_safetensors_metadata(path) == {"version": "2", "extra": "yes"}


def test_returned_header_is_a_copy(checkpoint_factory):
    path = checkpoint_factory("model")
    read_safetensors_header(path)["injected"] = True
    assert "injected" not in read_safetensors_header(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_safetensors_header(tmp_path / "missing.safetensors")


def test_bad_type():
    with pytest.raises(TypeError):
        read_safetensors_header(None)


def test_truncated_file(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(b"\x01\x02")
    with pytest.raises(ValueError, match="truncated"):
        read_safetensors_header(path)


def test_bogus_header_length(tmp_path):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", 10**9) + b"{}")
    with pytest.raises(ValueError, match="bad header length"):
        read_safetensors_header(path)
