#!/usr/bin/env python
"""
Streaming Checkpoint Merge

Merges safetensors checkpoints one tensor at a time: inputs are read through
memory-mapped safe_open() handles and every merged tensor is written straight
to disk, so peak memory stays at a few tensors regardless of model size.

Results are identical to the in-memory merge_checkpoints(): each tensor goes
through the same merge_tensors() call.
"""

import json
import logging
import os
import struct
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from safetensors import safe_open

from metastable.exceptions import MergeError
from metastable.merge.files import backup_existing as backup_existing_file
from metastable.merge.files import empty_accelerator_cache
from metastable.merge.inspect import CheckpointInfo
from metastable.merge.methods import MergeMethod, merge_tensors

logger = logging.getLogger(__name__)

# safetensors dtype name <-> torch dtype
_SAFETENSORS_DTYPES: Dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}
_TORCH_DTYPES = {dtype: name for name, dtype in _SAFETENSORS_DTYPES.items()}

# Header length field (little-endian u64) and header alignment
_HEADER_LEN_SIZE = 8
_ALIGNMENT = 8


def torch_dtype(safetensors_dtype: str) -> torch.dtype:
    """Torch dtype for a safetensors dtype name."""
    try:
        return _SAFETENSORS_DTYPES[safetensors_dtype]
    except KeyError:
        raise MergeError(f"Unsupported tensor dtype: {safetensors_dtype}") from None


def safetensors_dtype(dtype: torch.dtype) -> str:
    """Safetensors dtype name for a torch dtype."""
    try:
        return _TORCH_DTYPES[dtype]
    except KeyError:
        raise MergeError(f"Unsupported tensor dtype: {dtype}") from None


def _numel(shape: Sequence[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


class SafetensorsStreamWriter:
    """
    Write a safetensors file one tensor at a time.

    The header is computed up front from the declared (name, dtype, shape)
    entries, so tensors can be written as they are produced. Entries are
    stored largest dtype first (as safetensors does, keeping data aligned);
    write them in the order given by `names`.

    The file is written to a temporary path next to the output and renamed
    into place only once every tensor has been written; on any error the
    temporary file is removed and the output is left untouched.

    Usage:
        with SafetensorsStreamWriter(path, entries) as writer:
            for name in writer.names:
                writer.write(name, tensor_for(name))
    """

    def __init__(
        self,
        path: Union[str, Path],
        entries: Sequence[Tuple[str, torch.dtype, Sequence[int]]],
        metadata: Optional[Mapping[str, str]] = None,
        backup_existing: bool = True,
    ):
        """
        Args:
            path: Output file
            entries: (name, dtype, shape) of every tensor
            metadata: Optional string metadata stored in the header
            backup_existing: Rename an existing output file to a backup
                instead of overwriting it
        """
        self.path = Path(path)
        self.metadata = dict(metadata or {})
        self.backup_existing = backup_existing
        ordered = sorted(entries, key=lambda e: -e[1].itemsize)
        self._entries = [(name, dtype, tuple(shape)) for name, dtype, shape in ordered]
        self.names: List[str] = [name for name, _, _ in self._entries]
        if len(set(self.names)) != len(self.names):
            raise MergeError("Duplicate tensor names in safetensors entries")
        self._tmp_path = self.path.with_name(f"{self.path.name}.tmp-{os.getpid()}")
        self._file = None
        self._next = 0

    def _header(self) -> bytes:
        header: Dict[str, object] = {}
        if self.metadata:
            header["__metadata__"] = {str(k): str(v) for k, v in self.metadata.items()}
        offset = 0
        for name, dtype, shape in self._entries:
            size = _numel(shape) * dtype.itemsize
            header[name] = {
                "dtype": safetensors_dtype(dtype),
                "shape": list(shape),
                "data_offsets": [offset, offset + size],
            }
            offset += size
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        padding = -(_HEADER_LEN_SIZE + len(encoded)) % _ALIGNMENT
        return encoded + b" " * padding

    def __enter__(self) -> "SafetensorsStreamWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        header = self._header()
        self._file = open(self._tmp_path, "wb")
        self._file.write(struct.pack("<Q", len(header)))
        self._file.write(header)
        return self

    def write(self, name: str, tensor: torch.Tensor) -> None:
        """Write the next tensor; name, dtype and shape must match its entry."""
        if self._file is None:
            raise MergeError("SafetensorsStreamWriter is not open")
        if self._next >= len(self._entries):
            raise MergeError(f"Unexpected tensor '{name}': all tensors already written")
        expected, dtype, shape = self._entries[self._next]
        if name != expected:
            raise MergeError(f"Tensor '{name}' written out of order; expected '{expected}'")
        if tensor.dtype != dtype or tuple(tensor.shape) != shape:
            raise MergeError(
                f"Tensor '{name}' is {tensor.dtype} {tuple(tensor.shape)}; "
                f"declared {dtype} {shape}"
            )
        data = tensor.detach().to("cpu").contiguous().reshape(-1).view(torch.uint8)
        self._file.write(memoryview(data.numpy()))
        self._next += 1

    def __exit__(self, exc_type, exc, tb) -> bool:
        complete = exc_type is None and self._next == len(self._entries)
        try:
            if complete:
                self._file.flush()
                os.fsync(self._file.fileno())
            self._file.close()
            if complete:
                if self.backup_existing:
                    backup_existing_file(self.path)
                os.replace(self._tmp_path, self.path)
        finally:
            if self._tmp_path.exists():
                self._tmp_path.unlink()
            self._file = None
        if exc_type is None and not complete:
            raise MergeError(
                f"Only {self._next} of {len(self._entries)} tensors were written to {self.path}"
            )
        return False


@dataclass(frozen=True)
class MergeStep:
    """One merge step: blend a target into the running result."""

    target: CheckpointInfo
    method: MergeMethod
    alpha: float


@dataclass(frozen=True)
class StreamMergeResult:
    """Outcome of stream_merge()."""

    path: Path
    merged_keys: int
    copied_keys: int


def stream_merge(
    base: CheckpointInfo,
    steps: Sequence[MergeStep],
    output_path: Union[str, Path],
    *,
    reference: Optional[CheckpointInfo] = None,
    device: str = "cpu",
    output_dtype: Optional[torch.dtype] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
    chunk_size: Optional[int] = None,
    backup_existing: bool = True,
) -> StreamMergeResult:
    """
    Merge checkpoints tensor by tensor and stream the result to disk.

    For every merge key of the base, each step applies
    merge_tensors(current, target, method, alpha, base_tensor=reference) in
    order. Without a reference, ADDITIVE/SUBTRACT use the running result
    (as merge_checkpoints() does); recipes pass the original base. Copied
    keys of the base (all-in-one checkpoints) are written unchanged. The
    output uses the base file's key names.

    Inputs must already be compatible (see check_compatible()).

    Args:
        base: Base checkpoint
        steps: Merge steps, applied in order
        output_path: Output safetensors file
        reference: Reference checkpoint for ADDITIVE/SUBTRACT steps
        device: Device the merge math runs on
        output_dtype: dtype of merged tensors (None keeps the base dtype);
            copied tensors keep theirs
        progress_callback: Called with progress in [0, 1]
        chunk_size: Release accelerator memory every chunk_size merged keys
        backup_existing: Rename an existing output instead of overwriting

    Returns:
        Output path and the number of merged and copied tensors
    """
    sources: Dict[str, Tuple[str, Optional[str]]] = {}  # file key -> (kind, merge key)
    entries = []
    for key, entry in base.merge_keys.items():
        dtype = output_dtype or torch_dtype(entry.dtype)
        entries.append((entry.file_key, dtype, entry.shape))
        sources[entry.file_key] = ("merge", key)
    for file_key, entry in base.copy_keys.items():
        entries.append((file_key, torch_dtype(entry.dtype), entry.shape))
        sources[file_key] = ("copy", None)

    total = len(entries)
    merged = 0
    with ExitStack() as stack:
        def _open(info: CheckpointInfo):
            return stack.enter_context(safe_open(str(info.path), framework="pt", device="cpu"))

        base_handle = _open(base)
        step_handles = [(step, _open(step.target)) for step in steps]
        reference_handle = _open(reference) if reference is not None else None
        writer = stack.enter_context(
            SafetensorsStreamWriter(output_path, entries, backup_existing=backup_existing)
        )

        for i, file_key in enumerate(writer.names):
            if progress_callback:
                progress_callback(i / total)

            kind, key = sources[file_key]
            tensor = base_handle.get_tensor(file_key)
            if kind == "copy":
                writer.write(file_key, tensor)
                continue

            current = tensor.to(device)
            for step, handle in step_handles:
                target = handle.get_tensor(step.target.merge_keys[key].file_key).to(device)
                reference_tensor = None
                if reference_handle is not None and step.method in (
                    MergeMethod.ADDITIVE, MergeMethod.SUBTRACT
                ):
                    reference_tensor = reference_handle.get_tensor(
                        reference.merge_keys[key].file_key
                    ).to(device)
                current = merge_tensors(
                    current, target, step.method, step.alpha, base_tensor=reference_tensor
                )
            if output_dtype is not None:
                current = current.to(output_dtype)
            writer.write(file_key, current)

            merged += 1
            if chunk_size and merged % chunk_size == 0:
                empty_accelerator_cache()

    if progress_callback:
        progress_callback(1.0)

    logger.info("Streamed merge saved: %s (%d merged, %d copied)",
                output_path, merged, total - merged)
    return StreamMergeResult(Path(output_path), merged, total - merged)
