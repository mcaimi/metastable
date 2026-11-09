#!/usr/bin/env python
"""
Model File Helpers

Safetensors header parsing and model directory enumeration.
"""

import json
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Union

# Safetensors header length (8 bytes for metadata length field)
SFT_HEADER_LEN = 8


def _to_path(filename: Union[str, Path]) -> Path:
    if isinstance(filename, (str, Path)):
        return Path(filename)
    raise TypeError(
        f"Expected str or Path, got {type(filename).__name__}"
    )


def enumerate_models(path: Union[str, Path]) -> Dict[str, Path]:
    """
    Enumerate all safetensors model files under a directory (recursive).

    Not cached: directories change at runtime (e.g. merge outputs), and a
    glob is cheap compared to the staleness bugs a cache would introduce.

    Args:
        path: Directory to search for models

    Returns:
        Dictionary mapping filename to full Path
    """
    model_path = _to_path(path)
    return {
        filepath.name: filepath
        for filepath in sorted(model_path.glob("**/*.safetensors"))
    }


@lru_cache(maxsize=128)
def _read_header_cached(filepath_str: str, mtime_ns: int, size: int) -> Dict[str, Any]:
    """Read a safetensors header; mtime/size are part of the cache key only."""
    with open(filepath_str, "rb") as f:
        header_bytes = f.read(SFT_HEADER_LEN)
        if len(header_bytes) != SFT_HEADER_LEN:
            raise ValueError(f"Not a safetensors file (truncated header): {filepath_str}")
        # Header length is a little-endian unsigned long long
        metadata_len = struct.unpack("<Q", header_bytes)[0]
        if metadata_len > size - SFT_HEADER_LEN:
            raise ValueError(f"Not a safetensors file (bad header length): {filepath_str}")
        return json.loads(f.read(metadata_len))


def read_safetensors_header(filename: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse the JSON header of a safetensors file.

    The header contains the tensor index plus the optional "__metadata__"
    entry. Results are cached per (path, mtime, size), so a rewritten file
    is re-read automatically.

    Args:
        filename: Path to the safetensors file

    Returns:
        Parsed header dictionary

    Raises:
        TypeError: If filename is not str or Path
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a valid safetensors file
    """
    filepath = _to_path(filename).absolute()
    if not filepath.is_file():
        raise FileNotFoundError(f"Safetensors file not found: {filepath}")
    stat = filepath.stat()
    # Return a copy so callers can't mutate the cached value
    return dict(_read_header_cached(str(filepath), stat.st_mtime_ns, stat.st_size))


def read_safetensors_metadata(filename: Union[str, Path]) -> Dict[str, str]:
    """
    Return only the user metadata ("__metadata__") of a safetensors file.

    Returns:
        Metadata dictionary, empty if the file has none
    """
    return dict(read_safetensors_header(filename).get("__metadata__", {}))
