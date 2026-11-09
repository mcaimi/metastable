#!/usr/bin/env python
"""
Custom Exception Classes

Domain-specific exceptions raised by metastable. Every exception derives
from MetastableError so callers can catch library failures with one clause.
"""


class MetastableError(Exception):
    """Base class for every exception raised by metastable."""
    pass


class CheckpointLoadError(MetastableError):
    """
    Exception raised when a checkpoint file cannot be loaded.

    This can occur due to:
    - File not found
    - Invalid file format
    - Corrupted checkpoint data
    - Insufficient memory
    """
    pass


class MergeError(MetastableError):
    """
    Exception raised when a merge operation fails.

    This can occur due to:
    - Incompatible model architectures
    - Shape mismatches between tensors
    - Unsupported merge method
    - Memory allocation failures
    """
    pass


class ConfigurationError(MetastableError):
    """
    Exception raised when configuration loading or validation fails.

    This can occur due to:
    - Unknown device backend or dtype
    - Requested backend not available in the running PyTorch build
    - Malformed configuration mappings
    """
    pass


class PipelineError(MetastableError):
    """
    Exception raised when a pipeline operation fails.

    This can occur due to:
    - Model not loaded
    - Invalid inference parameters (e.g. unknown scheduler)
    - Device/memory issues
    """
    pass
