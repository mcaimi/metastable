"""Checkpoint merging: tensor methods, checkpoint ops and merge pipelines."""

from metastable.merge.methods import MergeMethod, linear, merge_tensors, slerp
from metastable.merge.ops import (
    MergeConfig,
    batch_merge_checkpoints,
    load_checkpoint_dict,
    merge_checkpoints,
    save_checkpoint,
    validate_checkpoints,
)
from metastable.merge.pipeline import (
    ARCHITECTURES,
    ILLUSTRIOUS_ARCHITECTURE,
    SD15_ARCHITECTURE,
    SDXL_ARCHITECTURE,
    IllustriousMergePipeline,
    MergePipeline,
    ModelArchitectureInfo,
    SD15MergePipeline,
    SDXLMergePipeline,
)

__all__ = [
    "ARCHITECTURES",
    "ILLUSTRIOUS_ARCHITECTURE",
    "IllustriousMergePipeline",
    "MergeConfig",
    "MergeMethod",
    "MergePipeline",
    "ModelArchitectureInfo",
    "SD15MergePipeline",
    "SD15_ARCHITECTURE",
    "SDXLMergePipeline",
    "SDXL_ARCHITECTURE",
    "batch_merge_checkpoints",
    "linear",
    "load_checkpoint_dict",
    "merge_checkpoints",
    "merge_tensors",
    "save_checkpoint",
    "slerp",
    "validate_checkpoints",
]
