"""Checkpoint merging: tensor methods, checkpoint ops and merge pipelines."""

from metastable.merge.inspect import (
    CheckpointInfo,
    TensorEntry,
    check_compatible,
    inspect_checkpoint,
)
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
    FLUX_ARCHITECTURE,
    ILLUSTRIOUS_ARCHITECTURE,
    SD15_ARCHITECTURE,
    SDXL_ARCHITECTURE,
    ZIMAGE_ARCHITECTURE,
    FluxMergePipeline,
    IllustriousMergePipeline,
    MergePipeline,
    ModelArchitectureInfo,
    SD15MergePipeline,
    SDXLMergePipeline,
    ZImageMergePipeline,
)
from metastable.merge.stream import (
    MergeStep,
    SafetensorsStreamWriter,
    StreamMergeResult,
    stream_merge,
)

__all__ = [
    "ARCHITECTURES",
    "CheckpointInfo",
    "FLUX_ARCHITECTURE",
    "FluxMergePipeline",
    "ILLUSTRIOUS_ARCHITECTURE",
    "IllustriousMergePipeline",
    "MergeConfig",
    "MergeMethod",
    "MergePipeline",
    "MergeStep",
    "ModelArchitectureInfo",
    "SD15MergePipeline",
    "SD15_ARCHITECTURE",
    "SDXLMergePipeline",
    "SDXL_ARCHITECTURE",
    "SafetensorsStreamWriter",
    "StreamMergeResult",
    "TensorEntry",
    "ZIMAGE_ARCHITECTURE",
    "ZImageMergePipeline",
    "batch_merge_checkpoints",
    "check_compatible",
    "inspect_checkpoint",
    "linear",
    "load_checkpoint_dict",
    "merge_checkpoints",
    "merge_tensors",
    "save_checkpoint",
    "slerp",
    "stream_merge",
    "validate_checkpoints",
]
