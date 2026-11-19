#!/usr/bin/env python
"""
Transformer-only Pipeline Generator Base

Base class for architectures (Flux, Z-Image) whose checkpoints only ship the
diffusion transformer. Text encoders, tokenizers, VAE and scheduler are loaded
once from a shared local diffusers-format directory and reused across every
checkpoint of that architecture.

Because those modules are shared, every generator using the same components
directory also shares one lock: inference, LoRA fusion and placement changes
on any of them run one at a time (model CPU offload moves shared modules
between devices, which would break a concurrent run).
"""

import logging
import threading
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from accelerate.hooks import remove_hook_from_module

from metastable.device import DeviceSettings
from metastable.pipelines.base import BasePipelineGenerator

logger = logging.getLogger(__name__)


# resolved components directory -> lock shared by its generators
_COMPONENT_LOCKS: Dict[str, threading.RLock] = {}
_COMPONENT_LOCKS_GUARD = threading.Lock()


def components_lock(components_dir: Union[str, Path]) -> threading.RLock:
    """Return the lock shared by every generator using components_dir.

    Keyed by the resolved path only (not dtype): a coarser key can only
    serialize more than needed, never race.
    """
    key = str(Path(components_dir).resolve())
    with _COMPONENT_LOCKS_GUARD:
        return _COMPONENT_LOCKS.setdefault(key, threading.RLock())


def _remove_offload_hooks(components: dict) -> None:
    """Strip accelerate hooks from shared modules.

    A shared module's offload hook references the previous model in the
    offload chain of the last pipeline that ran (e.g. the VAE's hook holds
    that pipeline's transformer), keeping it alive after its generator is
    dropped. Pipelines still using the modules reinstall their hooks before
    their next run.
    """
    for module in components.values():
        if isinstance(module, torch.nn.Module) and hasattr(module, "_hf_hook"):
            remove_hook_from_module(module, recurse=True)


class TransformerPipelineGenerator(BasePipelineGenerator):
    """
    Base generator for transformer-only checkpoints plus shared components.

    Subclasses implement the three loader hooks (_load_shared_components,
    _load_component, _load_transformer) and get_resolutions().
    """

    # Human-readable architecture name used in log messages
    DISPLAY_NAME = "transformer"

    def __init__(
        self,
        model_checkpoint: Union[str, Path],
        components_dir: Union[str, Path],
        device: Optional[DeviceSettings] = None,
    ):
        """
        Args:
            model_checkpoint: Path to the transformer-only safetensors file
            components_dir: Shared diffusers-format directory holding the
                text encoder(s), tokenizer(s), VAE and scheduler. If it also
                contains transformer/config.json, that config is used to
                load the checkpoint instead of one fetched from the Hub.
            device: Explicit device settings (see BasePipelineGenerator)
        """
        super().__init__(model_checkpoint, device=device)
        self.components_dir: str = str(components_dir)
        self._lock = components_lock(self.components_dir)

    @abstractmethod
    def _load_shared_components(self) -> Dict[str, Any]:
        """Return the cached shared components for (components_dir, dtype).

        The returned modules must never be modified in place: see
        _isolate_lora_targets().
        """

    @abstractmethod
    def _load_component(self, name: str) -> Any:
        """Load a single component from components_dir (uncached)."""

    @abstractmethod
    def _load_transformer(self, **kwargs) -> Any:
        """Load the transformer from the checkpoint with from_single_file()."""

    def _init_device(self) -> None:
        """Initialize GPU/device settings, preferring bf16 on accelerators.

        bf16 is only auto-selected when the user hasn't pinned a dtype in the
        device config: ROCm hosts (especially older RDNA GPUs like the RX
        5500 XT) default to float16, and an explicit dtype always wins.
        """
        super()._init_device()
        if self.accelerator != "cpu" and self.dtype_source == "auto":
            self.dtype = torch.bfloat16

    def _transformer_load_kwargs(self) -> Dict[str, Any]:
        """Keyword arguments for _load_transformer().

        Uses components_dir/transformer/config.json when present, so loading
        works offline and the checkpoint gets the config that matches it
        rather than the one diffusers infers from its keys.
        """
        kwargs: Dict[str, Any] = {"torch_dtype": self.dtype}
        if (Path(self.components_dir) / "transformer" / "config.json").is_file():
            kwargs.update(config=self.components_dir, subfolder="transformer")
        return kwargs

    def _build(self, pipeline_class, label: str):
        """Assemble a pipeline from the checkpoint transformer and the
        shared components.

        Stale offload hooks are stripped from the shared modules before the
        transformer is loaded, so a previous checkpoint's transformer is
        released first rather than coexisting with the new one in memory.
        Each pipeline gets its own scheduler: schedulers carry per-run state.
        """
        self._init_device()
        with self._lock:
            shared = dict(self._load_shared_components())
            _remove_offload_hooks(shared)
        scheduler = shared["scheduler"]
        shared["scheduler"] = type(scheduler).from_config(scheduler.config)

        logger.info(
            "Loading %s %stransformer checkpoint: %s",
            self.DISPLAY_NAME,
            label,
            self.model_checkpoint,
        )
        transformer = self._load_transformer(**self._transformer_load_kwargs())
        return pipeline_class(transformer=transformer, **shared)

    def load_pipeline(self) -> None:
        """Load the generation pipeline."""
        self.pipeline = self._build(self.PIPELINE_CLASS, "")

    def load_inpaint_pipeline(self) -> None:
        """Load the inpainting pipeline."""
        self.inpaint_pipeline = self._build(self.INPAINT_PIPELINE_CLASS, "inpaint ")

    def to_device(self) -> None:
        """Move pipeline(s) to the configured compute device.

        These models are large enough that a naive .to(device) often won't
        fit in consumer VRAM, so model CPU offload is used instead whenever
        an accelerator is available. enable_model_cpu_offload() raises on a
        CPU-only host, so fall back to the base behavior there.
        """
        if self.accelerator == "cpu":
            super().to_device()
            return

        with self._lock:
            for pipeline in self.pipelines:
                self._enable_offload(pipeline)
            self._placement = "offload"

    def _isolate_lora_targets(self, pipeline) -> None:
        """Give the pipeline private copies of shared LoRA-loadable components.

        Fusing a LoRA rewrites weights in place. Components that are still
        the cached shared instances (e.g. Flux's CLIP text encoder) are
        reloaded from disk first, so the fused weights never leak into the
        cache or into other pipelines. Components a LoRA can't touch stay
        shared.
        """
        shared = self._load_shared_components()
        isolated = False
        for name in getattr(pipeline, "_lora_loadable_modules", ()):
            if name not in shared or getattr(pipeline, name, None) is not shared[name]:
                continue
            logger.info("Loading private %s %s for LoRA fusion", self.DISPLAY_NAME, name)
            private = self._load_component(name)
            if self._placement == "device":
                private = private.to(self.torch_device)
            setattr(pipeline, name, private)
            isolated = True

        if isolated and self._placement == "offload":
            # New modules carry no offload hooks; reinstall them
            self._enable_offload(pipeline)
