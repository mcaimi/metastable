#!/usr/bin/env python
"""
Stable Diffusion 1.5 Pipeline Generator
"""

from typing import Dict, Tuple

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline

from metastable.pipelines.base import BasePipelineGenerator


class SD15PipelineGenerator(BasePipelineGenerator):
    """SD1.5 Pipeline Generator for text-to-image and inpainting."""

    PIPELINE_CLASS = StableDiffusionPipeline
    INPAINT_PIPELINE_CLASS = StableDiffusionInpaintPipeline
    DEFAULT_WIDTH = 512
    DEFAULT_HEIGHT = 512
    MODEL_TYPE = "sd15"

    @staticmethod
    def get_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported SD1.5 resolutions."""
        return {
            "512x512": (512, 512),
            "512x768": (512, 768),
            "768x512": (768, 512),
        }

    def load_pipeline(self) -> None:
        """Load the SD1.5 generation pipeline."""
        self.pipeline = self._load_single_file(self.PIPELINE_CLASS, "SD1.5")

    def load_inpaint_pipeline(self) -> None:
        """Load the SD1.5 inpainting pipeline."""
        self.inpaint_pipeline = self._load_single_file(
            self.INPAINT_PIPELINE_CLASS, "SD1.5 inpaint"
        )
