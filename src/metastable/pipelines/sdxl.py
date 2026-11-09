#!/usr/bin/env python
"""
Stable Diffusion XL Pipeline Generator
"""

from typing import Dict, Tuple

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLInpaintPipeline

from metastable.pipelines.base import BasePipelineGenerator


class SDXLPipelineGenerator(BasePipelineGenerator):
    """SDXL Pipeline Generator for text-to-image and inpainting."""

    PIPELINE_CLASS = StableDiffusionXLPipeline
    INPAINT_PIPELINE_CLASS = StableDiffusionXLInpaintPipeline
    DEFAULT_WIDTH = 1024
    DEFAULT_HEIGHT = 1024
    MODEL_TYPE = "sdxl"

    @staticmethod
    def get_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported SDXL resolutions."""
        return {
            "1024x1024": (1024, 1024),
            "1152x896": (1152, 896),
            "896x1152": (896, 1152),
            "1216x832": (1216, 832),
            "832x1216": (832, 1216),
            "1344x768": (1344, 768),
            "768x1344": (768, 1344),
            "1536x640": (1536, 640),
            "640x1536": (640, 1536),
        }

    def load_pipeline(self) -> None:
        """Load the SDXL generation pipeline."""
        self.pipeline = self._load_single_file(self.PIPELINE_CLASS, "SDXL")

    def load_inpaint_pipeline(self) -> None:
        """Load the SDXL inpainting pipeline."""
        self.inpaint_pipeline = self._load_single_file(
            self.INPAINT_PIPELINE_CLASS, "SDXL inpaint"
        )
