#!/usr/bin/env python
"""
Illustrious XL Pipeline Generator

Illustrious XL is architecturally SDXL-based but supports higher native
resolutions (up to 1536x1536).
"""

from typing import Dict, Tuple

from metastable.pipelines.sdxl import SDXLPipelineGenerator


class IllustriousPipelineGenerator(SDXLPipelineGenerator):
    """Illustrious XL Pipeline Generator for text-to-image and inpainting."""

    MODEL_TYPE = "illustrious"

    @staticmethod
    def get_resolutions() -> Dict[str, Tuple[int, int]]:
        """Return supported Illustrious XL resolutions."""
        return {
            **SDXLPipelineGenerator.get_resolutions(),
            "1280x1280": (1280, 1280),
            "1536x1536": (1536, 1536),
        }

    def load_pipeline(self) -> None:
        """Load the Illustrious XL generation pipeline."""
        self.pipeline = self._load_single_file(self.PIPELINE_CLASS, "Illustrious XL")

    def load_inpaint_pipeline(self) -> None:
        """Load the Illustrious XL inpainting pipeline."""
        self.inpaint_pipeline = self._load_single_file(
            self.INPAINT_PIPELINE_CLASS, "Illustrious XL inpaint"
        )
