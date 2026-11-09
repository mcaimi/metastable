#!/usr/bin/env python
"""
Generation Metadata Container
"""

from typing import Any, Dict


class GenerationMetadata:
    """
    Container for image generation metadata with dynamic attribute access.

    Expects a dictionary with an "output_parameters" entry shaped like the
    output of format_metadata() ({"instances": [{...}]}). Top-level keys and
    the first instance's keys are exposed as attributes.
    """

    def __init__(self, metadict: Dict[str, Any]) -> None:
        """
        Initialize metadata from a dictionary.

        Args:
            metadict: Dictionary containing generation metadata

        Raises:
            ValueError: If metadata dictionary is invalid or missing required fields
        """
        self.metadata: Dict[str, Any] = metadict

        try:
            for k in self.metadata.keys():
                setattr(self, f"{k}", self.metadata.get(k))

            # Access generation data specifically
            instance_parms: Dict[str, Any] = self.output_parameters.get("instances")[0]

            for k in instance_parms.keys():
                setattr(self, f"{k}", instance_parms.get(k))
        except (AttributeError, KeyError, TypeError, IndexError) as e:
            raise ValueError(f"Cannot load metadata from dictionary: {e}") from e
