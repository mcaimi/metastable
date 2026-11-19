#!/usr/bin/env python
"""
Merge two checkpoints with metastable.

    uv run python examples/merge.py sdxl base.safetensors target.safetensors out.safetensors \
        --method slerp --alpha 0.3
    uv run python examples/merge.py zimage base.safetensors target.safetensors out.safetensors \
        --alpha 0.4 --output-dtype bf16

alpha is the weight of the *target* model for every method.
"""

import argparse
import logging

from metastable.merge import ARCHITECTURES, MergeConfig, MergePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_type", choices=list(ARCHITECTURES))
    parser.add_argument("base")
    parser.add_argument("target")
    parser.add_argument("output")
    parser.add_argument("--method", default="linear", choices=["linear", "slerp", "additive", "subtract"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", help="device the merge math runs on")
    parser.add_argument("--output-dtype", default=None,
                        help="bf16 | float16 | float32 (default: keep the base model's dtype)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pipeline = MergePipeline.for_model_type(args.model_type, device=args.device)
    config = MergeConfig(method=args.method, alpha=args.alpha, device=pipeline.device,
                         output_dtype=args.output_dtype)
    print(pipeline.merge(args.base, args.target, config, args.output))


if __name__ == "__main__":
    main()
