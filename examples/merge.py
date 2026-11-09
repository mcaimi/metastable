#!/usr/bin/env python
"""
Merge two checkpoints with metastable.

    uv run python examples/merge.py sdxl base.safetensors target.safetensors out.safetensors \
        --method slerp --alpha 0.3

alpha is the weight of the *target* model for every method.
"""

import argparse
import logging

from metastable.merge import MergeConfig, MergePipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_type", choices=["sd15", "sdxl", "illustrious"])
    parser.add_argument("base")
    parser.add_argument("target")
    parser.add_argument("output")
    parser.add_argument("--method", default="linear", choices=["linear", "slerp", "additive", "subtract"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", help="device checkpoints are loaded to")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    pipeline = MergePipeline.for_model_type(args.model_type, device=args.device)
    config = MergeConfig(method=args.method, alpha=args.alpha, device=pipeline.device)
    print(pipeline.merge(args.base, args.target, config, args.output))


if __name__ == "__main__":
    main()
