#!/usr/bin/env python
"""
Minimal standalone text-to-image app built on metastable.

    uv run python examples/txt2img.py sdxl /path/to/model.safetensors "a lighthouse at dusk"
    uv run python examples/txt2img.py flux /path/to/flux.safetensors "a fox" \
        --components /path/to/flux_components --scheduler "FlowMatch Euler" --cfg 3.5
"""

import argparse
import json
import logging

from PIL import Image

import metastable
from metastable.pipelines import LoraSpec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_type", choices=metastable.available_model_types())
    parser.add_argument("checkpoint")
    parser.add_argument("prompt")
    parser.add_argument("--negative", default="")
    parser.add_argument("--components", help="Flux shared components directory")
    parser.add_argument("--scheduler", default=None, help="scheduler name (default: keep the model's)")
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--size", default=None, help="WxH (default: model default)")
    parser.add_argument("--lora", action="append", default=[], help="path[:strength], repeatable")
    parser.add_argument("--vae", help="custom VAE checkpoint")
    parser.add_argument("--backend", default="auto", help="auto | cuda | rocm | mps | cpu")
    parser.add_argument("--output", default="output.png")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    metastable.configure_device({"backend": args.backend})

    kwargs = {"components_dir": args.components} if args.model_type == "flux" else {}
    generator = metastable.create_generator(args.model_type, args.checkpoint, **kwargs)
    generator.load_pipeline()

    if args.lora:
        specs = []
        for entry in args.lora:
            path, _, strength = entry.partition(":")
            specs.append(LoraSpec(path, float(strength) if strength else 0.5))
        generator.add_loras(specs)
    if args.vae:
        generator.load_custom_vae(args.vae)
    generator.to_device()

    if args.size:
        width, height = (int(v) for v in args.size.lower().split("x"))
    else:
        width, height = generator.DEFAULT_WIDTH, generator.DEFAULT_HEIGHT

    image, metadata = generator.forward(
        positive_prompt=args.prompt,
        negative_prompt=args.negative,
        scheduler_type=args.scheduler,
        steps=args.steps,
        width=width,
        height=height,
        cfg=args.cfg,
        seed=args.seed,
    )
    Image.fromarray(image).save(args.output)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
