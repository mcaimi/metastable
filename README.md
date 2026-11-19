# metastable

Stable Diffusion middleware: pipeline generators, compute-device selection and
checkpoint merging, originally developed as part of [sd-api-backend](https://github.com/mcaimi/sd-api-backend)
and now extracted so the same engine can power API servers and standalone applications.

- **Pipelines** – SD 1.5, SDXL, Illustrious XL, Flux and Z-Image (Base and
  Turbo) text-to-image and inpainting, LoRA fusion, custom VAEs, scheduler
  selection
- **Devices** – CUDA, ROCm (incl. `gfx_override` for unsupported AMD cards),
  Apple MPS and CPU, with automatic dtype selection
- **Merging** – linear / SLERP / additive / subtract merges, batch merges and
  multi-step recipes (SD 1.5, SDXL, Illustrious XL)
- **Telemetry** (optional) – GPU memory/utilization and host stats

`import metastable` is cheap: heavy dependencies load on first use.

## Installation

```bash
uv add metastable                 # or: pip install metastable
uv add "metastable[telemetry]"    # + psutil for host/MPS stats
```

Local development / use from a sibling project:

```toml
# consumer pyproject.toml
[tool.uv.sources]
metastable = { path = "../sd-lib", editable = true }
```

`torch` is declared without a specific build. The consuming application picks
the CPU / CUDA / ROCm wheel index (see sd-api-backend's `pyproject.linux.toml`).

## Quick start

```python
import metastable
from PIL import Image

metastable.configure_device({"backend": "auto"})   # optional; auto-detects otherwise

gen = metastable.create_generator("sdxl", "models/sdxl/model.safetensors")
gen.load_pipeline()
gen.add_loras([metastable.LoraSpec("models/loras/style.safetensors", strength=0.7)])
gen.to_device()

image, metadata = gen.forward(
    positive_prompt="a lighthouse at dusk",
    negative_prompt="blurry",
    scheduler_type="DPM++ 2M",   # None keeps the model's scheduler
    steps=25, width=1024, height=1024, cfg=7.0,
    seed=-1,                     # negative = random; metadata reports the seed used
)
Image.fromarray(image).save("out.png")
```

Flux needs the shared text-encoder/VAE directory:

```python
gen = metastable.create_generator(
    "flux", "models/flux/flux1-dev.safetensors", components_dir="models/flux_components"
)
gen.load_pipeline()
gen.to_device()              # model CPU offload on accelerators
image, meta = gen.forward("a fox", "", "FlowMatch Euler", 28, 1024, 1024, 3.5, 42,
                          true_cfg_scale=1.0)
```

The text encoders, VAE and scheduler in `components_dir` are loaded once and
shared by every Flux pipeline. Before LoRAs are fused, a pipeline gets its own
copy of each component a LoRA can modify (the CLIP text encoder), so fused
weights never leak into other pipelines. T5 and the VAE stay shared.

Because modules are shared, every generator using the same `components_dir`
shares one lock (`gen.lock`): generations, LoRA fusion and placement changes
on any of them run one at a time, and concurrent requests queue up. Use
`gen.lock.acquire(blocking=False)` to report "busy" instead. Single-file
generators (SD 1.5, SDXL, Illustrious) only serialize calls on the same
generator. Generators with different components directories (e.g. Flux and
Z-Image) are locked separately and may run concurrently, competing for VRAM.

If `components_dir` also contains `transformer/config.json`, it is used to
load the checkpoint, so loading works offline; otherwise diffusers infers
the config from the checkpoint keys and may fetch it from the Hugging Face
Hub.

Z-Image (Base and Turbo) works the same way, with a components directory in
the `Tongyi-MAI/Z-Image` or `Tongyi-MAI/Z-Image-Turbo` layout (Qwen3 text
encoder, tokenizer, VAE, scheduler):

```python
gen = metastable.create_generator(
    "zimage", "models/zimage/z_image_turbo_bf16.safetensors",
    components_dir="models/zimage_turbo_components",
)
gen.load_pipeline()
gen.to_device()
image, meta = gen.forward("a fox", "", None, 8, 1024, 1024, 0.0, 42)    # Turbo
```

One `zimage` model type serves both checkpoints; pick the settings to match:

| checkpoint | steps | `cfg` (`guidance_scale`) | negative prompt   |
|------------|-------|--------------------------|-------------------|
| Turbo      | ~8    | `0.0`                    | ignored (CFG off) |
| Base       | 28–50 | ~3–5                     | used              |

CFG is enabled whenever `cfg > 0`. Pair each checkpoint with the components
directory from its own repository: the text encoder and VAE are identical,
but the scheduler shift differs (3.0 for Turbo, 6.0 for Base). Each
directory loads its own copy of the Qwen3 text encoder.
`forward_inpaint()` takes an optional `strength` (pipeline default 1.0).

Flux and Z-Image use the flow-matching schedulers in `FLOW_MATCH_SCHEDULERS`
(renamed from `FLUX_SCHEDULERS`; the old name is no longer available).
`"FlowMatch Heun"` has been removed: both pipelines pass custom sigmas to the
scheduler, which it does not support.

Inpainting:

```python
gen.load_inpaint_pipeline()
gen.to_device()
image, meta = gen.forward_inpaint("a red door", "", base_image, mask, "Euler a",
                                  steps=30, cfg=7.0, seed=1)
```

Any extra keyword argument to `forward()` / `forward_inpaint()` is passed
straight to the diffusers pipeline call.

See [`examples/`](examples/) for runnable scripts.

## Device configuration

```python
metastable.configure_device({
    "backend": "auto",         # auto | cuda | rocm | mps | cpu
    "index": 0,                # GPU ordinal
    "dtype": "auto",           # auto | float16 | bfloat16 | float32
    "gfx_override": "10.3.0",  # ROCm: HSA_OVERRIDE_GFX_VERSION (e.g. RX 5500 XT)
    "env": {},                 # extra env vars applied before HIP/CUDA init
})
```

Call it once at startup, before anything else touches `torch.cuda`. Any
mapping or object with `to_dict()` works. Without it, the library
auto-detects mps → cuda/rocm → cpu. To override per generator, pass
`device=DeviceSettings(...)` (e.g. `DeviceSettings.cpu()`) to its constructor
or to `create_generator()`.

Flux and Z-Image default to bfloat16 on accelerators unless `dtype` is set
explicitly.

## Merging

```python
from metastable.merge import MergeConfig, MergePipeline

pipe = MergePipeline.for_model_type("sdxl", device="cpu")
pipe.merge("base.safetensors", "target.safetensors",
           MergeConfig(method="slerp", alpha=0.3), "merged.safetensors")

recipe = pipe.create_merge_recipe("base.safetensors", [
    {"target_model": "a.safetensors", "method": "linear", "alpha": 0.5},
    {"target_model": "b.safetensors", "method": "additive", "alpha": 0.2},
])
pipe.execute_merge_recipe(recipe, "recipe.safetensors")
pipe.clear_cache()
```

**`alpha` is always the weight of the target model:**

| method   | result |
|----------|--------|
| linear   | `(1 - alpha) * base + alpha * target` |
| slerp    | spherical interpolation, `alpha=0` → base, `alpha=1` → target |
| additive | `base + alpha * (target - reference)` |
| subtract | `base - alpha * (target - reference)` |

`reference` defaults to the base. In recipes it is the original base model.
`MergeConfig(chunk_size=N)` frees accelerator memory every N tensors.

## Development

```bash
uv sync
uv run pytest                      # CPU-only, no model downloads
uv run pytest --cov=metastable
```

## License

This project is released under the GPL v3
