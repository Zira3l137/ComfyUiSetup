# ComfyUI setup: FLUX 3 Klein + Qwen-Image-2.1 (RTX 3060 12 GB, 16 GB RAM)

## Install

1. Hugging Face token (needed once, for the gated FLUX 3 Klein files):
   - accept the license at https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy
   - create a **Read** token at https://huggingface.co/settings/tokens
   - in Nushell: `$env.HF_TOKEN = "hf_..."` (add it to `env.nu` to keep it; never commit it)
2. From this folder: `nu install-comfyui.nu C:\AI\ComfyUI`  (SSD; about 54 GB with all defaults; `--no-detailer` saves 1.8 GB)

## Day to day (run from the install folder)

| Script | What it does |
|---|---|
| `nu launch.nu` | Start ComfyUI with the tuning below (Manager on, previews on, opens the browser). `--stock` = plain defaults, `--no-browser`; other flags pass through |
| `nu gen.nu klein "prompt"` | Generate from the terminal. `--ref img.png` (repeatable) = edit mode, `--upsample` = prompt rewrite, `--seed`, `--width/--height`, `--out dir`. Run `nu gen.nu` for all options |
| `nu gen.nu qwen "prompt"` | Same with Qwen-Image-2.1 (transparent PNGs when asked for) |
| `nu upscale.nu img.png` | AI upscale: SeedVR2 2x (faithful). `--scale 4`; `--creative` = FLUX 3 Klein redraw with new detail (`--denoise`, `--prompt`, `--megapixels`) |
| `nu detail.nu img.png [ref.png ...]` | Region detailer ("ADetailer"): SAM3 finds faces (`--detect "face:2, hand:4"`), or `--mask m.png` / `--box "x,y,w,h"`; each region is redone by Klein at ~1 MP, kept close to the reference images, and pasted back. `--denoise`, `--prompt` |
| `nu free.nu` | Unload models and free RAM (do this when switching Klein to Qwen) |
| `nu status.nu` | Installed models + sizes, repo versions, live RAM/VRAM |
| `nu update.nu` | Pull ComfyUI + custom nodes, refresh dependencies (stop ComfyUI first) |

## Workflows (ComfyUI sidebar > Workflows > tailored)

Every workflow has the same layout: notes on the far left, then the settings in two groups, **Main settings** and **Fine-tuning** (each node's title says what it does and what's sensible), then the pipeline left to right. The nodes those settings feed show the values greyed out; nothing needs changing there. No subgraphs: everything is visible.

- **Klein - Text to Image**: distilled INT8, 16 steps (template values). Prompt upsampler off by default.
- **Klein - Edit (multi-reference)**: write the instruction in *Prompt* (`<ref_image_0>`, `<obj_1>` for boxes). *Reference 2* is bypassed; Ctrl+B enables it. Prompt upsampler off by default.
- **Qwen - Text to Image (GGUF)**: Q4_K + W4A8 encoder, 25 steps, KV cache int8.
- **Qwen - Edit (GGUF)**: up to 10 references; output matches the first reference's size.
- **Qwen - Remove Background (GGUF)**: RGBA PNG cut-outs.

- **Upscale - SeedVR2 (faithful)**: restores detail at 2x (or 4x) and keeps the image as it is. Before/after slider included.
- **Upscale - Klein (creative)**: enlarges to ~4 MP, then Klein repaints at denoise 0.35 while looking at the original. Raise denoise for more invented detail, lower it to stay closer.

- **Detail - Klein (manual mask)**: right-click the image > *Open in MaskEditor*, paint what to redo. Each painted blob is cropped with context, enlarged to ~1 MP, repainted by Klein (denoise 0.4) and pasted back with a soft edge. *Reference 1* (`<ref_image_0>`) keeps faces/objects close to your reference; bypass it (Ctrl+B) if you have none.
- **Detail - Klein (auto detect)**: same, but SAM3 finds the regions from words: `face:4`, `face:2, hand:4`. *Detected mask* shows what it found.

Outputs land in `ComfyUI/output/klein/`, `ComfyUI/output/qwen/`, `ComfyUI/output/upscale/` and `ComfyUI/output/detail/`.

With 16 GB RAM, keep the upsampler/enhancer off unless you need it, and use one model family per session (or `free.nu` between them).

## Performance tuning

`launch.nu` adds two flags unless you pass `--stock`:

- `--use-ck-attention`: ComfyUI's built-in INT8 attention (RTX 20-series and newer). Speeds up sampling; quality is usually indistinguishable.
- `--cache-ram 3`: keep 3 GB of RAM free for Windows and the browser (default is 2 GB).

Already on by default, no flags needed: dynamic VRAM (makes `--lowvram` a no-op), async offload, the model compiler and CUDA graphs, and fast-disk streaming when models are on NVMe.

Compare tuned vs stock with the same seed (run each twice; the first run includes loading):

```nu
nu gen.nu klein "a red fox in snow" --seed 1      # look at "finished in Ns"
```

Worth trying if RAM is the bottleneck (Task Manager shows heavy paging or stutter):
`nu launch.nu --disable-pinned-memory`. On Windows ComfyUI may pin up to 40% of RAM (6.4 GB here) for faster transfers; pinned memory cannot be paged out.

Workflow side: Qwen references are fed at 1024 px (`resolution`), keep batch size 1, and leave the upsampler/enhancer off unless needed.
