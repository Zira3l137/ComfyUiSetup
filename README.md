# ComfyUI setup: FLUX 3 Klein + Qwen-Image-2.1 (RTX 3060 12 GB, 16 GB RAM)

## Install

1. Hugging Face token (needed once, for the gated FLUX 3 Klein files):
   - accept the license at https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy
   - create a **Read** token at https://huggingface.co/settings/tokens
   - in Nushell: `$env.HF_TOKEN = "hf_..."` (add it to `env.nu` to keep it; never commit it)
2. From this folder: `nu install-comfyui.nu C:\AI\ComfyUI`  (SSD; about 48 GB with all defaults)

## Day to day (run from the install folder)

| Script | What it does |
|---|---|
| `nu launch.nu` | Start ComfyUI (Manager on, previews on, opens the browser). Extra flags pass through: `--lowvram`, `--port 8189`, `--no-browser` |
| `nu gen.nu klein "prompt"` | Generate from the terminal. `--ref img.png` (repeatable) = edit mode, `--upsample` = prompt rewrite, `--seed`, `--width/--height`, `--out dir`. Run `nu gen.nu` for all options |
| `nu gen.nu qwen "prompt"` | Same with Qwen-Image-2.1 (transparent PNGs when asked for) |
| `nu free.nu` | Unload models and free RAM (do this when switching Klein to Qwen) |
| `nu status.nu` | Installed models + sizes, repo versions, live RAM/VRAM |
| `nu update.nu` | Pull ComfyUI + custom nodes, refresh dependencies (stop ComfyUI first) |

## Workflows (ComfyUI sidebar > Workflows > tailored)

- **Klein - Text to Image**: distilled INT8, 15 steps. Subgraph switch = upsampler (off by default).
- **Klein - Edit (multi-reference)**: write the instruction in *Flux3LayoutPrompt* (`<ref_image_0>`, `<obj_1>` for boxes). *Reference 2* is bypassed; Ctrl+B enables it. Upsampler switch: true = on.
- **Qwen - Text to Image (GGUF)**: Q4_K + W4A8 encoder, 25 steps, KV cache int8.
- **Qwen - Edit (GGUF)**: up to 10 references; output matches the first reference's size.
- **Qwen - Remove Background (GGUF)**: RGBA PNG cut-outs.

Outputs land in `ComfyUI/output/klein/` and `ComfyUI/output/qwen/`.

With 16 GB RAM, keep the upsampler/enhancer off unless you need it, and use one model family per session (or `free.nu` between them).
