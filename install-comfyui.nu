# install-comfyui.nu
# ──────────────────────────────────────────────────────────────────────────────
# Installs ComfyUI + ComfyUI-Manager + a GGUF loader, downloads the models, and
# copies the tailored workflows and helper scripts from this folder:
#   • FLUX 3 Image Klein (INT8, distilled)  - everyday generation / editing / multi-ref
#   • FLUX 3 Klein prompt upsampler (W4A8)  - the "ChatGPT-style" prompt rewriter
#   • Qwen-Image-2.1 (GGUF Q4_K)            - transparent (RGBA) output, mask edits
#   • Qwen-Image-2.1 prompt enhancers (W4A8, t2i + edit)
#   • SeedVR2 3B (INT8)                     - faithful AI upscaling
#
# Target: Windows 11, RTX 3060 12 GB, 16 GB RAM, Nushell 0.114, uv, git.
#
# Usage (run from the folder this file is in, next to scripts/, tools/, workflows/):
#   nu install-comfyui.nu C:\AI\ComfyUI                    # everything, defaults
#   nu install-comfyui.nu C:\AI\ComfyUI --no-qwen          # FLUX 3 Klein only
#   nu install-comfyui.nu C:\AI\ComfyUI --no-upsampler     # skip the 6.7 GB Klein upsampler
#   nu install-comfyui.nu C:\AI\ComfyUI --no-qwen-enhancer # skip the 2 x 6.7 GB Qwen enhancers
#   nu install-comfyui.nu C:\AI\ComfyUI --seedvr2-7b       # also the 8.3 GB SeedVR2 7B upscaler
#   nu install-comfyui.nu --help                            # list all options
#
# Re-running is safe: repos are fast-forwarded (git pull), downloaded model
# files are skipped, and scripts/workflows are refreshed.
#
# Requires the HF_TOKEN environment variable (a Hugging Face "Read" token),
# because the FLUX 3 Klein repo from Black Forest Labs is gated.
# ──────────────────────────────────────────────────────────────────────────────

# Hugging Face repos. The Comfy-packaged repos use the same folder layout as
# ComfyUI/models (diffusion_models/, text_encoders/, vae/), so files download
# straight into the models folder.
const KLEIN_REPO = "black-forest-labs/flux-3-image-klein-comfy"   # gated: accept license on HF first
const QWEN_REPO  = "Comfy-Org/Qwen-Image-2.1"
const QWEN_GGUF_REPO = "leejet/Qwen-Image-2.1-GGUF"
const SEEDVR2_REPO = "Comfy-Org/SeedVR2"

# The leejet fork of ComfyUI-GGUF: its loader lists the qwen_image21 architecture.
# The original city96 node is reported to fail on these files with
# "Unknown model architecture!", and the GGUF model card points to this fork.
const GGUF_NODE_REPO = "https://github.com/leejet/ComfyUI-GGUF"
const COMFY_REPO     = "https://github.com/Comfy-Org/ComfyUI"
const DEFAULT_GGUF   = "qwen_image_2.1-Q4_K.gguf"   # what the bundled workflows reference

# ── helpers ───────────────────────────────────────────────────────────────────

# Print a section header so the long run is easy to follow.
def step [msg: string] {
    print $"\n(ansi cyan_bold)==> ($msg)(ansi reset)"
}

# Clone a git repo, or fast-forward it if it is already there.
def git-sync [url: string, dest: path] {
    if ($dest | path join ".git" | path exists) {
        print $"  updating ($dest)"
        ^git -C $dest pull --ff-only
    } else {
        print $"  cloning ($url)"
        ^git clone --depth 1 $url $dest
    }
}

# Download one file from a Hugging Face repo into `dest`.
# `file` is the path inside the repo and keeps that relative path under `dest`:
# "vae/x.safetensors" with dest = models/ ends up at models/vae/x.safetensors.
# Uses the official `hf` CLI through uvx (nothing installed globally); it resumes
# partial downloads and reads HF_TOKEN from the environment.
def hf-get [repo: string, file: string, dest: path] {
    let target = ($dest | path join $file)
    if ($target | path exists) {
        print $"  (ansi green)✓(ansi reset) ($file) already present, skipping"
        return
    }
    print $"  (ansi yellow)↓(ansi reset) ($repo) :: ($file)"
    ^uvx --from huggingface_hub hf download $repo $file --local-dir $dest
}

# ── main ──────────────────────────────────────────────────────────────────────

# Install ComfyUI with FLUX 3 Image Klein and Qwen-Image-2.1 for a 12 GB NVIDIA card.
def main [
    install_dir: path               # Root folder, gets ComfyUI, .venv, tools and the scripts
    --python: string = "3.13"       # Python for the venv. ComfyUI recommends 3.13; uv downloads it if missing
    --torch-index: string = "cu130" # PyTorch CUDA build. ComfyUI README: RTX 20-series and newer need cu130+
    --no-upsampler                  # Skip the 6.7 GB FLUX 3 Klein prompt upsampler
    --no-qwen                       # Skip Qwen-Image-2.1 entirely
    --no-qwen-enhancer              # Skip the two 6.7 GB Qwen prompt enhancers
    --no-seedvr2                    # Skip the SeedVR2 upscaler
    --seedvr2-7b                    # Also fetch the bigger 8.3 GB SeedVR2 7B model
    --qwen-format: string = "gguf"  # "gguf" (bundled workflows use it) or "int8" (official file, no custom node)
    --qwen-quant: string = "Q4_K"   # GGUF quant from the leejet repo: Q2_K Q3_K Q4_0 Q4_K Q5_0 Q6_K Q8_0
] {
    let pkg = $env.FILE_PWD     # the folder this installer was run from

    # ── 0. sanity checks ──────────────────────────────────────────────────────
    step "Checking prerequisites"

    for tool in [git uv uvx] {
        if (which $tool | is-empty) {
            error make { msg: $"`($tool)` not found on PATH" }
        }
    }
    for sub in [scripts tools workflows] {
        if not ($pkg | path join $sub | path exists) {
            error make { msg: $"Missing ($pkg | path join $sub). Keep install-comfyui.nu next to scripts, tools and workflows." }
        }
    }

    if $qwen_format not-in [gguf int8] {
        error make { msg: $"--qwen-format must be 'gguf' or 'int8', got '($qwen_format)'" }
    }
    let quants = [Q2_K Q3_K Q4_0 Q4_K Q5_0 Q6_K Q8_0]
    if $qwen_quant not-in $quants {
        error make { msg: $"--qwen-quant must be one of ($quants | str join ', ')" }
    }

    # Fail early instead of after a 15-minute PyTorch install.
    if ($env.HF_TOKEN? | is-empty) {
        error make {
            msg: "HF_TOKEN is not set. Create a Read token at https://huggingface.co/settings/tokens, accept the license at https://huggingface.co/black-forest-labs/flux-3-image-klein-comfy, then: $env.HF_TOKEN = 'hf_...'"
        }
    }

    # Rough disk budget, so you know before the downloads start.
    let klein_gb = if $no_upsampler { 11.0 } else { 17.7 }
    let qwen_base_gb = if $qwen_format == "int8" { 14.3 } else { 11.2 }
    let qwen_gb = if $no_qwen { 0.0 } else if $no_qwen_enhancer { $qwen_base_gb } else { $qwen_base_gb + 13.4 }
    let seedvr2_gb = if $no_seedvr2 { 0.0 } else if $seedvr2_7b { 12.3 } else { 4.0 }
    let total_gb = $klein_gb + $qwen_gb + $seedvr2_gb + 6.0   # + about 6 GB for ComfyUI and the venv
    print $"  install dir : ($install_dir)"
    print $"  disk needed : about ($total_gb | math round --precision 1) GB, plus about 3 GB in the uv cache on drive C"

    let comfy = ($install_dir | path join "ComfyUI")
    let venv  = ($install_dir | path join ".venv")
    let py    = ($venv | path join "Scripts" "python.exe")
    let models = ($comfy | path join "models")
    let gguf_name = $"qwen_image_2.1-($qwen_quant).gguf"

    mkdir $install_dir

    # ── 1. ComfyUI source ─────────────────────────────────────────────────────
    step "Fetching ComfyUI"
    git-sync $COMFY_REPO $comfy

    # ── 2. Python environment ────────────────────────────────────────────────
    # The venv sits next to the repo, not inside it, so `git pull` never touches it.
    step $"Creating Python ($python) venv"
    if not ($py | path exists) {
        ^uv venv $venv --python $python
    } else {
        print "  venv exists, reusing"
    }

    # PyTorch FIRST, from the CUDA index. requirements.txt lists `torch` without a
    # version, so uv then sees it as satisfied and keeps this CUDA build.
    step $"Installing PyTorch for ($torch_index)"
    ^uv pip install --python $py torch torchvision torchaudio --index-url $"https://download.pytorch.org/whl/($torch_index)"

    step "Installing ComfyUI requirements"
    ^uv pip install --python $py -r ($comfy | path join "requirements.txt")

    # ComfyUI-Manager ships as a pip package now, switched on at launch with
    # --enable-manager (launch.nu does that).
    let mgr_req = ($comfy | path join "manager_requirements.txt")
    if ($mgr_req | path exists) {
        step "Installing ComfyUI-Manager"
        ^uv pip install --python $py -r $mgr_req
    }

    # Confirm the GPU is visible before downloading tens of GB of weights.
    step "Verifying CUDA"
    ^$py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('  torch', torch.__version__, '|', torch.cuda.get_device_name(0))"

    # ── 3. GGUF loader node (only for Qwen GGUF) ─────────────────────────────
    if (not $no_qwen) and $qwen_format == "gguf" {
        step "Installing ComfyUI-GGUF (leejet fork)"
        let node = ($comfy | path join "custom_nodes" "ComfyUI-GGUF")
        # Another fork in the same folder would shadow this one, so stop if found.
        if ($node | path join ".git" | path exists) {
            let origin = (^git -C $node remote get-url origin | str trim)
            if not ($origin | str contains "leejet") {
                error make { msg: $"($node) is a different fork: ($origin). Delete that folder and re-run." }
            }
        }
        git-sync $GGUF_NODE_REPO $node
        let node_req = ($node | path join "requirements.txt")
        if ($node_req | path exists) { ^uv pip install --python $py -r $node_req }
        ^uv pip install --python $py --upgrade gguf
    }

    # ── 4. FLUX 3 Image Klein ────────────────────────────────────────────────
    # The distilled checkpoint (15-16 steps). The "base" one needs about 50 steps.
    step "Downloading FLUX 3 Image Klein (INT8, distilled)"
    hf-get $KLEIN_REPO "diffusion_models/flux3-image-klein-int8-convrot.safetensors" $models  # 5.8 GB
    hf-get $KLEIN_REPO "text_encoders/qwen3vl_4b_int8_convrot.safetensors" $models            # 4.9 GB
    hf-get $KLEIN_REPO "vae/flux2-vae.safetensors" $models                                     # 0.3 GB
    if not $no_upsampler {
        # W4A8 rather than the 9.7 GB INT8 build: smaller, and the INT8 one was much
        # slower on low-VRAM cards in testing.
        hf-get $KLEIN_REPO "text_encoders/qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors" $models  # 6.7 GB
    }

    # ── 5. Qwen-Image-2.1 ────────────────────────────────────────────────────
    if not $no_qwen {
        step $"Downloading Qwen-Image-2.1 as ($qwen_format)"
        # Qwen3-VL-8B text encoder (W4A8, the low-RAM one) + VAE, used by both formats.
        hf-get $QWEN_REPO "text_encoders/qwen3vl_8b_w4a8.safetensors" $models   # 6.3 GB
        hf-get $QWEN_REPO "vae/qwen_image_2.1_vae_bf16.safetensors" $models     # 0.7 GB

        if $qwen_format == "gguf" {
            # GGUF image models go in models/unet, where "Unet Loader (GGUF)" looks.
            hf-get $QWEN_GGUF_REPO $gguf_name ($models | path join "unet")
        } else {
            hf-get $QWEN_REPO "diffusion_models/qwen_image_2.1_int8_convrot.safetensors" $models  # 7.3 GB
        }

        if not $no_qwen_enhancer {
            # The Qwen workflows wire these in behind a switch (off by default).
            # ComfyUI checks every connected loader before a run, so the files must
            # exist even while the switch is off.
            hf-get $QWEN_REPO "text_encoders/qwen3.5_9b_qwen_image_2.1_pe_t2i_w4a8.safetensors" $models  # 6.7 GB
            hf-get $QWEN_REPO "text_encoders/qwen3.5_9b_qwen_image_2.1_pe_i2i_w4a8.safetensors" $models  # 6.7 GB
        }
    }

    # ── 5b. SeedVR2 upscaler ─────────────────────────────────────────────────
    if not $no_seedvr2 {
        step "Downloading SeedVR2 (INT8)"
        hf-get $SEEDVR2_REPO "diffusion_models/seedvr2_3b_int8_convrot.safetensors" $models  # 3.5 GB
        hf-get $SEEDVR2_REPO "vae/seedvr2_ema_vae_fp16.safetensors" $models                   # 0.5 GB
        if $seedvr2_7b {
            hf-get $SEEDVR2_REPO "diffusion_models/seedvr2_7b_int8_convrot.safetensors" $models  # 8.3 GB
        }
    }

    # ── 6. helper scripts and the generator ──────────────────────────────────
    step "Copying helper scripts"
    for f in (ls ($pkg | path join "scripts") | get name) {
        cp $f $install_dir
        print $"  ($f | path basename)"
    }
    let tools_dir = ($install_dir | path join "tools")
    mkdir $tools_dir
    for f in (ls ($pkg | path join "tools") | where type == file | get name) {
        cp $f $tools_dir
    }
    # Tell the generator which GGUF quant was downloaded (only if not the default).
    let cfg = ($tools_dir | path join "config.json")
    if $gguf_name != $DEFAULT_GGUF {
        { qwen_gguf: $gguf_name } | to json | save --force $cfg
    } else if ($cfg | path exists) {
        rm $cfg
    }

    # ── 7. workflows ─────────────────────────────────────────────────────────
    # They appear in the ComfyUI sidebar under Workflows > tailored.
    step "Copying workflows"
    let wf_dest = ($comfy | path join "user" "default" "workflows" "tailored")
    mkdir $wf_dest
    for f in (ls ($pkg | path join "workflows") | where type == file | get name) {
        let name = ($f | path basename)
        if $no_qwen and ($name | str starts-with "Qwen") { continue }
        if $no_seedvr2 and ($name | str contains "SeedVR2") { continue }
        let dest = ($wf_dest | path join $name)
        if $gguf_name == $DEFAULT_GGUF {
            cp $f $dest
        } else {
            # Point the Qwen workflows at the quant that was actually downloaded.
            open $f | to json | str replace --all $DEFAULT_GGUF $gguf_name | save --force $dest
        }
        print $"  ($name)"
    }
    if (not $no_qwen) and $qwen_format == "int8" {
        print $"  (ansi yellow)note(ansi reset): the Qwen workflows use the GGUF loader. With int8, swap their"
        print "  'Unet Loader (GGUF)' for 'Load Diffusion Model', or use: nu gen.nu qwen ... --qwen-int8"
    }
    if (not $no_qwen) and $no_qwen_enhancer {
        print $"  (ansi yellow)note(ansi reset): without the Qwen enhancers, the Qwen text-to-image and edit"
        print "  workflows will report a missing PE model. Delete that loader node, or re-run without the flag."
    }

    step "Done"
    print $"  Start:   nu '($install_dir | path join launch.nu)'"
    print "  Then:    Workflows sidebar > tailored, or from a terminal: nu gen.nu klein \"a red fox in snow\""
    print "  Upscale: nu upscale.nu photo.png   (add --creative for the FLUX 3 Klein redraw)"
    print "  Also:    status.nu (models, memory), free.nu (unload models), update.nu (pull + deps)"
}
