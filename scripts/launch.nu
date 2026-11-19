# launch.nu - start ComfyUI for this install.
#
#   nu launch.nu                      tuned settings, Manager on, opens the browser
#   nu launch.nu --stock              plain ComfyUI defaults (to compare or troubleshoot)
#   nu launch.nu --no-browser         do not open the browser
#   nu launch.nu --disable-pinned-memory   any other flag goes straight to ComfyUI
#
# --stock and --no-browser are handled here; everything else is passed through,
# after the tuned flags, so a flag you pass wins over the same one set here.
#
# What the tuning does (RTX 3060 12 GB, 16 GB RAM):
#   --use-ck-attention   Comfy Kitchen INT8 attention: a built-in SageAttention-style
#                        kernel for RTX 20-series and newer. Faster sampling, nothing
#                        to install. If ComfyUI exits with "Comfy Kitchen attention is
#                        unavailable", use --stock and report it.
#   --cache-ram 3        Keep 3 GB of RAM free: cached results of the current workflow
#                        are dropped below that (default 2 GB on 16 GB machines), so
#                        Windows and the browser are squeezed less.
#
# Already right by default, so not set here: dynamic VRAM, async offload, the
# Comfy model compiler and CUDA graphs, and fast-disk streaming (auto-enabled when
# the models sit on an NVMe drive). --lowvram does nothing while dynamic VRAM is on.

def --wrapped main [...rest] {
    let root = $env.FILE_PWD
    let py = ($root | path join ".venv" "Scripts" "python.exe")
    let comfy = ($root | path join "ComfyUI")
    if not ($py | path exists) {
        error make { msg: $"No Python venv at ($py). Run install-comfyui.nu first." }
    }

    let ours = ["--no-browser" "--stock"]
    let browser = if ("--no-browser" in $rest) { [] } else { ["--auto-launch"] }
    let passthru = ($rest | where $it not-in $ours)

    # ComfyUI allows only one flag from each of these groups, so a flag you pass
    # replaces the tuned one instead of clashing with it.
    let attn_flags = ["--use-split-cross-attention" "--use-quad-cross-attention" "--use-pytorch-cross-attention" "--use-sage-attention" "--use-flash-attention" "--use-ck-attention"]
    let cache_flags = ["--cache-ram" "--cache-classic" "--cache-lru" "--cache-none" "--high-ram"]
    let stock = ("--stock" in $rest)
    let attn = if $stock or ($passthru | any { |a| $a in $attn_flags }) { [] } else { ["--use-ck-attention"] }
    let cache = if $stock or ($passthru | any { |a| $a in $cache_flags }) { [] } else { ["--cache-ram" "3"] }
    let tuning = ($attn | append $cache)

    if ($tuning | is-empty) {
        print "Starting with stock ComfyUI settings."
    } else {
        print $"Starting with tuning: ($tuning | str join ' ')"
    }

    cd $comfy
    ^$py main.py --enable-manager --preview-method auto ...$tuning ...$browser ...$passthru
}
