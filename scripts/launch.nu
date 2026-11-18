# launch.nu - start ComfyUI for this install.
#
#   nu launch.nu                      Manager on, live previews, opens the browser
#   nu launch.nu --no-browser         do not open the browser (handled here)
#   nu launch.nu --lowvram            any other flag goes straight to ComfyUI
#   nu launch.nu --listen 0.0.0.0 --port 8189
#
# Defaults that suit a 12 GB card + 16 GB RAM are ComfyUI's own: dynamic VRAM
# loading and RAM-pressure caching are both on by default, so no extra flags.

def --wrapped main [...rest] {
    let root = $env.FILE_PWD
    let py = ($root | path join ".venv" "Scripts" "python.exe")
    let comfy = ($root | path join "ComfyUI")
    if not ($py | path exists) {
        error make { msg: $"No Python venv at ($py). Run install-comfyui.nu first." }
    }

    # --no-browser is ours; strip it before handing the rest to ComfyUI.
    let browser = if ("--no-browser" in $rest) { [] } else { ["--auto-launch"] }
    let passthru = ($rest | where $it != "--no-browser")

    cd $comfy
    ^$py main.py --enable-manager --preview-method auto ...$browser ...$passthru
}
