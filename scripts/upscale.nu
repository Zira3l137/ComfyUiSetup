# upscale.nu - AI upscaling through a running ComfyUI (start it with launch.nu).
#
#   nu upscale.nu photo.png                       SeedVR2, 2x, stays faithful to the original
#   nu upscale.nu photo.png --scale 4             SeedVR2, 4x (slow for large inputs)
#   nu upscale.nu photo.png --creative            FLUX 3 Klein redraw to about 4 MP, adds new detail
#   nu upscale.nu photo.png --creative --denoise 0.45 --prompt "visible skin pores, sharp fabric weave"
#   nu upscale.nu photo.png --creative --seed 7 --out renders
#
# Two kinds of upscale:
#   SeedVR2 (default)  restores detail at the new size and keeps everything else as is.
#   --creative         enlarges, then FLUX 3 Klein repaints lightly while looking at the
#                      original. --denoise sets how much is repainted: 0.25 subtle,
#                      0.35 default, 0.5 strong. --prompt names the detail you want.
# Chaining works: a --creative pass first, then SeedVR2 on its result for more size.
#
# Prints the path of the saved image (ComfyUI/output/upscale/).

def main [
    image: path                       # Image to upscale
    --creative                        # Use the FLUX 3 Klein redraw instead of SeedVR2
    --scale: float = 2.0              # SeedVR2: upscale factor
    --model: string = "3b"            # SeedVR2: 3b or 7b, 7b needs the installer flag --seedvr2-7b
    --color: string = "lab"           # SeedVR2: colour match lab, wavelet, adain or none
    --megapixels: float = 4.0         # Creative: output size in megapixels
    --denoise: float = 0.35           # Creative: how much is repainted
    --prompt: string = ""             # Creative: detail to ask for, optional
    --seed: int                       # Fixed seed, random if omitted
    --out: path                       # Also copy the result into this folder
] {
    let root = $env.FILE_PWD
    let py = ($root | path join ".venv" "Scripts" "python.exe")
    let script = ($root | path join "tools" "comfy_gen.py")

    mut args = ["--comfy-root" ($root | path join "ComfyUI")]
    if $creative {
        # The prompt goes right after the mode name, where the generator expects it.
        $args = ($args | append ["klein-up"])
        if ($prompt | is-not-empty) { $args = ($args | append [$prompt]) }
        $args = ($args | append ["--ref" $image "--megapixels" $megapixels "--denoise" $denoise])
    } else {
        $args = ($args | append ["seedvr2" "--ref" $image "--scale" $scale "--seedvr2-model" $model "--color" $color])
    }
    if $seed != null { $args = ($args | append ["--seed" $seed]) }
    if $out != null { $args = ($args | append ["--out" $out]) }

    ^$py $script ...$args
}
