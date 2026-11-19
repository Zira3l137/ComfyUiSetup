# upscale.nu - AI upscaling through a running ComfyUI (start it with launch.nu).
#
#   nu upscale.nu photo.png                       SeedVR2, 2x, stays faithful to the original
#   nu upscale.nu photo.png --scale 4             SeedVR2, 4x (slow for large inputs)
#   nu upscale.nu photo.png --creative            SeedVR2 to 2x, then FLUX 3 Klein repaints: new detail
#   nu upscale.nu photo.png --creative --no-original-ref --denoise 0.5   even more new detail (may drift)
#   nu upscale.nu photo.png --creative --scale 1.5 --denoise 0.45 --prompt "visible skin pores, sharp fabric weave"
#   nu upscale.nu photo.png face.png --creative   extra reference(s) keep the face from drifting
#   nu upscale.nu photo.png --creative --seed 7 --out renders
#
# Two kinds of upscale:
#   SeedVR2 (default)  restores detail at the new size and keeps everything else as is.
#   --creative         "hi-res fix": SeedVR2 enlarges and restores detail first (--soft
#                      uses a plain lanczos stretch instead), then FLUX 3 Klein repaints
#                      while looking at the original. --denoise sets how much is repainted:
#                      0.3 subtle, 0.4 default, 0.55 strong. --no-original-ref stops it
#                      looking at the original: more new detail, more drift.
#                      --prompt replaces the whole prompt (mention <ref_image_0> for the
#                      original yourself).
#                      Output = original x --scale, rounded down to a multiple of 16 px;
#                      keep it around 4 MP or less on 12 GB (SeedVR2 for anything bigger).
#                      Images after the first are extra references (a clean face close-up,
#                      an object) that Klein matches, so the subject doesn't drift.
# Chaining works: a --creative pass first, then SeedVR2 on its result for more size.
#
# Prints the path of the saved image (ComfyUI/output/upscale/).

def main [
    image: path                       # Image to upscale
    ...refs: path                     # Creative only: extra reference images (face, object)
    --creative                        # Use the FLUX 3 Klein redraw instead of SeedVR2
    --soft                            # Creative: plain lanczos first pass instead of SeedVR2
    --no-original-ref                 # Creative: don't feed the original as a reference
    --scale: float = 2.0              # Output = original size x this
    --model: string = "3b"            # SeedVR2 (also creative's first pass): 3b or 7b (installer --seedvr2-7b)
    --color: string = "lab"           # SeedVR2 (also creative's first pass): lab, wavelet, adain or none
    --megapixels: float               # Creative: output size in megapixels instead of --scale
    --denoise: float = 0.4            # Creative: how much is repainted
    --prompt: string = ""             # Creative: detail to ask for, optional
    --seed: int                       # Fixed seed, random if omitted
    --out: path                       # Also copy the result into this folder
] {
    let root = $env.FILE_PWD
    let py = ($root | path join ".venv" "Scripts" "python.exe")
    let script = ($root | path join "tools" "comfy_gen.py")

    # list<any>: the numbers go to the external as they are (a list<string> would reject them).
    mut args: list<any> = ["--comfy-root" ($root | path join "ComfyUI")]
    if $creative {
        # The prompt goes right after the mode name, where the generator expects it.
        $args = ($args | append ["klein-up"])
        if ($prompt | is-not-empty) { $args = ($args | append [$prompt]) }
        $args = ($args | append ["--ref" $image "--scale" $scale "--denoise" $denoise])
        for r in $refs { $args = ($args | append ["--ref" $r]) }
        if $megapixels != null { $args = ($args | append ["--megapixels" $megapixels]) }
        if $soft { $args = ($args | append ["--first-pass" "lanczos"]) }
        if $no_original_ref { $args = ($args | append ["--no-original-ref"]) }
        # SeedVR2 first-pass settings apply to the creative mode too
        $args = ($args | append ["--seedvr2-model" $model "--color" $color])
    } else {
        if ($refs | is-not-empty) {
            error make { msg: "extra reference images only work with --creative" }
        }
        $args = ($args | append ["seedvr2" "--ref" $image "--scale" $scale "--seedvr2-model" $model "--color" $color])
    }
    if $seed != null { $args = ($args | append ["--seed" $seed]) }
    if $out != null { $args = ($args | append ["--out" $out]) }

    ^$py $script ...$args
}
