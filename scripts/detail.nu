# detail.nu - ADetailer-style region repaint with FLUX 3 Klein (start ComfyUI with launch.nu first).
#
#   nu detail.nu shot.png                                   find faces (SAM3) and redo them at ~1 MP each
#   nu detail.nu shot.png face.png                          ...and keep them close to the face in face.png
#   nu detail.nu shot.png face.png outfit.png               more references: every extra image counts
#   nu detail.nu shot.png --detect "face:2, hand:4"         up to 2 faces and 4 hands
#   nu detail.nu shot.png --mask hilt_mask.png --prompt "an ornate steel sword hilt wrapped in leather"
#   nu detail.nu shot.png --box "820,140,180,220"           a rectangle: x,y,width,height in pixels
#   nu detail.nu shot.png --box "820,140,180,220 60,600,120,120"   several rectangles
#   nu detail.nu shot.png face.png --denoise 0.55 --seed 7 --out renders
#
# Each region is cropped with context around it, enlarged so its short side is --size
# px, repainted by Klein at --denoise, scaled back down and pasted in with a soft edge.
# Reference images (every image after the first) go to Klein for each region, so a
# face stays the face in the reference instead of drifting.
#
# Where to work: SAM3 detection by default (--detect, plain words, ":N" = up to N),
# or --mask (white = repaint, same size as the image), or --box.
# --denoise: 0.3 subtle, 0.4 default, 0.5-0.6 redraws broken hands/faces but may drift.
#
# Prints the path of the saved image (ComfyUI/output/detail/). Nothing detected =
# the image comes back unchanged.

def main [
    image: path                       # Image to fix
    ...refs: path                     # Reference images (identity, outfit, object), optional
    --detect: string = "face:4"       # What SAM3 should find, e.g. "face:2, hand:4"
    --mask: path                      # Mask image, white = repaint (instead of --detect)
    --box: string                     # "x,y,w,h" rectangles separated by spaces (instead of --detect)
    --prompt: string = ""             # What the regions should look like; default prompt if empty
    --denoise: float = 0.4            # How much is repainted
    --threshold: float = 0.5          # SAM3 confidence: higher = fewer, surer detections
    --size: int = 1024                # Short side each region is enlarged to
    --crop-factor: float = 2.5        # Context around each region, as a multiple of its size
    --grow: int                       # Grow the mask by N px (default 16 for --detect, 8 otherwise)
    --steps: int = 16                 # Sampling steps
    --seed: int                       # Fixed seed, random if omitted
    --out: path                       # Also copy the result into this folder
] {
    let root = $env.FILE_PWD
    let py = ($root | path join ".venv" "Scripts" "python.exe")
    let script = ($root | path join "tools" "comfy_gen.py")

    if $mask != null and $box != null {
        error make { msg: "use either --mask or --box, not both" }
    }

    # The prompt goes right after the mode name, where the generator expects it.
    # list<any>: the numbers go to the external as they are (a list<string> would reject them).
    mut args: list<any> = ["--comfy-root" ($root | path join "ComfyUI") "detail"]
    if ($prompt | is-not-empty) { $args = ($args | append [$prompt]) }
    $args = ($args | append [
        "--image" $image "--denoise" $denoise "--threshold" $threshold
        "--size" $size "--crop-factor" $crop_factor "--steps" $steps
    ])
    for r in $refs { $args = ($args | append ["--ref" $r]) }

    if $mask != null {
        $args = ($args | append ["--mask" $mask])
    } else if $box != null {
        for b in ($box | split row --regex '[\s;]+' | where { |s| $s | is-not-empty }) {
            $args = ($args | append ["--box" $b])
        }
    } else {
        $args = ($args | append ["--detect" $detect])
    }
    if $grow != null { $args = ($args | append ["--grow" $grow]) }
    if $seed != null { $args = ($args | append ["--seed" $seed]) }
    if $out != null { $args = ($args | append ["--out" $out]) }

    ^$py $script ...$args
}
