# gen.nu - generate or edit images from the terminal through a running ComfyUI.
#
#   nu gen.nu klein "a red fox in fresh snow, golden hour"
#   nu gen.nu klein "make it night time" --ref photo.png
#   nu gen.nu klein "the cat from image 1 wearing the hat from image 2" --ref cat.png --ref hat.png
#   nu gen.nu qwen "a game-ready rusty sword icon on a transparent background"
#   nu gen.nu qwen "replace the jacket with the one in image 2" --ref model.png --ref jacket.png
#   nu gen.nu klein "castle" --upsample --seed 42 --width 1344 --height 768 --out renders
#   nu gen.nu klein "castle" --print-graph | save castle.json   # API graph; drag into ComfyUI
#   nu gen.nu                                                    # full option list
#
# --ref switches to edit mode (repeat it for more references: Klein and Qwen
# both take up to 10). --upsample runs the model's prompt enhancer first, the
# ChatGPT-style rewrite; it loads a 6.7 GB model, so expect it to be slower.
# Prints one path per saved image, so it pipes:  nu gen.nu klein "x" | lines | each { start $in }
#
# Settings match the official templates; ComfyUI must already be running (launch.nu).

def --wrapped main [...args] {
    let root = $env.FILE_PWD
    let py = ($root | path join ".venv" "Scripts" "python.exe")
    let script = ($root | path join "tools" "comfy_gen.py")
    if ($args | is-empty) {
        ^$py $script --help
        return
    }
    ^$py $script --comfy-root ($root | path join "ComfyUI") ...$args
}
