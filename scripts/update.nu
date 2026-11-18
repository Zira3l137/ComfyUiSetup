# update.nu - update ComfyUI, its custom nodes and their Python dependencies.
#
#   nu update.nu              pull everything, reinstall requirements, check CUDA
#   nu update.nu --no-deps    only git pull (fast; use when nothing new is needed)
#
# Stop ComfyUI before running this: Windows locks files that are in use.
# PyTorch is never touched here: requirements.txt lists torch without a version,
# so the CUDA build from the installer stays in place.

def main [
    --no-deps   # Skip the pip installs, only pull the repos
] {
    let root = $env.FILE_PWD
    let py = ($root | path join ".venv" "Scripts" "python.exe")
    let comfy = ($root | path join "ComfyUI")
    let nodes_dir = ($comfy | path join "custom_nodes")

    print $"(ansi cyan_bold)==> ComfyUI(ansi reset)"
    ^git -C $comfy pull --ff-only

    # Every custom node that is a git checkout gets pulled, including ones
    # installed later through ComfyUI-Manager.
    let node_repos = (ls $nodes_dir
        | where type == dir
        | get name
        | where { |d| ($d | path join ".git" | path exists) })
    for repo in $node_repos {
        print $"(ansi cyan_bold)==> ($repo | path basename)(ansi reset)"
        ^git -C $repo pull --ff-only
    }

    if $no_deps { return }

    print $"(ansi cyan_bold)==> Python requirements(ansi reset)"
    let req_files = ([
        ($comfy | path join "requirements.txt")
        ($comfy | path join "manager_requirements.txt")
    ] | append ($node_repos | each { |d| $d | path join "requirements.txt" })
      | where { |f| $f | path exists })
    for req in $req_files {
        print $"  ($req)"
        ^uv pip install --python $py -r $req
    }

    print $"(ansi cyan_bold)==> CUDA check(ansi reset)"
    ^$py -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('  torch', torch.__version__, '|', torch.cuda.get_device_name(0))"
}
