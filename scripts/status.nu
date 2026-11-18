# status.nu - what is installed, how much space it takes, and live memory use.
#
#   nu status.nu                  models, versions, and server stats if running
#   nu status.nu --port 8189      if ComfyUI runs on another port

def main [
    --host: string = "127.0.0.1"
    --port: int = 8188
] {
    let root = $env.FILE_PWD
    let comfy = ($root | path join "ComfyUI")

    # ── models ───────────────────────────────────────────────────────────────
    # Relative glob after cd, so Windows backslashes never end up in the pattern.
    cd ($comfy | path join "models")
    let models = (ls **/*
        | where type == file and name =~ '\.(safetensors|gguf)$'
        | select name size
        | insert folder { |r| $r.name | path dirname }
        | update name { |r| $r.name | path basename }
        | select folder name size
        | sort-by folder name)
    print $"(ansi cyan_bold)Models(ansi reset)"
    if ($models | is-empty) {
        print "  none found"
    } else {
        print ($models | table --index false)
        print $"  total: ($models | get size | math sum)"
    }

    # ── versions ─────────────────────────────────────────────────────────────
    print $"\n(ansi cyan_bold)Repos(ansi reset)"
    let repos = ([$comfy] | append (ls ($comfy | path join "custom_nodes")
        | where type == dir
        | get name
        | where { |d| ($d | path join ".git" | path exists) }))
    print ($repos | each { |r| {
        repo: ($r | path basename)
        commit: (^git -C $r rev-parse --short HEAD | str trim)
        date: (^git -C $r log -1 --format=%cs | str trim)
    } } | table --index false)

    # ── live server stats ────────────────────────────────────────────────────
    print $"\n(ansi cyan_bold)Server(ansi reset)"
    let url = $"http://($host):($port)/system_stats"
    let stats = try { http get $url } catch { null }
    if $stats == null {
        print $"  not running at ($host):($port)"
        return
    }
    let sys = $stats.system
    print $"  ComfyUI ($sys.comfyui_version), PyTorch ($sys.pytorch_version)"
    print $"  RAM   free ($sys.ram_free | into filesize) of ($sys.ram_total | into filesize)"
    for dev in $stats.devices {
        print $"  VRAM  free ($dev.vram_free | into filesize) of ($dev.vram_total | into filesize)  [($dev.name)]"
    }
}
