# free.nu - make a running ComfyUI unload its models and release memory.
#
#   nu free.nu
#
# Useful with 16 GB RAM when switching between FLUX 3 Klein and Qwen-Image:
# otherwise the previous model may stay cached and push Windows into the pagefile.
# The unload happens between jobs, so it is safe to call while something is queued.

def main [
    --host: string = "127.0.0.1"
    --port: int = 8188
] {
    let url = $"http://($host):($port)/free"
    try {
        http post --content-type application/json $url { unload_models: true, free_memory: true }
        print "ComfyUI will unload its models and free memory."
    } catch {
        print $"ComfyUI is not reachable at ($host):($port)."
    }
}
