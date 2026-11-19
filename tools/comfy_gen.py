"""Command-line image generation through a running ComfyUI (called by gen.nu).

Builds an API-format prompt (FLUX 3 Image Klein, Qwen-Image-2.1, SeedVR2, or the Klein
region detailer), queues it on the local server, waits for it to finish and prints the
paths of the saved images.
Settings mirror the official ComfyUI templates. Standard library only.

  python comfy_gen.py --comfy-root C:/AI/ComfyUI/ComfyUI klein "a red fox in snow"
  python comfy_gen.py ... qwen "replace the jacket with the one in image 2" --ref a.png --ref b.png
  python comfy_gen.py ... detail --image shot.png --detect "face:2" --ref face.png
"""
import argparse
import json
import random
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYSTEM_PROMPTS = json.loads((HERE / "system_prompts.json").read_text(encoding="utf-8"))

KLEIN = {
    "unet": "flux3-image-klein-int8-convrot.safetensors",
    "te": "qwen3vl_4b_int8_convrot.safetensors",
    "vae": "flux2-vae.safetensors",
    "pup": "qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors",
}
QWEN = {
    "gguf": "qwen_image_2.1-Q4_K.gguf",
    "int8": "qwen_image_2.1_int8_convrot.safetensors",
    "te": "qwen3vl_8b_w4a8.safetensors",
    "vae": "qwen_image_2.1_vae_bf16.safetensors",
    "pe_t2i": "qwen3.5_9b_qwen_image_2.1_pe_t2i_w4a8.safetensors",
    "pe_i2i": "qwen3.5_9b_qwen_image_2.1_pe_i2i_w4a8.safetensors",
}
SEEDVR2 = {
    "3b": "seedvr2_3b_int8_convrot.safetensors",
    "7b": "seedvr2_7b_int8_convrot.safetensors",
    "vae": "seedvr2_ema_vae_fp16.safetensors",
}
# The installer writes config.json when a non-default GGUF quant was chosen.
_cfg = HERE / "config.json"
if _cfg.is_file():
    QWEN["gguf"] = json.loads(_cfg.read_text(encoding="utf-8")).get("qwen_gguf", QWEN["gguf"])


def text_generate(clip, prompt, seed, sys_prompt_node, *, max_length, top_k, rep_pen, presence, mtp, image=None):
    """TextGenerate inputs as the templates set them (dynamic-combo keys are dotted)."""
    inputs = {
        "clip": clip, "prompt": prompt, "max_length": max_length,
        "sampling_mode": "on",
        "sampling_mode.temperature": 1.0, "sampling_mode.top_k": top_k, "sampling_mode.top_p": 0.95,
        "sampling_mode.min_p": 0.0, "sampling_mode.repetition_penalty": rep_pen,
        "sampling_mode.seed": seed, "sampling_mode.presence_penalty": presence,
        "thinking": False, "use_default_template": True, "mtp": mtp,
        "system_prompt": [sys_prompt_node, 0],
    }
    if image is not None:
        inputs["image"] = image
    return {"class_type": "TextGenerate", "inputs": inputs}


def build_klein(a, refs):
    """FLUX 3 Image Klein, distilled. t2i: 15 steps er_sde; edit: 16 steps euler (template values)."""
    edit = bool(refs)
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": KLEIN["unet"], "weight_dtype": "default"}},
        "te": {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN["te"], "type": "flux3", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN["vae"]}},
    }
    for i, name in enumerate(refs):
        g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}

    text = a.prompt
    if a.upsample:
        g["pup_te"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN["pup"], "type": "flux3", "device": "default"}}
        g["pup_sys"] = {"class_type": "PrimitiveStringMultiline",
                        "inputs": {"value": SYSTEM_PROMPTS["klein_edit" if edit else "klein_t2i"]}}
        image = None
        if edit:
            g["pup_imgs"] = {"class_type": "BatchImagesNode",
                             "inputs": {f"images.image{i}": [f"ref{i}", 0] for i in range(len(refs))}}
            image = ["pup_imgs", 0]
        g["pup"] = text_generate(["pup_te", 0], a.prompt, a.seed, "pup_sys", max_length=3584, top_k=1,
                                 rep_pen=1.05, presence=0.0 if edit else 1.5, mtp="off", image=image)
        text = ["pup", 0]

    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": text}}
    if edit:
        g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
        rml = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
               "resolution": "resize", "resolution.megapixels": a.ref_mp}
        rml.update({f"images.image{i}": [f"ref{i}", 0] for i in range(len(refs))})
        g["refs"] = {"class_type": "ReferenceMultiLatent", "inputs": rml}
        pos, neg, latent = ["refs", 0], ["refs", 1], ["refs", 2]
        steps, sampler = a.steps or 16, "euler"
    else:
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": ""}}
        pos, neg, latent = ["pos", 0], ["neg", 0], None
        steps, sampler = a.steps or 15, "er_sde"
    if latent is None or a.width:
        g["latent"] = {"class_type": "EmptyFlux2LatentImage",
                       "inputs": {"width": a.width or 1024, "height": a.height or a.width or 1024, "batch_size": 1}}
        latent = ["latent", 0]
    g["sample"] = {"class_type": "KSampler", "inputs": {
        "model": ["unet", 0], "seed": a.seed, "steps": steps, "cfg": 1.0, "sampler_name": sampler,
        "scheduler": "simple", "positive": pos, "negative": neg, "latent_image": latent, "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["decode", 0], "filename_prefix": f"klein/{'edit' if edit else 't2i'}"}}
    return g


def build_qwen(a, refs):
    """Qwen-Image-2.1 (GGUF by default). 25 steps euler/simple, cfg 1 (template values)."""
    edit = bool(refs)
    if a.qwen_int8:
        unet = {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN["int8"], "weight_dtype": "default"}}
    else:
        unet = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": QWEN["gguf"]}}
    g = {
        "unet": unet,
        # KV cache at int8: half the memory at about bf16 accuracy (node tooltip).
        "cache": {"class_type": "QwenImage21Cache", "inputs": {"model": ["unet", 0], "device": "auto", "dtype": "int8"}},
        "te": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN["te"], "type": "qwen_image", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN["vae"]}},
    }
    for i, name in enumerate(refs):
        g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}

    text = a.prompt
    if a.upsample:
        g["pe_te"] = {"class_type": "CLIPLoader", "inputs": {
            "clip_name": QWEN["pe_i2i" if edit else "pe_t2i"],
            "type": "qwen_image" if edit else "stable_diffusion", "device": "default"}}
        g["pe_sys"] = {"class_type": "PrimitiveStringMultiline",
                       "inputs": {"value": SYSTEM_PROMPTS["qwen_edit" if edit else "qwen_t2i"]}}
        image = None
        if edit:
            g["pe_imgs"] = {"class_type": "BatchImagesNode",
                            "inputs": {f"images.image{i}": [f"ref{i}", 0] for i in range(len(refs))}}
            image = ["pe_imgs", 0]
        g["pe"] = text_generate(["pe_te", 0], a.prompt, a.seed, "pe_sys", max_length=4096, top_k=20,
                                rep_pen=1.0, presence=0.0 if edit else 1.5, mtp="auto", image=image)
        text = ["pe", 0]

    enc = {"clip": ["te", 0], "prompt": text, "negative_prompt": a.negative,
           "resolution": a.ref_res if edit else 1024}
    if edit:
        enc["vae"] = ["vae", 0]
        enc.update({f"images.image_{i + 1}": [f"ref{i}", 0] for i in range(len(refs))})
    g["enc"] = {"class_type": "TextEncodeQwenImage21", "inputs": enc}

    if edit and not a.width:
        latent = ["enc", 2]       # matches the first reference's size, so the edit doesn't shift
    else:
        g["latent"] = {"class_type": "EmptyLatentImage",
                       "inputs": {"width": a.width or 1024, "height": a.height or a.width or 1024, "batch_size": 1}}
        latent = ["latent", 0]
    g["sample"] = {"class_type": "KSampler", "inputs": {
        "model": ["cache", 0], "seed": a.seed, "steps": a.steps or 25, "cfg": 1.0, "sampler_name": "euler",
        "scheduler": "simple", "positive": ["enc", 0], "negative": ["enc", 1], "latent_image": latent, "denoise": 1.0}}
    g["decode"] = {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}}
    # Save Image Advanced keeps the alpha channel ("auto"), so transparent results stay transparent.
    g["save"] = {"class_type": "SaveImageAdvanced", "inputs": {
        "images": ["decode", 0], "filename_prefix": f"qwen/{'edit' if edit else 't2i'}",
        "format": "png", "format.bit_depth": "8-bit", "format.input_color_space": "sRGB"}}
    return g


def build_seedvr2(a, refs):
    """Faithful upscale with SeedVR2: one diffusion step that restores detail at the new size.
    Mirrors the official SeedVR2 template (tiled VAE, alpha kept, colour matched to the original)."""
    if len(refs) != 1:
        sys.exit("seedvr2 needs exactly one --ref image")
    return {
        "img": {"class_type": "LoadImage", "inputs": {"image": refs[0]}},
        "rgba": {"class_type": "JoinImageWithAlpha", "inputs": {"image": ["img", 0], "alpha": ["img", 1]}},
        "resize": {"class_type": "ResizeImageMaskNode", "inputs": {
            "input": ["rgba", 0], "resize_type": "scale by multiplier",
            "resize_type.multiplier": a.scale, "scale_method": "lanczos"}},
        "pre": {"class_type": "SeedVR2Preprocess", "inputs": {"resized_images": ["resize", 0]}},
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": SEEDVR2[a.seedvr2_model], "weight_dtype": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": SEEDVR2["vae"]}},
        "enc": {"class_type": "VAEEncodeTiled", "inputs": {
            "pixels": ["pre", 0], "vae": ["vae", 0], "tile_size": 512, "overlap": 128,
            "temporal_size": 4096, "temporal_overlap": 8}},
        "cond": {"class_type": "SeedVR2Conditioning", "inputs": {"model": ["unet", 0], "vae_conditioning": ["enc", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["unet", 0], "seed": a.seed, "steps": 1, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "simple", "positive": ["cond", 0], "negative": ["cond", 1],
            "latent_image": ["enc", 0], "denoise": 1.0}},
        "dec": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["sample", 0], "vae": ["vae", 0], "tile_size": 512, "overlap": 128,
            "temporal_size": 4096, "temporal_overlap": 8}},
        "post": {"class_type": "SeedVR2PostProcessing", "inputs": {
            "images": ["dec", 0], "original_resized_images": ["resize", 0],
            "color_correction_method": a.color}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["post", 0], "filename_prefix": "upscale/seedvr2"}},
    }


def build_klein_upscale(a, refs):
    """Creative upscale ("hires fix") with FLUX 3 Klein: enlarge, then repaint at partial
    denoise while the original is also fed in as a reference, so identity and layout hold."""
    if len(refs) != 1:
        sys.exit("klein-up needs exactly one --ref image")
    prompt = a.prompt or DEFAULT_UPSCALE_PROMPT
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": KLEIN["unet"], "weight_dtype": "default"}},
        "te": {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN["te"], "type": "flux3", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN["vae"]}},
        "img": {"class_type": "LoadImage", "inputs": {"image": refs[0]}},
        "big": {"class_type": "ImageScaleToTotalPixels", "inputs": {
            "image": ["img", 0], "upscale_method": "lanczos", "megapixels": a.megapixels, "resolution_steps": 16}},
        "init": {"class_type": "VAEEncode", "inputs": {"pixels": ["big", 0], "vae": ["vae", 0]}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": prompt}},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}},
        "refs": {"class_type": "ReferenceMultiLatent", "inputs": {
            "positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
            "resolution": "resize", "resolution.megapixels": a.ref_mp, "images.image0": ["img", 0]}},
        "sample": {"class_type": "KSampler", "inputs": {
            "model": ["unet", 0], "seed": a.seed, "steps": a.steps or 16, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "simple", "positive": ["refs", 0], "negative": ["refs", 1],
            "latent_image": ["init", 0], "denoise": a.denoise}},
        "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": "upscale/klein"}},
    }


DEFAULT_UPSCALE_PROMPT = (
    "A high-resolution version of <ref_image_0>: identical composition, subjects, colors and lighting, "
    "with sharp focus and fine, natural detail in every texture.")

SAM3_CKPT = "sam3.1_multiplex_fp16.safetensors"


def detail_prompt(n_refs):
    """Default detailer prompt. With references, it names them so Klein matches them."""
    base = ("A sharp, high-resolution close-up with fine, natural detail and realistic texture. "
            "Same pose, expression, colors and lighting.")
    if not n_refs:
        return base
    tags = ", ".join(f"<ref_image_{i}>" for i in range(n_refs))
    return (f"{base} The subject looks exactly like {tags}: "
            "same face, features, hair, clothing, colors and materials.")


def build_detail(a, refs):
    """ADetailer-style inpaint with FLUX 3 Klein (Impact Pack detailer).

    Each masked region is cropped with context (crop_factor), enlarged so its short side
    is about --size px, repainted at partial denoise, scaled back and pasted through a
    feathered mask. The --ref images are fed to Klein as references for every region, so
    faces and details stay close to them. The region comes from SAM3 text detection
    (--detect), a mask image (--mask) or rectangles (--box)."""
    manual = bool(a.mask or a.box)
    g = {
        "unet": {"class_type": "UNETLoader", "inputs": {"unet_name": KLEIN["unet"], "weight_dtype": "default"}},
        "te": {"class_type": "CLIPLoader", "inputs": {"clip_name": KLEIN["te"], "type": "flux3", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": KLEIN["vae"]}},
        "img": {"class_type": "LoadImage", "inputs": {"image": a.image_name}},
    }

    # ── where to work ──
    if a.mask:
        # White = repaint. The red channel works for grayscale and RGB mask files alike.
        g["mask"] = {"class_type": "LoadImageMask", "inputs": {"image": a.mask_name, "channel": "red"}}
        mask = ["mask", 0]
    elif a.box:
        g["size"] = {"class_type": "GetImageSize", "inputs": {"image": ["img", 0]}}
        g["box_base"] = {"class_type": "SolidMask", "inputs": {"value": 0.0, "width": ["size", 0], "height": ["size", 1]}}
        mask = ["box_base", 0]
        for i, (x, y, w, h) in enumerate(a.box):
            g[f"box{i}"] = {"class_type": "SolidMask", "inputs": {"value": 1.0, "width": w, "height": h}}
            g[f"box{i}_add"] = {"class_type": "MaskComposite", "inputs": {
                "destination": mask, "source": [f"box{i}", 0], "x": x, "y": y, "operation": "add"}}
            mask = [f"box{i}_add", 0]
    else:
        # SAM3 (built into ComfyUI): open-vocabulary detection. "face:4, hand:2" = up to
        # 4 faces and 2 hands; the masks of everything found are merged.
        g["sam"] = {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM3_CKPT}}
        g["sam_text"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["sam", 1], "text": a.detect}}
        g["detect"] = {"class_type": "SAM3_Detect", "inputs": {
            "model": ["sam", 0], "image": ["img", 0], "conditioning": ["sam_text", 0],
            "threshold": a.threshold, "refine_iterations": 2, "individual_masks": False}}
        mask = ["detect", 0]

    grow = a.grow if a.grow is not None else (8 if manual else 16)
    g["grow"] = {"class_type": "GrowMask", "inputs": {"mask": mask, "expand": grow, "tapered_corners": True}}
    # combined=False: every separate blob becomes its own region (each face on its own).
    # contour_fill: holes inside a region are filled, so a painted outline works too.
    g["segs"] = {"class_type": "MaskToSEGS", "inputs": {
        "mask": ["grow", 0], "combined": False, "crop_factor": a.crop_factor, "bbox_fill": False,
        "drop_size": 10 if manual else 32, "contour_fill": True}}

    # ── what to paint: prompt + references (same as Klein edit) ──
    for i, name in enumerate(refs):
        g[f"ref{i}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": ["te", 0], "text": a.prompt or detail_prompt(len(refs))}}
    g["neg"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos", 0]}}
    rml = {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
           "resolution": "resize", "resolution.megapixels": a.ref_mp}
    rml.update({f"images.image{i}": [f"ref{i}", 0] for i in range(len(refs))})
    g["refs"] = {"class_type": "ReferenceMultiLatent", "inputs": rml}

    # guide_size_for=False ("crop_region"): the crop's short side is scaled to --size,
    # the long side capped at --max-size. Klein works best at about 1 MP.
    g["detail"] = {"class_type": "DetailerForEach", "inputs": {
        "image": ["img", 0], "segs": ["segs", 0], "model": ["unet", 0], "clip": ["te", 0], "vae": ["vae", 0],
        "guide_size": a.size, "guide_size_for": False, "max_size": a.max_size, "seed": a.seed,
        "steps": a.steps or 16, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
        "positive": ["refs", 0], "negative": ["refs", 1], "denoise": a.denoise,
        "feather": a.feather, "noise_mask": True, "force_inpaint": True, "wildcard": "", "cycle": 1,
        "inpaint_model": False, "noise_mask_feather": 20, "tiled_encode": False, "tiled_decode": False}}
    g["save"] = {"class_type": "SaveImage", "inputs": {
        "images": ["detail", 0], "filename_prefix": f"detail/{'manual' if manual else 'auto'}"}}
    return g


BUILDERS = {"klein": build_klein, "qwen": build_qwen, "seedvr2": build_seedvr2, "klein-up": build_klein_upscale,
            "detail": build_detail}


# ── talking to the server ─────────────────────────────────────────────────────

def api(base, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read()
    return json.loads(body) if body else None


def stage_refs(paths, comfy_root):
    """Copy reference images into ComfyUI/input/gen/ and return their LoadImage names."""
    dest = Path(comfy_root) / "input" / "gen"
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for p in map(Path, paths):
        if not p.is_file():
            sys.exit(f"reference not found: {p}")
        target = dest / f"{p.stem}_{uuid.uuid4().hex[:8]}{p.suffix.lower()}"
        shutil.copy2(p, target)
        names.append(f"gen/{target.name}")
    return names


def print_node_errors(err):
    print("ComfyUI rejected the prompt:", err.get("error", {}).get("message", err), file=sys.stderr)
    for nid, info in (err.get("node_errors") or {}).items():
        for e in info.get("errors", []):
            print(f"  [{nid} {info.get('class_type')}] {e.get('message')}: {e.get('details')}", file=sys.stderr)


def run(a):
    if a.print_graph:   # no server needed; inputs keep their own names
        a.image_name = a.image and f"gen/{Path(a.image).name}"
        a.mask_name = a.mask and f"gen/{Path(a.mask).name}"
        graph = BUILDERS[a.model](a, [f"gen/{Path(r).name}" for r in a.ref])
        print(json.dumps(graph, indent=1, ensure_ascii=False))
        return

    base = f"http://{a.host}:{a.port}"
    try:
        api(base, "/system_stats")
    except (urllib.error.URLError, OSError):
        sys.exit(f"ComfyUI is not reachable at {base}. Start it with `nu launch.nu` first.")

    refs = stage_refs(a.ref, a.comfy_root) if a.ref else []
    a.image_name = stage_refs([a.image], a.comfy_root)[0] if a.image else None
    a.mask_name = stage_refs([a.mask], a.comfy_root)[0] if a.mask else None
    graph = BUILDERS[a.model](a, refs)

    try:
        res = api(base, "/prompt", {"prompt": graph, "client_id": uuid.uuid4().hex})
    except urllib.error.HTTPError as e:
        print_node_errors(json.loads(e.read() or b"{}"))
        sys.exit(1)
    pid = res["prompt_id"]
    if a.model == "detail":
        mode = "manual" if (a.mask or a.box) else f"auto [{a.detect}]"
    else:
        mode = "upscale" if a.model in ("seedvr2", "klein-up") else ("edit" if refs else "t2i")
    print(f"queued {a.model} {mode}  seed={a.seed}  id={pid}", file=sys.stderr)

    t0 = time.time()
    while True:
        time.sleep(1)
        hist = api(base, f"/history/{pid}") or {}
        if pid in hist:
            entry = hist[pid]
            break
        print(f"\r  waiting... {time.time() - t0:5.0f}s", end="", file=sys.stderr)
    print(f"\r  finished in {time.time() - t0:.0f}s      ", file=sys.stderr)

    status = entry.get("status", {})
    if status.get("status_str") == "error":
        for kind, msg in status.get("messages", []):
            if kind == "execution_error":
                print(f"  error in {msg.get('node_type')}: {msg.get('exception_message')}", file=sys.stderr)
        sys.exit(1)

    out_root = Path(a.comfy_root) / "output"
    for node_out in entry.get("outputs", {}).values():
        for img in node_out.get("images", []):
            if img.get("type") != "output":
                continue
            src = out_root / img.get("subfolder", "") / img["filename"]
            if a.out:
                Path(a.out).mkdir(parents=True, exist_ok=True)
                src = Path(shutil.copy2(src, Path(a.out) / img["filename"]))
            print(src)      # stdout: one path per image, easy to pipe in Nushell


def parse_box(text):
    try:
        x, y, w, h = (int(v) for v in text.replace(" ", "").split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected X,Y,W,H in pixels, got {text!r}")
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        raise argparse.ArgumentTypeError(f"box needs x, y >= 0 and positive width/height: {text!r}")
    return x, y, w, h


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--comfy-root", required=True, help="the ComfyUI folder (contains main.py)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8188)
    p.add_argument("model", choices=sorted(BUILDERS))
    p.add_argument("prompt", nargs="?", default="", help="optional for seedvr2 / klein-up / detail")
    p.add_argument("--ref", action="append", default=[],
                   help="reference image; repeat for more (edit mode; for detail: identity references)")
    p.add_argument("--upsample", action="store_true", help="rewrite the prompt with the model's prompt enhancer first")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--steps", type=int, default=None, help="default: klein 15 (t2i) / 16 (edit, detail), qwen 25")
    p.add_argument("--width", type=int, default=None, help="default 1024; for edits, default = first reference's size")
    p.add_argument("--height", type=int, default=None, help="default = width")
    p.add_argument("--negative", default="", help="qwen only")
    p.add_argument("--ref-mp", type=float, default=1.0, help="klein: megapixels each reference is resized to")
    p.add_argument("--ref-res", type=int, default=1024, help="qwen: references are resized to about NxN (0 = full size)")
    p.add_argument("--qwen-int8", action="store_true", help="use the official INT8 Qwen model instead of the GGUF")
    p.add_argument("--scale", type=float, default=2.0, help="seedvr2: upscale factor (2 or 4 are typical)")
    p.add_argument("--seedvr2-model", choices=["3b", "7b"], default="3b",
                   help="seedvr2: model size (7b needs the installer's --seedvr2-7b)")
    p.add_argument("--color", choices=["lab", "wavelet", "adain", "none"], default="lab",
                   help="seedvr2: colour matching to the original (lab = most faithful)")
    p.add_argument("--megapixels", type=float, default=4.0, help="klein-up: output size in megapixels (4 = 2x of 1 MP)")
    p.add_argument("--denoise", type=float, default=None,
                   help="klein-up / detail: how much is repainted (klein-up default 0.35, detail default 0.4)")
    d = p.add_argument_group("detail (ADetailer-style region repaint with Klein; --ref = identity references)")
    d.add_argument("--image", default=None, help="detail: the image to fix")
    d.add_argument("--detect", default="face:4",
                   help='detail: what SAM3 should find, e.g. "face:4, hand:2" (up to N of each). Default face:4')
    d.add_argument("--mask", default=None, help="detail: mask image, white = repaint (replaces --detect)")
    d.add_argument("--box", action="append", default=[], type=parse_box, metavar="X,Y,W,H",
                   help="detail: rectangle to repaint in pixels; repeatable (replaces --detect)")
    d.add_argument("--threshold", type=float, default=0.5, help="detail: SAM3 confidence threshold")
    d.add_argument("--size", type=int, default=1024, help="detail: each crop's short side is scaled to this")
    d.add_argument("--max-size", type=int, default=1536, help="detail: cap for each crop's long side")
    d.add_argument("--crop-factor", type=float, default=2.5, help="detail: context around a region (x its size)")
    d.add_argument("--grow", type=int, default=None, help="detail: grow the mask by N px (default 16 detect, 8 mask/box)")
    d.add_argument("--feather", type=int, default=8, help="detail: blur of the paste-back edge in px")
    p.add_argument("--out", default=None, help="also copy results into this folder")
    p.add_argument("--print-graph", action="store_true", help="print the API prompt instead of queueing it")
    a = p.parse_args(argv)
    if a.seed is None:
        a.seed = random.randrange(2**48)
    if not a.prompt and a.model in ("klein", "qwen"):
        p.error("a prompt is required for klein and qwen")
    if a.denoise is None:
        a.denoise = 0.4 if a.model == "detail" else 0.35
    if a.model == "detail":
        if not a.image:
            p.error("detail needs --image")
        if a.mask and a.box:
            p.error("use either --mask or --box, not both")
    elif a.image or a.mask or a.box:
        p.error("--image, --mask and --box are for the detail mode")
    run(a)


if __name__ == "__main__":
    main()
