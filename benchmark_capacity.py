"""
Benchmark Pi3 / Pi3X capacity at different resolutions.
Uses exponential probing (1, 2, 4, 8, ...) to quickly find the max power-of-2
number of images that fits in GPU memory. No slow binary search.
"""
import argparse
import gc
import json
import time
from datetime import datetime

import torch


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def try_forward(model, model_type, n_images, H, W, device, dtype):
    """Try a forward pass with n_images. Returns (success, peak_vram_mb, elapsed_sec)."""
    clear_gpu()
    try:
        imgs = torch.rand(1, n_images, 3, H, W, device=device, dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                if model_type == "pi3":
                    _ = model(imgs)
                else:
                    _ = model(imgs=imgs)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        del imgs, _
        clear_gpu()
        return True, peak_mb, elapsed
    except torch.cuda.OutOfMemoryError:
        clear_gpu()
        return False, -1, -1
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            clear_gpu()
            return False, -1, -1
        raise


def probe_max_images(model, model_type, H, W, device, dtype, max_n=8192):
    """Exponential probing: try N = 1, 2, 4, 8, ... until OOM.
    Returns (max_n, peak_vram_mb, elapsed_sec) for the last successful N."""
    patches_per_image = (H // 14) * (W // 14)
    n = 1
    best_n, best_peak, best_elapsed = 0, -1, -1

    while n <= max_n:
        total_patches = n * patches_per_image
        print(f"    N={n:>5d} (total patches: {total_patches:>7d}) ...", end=" ", flush=True)
        ok, peak_mb, elapsed = try_forward(model, model_type, n, H, W, device, dtype)
        if ok:
            print(f"OK   {peak_mb:>7.0f} MB  {elapsed:>6.2f}s  ({n/elapsed:>6.1f} img/s)")
            best_n, best_peak, best_elapsed = n, peak_mb, elapsed
            n *= 2
        else:
            print("OOM")
            break

    return best_n, best_peak, best_elapsed


def load_model(model_type, ckpt, use_mm, device):
    if model_type == "pi3x":
        from pi3.models.pi3x import Pi3X
        if ckpt is not None:
            model = Pi3X(use_multimodal=use_mm).eval()
            if ckpt.endswith('.safetensors'):
                from safetensors.torch import load_file
                weight = load_file(ckpt)
            else:
                weight = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(weight, strict=False)
        else:
            model = Pi3X.from_pretrained("yyfz233/Pi3X").eval()
            if not use_mm:
                model.disable_multimodal()
    elif model_type == "pi3":
        from pi3.models.pi3 import Pi3
        if ckpt is not None:
            model = Pi3().eval()
            if ckpt.endswith('.safetensors'):
                from safetensors.torch import load_file
                weight = load_file(ckpt)
            else:
                weight = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(weight)
        else:
            model = Pi3.from_pretrained("yyfz233/Pi3").eval()
    else:
        raise ValueError(f"Unknown model type: {model_type}, use pi3 or pi3x")

    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Pi3/Pi3X capacity at different resolutions")
    parser.add_argument("--model", type=str, default="pi3x", choices=["pi3", "pi3x"],
                        help="Model type (default: pi3x)")
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path (default: HuggingFace)")
    parser.add_argument("--multimodal", action="store_true",
                        help="[pi3x only] Enable multimodal branch")
    parser.add_argument("--max-n", type=int, default=8192, help="Max N to probe (default: 8192)")
    parser.add_argument("--resolutions", type=str, default=None,
                        help="Custom resolutions: HxW,HxW,...  e.g. 224x224,504x504")
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save results to JSON file")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    gpu_name = torch.cuda.get_device_name(0)
    gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    gpu_total_gb = gpu_total_mb / 1024

    model_type = args.model
    use_mm = args.multimodal and model_type == "pi3x"

    print(f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB)")
    print(f"Model: {model_type.upper()}")
    print(f"Dtype: {dtype}")
    if model_type == "pi3x":
        print(f"Multimodal: {'enabled' if use_mm else 'disabled'}")
    print()

    print("Loading model...")
    model = load_model(model_type, args.ckpt, use_mm, device)
    clear_gpu()

    model_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    print(f"Model VRAM: {model_mem_mb:.0f} MB")
    print()

    if args.resolutions:
        resolutions = []
        for r in args.resolutions.split(","):
            h, w = r.strip().split("x")
            resolutions.append((int(h), int(w)))
    else:
        resolutions = [
            (224, 224),    # ~50K px
            (336, 336),    # ~113K px
            (336, 504),    # ~169K px (3:2)
            (364, 504),    # ~183K px
            (378, 672),    # ~254K px (near PIXEL_LIMIT=255K, 9:16)
            (504, 504),    # ~254K px (near PIXEL_LIMIT=255K, 1:1)
            (518, 518),    # DINOv2 default
            (672, 672),    # ~452K px
            (504, 896),    # ~452K px (9:16)
        ]

    checked = []
    for h, w in resolutions:
        h14 = (h // 14) * 14
        w14 = (w // 14) * 14
        if h14 != h or w14 != w:
            print(f"  Note: {h}x{w} -> {h14}x{w14} (multiple of 14)")
        if h14 > 0 and w14 > 0:
            checked.append((h14, w14))
    resolutions = checked

    results = []
    for h, w in resolutions:
        pixels = h * w
        patches_per_image = (h // 14) * (w // 14)
        print(f"[{h}x{w}] ({pixels / 1000:.0f}K px, {patches_per_image} patches/image)")

        max_n, peak_mb, elapsed = probe_max_images(
            model, model_type, h, w, device, dtype, max_n=args.max_n)

        total_patches = max_n * patches_per_image if max_n > 0 else 0
        vram_util = peak_mb / gpu_total_mb * 100 if peak_mb > 0 else 0
        vram_per_img = (peak_mb - model_mem_mb) / max_n if max_n > 0 and peak_mb > 0 else 0
        throughput = max_n / elapsed if max_n > 0 and elapsed > 0 else 0

        row = {
            "Resolution": f"{h}x{w}",
            "Pixels": f"{pixels / 1000:.0f}K",
            "Patches/img": patches_per_image,
            "Max N": max_n if max_n > 0 else "< 1",
            "Total patches": total_patches if max_n > 0 else "N/A",
            "Peak VRAM (MB)": f"{peak_mb:.0f}" if peak_mb > 0 else "N/A",
            "VRAM util %": f"{vram_util:.1f}" if peak_mb > 0 else "N/A",
            "MB/img": f"{vram_per_img:.1f}" if vram_per_img > 0 else "N/A",
            "Time (s)": f"{elapsed:.2f}" if elapsed > 0 else "N/A",
            "Img/s": f"{throughput:.1f}" if throughput > 0 else "N/A",
        }
        results.append(row)

        print(f"  => N={max_n} | patches={total_patches} | "
              f"VRAM: {peak_mb:.0f} MB ({vram_util:.1f}%) | "
              f"~{vram_per_img:.1f} MB/img | "
              f"{elapsed:.2f}s | {throughput:.1f} img/s")
        print()

    # Summary table
    print("=" * 110)
    print(f"GPU: {gpu_name} ({gpu_total_gb:.1f} GB) | Model: {model_type.upper()} | "
          f"Dtype: {dtype} | Model VRAM: {model_mem_mb:.0f} MB"
          + (f" | Multimodal: {'enabled' if use_mm else 'disabled'}" if model_type == "pi3x" else ""))
    print("=" * 110)
    try:
        from tabulate import tabulate
        print(tabulate(results, headers="keys", tablefmt="grid"))
    except ImportError:
        cols = list(results[0].keys())
        header = " | ".join(f"{c:>14s}" for c in cols)
        print(header)
        print("-" * len(header))
        for r in results:
            print(" | ".join(f"{str(r[c]):>14s}" for c in cols))

    if args.save_json:
        report = {
            "timestamp": datetime.now().isoformat(),
            "gpu": gpu_name,
            "gpu_total_mb": round(gpu_total_mb),
            "model": model_type,
            "dtype": str(dtype),
            "multimodal": use_mm,
            "model_vram_mb": round(model_mem_mb),
            "results": results,
        }
        with open(args.save_json, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.save_json}")


if __name__ == "__main__":
    main()
