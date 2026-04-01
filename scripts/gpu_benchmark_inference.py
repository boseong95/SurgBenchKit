"""
Benchmark inference throughput and GPU memory for each model.

Profiles:
  - GPU memory footprint per model
  - Inference throughput (frames/sec) with warm-up
  - Estimates max concurrent models that fit in GPU memory
  - Saves results as JSON for reuse across instance types

Usage:
    python scripts/benchmark_inference.py                    # benchmark all models
    python scripts/benchmark_inference.py --model Qwen2-VL-7B-Instruct
    python scripts/benchmark_inference.py --num-frames 20    # more frames for accurate timing
"""

import argparse
import gc
import json
import os
import sys
import time
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

GENERATIVE_MODELS = [
    "Qwen2-VL-7B-Instruct",
    "InternVL3-8B",
    "llava_next_vicuna_7b",
    "paligemma-3b-mix-448",
    "MedGemma-4B",
]

TEST_PROMPT = (
    'What surgical instrument is visible in this image? '
    'Use this JSON schema: {"instrument": str} and avoid line breaks.'
)


def get_gpu_memory():
    """Return (used_MB, total_MB) for GPU 0."""
    import torch
    if not torch.cuda.is_available():
        return 0, 0
    total = torch.cuda.get_device_properties(0).total_mem / 1024**2
    used = torch.cuda.memory_allocated(0) / 1024**2
    return used, total


def get_gpu_memory_nvidia_smi():
    """Get actual GPU memory used via nvidia-smi (more accurate for fragmented memory)."""
    import subprocess
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    )
    used, total = result.stdout.strip().split(", ")
    return int(used), int(total)


def create_test_images(tmp_dir, n=30, size=448):
    """Create N synthetic test images."""
    paths = []
    for i in range(n):
        path = os.path.join(tmp_dir, f"bench_{i}.jpg")
        img = Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))
        img.save(path)
        paths.append(path)
    return paths


def benchmark_model(model_name, image_paths, num_warmup=3, num_frames=10):
    """Benchmark a single model. Returns dict with results."""
    import torch
    from vlmeval.config import supported_VLM

    result = {
        "model": model_name,
        "status": "FAIL",
    }

    # Measure baseline GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()
    mem_before, mem_total = get_gpu_memory_nvidia_smi()

    # Load model
    print(f"  Loading {model_name}...")
    t0 = time.time()
    try:
        model = supported_VLM[model_name]()
        if not hasattr(model, "name"):
            model.name = model_name
    except Exception as e:
        result["error"] = str(e)
        print(f"  FAILED to load: {e}")
        return result

    load_time = time.time() - t0
    mem_after_load, _ = get_gpu_memory_nvidia_smi()
    model_mem = mem_after_load - mem_before

    print(f"  Loaded in {load_time:.1f}s | GPU memory: {model_mem} MB")

    # Warm-up inference
    print(f"  Warm-up ({num_warmup} frames)...")
    for i in range(num_warmup):
        try:
            model.generate([image_paths[i % len(image_paths)], TEST_PROMPT])
        except Exception as e:
            result["error"] = f"Inference failed: {e}"
            print(f"  FAILED during warm-up: {e}")
            del model
            torch.cuda.empty_cache()
            gc.collect()
            return result

    mem_after_infer, _ = get_gpu_memory_nvidia_smi()
    peak_mem = mem_after_infer - mem_before

    # Timed benchmark
    print(f"  Benchmarking ({num_frames} frames)...")
    latencies = []
    for i in range(num_frames):
        img_path = image_paths[i % len(image_paths)]
        t_start = time.time()
        model.generate([img_path, TEST_PROMPT])
        latencies.append(time.time() - t_start)

    # Compute stats
    avg_latency = np.mean(latencies)
    p50_latency = np.median(latencies)
    p95_latency = np.percentile(latencies, 95)
    throughput = 1.0 / avg_latency if avg_latency > 0 else 0

    # Max concurrent models estimate
    max_concurrent = max(1, int(mem_total * 0.90 / peak_mem)) if peak_mem > 0 else 1

    result.update({
        "status": "PASS",
        "load_time_s": round(load_time, 1),
        "model_memory_mb": model_mem,
        "peak_memory_mb": peak_mem,
        "gpu_total_mb": mem_total,
        "avg_latency_s": round(avg_latency, 3),
        "p50_latency_s": round(p50_latency, 3),
        "p95_latency_s": round(p95_latency, 3),
        "throughput_fps": round(throughput, 2),
        "max_concurrent_models": max_concurrent,
        "num_frames_benchmarked": num_frames,
    })

    print(f"  Throughput: {throughput:.2f} fps | Avg latency: {avg_latency:.3f}s | "
          f"Peak GPU mem: {peak_mem} MB | Max concurrent: {max_concurrent}")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Benchmark model inference throughput")
    parser.add_argument("--model", type=str, default=None, help="Specific model to benchmark")
    parser.add_argument("--num-frames", type=int, default=10, help="Number of frames for timed benchmark")
    parser.add_argument("--num-warmup", type=int, default=3, help="Number of warm-up frames")
    parser.add_argument("--output", type=str, default="outputs/model_test/benchmark_results.json",
                        help="Output JSON path")
    args = parser.parse_args()

    models = [args.model] if args.model else GENERATIVE_MODELS

    # Create test images
    tmp_dir = tempfile.mkdtemp(prefix="surgbench_bench_")
    image_paths = create_test_images(tmp_dir, n=max(30, args.num_frames + args.num_warmup))
    print(f"Created {len(image_paths)} test images in {tmp_dir}")

    # Run benchmarks
    results = []
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {model_name}")
        print(f"{'='*60}")
        res = benchmark_model(model_name, image_paths, args.num_warmup, args.num_frames)
        results.append(res)

    # Cleanup test images
    for p in image_paths:
        os.remove(p)
    os.rmdir(tmp_dir)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Merge with existing results if file exists
    existing = []
    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            existing = json.load(f)
        existing_names = {r["model"] for r in existing}
        for r in results:
            if r["model"] in existing_names:
                existing = [e for e in existing if e["model"] != r["model"]]
            existing.append(r)
        results = existing

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<28} {'FPS':>6} {'Latency':>8} {'GPU MB':>7} {'MaxConc':>8}")
    print("-" * 60)
    for r in results:
        if r["status"] == "PASS":
            print(f"{r['model']:<28} {r['throughput_fps']:>6.2f} {r['avg_latency_s']:>7.3f}s "
                  f"{r['peak_memory_mb']:>7} {r['max_concurrent_models']:>8}")
        else:
            print(f"{r['model']:<28} {'FAILED':>6}")

    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDATIONS")
    print(f"{'='*60}")
    passed = [r for r in results if r["status"] == "PASS"]
    if passed:
        gpu_total = passed[0]["gpu_total_mb"]
        print(f"GPU: {gpu_total} MB total")
        for r in passed:
            est_frames = 14259  # total HeiChole test frames
            est_time_min = est_frames / r["throughput_fps"] / 60
            tasks = 3  # phase, tool, action
            total_min = est_time_min * tasks
            print(f"\n  {r['model']}:")
            print(f"    Sequential (1 process):  ~{est_time_min:.0f} min/task, ~{total_min:.0f} min total")
            if r["max_concurrent_models"] > 1:
                par_min = total_min / min(r["max_concurrent_models"], tasks)
                print(f"    Parallel ({min(r['max_concurrent_models'], tasks)} processes): ~{par_min:.0f} min total")
            print(f"    With max_samples=100:    ~{100 / r['throughput_fps'] / 60 * tasks:.1f} min total")

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
