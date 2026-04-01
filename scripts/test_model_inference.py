"""
Test script to verify basic inference pipeline for each supported model.

Usage:
    # Test a specific model
    python scripts/test_model_inference.py --model Qwen2-VL-7B-Instruct

    # Test all local (non-API) generative models
    python scripts/test_model_inference.py --group generative

    # Test all contrastive models
    python scripts/test_model_inference.py --group contrastive

    # Test all models
    python scripts/test_model_inference.py --group all

    # List available models
    python scripts/test_model_inference.py --list
"""

import argparse
import os
import sys
import time
import traceback
import tempfile

import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from vlmeval.config import supported_VLM


# Models that have config yamls and are meant to run locally
GENERATIVE_MODELS = [
    "Qwen2-VL-7B-Instruct",
    "InternVL3-8B",
    "llava_next_vicuna_7b",
    "paligemma-3b-mix-448",
    "MedGemma-4B",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking",
]

CONTRASTIVE_MODELS = [
    "CLIP",
    "OpenCLIP",
]

# SurgVLP may not be available due to dependency issues
try:
    if "SurgVLP" in supported_VLM:
        CONTRASTIVE_MODELS.append("SurgVLP")
except Exception:
    pass

TEST_PROMPT = (
    'What surgical instrument is visible in this image? '
    'Use this JSON schema: {"instrument": str} and avoid line breaks.'
)

TEST_CAPTIONS = [
    "A surgical scene containing a grasper.",
    "A surgical scene containing scissors.",
    "A surgical scene containing a hook.",
]


def create_test_image(path):
    """Create a simple synthetic test image (RGB noise)."""
    img = Image.fromarray(
        np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)
    )
    img.save(path)
    return path


def test_generative_model(model_name, image_path):
    """Test a generative model's basic inference pipeline."""
    print(f"\n{'='*60}")
    print(f"Testing generative model: {model_name}")
    print(f"{'='*60}")

    # Step 1: Instantiate model
    print(f"  [1/3] Loading model '{model_name}'...")
    t0 = time.time()
    model = supported_VLM[model_name]()
    if not hasattr(model, "name"):
        model.name = model_name
    load_time = time.time() - t0
    print(f"         Loaded in {load_time:.1f}s")

    # Step 2: Run inference
    print(f"  [2/3] Running inference...")
    t0 = time.time()
    output = model.generate([image_path, TEST_PROMPT])
    infer_time = time.time() - t0
    print(f"         Inference took {infer_time:.1f}s")

    # Step 3: Validate output
    print(f"  [3/3] Validating output...")
    assert output is not None, "Output is None"
    assert isinstance(output, (str, int)), f"Unexpected output type: {type(output)}"
    if isinstance(output, str):
        assert len(output) > 0, "Output is empty string"
    print(f"         Output (truncated): {str(output)[:200]}")

    print(f"\n  PASSED: {model_name}")
    return True


def test_contrastive_model(model_name, image_path):
    """Test a contrastive model's basic inference pipeline."""
    print(f"\n{'='*60}")
    print(f"Testing contrastive model: {model_name}")
    print(f"{'='*60}")

    # Step 1: Instantiate model
    print(f"  [1/3] Loading model '{model_name}'...")
    t0 = time.time()
    model = supported_VLM[model_name](eval_type="singlelabel")
    if not hasattr(model, "name"):
        model.name = model_name
    load_time = time.time() - t0
    print(f"         Loaded in {load_time:.1f}s")

    # Step 2: Run inference
    print(f"  [2/3] Running inference...")
    t0 = time.time()
    output = model(TEST_CAPTIONS, image_path)
    infer_time = time.time() - t0
    print(f"         Inference took {infer_time:.1f}s")

    # Step 3: Validate output
    print(f"  [3/3] Validating output...")
    assert output is not None, "Output is None"
    assert hasattr(output, "shape"), f"Expected numpy array, got {type(output)}"
    assert output.shape[-1] == len(TEST_CAPTIONS), (
        f"Expected {len(TEST_CAPTIONS)} scores, got shape {output.shape}"
    )
    print(f"         Similarity scores: {output}")

    print(f"\n  PASSED: {model_name}")
    return True


def list_models():
    """List all testable models."""
    print("Generative models:")
    for m in GENERATIVE_MODELS:
        available = m in supported_VLM
        status = "available" if available else "NOT REGISTERED"
        print(f"  - {m} ({status})")

    print("\nContrastive models:")
    for m in CONTRASTIVE_MODELS:
        available = m in supported_VLM
        status = "available" if available else "NOT REGISTERED"
        print(f"  - {m} ({status})")


def main():
    parser = argparse.ArgumentParser(description="Test model inference pipelines")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Specific model name to test",
    )
    parser.add_argument(
        "--group",
        type=str,
        choices=["generative", "contrastive", "all"],
        default=None,
        help="Test a group of models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    if args.model is None and args.group is None:
        parser.print_help()
        return

    # Create a temporary test image
    tmp_dir = tempfile.mkdtemp(prefix="surgbench_test_")
    image_path = os.path.join(tmp_dir, "test_image.jpg")
    create_test_image(image_path)
    print(f"Test image created at: {image_path}")

    # Build test list
    models_to_test = []
    if args.model:
        # Determine if it's contrastive or generative
        is_contrastive = args.model in CONTRASTIVE_MODELS
        models_to_test.append((args.model, is_contrastive))
    elif args.group in ("generative", "all"):
        for m in GENERATIVE_MODELS:
            models_to_test.append((m, False))
    if args.group in ("contrastive", "all"):
        for m in CONTRASTIVE_MODELS:
            models_to_test.append((m, True))

    # Run tests
    results = {}
    for model_name, is_contrastive in models_to_test:
        if model_name not in supported_VLM:
            print(f"\nSKIPPED: {model_name} (not registered in supported_VLM)")
            results[model_name] = "SKIPPED"
            continue
        try:
            if is_contrastive:
                test_contrastive_model(model_name, image_path)
            else:
                test_generative_model(model_name, image_path)
            results[model_name] = "PASSED"
        except Exception as e:
            print(f"\n  FAILED: {model_name}")
            print(f"  Error: {e}")
            traceback.print_exc()
            results[model_name] = f"FAILED: {e}"

    # Cleanup
    os.remove(image_path)
    os.rmdir(tmp_dir)

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    for model_name, status in results.items():
        icon = "PASS" if status == "PASSED" else ("SKIP" if status == "SKIPPED" else "FAIL")
        print(f"  [{icon}] {model_name}")

    n_passed = sum(1 for s in results.values() if s == "PASSED")
    n_failed = sum(1 for s in results.values() if "FAILED" in s)
    n_skipped = sum(1 for s in results.values() if s == "SKIPPED")
    print(f"\n  Total: {len(results)} | Passed: {n_passed} | Failed: {n_failed} | Skipped: {n_skipped}")

    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
