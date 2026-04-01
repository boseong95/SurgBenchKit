#!/bin/bash
# Test each model's basic inference pipeline individually.
# Usage:
#   bash scripts/test_all_models.sh                    # test all models one by one
#   bash scripts/test_all_models.sh Qwen2-VL-7B-Instruct  # test a specific model
#   bash scripts/test_all_models.sh --group generative     # test generative models only
#   bash scripts/test_all_models.sh --group contrastive    # test contrastive models only

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -gt 0 ]; then
    # Pass arguments directly
    python3 scripts/test_model_inference.py "$@"
else
    # Test each model individually so one failure doesn't block the rest
    GENERATIVE_MODELS=(
        "Qwen2-VL-7B-Instruct"
        "InternVL3-8B"
        "llava_next_vicuna_7b"
        "paligemma-3b-mix-448"
        "MedGemma-4B"
        "Qwen3-VL-8B-Instruct"
        "Qwen3-VL-8B-Thinking"
    )

    CONTRASTIVE_MODELS=(
        "CLIP"
        "OpenCLIP"
    )

    PASSED=0
    FAILED=0
    SKIPPED=0
    FAIL_LIST=""

    for model in "${GENERATIVE_MODELS[@]}" "${CONTRASTIVE_MODELS[@]}"; do
        echo ""
        echo "============================================================"
        echo "  Testing: $model"
        echo "============================================================"
        if python3 scripts/test_model_inference.py --model "$model"; then
            PASSED=$((PASSED + 1))
        else
            exit_code=$?
            if [ $exit_code -eq 2 ]; then
                SKIPPED=$((SKIPPED + 1))
            else
                FAILED=$((FAILED + 1))
                FAIL_LIST="$FAIL_LIST $model"
            fi
        fi
    done

    echo ""
    echo "============================================================"
    echo "  FINAL SUMMARY"
    echo "============================================================"
    echo "  Passed:  $PASSED"
    echo "  Failed:  $FAILED"
    echo "  Skipped: $SKIPPED"
    if [ -n "$FAIL_LIST" ]; then
        echo "  Failed models:$FAIL_LIST"
        exit 1
    fi
fi
