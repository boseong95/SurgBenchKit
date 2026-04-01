#!/bin/bash
# Run all models x CholecT45 triplet recognition (zero-shot).
#
# Task: cholect45_triplet_recognition
#
# Strategy: same GPU grouping as other scripts
#
# Usage:
#   bash scripts/run_cholect45_all.sh              # run all
#   bash scripts/run_cholect45_all.sh --dry-run    # show plan without running

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
fi

TASKS=(
    "cholect45_triplet_recognition"
)
NUM_TASKS=${#TASKS[@]}

declare -A MODEL_CONFIGS
MODEL_CONFIGS["Qwen2-VL-7B-Instruct"]="Qwen2-VL"
MODEL_CONFIGS["InternVL3-8B"]="InternVL3"
MODEL_CONFIGS["llava_next_vicuna_7b"]="llava_next_vicuna_7b"
MODEL_CONFIGS["paligemma-3b-mix-448"]="PaliGemma"
MODEL_CONFIGS["MedGemma-4B"]="MedGemma-4B"
MODEL_CONFIGS["Qwen3-VL-8B-Instruct"]="Qwen3-VL-Instruct"
MODEL_CONFIGS["Qwen3-VL-8B-Thinking"]="Qwen3-VL-Thinking"

GROUP1=("Qwen2-VL-7B-Instruct" "InternVL3-8B" "paligemma-3b-mix-448")
GROUP2=("llava_next_vicuna_7b" "MedGemma-4B")
GROUP3=("Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Thinking")

LOG_DIR="outputs/model_test/cholect45_runs"
mkdir -p "$LOG_DIR"

TOTAL_PASS=0
TOTAL_FAIL=0
START_TIME=$(date +%s)

run_model() {
    local model="$1"
    local config="${MODEL_CONFIGS[$model]}"
    local pass=0 fail=0

    echo "  [$model] Starting $NUM_TASKS task(s) sequentially..."
    for task in "${TASKS[@]}"; do
        local log="$LOG_DIR/${model}__${task}.log"
        echo "  [$model] Running: $task"
        if python3 eval.py model="$config" task="$task" +task.max_samples=100 override_outputs=True > "$log" 2>&1; then
            echo "  [$model] [PASS] $task"
            pass=$((pass+1))
        else
            echo "  [$model] [FAIL] $task — see $log"
            fail=$((fail+1))
        fi
    done
    return $fail
}

run_group() {
    local group_name="$1"
    shift
    local models=("$@")

    echo ""
    echo "================================================================"
    echo "  $group_name: ${models[*]}"
    echo "================================================================"

    if [ "$DRY_RUN" = true ]; then
        for model in "${models[@]}"; do
            for task in "${TASKS[@]}"; do
                echo "  [DRY-RUN] python3 eval.py model=${MODEL_CONFIGS[$model]} task=$task +task.max_samples=100 override_outputs=True"
            done
        done
        return
    fi

    local PIDS=()
    local MNAMES=()

    for model in "${models[@]}"; do
        run_model "$model" &
        PIDS+=($!)
        MNAMES+=("$model")
    done

    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            echo "  [${MNAMES[$i]}] All tasks passed"
            TOTAL_PASS=$((TOTAL_PASS + NUM_TASKS))
        else
            fail_count=$?
            echo "  [${MNAMES[$i]}] $fail_count task(s) failed"
            TOTAL_FAIL=$((TOTAL_FAIL + fail_count))
            TOTAL_PASS=$((TOTAL_PASS + NUM_TASKS - fail_count))
        fi
    done
}

TOTAL_JOBS=$(( (${#GROUP1[@]} + ${#GROUP2[@]} + ${#GROUP3[@]}) * NUM_TASKS ))

echo "================================================================"
echo "  CholecT45 Triplet Recognition Inference Run"
echo "  Total jobs: $TOTAL_JOBS (7 models x $NUM_TASKS task)"
echo "  Tasks: ${TASKS[*]}"
echo "  Log dir: $LOG_DIR"
echo "  Start: $(date)"
echo "================================================================"

run_group "Group 1 (~39 GB)" "${GROUP1[@]}"
run_group "Group 2 (~52 GB)" "${GROUP2[@]}"
run_group "Group 3 (~36 GB)" "${GROUP3[@]}"

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "================================================================"
echo "  FINAL SUMMARY"
echo "================================================================"
echo "  Passed: $TOTAL_PASS / $TOTAL_JOBS"
echo "  Failed: $TOTAL_FAIL / $TOTAL_JOBS"
echo "  Elapsed: ${ELAPSED} min"
echo "  Logs: $LOG_DIR/"
echo "================================================================"

if [ "$TOTAL_FAIL" -gt 0 ]; then
    echo ""
    echo "Failed logs:"
    for f in "$LOG_DIR"/*.log; do
        if grep -q "Exception\|Error\|Traceback" "$f" 2>/dev/null; then
            echo "  $f:"
            grep "Exception\|Error" "$f" | tail -2 | sed 's/^/    /'
        fi
    done
    exit 1
fi
