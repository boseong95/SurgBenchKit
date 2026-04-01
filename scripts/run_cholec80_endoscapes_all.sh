#!/bin/bash
# Run all models x all Cholec80 + Endoscapes tasks (zero-shot).
#
# Tasks:
#   - endoscapes_cvs_assessment
#   - cholec80_phase_recognition
#   - cholec80_tool_recognition
#
# Strategy:
#   - Tasks run SEQUENTIALLY per model (avoids loading multiple copies)
#   - Different models can run in PARALLEL if they fit in GPU memory
#   - Model groups are defined by memory budget (~98 GB total, use ~90%)
#
# Usage:
#   bash scripts/run_cholec80_endoscapes_all.sh              # run all
#   bash scripts/run_cholec80_endoscapes_all.sh --dry-run    # show plan without running

set -uo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
fi

TASKS=(
    "endoscapes_cvs_assessment"
    "cholec80_phase_recognition"
    "cholec80_tool_recognition"
)
NUM_TASKS=${#TASKS[@]}

# Map model names to config yaml filenames
declare -A MODEL_CONFIGS
MODEL_CONFIGS["Qwen2-VL-7B-Instruct"]="Qwen2-VL"
MODEL_CONFIGS["InternVL3-8B"]="InternVL3"
MODEL_CONFIGS["llava_next_vicuna_7b"]="llava_next_vicuna_7b"
MODEL_CONFIGS["paligemma-3b-mix-448"]="PaliGemma"
MODEL_CONFIGS["MedGemma-4B"]="MedGemma-4B"
MODEL_CONFIGS["Qwen3-VL-8B-Instruct"]="Qwen3-VL-Instruct"
MODEL_CONFIGS["Qwen3-VL-8B-Thinking"]="Qwen3-VL-Thinking"

# Parallel groups: models that fit together in ~90 GB GPU memory
# Qwen2(18)+InternVL3(15)+paligemma(6)=39GB, llava(26)+MedGemma(26)=52GB, Qwen3(18)+Qwen3(18)=36GB
GROUP1=("Qwen2-VL-7B-Instruct" "InternVL3-8B" "paligemma-3b-mix-448")
GROUP2=("llava_next_vicuna_7b" "MedGemma-4B")
GROUP3=("Qwen3-VL-8B-Instruct" "Qwen3-VL-8B-Thinking")

LOG_DIR="outputs/model_test/cholec80_endoscapes_runs"
mkdir -p "$LOG_DIR"

TOTAL_PASS=0
TOTAL_FAIL=0
START_TIME=$(date +%s)

# Run one model: all tasks sequentially
run_model() {
    local model="$1"
    local config="${MODEL_CONFIGS[$model]}"
    local pass=0 fail=0

    echo "  [$model] Starting $NUM_TASKS tasks sequentially..."
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

# Run a group of models in parallel (each model runs its tasks sequentially)
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
echo "  Cholec80 + Endoscapes Inference Run"
echo "  Total jobs: $TOTAL_JOBS (7 models x $NUM_TASKS tasks)"
echo "  Tasks: ${TASKS[*]}"
echo "  Tasks sequential, models parallel within groups"
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
