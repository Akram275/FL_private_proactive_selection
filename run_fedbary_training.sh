#!/bin/bash
# FedBary baseline: train FedAvg on the federation selected by
# optimization/fedbary_selection.py (Wasserstein-barycenter-distance
# valuation, no-validation-set setting), for every task/k, 3 seeds each.
# Directly comparable to Optimal_FedAvg in Convergence2/ -- same training
# protocol (plain FedAvg, full participation), different federation-search
# method. See optimization/fedbary_selection.py's module docstring for the
# practical deviations from the original FedBary paper.
#
# Same per-seed-isolated, resume-safe pattern as run_fl_baseline_training.sh
# (one python3 process per seed, exit-code + output-length checked before
# logging success, skip already-valid seed CSVs on rerun).

set -e
set -o pipefail

TASKS=("ACSIncome" "ACSEmployment" "ACSPublicCoverage" "ACSTravelTime")
K_VALUES=(5 10 15)
SEEDS=(0 1 2)
FEDERATIONS_CSV="${FEDERATIONS_CSV:-results/fedbary_federations.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-Convergence2}"
FOLDER_NAME="${FOLDER_NAME:-FedBary_FedAvg}"
JOBS="${JOBS:-1}"

cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
FED_CSV_ABS="${PROJECT_ROOT}/${FEDERATIONS_CSV}"
OUT_DIR_ABS="${PROJECT_ROOT}/${OUTPUT_DIR}"
LOG_DIR="${PROJECT_ROOT}/results/fedbary_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo "FedBary FL Training (per-seed isolated) - Started $(date)"
echo "Tasks: ${TASKS[*]}  K: ${K_VALUES[*]}  Seeds: ${SEEDS[*]}  Workers: $JOBS"
echo "Federations CSV: $FED_CSV_ABS"
echo "Output folder: $OUT_DIR_ABS/{task}/fixed_size_{k}/$FOLDER_NAME"
echo "Logs: $LOG_DIR"
echo "======================================================"

seed_already_done() {
    local task="$1" k="$2" seed="$3"
    local f="${OUT_DIR_ABS}/${task}/fixed_size_${k}/${FOLDER_NAME}/seed_${seed}.csv"
    [ -f "$f" ] && [ "$(wc -l < "$f")" -eq 51 ]
}

run_one_seed() {
    local task="$1" k="$2" seed="$3"

    if seed_already_done "$task" "$k" "$seed"; then
        echo "SKIP (already valid) $task k=$k seed=$seed"
        return 0
    fi

    local job_log="${LOG_DIR}/${task}_k${k}_seed${seed}.log"

    if python3 "${PROJECT_ROOT}/FL_training/FolkTables_FL.py" \
        --task "$task" --k "$k" --n-seeds 1 --seed-start "$seed" \
        --federations-csv "$FED_CSV_ABS" --output-dir "$OUT_DIR_ABS" \
        --only-optimal --optimal-folder-name "$FOLDER_NAME" \
        > "$job_log" 2>&1
    then
        if seed_already_done "$task" "$k" "$seed"; then
            echo "OK $task k=$k seed=$seed (log: $job_log)"
        else
            echo "FAILED (exit 0 but output missing/short) $task k=$k seed=$seed (log: $job_log)"
        fi
    else
        echo "FAILED (python3 exit $?) $task k=$k seed=$seed (log: $job_log)"
    fi
}
export -f run_one_seed seed_already_done
export FED_CSV_ABS OUT_DIR_ABS FOLDER_NAME LOG_DIR PROJECT_ROOT

for TASK in "${TASKS[@]}"; do
    for K in "${K_VALUES[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            printf '%s\t%s\t%s\n' "$TASK" "$K" "$SEED"
        done
    done
done | xargs -P "$JOBS" -n 3 bash -c 'run_one_seed "$0" "$1" "$2"'

echo ""
echo "======================================================"
echo "All FedBary training units processed - $(date)"
echo "Results in: $OUT_DIR_ABS"
echo "======================================================"
