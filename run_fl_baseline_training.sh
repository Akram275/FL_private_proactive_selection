#!/bin/bash
# Full FL training reproduction: optimal federation (new calibrated-PFL,
# fully-annealed SA search) vs. random federations across every baseline
# from Table II (FedAvg, FedProx, SCAFFOLD aggregation; UCB-CS, FedSampling,
# Threshold client selection), 3 random federations (fresh per seed) per
# task/k, matching the paper's Fig. 4/5 methodology and the README's
# documented pipeline.
#
# Each SEED is its own python3 process (not each task/k/config -- that was
# the first version's mistake). run_exp() leaks TensorFlow/Keras memory
# across repeated in-process calls; a "base" config invocation doing 3 seeds
# x 2 calls (random+optimal) in one process grew to 11-14GB and got OOM-killed
# 3 times during the first run, all silently recorded as "Completed" because
# the old script never checked python3's exit code. This version: (1) isolates
# every seed as its own process via --seed-start/--n-seeds 1, so no process
# can accumulate more than ~2 run_exp calls, (2) verifies actual output (exit
# code + expected row count) before logging success, (3) skips any (task, k,
# config, seed) combo that already has a valid 51-row seed CSV, so reruns
# only backfill what's missing instead of redoing everything.

set -e
set -o pipefail

TASKS=("ACSIncome" "ACSEmployment" "ACSPublicCoverage" "ACSTravelTime")
K_VALUES=(5 10 15)
SEEDS=(0 1 2)
FEDERATIONS_CSV="${FEDERATIONS_CSV:-results/full_20260918_c1anneal/all_federations.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-Convergence2}"
JOBS="${JOBS:-2}"

cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
FED_CSV_ABS="${PROJECT_ROOT}/${FEDERATIONS_CSV}"
OUT_DIR_ABS="${PROJECT_ROOT}/${OUTPUT_DIR}"
LOG_DIR="${PROJECT_ROOT}/results/fl_baseline_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo "FL Baseline Training (per-seed isolated) - Started $(date)"
echo "Tasks: ${TASKS[*]}  K: ${K_VALUES[*]}  Seeds: ${SEEDS[*]}  Workers: $JOBS"
echo "Federations CSV: $FED_CSV_ABS"
echo "Output: $OUT_DIR_ABS"
echo "Logs: $LOG_DIR"
echo "======================================================"

# Expected output folder for a given config, relative to
# Convergence2/{task}/fixed_size_{k}/. "base" is split into base_random and
# base_optimal so every process does exactly one run_exp() call -- k=15's
# base config doing both random+optimal in one process (2 calls, 15 clients
# each) still grew to 6-8GB and pushed the system back to the swap-thrashing
# edge, even with per-seed isolation. One call per process is the level that
# actually held up reliably (matches the PFL dual-weight validation fix).
config_folder() {
    case "$1" in
        base_random)  echo "Random_FedAvg" ;;
        base_optimal) echo "Optimal_FedAvg" ;;
        scaffold)     echo "Random_SCAFFOLD" ;;
        fedprox)      echo "Random_FedProx" ;;
        ucb)          echo "Random_FedAvg_UCB" ;;
        fedsampling)  echo "Random_FedAvg_FedSamp" ;;
        threshold)    echo "Random_FedAvg_Threshold" ;;
    esac
}
config_extra_args() {
    case "$1" in
        base_random)  echo "--skip-optimal" ;;
        base_optimal) echo "--only-optimal" ;;
        scaffold)     echo "--random-agg scaffold --skip-optimal" ;;
        fedprox)      echo "--random-agg fedprox --skip-optimal" ;;
        ucb)          echo "--client-selection ucb --skip-optimal" ;;
        fedsampling)  echo "--client-selection fedsampling --skip-optimal" ;;
        threshold)    echo "--client-selection threshold --skip-optimal" ;;
    esac
}

# A (task,k,config,seed) unit is done if its expected seed_N.csv already has
# 51 lines (1 header + 50 rounds).
seed_already_done() {
    local task="$1" k="$2" config="$3" seed="$4"
    local base_path="${OUT_DIR_ABS}/${task}/fixed_size_${k}"
    local folder
    folder="$(config_folder "$config")"
    local f="${base_path}/${folder}/seed_${seed}.csv"
    [ -f "$f" ] && [ "$(wc -l < "$f")" -eq 51 ]
}

run_one_seed() {
    local task="$1"
    local k="$2"
    local config="$3"
    local seed="$4"

    if seed_already_done "$task" "$k" "$config" "$seed"; then
        echo "SKIP (already valid) $task k=$k config=$config seed=$seed"
        return 0
    fi

    local extra_args
    extra_args="$(config_extra_args "$config")"
    local job_log="${LOG_DIR}/${task}_k${k}_${config}_seed${seed}.log"

    if python3 "${PROJECT_ROOT}/FL_training/FolkTables_FL.py" \
        --task "$task" --k "$k" --n-seeds 1 --seed-start "$seed" \
        --federations-csv "$FED_CSV_ABS" --output-dir "$OUT_DIR_ABS" \
        $extra_args \
        > "$job_log" 2>&1
    then
        if seed_already_done "$task" "$k" "$config" "$seed"; then
            echo "OK $task k=$k config=$config seed=$seed (log: $job_log)"
        else
            echo "FAILED (exit 0 but output missing/short) $task k=$k config=$config seed=$seed (log: $job_log)"
        fi
    else
        echo "FAILED (python3 exit $?) $task k=$k config=$config seed=$seed (log: $job_log)"
    fi
}
export -f run_one_seed seed_already_done config_folder config_extra_args
export FED_CSV_ABS OUT_DIR_ABS LOG_DIR PROJECT_ROOT

CONFIGS=(base_random base_optimal scaffold fedprox ucb fedsampling threshold)
for TASK in "${TASKS[@]}"; do
    for K in "${K_VALUES[@]}"; do
        for CFG in "${CONFIGS[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                printf '%s\t%s\t%s\t%s\n' "$TASK" "$K" "$CFG" "$SEED"
            done
        done
    done
done | xargs -P "$JOBS" -n 4 bash -c 'run_one_seed "$0" "$1" "$2" "$3"'

echo ""
echo "======================================================"
echo "All FL baseline training units processed - $(date)"
echo "Results in: $OUT_DIR_ABS"
echo "======================================================"
