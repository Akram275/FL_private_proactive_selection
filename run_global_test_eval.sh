#!/bin/bash
# One-off experiment: k=5 only, every baseline (7 from the main campaign +
# FedBary), single seed, evaluated each round against a FIXED test set drawn
# from ALL 50 states -- not just the training federation's own states. This
# isolates "how good is the federation choice" from "which population is the
# model being tested on", per the discussion that motivated this run.
#
# Output goes to Convergence2_globaltest/ (NOT Convergence2/) since this is a
# different evaluation methodology from the main campaign and must not be
# confused with or overwrite those results.
#
# Same per-run isolation + resume-safe pattern as the other campaign scripts.
# The global eval set (all 50 states, preprocessed) is expensive to build
# (~2-3 min) so it's cached per task in results/global_eval_cache_<task>.pkl
# and reused across all 8 configs for that task.

set -e
set -o pipefail

TASKS=("ACSIncome" "ACSEmployment" "ACSPublicCoverage" "ACSTravelTime")
K=5
SEED=0
MAIN_FEDERATIONS_CSV="${MAIN_FEDERATIONS_CSV:-results/full_20260918_c1anneal/all_federations.csv}"
FEDBARY_FEDERATIONS_CSV="${FEDBARY_FEDERATIONS_CSV:-results/fedbary_federations.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-Convergence2_globaltest}"
JOBS="${JOBS:-1}"

cd "$(dirname "$0")"
PROJECT_ROOT="$(pwd)"
OUT_DIR_ABS="${PROJECT_ROOT}/${OUTPUT_DIR}"
LOG_DIR="${PROJECT_ROOT}/results/globaltest_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================================"
echo "Global-population-test evaluation (k=5 only, single seed) - Started $(date)"
echo "Tasks: ${TASKS[*]}  Workers: $JOBS"
echo "Output: $OUT_DIR_ABS"
echo "Logs: $LOG_DIR"
echo "======================================================"

config_folder() {
    case "$1" in
        base_random)  echo "Random_FedAvg" ;;
        base_optimal) echo "Optimal_FedAvg" ;;
        scaffold)     echo "Random_SCAFFOLD" ;;
        fedprox)      echo "Random_FedProx" ;;
        ucb)          echo "Random_FedAvg_UCB" ;;
        fedsampling)  echo "Random_FedAvg_FedSamp" ;;
        threshold)    echo "Random_FedAvg_Threshold" ;;
        fedbary)      echo "FedBary_FedAvg" ;;
    esac
}
config_extra_args() {
    local task="$1" config="$2"
    case "$config" in
        base_random)  echo "--federations-csv ${PROJECT_ROOT}/${MAIN_FEDERATIONS_CSV} --skip-optimal" ;;
        base_optimal) echo "--federations-csv ${PROJECT_ROOT}/${MAIN_FEDERATIONS_CSV} --only-optimal" ;;
        scaffold)     echo "--federations-csv ${PROJECT_ROOT}/${MAIN_FEDERATIONS_CSV} --random-agg scaffold --skip-optimal" ;;
        fedprox)      echo "--federations-csv ${PROJECT_ROOT}/${MAIN_FEDERATIONS_CSV} --random-agg fedprox --skip-optimal" ;;
        ucb)          echo "--federations-csv ${PROJECT_ROOT}/${MAIN_FEDERATIONS_CSV} --client-selection ucb --skip-optimal" ;;
        fedsampling)  echo "--federations-csv ${PROJECT_ROOT}/${MAIN_FEDERATIONS_CSV} --client-selection fedsampling --skip-optimal" ;;
        threshold)    echo "--federations-csv ${PROJECT_ROOT}/${MAIN_FEDERATIONS_CSV} --client-selection threshold --skip-optimal" ;;
        fedbary)      echo "--federations-csv ${PROJECT_ROOT}/${FEDBARY_FEDERATIONS_CSV} --only-optimal --optimal-folder-name FedBary_FedAvg" ;;
    esac
}

seed_already_done() {
    local task="$1" config="$2"
    local folder
    folder="$(config_folder "$config")"
    local f="${OUT_DIR_ABS}/${task}/fixed_size_${K}/${folder}/seed_${SEED}.csv"
    [ -f "$f" ] && [ "$(wc -l < "$f")" -eq 51 ]
}

run_one_config() {
    local task="$1" config="$2"

    if seed_already_done "$task" "$config"; then
        echo "SKIP (already valid) $task config=$config"
        return 0
    fi

    local extra_args
    extra_args="$(config_extra_args "$task" "$config")"
    local job_log="${LOG_DIR}/${task}_${config}.log"
    local eval_cache="${PROJECT_ROOT}/results/global_eval_cache_${task}.pkl"

    if python3 "${PROJECT_ROOT}/FL_training/FolkTables_FL.py" \
        --task "$task" --k "$K" --n-seeds 1 --seed-start "$SEED" \
        --output-dir "$OUT_DIR_ABS" \
        --global-test --global-eval-cache "$eval_cache" \
        $extra_args \
        > "$job_log" 2>&1
    then
        if seed_already_done "$task" "$config"; then
            echo "OK $task config=$config (log: $job_log)"
        else
            echo "FAILED (exit 0 but output missing/short) $task config=$config (log: $job_log)"
        fi
    else
        echo "FAILED (python3 exit $?) $task config=$config (log: $job_log)"
    fi
}
export -f run_one_config seed_already_done config_folder config_extra_args
export MAIN_FEDERATIONS_CSV FEDBARY_FEDERATIONS_CSV OUT_DIR_ABS LOG_DIR PROJECT_ROOT K SEED

CONFIGS=(base_optimal base_random scaffold fedprox ucb fedsampling threshold)
for TASK in "${TASKS[@]}"; do
    for CFG in "${CONFIGS[@]}"; do
        printf '%s\t%s\n' "$TASK" "$CFG"
    done
done | xargs -P "$JOBS" -n 2 bash -c 'run_one_config "$0" "$1"'

echo ""
echo "======================================================"
echo "All global-test evaluation units processed - $(date)"
echo "Results in: $OUT_DIR_ABS"
echo "======================================================"
