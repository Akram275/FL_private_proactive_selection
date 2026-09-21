#!/bin/bash
# Run Simulated Annealing experiments for all tasks and k values, in parallel.
# Uses optimized PFL weights: α_ST=2.0, α_SN=0.8889, β_NN=0.1111, δ_NT=1.3333
#
# Each (task, k) combination is a fully independent SA search over its own
# client population's MI components -- running them on separate cores does
# not touch the SA Markov chain, cooling schedule, or acceptance rule of any
# individual search, so per-search convergence guarantees are unaffected.

set -e
set -o pipefail

# Configuration
TASKS=("ACSIncome" "ACSEmployment" "ACSPublicCoverage" "ACSMobility" "ACSTravelTime")
K_VALUES=(5 10 15)
EPSILON=0.05
RUNS=1  # Single run for each
JOBS="${JOBS:-$(nproc)}"  # Concurrent (task, k) combos; override with JOBS=N ./run_sa_experiments.sh
OUTPUT_DIR="results/sa_optimal_$(date +%Y%m%d)"
PARTIAL_DIR="${OUTPUT_DIR}/partial"
MASTER_CSV="${OUTPUT_DIR}/master_results.csv"

# SA parameters (tuned for good exploration)
SA_TEMP=1.0
SA_COOLING=0.95
SA_MIN_TEMP=1e-4
SA_MAX_ITER=4000
SA_ITER_PER_TEMP=25

echo "========================================"
echo "Simulated Annealing Experiments (parallel, ${JOBS} workers)"
echo "========================================"
echo "Tasks: ${TASKS[*]}"
echo "K values: ${K_VALUES[*]}"
echo "Epsilon: ${EPSILON}"
echo "Runs per config: ${RUNS}"
echo "Output: ${OUTPUT_DIR}"
echo "========================================"

mkdir -p "${OUTPUT_DIR}" "${PARTIAL_DIR}"

LOG_FILE="${OUTPUT_DIR}/run.log"
echo "Starting experiments at $(date)" | tee "${LOG_FILE}"

# Runs one (task, k) combo, writing its own partial CSV/log so concurrent
# jobs never write to the same file (avoids interleaved rows / duplicate
# headers that a shared --master-csv would hit under -P).
run_one() {
    local task="$1"
    local k="$2"
    local partial_csv="${PARTIAL_DIR}/${task}_k${k}.csv"
    local job_log="${PARTIAL_DIR}/${task}_k${k}.log"

    python3 run_optimal_federation.py \
        --task "${task}" \
        --k "${k}" \
        --epsilon "${EPSILON}" \
        --method sa \
        --runs "${RUNS}" \
        --output "${OUTPUT_DIR}" \
        --master-csv "${partial_csv}" \
        --sa-temp "${SA_TEMP}" \
        --sa-cooling "${SA_COOLING}" \
        --sa-min-temp "${SA_MIN_TEMP}" \
        --sa-max-iter "${SA_MAX_ITER}" \
        --sa-iter-per-temp "${SA_ITER_PER_TEMP}" \
        > "${job_log}" 2>&1
    echo "Completed ${task} k=${k} (log: ${job_log})"
}
export -f run_one
export EPSILON RUNS OUTPUT_DIR PARTIAL_DIR SA_TEMP SA_COOLING SA_MIN_TEMP SA_MAX_ITER SA_ITER_PER_TEMP

# Build the (task, k) job list and fan out across $JOBS workers.
for task in "${TASKS[@]}"; do
    for k in "${K_VALUES[@]}"; do
        printf '%s\t%s\n' "${task}" "${k}"
    done
done | xargs -P "${JOBS}" -n 2 bash -c 'run_one "$0" "$1"' | tee -a "${LOG_FILE}"

# Merge per-combo CSVs into one master CSV (header from the first file only).
first=1
for f in "${PARTIAL_DIR}"/*.csv; do
    if [ "${first}" -eq 1 ]; then
        cat "${f}" > "${MASTER_CSV}"
        first=0
    else
        tail -n +2 "${f}" >> "${MASTER_CSV}"
    fi
done

echo ""
echo "========================================"
echo "All experiments completed at $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "Master CSV: ${MASTER_CSV}"
echo "Per-job logs: ${PARTIAL_DIR}/*.log"
echo "========================================"
