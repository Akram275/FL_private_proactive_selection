#!/bin/bash
# Full Federation Search Script, in parallel.
# Runs SA-based optimal federation selection for all tasks and sizes.
#
# Each (task, k) combination is a fully independent SA search (its own
# n_runs=SA_RUNS multi-start loop over its own client population) -- running
# combos on separate cores does not touch the SA Markov chain, cooling
# schedule, or acceptance rule of any individual search, so per-search
# convergence guarantees are unaffected.

set -e
set -o pipefail

# Configuration -- matches paper Table III/IV exactly (4 tasks, no ACSMobility)
TASKS=("ACSIncome" "ACSEmployment" "ACSPublicCoverage" "ACSTravelTime")
K_VALUES=(5 10 15)
EPSILON=1.0
SA_RUNS=1  # PFL std was ~0 across 5 runs at the fully-annealed schedule below; 1 run suffices
JOBS="${JOBS:-$(nproc)}"  # Concurrent (task, k) combos; override with JOBS=N ./run_full_exp.sh
N_STATES=50

# SA Parameters (Table III base), with iterations_per_temp and max_iterations
# scaled per k (computed in run_one): iter_per_temp(k) = k*(N_STATES-k) (c=1,
# literature-grounded neighborhood coverage), max_iter(k) = iter_per_temp(k) *
# LEVELS, where LEVELS = ceil(log(min_temp/T0)/log(cooling)) is the number of
# cooling steps needed to actually reach min_temp -- this restores a full
# anneal at the larger per-level sweep instead of truncating mid-cooling.
SA_TEMP=1.0
SA_COOLING=0.98
SA_MIN_TEMP=1e-4
LEVELS=456

# Output
OUTPUT_DIR="results/full_$(date +%Y%m%d)_c1anneal_arbweights"
PARTIAL_DIR="${OUTPUT_DIR}/partial"
MASTER_CSV="${OUTPUT_DIR}/all_federations.csv"
LOG_FILE="${OUTPUT_DIR}/run.log"

# Create output directory
mkdir -p "$OUTPUT_DIR" "$PARTIAL_DIR"

echo "======================================================" | tee -a "$LOG_FILE"
echo "Full Federation Search - Started $(date)" | tee -a "$LOG_FILE"
echo "Tasks: ${TASKS[*]}" | tee -a "$LOG_FILE"
echo "K values: ${K_VALUES[*]}" | tee -a "$LOG_FILE"
echo "Epsilon: $EPSILON" | tee -a "$LOG_FILE"
echo "Workers: $JOBS" | tee -a "$LOG_FILE"
echo "Output: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"

# Runs one (task, k) combo, writing its own partial CSV/log so concurrent
# jobs never write to the same file (avoids interleaved rows / duplicate
# headers that a shared --master-csv would hit under -P).
run_one() {
    local task="$1"
    local k="$2"
    local iter_per_temp=$(( k * (N_STATES - k) ))
    local max_iter=$(( iter_per_temp * LEVELS ))
    local partial_csv="${PARTIAL_DIR}/${task}_k${k}.csv"
    local job_log="${PARTIAL_DIR}/${task}_k${k}.log"

    python3 run_optimal_federation.py \
        --task "$task" \
        --k "$k" \
        --epsilon "$EPSILON" \
        --method sa \
        --runs "$SA_RUNS" \
        --sa-temp "$SA_TEMP" \
        --sa-cooling "$SA_COOLING" \
        --sa-min-temp "$SA_MIN_TEMP" \
        --sa-max-iter "$max_iter" \
        --sa-iter-per-temp "$iter_per_temp" \
        --output "$OUTPUT_DIR" \
        --master-csv "$partial_csv" \
        --quiet \
        > "$job_log" 2>&1
    echo "Completed $task k=$k (iter_per_temp=$iter_per_temp, max_iter=$max_iter, log: $job_log)"
}
export -f run_one
export EPSILON SA_RUNS SA_TEMP SA_COOLING SA_MIN_TEMP LEVELS N_STATES OUTPUT_DIR PARTIAL_DIR

# Build the (task, k) job list and fan out across $JOBS workers.
for TASK in "${TASKS[@]}"; do
    for K in "${K_VALUES[@]}"; do
        printf '%s\t%s\n' "$TASK" "$K"
    done
done | xargs -P "$JOBS" -n 2 bash -c 'run_one "$0" "$1"' | tee -a "$LOG_FILE"

# Merge per-combo CSVs into one master CSV (header from the first file only).
first=1
for f in "${PARTIAL_DIR}"/*.csv; do
    if [ "$first" -eq 1 ]; then
        cat "$f" > "$MASTER_CSV"
        first=0
    else
        tail -n +2 "$f" >> "$MASTER_CSV"
    fi
done

echo "" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
echo "All runs completed - $(date)" | tee -a "$LOG_FILE"
echo "Results saved to: $MASTER_CSV" | tee -a "$LOG_FILE"
echo "======================================================" | tee -a "$LOG_FILE"
