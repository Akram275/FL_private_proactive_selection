# FL Private Proactive Selection

**Federated learning with privacy-aware, MI-driven client selection using differential privacy.**

This repository contains the full pipeline accompanying the paper:

> *Distributed Data Valuation for Fair and Private Federated Learning*

---

## Overview

The project has three main components:

| Component | Entry point | Description |
|---|---|---|
| **Optimal federation search** | `run_optimal_federation.py` | Simulated-annealing search over US state subsets, maximising a DP-noisy mutual information objective |
| **FL training** | `FolkTables_FL.py` | FedAvg training on ACS Folktables tasks using the selected federation |
| **DP privacy audit** | `dp_contingency_audit.py` | LRT-based membership inference audit on the released noisy contingency tables |

---

## Repository Structure

```
.
├── run_optimal_federation.py   # Federation search entry point
├── task_config.py              # ACS task definitions and DP parameters
├── mi_utils.py                 # DP-noisy mutual information engine
├── optimization.py             # Simulated annealing + greedy selection
├── reporting.py                # Result formatting
│
├── FolkTables_FL.py            # FL training loop (FedAvg)
├── acs_preprocessing.py        # ACS feature encoding and scaling
├── pfl_from_dataframe.py       # FL data partitioning helpers
├── fl_aggregation.py           # Server-side aggregation methods
├── client_selection.py         # Per-round client selection policies
│
├── dp_contingency_audit.py     # Standalone DP privacy audit (LRT)
│
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Akram275/FL_private_proactive_selection.git
cd FL_private_proactive_selection
pip install -r requirements.txt
```

ACS data is downloaded automatically by [folktables](https://github.com/socialfoundations/folktables) on first run.

---

## Usage

### 1. Find the optimal federation (simulated annealing + DP)

```bash
python run_optimal_federation.py \
    --task ACSIncome \
    --k 5 \
    --epsilon 1.0 \
    --delta 1e-5 \
    --method sa \
    --output results/
```

This produces `results/best_federations.csv` consumed by the FL training script.

### 2. Run federated training

The FL script supports multiple aggregation algorithms and client-selection strategies. Below is the full set of commands for **k=15** across all datasets (repeat analogously for k=5 and k=10 by changing `--k`).

```bash
# ============================================================
# Full Pipeline for k=15 - All Datasets
# ============================================================

# ---------------------------------------------------------
# ACSIncome k=15
# ---------------------------------------------------------
# 1. Optimal (FedAvg) + Random (FedAvg)
python3 FolkTables_FL.py --task ACSIncome --k 15 --n-seeds 3

# 2. Random with SCAFFOLD
python3 FolkTables_FL.py --task ACSIncome --k 15 --random-agg scaffold --skip-optimal --n-seeds 3

# 3. Random with FedProx
python3 FolkTables_FL.py --task ACSIncome --k 15 --random-agg fedprox --skip-optimal --n-seeds 3

# 4. Random with UCB-CS + FedAvg
python3 FolkTables_FL.py --task ACSIncome --k 15 --client-selection ucb --skip-optimal --n-seeds 3

# 5. Random with FedSampling + FedAvg
python3 FolkTables_FL.py --task ACSIncome --k 15 --client-selection fedsampling --skip-optimal --n-seeds 3

# 6. Random with Threshold + FedAvg
python3 FolkTables_FL.py --task ACSIncome --k 15 --client-selection threshold --skip-optimal --n-seeds 3

# ---------------------------------------------------------
# ACSEmployment k=15
# ---------------------------------------------------------
python3 FolkTables_FL.py --task ACSEmployment --k 15 --n-seeds 3
python3 FolkTables_FL.py --task ACSEmployment --k 15 --random-agg scaffold --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSEmployment --k 15 --random-agg fedprox --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSEmployment --k 15 --client-selection ucb --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSEmployment --k 15 --client-selection fedsampling --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSEmployment --k 15 --client-selection threshold --skip-optimal --n-seeds 3

# ---------------------------------------------------------
# ACSMobility k=15
# ---------------------------------------------------------
python3 FolkTables_FL.py --task ACSMobility --k 15 --n-seeds 3
python3 FolkTables_FL.py --task ACSMobility --k 15 --random-agg scaffold --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSMobility --k 15 --random-agg fedprox --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSMobility --k 15 --client-selection ucb --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSMobility --k 15 --client-selection fedsampling --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSMobility --k 15 --client-selection threshold --skip-optimal --n-seeds 3

# ---------------------------------------------------------
# ACSPublicCoverage k=15
# ---------------------------------------------------------
python3 FolkTables_FL.py --task ACSPublicCoverage --k 15 --n-seeds 3
python3 FolkTables_FL.py --task ACSPublicCoverage --k 15 --random-agg scaffold --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSPublicCoverage --k 15 --random-agg fedprox --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSPublicCoverage --k 15 --client-selection ucb --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSPublicCoverage --k 15 --client-selection fedsampling --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSPublicCoverage --k 15 --client-selection threshold --skip-optimal --n-seeds 3

# ---------------------------------------------------------
# ACSTravelTime k=15
# ---------------------------------------------------------
python3 FolkTables_FL.py --task ACSTravelTime --k 15 --n-seeds 3
python3 FolkTables_FL.py --task ACSTravelTime --k 15 --random-agg scaffold --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSTravelTime --k 15 --random-agg fedprox --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSTravelTime --k 15 --client-selection ucb --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSTravelTime --k 15 --client-selection fedsampling --skip-optimal --n-seeds 3
python3 FolkTables_FL.py --task ACSTravelTime --k 15 --client-selection threshold --skip-optimal --n-seeds 3
```

### 3. Run the DP privacy audit

```bash
python dp_contingency_audit.py \
    --tasks ACSIncome ACSEmployment ACSPublicCoverage ACSTravelTime \
    --states CA TX NY FL \
    --epsilons 0.1 0.5 1.0 2.0 5.0 inf \
    --output-dir results/dp_audit/
```

### Execution options

To see all options at runtime:

```bash
python run_optimal_federation.py --help
python FolkTables_FL.py --help
python dp_contingency_audit.py --help
```

`run_optimal_federation.py` (federation search)

| Option | Meaning |
|---|---|
| `--task`, `-t` | Task name (`ACSIncome`, `ACSEmployment`, `ACSPublicCoverage`, `ACSMobility`, `ACSTravelTime`) |
| `--k`, `-k` | Federation size |
| `--epsilon`, `-e` | DP epsilon (`inf` for no-noise baseline) |
| `--delta`, `-d` | DP delta |
| `--method`, `-m` | Search method: `greedy`, `sa`, `both` |
| `--runs`, `-r` | Number of SA runs |
| `--states`, `-s` | Comma-separated subset of states |
| `--year`, `-y` | ACS survey year |
| `--output`, `-o` | Output directory |
| `--sa-temp` | SA initial temperature |
| `--sa-cooling` | SA cooling rate |
| `--sa-min-temp` | SA minimum temperature |
| `--sa-max-iter` | SA maximum iterations |
| `--sa-iter-per-temp` | SA iterations per temperature level |
| `--master-csv` | Append run summaries to a master CSV |
| `--list-tasks` | Print available tasks and exit |
| `--task-info` | Print detailed info for one task |
| `--quiet`, `-q` | Reduce console output |

`FolkTables_FL.py` (FL training)

| Option | Meaning |
|---|---|
| `--task` | Task to run (or all tasks if omitted) |
| `--k` | Federation size to run (or all available values if omitted) |
| `--federations-csv` | Federation file path (for optimal state sets) |
| `--output-dir` | Directory for convergence logs |
| `--n-seeds` | Number of random seeds |
| `--seed-start` | Starting seed index |
| `--list-available` | List available tasks and `k` values from federation CSV |
| `--random-agg` | Aggregator for random federation (`fedavg`, `fedprox`, `fedadam`, `scaffold`, ...) |
| `--skip-optimal` | Skip optimal federation runs |
| `--only-optimal` | Run only optimal federation |
| `--client-selection` | Sampling strategy: `full`, `random`, `ucb`, `threshold`, `power_of_choice`, `fedsampling` |
| `--participation-rate` | Fraction of clients per round |
| `--ucb-exploration` | UCB exploration coefficient |
| `--ucb-loss-decay` | UCB loss decay factor |
| `--threshold-percentile` | Percentile cutoff for threshold selector |
| `--threshold-theta` | Mean reversion parameter for threshold selector |
| `--poc-d-choices` | Number of candidate clients in power-of-choice |
| `--fedprox-mu` | FedProx proximal coefficient |
| `--fedadam-server-lr` | FedAdam server learning rate |
| `--fedadam-beta1` | FedAdam beta1 |
| `--fedadam-beta2` | FedAdam beta2 |
| `--fedadam-tau` | FedAdam tau |
| `--scaffold-local-lr` | SCAFFOLD local learning rate |

`dp_contingency_audit.py` (DP privacy audit)

| Option | Meaning |
|---|---|
| `--tasks` | Tasks to audit |
| `--states` | States to audit |
| `--epsilons` | Epsilon grid (`inf` supported) |
| `--delta` | DP delta |
| `--fpr` | Target FPR for TPR@FPR |
| `--survey-year` | ACS survey year |
| `--horizon` | ACS horizon (`1-Year`, etc.) |
| `--max-records-per-state` | Sampling cap per state/task |
| `--seed` | Random seed |
| `--joint-scale-with-m` | Scale joint-table sigma by sqrt(number of tables) |
| `--exact-total-knowledge` | Assume attacker knows exact total count `N` |
| `--output-dir` | Output directory for plots and CSV metrics |

---

## Privacy Guarantee

The server releases one noisy contingency table per feature per client, calibrated with the Gaussian mechanism:

$$\sigma = \frac{\Delta f \cdot \sqrt{2 \ln(1.25/\delta)}}{\varepsilon}$$

The audit confirms that at $\varepsilon_1 = 1.0$ the LRT membership inference attack is indistinguishable from random guessing (AUC $\approx 0.5$, TPR@1%FPR $\approx 1\%$) across all four ACS tasks and 50 US states.

---

## License

MIT
