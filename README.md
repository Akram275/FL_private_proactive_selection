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
    --output-dir results/
```

This produces `results/best_federations.csv` consumed by the FL training script.

### 2. Run federated training

```bash
python FolkTables_FL.py \
    --task ACSIncome \
    --federations-csv results/best_federations2.csv \
    --rounds 50 \
    --output-dir Convergence2/
```

### 3. Run the DP privacy audit

```bash
python dp_contingency_audit.py \
    --tasks ACSIncome ACSEmployment ACSPublicCoverage ACSTravelTime \
    --states CA TX NY FL \
    --epsilons 0.1 0.5 1.0 2.0 5.0 inf \
    --output-dir results/dp_audit/
```

---

## Privacy Guarantee

The server releases one noisy contingency table per feature per client, calibrated with the Gaussian mechanism:

$$\sigma = \frac{\Delta f \cdot \sqrt{2 \ln(1.25/\delta)}}{\varepsilon}$$

The audit confirms that at $\varepsilon_1 = 1.0$ the LRT membership inference attack is indistinguishable from random guessing (AUC $\approx 0.5$, TPR@1%FPR $\approx 1\%$) across all four ACS tasks and 50 US states.

---

## License

MIT
