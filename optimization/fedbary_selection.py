"""
FedBary federation-selection baseline.

Reproduces the core idea of Li et al., "Data Valuation and Detections in
Federated Learning" (arXiv:2311.05304): value each client (here, a US state)
by its Wasserstein distance to a target distribution Q, then select the
states with the smallest distance.

Three deliberate deviations from the paper, needed to make it run at this
project's data scale (tens of thousands of rows per state, vs. the paper's
own much smaller per-client sizes):

1. Sinkhorn OT (entropic regularization, via the POT library) instead of
   the paper's exact network-simplex OT. Exact OT is O(m^3 log m) per
   client per iteration, which is infeasible here; Sinkhorn is the
   standard tractable substitute.
2. Distances are computed directly (barycenter, then per-state distance to
   it) rather than by replaying the paper's iterative client-server
   "interpolating measure" exchange (their Algorithm 1). That protocol
   exists to let two parties approximate a Wasserstein distance without
   either one seeing the other's raw samples over a real network. This is
   a single-machine research simulation with direct access to every
   state's data already (same convention this repo's SA/PFL search uses),
   so the direct computation is equivalent in effect and far simpler.
3. No validation dataset exists in this pipeline, so we use the paper's
   own "Without Validation data" setting (Section 5.1): Q is the
   approximate Wasserstein barycenter of all N=50 candidate states'
   distributions, and a state's value is the *inverse* of its distance to
   that barycenter -- i.e. states closest to the collective distribution
   are considered most relevant, states far from it are treated as
   outliers/noisy (mirroring the paper's own case-5 experiment).

Each state's local distribution is also capped to SUBSAMPLE_SIZE rows
before running OT, purely to keep the O(m^2) Sinkhorn cost matrices a
tractable size in memory and time.

Usage:
    python optimization/fedbary_selection.py --task ACSIncome --k 5
    python optimization/fedbary_selection.py --task ACSIncome --k 5 --k 10 --k 15
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_training"))

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.linalg import sqrtm
import ot

from folktables import ACSDataSource
from FolkTables_FL import TASK_OBJECTS
from pfl_from_dataframe import all_states
from acs_preprocessing import preprocess_acs_data

SUBSAMPLE_SIZE = 1500
SINKHORN_REG = 0.05
BARYCENTER_SUPPORT_SIZE = 300
BARYCENTER_ITERS = 8
RNG_SEED = 0


def load_all_states_features(task: str, states: list, rng: np.random.Generator) -> dict:
    """Download + preprocess every state's data together (so one-hot columns
    line up across states), then split back apart and subsample per state."""
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    task_obj = TASK_OBJECTS[task]

    raw_features, labels, counts = [], [], []
    for state in states:
        acs_data = data_source.get_data(states=[state], download=True)
        features, lbl, _ = task_obj.df_to_pandas(acs_data)
        raw_features.append(features)
        labels.append(np.asarray(lbl).astype(int).ravel())
        counts.append(len(features))

    combined_raw = pd.concat(raw_features, ignore_index=True)
    combined_processed = preprocess_acs_data(combined_raw, verbose=False)
    X = combined_processed.values.astype(float)

    per_state = {}
    start = 0
    for state, n, y in zip(states, counts, labels):
        end = start + n
        Xi, yi = X[start:end], y[:n]
        if len(Xi) > SUBSAMPLE_SIZE:
            idx = rng.choice(len(Xi), size=SUBSAMPLE_SIZE, replace=False)
            Xi, yi = Xi[idx], yi[idx]
        per_state[state] = (Xi, yi)
        start = end
    return per_state


def augment(Xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
    """Stack raw features with per-class Gaussian stats (mean + sqrt-cov),
    matching the augmented distance of Eq. 8 in the paper."""
    n, d = Xi.shape
    aug = np.zeros((n, d + d + d * d))
    for cls in np.unique(yi):
        mask = yi == cls
        Xc = Xi[mask]
        mean = Xc.mean(axis=0)
        cov = np.cov(Xc, rowvar=False) if len(Xc) > 1 else np.zeros((d, d))
        cov = cov + np.eye(d) * 1e-6
        cov_sqrt = np.real(sqrtm(cov)).ravel()
        aug[mask] = np.concatenate([
            Xi[mask],
            np.tile(mean, (mask.sum(), 1)),
            np.tile(cov_sqrt, (mask.sum(), 1)),
        ], axis=1)
    return aug


def sinkhorn_distance(A: np.ndarray, B: np.ndarray, reg: float = SINKHORN_REG) -> float:
    a = np.full(len(A), 1.0 / len(A))
    b = np.full(len(B), 1.0 / len(B))
    M = ot.dist(A, B, metric='sqeuclidean')
    M = M / (M.max() + 1e-12)
    return float(ot.sinkhorn2(a, b, M, reg))


def wasserstein_barycenter(point_sets: list, rng: np.random.Generator) -> np.ndarray:
    """Approximate free-support Wasserstein barycenter of several point clouds."""
    all_points = np.concatenate(point_sets, axis=0)
    support = min(BARYCENTER_SUPPORT_SIZE, len(all_points))
    init_idx = rng.choice(len(all_points), size=support, replace=False)
    X_init = all_points[init_idx]
    measure_weights = [np.full(len(p), 1.0 / len(p)) for p in point_sets]
    b_init = np.full(support, 1.0 / support)
    return ot.lp.free_support_barycenter(
        point_sets, measure_weights, X_init, b_init, numItermax=BARYCENTER_ITERS
    )


def compute_state_distances(task: str, states: list = None) -> dict:
    """Compute every state's FedBary distance to the no-validation-set
    barycenter. This ranking doesn't depend on k, so it's computed once per
    task and reused for every federation size."""
    states = states or all_states
    rng = np.random.default_rng(RNG_SEED)

    print(f"[{task}] Loading + preprocessing {len(states)} states...")
    per_state = load_all_states_features(task, states, rng)

    print(f"[{task}] Building augmented per-class distributions...")
    augmented = {s: augment(X, y) for s, (X, y) in per_state.items()}

    print(f"[{task}] Approximating Wasserstein barycenter Q (no validation set)...")
    Q = wasserstein_barycenter(list(augmented.values()), rng)

    print(f"[{task}] Computing Sinkhorn distance of each state to Q...")
    distances = {s: sinkhorn_distance(A, Q) for s, A in augmented.items()}
    return distances


def select_top_k(distances: dict, k: int) -> list:
    """Smallest distance = most valuable (closest to the collective / least
    outlier-like, per the paper's no-validation-set interpretation)."""
    ranked = sorted(distances, key=distances.get)
    return ranked[:k]


def append_to_csv(csv_path: str, task: str, k: int, selected: list):
    row = pd.DataFrame([{'task': task, 'k': k, 'selected_states': ','.join(selected)}])
    if os.path.exists(csv_path):
        existing = pd.read_csv(csv_path)
        existing = existing[~((existing['task'] == task) & (existing['k'] == k))]
        row = pd.concat([existing, row], ignore_index=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True)
    parser.add_argument('--k', type=int, action='append', required=True,
                         help='Federation size; repeat --k to run several sizes')
    parser.add_argument('--output-csv', default='results/fedbary_federations.csv')
    args = parser.parse_args()

    distances = compute_state_distances(args.task)
    ranked = sorted(distances, key=distances.get)
    print(f"[{args.task}] Full ranking (smallest distance = most valuable):")
    for s in ranked:
        print(f"    {s}: {distances[s]:.5f}")

    for k in args.k:
        selected = select_top_k(distances, k)
        print(f"[{args.task} k={k}] Selected federation: {selected}")
        append_to_csv(args.output_csv, args.task, k, selected)


if __name__ == '__main__':
    main()
