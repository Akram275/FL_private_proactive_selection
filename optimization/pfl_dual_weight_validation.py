"""
PFL Dual-Weight Validation Experiment

Validates whether PFL remains a meaningful proxy for real FL training outcomes
under the fixed, arbitrary weights (1.0 fairness / 1.0 utility) now used as
the deployed default, compared against the previously calibrated weights --
without needing to re-run FL training twice (the training itself doesn't
depend on PFL weights, only the score computed from its resulting stats does).

For each of N randomly sampled federations of size k:
1. Load the federation's combined raw data once.
2. Compute PFL under (a) the calibrated weights and (b) the arbitrary weights,
   from that same combined data (no re-loading).
3. Train FL once (n_seeds runs, averaged), collect utility + fairness metrics.
4. Save incrementally; plot PFL vs metrics and report Spearman correlations
   for both weight sets side by side.

Usage:
    python pfl_dual_weight_validation.py --task ACSIncome --k 5 --n-federations 20
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_training"))

import argparse
import json
import random
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from folktables import ACSDataSource
from pfl_from_dataframe import compute_PFL_of_dataframe, all_states
from FolkTables_FL import TASK_OBJECTS

_WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_fl_train_worker.py")


def run_federation_experiment_subprocess(states: list, task: str, n_seeds: int, max_iterations: int) -> dict:
    """
    Runs FL training for one federation in a fresh subprocess (see
    _fl_train_worker.py for why: run_exp/Keras leaks memory across repeated
    in-process calls, and is shared with the production FL training pipeline,
    so isolating it here avoids both the leak and touching that pipeline.
    """
    with tempfile.NamedTemporaryFile(mode='r', suffix='.json', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [sys.executable, _WORKER_SCRIPT,
             '--task', task, '--states', ','.join(states),
             '--n-seeds', str(n_seeds), '--max-iterations', str(max_iterations),
             '--output', tmp_path],
            check=True,
        )
        with open(tmp_path) as f:
            return json.load(f)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# The two weight configurations under comparison
WEIGHTS_CALIBRATED = {'alpha_ST': 2.0, 'alpha_SN': 0.8889, 'beta_NN': 0.1111, 'delta_NT': 1.3333}
WEIGHTS_ARBITRARY = {'alpha_ST': 1.0, 'alpha_SN': 1.0, 'beta_NN': 1.0, 'delta_NT': 1.0}


def load_combined_federation_df(states: list, task: str) -> pd.DataFrame:
    """Load and combine raw ACS data for a federation once (reused for both PFL evaluations)."""
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    task_obj = TASK_OBJECTS[task]

    all_features, all_labels = [], []
    for state in states:
        acs_data = data_source.get_data(states=[state], download=True)
        features, labels, _ = task_obj.df_to_pandas(acs_data)
        all_features.append(features)
        all_labels.append(labels)

    df = pd.concat(all_features, ignore_index=True)
    df['label'] = pd.concat(all_labels, ignore_index=True).values
    return df


def plot_dual_scatter(ax, x, y, yerr, xlabel, ylabel, title, color):
    ax.errorbar(x, y, yerr=yerr, fmt='o', color=color, alpha=0.7, markersize=7, capsize=3, elinewidth=1)
    slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
    x_line = np.linspace(min(x), max(x), 100)
    ax.plot(x_line, slope * x_line + intercept, '--', color='red', linewidth=1.5)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    textstr = f'Pearson r = {r_value:.3f}\nSpearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    return r_value, p_value, spearman_r, spearman_p


def create_dual_validation_plots(df: pd.DataFrame, output_path: Path, task: str, k: int):
    metrics = [
        ('loss_mean', 'loss_std', 'Final Validation Loss', 'coral'),
        ('accuracy_mean', 'accuracy_std', 'Final Validation Accuracy', 'seagreen'),
        ('eod_mean', 'eod_std', '|EOD|', 'steelblue'),
        ('spd_mean', 'spd_std', '|SPD|', 'purple'),
    ]
    correlations = {}
    for weight_label, pfl_col in [('calibrated', 'pfl_calibrated'), ('arbitrary', 'pfl_arbitrary')]:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        pfl = df[pfl_col].values
        for ax, (mcol, ecol, mlabel, color) in zip(axes.flat, metrics):
            r, p, sr, sp = plot_dual_scatter(
                ax, pfl, df[mcol].values, df[ecol].values,
                xlabel=f'PFL ({weight_label} weights)', ylabel=mlabel,
                title=f'{mlabel} vs PFL ({weight_label})', color=color,
            )
            correlations[f'{weight_label}_{mcol.replace("_mean","")}_pearson_r'] = r
            correlations[f'{weight_label}_{mcol.replace("_mean","")}_pearson_p'] = p
            correlations[f'{weight_label}_{mcol.replace("_mean","")}_spearman_r'] = sr
            correlations[f'{weight_label}_{mcol.replace("_mean","")}_spearman_p'] = sp
        plt.suptitle(f'PFL Validation: {task} (k={k}) -- {weight_label} weights', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plot_path = output_path / f'scatter_plots_{weight_label}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to {plot_path}")
    return correlations


def run(task: str, k: int, n_federations: int, n_seeds: int, max_iterations: int, output_dir: str):
    output_path = Path(output_dir) / task / f'k={k}_dual_weight'
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / 'results.csv'

    start_id = 0
    already_used = set()
    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        if len(existing) > 0:
            start_id = existing['federation_id'].max() + 1
            already_used = {frozenset(s.split(',')) for s in existing['states']}
        print(f"Appending to existing file with {len(existing)} federations (starting at id={start_id})")

    # Dedicated RNG for federation sampling, seeded from OS entropy -- NOT the
    # global numpy RNG, which run_federation_experiment() reseeds deterministically
    # (np.random.seed(0..n_seeds-1)) for FL training reproducibility on every
    # federation. Sampling off the global RNG would make the federation sequence
    # itself deterministic across fresh process restarts, silently replaying the
    # same federations after an interrupt-and-resume instead of extending coverage.
    sampling_rng = np.random.default_rng()

    def sample_unique_federation(k: int) -> list:
        while True:
            states = sorted(sampling_rng.choice(all_states, size=k, replace=False).tolist())
            if frozenset(states) not in already_used:
                already_used.add(frozenset(states))
                return states

    print("=" * 70)
    print(f"PFL Dual-Weight Validation: task={task} k={k} n_federations={n_federations} "
          f"n_seeds={n_seeds} max_iterations={max_iterations}")
    print(f"Calibrated weights: {WEIGHTS_CALIBRATED}")
    print(f"Arbitrary weights:  {WEIGHTS_ARBITRARY}")
    print("=" * 70)

    for i in tqdm(range(n_federations), desc="Federations"):
        states = sample_unique_federation(k)
        states_str = ','.join(states)
        print(f"\n[{i+1}/{n_federations}] Federation: {states}")

        print("  Loading combined data + computing PFL under both weight sets...")
        df = load_combined_federation_df(states, task)
        pfl_calibrated = compute_PFL_of_dataframe(df, weights=WEIGHTS_CALIBRATED)
        pfl_arbitrary = compute_PFL_of_dataframe(df, weights=WEIGHTS_ARBITRARY)
        print(f"  PFL (calibrated) = {pfl_calibrated:.4f}  |  PFL (arbitrary) = {pfl_arbitrary:.4f}")

        print("  Training FL model...")
        metrics = run_federation_experiment_subprocess(states=states, task=task, n_seeds=n_seeds, max_iterations=max_iterations)

        result = {
            'federation_id': start_id + i,
            'states': states_str,
            'k': k,
            'pfl_calibrated': pfl_calibrated,
            'pfl_arbitrary': pfl_arbitrary,
            **metrics,
        }
        result_df = pd.DataFrame([result])
        result_df.to_csv(csv_path, index=False, mode='a' if csv_path.exists() else 'w',
                          header=not csv_path.exists())
        os.sync()
        print(f"  Metrics: Acc={metrics['accuracy_mean']:.4f} EOD={metrics['eod_mean']:.4f} "
              f"SPD={metrics['spd_mean']:.4f}  (saved to {csv_path})")

    results_df = pd.read_csv(csv_path)
    print(f"\nTotal federations in {csv_path}: {len(results_df)}")

    correlations = create_dual_validation_plots(results_df, output_path, task, k)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = {
        'task': task, 'k': k, 'n_federations': len(results_df), 'n_seeds': n_seeds,
        'max_iterations': max_iterations,
        'weights_calibrated': WEIGHTS_CALIBRATED, 'weights_arbitrary': WEIGHTS_ARBITRARY,
        'correlations': correlations, 'timestamp': timestamp,
    }
    summary_path = output_path / f'summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SPEARMAN CORRELATION SUMMARY (calibrated vs arbitrary weights)")
    print("=" * 70)
    print(f"{'Metric':<12} {'Calibrated ρ':<15} {'p':<12} {'Arbitrary ρ':<15} {'p':<12}")
    for metric in ['loss', 'accuracy', 'eod', 'spd']:
        cr = correlations[f'calibrated_{metric}_spearman_r']
        cp = correlations[f'calibrated_{metric}_spearman_p']
        ar = correlations[f'arbitrary_{metric}_spearman_r']
        ap = correlations[f'arbitrary_{metric}_spearman_p']
        print(f"{metric:<12} {cr:<15.3f} {cp:<12.2e} {ar:<15.3f} {ap:<12.2e}")
    print("=" * 70)

    return results_df, correlations


def main():
    parser = argparse.ArgumentParser(description='PFL Dual-Weight Validation Experiment')
    parser.add_argument('--task', type=str, default='ACSIncome',
                        choices=['ACSIncome', 'ACSEmployment', 'ACSPublicCoverage', 'ACSMobility', 'ACSTravelTime'])
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--n-federations', type=int, default=20)
    parser.add_argument('--n-seeds', type=int, default=3)
    parser.add_argument('--max-iterations', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='PFL_validation')
    args = parser.parse_args()

    run(task=args.task, k=args.k, n_federations=args.n_federations, n_seeds=args.n_seeds,
        max_iterations=args.max_iterations, output_dir=args.output_dir)


if __name__ == '__main__':
    main()
