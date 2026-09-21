"""
PFL Validation Experiment

This script validates that PFL is a reliable measure of federation potential by:
1. Sampling N random federations of size k
2. Computing PFL for each federation (with redundancy weighting β_NN = 0.5)
3. Training FL models on each federation
4. Storing and plotting scatter plots of metrics vs PFL:
   - Final validation loss vs PFL (expect positive correlation)
   - Final validation accuracy vs PFL (expect negative correlation)
   - EOD vs PFL (expect positive correlation - higher PFL = more bias)
   - SPD vs PFL (expect positive correlation - higher PFL = more bias)

Usage:
    python pfl_validation_experiment.py --task ACSIncome --k 5 --n-federations 20 --n-seeds 3
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_training"))

import numpy as np
import pandas as pd
import argparse
import csv
import json
import random
from pathlib import Path
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from folktables import ACSDataSource, ACSIncome, ACSEmployment, ACSPublicCoverage, ACSMobility, ACSTravelTime
from pfl_from_dataframe import compute_PFL_of_dataframe, all_states
from FolkTables_FL import run_exp, TASK_OBJECTS

# Professional plotting style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.dpi'] = 150

# PFL Weights with redundancy penalty (optimized)
PFL_WEIGHTS = {
    'alpha_ST': 2.0,      # Direct bias MI(S, T)
    'alpha_SN': 0.8889,   # Proxy bias MI(S, N_k)
    'beta_NN': 0.1111,    # Feature redundancy MI(N_i, N_j)
    'delta_NT': 1.3333    # Utility reward MI(N_k, T)
}


def sample_random_federation(k: int, exclude: list = None) -> list:
    """Sample a random federation of k states."""
    available = [s for s in all_states if s not in (exclude or [])]
    return list(np.random.choice(available, size=k, replace=False))


def compute_pfl_for_federation(states: list, task: str) -> float:
    """
    Compute PFL score for a federation with the updated weights.
    
    Args:
        states: List of state codes
        task: Task name (ACSIncome, ACSEmployment, etc.)
    
    Returns:
        PFL score
    """
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    task_obj = TASK_OBJECTS[task]
    
    # Load and combine data from all states
    all_features = []
    all_labels = []
    
    for state in states:
        acs_data = data_source.get_data(states=[state], download=True)
        features, labels, _ = task_obj.df_to_pandas(acs_data)
        all_features.append(features)
        all_labels.append(labels)
    
    # Combine into single DataFrame
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = pd.concat(all_labels, ignore_index=True)
    
    # Create single DataFrame with label column
    df = combined_features.copy()
    df['label'] = combined_labels.values
    
    # Compute PFL with redundancy weighting
    pfl_score = compute_PFL_of_dataframe(df, weights=PFL_WEIGHTS)
    
    return pfl_score


def run_federation_experiment(states: list, task: str, n_seeds: int = 3, 
                               max_iterations: int = 50) -> dict:
    """
    Train FL model on a federation and collect final metrics.
    
    Args:
        states: List of state codes
        task: Task name
        n_seeds: Number of random seeds to average over
        max_iterations: Number of FL rounds
    
    Returns:
        Dict with averaged final metrics
    """
    all_final_metrics = {
        'loss': [],
        'accuracy': [],
        'recall': [],
        'precision': [],
        'f1': [],
        'eod': [],
        'spd': [],
        'mad': []
    }
    
    for seed in range(n_seeds):
        np.random.seed(seed)
        random.seed(seed)
        
        try:
            # Run FL training
            scores = run_exp(
                task=task,
                states=states,
                epochs=1,
                max_iterations=max_iterations,
                centralized_test=False,
                aggregation_method='fedavg'
            )
            
            # Get final round metrics
            # scores format: [loss, acc, recall, precision, eod, spd, mad]
            final_scores = scores[-1]
            
            all_final_metrics['loss'].append(final_scores[0])
            all_final_metrics['accuracy'].append(final_scores[1])
            all_final_metrics['recall'].append(final_scores[2])
            all_final_metrics['precision'].append(final_scores[3])
            # Compute F1 score from precision and recall
            precision = final_scores[3]
            recall = final_scores[2]
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
            else:
                f1 = 0.0
            all_final_metrics['f1'].append(f1)
            all_final_metrics['eod'].append(abs(final_scores[4]))  # Use absolute value
            all_final_metrics['spd'].append(abs(final_scores[5]))  # Use absolute value
            all_final_metrics['mad'].append(abs(final_scores[6]))  # Use absolute value
            
        except Exception as e:
            print(f"Error training federation {states} seed {seed}: {e}")
            continue
    
    # Average over seeds
    return {
        'loss_mean': np.mean(all_final_metrics['loss']),
        'loss_std': np.std(all_final_metrics['loss']),
        'accuracy_mean': np.mean(all_final_metrics['accuracy']),
        'accuracy_std': np.std(all_final_metrics['accuracy']),
        'f1_mean': np.mean(all_final_metrics['f1']),
        'f1_std': np.std(all_final_metrics['f1']),
        'eod_mean': np.mean(all_final_metrics['eod']),
        'eod_std': np.std(all_final_metrics['eod']),
        'spd_mean': np.mean(all_final_metrics['spd']),
        'spd_std': np.std(all_final_metrics['spd']),
        'mad_mean': np.mean(all_final_metrics['mad']),
        'mad_std': np.std(all_final_metrics['mad']),
        'n_successful_seeds': len(all_final_metrics['loss'])
    }


def plot_scatter_with_correlation(ax, x, y, xerr=None, yerr=None, 
                                   xlabel='PFL', ylabel='Metric',
                                   title='', color='steelblue'):
    """
    Create scatter plot with correlation line and statistics.
    """
    # Scatter points with error bars
    if xerr is not None or yerr is not None:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', color=color, 
                    alpha=0.7, markersize=8, capsize=3, elinewidth=1)
    else:
        ax.scatter(x, y, c=color, alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    
    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_line = np.linspace(min(x), max(x), 100)
    y_line = slope * x_line + intercept
    
    # Correlation line
    ax.plot(x_line, y_line, '--', color='red', linewidth=2, 
            label=f'r = {r_value:.3f}, p = {p_value:.3e}')
    
    # Spearman correlation (rank-based, more robust)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=10)
    
    # Add text box with statistics
    textstr = f'Pearson r = {r_value:.3f}\nSpearman ρ = {spearman_r:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    return r_value, p_value, spearman_r, spearman_p


def create_validation_plots(results_df: pd.DataFrame, output_dir: Path, task: str, k: int):
    """
    Create 2x2 scatter plots of metrics vs PFL.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    pfl = results_df['pfl'].values
    
    # Plot 1: Loss vs PFL (expect positive correlation)
    ax1 = axes[0, 0]
    loss = results_df['loss_mean'].values
    loss_err = results_df['loss_std'].values
    r_loss, p_loss, _, _ = plot_scatter_with_correlation(
        ax1, pfl, loss, yerr=loss_err,
        xlabel='PFL Score', ylabel='Final Validation Loss',
        title='Loss vs PFL (Expected: Positive Correlation)',
        color='coral'
    )
    
    # Plot 2: Accuracy vs PFL (expect negative correlation)
    ax2 = axes[0, 1]
    acc = results_df['accuracy_mean'].values
    acc_err = results_df['accuracy_std'].values
    r_acc, p_acc, _, _ = plot_scatter_with_correlation(
        ax2, pfl, acc, yerr=acc_err,
        xlabel='PFL Score', ylabel='Final Validation Accuracy',
        title='Accuracy vs PFL (Expected: Negative Correlation)',
        color='seagreen'
    )
    
    # Plot 3: EOD vs PFL (expect positive correlation)
    ax3 = axes[1, 0]
    eod = results_df['eod_mean'].values
    eod_err = results_df['eod_std'].values
    r_eod, p_eod, _, _ = plot_scatter_with_correlation(
        ax3, pfl, eod, yerr=eod_err,
        xlabel='PFL Score', ylabel='|EOD| (Equalized Odds Diff)',
        title='EOD vs PFL (Expected: Positive Correlation)',
        color='steelblue'
    )
    
    # Plot 4: SPD vs PFL (expect positive correlation)
    ax4 = axes[1, 1]
    spd = results_df['spd_mean'].values
    spd_err = results_df['spd_std'].values
    r_spd, p_spd, _, _ = plot_scatter_with_correlation(
        ax4, pfl, spd, yerr=spd_err,
        xlabel='PFL Score', ylabel='|SPD| (Statistical Parity Diff)',
        title='SPD vs PFL (Expected: Positive Correlation)',
        color='purple'
    )
    
    plt.suptitle(f'PFL Validation: {task} (k={k})\nPFL Weights: α_ST=2.0, α_SN=0.89, β_NN=0.11, δ_NT=1.33',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = output_dir / f'scatter_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSaved plot to {plot_path}")
    
    return {
        'loss_r': r_loss, 'loss_p': p_loss,
        'accuracy_r': r_acc, 'accuracy_p': p_acc,
        'eod_r': r_eod, 'eod_p': p_eod,
        'spd_r': r_spd, 'spd_p': p_spd
    }


def run_pfl_validation_experiment(task: str, k: int, n_federations: int = 20,
                                   n_seeds: int = 3, max_iterations: int = 50,
                                   output_dir: str = 'PFL_validation'):
    """
    Main experiment function.
    """
    # Create folder structure: PFL_validation/{task}/k={k}/
    output_path = Path(output_dir) / task / f'k={k}'
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Fixed CSV file path - append to existing results
    csv_path = output_path / 'results.csv'
    
    # Check if file exists to determine if we need to write header
    file_exists = csv_path.exists()
    
    # Get starting federation_id from existing file
    start_id = 0
    if file_exists:
        existing_df = pd.read_csv(csv_path)
        if len(existing_df) > 0:
            start_id = existing_df['federation_id'].max() + 1
        print(f"Appending to existing file with {len(existing_df)} federations (starting at id={start_id})")
    
    print("="*70)
    print(f"PFL Validation Experiment")
    print(f"Task: {task}, k: {k}, N federations: {n_federations}")
    print(f"Seeds per federation: {n_seeds}, FL rounds: {max_iterations}")
    print(f"PFL Weights: {PFL_WEIGHTS}")
    print(f"Output: {csv_path}")
    print("="*70)
    
    results = []
    
    for i in tqdm(range(n_federations), desc="Sampling federations"):
        # Sample random federation
        states = sample_random_federation(k)
        states_str = ','.join(sorted(states))
        
        print(f"\n[{i+1}/{n_federations}] Federation: {states}")
        
        # Compute PFL
        print("  Computing PFL...")
        pfl_score = compute_pfl_for_federation(states, task)
        print(f"  PFL = {pfl_score:.4f}")
        
        # Train FL model and get final metrics
        print("  Training FL model...")
        metrics = run_federation_experiment(
            states=states,
            task=task,
            n_seeds=n_seeds,
            max_iterations=max_iterations
        )
        
        # Store results with correct federation_id
        result = {
            'federation_id': start_id + i,
            'states': states_str,
            'k': k,
            'pfl': pfl_score,
            **metrics
        }
        results.append(result)
        
        # Save incrementally after each run (append mode)
        result_df = pd.DataFrame([result])
        if not file_exists and i == 0:
            # First run, new file - write with header
            result_df.to_csv(csv_path, index=False, mode='w')
            file_exists = True
        else:
            # Append without header
            result_df.to_csv(csv_path, index=False, mode='a', header=False)
        
        # Force flush to disk
        os.sync()
        
        print(f"  Final metrics: Loss={metrics['loss_mean']:.4f}±{metrics['loss_std']:.4f}, "
              f"Acc={metrics['accuracy_mean']:.4f}±{metrics['accuracy_std']:.4f}")
        print(f"  Fairness: EOD={metrics['eod_mean']:.4f}±{metrics['eod_std']:.4f}, "
              f"SPD={metrics['spd_mean']:.4f}±{metrics['spd_std']:.4f}")
        print(f"  Saved to {csv_path}")
    
    # Load ALL results (including previous runs) for plotting
    results_df = pd.read_csv(csv_path)
    
    print(f"\nTotal federations in {csv_path}: {len(results_df)}")
    
    # Create plots
    correlations = create_validation_plots(results_df, output_path, task, k)
    
    # Save summary
    summary = {
        'task': task,
        'k': k,
        'n_federations': n_federations,
        'n_seeds': n_seeds,
        'max_iterations': max_iterations,
        'pfl_weights': PFL_WEIGHTS,
        'correlations': correlations,
        'timestamp': timestamp
    }
    
    summary_path = output_path / f'summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("CORRELATION SUMMARY")
    print("="*70)
    print(f"{'Metric':<15} {'Pearson r':<12} {'p-value':<15} {'Significant?':<12}")
    print("-"*55)
    
    for metric in ['loss', 'accuracy', 'eod', 'spd']:
        r = correlations[f'{metric}_r']
        p = correlations[f'{metric}_p']
        sig = "Yes" if p < 0.05 else "No"
        print(f"{metric:<15} {r:<12.4f} {p:<15.2e} {sig:<12}")
    
    print("="*70)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if correlations['loss_r'] > 0 and correlations['loss_p'] < 0.05:
        print("✓ Higher PFL → Higher Loss (worse utility) - AS EXPECTED")
    else:
        print("✗ Loss correlation not as expected")
        
    if correlations['accuracy_r'] < 0 and correlations['accuracy_p'] < 0.05:
        print("✓ Higher PFL → Lower Accuracy (worse utility) - AS EXPECTED")
    else:
        print("✗ Accuracy correlation not as expected")
        
    if correlations['eod_r'] > 0 and correlations['eod_p'] < 0.05:
        print("✓ Higher PFL → Higher |EOD| (worse fairness) - AS EXPECTED")
    else:
        print("✗ EOD correlation not as expected")
        
    if correlations['spd_r'] > 0 and correlations['spd_p'] < 0.05:
        print("✓ Higher PFL → Higher |SPD| (worse fairness) - AS EXPECTED")
    else:
        print("✗ SPD correlation not as expected")
    
    return results_df, correlations


def main():
    parser = argparse.ArgumentParser(description='PFL Validation Experiment')
    parser.add_argument('--task', type=str, default='ACSIncome',
                        choices=['ACSIncome', 'ACSEmployment', 'ACSPublicCoverage', 
                                'ACSMobility', 'ACSTravelTime'],
                        help='FolkTables task')
    parser.add_argument('--k', type=int, default=5,
                        help='Federation size (number of states)')
    parser.add_argument('--n-federations', type=int, default=20,
                        help='Number of random federations to sample')
    parser.add_argument('--n-seeds', type=int, default=3,
                        help='Number of random seeds per federation')
    parser.add_argument('--max-iterations', type=int, default=50,
                        help='Number of FL rounds')
    parser.add_argument('--output-dir', type=str, default='PFL_validation',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    run_pfl_validation_experiment(
        task=args.task,
        k=args.k,
        n_federations=args.n_federations,
        n_seeds=args.n_seeds,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
