"""
Optimize PFL Weights for Best Correlation with Training Metrics

This script finds the optimal PFL weight configuration (α_ST, α_SN, β_NN, δ_NT)
that maximizes Spearman correlation between PFL and observed training metrics.

The optimization:
1. Loads existing results with federation states and metrics
2. Recomputes MI components for each federation
3. Searches for weights that maximize a combined correlation objective
4. Reports optimal weights and their correlations

Usage:
    python optimize_pfl_weights.py --csv PFL_validation/ACSIncome/k=5/results.csv
    python optimize_pfl_weights.py --csv PFL_validation/ACSIncome/k=5/results.csv --method differential_evolution
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_training"))

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from datetime import datetime
from scipy import stats, optimize
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from folktables import ACSDataSource
from FolkTables_FL import TASK_OBJECTS
from pfl_from_dataframe import calculate_mi, SENSITIVE_VARIABLE_NAME, NON_SENSITIVE_VARIABLE_NAMES
from itertools import combinations

# Professional plotting style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['figure.dpi'] = 150


def compute_mi_components(states: list, task: str) -> dict:
    """
    Compute individual MI components for a federation.
    
    Returns:
        dict with MI_ST, sum_MI_SN, sum_MI_NN, sum_MI_NT
    """
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    task_obj = TASK_OBJECTS[task]
    
    # Load and combine data
    all_features = []
    all_labels = []
    
    for state in states:
        acs_data = data_source.get_data(states=[state], download=True)
        features, labels, _ = task_obj.df_to_pandas(acs_data)
        all_features.append(features)
        all_labels.append(labels)
    
    df = pd.concat(all_features, ignore_index=True)
    df['label'] = pd.concat(all_labels, ignore_index=True).values
    
    sensitive_var = SENSITIVE_VARIABLE_NAME
    target_var = 'label'
    non_sensitive_vars = [v for v in NON_SENSITIVE_VARIABLE_NAMES if v in df.columns]
    
    # Compute MI components
    # 1. MI(S, T) - direct bias
    mi_st = calculate_mi(df[sensitive_var], df[target_var])
    if pd.isna(mi_st):
        mi_st = 0.0
    
    # 2. sum MI(S, N_k) - proxy bias
    sum_mi_sn = 0.0
    for ns_var in non_sensitive_vars:
        mi = calculate_mi(df[sensitive_var], df[ns_var])
        if not pd.isna(mi):
            sum_mi_sn += mi
    
    # 3. sum MI(N_i, N_j) - redundancy
    sum_mi_nn = 0.0
    for ns1, ns2 in combinations(non_sensitive_vars, 2):
        mi = calculate_mi(df[ns1], df[ns2])
        if not pd.isna(mi):
            sum_mi_nn += mi
    
    # 4. sum MI(N_k, T) - utility
    sum_mi_nt = 0.0
    for ns_var in non_sensitive_vars:
        mi = calculate_mi(df[ns_var], df[target_var])
        if not pd.isna(mi):
            sum_mi_nt += mi
    
    return {
        'mi_st': mi_st,
        'sum_mi_sn': sum_mi_sn,
        'sum_mi_nn': sum_mi_nn,
        'sum_mi_nt': sum_mi_nt
    }


def compute_pfl_from_components(mi_components: dict, weights: dict) -> float:
    """
    Compute PFL score from MI components and weights.
    
    PFL = α_ST * MI(S,T) + α_SN * Σ MI(S,N_k) + β_NN * Σ MI(N_i,N_j) - δ_NT * Σ MI(N_k,T)
    """
    return (
        weights['alpha_ST'] * mi_components['mi_st'] +
        weights['alpha_SN'] * mi_components['sum_mi_sn'] +
        weights['beta_NN'] * mi_components['sum_mi_nn'] -
        weights['delta_NT'] * mi_components['sum_mi_nt']
    )


def compute_correlations(pfl_values: np.ndarray, metrics_df: pd.DataFrame) -> dict:
    """
    Compute Spearman correlations between PFL and all metrics.
    """
    correlations = {}
    
    for metric in ['loss_mean', 'accuracy_mean', 'eod_mean', 'spd_mean']:
        if metric in metrics_df.columns:
            rho, p = stats.spearmanr(pfl_values, metrics_df[metric].values)
            correlations[metric] = {'rho': rho, 'p': p}
    
    return correlations


def correlation_objective(weights_array: np.ndarray, mi_data: list, 
                          metrics_df: pd.DataFrame, 
                          objective_type: str = 'combined') -> float:
    """
    Objective function for optimization.
    
    Args:
        weights_array: [alpha_ST, alpha_SN, beta_NN, delta_NT]
        mi_data: List of MI component dicts for each federation
        metrics_df: DataFrame with training metrics
        objective_type: 'combined', 'fairness', 'utility', or 'all'
    
    Returns:
        Negative score (for minimization)
    """
    weights = {
        'alpha_ST': weights_array[0],
        'alpha_SN': weights_array[1],
        'beta_NN': weights_array[2],
        'delta_NT': weights_array[3]
    }
    
    # Compute PFL for each federation
    pfl_values = np.array([
        compute_pfl_from_components(mi, weights) for mi in mi_data
    ])
    
    # Handle constant PFL (no variance)
    if np.std(pfl_values) < 1e-10:
        return 1000.0  # Penalty for degenerate solution
    
    correlations = compute_correlations(pfl_values, metrics_df)
    
    # Compute objective based on type
    if objective_type == 'combined':
        # Combined: maximize |rho| for all metrics with correct sign
        # Loss: want positive correlation (higher PFL = higher loss)
        # Accuracy: want negative correlation (higher PFL = lower accuracy)
        # EOD, SPD: want positive correlation (higher PFL = more bias)
        score = 0.0
        if 'loss_mean' in correlations:
            score += correlations['loss_mean']['rho']  # Want positive
        if 'accuracy_mean' in correlations:
            score -= correlations['accuracy_mean']['rho']  # Want negative (so subtract)
        if 'eod_mean' in correlations:
            score += correlations['eod_mean']['rho']  # Want positive
        if 'spd_mean' in correlations:
            score += correlations['spd_mean']['rho']  # Want positive
        
    elif objective_type == 'fairness':
        # Only fairness metrics
        score = 0.0
        if 'eod_mean' in correlations:
            score += correlations['eod_mean']['rho']
        if 'spd_mean' in correlations:
            score += correlations['spd_mean']['rho']
            
    elif objective_type == 'utility':
        # Only utility metrics
        score = 0.0
        if 'loss_mean' in correlations:
            score += correlations['loss_mean']['rho']
        if 'accuracy_mean' in correlations:
            score -= correlations['accuracy_mean']['rho']
            
    elif objective_type == 'all_abs':
        # Maximize sum of absolute correlations
        score = sum(abs(c['rho']) for c in correlations.values())
    
    else:  # 'all' - just sum
        score = sum(c['rho'] for c in correlations.values())
    
    return -score  # Negative because we minimize


def grid_search_weights(mi_data: list, metrics_df: pd.DataFrame,
                        objective_type: str = 'combined',
                        resolution: int = 10) -> tuple:
    """
    Grid search for optimal weights.
    """
    # Define search grid
    alpha_st_range = np.linspace(0, 2, resolution)
    alpha_sn_range = np.linspace(0, 2, resolution)
    beta_nn_range = np.linspace(0, 1, resolution)
    delta_nt_range = np.linspace(0, 2, resolution)
    
    best_score = float('inf')
    best_weights = None
    
    total = resolution ** 4
    
    print(f"Grid search with {total} combinations...")
    
    for alpha_st, alpha_sn, beta_nn, delta_nt in tqdm(
        product(alpha_st_range, alpha_sn_range, beta_nn_range, delta_nt_range),
        total=total, desc="Grid search"
    ):
        weights_array = np.array([alpha_st, alpha_sn, beta_nn, delta_nt])
        score = correlation_objective(weights_array, mi_data, metrics_df, objective_type)
        
        if score < best_score:
            best_score = score
            best_weights = weights_array
    
    return best_weights, -best_score


def optimize_weights_scipy(mi_data: list, metrics_df: pd.DataFrame,
                           objective_type: str = 'combined',
                           method: str = 'differential_evolution') -> tuple:
    """
    Use scipy optimization for optimal weights.
    """
    # Bounds for weights
    bounds = [(0, 3), (0, 3), (0, 2), (0, 3)]  # alpha_ST, alpha_SN, beta_NN, delta_NT
    
    print(f"Optimizing with {method}...")
    
    if method == 'differential_evolution':
        result = optimize.differential_evolution(
            correlation_objective,
            bounds=bounds,
            args=(mi_data, metrics_df, objective_type),
            maxiter=500,
            seed=42,
            disp=True,
            polish=True
        )
    elif method == 'dual_annealing':
        result = optimize.dual_annealing(
            correlation_objective,
            bounds=bounds,
            args=(mi_data, metrics_df, objective_type),
            maxiter=1000,
            seed=42
        )
    elif method == 'basinhopping':
        x0 = np.array([1.0, 1.0, 0.5, 1.0])  # Initial guess
        minimizer_kwargs = {
            'args': (mi_data, metrics_df, objective_type),
            'bounds': bounds
        }
        result = optimize.basinhopping(
            correlation_objective,
            x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=200,
            seed=42
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result.x, -result.fun


def evaluate_weights(weights_array: np.ndarray, mi_data: list, 
                     metrics_df: pd.DataFrame) -> dict:
    """
    Evaluate a weight configuration and return detailed results.
    """
    weights = {
        'alpha_ST': weights_array[0],
        'alpha_SN': weights_array[1],
        'beta_NN': weights_array[2],
        'delta_NT': weights_array[3]
    }
    
    # Compute PFL for each federation
    pfl_values = np.array([
        compute_pfl_from_components(mi, weights) for mi in mi_data
    ])
    
    correlations = compute_correlations(pfl_values, metrics_df)
    
    return {
        'weights': weights,
        'pfl_values': pfl_values,
        'correlations': correlations,
        'pfl_range': (pfl_values.min(), pfl_values.max()),
        'pfl_std': pfl_values.std()
    }


def plot_weight_comparison(results: dict, output_dir: Path):
    """
    Plot comparison of original vs optimized weights.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    original = results['original']
    optimized = results['optimized']
    metrics_df = results['metrics_df']
    
    metrics_info = [
        ('loss_mean', 'Loss', 'coral', 'positive'),
        ('accuracy_mean', 'Accuracy', 'seagreen', 'negative'),
        ('eod_mean', '|EOD|', 'steelblue', 'positive'),
        ('spd_mean', '|SPD|', 'purple', 'positive')
    ]
    
    for ax, (metric, label, color, expected) in zip(axes.flatten(), metrics_info):
        y = metrics_df[metric].values
        
        # Original PFL
        ax.scatter(original['pfl_values'], y, c=color, alpha=0.5, s=40, 
                   label='Original', marker='o')
        
        # Optimized PFL
        ax.scatter(optimized['pfl_values'], y, c='red', alpha=0.7, s=40,
                   label='Optimized', marker='x')
        
        # Correlation info
        orig_rho = original['correlations'][metric]['rho']
        opt_rho = optimized['correlations'][metric]['rho']
        
        textstr = f"Original ρ = {orig_rho:.3f}\nOptimized ρ = {opt_rho:.3f}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel('PFL Score')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs PFL (Expected: {expected} correlation)')
        ax.legend()
    
    plt.suptitle('Comparison: Original vs Optimized PFL Weights', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = output_dir / 'weight_optimization_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved comparison plot to {plot_path}")


def plot_weight_sensitivity(mi_data: list, metrics_df: pd.DataFrame, 
                            output_dir: Path, base_weights: np.ndarray):
    """
    Plot sensitivity of correlations to each weight.
    """
    weight_names = ['α_ST', 'α_SN', 'β_NN', 'δ_NT']
    weight_ranges = [
        np.linspace(0, 3, 30),
        np.linspace(0, 3, 30),
        np.linspace(0, 2, 30),
        np.linspace(0, 3, 30)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax, (name, w_range, idx) in zip(axes.flatten(), zip(weight_names, weight_ranges, range(4))):
        correlations_by_weight = {
            'loss': [], 'accuracy': [], 'eod': [], 'spd': []
        }
        
        for w_val in w_range:
            weights = base_weights.copy()
            weights[idx] = w_val
            
            pfl_values = np.array([
                compute_pfl_from_components(mi, {
                    'alpha_ST': weights[0], 'alpha_SN': weights[1],
                    'beta_NN': weights[2], 'delta_NT': weights[3]
                }) for mi in mi_data
            ])
            
            if np.std(pfl_values) > 1e-10:
                for metric, key in [('loss_mean', 'loss'), ('accuracy_mean', 'accuracy'),
                                   ('eod_mean', 'eod'), ('spd_mean', 'spd')]:
                    rho, _ = stats.spearmanr(pfl_values, metrics_df[metric].values)
                    correlations_by_weight[key].append(rho)
            else:
                for key in correlations_by_weight:
                    correlations_by_weight[key].append(np.nan)
        
        ax.plot(w_range, correlations_by_weight['loss'], label='Loss', color='coral')
        ax.plot(w_range, correlations_by_weight['accuracy'], label='Accuracy', color='seagreen')
        ax.plot(w_range, correlations_by_weight['eod'], label='EOD', color='steelblue')
        ax.plot(w_range, correlations_by_weight['spd'], label='SPD', color='purple')
        
        ax.axvline(x=base_weights[idx], color='red', linestyle='--', alpha=0.7, label='Optimal')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        ax.set_xlabel(f'{name} value')
        ax.set_ylabel('Spearman ρ')
        ax.set_title(f'Sensitivity to {name}')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Weight Sensitivity Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    plot_path = output_dir / 'weight_sensitivity.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Saved sensitivity plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Optimize PFL weights for best correlation')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to results CSV file')
    parser.add_argument('--task', type=str, default='ACSIncome',
                        choices=['ACSIncome', 'ACSEmployment', 'ACSPublicCoverage',
                                'ACSMobility', 'ACSTravelTime'],
                        help='FolkTables task')
    parser.add_argument('--method', type=str, default='differential_evolution',
                        choices=['grid_search', 'differential_evolution', 
                                'dual_annealing', 'basinhopping'],
                        help='Optimization method')
    parser.add_argument('--objective', type=str, default='combined',
                        choices=['combined', 'fairness', 'utility', 'all_abs'],
                        help='Objective type')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--grid-resolution', type=int, default=8,
                        help='Grid search resolution per dimension')
    
    args = parser.parse_args()
    
    # Load results
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV not found: {csv_path}")
        return
    
    results_df = pd.read_csv(csv_path)
    print(f"Loaded {len(results_df)} federations from {csv_path}")
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Infer task from path if possible
    task = args.task
    for part in csv_path.parts:
        if part.startswith('ACS'):
            task = part
            break
    
    print(f"Task: {task}")
    
    # Compute MI components for each federation
    print("\nComputing MI components for each federation...")
    mi_data = []
    
    for _, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Computing MI"):
        states = row['states'].split(',')
        mi_components = compute_mi_components(states, task)
        mi_data.append(mi_components)
    
    print(f"Computed MI components for {len(mi_data)} federations")
    
    # Original weights
    original_weights = np.array([1.0, 1.0, 0.5, 1.0])
    original_eval = evaluate_weights(original_weights, mi_data, results_df)
    
    print("\n" + "="*70)
    print("ORIGINAL WEIGHTS: α_ST=1.0, α_SN=1.0, β_NN=0.5, δ_NT=1.0")
    print("="*70)
    for metric, corr in original_eval['correlations'].items():
        print(f"  {metric}: ρ = {corr['rho']:.4f} (p = {corr['p']:.2e})")
    
    # Optimize
    print(f"\nOptimizing with objective={args.objective}, method={args.method}...")
    
    if args.method == 'grid_search':
        best_weights, best_score = grid_search_weights(
            mi_data, results_df, args.objective, args.grid_resolution
        )
    else:
        best_weights, best_score = optimize_weights_scipy(
            mi_data, results_df, args.objective, args.method
        )
    
    optimized_eval = evaluate_weights(best_weights, mi_data, results_df)
    
    print("\n" + "="*70)
    print(f"OPTIMIZED WEIGHTS (objective={args.objective})")
    print("="*70)
    print(f"  α_ST = {best_weights[0]:.4f}")
    print(f"  α_SN = {best_weights[1]:.4f}")
    print(f"  β_NN = {best_weights[2]:.4f}")
    print(f"  δ_NT = {best_weights[3]:.4f}")
    print(f"\nCorrelations:")
    for metric, corr in optimized_eval['correlations'].items():
        print(f"  {metric}: ρ = {corr['rho']:.4f} (p = {corr['p']:.2e})")
    
    # Improvement summary
    print("\n" + "="*70)
    print("IMPROVEMENT SUMMARY")
    print("="*70)
    print(f"{'Metric':<15} {'Original ρ':<12} {'Optimized ρ':<12} {'Change':<12}")
    print("-"*55)
    for metric in ['loss_mean', 'accuracy_mean', 'eod_mean', 'spd_mean']:
        orig_rho = original_eval['correlations'][metric]['rho']
        opt_rho = optimized_eval['correlations'][metric]['rho']
        change = opt_rho - orig_rho
        print(f"{metric:<15} {orig_rho:<12.4f} {opt_rho:<12.4f} {change:+.4f}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    results = {
        'original': original_eval,
        'optimized': optimized_eval,
        'metrics_df': results_df,
        'mi_data': mi_data
    }
    
    # Save optimized weights
    weight_result = {
        'original_weights': {
            'alpha_ST': 1.0, 'alpha_SN': 1.0, 'beta_NN': 0.5, 'delta_NT': 1.0
        },
        'optimized_weights': {
            'alpha_ST': float(best_weights[0]),
            'alpha_SN': float(best_weights[1]),
            'beta_NN': float(best_weights[2]),
            'delta_NT': float(best_weights[3])
        },
        'objective': args.objective,
        'method': args.method,
        'n_federations': len(results_df),
        'original_correlations': {k: {'rho': float(v['rho']), 'p': float(v['p'])} 
                                  for k, v in original_eval['correlations'].items()},
        'optimized_correlations': {k: {'rho': float(v['rho']), 'p': float(v['p'])} 
                                   for k, v in optimized_eval['correlations'].items()},
        'timestamp': timestamp
    }
    
    json_path = output_dir / f'optimized_weights_{timestamp}.json'
    with open(json_path, 'w') as f:
        json.dump(weight_result, f, indent=2)
    print(f"\nSaved optimized weights to {json_path}")
    
    # Create plots
    print("\nCreating comparison plots...")
    plot_weight_comparison(results, output_dir)
    
    print("\nCreating sensitivity plots...")
    plot_weight_sensitivity(mi_data, results_df, output_dir, best_weights)
    
    print("\n" + "="*70)
    print("Done!")
    print(f"Output saved to: {output_dir}")


if __name__ == '__main__':
    main()
