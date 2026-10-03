"""
Plot and Analyze PFL Validation Results

This script reads existing PFL validation results from CSV and creates:
1. Scatter plots of metrics vs PFL
2. Correlation analysis (Pearson, Spearman, Kendall)
3. Summary statistics and tables

IMPORTANT: PFL values are recomputed using the optimal weights, not read from CSV.

Usage:
    python plot_pfl_validation.py --csv PFL_validation/ACSIncome/k=5/results.csv
    python plot_pfl_validation.py --csv PFL_validation/ACSIncome/k=5/results.csv --output-dir plots/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FL_training"))

import numpy as np
import pandas as pd
import argparse
import json
from pathlib import Path
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from tqdm import tqdm

from folktables import ACSDataSource
from FolkTables_FL import TASK_OBJECTS
from pfl_from_dataframe import compute_PFL_of_dataframe

# Optimized PFL Weights (keys match pfl_from_dataframe.py)
PFL_WEIGHTS = {
    'alpha_ST': 2.0,      # Direct bias MI(S, T)
    'alpha_SN': 0.8889,   # Proxy bias MI(S, N_k)
    'beta_NN': 0.1111,    # Feature redundancy MI(N_i, N_j)
    'delta_NT': 1.3333    # Utility reward MI(N_k, T)
}

# Professional plotting style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.dpi'] = 150


def recompute_pfl_for_federation(states: list, task: str) -> float:
    """
    Recompute PFL score for a federation using the optimal weights.
    """
    data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    task_obj = TASK_OBJECTS[task]
    
    all_features = []
    all_labels = []
    
    for state in states:
        acs_data = data_source.get_data(states=[state], download=True)
        features, labels, _ = task_obj.df_to_pandas(acs_data)
        all_features.append(features)
        all_labels.append(labels)
    
    combined_features = pd.concat(all_features, ignore_index=True)
    combined_labels = pd.concat(all_labels, ignore_index=True)
    
    df = combined_features.copy()
    df['label'] = combined_labels.values
    
    return compute_PFL_of_dataframe(df, weights=PFL_WEIGHTS)


def recompute_all_pfls(results_df: pd.DataFrame, task: str) -> np.ndarray:
    """
    Recompute PFL for all federations in the results DataFrame.
    """
    print(f"\nRecomputing PFL with optimal weights: {PFL_WEIGHTS}")
    
    pfl_values = []
    for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Recomputing PFL"):
        states = row['states'].split(',')
        pfl = recompute_pfl_for_federation(states, task)
        pfl_values.append(pfl)
    
    return np.array(pfl_values)


def compute_all_correlations(x, y):
    """
    Compute multiple correlation measures between x and y.
    
    Returns:
        dict with Pearson, Spearman, and Kendall correlations + p-values
    """
    # Pearson (linear)
    pearson_r, pearson_p = stats.pearsonr(x, y)
    
    # Spearman (rank-based, monotonic)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    # Kendall (rank-based, robust with small samples)
    kendall_tau, kendall_p = stats.kendalltau(x, y)
    
    # R² (coefficient of determination)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = r_value ** 2
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_rho': spearman_r,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
        'r_squared': r_squared,
        'slope': slope,
        'intercept': intercept
    }


def plot_scatter_with_correlation(ax, x, y, xerr=None, yerr=None,
                                   xlabel='PFL', ylabel='Metric',
                                   title='', color='steelblue',
                                   show_regression=True,
                                   label_fontsize=22, tick_fontsize=22,
                                   corr_fontsize=22, marker_size=40, n_ticks=4):
    """
    Create scatter plot with correlation line and statistics.
    """
    # Scatter points with error bars
    if yerr is not None:
        ax.errorbar(x, y, yerr=yerr, fmt='o', color=color,
                    alpha=0.7, markersize=6, capsize=2, elinewidth=1)
    else:
        ax.scatter(x, y, c=color, alpha=0.7, s=marker_size, edgecolors='white', linewidth=0.5)

    # Compute correlations
    corr = compute_all_correlations(x, y)

    if show_regression:
        # Linear regression line
        x_line = np.linspace(min(x), max(x), 100)
        y_line = corr['slope'] * x_line + corr['intercept']
        ax.plot(x_line, y_line, '--', color='red', linewidth=1.5)

    ax.set_xlabel(xlabel, fontsize=label_fontsize)
    ax.set_ylabel(ylabel, fontsize=label_fontsize)
    ax.set_title(title)

    # Make plot borders (spines) more bold
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Make tick marks bolder
    ax.tick_params(axis='both', width=1.5, length=5, labelsize=tick_fontsize)

    # Fewer ticks so larger tick labels don't collide with each other
    ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks))

    # Add text box with statistics (Spearman only). Anchor it in whichever
    # top corner is emptiest given the trend direction, so the larger label
    # doesn't sit on top of the point cluster (e.g. negative correlations
    # cluster points near the top-left).
    textstr = f"ρ = {corr['spearman_rho']:.3f}"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    if corr['spearman_rho'] < 0:
        text_x, ha = 0.95, 'right'
    else:
        text_x, ha = 0.05, 'left'
    ax.text(text_x, 0.95, textstr, transform=ax.transAxes, fontsize=corr_fontsize,
            verticalalignment='top', horizontalalignment=ha, bbox=props)

    return corr


def create_validation_plots(results_df: pd.DataFrame, output_dir: Path, 
                            task: str = None, k: int = None):
    """
    Create 2x2 scatter plots of metrics vs PFL.
    """
    # Single wide row instead of a near-square 2x2 grid: same content, same
    # per-panel readability, but ~3x less vertical space at a given print
    # width (a 2x2 at 14x12in is nearly square; a 1x4 row is a short, wide
    # strip -- most figure placements in the paper scale to a fixed column
    # width, so height is what actually shrinks).
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))

    pfl = results_df['pfl'].values

    # Infer task and k from data if not provided
    if task is None:
        task = "Unknown"
    if k is None and 'k' in results_df.columns:
        k = results_df['k'].iloc[0]

    correlations = {}

    # Plot 1: F1 vs PFL (expect negative correlation - higher PFL = worse performance)
    ax1 = axes[0]
    f1 = results_df['f1_mean'].values
    f1_err = results_df['f1_std'].values if 'f1_std' in results_df.columns else None
    corr_f1 = plot_scatter_with_correlation(
        ax1, pfl, f1, yerr=f1_err,
        xlabel='PFL Score', ylabel='F1 Score',
        title='',
        color='black'
    )
    correlations['f1'] = corr_f1

    # Plot 2: Accuracy vs PFL (expect negative correlation)
    ax2 = axes[1]
    acc = results_df['accuracy_mean'].values
    acc_err = results_df['accuracy_std'].values if 'accuracy_std' in results_df.columns else None
    corr_acc = plot_scatter_with_correlation(
        ax2, pfl, acc, yerr=acc_err,
        xlabel='PFL Score', ylabel='Accuracy',
        title='',
        color='black'
    )
    correlations['accuracy'] = corr_acc

    # Plot 3: EOD vs PFL (expect positive correlation)
    ax3 = axes[2]
    eod = results_df['eod_mean'].values
    eod_err = results_df['eod_std'].values if 'eod_std' in results_df.columns else None
    corr_eod = plot_scatter_with_correlation(
        ax3, pfl, eod, yerr=eod_err,
        xlabel='PFL Score', ylabel='|EOD|',
        title='',
        color='black'
    )
    correlations['eod'] = corr_eod

    # Plot 4: MAD vs PFL (expect positive correlation)
    ax4 = axes[3]
    mad = results_df['mad_mean'].values
    mad_err = results_df['mad_std'].values if 'mad_std' in results_df.columns else None
    corr_mad = plot_scatter_with_correlation(
        ax4, pfl, mad, yerr=mad_err,
        xlabel='PFL Score', ylabel='|MAD|',
        title='',
        color='black'
    )
    correlations['mad'] = corr_mad

    plt.tight_layout()

    plot_path = output_dir / 'scatter_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plots to {plot_path}")

    plt.close(fig)

    return correlations


def create_correlation_heatmap(results_df: pd.DataFrame, output_dir: Path):
    """
    Create a heatmap showing correlations between PFL and all metrics.
    """
    # Select numeric columns
    metrics = ['pfl', 'f1_mean', 'accuracy_mean', 'eod_mean', 'mad_mean']
    if 'mad_mean' in results_df.columns:
        metrics.append('mad_mean')
    
    # Filter to existing columns
    metrics = [m for m in metrics if m in results_df.columns]
    
    # Compute correlation matrix (Spearman)
    corr_matrix = results_df[metrics].corr(method='spearman')
    
    # Rename for display
    rename_map = {
        'pfl': 'PFL',
        'f1_mean': 'F1',
        'accuracy_mean': 'Accuracy',
        'eod_mean': '|EOD|',
        'mad_mean': '|MAD|',
        'mad_mean': '|MAD|'
    }
    corr_matrix = corr_matrix.rename(index=rename_map, columns=rename_map)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
                center=0, vmin=-1, vmax=1, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_dir / 'correlation_heatmap.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved correlation heatmap to {plot_path}")
    
    plt.show()
    
    return corr_matrix


def print_correlation_summary(correlations: dict):
    """
    Print formatted correlation summary table.
    """
    print("\n" + "="*80)
    print("CORRELATION SUMMARY (Spearman ρ is primary metric)")
    print("="*80)
    
    header = f"{'Metric':<12} {'Spearman ρ':<12} {'p-value':<12} {'Pearson r':<12} {'Kendall τ':<12} {'Sig?':<8}"
    print(header)
    print("-"*80)
    
    for metric, corr in correlations.items():
        sig = "Yes" if corr['spearman_p'] < 0.05 else "No"
        row = f"{metric:<12} {corr['spearman_rho']:<12.4f} {corr['spearman_p']:<12.2e} {corr['pearson_r']:<12.4f} {corr['kendall_tau']:<12.4f} {sig:<8}"
        print(row)
    
    print("="*80)
    
    # Interpretation
    print("\nINTERPRETATION:")
    
    expectations = {
        'f1': ('negative', lambda r: r < 0),
        'accuracy': ('negative', lambda r: r < 0),
        'eod': ('positive', lambda r: r > 0),
        'mad': ('positive', lambda r: r > 0)
    }
    
    for metric, (expected_dir, check_fn) in expectations.items():
        if metric in correlations:
            corr = correlations[metric]
            rho = corr['spearman_rho']
            p = corr['spearman_p']
            
            if check_fn(rho) and p < 0.05:
                print(f"✓ {metric.upper()}: ρ={rho:.3f}, p={p:.2e} → {expected_dir} correlation AS EXPECTED")
            elif check_fn(rho) and p >= 0.05:
                print(f"~ {metric.upper()}: ρ={rho:.3f}, p={p:.2e} → {expected_dir} but NOT SIGNIFICANT")
            else:
                print(f"✗ {metric.upper()}: ρ={rho:.3f}, p={p:.2e} → NOT as expected (wanted {expected_dir})")


def save_correlation_results(correlations: dict, output_dir: Path, 
                              results_df: pd.DataFrame):
    """
    Save correlation results to JSON file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Flatten correlations for JSON
    flat_corr = {}
    for metric, corr in correlations.items():
        for key, value in corr.items():
            flat_corr[f'{metric}_{key}'] = float(value)
    
    summary = {
        'n_federations': len(results_df),
        'pfl_range': [float(results_df['pfl'].min()), float(results_df['pfl'].max())],
        'pfl_weights': PFL_WEIGHTS,  # Record which weights were used
        'correlations': flat_corr,
        'timestamp': timestamp
    }
    
    if 'k' in results_df.columns:
        summary['k'] = int(results_df['k'].iloc[0])
    
    summary_path = output_dir / f'correlation_summary_{timestamp}.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved correlation summary to {summary_path}")
    
    return summary


def create_latex_table(correlations: dict, output_dir: Path):
    """
    Generate LaTeX table of correlation results.
    """
    latex = r"""\begin{table}[htbp]
\centering
\caption{Correlation between PFL and FL Training Metrics}
\label{tab:pfl_correlation}
\begin{tabular}{lcccc}
\toprule
\textbf{Metric} & \textbf{Spearman $\rho$} & \textbf{p-value} & \textbf{Pearson $r$} & \textbf{Expected} \\
\midrule
"""
    
    expected_map = {
        'f1': 'Negative',
        'accuracy': 'Negative',
        'eod': 'Positive',
        'mad': 'Positive'
    }
    
    for metric, corr in correlations.items():
        rho = corr['spearman_rho']
        p = corr['spearman_p']
        r = corr['pearson_r']
        expected = expected_map.get(metric, '---')
        
        # Format p-value
        if p < 0.001:
            p_str = f"$<$0.001"
        else:
            p_str = f"{p:.3f}"
        
        # Add significance marker
        sig_marker = "*" if p < 0.05 else ""
        if p < 0.01:
            sig_marker = "**"
        if p < 0.001:
            sig_marker = "***"
        
        latex += f"{metric.upper()} & {rho:.3f}{sig_marker} & {p_str} & {r:.3f} & {expected} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\vspace{2mm}
\footnotesize{$^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$}
\end{table}
"""
    
    latex_path = output_dir / 'correlation_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)
    
    print(f"Saved LaTeX table to {latex_path}")
    
    return latex


def main():
    parser = argparse.ArgumentParser(description='Plot and analyze PFL validation results')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to results CSV file')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: same as CSV)')
    parser.add_argument('--task', type=str, default=None,
                        help='Task name for plot title')
    parser.add_argument('--k', type=int, default=None,
                        help='Federation size for plot title')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display plots (just save)')
    parser.add_argument('--skip-recompute', action='store_true',
                        help='Skip PFL recomputation (use values from CSV)')
    
    args = parser.parse_args()
    
    # Load data
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    results_df = pd.read_csv(csv_path)
    print(f"Loaded {len(results_df)} federations from {csv_path}")
    
    # Infer task early (needed for recomputation)
    task = args.task
    if task is None:
        # Try to infer from path like PFL_validation/ACSIncome/k=5/results.csv
        parts = csv_path.parts
        for part in parts:
            if part.startswith('ACS'):
                task = part
                break
    
    # Recompute PFL values with optimal weights (unless skipped)
    if not args.skip_recompute:
        if task is None:
            print("Warning: Task not specified, cannot recompute PFL. Using CSV values.")
        else:
            print("\n" + "="*80)
            print(f"Recomputing PFL values with optimal weights:")
            print(f"  α_ST={PFL_WEIGHTS['alpha_ST']:.4f}, α_SN={PFL_WEIGHTS['alpha_SN']:.4f}")
            print(f"  β_NN={PFL_WEIGHTS['beta_NN']:.4f}, δ_NT={PFL_WEIGHTS['delta_NT']:.4f}")
            
            old_pfl = results_df['pfl'].copy()
            new_pfl = recompute_all_pfls(results_df, task)
            results_df['pfl'] = new_pfl
            
            # Report change
            pfl_diff = (results_df['pfl'] - old_pfl).abs()
            print(f"  PFL value changes: mean={pfl_diff.mean():.4f}, max={pfl_diff.max():.4f}")
    else:
        print("Skipping PFL recomputation (using values from CSV)")
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = csv_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create plots
    if args.no_show:
        plt.switch_backend('Agg')
    
    print("\n" + "="*80)
    print("Creating scatter plots...")
    correlations = create_validation_plots(results_df, output_dir, task=task, k=args.k)
    
    print("\nCreating correlation heatmap...")
    create_correlation_heatmap(results_df, output_dir)
    
    # Print and save results
    print_correlation_summary(correlations)
    save_correlation_results(correlations, output_dir, results_df)
    create_latex_table(correlations, output_dir)
    
    print("\n" + "="*80)
    print("Done!")
    print(f"Output files saved to: {output_dir}")


if __name__ == '__main__':
    main()
