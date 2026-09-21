"""
Verification of MI Noise Propagation: Var(ΔMI) = O(σ²/N²)

This script verifies the analytical result that noise propagates from 
contingency counts to mutual information as a centered Gaussian with 
variance proportional to σ²/N², where:
- σ: standard deviation of noise added to counts
- N: total number of samples in the dataset (sum of contingency table)

The script:
1. Fixes σ and varies N across several orders of magnitude
2. Runs Monte Carlo simulations for each N
3. Verifies:
   - The MI error distribution is centered (mean ≈ 0)
   - Var(ΔMI) scales as 1/N² (slope of -2 on log-log plot)
   - Var(ΔMI) * N² / σ² ≈ constant
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, normaltest
from scipy.optimize import curve_fit
import seaborn as sns

# Set professional plotting style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.dpi'] = 150


def mutual_information_from_counts(counts, N):
    """Compute MI (in nats) from contingency counts."""
    counts = np.array(counts, dtype=float)
    p = counts / N
    p_i = p.sum(axis=1, keepdims=True)
    p_j = p.sum(axis=0, keepdims=True)
    # Avoid log(0) by clipping
    eps = 1e-12
    p_safe = np.clip(p, eps, None)
    p_i_safe = np.clip(p_i, eps, None)
    p_j_safe = np.clip(p_j, eps, None)
    mi = np.sum(p_safe * np.log(p_safe / (p_i_safe * p_j_safe)))
    return mi


def generate_contingency_table(N, shape=(5, 5), seed=None):
    """
    Generate a contingency table with total count N.
    Uses Dirichlet distribution for realistic probability allocation.
    """
    rng = np.random.default_rng(seed)
    n_cells = shape[0] * shape[1]
    # Dirichlet for random probability distribution
    probs = rng.dirichlet(np.ones(n_cells))
    counts = (probs * N).reshape(shape)
    # Ensure integer counts sum to N
    counts = np.round(counts).astype(float)
    # Adjust to match N exactly
    diff = N - counts.sum()
    counts[0, 0] += diff
    return counts


def simulate_mi_errors(true_counts, sigma, n_sim=5000):
    """
    Monte Carlo simulation: add Gaussian noise to counts, compute MI.
    Returns array of MI errors (noisy MI - true MI).
    """
    N = true_counts.sum()
    mi_true = mutual_information_from_counts(true_counts, N)
    
    errors = []
    H, W = true_counts.shape
    for _ in range(n_sim):
        noise = np.random.normal(loc=0.0, scale=sigma, size=(H, W))
        noisy_counts = true_counts + noise
        mi_noisy = mutual_information_from_counts(noisy_counts, N)
        errors.append(mi_noisy - mi_true)
    
    return np.array(errors), mi_true


def run_variance_scaling_experiment(sigma=10.0, N_values=None, n_sim=5000, 
                                     table_shape=(5, 5), seed=42):
    """
    Run experiment varying N to verify Var(ΔMI) ∝ σ²/N².
    """
    if N_values is None:
        N_values = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    
    results = {
        'N': [],
        'mean_error': [],
        'var_error': [],
        'std_error': [],
        'mi_true': [],
        'is_normal_shapiro': [],
        'is_normal_dagostino': [],
    }
    
    print(f"{'='*60}")
    print(f"Variance Scaling Experiment: σ = {sigma}")
    print(f"{'='*60}")
    print(f"{'N':>10} | {'Mean(ΔMI)':>12} | {'Var(ΔMI)':>12} | {'Var*N²/σ²':>12} | {'Normal?':>8}")
    print(f"{'-'*60}")
    
    for N in N_values:
        # Generate contingency table with this N
        true_counts = generate_contingency_table(N, shape=table_shape, seed=seed)
        
        # Run Monte Carlo
        errors, mi_true = simulate_mi_errors(true_counts, sigma, n_sim)
        
        # Statistics
        mean_err = errors.mean()
        var_err = errors.var(ddof=1)
        std_err = errors.std(ddof=1)
        
        # Normality tests
        _, p_shapiro = shapiro(errors[:500])  # Shapiro limited to 5000 samples
        _, p_dagostino = normaltest(errors)
        is_normal_s = p_shapiro > 0.05
        is_normal_d = p_dagostino > 0.05
        
        # Normalized variance: should be ~constant if Var ∝ σ²/N²
        normalized_var = var_err * (N**2) / (sigma**2)
        
        results['N'].append(N)
        results['mean_error'].append(mean_err)
        results['var_error'].append(var_err)
        results['std_error'].append(std_err)
        results['mi_true'].append(mi_true)
        results['is_normal_shapiro'].append(is_normal_s)
        results['is_normal_dagostino'].append(is_normal_d)
        
        normal_str = "Yes" if (is_normal_s or is_normal_d) else "No"
        print(f"{N:>10} | {mean_err:>12.2e} | {var_err:>12.2e} | {normalized_var:>12.4f} | {normal_str:>8}")
    
    return results


def plot_variance_scaling(results, sigma):
    """
    Create plots to verify the O(σ²/N²) scaling.
    """
    N = np.array(results['N'])
    var_err = np.array(results['var_error'])
    mean_err = np.array(results['mean_error'])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ============================================================
    # Plot 1: Log-log plot of Variance vs N (should have slope -2)
    # ============================================================
    ax1 = axes[0, 0]
    ax1.loglog(N, var_err, 'o-', color='steelblue', markersize=8, label='Empirical Var(ΔMI)')
    
    # Fit power law: Var = C * N^α
    log_N = np.log(N)
    log_var = np.log(var_err)
    slope, intercept = np.polyfit(log_N, log_var, 1)
    fitted_var = np.exp(intercept) * N**slope
    ax1.loglog(N, fitted_var, '--', color='red', linewidth=2, 
               label=f'Fit: slope = {slope:.2f}')
    
    # Theoretical line with slope -2
    C_theory = var_err[0] * N[0]**2  # Calibrate constant from first point
    theoretical_var = C_theory / N**2
    ax1.loglog(N, theoretical_var, ':', color='green', linewidth=2,
               label='Theoretical (slope = -2)')
    
    ax1.set_xlabel('N (dataset size)')
    ax1.set_ylabel('Var(ΔMI)')
    ax1.set_title(f'Variance Scaling (σ = {sigma})\nExpected slope: -2, Fitted slope: {slope:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 2: Normalized variance (Var * N² / σ²) should be constant
    # ============================================================
    ax2 = axes[0, 1]
    normalized_var = var_err * N**2 / sigma**2
    ax2.semilogx(N, normalized_var, 'o-', color='coral', markersize=8)
    ax2.axhline(y=normalized_var.mean(), color='green', linestyle='--', 
                linewidth=2, label=f'Mean = {normalized_var.mean():.4f}')
    ax2.fill_between(N, normalized_var.mean() - normalized_var.std(),
                     normalized_var.mean() + normalized_var.std(),
                     alpha=0.3, color='green')
    ax2.set_xlabel('N (dataset size)')
    ax2.set_ylabel('Var(ΔMI) × N² / σ²')
    ax2.set_title('Normalized Variance (should be constant)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 3: Mean error (should be ~0, centered distribution)
    # ============================================================
    ax3 = axes[1, 0]
    ax3.semilogx(N, mean_err, 'o-', color='purple', markersize=8)
    ax3.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Zero (centered)')
    ax3.fill_between(N, -np.array(results['std_error']), np.array(results['std_error']),
                     alpha=0.2, color='purple', label='±1 Std Dev')
    ax3.set_xlabel('N (dataset size)')
    ax3.set_ylabel('Mean(ΔMI)')
    ax3.set_title('Mean Error (should be ≈ 0)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ============================================================
    # Plot 4: Std Dev vs 1/N (should be linear)
    # ============================================================
    ax4 = axes[1, 1]
    std_err = np.array(results['std_error'])
    inv_N = 1 / N
    ax4.plot(inv_N, std_err, 'o', color='teal', markersize=8, label='Empirical Std(ΔMI)')
    
    # Linear fit: Std = C / N => Std = C * (1/N)
    slope_std, intercept_std = np.polyfit(inv_N, std_err, 1)
    fitted_std = slope_std * inv_N + intercept_std
    ax4.plot(inv_N, fitted_std, '--', color='red', linewidth=2,
             label=f'Linear fit: slope = {slope_std:.2f}')
    
    ax4.set_xlabel('1/N')
    ax4.set_ylabel('Std(ΔMI)')
    ax4.set_title('Std Dev vs 1/N (should be linear through origin)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mi_variance_scaling_verification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return slope


def plot_error_distributions(sigma=10.0, N_values=None, n_sim=5000, seed=42):
    """
    Plot error distributions for different N values to show Gaussianity.
    """
    if N_values is None:
        N_values = [1000, 5000, 20000, 100000]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, N in enumerate(N_values):
        ax = axes[idx]
        
        true_counts = generate_contingency_table(N, shape=(5, 5), seed=seed)
        errors, mi_true = simulate_mi_errors(true_counts, sigma, n_sim)
        
        # Histogram
        ax.hist(errors, bins=50, density=True, alpha=0.7, color='steelblue',
                edgecolor='white', label='Empirical')
        
        # Fitted Gaussian
        mu, std = errors.mean(), errors.std()
        x = np.linspace(errors.min(), errors.max(), 100)
        ax.plot(x, norm.pdf(x, mu, std), 'r-', linewidth=2,
                label=f'Gaussian(μ={mu:.2e}, σ={std:.2e})')
        
        ax.set_xlabel('ΔMI (error)')
        ax.set_ylabel('Density')
        ax.set_title(f'N = {N:,}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'MI Error Distributions (σ = {sigma})\nVerifying Gaussianity', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('mi_error_distributions_by_N.png', dpi=300, bbox_inches='tight')
    plt.show()


def verify_sigma_scaling(N=10000, sigma_values=None, n_sim=5000, seed=42):
    """
    Verify that Var(ΔMI) ∝ σ² by fixing N and varying σ.
    """
    if sigma_values is None:
        sigma_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    
    true_counts = generate_contingency_table(N, shape=(5, 5), seed=seed)
    
    variances = []
    for sigma in sigma_values:
        errors, _ = simulate_mi_errors(true_counts, sigma, n_sim)
        variances.append(errors.var(ddof=1))
    
    variances = np.array(variances)
    sigma_values = np.array(sigma_values)
    
    # Plot Var vs σ²
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Log-log plot
    ax1 = axes[0]
    ax1.loglog(sigma_values**2, variances, 'o-', color='steelblue', markersize=8)
    
    # Fit
    log_sigma2 = np.log(sigma_values**2)
    log_var = np.log(variances)
    slope, intercept = np.polyfit(log_sigma2, log_var, 1)
    fitted = np.exp(intercept) * (sigma_values**2)**slope
    ax1.loglog(sigma_values**2, fitted, '--', color='red', 
               label=f'Fit: slope = {slope:.2f}')
    
    ax1.set_xlabel('σ²')
    ax1.set_ylabel('Var(ΔMI)')
    ax1.set_title(f'Variance vs σ² (N = {N})\nExpected slope: 1, Fitted: {slope:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Normalized: Var / σ² should be constant
    ax2 = axes[1]
    normalized = variances / sigma_values**2
    ax2.plot(sigma_values, normalized, 'o-', color='coral', markersize=8)
    ax2.axhline(y=normalized.mean(), color='green', linestyle='--',
                label=f'Mean = {normalized.mean():.2e}')
    ax2.set_xlabel('σ')
    ax2.set_ylabel('Var(ΔMI) / σ²')
    ax2.set_title('Normalized by σ² (should be constant)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mi_variance_sigma_scaling.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return slope


if __name__ == '__main__':
    print("\n" + "="*70)
    print("VERIFICATION: Var(ΔMI) = O(σ²/N²)")
    print("="*70)
    
    # Parameters
    sigma = 10.0
    N_values = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    n_sim = 5000
    
    # Run main experiment: varying N
    print("\n[1] Varying N (fixed σ = {})".format(sigma))
    results = run_variance_scaling_experiment(
        sigma=sigma, N_values=N_values, n_sim=n_sim, seed=42
    )
    
    # Plot results
    slope = plot_variance_scaling(results, sigma)
    print(f"\nFitted slope on log-log plot: {slope:.3f} (expected: -2.0)")
    
    # Plot error distributions
    print("\n[2] Plotting error distributions...")
    plot_error_distributions(sigma=sigma, N_values=[1000, 5000, 20000, 100000], 
                            n_sim=n_sim, seed=42)
    
    # Verify σ scaling
    print("\n[3] Verifying σ² scaling (fixed N = 10000)...")
    sigma_slope = verify_sigma_scaling(N=10000, n_sim=n_sim, seed=42)
    print(f"Fitted slope for σ² scaling: {sigma_slope:.3f} (expected: 1.0)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"• Var(ΔMI) vs N: slope = {slope:.3f} (expected -2.0)")
    print(f"• Var(ΔMI) vs σ²: slope = {sigma_slope:.3f} (expected 1.0)")
    print(f"• Conclusion: Var(ΔMI) = O(σ²/N²) is {'VERIFIED' if abs(slope + 2) < 0.2 and abs(sigma_slope - 1) < 0.2 else 'NOT VERIFIED'}")
    print("="*70)
