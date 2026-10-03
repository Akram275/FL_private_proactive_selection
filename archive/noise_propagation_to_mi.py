import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

# Set professional plotting style
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['legend.fontsize'] = 22
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['figure.dpi'] = 300

# Mutual information (natural log) from counts matrix
def mutual_information_from_counts(counts, N):
    counts = np.array(counts, dtype=float)
    #N = counts.sum()
    p = counts / N
    p_i = p.sum(axis=1, keepdims=True)
    p_j = p.sum(axis=0, keepdims=True)
    # avoid zeros for log by clipping
    eps = 1e-12
    p_safe = np.clip(p, eps, None)
    p_i_safe = np.clip(p_i, eps, None)
    p_j_safe = np.clip(p_j, eps, None)
    mi = np.sum(p_safe * np.log(p_safe / (p_i_safe * p_j_safe)))
    return mi

# Compute predicted variance from document's derivative formula evaluated at true counts
def predicted_variance_from_true_counts(true_counts, sigma):
    counts = np.array(true_counts, dtype=float)
    N = counts.sum()
    p = counts / N
    p_i = p.sum(axis=1)
    p_j = p.sum(axis=0)
    mi_true = mutual_information_from_counts(counts, true_counts.sum())
    # PMI_kl = ln(p_kl / (p_k. * p_.l))
    eps = 1e-12
    p_safe = np.clip(p, eps, None)
    p_i_safe = np.clip(p_i, eps, None)
    p_j_safe = np.clip(p_j, eps, None)
    PMI = np.log(p_safe / (p_i_safe[:,None] * p_j_safe[None,:]))
    # g_kl_true = 1/N * (-PMI_kl - MI_true)
    g = ( -PMI - mi_true ) / N
    var_pred = sigma**2 * np.sum(g**2)
    return mi_true, var_pred

# Monte Carlo simulation of MI distribution
def simulate_mi(true_counts, sigma, n_sim=5000, clip_min=1e-8):
    counts = np.array(true_counts, dtype=float)
    sims = []
    H, W = counts.shape
    for _ in range(n_sim):
        noise = np.random.normal(loc=0.0, scale=sigma, size=(H,W))
        noisy = counts + noise
        # clip to avoid negative counts
        #noisy = np.clip(noisy, clip_min, None)
        sims.append(mutual_information_from_counts(noisy, true_counts.sum()))
    return np.array(sims)

# Example contingency table
rng = np.random.default_rng(seed=42)  # for reproducibility
true_counts = rng.integers(low=100, high=5000, size=(10, 10)).astype(float)

# Two noise levels
sigma_small = 1.0
sigma_large = 60.0

# Predicted stats
mi_true, var_pred_small = predicted_variance_from_true_counts(true_counts, sigma_small)
_, var_pred_large = predicted_variance_from_true_counts(true_counts, sigma_large)

# Monte Carlo simulations
n_sim = 5000
sims_small = simulate_mi(true_counts, sigma_small, n_sim=n_sim)
sims_large = simulate_mi(true_counts, sigma_large, n_sim=n_sim)

# Empirical stats
print("True MI (nats):", mi_true)

print("\n--- Small noise (sigma = {}) ---".format(sigma_small))
print("Predicted Var:", var_pred_small)
print("Predicted Std:", math.sqrt(var_pred_small))
print("Empirical Mean:", sims_small.mean())
print("Empirical Var:", sims_small.var(ddof=1))
print("Empirical Std:", math.sqrt(sims_small.var(ddof=1)))
print("True Bias : ", mi_true - sims_small.mean())

print("\n--- Large noise (sigma = {}) ---".format(sigma_large))
print("Predicted Var:", var_pred_large)
print("Predicted Std:", math.sqrt(var_pred_large))
print("Empirical Mean:", sims_large.mean())
print("Empirical Var:", sims_large.var(ddof=1))
print("Empirical Std:", math.sqrt(sims_large.var(ddof=1)))
print("True Bias : ", mi_true - sims_large.mean())

# Multiple SNR scenarios for box plots
# Signal strength represented by mean count value
mean_signal = true_counts.mean()
print(f"\nMean signal strength: {mean_signal:.2f}")

# Define various noise levels for different SNR
noise_levels = np.array([100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5])
snr_values = []  # Signal-to-Noise Ratio
errors_data = []  # Bias for each SNR level
std_data = []    # Standard deviation for each SNR level

print("\n--- SNR Analysis ---")
for sigma in noise_levels:
    snr = mean_signal / sigma  # Simple SNR metric
    sims = simulate_mi(true_counts, sigma, n_sim=n_sim)
    bias = mi_true - sims  # Error/bias for each simulation
    
    snr_values.append(snr)
    errors_data.append(bias)
    std_data.append(np.std(bias))
    
    print(f"SNR = {snr:.2f} (σ={sigma:.1f}): Mean bias = {bias.mean():.4f}, Std of bias = {np.std(bias):.4f}")

# Box plot: Bias across different SNRs
# Short, wide aspect (was 12x10, nearly square) instead of tall/square: at a
# fixed print width this cuts the printed height roughly in half, with the
# same absolute font sizes reading comparatively larger. The old fontsize=35
# legend for a single entry also dwarfed the plot -- shrunk to something
# proportionate.
fig, ax = plt.subplots(figsize=(12, 5))
bp1 = ax.boxplot(errors_data, labels=[f"{snr:.0e}".replace('e+0', 'e+').replace('e-0', 'e-') for snr in snr_values], patch_artist=True)
for patch in bp1['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
for median in bp1['medians']:
    median.set_color('red')
    median.set_linewidth(2)

ax.axhline(y=0, color='green', linestyle='--', linewidth=2, label='No Bias')
ax.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=22, fontweight='bold')
ax.set_ylabel('MI Estimation Error', fontsize=22, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
ax.legend(fontsize=18, loc='upper right')

plt.subplots_adjust(left=0.1, right=0.97, top=0.95, bottom=0.2)
plt.savefig('mi_error_boxplot_snr.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# Standard deviation of error vs SNR (separate figure)
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(snr_values, std_data, 'o-', color='coral', linewidth=2.5, markersize=8, label='Empirical Std Dev')
ax.fill_between(snr_values, 0, std_data, alpha=0.3, color='coral')
ax.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=11, fontweight='bold')
ax.set_ylabel('Standard Deviation of Error (nats)', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(fontsize=35, loc='upper right')

plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.12)
plt.savefig('mi_error_variance_snr.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional plot: Box plot with log scale
fig, ax = plt.subplots(figsize=(14, 8))
bp = ax.boxplot(errors_data, labels=[f"{snr:.0e}".replace('e+0', 'e+').replace('e-0', 'e-') for snr in snr_values], patch_artist=True, widths=0.6)

for patch in bp['boxes']:
    patch.set_facecolor('steelblue')
    patch.set_alpha(0.7)
    patch.set_linewidth(1.5)

for whisker in bp['whiskers']:
    whisker.set_linewidth(1.5)
    whisker.set_color('gray')

for median in bp['medians']:
    median.set_color('red')
    median.set_linewidth(2.5)

for flier in bp['fliers']:
    flier.set_marker('o')
    flier.set_markerfacecolor('lightcoral')
    flier.set_markersize(5)
    flier.set_alpha(0.6)

ax.axhline(y=0, color='green', linestyle='--', linewidth=2.5, label='No Bias (True MI)')
ax.set_xlabel('Signal-to-Noise Ratio (SNR)', fontsize=12, fontweight='bold')
ax.set_ylabel('MI Estimation Error (nats)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y', linestyle='--')
ax.legend(fontsize=25, loc='upper right')

# Add text box with statistics
textstr = f'Total simulations per SNR: {n_sim}\nMean signal level: {mean_signal:.1f}\nTrue MI: {mi_true:.4f} nats'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, family='monospace')

plt.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.12)
plt.savefig('mi_error_boxplot_detailed.png', dpi=300, bbox_inches='tight')
plt.show()
