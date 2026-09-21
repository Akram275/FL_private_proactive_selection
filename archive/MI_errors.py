import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, norm
from sklearn.metrics import mutual_info_score # Using sklearn for a direct MI calculation

# --- Configuration Parameters ---
NUM_ROWS = 8  # Number of categories for variable X
NUM_COLS = 5  # Number of categories for variable Y
TOTAL_TRUE_COUNT = 50000  # Total number of observations in the true table
NOISE_SIGMA = 200.0  # Standard deviation of the Gaussian noise added to each cell count
NUM_SAMPLES = 5000 # Number of Monte Carlo simulations
LOG_BASE = 10 # Use log base 2 for MI in bits (np.e for nats)
EPSILON = 1e-12 # Small constant to avoid log(0)

# --- Helper Functions ---

def generate_true_contingency_table(rows, cols, total_n):
    """Generates a random contingency table with integer counts."""
    num_cells = rows * cols
    print(f"Generating true table with dimensions: {rows}x{cols}, total count: {total_n}")

    if num_cells == 0:
        print("Error: Number of cells is zero (rows or cols is 0). Returning empty table.")
        return np.array([]).reshape((rows,cols))
    if total_n == 0:
        print("Warning: total_n is 0. Generating a table of all zeros.")
        return np.zeros((rows, cols), dtype=int)

    # Use Dirichlet distribution to generate probabilities
    # Alpha = 1.0 for each cell gives a distribution that's uniform on average over the simplex.
    # Smaller alpha values (e.g., 0.1) would lead to sparser probabilities (more zeros in table).
    # Larger alpha values (e.g., 10) would lead to more uniform probabilities.
    dirichlet_alpha = np.ones(num_cells) * 1.0

    flat_probs = np.random.dirichlet(dirichlet_alpha)

    # Generate counts using multinomial distribution
    # This ensures counts are integers and sum to total_n
    true_counts_flat = np.random.multinomial(total_n, flat_probs)
    true_table = true_counts_flat.reshape((rows, cols))

    # No complex zero fixup needed if using sklearn.metrics.mutual_info_score
    # as it handles cells with 0 counts correctly (p_ij = 0 for MI calculation).
    # If the user wants to avoid zeros for other reasons (e.g. manual PMI calc with logs),
    # they might need to ensure total_n is very large or add a small pseudocount later.
    # For this script's purpose, this generated table is fine. The nature of the
    # table (sparse or dense) will depend on rows, cols, and total_n.

    print("Generated True Contingency Table:\n", true_table)
    print("Sum of true table:", np.sum(true_table)) # Should always be total_n
    if np.any(true_table == 0):
        print("Note: The generated true table contains zero counts in some cells.")
        num_zero_cells = np.sum(true_table == 0)
        print(f"      Number of zero cells: {num_zero_cells}/{num_cells} ({num_zero_cells*100/num_cells:.1f}%)")
    return true_table

def calculate_mi_from_table(contingency_table, base=2):
    """
    Calculates Mutual Information from a contingency table.
    Using sklearn's mutual_info_score for robustness.
    """
    # Ensure the table is non-negative, as mutual_info_score expects counts.
    # The add_gaussian_noise function should already handle this by rounding and clipping.
    table_for_mi = np.maximum(contingency_table, 0).astype(int)

    if np.sum(table_for_mi) == 0:
        # If all counts are zero (e.g. after noise and clipping), MI is typically 0.
        return 0.0

    # sklearn.metrics.mutual_info_score calculates MI in nats (natural log)
    try:
        mi_nats = mutual_info_score(None, None, contingency=table_for_mi)
    except ValueError as e:
        # This can happen if sum of contingency table is 0, though we try to catch it.
        # Or if a row/column sum is zero, which sklearn might handle by returning 0 MI.
        print(f"Warning: mutual_info_score raised ValueError: {e}. Table:\n{table_for_mi}\nSum: {np.sum(table_for_mi)}. Returning 0.0 for MI.")
        return 0.0

    if base == 2:
        return mi_nats / np.log(2)
    elif base == np.e:
        return mi_nats
    else: # For other bases
        return mi_nats / np.log(base)

def add_gaussian_noise(true_table, sigma):
    """Adds Gaussian noise to each cell, rounds, and ensures counts are non-negative."""
    noise = np.random.normal(0, sigma, true_table.shape)
    noisy_table_float = true_table + noise

    # Round to nearest integer
    noisy_table_rounded = np.round(noisy_table_float)

    # Clip at zero to ensure non-negative counts

    noisy_table_clipped_int = np.maximum(noisy_table_rounded, 0).astype(int)
    #noisy_table_clipped_int = noisy_table_rounded.astype(int)
    return noisy_table_clipped_int


# --- Main Simulation ---
if __name__ == "__main__":
    print(f"Starting simulation with {NUM_SAMPLES} samples.")
    print(f"Table dimensions: {NUM_ROWS}x{NUM_COLS}")
    print(f"Total true count: {TOTAL_TRUE_COUNT}")
    print(f"Noise sigma: {NOISE_SIGMA}, Log base for MI: {'e' if LOG_BASE == np.e else LOG_BASE}\n")

    # 1. Generate true contingency table
    true_table = generate_true_contingency_table(NUM_ROWS, NUM_COLS, TOTAL_TRUE_COUNT)

    # 2. Calculate true MI
    mi_true = calculate_mi_from_table(true_table, base=LOG_BASE)
    print(f"True MI (MI_true): {mi_true:.4f} {'bits' if LOG_BASE==2 else 'nats'}\n")

    # 3. Simulate noisy tables and collect errors
    errors_mi = []
    noisy_mi_values = []

    for i in range(NUM_SAMPLES):
        if (i + 1) % (NUM_SAMPLES // 10) == 0:
            print(f"Processing sample {i+1}/{NUM_SAMPLES}...")

        # Add noise
        noisy_table = add_gaussian_noise(true_table, NOISE_SIGMA)

        # Calculate MI from noisy table
        mi_noisy = calculate_mi_from_table(noisy_table, base=LOG_BASE)
        noisy_mi_values.append(mi_noisy)

        # Calculate error
        error = mi_true - mi_noisy
        errors_mi.append(error)

    errors_mi = np.array(errors_mi)
    noisy_mi_values = np.array(noisy_mi_values)
    print("\nSimulation finished.")

    # --- Analysis and Plotting ---

    # Calculate empirical mean and std of the error
    mean_error_empirical = np.mean(errors_mi)
    std_error_empirical = np.std(errors_mi)

    print(f"\n--- Empirical Error E_MI = MI_true - MI' ---")
    print(f"Mean of error (E[E_MI]): {mean_error_empirical:.4f}")
    print(f"Std. dev of error (sqrt(Var(E_MI))): {std_error_empirical:.4f}")

    # Plotting the distribution of errors
    plt.figure(figsize=(14, 7)) # Increased figure size

    plt.subplot(1, 2, 1)
    count, bins, ignored = plt.hist(errors_mi, bins=50, density=True, alpha=0.7, label='Empirical Error')

    # Fit a Gaussian to the errors
    mu_fit, std_fit = norm.fit(errors_mi)
    # xmin, xmax = plt.xlim() # Use calculated bins for x range for better fit display
    x = np.linspace(bins[0], bins[-1], 100)
    p = norm.pdf(x, mu_fit, std_fit)
    plt.plot(x, p, 'k', linewidth=2, label=f'Gaussian Fit\n($\mu=${mu_fit:.3f}, $\sigma=${std_fit:.3f})')

    plt.title(f'Distribution of MI Error ($MI_{{true}} - MI\'$)')
    plt.xlabel(f"Error ({'bits' if LOG_BASE==2 else 'nats'})")
    plt.ylabel('Density')
    plt.legend(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    count_mi, bins_mi, ignored_mi = plt.hist(noisy_mi_values, bins=50, density=True, alpha=0.7, color='orange', label='Distribution of $MI\'$')
    plt.axvline(mi_true, color='blue', linestyle='dashed', linewidth=2, label=f'$MI_{{true}} = {mi_true:.3f}$')
    # xmin_mi, xmax_mi = plt.xlim() # Use calculated bins
    x_mi = np.linspace(bins_mi[0], bins_mi[-1], 100)
    # Fit Gaussian to noisy MI' values
    mu_mi_fit, std_mi_fit = norm.fit(noisy_mi_values)
    p_mi = norm.pdf(x_mi, mu_mi_fit, std_mi_fit)
    plt.plot(x_mi, p_mi, 'k', linewidth=2, label=f'Gaussian Fit for $MI\'$\n($\mu=${mu_mi_fit:.3f}, $\sigma=${std_mi_fit:.3f})')
    plt.title('Distribution of Noisy MI ($MI\'$)')
    plt.xlabel(f"$MI'$ ({'bits' if LOG_BASE==2 else 'nats'})")
    plt.ylabel('Density')
    plt.legend(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(f"MI Noise Simulation: {NUM_ROWS}x{NUM_COLS} table, N={TOTAL_TRUE_COUNT}, Noise $\sigma$={NOISE_SIGMA}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
    plt.show()

    print(f"\n--- Distribution of Noisy MI (MI') ---")
    print(f"Mean of MI': {np.mean(noisy_mi_values):.4f}")
    print(f"Std. dev of MI': {np.std(noisy_mi_values):.4f}")
    print(f"Bias (E[MI'] - MI_true): {np.mean(noisy_mi_values) - mi_true:.4f}")
    print(f"Note: Mean of error E[E_MI] = E[MI_true - MI'] = MI_true - E[MI']")
    print(f"      So, E[E_MI] should be approx. -(Bias): {-(np.mean(noisy_mi_values) - mi_true):.4f}")
    print(f"      This matches the empirical mean_error_empirical: {mean_error_empirical:.4f} (approx.)")
