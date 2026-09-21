import numpy as np
from sklearn.metrics import mutual_info_score
from scipy.optimize import minimize
import time

def generate_contingency_table(n_rows, n_cols):
    table = np.random.randint(0, 30, size=(n_rows, n_cols))
    return table

def compute_mutual_info(table):
    x_idx, y_idx = np.indices(table.shape)
    x_vals, y_vals = [], []
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            count = int(table[i, j])
            x_vals.extend([i] * count)
            y_vals.extend([j] * count)
    return mutual_info_score(x_vals, y_vals)


def compute_mutual_info_denoised(table, alpha=1e-3):
    """
    Computes mutual information from a noisy contingency table with Dirichlet smoothing.

    Args:
        table (np.ndarray): 2D array of noisy counts.
        alpha (float): smoothing constant for Dirichlet prior (default: 1e-3).

    Returns:
        float: mutual information estimate.
    """
    # Step 1: Add Dirichlet prior (smoothing)
    smoothed_table = table + alpha

    # Step 2: Normalize to get joint probabilities
    joint_prob = smoothed_table / np.sum(smoothed_table)

    # Step 3: Compute marginals
    px = np.sum(joint_prob, axis=1, keepdims=True)
    py = np.sum(joint_prob, axis=0, keepdims=True)

    # Step 4: Compute MI using the definition
    with np.errstate(divide='ignore', invalid='ignore'):
        log_term = np.log(joint_prob / (px @ py))
        log_term[np.isnan(log_term)] = 0
        log_term[np.isinf(log_term)] = 0

    mi = np.sum(joint_prob * log_term)
    return mi



def compute_mutual_info_bootstrap(table, alpha=1e-3, n_bootstrap=1000, seed=None, ci=0.95):
    """
    Estimates mutual information from a noisy contingency table with uncertainty via bootstrapping.

    Args:
        table (np.ndarray): 2D array of noisy counts.
        alpha (float): Dirichlet smoothing constant.
        n_bootstrap (int): Number of bootstrap samples.
        seed (int or None): Random seed.
        ci (float): Confidence level (e.g., 0.95 for 95% CI).

    Returns:
        tuple: (MI_estimate, (CI_lower, CI_upper), all_bootstrap_mis)
    """
    rng = np.random.default_rng(seed)

    def estimate_mi(smoothed_table):
        joint_prob = smoothed_table / np.sum(smoothed_table)
        px = np.sum(joint_prob, axis=1, keepdims=True)
        py = np.sum(joint_prob, axis=0, keepdims=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            log_term = np.log(joint_prob / (px @ py))
            log_term[np.isnan(log_term)] = 0
            log_term[np.isinf(log_term)] = 0

        return np.sum(joint_prob * log_term)

    # Smoothing the original table
    smoothed_table = table + alpha
    mi_estimate = estimate_mi(smoothed_table)

    # Bootstrap sampling
    flat = smoothed_table.flatten()
    n_cells = flat.shape[0]
    bootstrap_mis = []

    for _ in range(n_bootstrap):
        resample = rng.choice(n_cells, size=n_cells, replace=True)
        resampled_table = flat[resample].reshape(table.shape)
        bootstrap_mis.append(estimate_mi(resampled_table))

    # Compute confidence interval
    lower = np.percentile(bootstrap_mis, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_mis, (1 + ci) / 2 * 100)

    return mi_estimate, (lower, upper), bootstrap_mis
    #return mi_estimate




def compute_bayesian_smoothed_mi(table, alpha=1.0):
    """
    Compute mutual information using Bayesian smoothing (Dirichlet prior).

    Parameters:
    - table: 2D numpy array (contingency table)
    - alpha: prior strength (pseudo-count). Default = 1.0 (Laplace smoothing)

    Returns:
    - Smoothed mutual information estimate
    """
    table = np.array(table)
    n_rows, n_cols = table.shape
    k = n_rows * n_cols
    total = np.sum(table)

    # Smoothed counts (posterior mean)
    smoothed_table = table + alpha
    smoothed_total = total + alpha * k
    prob_matrix = smoothed_table / smoothed_total

    # Marginals
    px = np.sum(prob_matrix, axis=1)
    py = np.sum(prob_matrix, axis=0)

    # Compute mutual information
    mi = 0.0
    for i in range(n_rows):
        for j in range(n_cols):
            pij = prob_matrix[i, j]
            if pij > 0:
                mi += pij * np.log(pij / (px[i] * py[j]))

    return mi


def isotropic_gaussian_mechanism(table, sigma):
    noise = np.random.normal(0, sigma, size=table.shape)
    return table + noise

def rdgm_gaussian_mechanism(table, sigma):
    T = table.astype(float).flatten()
    p = T.shape[0]

    # Create projection matrix to enforce zero-sum (orthogonal to all-ones vector)
    one_vec = np.ones((p, 1))
    P = np.eye(p) - (one_vec @ one_vec.T) / p  # Projection onto sum-zero subspace

    # Apply Gaussian noise in sensitive subspace only
    noise = np.random.normal(0, sigma, size=p)
    projected_noise = P @ noise
    print(f'[Norm] of original noise {np.linalg.norm(noise)} projected noise {np.linalg.norm(projected_noise)}')
    print(f'[StdDev] of original noise {np.std(noise)} projected noise {np.std(projected_noise)}')

    T_noisy = T + projected_noise
    return T_noisy.reshape(table.shape)

def bayesian_smoothing(table, alpha=1.0):
    """
    Apply Bayesian (Dirichlet) smoothing to a noisy contingency table.

    Parameters:
        table (np.ndarray): 2D array of noisy counts.
        alpha (float): Pseudocount to add to each cell (default: 1.0).

    Returns:
        smoothed_probs (np.ndarray): Normalized 2D array of smoothed joint probabilities.
    """
    smoothed_counts = table + alpha
    total = np.sum(smoothed_counts)
    smoothed_probs = smoothed_counts
    return smoothed_probs


def james_stein_shrinkage(noisy_table, sigma, mu=None):
    T_noisy = noisy_table.flatten()
    p = T_noisy.shape[0]

    if mu is None:
        mu = np.full(p, 1/p)

    diff = T_noisy - mu
    norm_squared = np.sum(diff ** 2)

    shrinkage_factor = max(0, 1 - ((p - 2) * sigma**2) / norm_squared)
    T_shrunk = shrinkage_factor * diff + mu

    return T_shrunk.reshape(noisy_table.shape)

def rank_deficient_gaussian_with_shrinkage(contingency_table, sigma):
    # Flatten the contingency table into a 1D array
    T = contingency_table.astype(float).flatten()
    p = T.shape[0]  # number of categories (size of the contingency table)

    # Create the projection matrix to enforce zero-sum (orthogonal to all-ones vector)
    one_vec = np.ones((p, 1))
    P = np.eye(p) - (one_vec @ one_vec.T) / p  # Projection onto sum-zero subspace

    # Apply Gaussian noise in the sensitive subspace only
    noise = np.random.normal(0, sigma, size=p)
    projected_noise = P @ noise  # Project noise onto the subspace

    # Add the noise to the contingency table
    T_noisy = T + projected_noise

    # Now apply James-Stein shrinkage to the noisy table
    #T_shrunk = james_stein_shrinkage(T_noisy, sigma)
    T_shrunk = james_stein_shrinkage(np.maximum(T_noisy, 0), sigma)

    # Return the shrunk noisy table reshaped to the original contingency table shape
    return np.abs(T_shrunk.reshape(contingency_table.shape))


def structured_bayesian_smoothing(all_tables, alpha=1.0):
    """
    all_tables: list of 2D numpy arrays (one per (X,Y) pair), already summed over workers.
    alpha: pseudo-count for Dirichlet smoothing.
    """
    smoothed_tables = []
    total_sum = sum(np.sum(np.maximum(tbl, 0)) for tbl in all_tables)

    for table in all_tables:
        tbl_clipped = np.maximum(table, 0)
        tbl_smoothed = tbl_clipped + alpha
        smoothed_tables.append(tbl_smoothed)

    # Rescale all tables so that their total sums match original total_sum
    total_smoothed = sum(np.sum(tbl) for tbl in smoothed_tables)
    scaling_factor = total_sum / total_smoothed

    smoothed_tables = [tbl * scaling_factor for tbl in smoothed_tables]
    return smoothed_tables



# === Demo ===
use_smoothing = True
use_abs = True
use_clipping = False

while True :
    print('----------------------------------\n')
    true_table1 = generate_contingency_table(25, 25)
    true_table2 = generate_contingency_table(25, 25)

    true_table = true_table1+true_table2

    sigma = 50.0
    if use_abs :
        noisy_iso1  = np.abs(isotropic_gaussian_mechanism(true_table1, sigma))
        noisy_iso2  = np.abs(isotropic_gaussian_mechanism(true_table2, sigma))

        noisy_rdgm1 = np.abs(rdgm_gaussian_mechanism(true_table1, sigma))
        noisy_rdgm2 = np.abs(rdgm_gaussian_mechanism(true_table2, sigma))

        noisy_rdgmjs1 = np.abs(rank_deficient_gaussian_with_shrinkage(true_table1, sigma))
        noisy_rdgmjs2 = np.abs(rank_deficient_gaussian_with_shrinkage(true_table2, sigma))

    if use_clipping :
        noisy_iso1  = isotropic_gaussian_mechanism(true_table1, sigma).clip(min=0)
        noisy_iso2  = isotropic_gaussian_mechanism(true_table2, sigma).clip(min=0)

        noisy_rdgm1 = rdgm_gaussian_mechanism(true_table1, sigma).clip(min=0)
        noisy_rdgm2 = rdgm_gaussian_mechanism(true_table2, sigma).clip(min=0)

        noisy_rdgmjs1 = rank_deficient_gaussian_with_shrinkage(true_table1, sigma).clip(min=0)
        noisy_rdgmjs2 = rank_deficient_gaussian_with_shrinkage(true_table2, sigma).clip(min=0)

    else :
        noisy_iso1  = isotropic_gaussian_mechanism(true_table1, sigma)
        noisy_iso2  = isotropic_gaussian_mechanism(true_table2, sigma)

        noisy_rdgm1 = rdgm_gaussian_mechanism(true_table1, sigma)
        noisy_rdgm2 = rdgm_gaussian_mechanism(true_table2, sigma)

        noisy_rdgmjs1 = rank_deficient_gaussian_with_shrinkage(true_table1, sigma)
        noisy_rdgmjs2 = rank_deficient_gaussian_with_shrinkage(true_table2, sigma)

    if use_smoothing :
        noisy_iso = bayesian_smoothing(noisy_iso1 + noisy_iso2)
        noisy_rdgm = bayesian_smoothing(noisy_rdgm1 + noisy_rdgm2)
        noisy_rdgmjs = bayesian_smoothing(noisy_rdgmjs1 + noisy_rdgmjs2)

    else :
        noisy_iso = noisy_iso1 + noisy_iso2
        noisy_rdgm = noisy_rdgm1 + noisy_rdgm2
        noisy_rdgmjs = noisy_rdgmjs1 + noisy_rdgmjs2


    print("Isotropic error : %.4f" % np.linalg.norm((true_table - noisy_iso).flatten()))
    print("RDGM error : %.4f" % np.linalg.norm((true_table - noisy_rdgm).flatten()))
    print("RDGMJS error : %.4f" % np.linalg.norm((true_table - noisy_rdgmjs).flatten()))
    print("True Mutual Info:          %.4f" % compute_mutual_info(true_table))
    print("Isotropic Gaussian MI:     %.4f" % compute_mutual_info(noisy_iso))
    print("RDGM Projected Gaussian MI: %.4f" % compute_mutual_info(noisy_rdgm))
    print("RDGMJS Projected Gaussian MI: %.4f" % compute_mutual_info(noisy_rdgmjs))
    print('###### Denoised MI estimation #########')
    print("Isotropic Gaussian MI:     %.4f" % compute_mutual_info_denoised(noisy_iso))
    print("RDGM Projected Gaussian MI: %.4f" % compute_mutual_info_denoised(noisy_rdgm))
    print("RDGMJS Projected Gaussian MI: %.4f" % compute_mutual_info_denoised(noisy_rdgmjs))
    print('###### Basyian smoothing MI estimation #########')
    print("Isotropic Gaussian MI:     %.4f" % compute_bayesian_smoothed_mi(noisy_iso))
    print("RDGM Projected Gaussian MI: %.4f" % compute_bayesian_smoothed_mi(noisy_rdgm))
    print("RDGMJS Projected Gaussian MI: %.4f" % compute_bayesian_smoothed_mi(noisy_rdgmjs))
    print('###### Boostrap MI estimation #########')
    mi, (ci_lower, ci_upper), mis = compute_mutual_info_bootstrap(noisy_iso)
    print(f"Isotropic : Estimated MI: {mi:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    mi, (ci_lower, ci_upper), mis = compute_mutual_info_bootstrap(noisy_rdgm)
    print(f"RDGM : Estimated MI: {mi:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    mi, (ci_lower, ci_upper), mis = compute_mutual_info_bootstrap(noisy_rdgmjs)
    print(f"RDGMJS : Estimated MI: {mi:.4f}, 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")


#print("\nTrue Table:\n", true_table)
#print("\nNoisy RDGM Table (rounded):\n", np.round(noisy_rdgm))
#print("\nNoisy RDGMJS Table (rounded):\n", np.round(noisy_rdgmjs))
