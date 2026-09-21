import numpy as np
from sklearn.decomposition import NMF

def poisson_nmf(X, n_components=5, max_iter=500, tol=1e-7, random_state=0):
    np.random.seed(random_state)
    n, m = X.shape
    W = np.random.rand(n, n_components)
    H = np.random.rand(n_components, m)

    for iteration in range(max_iter):
        WH = W @ H + 1e-10  # avoid division by 0
        H *= (W.T @ (X / WH)) / (W.T @ np.ones_like(X))
        WH = W @ H + 1e-10
        W *= ((X / WH) @ H.T) / (np.ones_like(X) @ H.T)

        if iteration % 50 == 0:
            loss = np.sum(WH - X * np.log(WH + 1e-10))
            print(f"Iter {iteration}: Poisson loss = {loss:.2f}")

    return W @ H, W, H

def rowwise_normalize(M):
    row_sums = M.sum(axis=1, keepdims=True)
    return M / (row_sums + 1e-10)

def avg_total_variation(P, Q):
    """
    Computes average total variation distance across rows.
    Assumes P and Q are row-normalized (distributions).
    """
    return np.mean(0.5 * np.sum(np.abs(P - Q), axis=1))


def nmf_kl_denoise(X_noisy, n_components=2, max_iter=500, random_state=0):
    model = NMF(
        n_components=n_components,
        init='random',
        solver='mu',  # multiplicative updates
        beta_loss='kullback-leibler',
        max_iter=max_iter,
        random_state=random_state,
        tol=1e-4
    )
    W = model.fit_transform(X_noisy)
    H = model.components_
    return W @ H

def project_onto_simplex(v, z=1):
    """Project vector v onto the probability simplex of sum z"""
    v = np.maximum(v, 0)
    if v.sum() == z:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(u)+1) > (cssv - z))[0][-1]
    theta = (cssv[rho] - z) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def denoise_pgd_simplex(X_noisy, n_iter=100, lr=0.1, original_row_sums=None):
    """Denoise using projected gradient descent with simplex row constraints"""
    X_denoised = np.copy(X_noisy)
    if original_row_sums is None:
        original_row_sums = X_noisy.sum(axis=1)

    for it in range(n_iter):
        grad = X_denoised - X_noisy
        X_denoised -= lr * grad

        # Project each row to simplex
        for i in range(X_denoised.shape[0]):
            X_denoised[i] = project_onto_simplex(X_denoised[i], z=original_row_sums[i])

    return X_denoised



# Generate synthetic data
#
while True :
    #np.random.seed(np.random.randint(10000000))
    true_counts = np.random.poisson(20, size=(20, 10))

    # Add Gaussian noise
    noisy = true_counts + np.random.normal(0, 50.0, size=(20, 10))
    noisy = np.clip(noisy, 0, None)
    #noisy = np.abs(noisy)
    # Denoise using Poisson NMF
    denoised, W, H = poisson_nmf(noisy, n_components=5)
    pjd_denoised = denoise_pgd_simplex(noisy, n_iter=300, lr=0.2, original_row_sums=true_counts.sum(axis=1))
    kl_denoised = nmf_kl_denoise(noisy, max_iter=300)


    # Normalize to probability distributions (row-wise)
    true_dist = rowwise_normalize(true_counts)
    noisy_dist = rowwise_normalize(noisy)
    denoised_dist1 = rowwise_normalize(denoised)
    denoised_dist2 = rowwise_normalize(pjd_denoised)
    denoised_dist3 = rowwise_normalize(kl_denoised)

    # Measure distributional shift
    tvd_noisy = avg_total_variation(true_dist, noisy_dist)
    tvd_denoised = avg_total_variation(true_dist, denoised_dist1)

    #pj_noisy = avg_total_variation(true_dist, noisy_dist)
    pjd_denoised_dist = avg_total_variation(true_dist, denoised_dist2)
    kl_denoised_dist = avg_total_variation(true_dist, denoised_dist3)


    # Print
    print("\nOriginal counts (first 5 rows):")
    print(np.round(true_counts[:5]))
    print("\nNoisy counts (first 5 rows):")
    print(np.round(noisy[:5]))
    print("\nDenoised matrix (Poisson NMF, first 5 rows):")
    print(np.round(denoised[:5], 1))
    print("\nDenoised matrix (Projected Gradient first 5 rows):")
    print(np.round(pjd_denoised[:5], 1))


    print("#####Error estimation######")
    print('L2 : ')
    print(f"Error (L2 norm): Before = {np.linalg.norm((true_counts - noisy).flatten()):.2f}, Poisson NMF = {np.linalg.norm((true_counts - denoised).flatten()):.2f}")
    print(f"Error (L2 norm): Before = {np.linalg.norm((true_counts - noisy).flatten()):.2f}, Projected Grad = {np.linalg.norm((true_counts - pjd_denoised).flatten()):.2f}")
    print(f"Error (L2 norm): Before = {np.linalg.norm((true_counts - noisy).flatten()):.2f}, Projected Grad = {np.linalg.norm((true_counts - kl_denoised).flatten()):.2f}")
    print("Dist : ")
    print(f"Distributional shift (Avg. TVD): Before = {tvd_noisy:.4f}, After = {tvd_denoised:.4f}")
    print(f"Distributional shift (Avg. TVD): Before = {tvd_noisy:.4f}, After = {pjd_denoised_dist:.4f}")
    print(f"Distributional shift (Avg. TVD): Before = {tvd_noisy:.4f}, After = {kl_denoised_dist:.4f}")
