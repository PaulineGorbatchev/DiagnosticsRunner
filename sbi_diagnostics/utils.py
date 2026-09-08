import jax
import numpy as np


def summarize_1d(arr):
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    return mean, std, iqr


def sample_posterior(posterior, x_cond, n: int, seed: int):
    key = jax.random.PRNGKey(seed)
    s = posterior.sample(x=x_cond, num_samples=n, key=key)
    return np.asarray(s)


def sample_prior_resample_thetas(thetas, n: int, seed: int):
    """Proxy for the prior by resampling training thetas."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, thetas.shape[0], size=n)
    return np.asarray(thetas[idx])


def mmd_rbf(X, Y, gamma=None, max_points: int = 3000, seed: int = 0):
    """MMD² with RBF kernel on subsamples."""
    rng = np.random.default_rng(seed)
    n = min(max_points, X.shape[0])
    m = min(max_points, Y.shape[0])
    Xs = X[rng.choice(X.shape[0], n, replace=False)]
    Ys = Y[rng.choice(Y.shape[0], m, replace=False)]

    Z = np.vstack([Xs, Ys])
    if gamma is None:
        Z2 = Z[rng.choice(Z.shape[0], min(800, Z.shape[0]), replace=False)]
        dists = np.sum((Z2[:, None, :] - Z2[None, :, :]) ** 2, axis=-1)
        med = np.median(dists[dists > 0])
        gamma = 1.0 / (2.0 * med + 1e-12)

    def k(A, B):
        d2 = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d2)

    Kxx = k(Xs, Xs)
    Kyy = k(Ys, Ys)
    Kxy = k(Xs, Ys)
    np.fill_diagonal(Kxx, 0.0)
    np.fill_diagonal(Kyy, 0.0)

    mmd2 = (
        Kxx.sum() / (n * (n - 1) + 1e-12)
        + Kyy.sum() / (m * (m - 1) + 1e-12)
        - 2.0 * Kxy.mean()
    )
    return float(mmd2), float(gamma)
