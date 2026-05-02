import numpy as np

def mean_boot(x, ci=0.95, B=2000):
    """
    Percentile bootstrap CI for the mean of 1D samples.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    a = 1.0 - ci

    rng = np.random.default_rng()  # 不固定种子
    idx = rng.integers(0, n, size=(B, n))
    boot = x[idx].mean(axis=1)

    lo = np.quantile(boot, a / 2.0)
    hi = np.quantile(boot, 1.0 - a / 2.0)
    return float(lo), float(hi)


def fit_boot(n_list, Y, ci=0.95, B=2000, eps=1e-12):
    """
    Bootstraps means over trials then fits mean(n) ≈ a * b^n (log-space).
    Returns percentile bootstrap CIs for (a, b).

    n_list: (N,)
    Y:      (N, T)  (same T for each n)
    """
    rng = np.random.default_rng()  # 不固定种子

    n = np.asarray(n_list, dtype=float)
    Y = np.asarray(Y, dtype=float)
    N, T = Y.shape
    alpha = 1.0 - ci

    # log(mean) = log a + n log b  => linear regression on (1, n)
    X = np.column_stack([np.ones(N, dtype=float), n])      # (N,2)
    pinv = np.linalg.inv(X.T @ X) @ X.T                    # (2,N)

    # 每次对每个 n 抽 T 个（与原样本量一致）
    idx = rng.integers(0, T, size=(B, N, T))               # (B,N,T)

    # means: (B,N)
    means = Y[np.arange(N)[None, :, None], idx].mean(axis=2)

    logy = np.log(np.clip(means, eps, None))               # (B,N)
    beta = (pinv @ logy.T).T                               # (B,2)

    a_boot = np.exp(beta[:, 0])
    b_boot = np.exp(beta[:, 1])

    qlo, qhi = alpha/2, 1 - alpha/2
    a_ci = (float(np.quantile(a_boot, qlo)), float(np.quantile(a_boot, qhi)))
    b_ci = (float(np.quantile(b_boot, qlo)), float(np.quantile(b_boot, qhi)))
    return a_ci, b_ci
