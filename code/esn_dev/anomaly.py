import numpy as np
from scipy import special

def compute_window_stats(error, small_window, large_window):
    """
    Compute long-term mean, long-term std, and short-term mean 
    for nD input with time as the first axis.

    Parameters
    ----------
    error : ndarray
        An n-dimensional array with time as the first dimension (shape: [time, ...]).
    small_window : int
        Size of the small (short-term) window.
    large_window : int
        Size of the large (long-term) window.

    Returns
    -------
    lw_mu, lw_std, sw_mu : tuple of ndarrays
        Arrays of the same shape as `error`, representing 
        long-term mean, long-term std, and short-term mean.
    """
    time_len = error.shape[0]

    lw_mu  = np.zeros_like(error)
    lw_std = np.zeros_like(error)
    sw_mu  = np.zeros_like(error)

    # Initialize first timestep
    lw_mu[0]  = error[0]
    lw_std[0] = np.std(error[:2], axis=0)
    sw_mu[0]  = error[0]

    for i in range(1, time_len):
        lw_start = max(0, i - large_window + 1)
        sw_end   = min(i + small_window, time_len)

        lw_err = error[lw_start:i]
        sw_err = error[i:sw_end]

        lw_mu[i]  = np.mean(lw_err, axis=0)
        lw_std[i] = np.std(lw_err, axis=0)
        sw_mu[i]  = np.mean(sw_err, axis=0)

    return lw_mu, lw_std, sw_mu


def compute_sliding_score(error, lw_mu, lw_std, sw_mu, 
                          std_min=2.5, std_max=5.0, std_center=1.8):
    """
    Compute anomaly scores given precomputed statistics.

    1 = normal, 0 = extremely abnormal
    """
    time_len = error.shape[0]

    # Safe long-term std
    std_safe = np.clip(lw_std, std_min, std_max)

    # Normalized deviation
    x = np.maximum(0, sw_mu - lw_mu) / std_safe

    # Smooth score: higher x → less normal
    steepness = 6.0
    scores = special.expit(-steepness * (x - std_center))  # strictly in (0,1)

    # Warm-up period (assume normal)
    scores[:time_len // 10] = 1.0

    return scores, x


def cumulative_distribution(x):
    return special.erf(x/2.**.5)

def qfunction(x):
    return 1 - cumulative_distribution(x)



def sliding_score_1D(error, small_window, large_window):
    scores = np.empty(error.shape)
    lw_mu = np.zeros_like(scores)
    lw_std = np.zeros_like(scores)
    sw_mu = np.zeros_like(scores)

    lw_mu[0] = error[0]
    lw_std[0] = error[:2].std(axis=0)
    sw_mu[0] = error[0]

    for i in range(1, error.shape[0]):
        lw_start = max(0, i - large_window + 1)
        sw_end   = min(i + small_window, error.shape[0])

        lw_err = error[lw_start:i]
        sw_err = error[i:sw_end]

        lw_mu[i] = lw_err.mean(axis=0)
        lw_std[i] = lw_err.std(axis=0)
        sw_mu[i] = sw_err.mean(axis=0)

        x = np.maximum(0, sw_mu[i] - lw_mu[i]) /(lw_std[i] + (np.abs(lw_std[i])<1e-10))
        
        s = qfunction(x)
        scores[i] = s

    scores[:large_window//10] = 1.
    return scores, lw_mu, lw_std, sw_mu, x