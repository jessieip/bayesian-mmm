"""Geometric adstock (carryover) transform."""

from __future__ import annotations

import numpy as np


def geometric_adstock(x: np.ndarray, alpha: float) -> np.ndarray:
    """
    Apply geometric (infinite-lag) adstock to a 1-D spend series.

    Recursive definition (causal, week-by-week):
        x_adstock[0] = x[0]
        x_adstock[t] = x[t] + alpha * x_adstock[t - 1]   for t > 0

    Parameters
    ----------
    x : np.ndarray
        Raw weekly spend (or impressions); shape (n_periods,).
    alpha : float
        Retention / decay rate in (0, 1). Higher alpha = longer carryover.

    Returns
    -------
    np.ndarray
        Adstocked series, same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("geometric_adstock expects a 1-D array")

    out = np.empty_like(x)
    out[0] = x[0]
    for t in range(1, len(x)):
        out[t] = x[t] + alpha * out[t - 1]
    return out
