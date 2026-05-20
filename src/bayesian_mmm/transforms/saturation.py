"""Hill (Michaelis-Menten) saturation transform."""

from __future__ import annotations

import numpy as np


def hill_saturation(x: np.ndarray, slope: float, half_sat: float) -> np.ndarray:
    """
    Hill / Michaelis-Menten saturation on non-negative inputs.

        saturated = x^slope / (half_sat^slope + x^slope)

    Values lie in [0, 1]. At x = half_sat, output is 0.5 (for slope > 0).

    Parameters
    ----------
    x : np.ndarray
        Non-negative driver (e.g. scaled adstocked spend).
    slope : float
        Hill exponent (steepness of the curve).
    half_sat : float
        Half-saturation point (x at which response is 0.5).

    Returns
    -------
    np.ndarray
        Saturated response in [0, 1], same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)
    x_nonneg = np.maximum(x, 0.0)
    half_sat = max(float(half_sat), 1e-12)

    x_pow = np.power(x_nonneg, slope)
    k_pow = np.power(half_sat, slope)
    denom = k_pow + x_pow
    return np.where(denom > 0, x_pow / denom, 0.0)
