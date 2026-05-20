"""
Synthetic weekly MMM dataset with known ground-truth parameters.

Generative story (funnel):
    spend → geometric adstock → scale → Hill saturation → media contribution
    opportunities = baseline + Σ media + controls + noise
    sales = opportunities × close_rate + noise
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from bayesian_mmm import config as cfg
from bayesian_mmm.transforms.adstock import geometric_adstock
from bayesian_mmm.transforms.saturation import hill_saturation


def _week_dates(n_weeks: int, start: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n_weeks, freq="W-MON")


def _fourier_baseline(week_index: np.ndarray) -> np.ndarray:
    """Annual + semi-annual Fourier terms on week index 0..n-1."""
    t = week_index.astype(float)
    n = len(t)
    annual_phase = 2.0 * np.pi * t / 52.0
    baseline = np.full(n, cfg.BASELINE_INTERCEPT, dtype=float)

    for order, (coef_cos, coef_sin) in enumerate(cfg.BASELINE_FOURIER_COEFS, start=1):
        freq = order  # 1 = annual, 2 = semi-annual on weekly grid
        baseline += coef_cos * np.cos(freq * annual_phase)
        baseline += coef_sin * np.sin(freq * annual_phase)
    return baseline


def _q4_boost(dates: pd.DatetimeIndex) -> np.ndarray:
    """Additive Q4 lift (calendar months Oct–Dec)."""
    return np.where(dates.month.isin([10, 11, 12]), cfg.Q4_BOOST, 0.0).astype(float)


def _simulate_spend_series(
    rng: np.random.Generator,
    base_level: float,
    n_weeks: int,
    seasonal_phase: float,
) -> np.ndarray:
    """
    Log-normal AR(1) spend with mild seasonality and occasional pulses.
    """
    log_spend = np.log(base_level) + np.zeros(n_weeks)
    shock = rng.normal(0.0, cfg.SPEND_LOG_SIGMA, size=n_weeks)

    weeks = np.arange(n_weeks)
    season = cfg.SPEND_SEASONAL_AMPLITUDE * np.sin(
        2.0 * np.pi * weeks / 52.0 + seasonal_phase
    )

    for t in range(n_weeks):
        if t == 0:
            log_spend[t] = np.log(base_level) + shock[t] + season[t]
        else:
            log_spend[t] = (
                cfg.SPEND_AR_PHI * log_spend[t - 1]
                + (1.0 - cfg.SPEND_AR_PHI) * np.log(base_level)
                + shock[t]
                + season[t]
            )
        if rng.random() < cfg.SPEND_PULSE_PROB:
            log_spend[t] += np.log(cfg.SPEND_PULSE_SCALE)

    return np.exp(log_spend)


def _media_contribution(
    spend: np.ndarray,
    channel: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Adstock → scale → Hill → beta * saturated."""
    alpha = cfg.CHANNEL_ADSTOCK_ALPHA[channel]
    slope = cfg.CHANNEL_HILL_SLOPE[channel]
    half_sat = cfg.CHANNEL_HILL_HALF_SAT[channel]
    beta = cfg.CHANNEL_BETA[channel]

    adstocked = geometric_adstock(spend, alpha)
    scale = np.max(adstocked)
    if scale <= 0:
        scaled = np.zeros_like(adstocked)
    else:
        scaled = adstocked / scale

    saturated = hill_saturation(scaled, slope, half_sat)
    contribution = beta * saturated

    meta = {
        "adstock_alpha": alpha,
        "hill_slope": slope,
        "hill_half_sat": half_sat,
        "beta": beta,
        "adstock_scale_max": float(scale),
    }
    return contribution, meta


def _build_true_params(channel_meta: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Serialize every parameter used in the DGP for posterior validation."""
    return {
        "rng_seed": cfg.RNG_SEED,
        "n_weeks": cfg.N_WEEKS,
        "start_date": cfg.START_DATE,
        "media_channels": cfg.MEDIA_CHANNELS,
        "spend_columns": cfg.SPEND_COLUMNS,
        "control_columns": cfg.CONTROL_COLUMNS,
        "kpi_columns": cfg.KPI_COLUMNS,
        "channel_adstock_alpha": cfg.CHANNEL_ADSTOCK_ALPHA,
        "channel_hill_slope": cfg.CHANNEL_HILL_SLOPE,
        "channel_hill_half_sat": cfg.CHANNEL_HILL_HALF_SAT,
        "channel_beta": cfg.CHANNEL_BETA,
        "channel_adstock_scale_max": {
            ch: channel_meta[ch]["adstock_scale_max"] for ch in cfg.MEDIA_CHANNELS
        },
        "baseline": {
            "intercept": cfg.BASELINE_INTERCEPT,
            "fourier_order": cfg.BASELINE_FOURIER_ORDER,
            "fourier_coefs": cfg.BASELINE_FOURIER_COEFS,
            "q4_boost": cfg.Q4_BOOST,
        },
        "control_gamma": cfg.CONTROL_GAMMA,
        "true_close_rate": cfg.TRUE_CLOSE_RATE,
        "opportunities_noise_std": cfg.OPPORTUNITIES_NOISE_STD,
        "sales_noise_std": cfg.SALES_NOISE_STD,
        "spend_simulation": {
            "base_level": cfg.SPEND_BASE_LEVEL,
            "log_sigma": cfg.SPEND_LOG_SIGMA,
            "seasonal_amplitude": cfg.SPEND_SEASONAL_AMPLITUDE,
            "ar_phi": cfg.SPEND_AR_PHI,
            "pulse_prob": cfg.SPEND_PULSE_PROB,
            "pulse_scale": cfg.SPEND_PULSE_SCALE,
        },
        "competitor_spend_base": cfg.COMPETITOR_SPEND_BASE,
        "competitor_spend_sigma": cfg.COMPETITOR_SPEND_SIGMA,
        "google_trend_base": cfg.GOOGLE_TREND_BASE,
        "google_trend_sigma": cfg.GOOGLE_TREND_SIGMA,
        "google_trend_ar": cfg.GOOGLE_TREND_AR,
    }


def generate_synthetic_dataset(seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Simulate the full weekly MMM panel and return (dataframe, true_params).

    Parameters
    ----------
    seed : int, optional
        RNG seed; defaults to ``config.RNG_SEED``.
    """
    seed = cfg.RNG_SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    n = cfg.N_WEEKS
    dates = _week_dates(n, cfg.START_DATE)
    week_index = np.arange(n)

    # --- spends ---
    spend_data: dict[str, np.ndarray] = {}
    channel_meta: dict[str, dict[str, Any]] = {}
    for i, ch in enumerate(cfg.MEDIA_CHANNELS):
        col = f"{ch}_Spend"
        spend_data[col] = _simulate_spend_series(
            rng,
            cfg.SPEND_BASE_LEVEL[ch],
            n,
            seasonal_phase=float(i) * 0.7,
        )

    # --- controls ---
    log_comp = np.log(cfg.COMPETITOR_SPEND_BASE) + rng.normal(
        0.0, cfg.COMPETITOR_SPEND_SIGMA, size=n
    )
    competitor_spend = np.exp(log_comp)

    trend = np.empty(n)
    trend[0] = cfg.GOOGLE_TREND_BASE + rng.normal(0.0, cfg.GOOGLE_TREND_SIGMA)
    for t in range(1, n):
        trend[t] = (
            cfg.GOOGLE_TREND_BASE
            + cfg.GOOGLE_TREND_AR * (trend[t - 1] - cfg.GOOGLE_TREND_BASE)
            + rng.normal(0.0, cfg.GOOGLE_TREND_SIGMA)
        )
    google_trend_competitor = np.clip(trend, 0.0, None)

    # --- media → opportunities ---
    media_total = np.zeros(n)
    for ch in cfg.MEDIA_CHANNELS:
        col = f"{ch}_Spend"
        contrib, meta = _media_contribution(spend_data[col], ch)
        media_total += contrib
        channel_meta[ch] = meta

    baseline = _fourier_baseline(week_index) + _q4_boost(dates)
    control_effect = (
        cfg.CONTROL_GAMMA["competitor_spend"] * competitor_spend
        + cfg.CONTROL_GAMMA["google_trend_competitor"] * google_trend_competitor
    )
    opp_noise = rng.normal(0.0, cfg.OPPORTUNITIES_NOISE_STD, size=n)
    opportunities = baseline + media_total + control_effect + opp_noise
    opportunities = np.clip(opportunities, 50.0, None)

    # --- funnel: sales ---
    sales_noise = rng.normal(0.0, cfg.SALES_NOISE_STD, size=n)
    sales = opportunities * cfg.TRUE_CLOSE_RATE + sales_noise
    sales = np.clip(sales, 0.0, None)

    df = pd.DataFrame(
        {
            cfg.DATE_COLUMN: dates,
            **spend_data,
            "competitor_spend": competitor_spend,
            "google_trend_competitor": google_trend_competitor,
            "opportunities": opportunities,
            "sales": sales,
        }
    )

    true_params = _build_true_params(channel_meta)
    true_params["seed_used"] = seed
    return df, true_params


def save_synthetic_dataset(seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Generate and write ``weekly_mmm.csv`` and ``true_params.json``."""
    df, true_params = generate_synthetic_dataset(seed=seed)

    cfg.DATA_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.WEEKLY_MMM_CSV, index=False)

    with cfg.TRUE_PARAMS_JSON.open("w", encoding="utf-8") as f:
        json.dump(true_params, f, indent=2)

    return df, true_params
