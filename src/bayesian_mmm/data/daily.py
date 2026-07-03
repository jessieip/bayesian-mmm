"""
Stateful daily synthetic data generation.

Generative story (funnel, daily frequency):
    spend → geometric adstock → scale → Hill → media contribution
    opportunities = baseline + Σ media + controls + noise
    sales = opportunities × close_rate + noise
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from bayesian_mmm import config as cfg
from bayesian_mmm.data.state import SimulationState
from bayesian_mmm.transforms.saturation import hill_saturation


def _fourier_baseline_daily(day_index: int) -> float:
    """Annual + semi-annual Fourier on day index."""
    phase = 2.0 * np.pi * day_index / cfg.DAYS_PER_YEAR
    baseline = cfg.daily_baseline_intercept()
    for order, (coef_cos, coef_sin) in enumerate(cfg.daily_fourier_coefs(), start=1):
        baseline += coef_cos * np.cos(order * phase)
        baseline += coef_sin * np.sin(order * phase)
    return float(baseline)


def _q4_boost_daily(d: date) -> float:
    if d.month in (10, 11, 12):
        return cfg.daily_q4_boost()
    return 0.0


def _simulate_spend_day(
    rng: np.random.Generator,
    channel: str,
    day_index: int,
    seasonal_phase: float,
    log_state: float | None,
) -> tuple[float, float]:
    """One day of log-AR(1) spend; returns (spend, new_log_state)."""
    base = cfg.daily_spend_base(channel)
    log_base = np.log(base)
    shock = rng.normal(0.0, cfg.SPEND_LOG_SIGMA)
    season = cfg.SPEND_SEASONAL_AMPLITUDE * np.sin(
        2.0 * np.pi * day_index / cfg.DAYS_PER_YEAR + seasonal_phase
    )

    if log_state is None:
        log_spend = log_base + shock + season
    else:
        log_spend = (
            cfg.SPEND_AR_PHI * log_state
            + (1.0 - cfg.SPEND_AR_PHI) * log_base
            + shock
            + season
        )

    if rng.random() < cfg.daily_pulse_prob():
        log_spend += np.log(cfg.SPEND_PULSE_SCALE)

    return float(np.exp(log_spend)), float(log_spend)


def _media_contribution_day(
    spend: float,
    channel: str,
    adstock_prev: float,
    scale_max: float,
) -> tuple[float, float, float]:
    """
    One-step adstock → Hill → beta * saturated.

    Returns (contribution, new_adstock, new_scale_max).
    """
    alpha = cfg.daily_adstock_alpha(channel)
    adstock = spend + alpha * adstock_prev
    scale_max = max(scale_max, adstock, 1e-12)
    scaled = adstock / scale_max

    saturated = float(
        hill_saturation(
            np.array([scaled]),
            cfg.CHANNEL_HILL_SLOPE[channel],
            cfg.CHANNEL_HILL_HALF_SAT[channel],
        )[0]
    )
    contribution = cfg.daily_beta(channel) * saturated
    return contribution, adstock, scale_max


def generate_daily_range(
    start: date,
    end: date,
    state: SimulationState,
) -> tuple[pd.DataFrame, SimulationState]:
    """
    Generate daily rows for [start, end] inclusive, updating ``state`` in place.

    Parameters
    ----------
    start, end : date
        Inclusive calendar range.
    state : SimulationState
        Loaded or fresh state; mutated and returned.
    """
    if start > end:
        return pd.DataFrame(), state

    rng = np.random.default_rng(state.rng_seed + state.day_index)
    rows: list[dict] = []
    current = start

    while current <= end:
        day_idx = state.day_index
        row: dict = {cfg.DATE_COLUMN: pd.Timestamp(current)}

        # --- channel spends ---
        for i, ch in enumerate(cfg.MEDIA_CHANNELS):
            col = f"{ch}_Spend"
            log_prev = state.spend_log_state.get(ch)
            spend, log_new = _simulate_spend_day(
                rng, ch, day_idx, float(i) * 0.7, log_prev
            )
            state.spend_log_state[ch] = log_new
            row[col] = spend

        # --- controls ---
        log_comp = np.log(cfg.daily_competitor_spend_base()) + rng.normal(
            0.0, cfg.COMPETITOR_SPEND_SIGMA
        )
        row["competitor_spend"] = float(np.exp(log_comp))

        state.google_trend_last = (
            cfg.GOOGLE_TREND_BASE
            + cfg.GOOGLE_TREND_AR * (state.google_trend_last - cfg.GOOGLE_TREND_BASE)
            + rng.normal(0.0, cfg.GOOGLE_TREND_SIGMA)
        )
        row["google_trend_competitor"] = float(max(state.google_trend_last, 0.0))

        # --- media → opportunities ---
        media_total = 0.0
        for ch in cfg.MEDIA_CHANNELS:
            col = f"{ch}_Spend"
            contrib, adstock_new, scale_new = _media_contribution_day(
                row[col],
                ch,
                state.adstock_state.get(ch, 0.0),
                state.adstock_scale_max.get(ch, 1e-12),
            )
            state.adstock_state[ch] = adstock_new
            state.adstock_scale_max[ch] = scale_new
            media_total += contrib

        baseline = _fourier_baseline_daily(day_idx) + _q4_boost_daily(current)
        control_effect = (
            cfg.CONTROL_GAMMA["competitor_spend"] * row["competitor_spend"]
            + cfg.CONTROL_GAMMA["google_trend_competitor"]
            * row["google_trend_competitor"]
        )
        opp_noise = rng.normal(0.0, cfg.daily_opportunities_noise_std())
        opportunities = baseline + media_total + control_effect + opp_noise
        row["opportunities"] = float(max(opportunities, 5.0))

        sales_noise = rng.normal(0.0, cfg.daily_sales_noise_std())
        row["sales"] = float(
            max(row["opportunities"] * cfg.TRUE_CLOSE_RATE + sales_noise, 0.0)
        )

        rows.append(row)
        state.day_index += 1
        state.last_daily_date = current.isoformat()
        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    return df, state
