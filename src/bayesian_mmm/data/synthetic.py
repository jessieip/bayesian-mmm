"""
Synthetic MMM data pipeline: daily generation, weekly aggregation, refresh.

Generative story (funnel):
    spend → geometric adstock → scale → Hill saturation → media contribution
    opportunities = baseline + Σ media + controls + noise
    sales = opportunities × close_rate + noise
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Literal

import pandas as pd

from bayesian_mmm import config as cfg
from bayesian_mmm.data.daily import generate_daily_range
from bayesian_mmm.data.state import SimulationState, load_state, save_state
from bayesian_mmm.data.supabase_io import (
    get_supabase_client,
    last_covered_day_from_supabase,
    upsert_weekly_rows,
)
from bayesian_mmm.data.weekly_aggregate import (
    daily_to_weekly,
    filter_weekly_for_dates,
)

Mode = Literal["bootstrap", "refresh", "auto"]


@dataclass
class RefreshResult:
    """Summary returned by :func:`refresh_synthetic_data`."""

    mode: str
    daily_rows_added: int
    daily_total_rows: int
    weekly_rows_updated: int
    weekly_upserted: int
    gap_start: date | None
    gap_end: date | None
    last_covered_day: date | None
    generate_through: date
    message: str
    daily: pd.DataFrame
    weekly: pd.DataFrame


def _build_true_params() -> dict[str, Any]:
    """Static DGP parameters for posterior validation (written once on bootstrap)."""
    return {
        "rng_seed": cfg.RNG_SEED,
        "granularity": "daily",
        "media_channels": cfg.MEDIA_CHANNELS,
        "spend_columns": cfg.SPEND_COLUMNS,
        "control_columns": cfg.CONTROL_COLUMNS,
        "kpi_columns": cfg.KPI_COLUMNS,
        "channel_adstock_alpha_weekly": cfg.CHANNEL_ADSTOCK_ALPHA,
        "channel_adstock_alpha_daily": {
            ch: cfg.daily_adstock_alpha(ch) for ch in cfg.MEDIA_CHANNELS
        },
        "channel_hill_slope": cfg.CHANNEL_HILL_SLOPE,
        "channel_hill_half_sat": cfg.CHANNEL_HILL_HALF_SAT,
        "channel_beta_weekly": cfg.CHANNEL_BETA,
        "channel_beta_daily": {ch: cfg.daily_beta(ch) for ch in cfg.MEDIA_CHANNELS},
        "baseline": {
            "intercept_weekly": cfg.BASELINE_INTERCEPT,
            "intercept_daily": cfg.daily_baseline_intercept(),
            "fourier_order": cfg.BASELINE_FOURIER_ORDER,
            "fourier_coefs_weekly": cfg.BASELINE_FOURIER_COEFS,
            "fourier_coefs_daily": cfg.daily_fourier_coefs(),
            "q4_boost_weekly": cfg.Q4_BOOST,
            "q4_boost_daily": cfg.daily_q4_boost(),
        },
        "control_gamma": cfg.CONTROL_GAMMA,
        "true_close_rate": cfg.TRUE_CLOSE_RATE,
        "opportunities_noise_std_weekly": cfg.OPPORTUNITIES_NOISE_STD,
        "opportunities_noise_std_daily": cfg.daily_opportunities_noise_std(),
        "sales_noise_std_weekly": cfg.SALES_NOISE_STD,
        "sales_noise_std_daily": cfg.daily_sales_noise_std(),
        "spend_simulation": {
            "base_level_weekly": cfg.SPEND_BASE_LEVEL,
            "base_level_daily": {
                ch: cfg.daily_spend_base(ch) for ch in cfg.MEDIA_CHANNELS
            },
            "log_sigma": cfg.SPEND_LOG_SIGMA,
            "seasonal_amplitude": cfg.SPEND_SEASONAL_AMPLITUDE,
            "ar_phi": cfg.SPEND_AR_PHI,
            "pulse_prob_weekly": cfg.SPEND_PULSE_PROB,
            "pulse_prob_daily": cfg.daily_pulse_prob(),
            "pulse_scale": cfg.SPEND_PULSE_SCALE,
        },
        "competitor_spend_base_weekly": cfg.COMPETITOR_SPEND_BASE,
        "competitor_spend_base_daily": cfg.daily_competitor_spend_base(),
        "google_trend_base": cfg.GOOGLE_TREND_BASE,
        "google_trend_sigma": cfg.GOOGLE_TREND_SIGMA,
        "google_trend_ar": cfg.GOOGLE_TREND_AR,
        "weekly_aggregation": {
            "freq": "W-MON",
            "spend": "sum",
            "competitor_spend": "sum",
            "google_trend_competitor": "mean",
            "opportunities": "sum",
            "sales": "sum",
        },
        "supabase": {
            "weekly_table": cfg.SUPABASE_WEEKLY_TABLE,
            "date_is_week_start": cfg.DATE_IS_WEEK_START,
        },
    }


def _resolve_generate_through(as_of: date | None) -> date:
    """Never generate today's data — through yesterday relative to ``as_of``."""
    ref = date.today() if as_of is None else as_of
    return ref - timedelta(days=1)


def _load_daily_csv() -> pd.DataFrame:
    if not cfg.DAILY_MMM_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(cfg.DAILY_MMM_CSV, parse_dates=[cfg.DATE_COLUMN])
    return df.sort_values(cfg.DATE_COLUMN).reset_index(drop=True)


def _append_daily(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return new.sort_values(cfg.DATE_COLUMN).reset_index(drop=True)
    if new.empty:
        return existing
    combined = pd.concat([existing, new], ignore_index=True)
    combined[cfg.DATE_COLUMN] = pd.to_datetime(combined[cfg.DATE_COLUMN])
    combined = combined.drop_duplicates(subset=[cfg.DATE_COLUMN], keep="last")
    return combined.sort_values(cfg.DATE_COLUMN).reset_index(drop=True)


def _write_outputs(
    daily: pd.DataFrame, weekly: pd.DataFrame, write_true_params: bool
) -> None:
    cfg.DATA_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    daily.to_csv(cfg.DAILY_MMM_CSV, index=False)
    weekly.to_csv(cfg.WEEKLY_MMM_CSV, index=False)
    if write_true_params or not cfg.TRUE_PARAMS_JSON.exists():
        with cfg.TRUE_PARAMS_JSON.open("w", encoding="utf-8") as f:
            json.dump(_build_true_params(), f, indent=2)


def refresh_synthetic_data(
    *,
    mode: Mode = "auto",
    as_of: date | None = None,
    bootstrap_end: date | None = None,
    bootstrap_years: int | None = None,
    last_covered_day: date | None = None,
    skip_supabase: bool = False,
    seed: int | None = None,
) -> RefreshResult:
    """
    Bootstrap or incrementally refresh synthetic daily → weekly MMM data.

    Parameters
    ----------
    mode
        ``bootstrap`` | ``refresh`` | ``auto`` (bootstrap if no local daily file).
    as_of
        Reference "today" (default: calendar today). Data generated through ``as_of - 1``.
    bootstrap_end
        Last day of bootstrap window (default: generate_through).
    bootstrap_years
        Years of history for bootstrap (default: config).
    last_covered_day
        Override Supabase-derived last covered day (for testing).
    skip_supabase
        If True, do not read from or write to Supabase.
    seed
        RNG seed for fresh bootstrap state.
    """
    generate_through = _resolve_generate_through(as_of)
    years = cfg.BOOTSTRAP_YEARS if bootstrap_years is None else bootstrap_years
    existing_daily = _load_daily_csv()
    state = load_state()

    # --- decide mode ---
    resolved_mode = mode
    if mode == "auto":
        resolved_mode = "bootstrap" if existing_daily.empty else "refresh"

    gap_start: date | None = None
    gap_end: date | None = None
    last_cov: date | None = last_covered_day

    if resolved_mode == "bootstrap":
        end = bootstrap_end if bootstrap_end is not None else generate_through
        start = end - timedelta(days=int(years * cfg.DAYS_PER_YEAR))
        gap_start, gap_end = start, end
        state = SimulationState.fresh(seed=seed)
        new_daily, state = generate_daily_range(start, end, state)
        daily = new_daily
        write_true_params = True
        message = f"Bootstrap daily {start} to {end}"
    else:
        # refresh
        if last_cov is None and not skip_supabase:
            try:
                client = get_supabase_client()
                last_cov = last_covered_day_from_supabase(client)
            except Exception:
                last_cov = None

        if last_cov is None and state and state.last_daily_date:
            last_cov = date.fromisoformat(state.last_daily_date)
        elif last_cov is None and not existing_daily.empty:
            last_cov = pd.to_datetime(existing_daily[cfg.DATE_COLUMN].max()).date()

        if last_cov is None:
            raise RuntimeError(
                "Cannot refresh: no last_covered_day from Supabase, state, or daily CSV. "
                "Run bootstrap first or pass --last-covered-day."
            )

        gap_start = last_cov + timedelta(days=1)
        gap_end = generate_through

        if gap_start > gap_end:
            weekly = daily_to_weekly(existing_daily)
            return RefreshResult(
                mode="refresh",
                daily_rows_added=0,
                daily_total_rows=len(existing_daily),
                weekly_rows_updated=0,
                weekly_upserted=0,
                gap_start=gap_start,
                gap_end=gap_end,
                last_covered_day=last_cov,
                generate_through=generate_through,
                message="Already up to date",
                daily=existing_daily,
                weekly=weekly,
            )

        if state is None:
            raise RuntimeError(
                "Cannot refresh: simulation_state.json missing. Re-run bootstrap."
            )

        new_daily, state = generate_daily_range(gap_start, gap_end, state)
        daily = _append_daily(existing_daily, new_daily)
        write_true_params = False
        message = f"Refresh daily {gap_start} to {gap_end}"

    save_state(state)
    weekly = daily_to_weekly(daily)
    _write_outputs(daily, weekly, write_true_params=write_true_params)

    affected_weekly = (
        filter_weekly_for_dates(weekly, new_daily[cfg.DATE_COLUMN])
        if resolved_mode == "refresh"
        else weekly
    )

    upserted = 0
    if not skip_supabase:
        try:
            client = get_supabase_client()
            if resolved_mode == "bootstrap":
                upserted = upsert_weekly_rows(client, weekly)
            else:
                upserted = upsert_weekly_rows(client, affected_weekly)
        except Exception as exc:
            message += f" (Supabase upsert skipped: {exc})"

    new_rows = len(new_daily) if resolved_mode == "bootstrap" else len(new_daily)
    return RefreshResult(
        mode=resolved_mode,
        daily_rows_added=new_rows,
        daily_total_rows=len(daily),
        weekly_rows_updated=len(affected_weekly)
        if resolved_mode == "refresh"
        else len(weekly),
        weekly_upserted=upserted,
        gap_start=gap_start,
        gap_end=gap_end,
        last_covered_day=last_cov,
        generate_through=generate_through,
        message=message,
        daily=daily,
        weekly=weekly,
    )


# ---------------------------------------------------------------------------
# Backward-compatible weekly one-shot API (deprecated)
# ---------------------------------------------------------------------------


def generate_synthetic_dataset(seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Legacy: bootstrap daily data and return weekly aggregated panel.

    Prefer :func:`refresh_synthetic_data` for incremental updates.
    """
    result = refresh_synthetic_data(mode="bootstrap", seed=seed, skip_supabase=True)
    true_params = _build_true_params()
    true_params["seed_used"] = cfg.RNG_SEED if seed is None else seed
    return result.weekly, true_params


def save_synthetic_dataset(seed: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Legacy: bootstrap and write CSV + JSON (no Supabase)."""
    weekly, true_params = generate_synthetic_dataset(seed=seed)
    return weekly, true_params
