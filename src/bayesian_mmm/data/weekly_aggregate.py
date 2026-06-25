"""Aggregate daily MMM panel to weekly (Monday week-start)."""

from __future__ import annotations

from datetime import date

import pandas as pd

from bayesian_mmm import config as cfg


def _weekly_agg_dict() -> dict[str, str]:
    return {
        **{col: "sum" for col in cfg.SPEND_COLUMNS},
        "competitor_spend": "sum",
        "google_trend_competitor": "mean",
        "opportunities": "sum",
        "sales": "sum",
    }


def daily_to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    """
    Roll daily panel up to weeks starting Monday (Mon–Sun).

    Spend / KPI columns are summed; ``google_trend_competitor`` is averaged.
    Weekly ``date`` is the Monday that starts each week.
    """
    if daily.empty:
        return pd.DataFrame(
            columns=[
                cfg.DATE_COLUMN,
                *cfg.SPEND_COLUMNS,
                *cfg.CONTROL_COLUMNS,
                *cfg.KPI_COLUMNS,
            ]
        )

    df = daily.copy()
    df[cfg.DATE_COLUMN] = pd.to_datetime(df[cfg.DATE_COLUMN])
    df["_week_start"] = _monday_week_start(df[cfg.DATE_COLUMN])

    weekly = (
        df.groupby("_week_start", as_index=False)
        .agg(_weekly_agg_dict())
        .rename(columns={"_week_start": cfg.DATE_COLUMN})
    )
    return weekly


def _monday_week_start(dates: pd.Series) -> pd.Series:
    """Monday 00:00 of the ISO week containing each date (matches ``Grouper(freq='W-MON')``)."""
    dates = pd.to_datetime(dates).dt.normalize()
    return dates - pd.to_timedelta(dates.dt.weekday, unit="D")


def weeks_affected_by_dates(daily_dates: pd.Series) -> pd.DatetimeIndex:
    """Return unique week-start Mondays touched by the given daily dates."""
    week_starts = _monday_week_start(daily_dates)
    return pd.DatetimeIndex(sorted(week_starts.unique()))


def filter_weekly_for_dates(
    weekly: pd.DataFrame, daily_dates: pd.Series
) -> pd.DataFrame:
    """Subset weekly rows whose week-start is affected by ``daily_dates``."""
    if weekly.empty or daily_dates.empty:
        return weekly.iloc[0:0]

    affected = weeks_affected_by_dates(daily_dates)
    wk = weekly.copy()
    wk[cfg.DATE_COLUMN] = pd.to_datetime(wk[cfg.DATE_COLUMN]).dt.normalize()
    return wk[wk[cfg.DATE_COLUMN].isin(affected)].reset_index(drop=True)


def week_start_for_date(d: date) -> pd.Timestamp:
    """Monday of the ISO week containing ``d``."""
    ts = pd.Timestamp(d).normalize()
    return ts - pd.Timedelta(days=ts.weekday())
