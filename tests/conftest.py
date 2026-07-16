"""Shared pytest fixtures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import matplotlib.pyplot as plt

from mmm_model_prior import CHANNEL_COLUMNS, CONTROL_COLUMNS, N_CHANNELS


def _base_row(date: str, spend_multiplier: float = 1.0) -> dict:
    return {
        "date": date,
        "PPC_Brand_Spend": 1000.0 * spend_multiplier,
        "PPC_Generic_Spend": 800.0 * spend_multiplier,
        "Display_Spend": 600.0 * spend_multiplier,
        "Social_Spend": 400.0 * spend_multiplier,
        "TV_Spend": 2000.0 * spend_multiplier,
        "OOH_Spend": 500.0 * spend_multiplier,
        "Meta_Spend": 450.0 * spend_multiplier,
        "Yahoo_Spend": 150.0 * spend_multiplier,
        "competitor_spend": 3000.0 * spend_multiplier,
        "google_trend_competitor": 50.0,
        "opportunities": 100.0 * spend_multiplier,
        "sales": 12.0 * spend_multiplier,
    }

@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")



@pytest.fixture
def sample_weekly_rows() -> list[dict]:
    """Two rows out of chronological order for sort assertions."""
    return [
        _base_row("2026-06-08", spend_multiplier=1.2),
        _base_row("2026-06-01", spend_multiplier=1.0),
    ]


@pytest.fixture
def mock_supabase_client(sample_weekly_rows: list[dict]) -> MagicMock:
    """Chainable mock: client.table().select().execute()."""
    client = MagicMock()
    client.table.return_value.select.return_value.execute.return_value = MagicMock(
        data=sample_weekly_rows
    )
    return client


def build_mock_client(data: list[dict]) -> MagicMock:
    """Build a Supabase client mock returning the given response data."""
    client = MagicMock()
    client.table.return_value.select.return_value.execute.return_value = MagicMock(
        data=data
    )
    return client


@pytest.fixture
def sample_mmm_dataframe() -> pd.DataFrame:
    """Minimal weekly panel matching Supabase schema (2 rows, 4 channels)."""
    return pd.DataFrame(
        {
            "date": ["2026-06-01", "2026-06-08"],
            "PPC_Brand_Spend": [1000.0, 1000.0],
            "Display_Spend": [500.0, 500.0],
            "Meta_Spend": [300.0, 300.0],
            "Yahoo_Spend": [200.0, 200.0],
            "competitor_spend": [9999.0, 9999.0],
            "opportunities": [50.0, 60.0],
            "sales": [6.0, 7.0],
        }
    )


@pytest.fixture
def zero_spend_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PPC_Brand_Spend": [0.0, 0.0],
            "Display_Spend": [0.0, 0.0],
        }
    )


@pytest.fixture
def no_spend_dataframe() -> pd.DataFrame:
    return pd.DataFrame({"date": ["2026-06-01"], "sales": [10.0]})


@pytest.fixture
def full_mmm_dataset() -> pd.DataFrame:
    """Weekly panel with all 8 channels + controls (4 rows for plotting)."""
    rows = [
        _base_row("2026-01-06", 1.0),
        _base_row("2026-01-13", 1.05),
        _base_row("2026-01-20", 0.95),
        _base_row("2026-01-27", 1.1),
    ]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


@pytest.fixture
def valid_prior_sigma() -> list[float]:
    return [1.0] * N_CHANNELS


def build_mock_mmm(n_dates: int, *, n_chains: int = 2, n_draws: int = 5) -> MagicMock:
    """Minimal MMM mock for prior predictive plotting tests."""
    dates = pd.date_range("2026-01-06", periods=n_dates, freq="W-MON")
    y_prior = xr.DataArray(
        np.ones((n_chains, n_draws, n_dates)) * 100.0,
        dims=["chain", "draw", "date"],
        coords={"chain": range(n_chains), "draw": range(n_draws), "date": dates},
    )

    mock_mmm = MagicMock()
    mock_mmm.model.coords = {"date": dates}
    mock_mmm.idata.prior = {"y_original_scale": y_prior}
    return mock_mmm
