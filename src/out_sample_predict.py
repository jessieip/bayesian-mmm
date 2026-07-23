### Out of Sample Predictions
from __future__ import annotations

import logging
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from pymc_marketing.mmm.multidimensional import MMM

logger = logging.getLogger(__name__)
DEFAULT_N_POINTS: int = 15
DEFAULT_N_NEW: int = 5
DEFAULT_HDI_PROB: float = 0.94


def build_x_out_of_sample(
    x: pd.DataFrame,
    mmm: MMM,
    n_new: int = DEFAULT_N_NEW,
) -> pd.DataFrame:
    """Build future feature rows from the last observed date in ``x``."""
    dates = pd.to_datetime(x["date"])
    last_date = dates.max()

    # Continue the observed cadence when possible; fall back to weekly Monday.
    if len(dates) >= 2:
        step = dates.sort_values().diff().median()
        new_dates = pd.date_range(start=last_date + step, periods=n_new, freq=step)
    else:
        new_dates = pd.date_range(start=last_date, periods=1 + n_new, freq="W-MON")[1:]

    x_out_of_sample = pd.DataFrame({"date": new_dates})
    last_row = x.loc[x["date"].idxmax()]

    logger.info("Generating x_out_of_sample, y_out_of_sample...")
    channels = list(mmm.channel_columns)
    for ch in channels:
        x_out_of_sample[ch] = last_row[ch]

    logger.info("Generating x_out_of_sample control variables...")
    controls = getattr(mmm, "control_columns", None) or []

    for ctrl in controls:
        if ctrl in x.columns:
            x_out_of_sample[ctrl] = last_row[ctrl]
        else:
            logger.warning(f"Control variable '{ctrl}' is not found in x input")

    return x_out_of_sample


def build_x_for_prediction(
    x: pd.DataFrame,
    x_out_of_sample: pd.DataFrame,
    mmm: MMM,
) -> pd.DataFrame:
    """Prefix future rows with recent history so adstock has lag context.

    ``include_last_observations=True`` currently returns an empty ``date`` dim for
    multidimensional MMM, so we prepend up to ``adstock.l_max`` historical rows
    and sample with ``include_last_observations=False`` instead.
    """
    l_max_raw = getattr(getattr(mmm, "adstock", None), "l_max", 0)
    try:
        l_max = int(l_max_raw or 0)
    except (TypeError, ValueError):
        l_max = 0
    if l_max <= 0 or x.empty:
        return x_out_of_sample.reset_index(drop=True)

    cols = list(x_out_of_sample.columns)
    history = x.loc[:, cols].iloc[-min(l_max, len(x)) :].copy()
    return pd.concat([history, x_out_of_sample], ignore_index=True)


def y_original_scale_from_predictive(y_out_of_sample: Any) -> xr.DataArray:
    """Extract ``y_original_scale`` from a posterior-predictive return value.

    Matches the notebook contract:
    ``y_out["y_original_scale"].unstack().transpose(..., "date")``.
    """
    if "y_original_scale" in getattr(y_out_of_sample, "data_vars", y_out_of_sample):
        da = y_out_of_sample["y_original_scale"]
    elif hasattr(y_out_of_sample, "posterior_predictive"):
        da = y_out_of_sample.posterior_predictive["y_original_scale"]
    elif (
        isinstance(y_out_of_sample, dict)
        and "posterior_predictive" in y_out_of_sample
        and "y_original_scale" in y_out_of_sample["posterior_predictive"]
    ):
        da = y_out_of_sample["posterior_predictive"]["y_original_scale"]
    else:
        raise KeyError("Cannot identify y_original_scale in predictive output")

    if "sample" in da.dims and "date" not in da.dims:
        da = da.rename({"sample": "date"})

    return da.unstack().transpose(..., "date")


def _forecast_y_scale(
    x_out_of_sample: pd.DataFrame,
    y_out_original_scale: xr.DataArray,
) -> tuple[pd.DatetimeIndex, xr.DataArray]:
    """Restrict predictive scale to dates overlapping ``x_out_of_sample``."""
    forecast_dates = pd.DatetimeIndex(pd.to_datetime(x_out_of_sample["date"]))
    y_scale = y_out_original_scale
    if "date" not in y_scale.dims:
        raise ValueError("Predictive y_original_scale must include a 'date' dimension")

    y_scale = y_scale.assign_coords(date=pd.to_datetime(y_scale.coords["date"].values))
    overlap = pd.DatetimeIndex(y_scale.coords["date"].values).intersection(
        forecast_dates
    )
    if len(overlap) == 0:
        raise ValueError(
            "No overlapping dates between predictive output and "
            f"x_out_of_sample. predictive={list(y_scale.coords['date'].values)} "
            f"forecast={list(forecast_dates)}"
        )
    return overlap, y_scale.sel(date=overlap)


def summarize_out_of_sample(
    x_out_of_sample: pd.DataFrame,
    y_out_of_sample: Any,
    *,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> pd.DataFrame:
    """
    Build a table of forecast mean and HDI for Supabase ``predictions``.

    Returns a DataFrame with columns ``date``, ``mean``, ``hdi_lower``,
    ``hdi_upper``. HDI uses ArviZ's default-style interval (``hdi_prob=0.94``
    unless overridden).
    """
    y_out_original_scale = y_original_scale_from_predictive(y_out_of_sample)
    forecast_dates, y_scale = _forecast_y_scale(x_out_of_sample, y_out_original_scale)

    mean = y_scale.mean(dim=("chain", "draw"))
    hdi_ds = az.hdi(y_scale, hdi_prob=hdi_prob)

    # az.hdi returns a Dataset; pick the sole data variable when needed.
    if isinstance(hdi_ds, xr.Dataset):
        hdi_da = next(iter(hdi_ds.data_vars.values()))
    else:
        hdi_da = hdi_ds

    if "hdi" in hdi_da.dims:
        hdi_lower = hdi_da.sel(hdi="lower")
        hdi_upper = hdi_da.sel(hdi="higher")
    else:
        # Fallback when HDI is returned as (..., 2) along the last axis.
        hdi_lower = hdi_da.isel({hdi_da.dims[-1]: 0})
        hdi_upper = hdi_da.isel({hdi_da.dims[-1]: -1})

    rows = []
    for date in forecast_dates:
        rows.append(
            {
                "date": pd.Timestamp(date).to_pydatetime().replace(tzinfo=None),
                "mean": float(mean.sel(date=date).values),
                "hdi_lower": float(hdi_lower.sel(date=date).values),
                "hdi_upper": float(hdi_upper.sel(date=date).values),
            }
        )
    return pd.DataFrame(rows)


def plot_out_of_sample(
    x: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    x_out_of_sample: pd.DataFrame,
    y_out_original_scale: xr.DataArray,
    n_points: int = DEFAULT_N_POINTS,
) -> plt.Figure:
    """Plot historical actuals with forecast mean and HDI band."""
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")

    plot_x = pd.to_datetime(x["date"].iloc[-n_points:]).reset_index(drop=True)
    plot_y = y.iloc[-n_points:].reset_index(drop=True)

    if isinstance(plot_y, pd.DataFrame):
        plot_y = plot_y.iloc[:, 0]

    logger.info("Plotting historical actuals...")

    sns.lineplot(
        x=plot_x,
        y=plot_y,
        marker="o",
        markersize=7,
        color="blue",
        label="actuals",
        sort=False,
        ax=ax,
    )

    logger.info("Plotting forecasted value line chart...")

    forecast_dates, y_scale = _forecast_y_scale(x_out_of_sample, y_out_original_scale)

    y_for_hdi = (
        y_scale.stack(sample=("chain", "draw")).transpose("sample", "date").values
    )
    az.plot_hdi(
        forecast_dates.to_numpy(),
        y_scale,
        # y_for_hdi,
        smooth=False,
        fill_kwargs={"alpha": 0.25, "color": "C0"},
        ax=ax,
    )

    mean = y_scale.mean(dim=("chain", "draw"))
    mean.plot(
        ax=ax, marker="o", markersize=7, label="Forecast", color="C0", linestyle="--"
    )

    ax.set(ylabel="Original Target Scale (Opportunities")
    ax.set_title("Out of sample predictions for MMM", fontsize=18, fontweight="bold")
    ax.legend(loc="upper left")

    return fig


def out_sample_predict(
    x: pd.DataFrame,
    y: pd.DataFrame | pd.Series,
    mmm: MMM,
    n_points: int = DEFAULT_N_POINTS,
    n_new: int = DEFAULT_N_NEW,
    random_seed: int | np.random.Generator | None = 42,
) -> tuple[pd.DataFrame, Any, plt.Figure]:
    """
    Out of sample prediction for a trained MMM.

    Predicts the next ``n_new`` weeks of opportunities and plots them against
    the last ``n_points`` historical actuals.

    Args:
        x: Independent variables from the original dataframe (must include ``date``).
        y: Dependent variable (Series or single-column DataFrame).
        mmm: Trained MMM.
        n_points: Number of historical points shown on the plot.
        n_new: Number of future periods to forecast.
        random_seed: Seed or Generator forwarded to ``sample_posterior_predictive``.

    Returns:
        ``(x_out_of_sample, y_out_of_sample, fig)`` where ``y_out_of_sample`` is
        the posterior-predictive return value (typically an xarray Dataset).
    """
    plt.ioff()

    x_out_of_sample = build_x_out_of_sample(x, mmm, n_new=n_new)
    x_for_prediction = build_x_for_prediction(x, x_out_of_sample, mmm)

    logger.info(f"Sampling out-of-sample predictive for the next {n_new} periods...")

    seed = (
        np.random.default_rng(random_seed)
        if isinstance(random_seed, int)
        else random_seed
    )

    # include_last_observations=True returns an empty date dim on multidimensional
    # MMM; prepend history via build_x_for_prediction instead.
    y_out_of_sample = mmm.sample_posterior_predictive(
        X=x_for_prediction,
        extend_idata=False,
        include_last_observations=False,
        random_seed=seed,
    )

    y_out_original_scale = y_original_scale_from_predictive(y_out_of_sample)
    fig = plot_out_of_sample(
        x, y, x_out_of_sample, y_out_original_scale, n_points=n_points
    )

    plt.ion()

    logger.info("Plotting Out of Sample Line chart completed.")

    return x_out_of_sample, y_out_of_sample, fig
