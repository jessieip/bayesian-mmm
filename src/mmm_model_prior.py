"""MMM prior setup and prior predictive checks."""

from __future__ import annotations

import logging
import os
from typing import Any, Type
import xarray as xr

import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pymc_marketing.mmm import GeometricAdstock, HillSaturation
from pymc_marketing.mmm.multidimensional import MMM
from pymc_marketing.prior import Prior

logger = logging.getLogger(__name__)

CHANNEL_COLUMNS: list[str] = [
    "PPC_Brand_Spend",
    "PPC_Generic_Spend",
    "Display_Spend",
    "Social_Spend",
    "TV_Spend",
    "OOH_Spend",
    "Meta_Spend",
    "Yahoo_Spend",
]

CONTROL_COLUMNS: list[str] = ["competitor_spend", "google_trend_competitor"]

N_CHANNELS: int = len(CHANNEL_COLUMNS)

ADSTOCK_ALPHA_LIST: list[int] = [1, 1, 1, 1, 3, 3, 1, 1]
ADSTOCK_BETA_LIST: list[int] = [3, 3, 3, 3, 1, 1, 3, 3]

alpha_da = xr.DataArray(ADSTOCK_ALPHA_LIST, coords={'channel':CHANNEL_COLUMNS}, dims='channel')
beta_da = xr.DataArray(ADSTOCK_BETA_LIST, coords={'channel':CHANNEL_COLUMNS}, dims='channel')

DEFAULT_SAMPLER_CONFIG: dict[str, Any] = {
    "progressbar": True,
    "draws": 1000,
    "chains": 4,
    "nuts_sampler": "numpyro",
}


def _ensure_mingw_on_path(mingw_path: str = r"C:\msys64\mingw64\bin") -> None:
    """Append MinGW bin to PATH on Windows when present (JAX/numpyro)."""
    if os.path.exists(mingw_path) and mingw_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] += os.pathsep + mingw_path
        logger.debug("Added MinGW Path: %s", mingw_path)


def _validate_prior_sigma(prior_sigma: list[float], n_channels: int = N_CHANNELS) -> None:
    if len(prior_sigma) != n_channels:
        raise ValueError(
            f"Length of prior sigma is {len(prior_sigma)}. "
            f"Number of Marketing Channels is {n_channels}."
        )


def _build_model_config(prior_sigma: list[float]) -> dict[str, Prior]:
    n_channels = len(CHANNEL_COLUMNS)
    if len(ADSTOCK_ALPHA_LIST) != n_channels or len(ADSTOCK_BETA_LIST) != n_channels:
        raise ValueError(
            "The Length of Alpha List or Beta List unmatched with "
            "the number of marketing channels."
        )

    sigma_da = xr.DataArray(prior_sigma, coords={'channel': CHANNEL_COLUMNS}, dims='channel')

    return {
        "intercept": Prior("Normal", mu=0.5, sigma=2),
        "saturation_beta": Prior("HalfNormal", sigma=sigma_da, dims="channel"),
        "gamma_control": Prior("Normal", mu=0, sigma=0.5, dims="control"),
        "gamma_fourier": Prior("Laplace", mu=0, b=0.5, dims="fourier_mode"),
        "adstock_alpha": Prior(
            "Beta", alpha=alpha_da, beta=beta_da
        ),
        "saturation_slope": Prior("Gamma", alpha=3, beta=1, dims="channel"),
        "likelihood": Prior("Normal", sigma=Prior("HalfNormal", sigma=6)),
    }


def mmm_model_prior(
    dataset: pd.DataFrame,
    prior_sigma: list[float],
    *,
    prior_samples: int = 2000,
    mmm_cls: Type[MMM] = MMM,
) -> tuple[pd.DataFrame, pd.Series, MMM, plt.Figure]:
    """
    Set up MMM priors, run prior predictive sampling, and plot checks.

    Parameters
    ----------
    dataset
        Weekly panel from Supabase (spend, date, opportunities, controls).
    prior_sigma
        HalfNormal sigma per channel for ``saturation_beta`` (length must equal 8).
    prior_samples
        Number of prior predictive draws (default 2000).
    mmm_cls
        MMM class constructor (injectable for tests).

    Returns
    -------
    X, y, mmm, fig
        Feature matrix, target series, fitted MMM builder, prior predictive figure.
    """
    _ensure_mingw_on_path()
    _validate_prior_sigma(prior_sigma)

    y = dataset["opportunities"].copy()
    y.name = "y"

    required_cols = ["date"] + CHANNEL_COLUMNS + CONTROL_COLUMNS
    X = dataset[required_cols].copy()

    my_model_config = _build_model_config(prior_sigma)

    mmm = mmm_cls(
        model_config=my_model_config,
        sampler_config=DEFAULT_SAMPLER_CONFIG,
        date_column="date",
        target_column="y",
        adstock=GeometricAdstock(l_max=8),
        saturation=HillSaturation(),
        channel_columns=CHANNEL_COLUMNS,
        control_columns=CONTROL_COLUMNS,
        yearly_seasonality=2,
    )

    logger.info("Building PyMC MMM Graph...")
    mmm.build_model(X, y)

    mmm.add_original_scale_contribution_variable(
        var=[
            "channel_contribution",
            "control_contribution",
            "intercept_contribution",
            "yearly_seasonality_contribution",
            "y",
        ]
    )

    logger.info("Sampling prior predictive (%s samples)...", prior_samples)
    mmm.sample_prior_predictive(X, y, samples=prior_samples)

    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, hdi_prob in enumerate([0.94, 0.5]):
        az.plot_hdi(
            x=mmm.model.coords["date"],
            y=mmm.idata.prior["y_original_scale"],
            smooth=False,
            color="C0",
            hdi_prob=hdi_prob,
            fill_kwargs={"alpha": 0.3 + i * 0.1, "label": f"{hdi_prob: .0%} HDI"},
            ax=ax,
        )

    sns.lineplot(
        data=dataset,
        x="date",
        y="opportunities",
        color="black",
        label="observed(opportunities)",
        ax=ax,
    )

    ax.legend(loc="upper left")
    ax.set(xlabel="Date", ylabel="Opportunities")
    ax.set_title("Prior Predictive Checks", fontsize=18, fontweight="bold")
    plt.ion()

    logger.info("MMM Priors Initialisation and Plotting completed.")
    return X, y, mmm, fig
