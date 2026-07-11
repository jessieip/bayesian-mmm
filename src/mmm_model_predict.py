"""MMM posterior predictive checks, decomposition, and sensitivity plots."""

from __future__ import annotations

import logging
import warnings
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pymc_marketing.mmm.multidimensional import MMM

logger = logging.getLogger(__name__)

DEFAULT_HDI_PROB: float = 0.94
DEFAULT_SWEEP_VALUES: np.ndarray = np.linspace(0, 1.5, 12)
DEFAULT_DECOMPOSITION_SAVE_PATH: str = "mmm_contributions.png"
POSTERIOR_REF_VAL: float = 0.3

DECOMPOSITION_VARS: list[str] = [
    "channel_contribution_original_scale",
    "control_contribution_original_scale",
    "intercept_contribution_original_scale",
    "yearly_seasonality_contribution_original_scale",
]

RESULT_KEYS: list[str] = [
    "fig_posterior_predictive",
    "fig_decomposition",
    "fig_waterfall",
    "fig_adstock",
    "fig_saturation",
    "fig_sensitivity",
    "df_mean_contribution",
]


def mmm_model_predict(
    mmm: MMM,
    x: pd.DataFrame,
    df: pd.DataFrame,
    *,
    random_seed: int = 42,
    sweep_values: np.ndarray | None = None,
    decomposition_save_path: str | None = DEFAULT_DECOMPOSITION_SAVE_PATH,
) -> dict[str, plt.Figure | pd.DataFrame]:
    """
    Execute post model sampling and generate validation charts.
    - Plotting the predicted sales and observed sales.
    - Plotting the waterfall decomposition chart to understand the channels contribution and plot sensitivity analysis.
    Parameters:
        mmm: trained MMM
        x: DataFrame is used for prediction
        df: raw dataset including sales or opportunities.
        random_seed: seed forwarded to sample_posterior_predictive
        sweep_values: values for sensitivity sweep (defaults to linspace 0-1.5)
        decomposition_save_path: path to save decomposition figure; None skips save
    Returns:
        dict of figures and mean contribution DataFrame
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    plt.ioff()

    rng = np.random.default_rng(random_seed)
    logger.info("Sampling posterior predictive...")
    mmm.sample_posterior_predictive(X=x, random_seed=rng)

    logger.info("Plotting Posterior Predictive Charts...")

    fig_pp, axes_pp = mmm.plot.posterior_predictive(
        var=["y_original_scale"], hdi_prob=DEFAULT_HDI_PROB
    )
    target_col = mmm.target_column
    sns.lineplot(
        data=df,
        x=mmm.date_column,
        y=target_col,
        color="black",
        label=f"Observed ({target_col}",
        ax=axes_pp.flatten()[0],
    )

    fig_pp.suptitle("Posterior Predictive Check", fontsize=14, fontweight="bold")

    logger.info("Plotting Contribution Over Time Charts...")

    mmm.plot.contributions_over_time(
        var=["channel_contribution"], hdi_prob=DEFAULT_HDI_PROB
    )

    mmm.plot.contributions_over_time(
        var=["channel_contribution_original_scale"], hdi_prob=DEFAULT_HDI_PROB
    )

    fig_decomp, axes_decomp = mmm.plot.contributions_over_time(
        var=DECOMPOSITION_VARS,
        dims={"channel": mmm.channel_columns},
        hdi_prob=DEFAULT_HDI_PROB,
    )

    for ax in axes_decomp.flatten():
        legend = ax.get_legend()
        if legend is not None:
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize="small")

    if decomposition_save_path is not None:
        fig_decomp.savefig(decomposition_save_path, bbox_inches="tight")

    logger.info("Plotting waterfall and contribution chart...")

    mmm_waterfall, _ = mmm.plot.waterfall_components_decomposition()
    mmm_contribution = mmm.compute_mean_contributions_over_time()

    logger.info("Channels Adstock lagging effect:")
    alpha_var = (
        "adstock_alpha"
        if "adstock_alpha" in mmm.idata["posterior"].data_vars
        else "adstock_alpha"
    )

    az.plot_posterior(
        mmm.idata["posterior"],
        var_names=[alpha_var],
        ref_val={
            alpha_var: [
                {"channel": ch, "ref_val": POSTERIOR_REF_VAL} for ch in mmm.channel_columns
            ]
        },
    )

    fig_alpha = plt.gcf()
    fig_alpha.suptitle("Adstock Alpha Posterior", fontsize=16, fontweight="bold")

    logger.info("Predicted Channel Saturation Level...")

    slope_var = (
        "saturation_slope"
        if "saturation_slope" in mmm.idata["posterior"].data_vars
        else "saturation_slope"
    )

    az.plot_posterior(
        mmm.idata["posterior"],
        var_names=[slope_var],
        ref_val={
            slope_var: [
                {"channel": ch, "ref_val": POSTERIOR_REF_VAL} for ch in mmm.channel_columns
            ]
        },
    )
    fig_slope = plt.gcf()
    fig_slope.suptitle("Saturation Slope Posterior Distribution", fontsize=16, fontweight="bold")

    logger.info("Running and plotting sensitivity analysis sweep...")
    sweeps = sweep_values if sweep_values is not None else DEFAULT_SWEEP_VALUES
    mmm.sensitivity.run_sweep(
        sweep_values=sweeps,
        var_input="channel_data",
        var_names="channel_contribution_original_scale",
        extend_idata=True,
    )

    ax_sensitivity = mmm.plot.sensitivity_analysis(
        xlabel="Sweep multiplicative",
        ylabel="Total contribution over training period(Original Scale)",
        hue_dim="channel",
        x_sweep_axis="relative",
    )
    ax_sensitivity.axvline(1.0, color="black", linestyle="--", linewidth=1)
    fig_sensitivity = plt.gcf()
    plt.ion()
    logger.info("MMM Prediction and Analysis plotting completed.")

    return {
        "fig_posterior_predictive": fig_pp,
        "fig_decomposition": fig_decomp,
        "fig_waterfall": mmm_waterfall,
        "fig_adstock": fig_alpha,
        "fig_saturation": fig_slope,
        "fig_sensitivity": fig_sensitivity,
        "df_mean_contribution": mmm_contribution,
    }
