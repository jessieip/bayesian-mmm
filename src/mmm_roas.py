"""Channel ROAS posterior distributions with HDI overlays."""

from __future__ import annotations

import logging

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from pymc_marketing.mmm.multidimensional import MMM

DEFAULT_HDI_PROB: float = 0.94
DEFAULT_ROAS_SAVE_PATH: str = "mmm_roas.png"

RESULT_KEYS: list[str] = ["fig_roas"]

logger = logging.getLogger(__name__)


def mmm_roas(
    mmm: MMM,
    roas_save_path: str | None = DEFAULT_ROAS_SAVE_PATH,
    *,
    hdi_prob: float = DEFAULT_HDI_PROB,
) -> dict[str, plt.Figure]:
    """
    Calculate Channel ROAS and plotting posterior charts with HDI.

    Args:
        mmm: trained MMM
        roas_save_path: charts save path
        hdi_prob: highest density interval probability for HDI overlays

    Returns:
        dict: a dictionary includes channel ROAS charts.

    """
    plt.ioff()

    logger.info("Computing Channel ROAS graphs(Contribution over spend)...")

    roas = mmm.incrementality.contribution_over_spend(frequency="all_time").rename("roas")

    # calculate the number of channels and nrows
    channels = list(mmm.channel_columns)
    num_channels = len(channels)

    n_cols = 2
    n_rows = int(np.ceil(num_channels / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(14, 3 * n_rows), layout="constrained"
    )
    axes_flat = np.atleast_1d(axes).flatten()
    hdi_pct = int(round(hdi_prob * 100))

    logger.info(f"Plotting ROAS graphs with {num_channels} channels...")
    for i, channel in enumerate(channels):
        if i >= len(axes_flat):
            break

        channel_roas = roas.sel(channel=channel)

        az.plot_posterior(
            channel_roas, ax=axes_flat[i], hdi_prob=None, color="green", round_to=2
        )

        hdi_ds = az.hdi(channel_roas, hdi_prob=hdi_prob)
        hdi_low = float(hdi_ds.to_array().values.flatten()[0])
        hdi_high = float(hdi_ds.to_array().values.flatten()[-1])

        axes_flat[i].hlines(
            y=0,
            xmin=hdi_low,
            xmax=hdi_high,
            color="red",
            linewidth=3,
            label=f"{hdi_pct}% HDI",
        )

        axes_flat[i].text(hdi_low, 0.05, f"{hdi_low:.2f}", color="red", ha="center", weight="bold")
        axes_flat[i].text(hdi_high, 0.05, f"{hdi_high:.2f}", color="red", ha="center", weight="bold")

        title_ = channel.replace("_Spend", "") + " ROAS"
        axes_flat[i].set(title=title_)

        if i >= (n_rows - 1) * n_cols:
            axes_flat[i].set(xlabel="ROAS")
    # hide extra blank axes. e.g. if there's 7 channels, the 8th axes will be hidden
    for j in range(num_channels, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    fig.suptitle(
        f"ROAS Posterior Distributions with {hdi_pct}% HDI",
        fontsize=18,
        fontweight="bold",
        y=1.06,
    )

    if roas_save_path is not None:
        fig.savefig(roas_save_path, bbox_inches="tight")
        logger.info(f"ROAS charts completed and saved to {roas_save_path}")

    plt.ion()

    return {"fig_roas": fig}
