"""MMM training, diagnostics, and trace plotting."""

from __future__ import annotations

import logging
from typing import Any

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pymc_marketing.mmm.multidimensional import MMM

logger = logging.getLogger(__name__)

TARGET_SUMMARY_VARS: list[str] = [
    "adstock_alpha",
    "gamma_control",
    "gamma_fourier",
    "intercept",
    "saturation_beta",
    "saturation_slope",
    "sigma",
]

DEFAULT_FIT_KWARGS: dict[str, Any] = {
    "chains": 4,
    "draws": 1000,
    "target_accept": 0.8,
}


def mmm_model_train(
    x: pd.DataFrame,
    y: pd.Series,
    mmm: MMM,
    *,
    fit_kwargs: dict[str, Any] | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Model Training and plotting the tracing chart, has the model found the consistent ROI range via various chains?

    Arguments:
        x: the independent variables predicted the opportunities.
        y: the dependent variable that will be predicted via x.
        mmm: an instance of MMM
        fit_kwargs: optional overrides merged into default sampler settings
        random_seed: seed forwarded to mmm.fit as random_seed
    Returns:
        summary_df: the summary of saturation, lagging effect, noise, etc.
        trace_graph: matplotlib figure of parameter traces
    """
    rng = np.random.default_rng(random_seed)
    sampler_kwargs = {**DEFAULT_FIT_KWARGS, **(fit_kwargs or {})}

    logger.info(
        "Training MMM started (%s chains, %s draws)",
        sampler_kwargs["chains"],
        sampler_kwargs["draws"],
    )

    mmm.fit(
        X=x,
        y=y,
        random_seed=rng,
        **sampler_kwargs,
    )

    logger.info("training completed")
    logger.info(f"Model was trained using the {mmm.saturation.__class__.__name__} function")
    logger.info(f"and the {mmm.adstock.__class__.__name__} function")

    try:
        diverging_count = int(mmm.fit_result["sample_stats"]["diverging"].sum().item())
        if diverging_count > 0:
            logger.warning(f"Model Diagnostics: Found {diverging_count} diverging transitions!")
        else:
            logger.info("Model Diagnostics: 0 diverging transitions. Sampling is stable")
    except (AttributeError, KeyError) as e:
        logger.error(f"Failed to extract sample stats: {e}")
        diverging_count = 0

    available_var = [s for s in TARGET_SUMMARY_VARS if s in mmm.fit_result.data_vars]

    summary_df = az.summary(
        data=mmm.fit_result,
        var_names=available_var,
    )

    plt.ioff()

    az.plot_trace(
        data=mmm.fit_result,
        var_names=available_var,
        compact=True,
        backend_kwargs={"figsize": (12, 10), "layout": "constrained"},
    )

    trace_graph = plt.gcf()
    trace_graph.suptitle("Model Trace", fontsize=16, fontweight="bold")

    plt.ion()

    logger.info("Model Trace plot generated.")
    return summary_df, trace_graph
