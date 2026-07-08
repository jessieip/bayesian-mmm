"""Data processing for MMM prior configuration."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

EXCLUDED_SPEND_COLS = {"competitor_spend"}


def _get_spend_columns(dataset: pd.DataFrame) -> list[str]:
    """Return marketing spend columns, excluding competitor spend."""
    return [
        col
        for col in dataset.columns
        if col.lower().endswith("_spend") and col.lower() not in EXCLUDED_SPEND_COLS
    ]


def data_process(dataset: pd.DataFrame) -> list[float]:
    """
    Calculate Beta prior sigma per channel from historical spend share.

    Args:
     Weekly MMM data extracted from Supabase.

    Returns:
    list[float]: Prior sigma per channel, in DataFrame column order of spend columns.
    """
    spend_cols = _get_spend_columns(dataset)
    if not spend_cols:
        raise ValueError("No available marketing spend column in the dataset.")

    spend_per_channel = dataset[spend_cols].sum(axis=0)
    total = spend_per_channel.sum()
    if total == 0:
        raise ValueError("Total marketing spend is zero; cannot compute prior sigmas.")

    spend_pct = spend_per_channel / total
    n_channels = len(spend_cols)
    prior_sigma = n_channels * spend_pct

    logger.info(
        "Calculate channels prior sigmas for %s channels successfully.", n_channels
    )
    return prior_sigma.tolist()
