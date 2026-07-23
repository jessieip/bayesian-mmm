"""
Read-only Streamlit dashboard for Bayesian MMM forecasts.

Loads ``predictions`` and ``actuals`` from Supabase. Does **not** run the model.

Credentials (precedence):
1. ``.streamlit/secrets.toml`` nested ``[supabase]`` ``url`` / ``key``
2. ``.env`` ``SUPABASE_URL`` / ``SUPABASE_KEY``

Usage (from project root)::

    poetry run streamlit run src/streamlit_app.py

Prerequisites: credentials as above, and populated ``predictions`` / ``actuals`` tables
(e.g. after ``poetry run python scripts/run_pipeline_test.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Ensure ``src/`` is on the path when launched via ``streamlit run src/streamlit_app.py``.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from database import load_actuals_supabase, load_predictions_supabase  # noqa: E402

DEFAULT_ACTUAL_WEEKS = 15


@st.cache_data(ttl=300)
def _cached_predictions() -> pd.DataFrame:
    return load_predictions_supabase()


@st.cache_data(ttl=300)
def _cached_actuals() -> pd.DataFrame:
    return load_actuals_supabase()


def _build_forecast_figure(
    actuals: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    n_actual_weeks: int,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")

    if not actuals.empty:
        plot_actuals = actuals.iloc[-n_actual_weeks:]
        ax.plot(
            plot_actuals["date"],
            plot_actuals["opportunities"],
            marker="o",
            markersize=5,
            color="blue",
            label="actuals",
        )

    if not predictions.empty:
        ax.fill_between(
            predictions["date"],
            predictions["hdi_lower"],
            predictions["hdi_upper"],
            color="C0",
            alpha=0.25,
            label="94% HDI",
        )
        ax.plot(
            predictions["date"],
            predictions["mean"],
            marker="o",
            markersize=5,
            color="C0",
            linestyle="--",
            label="forecast mean",
        )

    ax.set_xlabel("Date")
    ax.set_ylabel("Opportunities")
    ax.set_title("Out of sample predictions for MMM")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    return fig


def main() -> None:
    st.set_page_config(page_title="Bayesian MMM Forecast", layout="wide")
    st.title("Bayesian MMM Forecast")
    st.caption(
        "Visualization only — results are loaded from Supabase. "
        "The model is not trained or sampled in this app."
    )

    with st.sidebar:
        st.header("Controls")
        n_actual_weeks = st.slider(
            "Historical actual weeks",
            min_value=4,
            max_value=52,
            value=DEFAULT_ACTUAL_WEEKS,
        )
        if st.button("Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    try:
        predictions = _cached_predictions()
        actuals = _cached_actuals()
    except ValueError as exc:
        st.error(
            f"{exc}\n\n"
            "Set credentials in `.streamlit/secrets.toml` under `[supabase]` "
            "(preferred for Streamlit Cloud), or in `.env` as "
            "`SUPABASE_URL` / `SUPABASE_KEY` for local development."
        )
        return
    except Exception as exc:
        st.error(f"Failed to load data from Supabase: {exc}")
        return

    if predictions.empty and actuals.empty:
        st.warning(
            "No predictions or actuals found. "
            "Run `poetry run python scripts/run_pipeline_test.py` first to populate "
            "the Supabase tables."
        )
        return

    if not predictions.empty:
        first = predictions.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Next forecast date", pd.Timestamp(first["date"]).date().isoformat())
        cols[1].metric("Forecast mean", f"{float(first['mean']):.1f}")
        cols[2].metric(
            "HDI interval",
            f"{float(first['hdi_lower']):.1f} – {float(first['hdi_upper']):.1f}",
        )
        if not actuals.empty:
            last_actual = float(actuals.iloc[-1]["opportunities"])
            cols[3].metric("Last actual", f"{last_actual:.1f}")

    fig = _build_forecast_figure(
        actuals, predictions, n_actual_weeks=n_actual_weeks
    )
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.subheader("Predictions")
        st.dataframe(predictions, use_container_width=True)
    with right:
        st.subheader("Actuals")
        st.dataframe(actuals, use_container_width=True)


if __name__ == "__main__":
    main()
