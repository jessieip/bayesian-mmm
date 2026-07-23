"""Supabase data access for production MMM pipeline."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
WEEKLY_TABLE = "marketing_spend_weekly"
PREDICTIONS_TABLE = "predictions"
ACTUALS_TABLE = "actuals"


def _default_env_path() -> Path:
    return DEFAULT_ENV_PATH


def _load_credentials(env_path: Path | str | None = None) -> tuple[str, str]:
    """Load Supabase URL and key from environment (after dotenv)."""
    path = _default_env_path() if env_path is None else Path(env_path)
    load_dotenv(dotenv_path=path)

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("Missing url or key in Environment variables.")
    return url, key


def _credentials_from_streamlit_secrets() -> tuple[str, str] | None:
    """Try nested ``st.secrets["supabase"]`` url/key; return None on miss."""
    try:
        import streamlit as st
    except ImportError:
        return None

    try:
        supabase_secrets = st.secrets["supabase"]
        url = supabase_secrets.get("url") if hasattr(supabase_secrets, "get") else (
            supabase_secrets["url"]
        )
        key = supabase_secrets.get("key") if hasattr(supabase_secrets, "get") else (
            supabase_secrets["key"]
        )
    except Exception:
        return None

    if not url or not key:
        return None
    return str(url), str(key)


def resolve_supabase_credentials(
    env_path: Path | str | None = None,
) -> tuple[str, str]:
    """
    Resolve Supabase credentials for Streamlit-aware callers.

    Precedence:
    1. ``.streamlit/secrets.toml`` nested ``[supabase]`` url/key via ``st.secrets``
    2. ``.env`` ``SUPABASE_URL`` / ``SUPABASE_KEY`` via ``_load_credentials``
    """
    from_secrets = _credentials_from_streamlit_secrets()
    if from_secrets is not None:
        logger.info("Using Supabase credentials from Streamlit secrets")
        return from_secrets

    logger.info("Falling back to Supabase credentials from environment / .env")
    return _load_credentials(env_path)


def _parse_date_sorted_frame(
    rows: list[dict],
    *,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """Build a sorted DataFrame with parsed dates and float numeric columns."""
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(by="date").reset_index(drop=True)


def _get_supabase_client(
    *,
    env_path: Path | str | None = None,
    client: Client | None = None,
    prefer_streamlit_secrets: bool = False,
) -> Client:
    if client is not None:
        return client
    if prefer_streamlit_secrets:
        url, key = resolve_supabase_credentials(env_path)
    else:
        url, key = _load_credentials(env_path)
    return create_client(url, key)


def _as_records(data: pd.DataFrame | list[dict] | None) -> list[dict]:
    if data is None:
        return []
    if isinstance(data, pd.DataFrame):
        if data.empty:
            return []
        return data.to_dict(orient="records")
    return list(data)


def _normalize_prediction_rows(rows: list[dict]) -> list[dict]:
    created_at = datetime.now(timezone.utc).isoformat()
    normalized: list[dict] = []
    for row in rows:
        normalized.append(
            {
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "mean": float(row["mean"]),
                "hdi_lower": float(row["hdi_lower"]),
                "hdi_upper": float(row["hdi_upper"]),
                "created_at": created_at,
            }
        )
    return normalized


def _normalize_actual_rows(rows: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for row in rows:
        normalized.append(
            {
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "opportunities": float(row["opportunities"]),
            }
        )
    return normalized


def extract_data_supabase(
    *,
    env_path: Path | str | None = None,
    client: Client | None = None,
) -> pd.DataFrame:
    """
    Load weekly MMM data from Supabase and return a sorted DataFrame.

    Parameters
    ----------
    env_path
        Path to ``.env`` file; defaults to project root ``.env``.
    client
        Optional pre-built Supabase client (for testing).

    Returns
    -------
    pd.DataFrame
        Rows from ``marketing_spend_weekly`` with ``date`` as datetime.
    """
    url, key = _load_credentials(env_path)

    try:
        logger.info("Connecting to Supabase...")
        supabase = client if client is not None else create_client(url, key)

        response = supabase.table(WEEKLY_TABLE).select("*").execute()

        if not response.data:
            logger.warning("No data returned from Supabase.")
            return pd.DataFrame()

        dataframe_raw = pd.DataFrame(response.data)
        dataframe_raw["date"] = pd.to_datetime(
            dataframe_raw["date"], format="%Y-%m-%d", errors="coerce"
        )

        if dataframe_raw["date"].isnull().any():
            logger.warning("No data returned from Supabase or found invalid dates.")

        dataframe_raw = dataframe_raw.sort_values(by="date").reset_index(drop=True)
        logger.info("Data Extraction Successful. With %s rows.", len(dataframe_raw))

        return dataframe_raw
    except Exception:
        logger.exception("Error happened during data extraction.")
        raise


def load_predictions_supabase(
    *,
    env_path: Path | str | None = None,
    client: Client | None = None,
) -> pd.DataFrame:
    """
    Load forecast rows from the ``predictions`` table.

    Uses Streamlit secrets when available, otherwise ``.env``.
    """
    try:
        logger.info("Loading predictions from Supabase...")
        supabase = _get_supabase_client(
            env_path=env_path,
            client=client,
            prefer_streamlit_secrets=True,
        )
        response = supabase.table(PREDICTIONS_TABLE).select("*").execute()
        if not response.data:
            logger.warning("No predictions returned from Supabase.")
            return pd.DataFrame()

        df = _parse_date_sorted_frame(
            response.data,
            numeric_cols=["mean", "hdi_lower", "hdi_upper"],
        )
        logger.info("Loaded %s prediction row(s).", len(df))
        return df
    except Exception:
        logger.exception("Error happened while loading predictions.")
        raise


def load_actuals_supabase(
    *,
    env_path: Path | str | None = None,
    client: Client | None = None,
) -> pd.DataFrame:
    """
    Load historical opportunities from the ``actuals`` table.

    Uses Streamlit secrets when available, otherwise ``.env``.
    """
    try:
        logger.info("Loading actuals from Supabase...")
        supabase = _get_supabase_client(
            env_path=env_path,
            client=client,
            prefer_streamlit_secrets=True,
        )
        response = supabase.table(ACTUALS_TABLE).select("*").execute()
        if not response.data:
            logger.warning("No actuals returned from Supabase.")
            return pd.DataFrame()

        df = _parse_date_sorted_frame(response.data, numeric_cols=["opportunities"])
        logger.info("Loaded %s actual row(s).", len(df))
        return df
    except Exception:
        logger.exception("Error happened while loading actuals.")
        raise


def save_results_to_supabase(
    result: dict[str, Any],
    *,
    env_path: Path | str | None = None,
    client: Client | None = None,
) -> None:
    """
    Upsert OOS predictions and historical actuals to Supabase.

    Args:
        result: Dictionary with:
            - ``predictions``: DataFrame or list[dict] with
              ``date``, ``mean``, ``hdi_lower``, ``hdi_upper``
            - ``actuals`` (optional): DataFrame or list[dict] with
              ``date``, ``opportunities``
        env_path: Path to ``.env``; defaults to project root ``.env``.
        client: Optional pre-built Supabase client (for testing).

    Raises:
        ValueError: If Supabase credentials are missing.
        Exception: Re-raised if upsert fails.
    """
    url, key = _load_credentials(env_path)
    prediction_rows = _normalize_prediction_rows(
        _as_records(result.get("predictions"))
    )

    if not prediction_rows:
        logger.warning("No predictions to save")
        return

    actual_rows = _normalize_actual_rows(_as_records(result.get("actuals")))

    try:
        logger.info("Connecting to Supabase...")
        supabase = client if client is not None else create_client(url, key)

        logger.info(
            "Upserting %s rows into %s...", len(prediction_rows), PREDICTIONS_TABLE
        )
        supabase.table(PREDICTIONS_TABLE).upsert(prediction_rows).execute()
        logger.info(
            "Successfully saved %s predictions to Supabase", len(prediction_rows)
        )

        if not actual_rows:
            logger.warning("No actuals to save")
            return

        logger.info("Upserting %s rows into %s...", len(actual_rows), ACTUALS_TABLE)
        supabase.table(ACTUALS_TABLE).upsert(actual_rows).execute()
        logger.info("Successfully saved %s actuals to Supabase", len(actual_rows))
    except Exception:
        logger.exception("Error happened during results upsert.")
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    dataset = extract_data_supabase()
