"""Supabase data access for production MMM pipeline."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
WEEKLY_TABLE = "marketing_spend_weekly"


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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    dataset = extract_data_supabase()
