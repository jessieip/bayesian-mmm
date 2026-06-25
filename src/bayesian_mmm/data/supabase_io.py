"""Supabase read/write for synthetic MMM refresh pipeline."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from supabase import Client, create_client

from bayesian_mmm import config as cfg


def get_supabase_client(project_root: Path | None = None) -> Client:
    """Create Supabase client from ``.env`` in project root."""
    root = cfg.PROJECT_ROOT if project_root is None else project_root
    load_dotenv(dotenv_path=root / ".env")
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in .env")
    return create_client(url, key)


def max_date_from_supabase(client: Client, table: str | None = None) -> date | None:
    """Return the max ``date`` stored in the Supabase table, or None if empty."""
    table = cfg.SUPABASE_WEEKLY_TABLE if table is None else table
    resp = (
        client.table(table).select("date").order("date", desc=True).limit(1).execute()
    )
    if not resp.data:
        return None

    first_row = resp.data[0]
    if isinstance(first_row, dict) and "date" in first_row:
        return pd.to_datetime(first_row["date"]).date()
    return None


def last_covered_day_from_supabase(
    client: Client,
    *,
    date_is_week_start: bool | None = None,
) -> date | None:
    """
    Map Supabase max(date) to the last calendar day covered by stored data.

    If ``date_is_week_start`` (default from config), max is Monday → +6 days = Sunday.
    """
    max_d = max_date_from_supabase(client)
    if max_d is None:
        return None

    is_week_start = (
        cfg.DATE_IS_WEEK_START if date_is_week_start is None else date_is_week_start
    )
    if is_week_start:
        return max_d + timedelta(days=6)
    return max_d


def dataframe_to_supabase_records(df: pd.DataFrame) -> list[dict]:
    """Format DataFrame rows for Supabase upsert (date as Y-m-d string)."""
    out = df.copy()
    out[cfg.DATE_COLUMN] = pd.to_datetime(out[cfg.DATE_COLUMN]).dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


def upsert_weekly_rows(client: Client, df_weeks: pd.DataFrame) -> int:
    """Upsert weekly rows to Supabase; return number of rows sent."""
    if df_weeks.empty:
        return 0
    records = dataframe_to_supabase_records(df_weeks)
    client.table(cfg.SUPABASE_WEEKLY_TABLE).upsert(records).execute()
    return len(records)
