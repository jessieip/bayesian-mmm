#!/usr/bin/env python
"""
Generate synthetic weekly MMM data and print verification summary.

Usage (from project root):
    poetry run python scripts/generate_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# Allow imports from src/ without installing the package (package-mode = false).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_mmm import config as cfg  # noqa: E402
from bayesian_mmm.data.synthetic import save_synthetic_dataset  # noqa: E402




def main() -> None:
    df, true_params = save_synthetic_dataset()

    dates = df[cfg.DATE_COLUMN]
    opp = df["opportunities"]
    sales = df["sales"]

    q4_mask = dates.dt.month.isin([10, 11, 12])
    q2_mask = dates.dt.month.isin([4, 5, 6])

    print("=" * 60)
    print("Synthetic MMM dataset generated")
    print("=" * 60)
    print(f"CSV:   {cfg.WEEKLY_MMM_CSV}")
    print(f"JSON:  {cfg.TRUE_PARAMS_JSON}")
    print(f"Rows:  {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")
    print()
    print("Date range:", dates.min().date(), "to", dates.max().date())
    print(f"Mean opportunities: {opp.mean():.2f}")
    print(f"Mean sales:         {sales.mean():.2f}")
    print(f"corr(sales, opportunities): {sales.corr(opp):.4f}")
    print(
        f"Q4 mean opportunities: {opp[q4_mask].mean():.2f}  "
        f"| Q2 mean: {opp[q2_mask].mean():.2f}  "
        f"| Q4/Q2 ratio: {opp[q4_mask].mean() / opp[q2_mask].mean():.3f}"
    )
    print()
    print("Ground-truth channel betas (opportunities scale):")
    for ch, beta in true_params["channel_beta"].items():
        half_sat = true_params["channel_hill_half_sat"][ch]
        alpha = true_params["channel_adstock_alpha"][ch]
        print(f"  {ch:12s}  beta={beta:7.1f}  alpha={alpha:.2f}  half_sat={half_sat:.2f}")
    print("=" * 60)

    # Read the environment variables from .env
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    # Initialise Supabase Client
    supabase: Client =  create_client(os.environ.get('SUPABASE_URL'), os.environ.get('SUPABASE_KEY'))

    df_n = df.copy()

    # convert to datetime format
    df_n[cfg.DATE_COLUMN] = pd.to_datetime(df_n[cfg.DATE_COLUMN], errors='coerce')
    # change the format to Y-m-d (string format)
    df_n[cfg.DATE_COLUMN] = df_n[cfg.DATE_COLUMN].dt.strftime("%Y-%m-%d")

    # Convert csv to dictionary that is required in Supabase
    df_dict = df_n.to_dict(orient = 'records')

    try:
        response = supabase.table('marketing_spend_weekly').upsert(df_dict).execute()
        print("Insert New Data Successful!")
    except Exception as e:
        print(f"Insert New Data Failure: {e}")


if __name__ == "__main__":
    main()
