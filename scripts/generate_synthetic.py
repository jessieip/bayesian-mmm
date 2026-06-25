#!/usr/bin/env python
"""
Generate synthetic MMM data: bootstrap or incremental daily refresh.

Usage (from project root):
    # Bootstrap 5 years ending 2026-06-10
    poetry run python scripts/generate_synthetic.py --mode bootstrap --bootstrap-end 2026-06-10

    # Daily refresh as of 2026-06-23 (generates through 2026-06-22)
    poetry run python scripts/generate_synthetic.py --mode refresh --as-of 2026-06-23

    # Refresh with explicit last covered day (testing without Supabase)
    poetry run python scripts/generate_synthetic.py --mode refresh --as-of 2026-06-23 \\
        --last-covered-day 2026-06-10 --skip-supabase
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# Allow imports from src/ without installing the package (package-mode = false).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bayesian_mmm import config as cfg  # noqa: E402
from bayesian_mmm.data.synthetic import refresh_synthetic_data  # noqa: E402


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic MMM daily refresh pipeline")
    parser.add_argument(
        "--mode",
        choices=["bootstrap", "refresh", "auto"],
        default="auto",
        help="bootstrap | refresh | auto (default: auto)",
    )
    parser.add_argument(
        "--as-of",
        type=_parse_date,
        default=None,
        help="Reference today (YYYY-MM-DD); generate through as_of - 1 day",
    )
    parser.add_argument(
        "--bootstrap-end",
        type=_parse_date,
        default=None,
        help="Last day of bootstrap window (default: as_of - 1)",
    )
    parser.add_argument(
        "--bootstrap-years",
        type=int,
        default=None,
        help=f"Bootstrap history length in years (default: {cfg.BOOTSTRAP_YEARS})",
    )
    parser.add_argument(
        "--last-covered-day",
        type=_parse_date,
        default=None,
        help="Override Supabase last covered day (testing)",
    )
    parser.add_argument(
        "--skip-supabase",
        action="store_true",
        help="Do not read from or write to Supabase",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for bootstrap")
    args = parser.parse_args()

    result = refresh_synthetic_data(
        mode=args.mode,
        as_of=args.as_of,
        bootstrap_end=args.bootstrap_end,
        bootstrap_years=args.bootstrap_years,
        last_covered_day=args.last_covered_day,
        skip_supabase=args.skip_supabase,
        seed=args.seed,
    )

    daily = result.daily
    weekly = result.weekly
    dates = daily[cfg.DATE_COLUMN] if not daily.empty else None

    print("=" * 60)
    print("Synthetic MMM refresh")
    print("=" * 60)
    print(f"Mode:               {result.mode}")
    print(f"Message:            {result.message}")
    print(f"Generate through:   {result.generate_through}")
    print(f"Last covered day:   {result.last_covered_day}")
    print(f"Gap:                {result.gap_start} to {result.gap_end}")
    print(f"Daily rows added:   {result.daily_rows_added}")
    print(f"Daily total rows:   {result.daily_total_rows}")
    print(f"Weekly rows updated:{result.weekly_rows_updated}")
    print(f"Weekly upserted:    {result.weekly_upserted}")
    print()
    print(f"Daily CSV:  {cfg.DAILY_MMM_CSV}")
    print(f"Weekly CSV: {cfg.WEEKLY_MMM_CSV}")
    print(f"State JSON: {cfg.SIMULATION_STATE_JSON}")
    print(f"Params JSON:{cfg.TRUE_PARAMS_JSON}")

    if dates is not None and not daily.empty:
        opp = daily["opportunities"]
        sales = daily["sales"]
        q4_mask = dates.dt.month.isin([10, 11, 12])
        q2_mask = dates.dt.month.isin([4, 5, 6])
        print()
        print("Daily date range:", dates.min().date(), "to", dates.max().date())
        print(f"Mean opportunities (daily): {opp.mean():.2f}")
        print(f"Mean sales (daily):         {sales.mean():.2f}")
        print(f"corr(sales, opportunities): {sales.corr(opp):.4f}")
        if q2_mask.any() and q4_mask.any():
            print(
                f"Q4 mean opportunities: {opp[q4_mask].mean():.2f}  "
                f"| Q2 mean: {opp[q2_mask].mean():.2f}  "
                f"| Q4/Q2 ratio: {opp[q4_mask].mean() / opp[q2_mask].mean():.3f}"
            )

    if not weekly.empty:
        wdates = pd.to_datetime(weekly[cfg.DATE_COLUMN])
        print()
        print(f"Weekly rows: {len(weekly)}")
        print("Weekly date range:", wdates.min().date(), "to", wdates.max().date())

    print("=" * 60)


if __name__ == "__main__":
    main()
