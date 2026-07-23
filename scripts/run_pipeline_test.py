#!/usr/bin/env python
"""
End-to-end MMM smoke pipeline: extract → train → OOS → upsert → verify.

Usage (from project root):
    poetry run python scripts/run_pipeline_test.py

    # Skip chart-heavy stages
    poetry run python scripts/run_pipeline_test.py --skip-predict --skip-roas

    # Stronger (still non-production) sampler
    poetry run python scripts/run_pipeline_test.py --chains 2 --draws 100 --prior-samples 100

Prerequisites:
    - ``.env`` with ``SUPABASE_URL`` and ``SUPABASE_KEY``
    - Non-empty ``marketing_spend_weekly`` table
    - ``predictions`` and ``actuals`` tables already created

This script uses SMOKE sampler settings by default (not production fit quality).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from supabase import create_client

# Allow imports from src/ without installing the package (package-mode = false).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from database import (  # noqa: E402
    PREDICTIONS_TABLE,
    _load_credentials,
    extract_data_supabase,
    save_results_to_supabase,
)
from mmm_model_predict import mmm_model_predict  # noqa: E402
from mmm_model_prior import mmm_model_prior  # noqa: E402
from mmm_model_train import mmm_model_train  # noqa: E402
from mmm_roas import mmm_roas  # noqa: E402
from out_sample_predict import out_sample_predict, summarize_out_of_sample  # noqa: E402
from processor import data_process  # noqa: E402

logger = logging.getLogger(__name__)


def verify_predictions_in_supabase(
    predictions: pd.DataFrame,
    *,
    client=None,
) -> pd.DataFrame:
    """Select upserted forecast rows and assert count matches local summary."""
    if predictions.empty:
        raise ValueError("No local predictions to verify against Supabase")

    forecast_dates = (
        pd.to_datetime(predictions["date"]).dt.strftime("%Y-%m-%d").tolist()
    )

    if client is None:
        url, key = _load_credentials()
        client = create_client(url, key)

    response = (
        client.table(PREDICTIONS_TABLE)
        .select("*")
        .in_("date", forecast_dates)
        .execute()
    )
    remote = pd.DataFrame(response.data or [])
    if remote.empty:
        raise RuntimeError(
            f"Verification failed: no rows in {PREDICTIONS_TABLE} for {forecast_dates}"
        )

    remote["date"] = pd.to_datetime(remote["date"]).dt.strftime("%Y-%m-%d")
    missing = sorted(set(forecast_dates) - set(remote["date"].tolist()))
    if missing:
        raise RuntimeError(
            f"Verification failed: missing prediction dates in Supabase: {missing}"
        )

    if len(remote) < len(forecast_dates):
        raise RuntimeError(
            f"Verification failed: expected >= {len(forecast_dates)} rows, "
            f"got {len(remote)}"
        )

    logger.info(
        "Verified %s prediction row(s) in Supabase for dates %s",
        len(remote),
        forecast_dates,
    )
    return remote.sort_values("date").reset_index(drop=True)


def run_pipeline(
    *,
    prior_samples: int = 50,
    chains: int = 2,
    draws: int = 50,
    n_new: int = 5,
    skip_predict: bool = False,
    skip_roas: bool = False,
) -> pd.DataFrame:
    """Run the MMM smoke pipeline and return verified Supabase prediction rows."""
    logger.info(
        "SMOKE sampler settings: prior_samples=%s chains=%s draws=%s n_new=%s "
        "(not a production fit)",
        prior_samples,
        chains,
        draws,
        n_new,
    )

    logger.info("=== 1/11 Extract ===")
    dataset = extract_data_supabase()
    if dataset.empty:
        raise RuntimeError("marketing_spend_weekly returned no rows; aborting pipeline")
    logger.info("Loaded %s weekly rows", len(dataset))

    logger.info("=== 2/11 Process prior sigmas ===")
    prior_sigma = data_process(dataset)
    logger.info("prior_sigma=%s", prior_sigma)

    logger.info("=== 3/11 Prior predictive ===")
    X, y, mmm, _fig_prior = mmm_model_prior(
        dataset,
        prior_sigma,
        prior_samples=prior_samples,
    )

    logger.info("=== 4/11 Train ===")
    mmm_model_train(
        X,
        y,
        mmm,
        fit_kwargs={"chains": chains, "draws": draws},
    )

    if not skip_predict:
        logger.info("=== 5/11 In-sample predict ===")
        plot_df = dataset.copy()
        plot_df["y"] = dataset["opportunities"]
        predict_result = mmm_model_predict(
            mmm,
            X,
            plot_df,
            decomposition_save_path=None,
            sweep_values=np.linspace(0, 1.0, 5),
        )
        logger.info(
            "Predict complete; mean contribution shape=%s",
            getattr(predict_result["df_mean_contribution"], "shape", None),
        )
    else:
        logger.info("=== 5/11 In-sample predict (skipped) ===")

    if not skip_roas:
        logger.info("=== 6/11 ROAS ===")
        mmm_roas(mmm, roas_save_path=None)
    else:
        logger.info("=== 6/11 ROAS (skipped) ===")

    logger.info("=== 7/11 Out-of-sample predict ===")
    x_oos, y_oos, _fig_oos = out_sample_predict(X, y, mmm, n_new=n_new)
    logger.info("OOS feature rows:\n%s", x_oos[["date"]].to_string(index=False))

    logger.info("=== 8/11 Summarize forecasts ===")
    predictions = summarize_out_of_sample(x_oos, y_oos)
    logger.info("Local predictions:\n%s", predictions.to_string(index=False))

    logger.info("=== 9/11 Build actuals ===")
    actuals = dataset[["date", "opportunities"]].copy()
    logger.info("Actuals rows=%s", len(actuals))

    logger.info("=== 10/11 Save to Supabase ===")
    save_results_to_supabase({"predictions": predictions, "actuals": actuals})

    logger.info("=== 11/11 Verify Supabase predictions ===")
    remote = verify_predictions_in_supabase(predictions)
    logger.info("Remote predictions:\n%s", remote.to_string(index=False))

    logger.info("PIPELINE OK: forecasts generated and verified in Supabase")
    return remote


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the full Bayesian MMM pipeline and Supabase upsert"
    )
    parser.add_argument(
        "--prior-samples",
        type=int,
        default=50,
        help="Prior predictive samples (default: 50)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=2,
        help="MCMC chains (default: 2)",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=50,
        help="MCMC draws per chain (default: 50)",
    )
    parser.add_argument(
        "--n-new",
        type=int,
        default=5,
        help="Out-of-sample horizon in weeks (default: 5)",
    )
    parser.add_argument(
        "--skip-predict",
        action="store_true",
        help="Skip in-sample predict / sensitivity charts",
    )
    parser.add_argument(
        "--skip-roas",
        action="store_true",
        help="Skip ROAS charts",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    try:
        run_pipeline(
            prior_samples=args.prior_samples,
            chains=args.chains,
            draws=args.draws,
            n_new=args.n_new,
            skip_predict=args.skip_predict,
            skip_roas=args.skip_roas,
        )
    except Exception:
        logger.exception("PIPELINE FAILED")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
