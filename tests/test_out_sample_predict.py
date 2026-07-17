"""Unit tests for src/out_sample_predict.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
import xarray as xr


from mmm_model_prior import CHANNEL_COLUMNS, CONTROL_COLUMNS, mmm_model_prior
from mmm_model_train import mmm_model_train
from out_sample_predict import (
    DEFAULT_N_NEW,
    DEFAULT_N_POINTS,
    build_x_for_prediction,
    build_x_out_of_sample,
    out_sample_predict,
    y_original_scale_from_predictive,
)


def split_xy(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = ["date"] + CHANNEL_COLUMNS + CONTROL_COLUMNS
    return dataset[feature_cols], dataset["opportunities"].rename("y")


def fake_posterior_predictive(
    dates: pd.DatetimeIndex,
    *,
    n_chains: int = 2,
    n_draws: int = 5,
    value: float = 100.0,
) -> dict:
    """Nested structure matching InferenceData-style access used by the module."""
    y_scale = xr.DataArray(
        np.full((n_chains, n_draws, len(dates)), value),
        dims=["chain", "draw", "date"],
        coords={
            "chain": range(n_chains),
            "draw": range(n_draws),
            "date": dates,
        },
        name="y_original_scale",
    )
    return {"posterior_predictive": {"y_original_scale": y_scale}}


def build_mock_oos_mmm(
    channels: list[str] | None = None,
    controls: list[str] | None = None,
    *,
    predictive: dict | None = None,
    l_max: int = 0,
) -> MagicMock:
    if channels is None:
        channels = list(CHANNEL_COLUMNS)
    if controls is None:
        controls = list(CONTROL_COLUMNS)

    mock_mmm = MagicMock()
    mock_mmm.channel_columns = channels
    mock_mmm.control_columns = controls
    mock_mmm.adstock.l_max = l_max

    def _sample(**kwargs):
        if predictive is not None:
            return predictive
        x_pred = kwargs["X"]
        return fake_posterior_predictive(pd.DatetimeIndex(x_pred["date"]))

    mock_mmm.sample_posterior_predictive.side_effect = _sample
    return mock_mmm


def _run_oos(
    dataset: pd.DataFrame,
    mock_mmm: MagicMock,
    **kwargs,
) -> tuple[pd.DataFrame, object, matplotlib.figure.Figure]:
    x, y = split_xy(dataset)
    return out_sample_predict(x, y, mock_mmm, **kwargs)


class TestBuildXOutOfSample:
    def test_future_dates_after_max_and_length(self, full_mmm_dataset):
        x, _ = split_xy(full_mmm_dataset)
        mock_mmm = build_mock_oos_mmm()
        x_oos = build_x_out_of_sample(x, mock_mmm, n_new=3)

        assert len(x_oos) == 3
        assert (x_oos["date"] > x["date"].max()).all()
        dates = pd.to_datetime(x["date"])
        step = dates.sort_values().diff().median()
        expected = pd.date_range(start=dates.max() + step, periods=3, freq=step)
        pd.testing.assert_index_equal(
            pd.DatetimeIndex(x_oos["date"]), expected, check_names=False
        )

    def test_channels_and_controls_from_max_date_row(self, full_mmm_dataset):
        x, _ = split_xy(full_mmm_dataset)
        # Put the chronologically last date first so iloc[-1] would be wrong
        x_unsorted = pd.concat([x.iloc[[-1]], x.iloc[:-1]], ignore_index=True)
        mock_mmm = build_mock_oos_mmm()
        x_oos = build_x_out_of_sample(x_unsorted, mock_mmm, n_new=2)

        last_row = x.loc[x["date"].idxmax()]
        for ch in CHANNEL_COLUMNS:
            assert (x_oos[ch] == last_row[ch]).all()
        for ctrl in CONTROL_COLUMNS:
            assert (x_oos[ctrl] == last_row[ctrl]).all()

    def test_missing_control_warns_and_skips_column(
        self, full_mmm_dataset, caplog: pytest.LogCaptureFixture
    ):
        x, _ = split_xy(full_mmm_dataset)
        x = x.drop(columns=["google_trend_competitor"])
        mock_mmm = build_mock_oos_mmm()

        with caplog.at_level("WARNING", logger="out_sample_predict"):
            x_oos = build_x_out_of_sample(x, mock_mmm, n_new=2)

        assert "google_trend_competitor" not in x_oos.columns
        assert "competitor_spend" in x_oos.columns
        assert "Control variable 'google_trend_competitor' is not found in x input" in (
            caplog.text
        )

    def test_default_n_new_length(self, full_mmm_dataset):
        x, _ = split_xy(full_mmm_dataset)
        x_oos = build_x_out_of_sample(x, build_mock_oos_mmm(), n_new=DEFAULT_N_NEW)
        assert len(x_oos) == DEFAULT_N_NEW


class TestBuildXForPrediction:
    def test_prepends_history_for_adstock_l_max(self, full_mmm_dataset):
        x, _ = split_xy(full_mmm_dataset)
        mock_mmm = build_mock_oos_mmm(l_max=2)
        x_oos = build_x_out_of_sample(x, mock_mmm, n_new=2)
        x_pred = build_x_for_prediction(x, x_oos, mock_mmm)
        assert len(x_pred) == 4
        pd.testing.assert_frame_equal(
            x_pred.iloc[-2:].reset_index(drop=True),
            x_oos.reset_index(drop=True),
        )

    def test_no_history_when_l_max_zero(self, full_mmm_dataset):
        x, _ = split_xy(full_mmm_dataset)
        mock_mmm = build_mock_oos_mmm(l_max=0)
        x_oos = build_x_out_of_sample(x, mock_mmm, n_new=2)
        x_pred = build_x_for_prediction(x, x_oos, mock_mmm)
        pd.testing.assert_frame_equal(
            x_pred.reset_index(drop=True), x_oos.reset_index(drop=True)
        )


class TestYOriginalScaleFromPredictive:
    def test_extracts_and_transposes_date_last(self):
        dates = pd.date_range("2026-02-03", periods=3, freq="W-MON")
        predictive = fake_posterior_predictive(dates)
        result = y_original_scale_from_predictive(predictive)
        assert result.dims[-1] == "date"
        assert list(result.coords["date"].values) == list(dates)

    def test_extracts_from_top_level_dataset_style(self):
        dates = pd.date_range("2026-02-03", periods=2, freq="W-MON")
        y_scale = fake_posterior_predictive(dates)["posterior_predictive"][
            "y_original_scale"
        ]
        result = y_original_scale_from_predictive({"y_original_scale": y_scale})
        assert result.dims[-1] == "date"
        assert result.shape[-1] == 2


class TestReturnTuple:
    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_returns_three_tuple(
        self, mock_lineplot, mock_plot_hdi, full_mmm_dataset
    ):
        result = _run_oos(full_mmm_dataset, build_mock_oos_mmm())
        assert isinstance(result, tuple)
        assert len(result) == 3

    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_x_oos_columns_and_fig_type(
        self, mock_lineplot, mock_plot_hdi, full_mmm_dataset
    ):
        x_oos, y_oos, fig = _run_oos(full_mmm_dataset, build_mock_oos_mmm())
        expected_cols = {"date", *CHANNEL_COLUMNS, *CONTROL_COLUMNS}
        assert set(x_oos.columns) == expected_cols
        assert isinstance(fig, matplotlib.figure.Figure)
        assert "posterior_predictive" in y_oos


class TestSamplePosteriorPredictive:
    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_called_with_expected_kwargs(
        self, mock_lineplot, mock_plot_hdi, full_mmm_dataset
    ):
        mock_mmm = build_mock_oos_mmm()
        x, y = split_xy(full_mmm_dataset)
        expected_x = build_x_out_of_sample(x, mock_mmm, n_new=DEFAULT_N_NEW)

        out_sample_predict(x, y, mock_mmm)

        mock_mmm.sample_posterior_predictive.assert_called_once()
        call_kwargs = mock_mmm.sample_posterior_predictive.call_args.kwargs
        assert call_kwargs["extend_idata"] is False
        assert call_kwargs["include_last_observations"] is False
        pd.testing.assert_frame_equal(call_kwargs["X"], expected_x)

    @patch("out_sample_predict.np.random.default_rng")
    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_random_seed_forwarded_via_default_rng(
        self,
        mock_lineplot,
        mock_plot_hdi,
        mock_default_rng,
        full_mmm_dataset,
    ):
        mock_rng = MagicMock()
        mock_default_rng.return_value = mock_rng
        mock_mmm = build_mock_oos_mmm()
        _run_oos(full_mmm_dataset, mock_mmm, random_seed=99)

        mock_default_rng.assert_called_once_with(99)
        assert (
            mock_mmm.sample_posterior_predictive.call_args.kwargs["random_seed"]
            is mock_rng
        )

    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_custom_n_new_passed_to_sampler(
        self, mock_lineplot, mock_plot_hdi, full_mmm_dataset
    ):
        mock_mmm = build_mock_oos_mmm()
        _run_oos(full_mmm_dataset, mock_mmm, n_new=3)
        x_arg = mock_mmm.sample_posterior_predictive.call_args.kwargs["X"]
        assert len(x_arg) == 3


class TestHistoryPlotWindow:
    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_lineplot_uses_last_n_points(
        self, mock_lineplot, mock_plot_hdi, full_mmm_dataset
    ):
        _run_oos(full_mmm_dataset, build_mock_oos_mmm(), n_points=2)
        mock_lineplot.assert_called_once()
        line_kwargs = mock_lineplot.call_args.kwargs
        assert len(line_kwargs["x"]) == 2
        assert len(line_kwargs["y"]) == 2

    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_dataframe_y_uses_first_column(
        self, mock_lineplot, mock_plot_hdi, full_mmm_dataset
    ):
        x, _ = split_xy(full_mmm_dataset)
        y_df = full_mmm_dataset[["opportunities", "sales"]]
        mock_mmm = build_mock_oos_mmm()
        out_sample_predict(x, y_df, mock_mmm, n_points=2)

        line_kwargs = mock_lineplot.call_args.kwargs
        expected = y_df.iloc[-2:, 0].reset_index(drop=True)
        pd.testing.assert_series_equal(
            line_kwargs["y"].reset_index(drop=True),
            expected,
            check_names=False,
        )


class TestForecastPlotting:
    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_plot_hdi_called_with_smooth_false(
        self, mock_lineplot, mock_plot_hdi, full_mmm_dataset
    ):
        _run_oos(full_mmm_dataset, build_mock_oos_mmm())
        mock_plot_hdi.assert_called_once()
        kwargs = mock_plot_hdi.call_args.kwargs
        assert kwargs["smooth"] is False
        assert kwargs["fill_kwargs"] == {"alpha": 0.25, "color": "C0"}
        # stacked (sample, date) numpy array for ArviZ
        y_arg = mock_plot_hdi.call_args.args[1]
        assert isinstance(y_arg, np.ndarray)
        assert y_arg.ndim == 2
        assert y_arg.shape[1] == DEFAULT_N_NEW

    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_plot_hdi_x_matches_future_dates(
        self, mock_lineplot, mock_plot_hdi, full_mmm_dataset
    ):
        x, y = split_xy(full_mmm_dataset)
        mock_mmm = build_mock_oos_mmm()
        x_oos, _, _ = out_sample_predict(x, y, mock_mmm)

        hdi_x = mock_plot_hdi.call_args.args[0]
        expected = pd.to_datetime(x_oos["date"]).to_numpy()
        np.testing.assert_array_equal(hdi_x, expected)

    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_mean_forecast_path_runs(
        self, mock_lineplot, mock_plot_hdi, full_mmm_dataset
    ):
        x_oos, _, fig = _run_oos(full_mmm_dataset, build_mock_oos_mmm())
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(x_oos) == DEFAULT_N_NEW


class TestMatplotlibMode:
    @patch("out_sample_predict.plt.ion")
    @patch("out_sample_predict.plt.ioff")
    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_ioff_before_ion(
        self,
        mock_lineplot,
        mock_plot_hdi,
        mock_ioff,
        mock_ion,
        full_mmm_dataset,
    ):
        call_order: list[str] = []
        mock_ioff.side_effect = lambda: call_order.append("ioff")
        mock_ion.side_effect = lambda: call_order.append("ion")
        _run_oos(full_mmm_dataset, build_mock_oos_mmm())
        mock_ioff.assert_called_once()
        mock_ion.assert_called_once()
        assert call_order == ["ioff", "ion"]


class TestLogging:
    @patch("out_sample_predict.az.plot_hdi")
    @patch("out_sample_predict.sns.lineplot")
    def test_logs_prediction_lifecycle(
        self,
        mock_lineplot,
        mock_plot_hdi,
        full_mmm_dataset,
        caplog: pytest.LogCaptureFixture,
    ):
        with caplog.at_level("INFO", logger="out_sample_predict"):
            _run_oos(full_mmm_dataset, build_mock_oos_mmm(), n_new=4)
        assert "Generating x_out_of_sample" in caplog.text
        assert "Sampling out-of-sample predictive for the next 4 periods" in caplog.text
        assert "Plotting historical actuals" in caplog.text
        assert "Plotting forecasted value line chart" in caplog.text
        assert "Plotting Out of Sample Line chart completed." in caplog.text

    def test_logs_missing_control_warning(
        self, full_mmm_dataset, caplog: pytest.LogCaptureFixture
    ):
        x, _ = split_xy(full_mmm_dataset)
        x = x.drop(columns=["competitor_spend"])
        mock_mmm = build_mock_oos_mmm()
        with caplog.at_level("WARNING", logger="out_sample_predict"):
            build_x_out_of_sample(x, mock_mmm, n_new=2)
        assert "Control variable 'competitor_spend' is not found in x input" in (
            caplog.text
        )


@pytest.mark.slow
class TestIntegrationSlow:
    def test_integration_prior_train_oos_smoke(
        self, full_mmm_dataset, valid_prior_sigma
    ):
        _, _, mmm, _ = mmm_model_prior(
            full_mmm_dataset,
            valid_prior_sigma,
            prior_samples=50,
        )
        x, y = split_xy(full_mmm_dataset)
        mmm_model_train(x, y, mmm, fit_kwargs={"chains": 2, "draws": 50})

        x_oos, y_oos, fig = out_sample_predict(x, y, mmm, n_new=2, n_points=3)

        assert len(x_oos) == 2
        assert isinstance(fig, matplotlib.figure.Figure)
        assert y_oos is not None

