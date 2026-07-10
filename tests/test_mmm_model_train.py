"""Unit tests for src/mmm_model_train.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib.figure
import pandas as pd
import pytest

from mmm_model_prior import CHANNEL_COLUMNS, CONTROL_COLUMNS, mmm_model_prior
from mmm_model_train import TARGET_SUMMARY_VARS, mmm_model_train


def split_xy(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = ["date"] + CHANNEL_COLUMNS + CONTROL_COLUMNS
    x = dataset[feature_cols]
    y = dataset["opportunities"].rename("y")
    return x, y


def build_mock_fit_mmm(
    *,
    diverging_count: int = 0,
    available_vars: list[str] | None = None,
    saturation_name: str = "HillSaturation",
    adstock_name: str = "GeometricAdstock",
    missing_sample_stats: bool = False,
) -> MagicMock:
    if available_vars is None:
        available_vars = list(TARGET_SUMMARY_VARS)

    mock_mmm = MagicMock()
    mock_mmm.saturation.__class__.__name__ = saturation_name
    mock_mmm.adstock.__class__.__name__ = adstock_name
    mock_mmm.fit_result.data_vars = available_vars

    if missing_sample_stats:
        mock_mmm.fit_result.__getitem__.side_effect = KeyError("sample_stats")
        return mock_mmm

    mock_diverging = MagicMock()
    mock_diverging.sum.return_value.item.return_value = diverging_count
    mock_sample_stats = {"diverging": mock_diverging}

    def fit_result_getitem(key: str):
        if key == "sample_stats":
            return mock_sample_stats
        raise KeyError(key)

    mock_mmm.fit_result.__getitem__.side_effect = fit_result_getitem
    return mock_mmm


def _run_train(
    dataset: pd.DataFrame,
    mock_mmm: MagicMock,
    *,
    fit_kwargs: dict | None = None,
    random_seed: int = 42,
):
    x, y = split_xy(dataset)
    return mmm_model_train(
        x,
        y,
        mock_mmm,
        fit_kwargs=fit_kwargs,
        random_seed=random_seed,
    )


def _setup_plot_mocks(mock_summary, mock_gcf, *, summary_df: pd.DataFrame | None = None):
    mock_summary.return_value = summary_df or pd.DataFrame({"mean": [1.0]})
    mock_gcf.return_value = matplotlib.figure.Figure()


class TestReturnTuple:
    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_returns_two_tuple(
        self, mock_summary, mock_plot_trace, mock_gcf, full_mmm_dataset
    ):
        mock_summary.return_value = pd.DataFrame({"mean": [1.0]})
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_fit_mmm()
        result = _run_train(full_mmm_dataset, mock_mmm)
        assert len(result) == 2

    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_returns_dataframe_for_summary(
        self, mock_summary, mock_plot_trace, mock_gcf, full_mmm_dataset
    ):
        expected = pd.DataFrame({"mean": [1.0], "sd": [0.1]}, index=["intercept"])
        mock_summary.return_value = expected
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_fit_mmm()
        summary_df, _ = _run_train(full_mmm_dataset, mock_mmm)
        assert isinstance(summary_df, pd.DataFrame)
        pd.testing.assert_frame_equal(summary_df, expected)

    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_returns_matplotlib_figure(
        self, mock_summary, mock_plot_trace, mock_gcf, full_mmm_dataset
    ):
        fig = matplotlib.figure.Figure()
        mock_summary.return_value = pd.DataFrame({"mean": [1.0]})
        mock_gcf.return_value = fig
        mock_mmm = build_mock_fit_mmm()
        _, trace_graph = _run_train(full_mmm_dataset, mock_mmm)
        assert isinstance(trace_graph, matplotlib.figure.Figure)
        assert trace_graph is fig


class TestFitInvocation:
    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_fit_called_with_x_and_y(
        self, mock_summary, mock_plot_trace, mock_gcf, full_mmm_dataset
    ):
        _setup_plot_mocks(mock_summary, mock_gcf)
        mock_mmm = build_mock_fit_mmm()
        x, y = split_xy(full_mmm_dataset)
        _run_train(full_mmm_dataset, mock_mmm)
        mock_mmm.fit.assert_called_once()
        call_kwargs = mock_mmm.fit.call_args.kwargs
        pd.testing.assert_frame_equal(call_kwargs["X"], x)
        pd.testing.assert_series_equal(call_kwargs["y"], y)

    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_fit_uses_default_sampler_args(
        self, mock_summary, mock_plot_trace, mock_gcf, full_mmm_dataset
    ):
        _setup_plot_mocks(mock_summary, mock_gcf)
        mock_mmm = build_mock_fit_mmm()
        _run_train(full_mmm_dataset, mock_mmm)
        call_kwargs = mock_mmm.fit.call_args.kwargs
        assert call_kwargs["chains"] == 4
        assert call_kwargs["draws"] == 1000
        assert call_kwargs["target_accept"] == 0.8

    @patch("mmm_model_train.np.random.default_rng")
    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_random_seed_forwarded(
        self, mock_summary, mock_plot_trace, mock_gcf, mock_default_rng, full_mmm_dataset
    ):
        mock_rng = MagicMock()
        mock_default_rng.return_value = mock_rng
        _setup_plot_mocks(mock_summary, mock_gcf)
        mock_mmm = build_mock_fit_mmm()
        _run_train(full_mmm_dataset, mock_mmm, random_seed=99)
        mock_default_rng.assert_called_once_with(99)
        assert mock_mmm.fit.call_args.kwargs["random_seed"] is mock_rng

    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_fit_kwargs_override_defaults(
        self, mock_summary, mock_plot_trace, mock_gcf, full_mmm_dataset
    ):
        _setup_plot_mocks(mock_summary, mock_gcf)
        mock_mmm = build_mock_fit_mmm()
        _run_train(
            full_mmm_dataset,
            mock_mmm,
            fit_kwargs={"chains": 2, "draws": 50},
        )
        call_kwargs = mock_mmm.fit.call_args.kwargs
        assert call_kwargs["chains"] == 2
        assert call_kwargs["draws"] == 50
        assert call_kwargs["target_accept"] == 0.8


class TestSummaryAndTrace:
    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_summary_receives_fit_result(
        self, mock_summary, mock_plot_trace, mock_gcf, full_mmm_dataset
    ):
        _setup_plot_mocks(mock_summary, mock_gcf)
        mock_mmm = build_mock_fit_mmm()
        _run_train(full_mmm_dataset, mock_mmm)
        assert mock_summary.call_args.kwargs["data"] is mock_mmm.fit_result

    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_summary_var_names_filtered(
        self, mock_summary, mock_plot_trace, mock_gcf, full_mmm_dataset
    ):
        _setup_plot_mocks(mock_summary, mock_gcf)
        available = ["intercept", "sigma", "missing_var"]
        mock_mmm = build_mock_fit_mmm(available_vars=available)
        _run_train(full_mmm_dataset, mock_mmm)
        expected = ["intercept", "sigma"]
        assert mock_summary.call_args.kwargs["var_names"] == expected

    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_plot_trace_uses_same_var_names(
        self, mock_summary, mock_plot_trace, mock_gcf, full_mmm_dataset
    ):
        _setup_plot_mocks(mock_summary, mock_gcf)
        available = ["adstock_alpha", "intercept"]
        mock_mmm = build_mock_fit_mmm(available_vars=available)
        _run_train(full_mmm_dataset, mock_mmm)
        plot_kwargs = mock_plot_trace.call_args.kwargs
        assert plot_kwargs["data"] is mock_mmm.fit_result
        assert plot_kwargs["var_names"] == available
        assert plot_kwargs["compact"] is True

    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_trace_figure_has_title(
        self, mock_summary, mock_plot_trace, mock_gcf, full_mmm_dataset
    ):
        mock_summary.return_value = pd.DataFrame({"mean": [1.0]})
        fig = matplotlib.figure.Figure()
        mock_gcf.return_value = fig
        mock_mmm = build_mock_fit_mmm()
        x, y = split_xy(full_mmm_dataset)
        _, trace_graph = mmm_model_train(x, y, mock_mmm)

        assert trace_graph._suptitle.get_text() == "Model Trace"


class TestDivergingDiagnostics:
    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_logs_warning_when_diverging(
        self,
        mock_summary,
        mock_plot_trace,
        mock_gcf,
        full_mmm_dataset,
        caplog: pytest.LogCaptureFixture,
    ):
        _setup_plot_mocks(mock_summary, mock_gcf)
        mock_mmm = build_mock_fit_mmm(diverging_count=3)
        with caplog.at_level("WARNING", logger="mmm_model_train"):
            _run_train(full_mmm_dataset, mock_mmm)
        assert "Found 3 diverging transitions" in caplog.text

    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_logs_stable_when_no_diverging(
        self,
        mock_summary,
        mock_plot_trace,
        mock_gcf,
        full_mmm_dataset,
        caplog: pytest.LogCaptureFixture,
    ):
        _setup_plot_mocks(mock_summary, mock_gcf)
        mock_mmm = build_mock_fit_mmm(diverging_count=0)
        with caplog.at_level("INFO", logger="mmm_model_train"):
            _run_train(full_mmm_dataset, mock_mmm)
        assert "0 diverging transitions. Sampling is stable" in caplog.text

    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_missing_sample_stats_logs_error_and_returns(
        self,
        mock_summary,
        mock_plot_trace,
        mock_gcf,
        full_mmm_dataset,
        caplog: pytest.LogCaptureFixture,
    ):
        _setup_plot_mocks(mock_summary, mock_gcf)
        mock_mmm = build_mock_fit_mmm(missing_sample_stats=True)

        with caplog.at_level("ERROR", logger="mmm_model_train"):
            summary_df, fig = _run_train(full_mmm_dataset, mock_mmm)

        assert "Failed to extract sample stats" in caplog.text
        assert isinstance(summary_df, pd.DataFrame)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestLogging:
    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_logs_training_lifecycle(
        self,
        mock_summary,
        mock_plot_trace,
        mock_gcf,
        full_mmm_dataset,
        caplog: pytest.LogCaptureFixture,
    ):
        _setup_plot_mocks(mock_summary, mock_gcf)
        mock_mmm = build_mock_fit_mmm(
            saturation_name="HillSaturation",
            adstock_name="GeometricAdstock",
        )
        with caplog.at_level("INFO", logger="mmm_model_train"):
            _run_train(full_mmm_dataset, mock_mmm)
        assert "Training MMM started (4 chains, 1000 draws)" in caplog.text
        assert "training completed" in caplog.text
        assert "HillSaturation" in caplog.text
        assert "GeometricAdstock" in caplog.text
        assert "Model Trace plot generated." in caplog.text


class TestMatplotlibMode:
    @patch("mmm_model_train.plt.ion")
    @patch("mmm_model_train.plt.ioff")
    @patch("mmm_model_train.plt.gcf")
    @patch("mmm_model_train.az.plot_trace")
    @patch("mmm_model_train.az.summary")
    def test_ioff_before_plot_ion_after(
        self,
        mock_summary,
        mock_plot_trace,
        mock_gcf,
        mock_ioff,
        mock_ion,
        full_mmm_dataset,
    ):
        call_order: list[str] = []
        mock_summary.return_value = pd.DataFrame({"mean": [1.0]})
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_ioff.side_effect = lambda: call_order.append("ioff")
        mock_plot_trace.side_effect = lambda *args, **kwargs: call_order.append("plot_trace")
        mock_ion.side_effect = lambda: call_order.append("ion")
        mock_mmm = build_mock_fit_mmm()
        _run_train(full_mmm_dataset, mock_mmm)
        mock_ioff.assert_called_once()
        mock_ion.assert_called_once()
        assert call_order == ["ioff", "plot_trace", "ion"]


@pytest.mark.slow
class TestIntegrationSlow:
    def test_integration_real_fit_smoke(self, full_mmm_dataset, valid_prior_sigma):
        _, _, mmm, _ = mmm_model_prior(
            full_mmm_dataset,
            valid_prior_sigma,
            prior_samples=50,
        )
        x, y = split_xy(full_mmm_dataset)
        summary_df, fig = mmm_model_train(
            x,
            y,
            mmm,
            fit_kwargs={"chains": 2, "draws": 50},
        )
        assert isinstance(summary_df, pd.DataFrame)
        assert not summary_df.empty
        assert isinstance(fig, matplotlib.figure.Figure)
        assert mmm.fit_result is not None
        assert len(mmm.fit_result.data_vars) > 0
