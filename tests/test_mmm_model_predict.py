"""Unit tests for src/mmm_model_predict.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from mmm_model_prior import CHANNEL_COLUMNS, CONTROL_COLUMNS, mmm_model_prior
from mmm_model_predict import (
    DECOMPOSITION_VARS,
    DEFAULT_HDI_PROB,
    DEFAULT_SWEEP_VALUES,
    POSTERIOR_REF_VAL,
    RESULT_KEYS,
    mmm_model_predict,
)
from mmm_model_train import mmm_model_train


def split_x(dataset: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["date"] + CHANNEL_COLUMNS + CONTROL_COLUMNS
    return dataset[feature_cols]


def _make_fig_axes(*, with_legend: bool = False) -> tuple[matplotlib.figure.Figure, np.ndarray]:
    fig = matplotlib.figure.Figure()
    ax = fig.add_subplot(111)
    if with_legend:
        ax.legend(["series"])
    return fig, np.array([[ax]])


def build_mock_predict_mmm(
    *,
    mean_contribution: pd.DataFrame | None = None,
) -> MagicMock:
    if mean_contribution is None:
        mean_contribution = pd.DataFrame(
            {"PPC_Brand_Spend": [10.0], "Display_Spend": [5.0]},
            index=["contribution"],
        )

    mock_mmm = MagicMock()
    mock_mmm.target_column = "opportunities"
    mock_mmm.date_column = "date"
    mock_mmm.channel_columns = list(CHANNEL_COLUMNS)
    mock_mmm.idata = {"posterior": MagicMock()}
    mock_mmm.idata["posterior"].data_vars = ["adstock_alpha", "saturation_slope"]

    fig_pp, axes_pp = _make_fig_axes()
    fig_contrib, axes_contrib = _make_fig_axes()
    fig_decomp, axes_decomp = _make_fig_axes(with_legend=True)
    fig_waterfall = matplotlib.figure.Figure()
    fig_waterfall.add_subplot(111)

    mock_mmm.plot.posterior_predictive.return_value = (fig_pp, axes_pp)
    mock_mmm.plot.contributions_over_time.side_effect = [
        (fig_contrib, axes_contrib),
        (fig_contrib, axes_contrib),
        (fig_decomp, axes_decomp),
    ]
    mock_mmm.plot.waterfall_components_decomposition.return_value = (
        fig_waterfall,
        fig_waterfall.axes[0],
    )
    mock_mmm.compute_mean_contributions_over_time.return_value = mean_contribution

    mock_ax_sensitivity = MagicMock()
    mock_mmm.plot.sensitivity_analysis.return_value = mock_ax_sensitivity

    return mock_mmm


def _run_predict(
    dataset: pd.DataFrame,
    mock_mmm: MagicMock,
    **kwargs,
) -> dict:
    x = split_x(dataset)
    return mmm_model_predict(mock_mmm, x, dataset, **kwargs)


class TestReturnDict:
    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_returns_dict_with_seven_keys(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        result = _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(RESULT_KEYS)

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_figure_values_are_figures(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        gcf_fig = matplotlib.figure.Figure()
        mock_gcf.return_value = gcf_fig
        mock_mmm = build_mock_predict_mmm()
        result = _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        figure_keys = [k for k in RESULT_KEYS if k != "df_mean_contribution"]
        for key in figure_keys:
            assert isinstance(result[key], matplotlib.figure.Figure)

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_mean_contribution_is_dataframe(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        result = _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        assert isinstance(result["df_mean_contribution"], pd.DataFrame)

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_mean_contribution_matches_mock_output(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        expected = pd.DataFrame({"TV_Spend": [20.0]}, index=["mean"])
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm(mean_contribution=expected)
        result = _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        pd.testing.assert_frame_equal(result["df_mean_contribution"], expected)


class TestPosteriorPredictive:
    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_sample_posterior_predictive_called_with_x(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        x = split_x(full_mmm_dataset)
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        mock_mmm.sample_posterior_predictive.assert_called_once()
        call_kwargs = mock_mmm.sample_posterior_predictive.call_args.kwargs
        pd.testing.assert_frame_equal(call_kwargs["X"], x)

    @patch("mmm_model_predict.np.random.default_rng")
    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_random_seed_forwarded(
        self,
        mock_lineplot,
        mock_plot_posterior,
        mock_gcf,
        mock_default_rng,
        full_mmm_dataset,
    ):
        mock_rng = MagicMock()
        mock_default_rng.return_value = mock_rng
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, random_seed=99, decomposition_save_path=None)
        mock_default_rng.assert_called_once_with(99)
        assert mock_mmm.sample_posterior_predictive.call_args.kwargs["random_seed"] is mock_rng

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_posterior_predictive_plot_args(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        mock_mmm.plot.posterior_predictive.assert_called_once_with(
            var=["y_original_scale"], hdi_prob=DEFAULT_HDI_PROB
        )

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_lineplot_and_posterior_title(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        result = _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        mock_lineplot.assert_called_once()
        line_kwargs = mock_lineplot.call_args.kwargs
        pd.testing.assert_frame_equal(line_kwargs["data"], full_mmm_dataset)
        assert line_kwargs["x"] == "date"
        assert line_kwargs["y"] == "opportunities"
        assert result["fig_posterior_predictive"]._suptitle.get_text() == "Posterior Predictive Check"


class TestContributionsOverTime:
    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_contributions_over_time_called_three_times(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        assert mock_mmm.plot.contributions_over_time.call_count == 3

        calls = mock_mmm.plot.contributions_over_time.call_args_list
        assert calls[0].kwargs["var"] == ["channel_contribution"]
        assert calls[1].kwargs["var"] == ["channel_contribution_original_scale"]
        assert calls[2].kwargs["var"] == DECOMPOSITION_VARS
        assert calls[2].kwargs["dims"] == {"channel": CHANNEL_COLUMNS}
        assert calls[0].kwargs["hdi_prob"] == DEFAULT_HDI_PROB

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_decomposition_savefig_called_with_default_path(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        with patch.object(matplotlib.figure.Figure, "savefig") as mock_savefig:
            _run_predict(full_mmm_dataset, mock_mmm)
            mock_savefig.assert_called_once_with("mmm_contributions.png", bbox_inches="tight")

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_decomposition_savefig_skipped_when_path_none(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        with patch.object(matplotlib.figure.Figure, "savefig") as mock_savefig:
            _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
            mock_savefig.assert_not_called()

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_legend_repositioning_runs_without_error(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)


class TestWaterfallAndMeanContribution:
    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_waterfall_called_once(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        mock_mmm.plot.waterfall_components_decomposition.assert_called_once()

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_compute_mean_contributions_called_once(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        mock_mmm.compute_mean_contributions_over_time.assert_called_once()


class TestPosteriorParameterPlots:
    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_plot_posterior_called_twice(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        assert mock_plot_posterior.call_count == 2

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_plot_posterior_uses_posterior_idata_and_ref_vals(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)

        alpha_call = mock_plot_posterior.call_args_list[0]
        slope_call = mock_plot_posterior.call_args_list[1]

        assert alpha_call.args[0] is mock_mmm.idata["posterior"]
        assert alpha_call.kwargs["var_names"] == ["adstock_alpha"]
        alpha_ref = alpha_call.kwargs["ref_val"]["adstock_alpha"]
        assert len(alpha_ref) == len(CHANNEL_COLUMNS)
        assert all(item["ref_val"] == POSTERIOR_REF_VAL for item in alpha_ref)

        assert slope_call.kwargs["var_names"] == ["saturation_slope"]
        slope_ref = slope_call.kwargs["ref_val"]["saturation_slope"]
        assert all(item["channel"] in CHANNEL_COLUMNS for item in slope_ref)

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_posterior_figure_suptitles(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        fig_alpha = matplotlib.figure.Figure()
        fig_slope = matplotlib.figure.Figure()
        fig_sensitivity = matplotlib.figure.Figure()
        mock_gcf.side_effect = [fig_alpha, fig_slope, fig_sensitivity]
        mock_mmm = build_mock_predict_mmm()
        result = _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        assert result["fig_adstock"]._suptitle.get_text() == "Adstock Alpha Posterior"
        assert (
            result["fig_saturation"]._suptitle.get_text()
            == "Saturation Slope Posterior Distribution"
        )
        assert result["fig_sensitivity"] is fig_sensitivity


class TestSensitivityAnalysis:
    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_run_sweep_called_with_defaults(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        sweep_kwargs = mock_mmm.sensitivity.run_sweep.call_args.kwargs
        np.testing.assert_array_equal(sweep_kwargs["sweep_values"], DEFAULT_SWEEP_VALUES)
        assert sweep_kwargs["var_input"] == "channel_data"
        assert sweep_kwargs["var_names"] == "channel_contribution_original_scale"
        assert sweep_kwargs["extend_idata"] is True

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_run_sweep_uses_injected_sweep_values(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        custom_sweeps = np.array([0.0, 0.5, 1.0])
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(
            full_mmm_dataset,
            mock_mmm,
            sweep_values=custom_sweeps,
            decomposition_save_path=None,
        )
        sweep_kwargs = mock_mmm.sensitivity.run_sweep.call_args.kwargs
        np.testing.assert_array_equal(sweep_kwargs["sweep_values"], custom_sweeps)

    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_sensitivity_plot_and_axvline(
        self, mock_lineplot, mock_plot_posterior, mock_gcf, full_mmm_dataset
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        mock_mmm.plot.sensitivity_analysis.assert_called_once_with(
            xlabel="Sweep multiplicative",
            ylabel="Total contribution over training period(Original Scale)",
            hue_dim="channel",
            x_sweep_axis="relative",
        )
        mock_mmm.plot.sensitivity_analysis.return_value.axvline.assert_called_once_with(
            1.0, color="black", linestyle="--", linewidth=1
        )


class TestMatplotlibMode:
    @patch("mmm_model_predict.plt.ion")
    @patch("mmm_model_predict.plt.ioff")
    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_ioff_before_ion(
        self,
        mock_lineplot,
        mock_plot_posterior,
        mock_gcf,
        mock_ioff,
        mock_ion,
        full_mmm_dataset,
    ):
        call_order: list[str] = []
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_ioff.side_effect = lambda: call_order.append("ioff")
        mock_ion.side_effect = lambda: call_order.append("ion")
        mock_mmm = build_mock_predict_mmm()
        _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        mock_ioff.assert_called_once()
        mock_ion.assert_called_once()
        assert call_order == ["ioff", "ion"]


class TestLogging:
    @patch("mmm_model_predict.plt.gcf")
    @patch("mmm_model_predict.az.plot_posterior")
    @patch("mmm_model_predict.sns.lineplot")
    def test_logs_prediction_lifecycle(
        self,
        mock_lineplot,
        mock_plot_posterior,
        mock_gcf,
        full_mmm_dataset,
        caplog: pytest.LogCaptureFixture,
    ):
        mock_gcf.return_value = matplotlib.figure.Figure()
        mock_mmm = build_mock_predict_mmm()
        with caplog.at_level("INFO", logger="mmm_model_predict"):
            _run_predict(full_mmm_dataset, mock_mmm, decomposition_save_path=None)
        assert "Sampling posterior predictive" in caplog.text
        assert "Plotting Posterior Predictive Charts" in caplog.text
        assert "Plotting Contribution Over Time Charts" in caplog.text
        assert "Plotting waterfall and contribution chart" in caplog.text
        assert "Channels Adstock lagging effect" in caplog.text
        assert "Predicted Channel Saturation Level" in caplog.text
        assert "Running and plotting sensitivity analysis sweep" in caplog.text
        assert "MMM Prediction and Analysis plotting completed." in caplog.text


@pytest.mark.slow
class TestIntegrationSlow:
    def test_integration_prior_train_predict_smoke(
        self, full_mmm_dataset, valid_prior_sigma
    ):
        _, _, mmm, _ = mmm_model_prior(
            full_mmm_dataset,
            valid_prior_sigma,
            prior_samples=50,
        )
        feature_cols = ["date"] + CHANNEL_COLUMNS + CONTROL_COLUMNS
        x = full_mmm_dataset[feature_cols]
        y = full_mmm_dataset["opportunities"].rename("y")
        mmm_model_train(x, y, mmm, fit_kwargs={"chains": 2, "draws": 50})

        custom_sweeps = np.linspace(0, 1.0, 5)
        plot_df = full_mmm_dataset.copy()
        plot_df["y"] = full_mmm_dataset["opportunities"]
        result = mmm_model_predict(
            mmm,
            x,
            plot_df,
            decomposition_save_path=None,
            sweep_values=custom_sweeps,
        )

        assert set(result.keys()) == set(RESULT_KEYS)
        assert isinstance(result["df_mean_contribution"], pd.DataFrame)
        assert not result["df_mean_contribution"].empty
        for key in RESULT_KEYS:
            if key == "df_mean_contribution":
                continue
            assert isinstance(result[key], matplotlib.figure.Figure)
