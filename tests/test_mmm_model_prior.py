"""Unit tests for src/mmm_model_prior.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib.figure
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pymc_marketing.mmm import GeometricAdstock, HillSaturation

from mmm_model_prior import (
    CHANNEL_COLUMNS,
    CONTROL_COLUMNS,
    mmm_model_prior,
)


def build_mock_mmm(n_dates: int, *, n_chains: int = 2, n_draws: int = 5) -> MagicMock:
    dates = pd.date_range("2026-01-06", periods=n_dates, freq="W-MON")
    y_prior = xr.DataArray(
        np.ones((n_chains, n_draws, n_dates)) * 100.0,
        dims=["chain", "draw", "date"],
        coords={"chain": range(n_chains), "draw": range(n_draws), "date": dates},
    )
    mock_mmm = MagicMock()
    mock_mmm.model.coords = {"date": dates}
    mock_mmm.idata.prior = {"y_original_scale": y_prior}
    return mock_mmm


def _run_with_mocks(
    dataset: pd.DataFrame,
    prior_sigma: list[float],
    mock_mmm_cls: MagicMock,
    *,
    prior_samples: int = 10,
):
    mock_mmm = build_mock_mmm(n_dates=len(dataset))
    mock_mmm_cls.return_value = mock_mmm
    return mmm_model_prior(
        dataset, prior_sigma, prior_samples=prior_samples, mmm_cls=mock_mmm_cls
    )


class TestPriorSigmaValidation:
    @patch("mmm_model_prior.MMM")
    def test_raises_when_prior_sigma_too_short(
        self, mock_mmm_cls, full_mmm_dataset, valid_prior_sigma
    ):
        with pytest.raises(ValueError, match="Length of prior sigma is 7"):
            mmm_model_prior(full_mmm_dataset, valid_prior_sigma[:7], mmm_cls=mock_mmm_cls)
        mock_mmm_cls.assert_not_called()

    @patch("mmm_model_prior.MMM")
    def test_raises_when_prior_sigma_too_long(
        self, mock_mmm_cls, full_mmm_dataset, valid_prior_sigma
    ):
        sigmas = valid_prior_sigma + [2.0]
        with pytest.raises(ValueError, match="Length of prior sigma is 9"):
            mmm_model_prior(full_mmm_dataset, sigmas, mmm_cls=mock_mmm_cls)
        mock_mmm_cls.assert_not_called()

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_accepts_eight_prior_sigmas(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        mock_mmm_cls.assert_called_once()


class TestReturnTuple:
    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_returns_four_tuple(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        result = _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        assert len(result) == 4

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_returns_dataframe_for_X(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        X, _, _, _ = _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        assert isinstance(X, pd.DataFrame)

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_returns_series_for_y(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        _, y, _, _ = _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        assert isinstance(y, pd.Series)
        assert y.name == "y"

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_returns_mmm_instance(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        _, _, mmm, _ = _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        assert mmm is mock_mmm_cls.return_value

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_returns_matplotlib_figure(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        _, _, _, fig = _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        assert isinstance(fig, matplotlib.figure.Figure)


class TestXYConstruction:
    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_X_has_required_columns(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        X, _, _, _ = _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        expected = ["date"] + CHANNEL_COLUMNS + CONTROL_COLUMNS
        assert list(X.columns) == expected

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_y_equals_opportunities_column(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        _, y, _, _ = _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            full_mmm_dataset["opportunities"].reset_index(drop=True),
            check_names=False,
        )
        assert y.name == "y"

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_X_row_count_matches_dataset(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        X, _, _, _ = _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        assert len(X) == len(full_mmm_dataset)

    @patch("mmm_model_prior.MMM")
    def test_raises_key_error_when_opportunities_missing(
        self, mock_mmm_cls, full_mmm_dataset, valid_prior_sigma
    ):
        df = full_mmm_dataset.drop(columns=["opportunities"])
        with pytest.raises(KeyError):
            mmm_model_prior(df, valid_prior_sigma, mmm_cls=mock_mmm_cls)


class TestMMMWiring:
    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_mmm_initialized_with_eight_channels(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        kwargs = mock_mmm_cls.call_args.kwargs
        assert kwargs["channel_columns"] == CHANNEL_COLUMNS
        assert kwargs["control_columns"] == CONTROL_COLUMNS
        assert kwargs["date_column"] == "date"
        assert kwargs["target_column"] == "y"
        assert kwargs["yearly_seasonality"] == 2

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_mmm_uses_hill_saturation_and_geometric_adstock(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        kwargs = mock_mmm_cls.call_args.kwargs
        assert isinstance(kwargs["saturation"], HillSaturation)
        assert isinstance(kwargs["adstock"], GeometricAdstock)
        assert kwargs["adstock"].l_max == 8

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_model_config_passes_prior_sigma_to_saturation_beta(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        _run_with_mocks(full_mmm_dataset, sigmas, mock_mmm_cls)
        model_config = mock_mmm_cls.call_args.kwargs["model_config"]
        beta_prior = model_config["saturation_beta"]
        assert list(beta_prior.parameters["sigma"]) == sigmas

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_build_model_called_with_X_y(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        X, y, mmm, _ = _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        mmm.build_model.assert_called_once_with(X, y)

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_sample_prior_predictive_called(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        X, y, mmm, _ = _run_with_mocks(
            full_mmm_dataset, valid_prior_sigma, mock_mmm_cls, prior_samples=10
        )
        mmm.sample_prior_predictive.assert_called_once_with(X, y, samples=10)


class TestFigurePlot:
    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_figure_title_and_labels(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        _, _, _, fig = _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        ax = fig.axes[0]
        assert "Prior Predictive" in ax.get_title()
        assert ax.get_xlabel() == "Date"
        assert ax.get_ylabel() == "Opportunities"

    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_plot_hdi_called_twice(
        self, mock_mmm_cls, mock_plot_hdi, full_mmm_dataset, valid_prior_sigma
    ):
        _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        assert mock_plot_hdi.call_count == 2
        hdi_probs = [call.kwargs["hdi_prob"] for call in mock_plot_hdi.call_args_list]
        assert hdi_probs == [0.94, 0.5]


class TestLogging:
    @patch("mmm_model_prior.az.plot_hdi")
    @patch("mmm_model_prior.MMM")
    def test_logs_initialisation_complete(
        self,
        mock_mmm_cls,
        mock_plot_hdi,
        full_mmm_dataset,
        valid_prior_sigma,
        caplog: pytest.LogCaptureFixture,
    ):
        with caplog.at_level("INFO", logger="mmm_model_prior"):
            _run_with_mocks(full_mmm_dataset, valid_prior_sigma, mock_mmm_cls)
        assert "MMM Priors Initialisation and Plotting completed" in caplog.text


@pytest.mark.slow
class TestIntegrationSlow:
    def test_integration_real_prior_predictive_smoke(
        self, full_mmm_dataset, valid_prior_sigma
    ):
        X, y, mmm, fig = mmm_model_prior(
            full_mmm_dataset, valid_prior_sigma, prior_samples=50
        )
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert y.name == "y"
        assert isinstance(fig, matplotlib.figure.Figure)
        assert mmm.idata is not None
        assert "prior" in mmm.idata
        assert len(X) == len(full_mmm_dataset)
        assert list(X.columns) == ["date"] + CHANNEL_COLUMNS + CONTROL_COLUMNS
