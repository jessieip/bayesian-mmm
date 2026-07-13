"""Unit tests for src/mmm_roas.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import matplotlib.figure
import numpy as np
import pytest
import xarray as xr

from mmm_model_prior import CHANNEL_COLUMNS
from mmm_roas import (
    DEFAULT_HDI_PROB,
    DEFAULT_ROAS_SAVE_PATH,
    RESULT_KEYS,
    mmm_roas,
)


def _fake_hdi(_data, hdi_prob=DEFAULT_HDI_PROB):
    mock_ds = MagicMock()
    mock_ds.to_array.return_value.values.flatten.return_value = np.array([1.0, 2.0])
    return mock_ds


def build_mock_roas_mmm(
    channels: list[str] | None = None,
) -> MagicMock:
    if channels is None:
        channels = list(CHANNEL_COLUMNS)

    n_channels = len(channels)
    roas = xr.DataArray(
        np.random.default_rng(0).normal(1.5, 0.3, (2, 5, n_channels)),
        dims=["chain", "draw", "channel"],
        coords={"chain": [0, 1], "draw": range(5), "channel": channels},
        name="roas",
    )

    mock_mmm = MagicMock()
    mock_mmm.channel_columns = channels
    mock_mmm.incrementality.contribution_over_spend.return_value = roas
    return mock_mmm


def _run_roas(mock_mmm: MagicMock, **kwargs) -> dict:
    if "roas_save_path" not in kwargs:
        kwargs["roas_save_path"] = None
    return mmm_roas(mock_mmm, **kwargs)


class TestReturnDict:
    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_returns_dict_with_expected_keys(self, mock_plot_posterior, mock_hdi):
        result = _run_roas(build_mock_roas_mmm())
        assert isinstance(result, dict)
        assert set(result.keys()) == set(RESULT_KEYS)

    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_fig_roas_is_figure(self, mock_plot_posterior, mock_hdi):
        result = _run_roas(build_mock_roas_mmm())
        assert isinstance(result["fig_roas"], matplotlib.figure.Figure)


class TestIncrementalityCall:
    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_contribution_over_spend_called_with_all_time(
        self, mock_plot_posterior, mock_hdi
    ):
        mock_mmm = build_mock_roas_mmm()
        _run_roas(mock_mmm)
        mock_mmm.incrementality.contribution_over_spend.assert_called_once_with(
            frequency="all_time"
        )


class TestPlotting:
    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_plot_posterior_called_once_per_channel(
        self, mock_plot_posterior, mock_hdi
    ):
        channels = ["PPC_Brand_Spend", "Display_Spend", "TV_Spend"]
        mock_mmm = build_mock_roas_mmm(channels)
        _run_roas(mock_mmm)
        assert mock_plot_posterior.call_count == len(channels)

    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_titles_strip_spend_suffix(self, mock_plot_posterior, mock_hdi):
        channels = ["PPC_Brand_Spend", "Display_Spend"]
        result = _run_roas(build_mock_roas_mmm(channels))
        titles = [ax.get_title() for ax in result["fig_roas"].axes]
        assert "PPC_Brand ROAS" in titles
        assert "Display ROAS" in titles

    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_hdi_uses_default_prob(self, mock_plot_posterior, mock_hdi):
        _run_roas(build_mock_roas_mmm(["Display_Spend"]))
        assert mock_hdi.call_count == 1
        assert mock_hdi.call_args.kwargs["hdi_prob"] == DEFAULT_HDI_PROB

    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_hdi_uses_injected_prob(self, mock_plot_posterior, mock_hdi):
        custom_prob = 0.89
        result = _run_roas(build_mock_roas_mmm(["Display_Spend"]), hdi_prob=custom_prob)
        assert mock_hdi.call_args.kwargs["hdi_prob"] == custom_prob
        assert "89% HDI" in result["fig_roas"]._suptitle.get_text()


class TestSavefig:
    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_savefig_called_with_default_path(self, mock_plot_posterior, mock_hdi):
        mock_mmm = build_mock_roas_mmm(["Display_Spend"])
        with patch.object(matplotlib.figure.Figure, "savefig") as mock_savefig:
            mmm_roas(mock_mmm)
            mock_savefig.assert_called_once_with(
                DEFAULT_ROAS_SAVE_PATH, bbox_inches="tight"
            )

    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_savefig_skipped_when_path_none(self, mock_plot_posterior, mock_hdi):
        mock_mmm = build_mock_roas_mmm(["Display_Spend"])
        with patch.object(matplotlib.figure.Figure, "savefig") as mock_savefig:
            _run_roas(mock_mmm, roas_save_path=None)
            mock_savefig.assert_not_called()


class TestMatplotlibMode:
    @patch("mmm_roas.plt.ion")
    @patch("mmm_roas.plt.ioff")
    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_ioff_before_ion(
        self, mock_plot_posterior, mock_hdi, mock_ioff, mock_ion
    ):
        call_order: list[str] = []
        mock_ioff.side_effect = lambda: call_order.append("ioff")
        mock_ion.side_effect = lambda: call_order.append("ion")
        _run_roas(build_mock_roas_mmm(["Display_Spend"]))
        mock_ioff.assert_called_once()
        mock_ion.assert_called_once()
        assert call_order == ["ioff", "ion"]


class TestLogging:
    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_logs_roas_lifecycle(
        self,
        mock_plot_posterior,
        mock_hdi,
        caplog: pytest.LogCaptureFixture,
    ):
        channels = ["Display_Spend", "TV_Spend"]
        with caplog.at_level("INFO", logger="mmm_roas"):
            _run_roas(build_mock_roas_mmm(channels))
        assert "Computing Channel ROAS graphs" in caplog.text
        assert f"Plotting ROAS graphs with {len(channels)} channels" in caplog.text

    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_logs_save_path(
        self,
        mock_plot_posterior,
        mock_hdi,
        caplog: pytest.LogCaptureFixture,
    ):
        save_path = "custom_roas.png"
        mock_mmm = build_mock_roas_mmm(["Display_Spend"])
        with (
            patch.object(matplotlib.figure.Figure, "savefig"),
            caplog.at_level("INFO", logger="mmm_roas"),
        ):
            mmm_roas(mock_mmm, roas_save_path=save_path)
        assert f"ROAS charts completed and saved to {save_path}" in caplog.text


class TestGridLayout:
    @patch("mmm_roas.az.hdi", side_effect=_fake_hdi)
    @patch("mmm_roas.az.plot_posterior")
    def test_odd_channel_count_hides_extra_axis(self, mock_plot_posterior, mock_hdi):
        channels = ["PPC_Brand_Spend", "Display_Spend", "TV_Spend"]
        result = _run_roas(build_mock_roas_mmm(channels))
        # 3 channels on a 2x2 grid => one unused axis removed
        assert len(result["fig_roas"].axes) == len(channels)
