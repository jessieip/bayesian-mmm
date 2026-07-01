"""Unit tests for src/processor.py."""

from __future__ import annotations

import pandas as pd
import pytest

from bayesian_mmm import config as cfg
from processor import _get_spend_columns, data_process


class TestGetSpendColumns:
    def test_detects_spend_columns_excludes_competitor(
        self, sample_mmm_dataframe: pd.DataFrame
    ):
        cols = _get_spend_columns(sample_mmm_dataframe)
        assert len(cols) == 4
        assert "competitor_spend" not in cols
        assert cols == [
            "PPC_Brand_Spend",
            "Display_Spend",
            "Meta_Spend",
            "Yahoo_Spend",
        ]

    def test_case_insensitive_spend_suffix(self):
        df = pd.DataFrame({"TV_spend": [100.0], "date": ["2026-06-01"]})
        cols = _get_spend_columns(df)
        assert cols == ["TV_spend"]

    def test_ignores_non_spend_columns(self, sample_mmm_dataframe: pd.DataFrame):
        cols = _get_spend_columns(sample_mmm_dataframe)
        for non_spend in ("date", "opportunities", "sales"):
            assert non_spend not in cols


class TestValidationErrors:
    def test_raises_when_no_spend_columns(self, no_spend_dataframe: pd.DataFrame):
        with pytest.raises(ValueError, match="No available marketing spend"):
            data_process(no_spend_dataframe)

    def test_raises_when_total_spend_is_zero(self, zero_spend_dataframe: pd.DataFrame):
        with pytest.raises(ValueError, match="Total marketing spend is zero"):
            data_process(zero_spend_dataframe)


class TestPriorSigmaCalculation:
    def test_equal_spend_returns_all_ones(self):
        df = pd.DataFrame(
            {
                "PPC_Brand_Spend": [1000.0, 1000.0],
                "Display_Spend": [1000.0, 1000.0],
                "Meta_Spend": [1000.0, 1000.0],
                "Yahoo_Spend": [1000.0, 1000.0],
            }
        )
        result = data_process(df)
        assert result == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_single_channel_returns_one(self):
        df = pd.DataFrame({"PPC_Brand_Spend": [100.0, 200.0, 300.0]})
        assert data_process(df) == [1.0]

    def test_unequal_spend_formula(self):
        df = pd.DataFrame(
            {
                "PPC_Brand_Spend": [4000.0],
                "Display_Spend": [1000.0],
            }
        )
        result = data_process(df)
        assert result[0] == pytest.approx(1.6)
        assert result[1] == pytest.approx(0.4)

    def test_return_length_matches_channel_count(
        self, sample_mmm_dataframe: pd.DataFrame
    ):
        result = data_process(sample_mmm_dataframe)
        assert len(result) == 4

    def test_return_order_matches_column_order(
        self, sample_mmm_dataframe: pd.DataFrame
    ):
        spend_cols = _get_spend_columns(sample_mmm_dataframe)
        result = data_process(sample_mmm_dataframe)
        totals = sample_mmm_dataframe[spend_cols].sum()
        total = totals.sum()
        expected = (len(spend_cols) * totals / total).tolist()
        assert result == pytest.approx(expected)

    def test_prior_sigma_sums_to_channel_count(
        self, sample_mmm_dataframe: pd.DataFrame
    ):
        result = data_process(sample_mmm_dataframe)
        assert sum(result) == pytest.approx(4.0)

    def test_works_with_realistic_schema_columns(self):
        data = {col: [1000.0] for col in cfg.SPEND_COLUMNS}
        data["competitor_spend"] = [5000.0]
        data["date"] = ["2026-06-01"]
        df = pd.DataFrame(data)
        result = data_process(df)
        assert len(result) == len(cfg.SPEND_COLUMNS)
        assert sum(result) == pytest.approx(float(len(cfg.SPEND_COLUMNS)))


class TestLogging:
    def test_logs_success_message(
        self, sample_mmm_dataframe: pd.DataFrame, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level("INFO", logger="processor"):
            data_process(sample_mmm_dataframe)
        assert "prior sigmas" in caplog.text.lower()
        assert "4 channels" in caplog.text
