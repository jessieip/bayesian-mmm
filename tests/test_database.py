"""Unit tests for src/database.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from database import WEEKLY_TABLE, extract_data_supabase


ENV = {"SUPABASE_URL": "http://test.supabase.co", "SUPABASE_KEY": "test-key"}


def build_mock_client(data: list[dict]) -> MagicMock:
    client = MagicMock()
    client.table.return_value.select.return_value.execute.return_value = MagicMock(
        data=data
    )
    return client


class TestCredentialValidation:
    @patch("database.load_dotenv")
    @patch.dict("os.environ", {"SUPABASE_URL": "", "SUPABASE_KEY": "key"}, clear=False)
    def test_raises_value_error_when_url_missing(self, mock_load_dotenv):
        with pytest.raises(ValueError, match="Missing url or key"):
            extract_data_supabase()

    @patch("database.load_dotenv")
    @patch.dict("os.environ", {"SUPABASE_URL": "http://test", "SUPABASE_KEY": ""}, clear=False)
    def test_raises_value_error_when_key_missing(self, mock_load_dotenv):
        with pytest.raises(ValueError, match="Missing url or key"):
            extract_data_supabase()

    @patch("database.load_dotenv")
    @patch.dict("os.environ", {"SUPABASE_URL": "", "SUPABASE_KEY": ""}, clear=False)
    def test_raises_value_error_when_both_missing(self, mock_load_dotenv):
        with pytest.raises(ValueError, match="Missing url or key"):
            extract_data_supabase()


class TestHappyPath:
    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_returns_sorted_dataframe_with_parsed_dates(
        self, mock_load_dotenv, mock_supabase_client, sample_weekly_rows
    ):
        df = extract_data_supabase(client=mock_supabase_client)

        assert len(df) == len(sample_weekly_rows)
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert df["date"].is_monotonic_increasing
        assert df.iloc[0]["date"] == pd.Timestamp("2026-06-01")
        assert df.iloc[1]["date"] == pd.Timestamp("2026-06-08")

    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_calls_correct_table_and_select(
        self, mock_load_dotenv, mock_supabase_client
    ):
        extract_data_supabase(client=mock_supabase_client)

        mock_supabase_client.table.assert_called_once_with(WEEKLY_TABLE)
        mock_supabase_client.table.return_value.select.assert_called_once_with("*")
        mock_supabase_client.table.return_value.select.return_value.execute.assert_called_once()

    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_preserves_all_columns_from_response(
        self, mock_load_dotenv, mock_supabase_client, sample_weekly_rows
    ):
        df = extract_data_supabase(client=mock_supabase_client)
        expected_columns = set(sample_weekly_rows[0].keys())
        assert set(df.columns) == expected_columns


class TestEmptyAndEdgeData:
    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_returns_empty_dataframe_when_no_data(self, mock_load_dotenv):
        client = build_mock_client([])
        df = extract_data_supabase(client=client)
        assert df.empty

    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_logs_warning_when_no_data(self, mock_load_dotenv, caplog):
        client = build_mock_client([])
        with caplog.at_level("WARNING"):
            extract_data_supabase(client=client)
        assert "No data returned from Supabase" in caplog.text


class TestInvalidDates:
    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_logs_warning_on_invalid_dates(self, mock_load_dotenv, caplog):
        row = {
            "date": "not-a-date",
            "PPC_Brand_Spend": 1000.0,
            "PPC_Generic_Spend": 800.0,
            "Display_Spend": 600.0,
            "Social_Spend": 400.0,
            "TV_Spend": 2000.0,
            "OOH_Spend": 500.0,
            "Meta_Spend": 450.0,
            "Yahoo_Spend": 150.0,
            "competitor_spend": 3000.0,
            "google_trend_competitor": 50.0,
            "opportunities": 100.0,
            "sales": 12.0,
        }
        client = build_mock_client([row])
        with caplog.at_level("WARNING"):
            df = extract_data_supabase(client=client)
        assert "invalid dates" in caplog.text.lower()
        assert isinstance(df, pd.DataFrame)


class TestErrorPropagation:
    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_reraises_supabase_exception(self, mock_load_dotenv, caplog):
        client = MagicMock()
        client.table.return_value.select.return_value.execute.side_effect = Exception(
            "network"
        )
        with caplog.at_level("ERROR"):
            with pytest.raises(Exception, match="network"):
                extract_data_supabase(client=client)
        assert "Error happened during data extraction" in caplog.text


class TestInjectableClient:
    @patch("database.load_dotenv")
    @patch("database.create_client")
    @patch.dict("os.environ", ENV, clear=False)
    def test_uses_injected_client_without_create_client(
        self, mock_create_client, mock_load_dotenv, mock_supabase_client
    ):
        extract_data_supabase(client=mock_supabase_client)
        mock_create_client.assert_not_called()
