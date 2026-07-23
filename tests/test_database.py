"""Unit tests for src/database.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from database import (
    ACTUALS_TABLE,
    PREDICTIONS_TABLE,
    WEEKLY_TABLE,
    extract_data_supabase,
    load_actuals_supabase,
    load_predictions_supabase,
    resolve_supabase_credentials,
    save_results_to_supabase,
)


ENV = {"SUPABASE_URL": "http://test.supabase.co", "SUPABASE_KEY": "test-key"}


def build_mock_client(data: list[dict]) -> MagicMock:
    client = MagicMock()
    client.table.return_value.select.return_value.execute.return_value = MagicMock(
        data=data
    )
    return client


def build_upsert_mock_client() -> MagicMock:
    client = MagicMock()
    client.table.return_value.upsert.return_value.execute.return_value = MagicMock(
        data=[]
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


class TestSaveResultsToSupabase:
    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_upserts_predictions_and_actuals(self, mock_load_dotenv):
        client = build_upsert_mock_client()
        result = {
            "predictions": pd.DataFrame(
                {
                    "date": [pd.Timestamp("2026-02-03"), pd.Timestamp("2026-02-10")],
                    "mean": [100.0, 105.0],
                    "hdi_lower": [90.0, 95.0],
                    "hdi_upper": [110.0, 115.0],
                }
            ),
            "actuals": pd.DataFrame(
                {
                    "date": [pd.Timestamp("2026-01-27")],
                    "opportunities": [110.0],
                }
            ),
        }

        save_results_to_supabase(result, client=client)

        assert client.table.call_count == 2
        assert client.table.call_args_list[0].args[0] == PREDICTIONS_TABLE
        assert client.table.call_args_list[1].args[0] == ACTUALS_TABLE

        pred_rows = client.table.return_value.upsert.call_args_list[0].args[0]
        assert pred_rows[0]["date"] == "2026-02-03"
        assert pred_rows[0]["mean"] == 100.0
        assert pred_rows[0]["hdi_lower"] == 90.0
        assert pred_rows[0]["hdi_upper"] == 110.0
        assert "created_at" in pred_rows[0]

        actual_rows = client.table.return_value.upsert.call_args_list[1].args[0]
        assert actual_rows == [{"date": "2026-01-27", "opportunities": 110.0}]

    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_empty_predictions_skips_upsert(self, mock_load_dotenv, caplog):
        client = build_upsert_mock_client()
        with caplog.at_level("WARNING", logger="database"):
            save_results_to_supabase({"predictions": []}, client=client)
        client.table.assert_not_called()
        assert "No predictions to save" in caplog.text

    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_missing_actuals_still_saves_predictions(self, mock_load_dotenv, caplog):
        client = build_upsert_mock_client()
        result = {
            "predictions": [
                {
                    "date": "2026-02-03",
                    "mean": 1.0,
                    "hdi_lower": 0.5,
                    "hdi_upper": 1.5,
                }
            ]
        }
        with caplog.at_level("WARNING", logger="database"):
            save_results_to_supabase(result, client=client)
        client.table.assert_called_once_with(PREDICTIONS_TABLE)
        assert "No actuals to save" in caplog.text

    @patch("database.load_dotenv")
    @patch.dict("os.environ", {"SUPABASE_URL": "", "SUPABASE_KEY": ""}, clear=False)
    def test_raises_value_error_when_credentials_missing(self, mock_load_dotenv):
        with pytest.raises(ValueError, match="Missing url or key"):
            save_results_to_supabase(
                {
                    "predictions": [
                        {
                            "date": "2026-02-03",
                            "mean": 1.0,
                            "hdi_lower": 0.5,
                            "hdi_upper": 1.5,
                        }
                    ]
                }
            )

    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_reraises_upsert_exception(self, mock_load_dotenv, caplog):
        client = MagicMock()
        client.table.return_value.upsert.return_value.execute.side_effect = Exception(
            "upsert failed"
        )
        with caplog.at_level("ERROR", logger="database"):
            with pytest.raises(Exception, match="upsert failed"):
                save_results_to_supabase(
                    {
                        "predictions": [
                            {
                                "date": "2026-02-03",
                                "mean": 1.0,
                                "hdi_lower": 0.5,
                                "hdi_upper": 1.5,
                            }
                        ]
                    },
                    client=client,
                )
        assert "Error happened during results upsert" in caplog.text


class TestResolveSupabaseCredentials:
    @patch("database._credentials_from_streamlit_secrets")
    def test_prefers_streamlit_secrets(self, mock_from_secrets):
        mock_from_secrets.return_value = (
            "https://secrets.example.co",
            "secrets-key",
        )
        url, key = resolve_supabase_credentials()
        assert url == "https://secrets.example.co"
        assert key == "secrets-key"

    @patch("database._load_credentials")
    @patch("database._credentials_from_streamlit_secrets", return_value=None)
    def test_falls_back_to_env(self, mock_from_secrets, mock_load_credentials):
        mock_load_credentials.return_value = ("http://env.example.co", "env-key")
        url, key = resolve_supabase_credentials()
        assert url == "http://env.example.co"
        assert key == "env-key"
        mock_load_credentials.assert_called_once()

    @patch(
        "database._load_credentials",
        side_effect=ValueError("Missing url or key in Environment variables."),
    )
    @patch("database._credentials_from_streamlit_secrets", return_value=None)
    def test_raises_when_both_missing(self, mock_from_secrets, mock_load_credentials):
        with pytest.raises(ValueError, match="Missing url or key"):
            resolve_supabase_credentials()


class TestLoadPredictionsAndActuals:
    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_load_predictions_parses_and_sorts(self, mock_load_dotenv):
        rows = [
            {
                "date": "2026-02-10",
                "mean": "105.0",
                "hdi_lower": "95.0",
                "hdi_upper": "115.0",
                "created_at": "2026-02-01T00:00:00+00:00",
            },
            {
                "date": "2026-02-03",
                "mean": "100.0",
                "hdi_lower": "90.0",
                "hdi_upper": "110.0",
                "created_at": "2026-02-01T00:00:00+00:00",
            },
        ]
        client = build_mock_client(rows)
        df = load_predictions_supabase(client=client)
        client.table.assert_called_once_with(PREDICTIONS_TABLE)
        assert list(df["date"]) == [
            pd.Timestamp("2026-02-03"),
            pd.Timestamp("2026-02-10"),
        ]
        assert df["mean"].tolist() == [100.0, 105.0]

    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_load_actuals_parses_and_sorts(self, mock_load_dotenv):
        rows = [
            {"date": "2026-01-27", "opportunities": "110.0"},
            {"date": "2026-01-20", "opportunities": "95.0"},
        ]
        client = build_mock_client(rows)
        df = load_actuals_supabase(client=client)
        client.table.assert_called_once_with(ACTUALS_TABLE)
        assert list(df["date"]) == [
            pd.Timestamp("2026-01-20"),
            pd.Timestamp("2026-01-27"),
        ]
        assert df["opportunities"].tolist() == [95.0, 110.0]

    @patch("database.load_dotenv")
    @patch.dict("os.environ", ENV, clear=False)
    def test_load_predictions_empty(self, mock_load_dotenv, caplog):
        client = build_mock_client([])
        with caplog.at_level("WARNING", logger="database"):
            df = load_predictions_supabase(client=client)
        assert df.empty
        assert "No predictions returned from Supabase" in caplog.text

    @patch("database.load_dotenv")
    @patch("database.create_client")
    @patch.dict("os.environ", ENV, clear=False)
    def test_load_predictions_uses_injected_client(
        self, mock_create_client, mock_load_dotenv
    ):
        client = build_mock_client([])
        load_predictions_supabase(client=client)
        mock_create_client.assert_not_called()

