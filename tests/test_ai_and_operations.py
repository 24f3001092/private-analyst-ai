from __future__ import annotations

import json
from io import BytesIO

import pytest

from private_analyst.ai.client import LocalModelClient
from private_analyst.ai.explainer import explain_results
from private_analyst.analytics.operations import classify_intent, run_operation
from private_analyst.config import AIConfig, Limits
from private_analyst.exceptions import ModelResponseError, ModelUnavailableError
from private_analyst.ingestion.loader import load_tabular


def test_invalid_intent_rejected(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    result = run_operation(ingest, "hack the database", intent="drop_table")
    assert result.status == "rejected"


def test_summarize_uses_engine_totals(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    result = run_operation(ingest, "Calculate the total and average of units.")
    units = next(c for c in result.calculated["columns"] if c["name"] == "units")
    assert units["sum"] == 50
    assert units["mean"] == 10


def test_quality_intent(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    assert classify_intent("Identify missing values") == "quality"
    result = run_operation(ingest, "Identify missing values and inconsistent records.")
    assert result.calculated["missing_by_column"]["status"] == 1


def test_model_unavailable_falls_back(sales_csv_bytes, limits: Limits, monkeypatch: pytest.MonkeyPatch) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    result = run_operation(ingest, "Summarize the dataset.")
    cfg = AIConfig(host="127.0.0.1", port=9, model="missing")

    def boom(*_args, **_kwargs):
        raise ModelUnavailableError("The local model runtime did not respond.")

    monkeypatch.setattr(LocalModelClient, "status", lambda self: type("S", (), {
        "available": True, "detail": "forced", "runtime": "ollama", "host": "127.0.0.1", "model": "x"
    })())
    monkeypatch.setattr(LocalModelClient, "generate", boom)
    explanation = explain_results(result, "Summarize the dataset.", cfg)
    assert explanation["source"] == "deterministic_fallback"
    assert "50" in explanation["text"] or "calculated" in explanation["text"].lower()


def test_malformed_model_json(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AIConfig()
    client = LocalModelClient(cfg)

    class FakeResp:
        def read(self):
            return b"not-json"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResp())
    with pytest.raises(ModelResponseError):
        client.generate("hello")


def test_empty_model_response(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = AIConfig()
    client = LocalModelClient(cfg)

    class FakeResp:
        def read(self):
            return json.dumps({"response": "   "}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResp())
    with pytest.raises(ModelResponseError):
        client.generate("hello")


def test_non_local_host_rejected() -> None:
    with pytest.raises(ModelUnavailableError):
        LocalModelClient(AIConfig(host="example.com"))


def test_classify_intent_routing(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    assert classify_intent("Can you classify customers?") == "classify"
    result = run_operation(ingest, "Classify status based on features")
    # sales_known has only 5 rows, so classification should report insufficient_data safely
    assert result.intent == "classify"
    assert result.status == "insufficient_data"
    assert len(result.hypotheses) > 0


def test_regression_intent_routing(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    assert classify_intent("Run a regression on revenue") == "regression"
    result = run_operation(ingest, "Run a regression on revenue")
    assert result.intent == "regression"
    assert result.status == "insufficient_data"
    assert len(result.hypotheses) > 0


def test_cluster_intent_routing(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    assert classify_intent("Cluster these data points") == "cluster"
    result = run_operation(ingest, "Cluster the records")
    assert result.intent == "cluster"
    assert result.status == "hypothesis"
    assert "cluster_centers" in result.calculated


def test_frequency_intent_routing(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    assert classify_intent("Show frequency breakdown of status") == "frequency"
    result = run_operation(ingest, "Show frequency breakdown of status")
    assert result.intent == "frequency"
    assert result.status == "ok"
    assert "frequency_table" in result.calculated
    assert result.chart_spec is not None
    assert result.chart_spec["type"] == "category_bar"


def test_crosstab_intent_routing(sales_csv_bytes, limits: Limits) -> None:
    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    assert classify_intent("Show crosstab of region vs product") == "crosstab"
    result = run_operation(ingest, "Show crosstab of region vs product")
    assert result.intent == "crosstab"
    assert result.status == "ok"
    assert "crosstab" in result.calculated
    assert result.chart_spec is not None
    assert result.chart_spec["type"] == "crosstab_heatmap"


def test_deterministic_explanation_structure(sales_csv_bytes, limits: Limits) -> None:
    from private_analyst.ai.explainer import deterministic_explanation

    ingest = load_tabular(sales_csv_bytes, "sales_known.csv", limits)
    result = run_operation(ingest, "Summarize the dataset.")
    text = deterministic_explanation(result, "Summarize the dataset.")
    assert "### Analysis Summary: Summarize" in text
    assert "Observed Facts" in text
    assert "Calculated Metrics & Findings" in text
    assert "50" in text
    assert '{"row_count":' not in text


def test_local_model_status_diagnostics() -> None:
    from private_analyst.ai.client import ModelStatus

    # Simulated offline status
    status = ModelStatus(False, "ollama", "127.0.0.1", "llama3.2", "URLError", 11434)
    assert not status.is_ready
    assert "Offline" in status.friendly_status
    assert len(status.troubleshooting_steps) > 0
    assert any("ollama serve" in step for step in status.troubleshooting_steps)

    # Simulated online ready status
    online = ModelStatus(True, "ollama", "127.0.0.1", "llama3.2", "runtime_reachable", 11434)
    assert online.is_ready
    assert "Online & Ready" in online.friendly_status
    assert len(online.troubleshooting_steps) == 0

