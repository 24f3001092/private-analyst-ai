from private_analyst.analytics.anomalies import linear_trend_forecast


def test_forecast_is_labelled_hypothesis() -> None:
    out = linear_trend_forecast([1.0, 2.0, 3.0, 4.0], periods=2)
    assert out["status"] == "hypothesis"
    assert "unvalidated" in out["caveat"].lower()
    assert len(out["forecast"]) == 2


def test_forecast_insufficient_data() -> None:
    out = linear_trend_forecast([1.0, 2.0], periods=1)
    assert out["status"] == "insufficient_data"
