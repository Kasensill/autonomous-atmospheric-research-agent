"""网页解码损坏与天气数值损坏的接收边界；所有网络响应均为假对象。"""

import sys
from types import SimpleNamespace

import pytest

from atmospheric_research_agent.agent.tools import source_reader, weather


@pytest.mark.parametrize("content", ["\ufffd" * 100, "正文\x00\x01" * 30, {"unexpected": "object"}])
def test_invalid_extracted_html_returns_diagnostic_error(monkeypatch, content):
    fake = SimpleNamespace(
        fetch_response=lambda *a, **kw: SimpleNamespace(data=b"<html>fixture</html>", status=200, headers={}),
        extract=lambda *a, **kw: content,
        extract_metadata=lambda *a, **kw: SimpleNamespace(title="资料"),
    )
    monkeypatch.setitem(sys.modules, "trafilatura", fake)
    result = source_reader.read_public_source("https://example.org/fixture")
    assert result["status"] == "error"
    assert result.get("content", "") == ""
    assert result["diagnostics"]["downloaded_bytes"] > 0


@pytest.mark.parametrize("field,value", [
    ("temperature_2m", "乱码"), ("temperature_2m", True),
    ("temperature_2m", float("nan")), ("temperature_2m", float("inf")),
    ("relative_humidity_2m", -1), ("relative_humidity_2m", 101),
    ("cloud_cover", 101), ("wind_direction_10m", 361),
    ("wind_speed_10m", -1), ("precipitation", -1),
])
def test_weather_rejects_invalid_numeric_values(monkeypatch, field, value):
    current = {"time": "2026-10-25T10:00", "temperature_2m": 12.0, field: value}
    responses = [({"results": [{"name": "北京", "latitude": 39.9, "longitude": 116.4,
                                "timezone": "Asia/Shanghai"}]}, "https://example.org/geo"),
                 ({"current": current, "timezone": "Asia/Shanghai"}, "https://example.org/weather")]
    monkeypatch.setattr(weather, "_get_json", lambda *a: responses.pop(0))
    result = weather.query_current_weather("北京")
    assert result["status"] == "error"
    assert "evidence_text" not in result
    assert field in result["message"]
