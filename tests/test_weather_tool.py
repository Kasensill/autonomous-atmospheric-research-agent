from atmospheric_research_agent.agent.tools import weather


def test_query_current_weather_normalizes_open_meteo_response(monkeypatch) -> None:
    responses = [
        (
            {
                "results": [
                    {
                        "name": "北京",
                        "country": "中国",
                        "admin1": "北京市",
                        "latitude": 39.9042,
                        "longitude": 116.4074,
                        "timezone": "Asia/Shanghai",
                    }
                ]
            },
            "https://example.test/geocoding",
        ),
        (
            {
                "timezone": "Asia/Shanghai",
                "current": {
                    "time": "2026-09-16T10:00",
                    "temperature_2m": 25.0,
                    "apparent_temperature": 26.0,
                    "relative_humidity_2m": 65,
                    "precipitation": 0.0,
                    "rain": 0.0,
                    "showers": 0.0,
                    "snowfall": 0.0,
                    "weather_code": 1,
                    "cloud_cover": 20,
                    "wind_speed_10m": 12.0,
                    "wind_direction_10m": 90,
                },
            },
            "https://example.test/weather",
        ),
    ]
    monkeypatch.setattr(weather, "_get_json", lambda *args: responses.pop(0))

    result = weather.query_current_weather("北京")

    assert result["status"] == "success"
    assert result["data_kind"] == "model_current_conditions"
    assert result["location"]["resolved_name"] == "北京"
    assert result["current"]["weather_description"] == "大部晴朗"
    assert result["source"]["type"] == "weather"
    assert "25.0°C" in result["evidence_text"]


def test_query_current_weather_rejects_blank_location() -> None:
    result = weather.query_current_weather("  ")

    assert result["status"] == "error"
    assert "地点不能为空" in result["message"]
