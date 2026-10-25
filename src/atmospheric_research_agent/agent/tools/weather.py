"""指定地区的当前天气查询工具（Open-Meteo）。"""

import json
from math import isfinite
from datetime import UTC, datetime
from zoneinfo import ZoneInfo
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain_core.tools import tool

from ..settings import WEATHER_REQUEST_TIMEOUT_SECONDS


GEOCODING_ENDPOINT = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
# Open-Meteo 使用 WMO Weather interpretation code；只保留 V1 当前会展示的中文描述。
WEATHER_CODE_LABELS = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "雾",
    48: "雾凇",
    51: "毛毛雨（弱）",
    53: "毛毛雨（中）",
    55: "毛毛雨（强）",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    80: "阵雨（弱）",
    81: "阵雨（中）",
    82: "阵雨（强）",
    95: "雷暴",
    96: "雷暴伴小冰雹",
    99: "雷暴伴大冰雹",
}


def _get_json(endpoint: str, parameters: dict[str, str | int | float]) -> tuple[dict[str, Any], str]:
    """请求公开 API，并同时返回实际请求 URL 供结果溯源。"""

    request_url = f"{endpoint}?{urlencode(parameters)}"
    request = Request(request_url, headers={"User-Agent": "AutonomousResearchAgent/0.1"})
    with urlopen(request, timeout=WEATHER_REQUEST_TIMEOUT_SECONDS) as response:  # noqa: S310 - endpoint 为本模块常量
        return json.loads(response.read().decode("utf-8")), request_url


def _resolve_location(location: str) -> tuple[dict[str, Any], str]:
    """使用地点名解析一个最匹配的坐标；找不到时明确报错。"""

    data, request_url = _get_json(
        GEOCODING_ENDPOINT,
        {"name": location, "count": 1, "language": "zh", "format": "json"},
    )
    results = data.get("results", [])
    if not results:
        raise ValueError(f"未找到地点“{location}”。请提供更具体的城市、地区或国家信息。")
    return results[0], request_url


def _weather_evidence_text(location: dict[str, Any], current: dict[str, Any], timezone: str) -> str:
    """把结构化当前条件压缩为可被 evidence 层引用的一段中性文字。"""

    weather_code = current.get("weather_code")
    condition = WEATHER_CODE_LABELS.get(weather_code, f"天气代码 {weather_code}")
    return (
        f"Open-Meteo 模型当前条件：{location['name']}（{location.get('country', '未标注国家')}，"
        f"{timezone}）在 {current.get('time', '未提供时间')} 的气温为 "
        f"{current.get('temperature_2m')}°C，体感温度 {current.get('apparent_temperature')}°C，"
        f"相对湿度 {current.get('relative_humidity_2m')}%，天气为{condition}，"
        f"降水量 {current.get('precipitation')} mm，10 米风速 {current.get('wind_speed_10m')} km/h，"
        f"云量 {current.get('cloud_cover', '未提供')}%，"
        f"10 米风向 {current.get('wind_direction_10m', '未提供')}°（气象风向）。"
    )


def query_current_weather(location: str) -> dict[str, Any]:
    """地点名 → 经纬度 → Open-Meteo 当前条件，并返回规范化结构。"""

    clean_location = location.strip()
    if not clean_location:
        return {
            "status": "error",
            "tool": "get_current_weather",
            "message": "地点不能为空。请提供具体城市或地区。",
        }

    try:
        resolved_location, geocoding_url = _resolve_location(clean_location)
        weather_data, weather_url = _get_json(
            FORECAST_ENDPOINT,
            {
                "latitude": resolved_location["latitude"],
                "longitude": resolved_location["longitude"],
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,"
                    "rain,showers,snowfall,weather_code,cloud_cover,wind_speed_10m,wind_direction_10m"
                ),
                "timezone": "auto",
            },
        )
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        return {
            "status": "error",
            "tool": "get_current_weather",
            "location": clean_location,
            "message": f"天气查询失败：{error}",
        }

    current = weather_data.get("current")
    if not current:
        return {
            "status": "error",
            "tool": "get_current_weather",
            "location": clean_location,
            "message": "天气服务没有返回当前条件。",
        }

    numeric_fields = (
        "temperature_2m", "apparent_temperature", "relative_humidity_2m", "precipitation",
        "rain", "showers", "snowfall", "weather_code", "cloud_cover", "wind_speed_10m", "wind_direction_10m",
    )
    invalid_fields = []
    if not isinstance(current, dict):
        invalid_fields.append("current")
    else:
        for field in numeric_fields:
            value = current.get(field)
            if value is None:  # 未提供的字段继续明确显示缺失，不把它补成零。
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                invalid_fields.append(field)
            elif field in {"relative_humidity_2m", "cloud_cover"} and not 0 <= value <= 100:
                invalid_fields.append(field)
            elif field == "wind_direction_10m" and not 0 <= value <= 360:
                invalid_fields.append(field)
            elif field in {"precipitation", "rain", "showers", "snowfall", "wind_speed_10m"} and value < 0:
                invalid_fields.append(field)
            elif field == "weather_code" and (value < 0 or int(value) != value):
                invalid_fields.append(field)
    if invalid_fields:
        return {"status": "error", "tool": "get_current_weather", "location": clean_location,
                "message": "天气服务返回无效的当前条件字段：" + ", ".join(invalid_fields),
                "diagnostics": {"invalid_fields": invalid_fields}}

    timezone = weather_data.get("timezone", resolved_location.get("timezone", "未标注时区"))
    weather_code = current.get("weather_code")
    location_data = {
        "requested_name": clean_location,
        "resolved_name": resolved_location["name"],
        "country": resolved_location.get("country"),
        "admin1": resolved_location.get("admin1"),
        "latitude": resolved_location["latitude"],
        "longitude": resolved_location["longitude"],
        "timezone": timezone,
    }
    normalized_current = {
        "time": current.get("time"),
        "temperature_c": current.get("temperature_2m"),
        "apparent_temperature_c": current.get("apparent_temperature"),
        "relative_humidity_percent": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation"),
        "rain_mm": current.get("rain"),
        "showers_mm": current.get("showers"),
        "snowfall_cm": current.get("snowfall"),
        "cloud_cover_percent": current.get("cloud_cover"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "wind_direction_degrees": current.get("wind_direction_10m"),
        "weather_code": weather_code,
        "weather_description": WEATHER_CODE_LABELS.get(weather_code, f"天气代码 {weather_code}"),
    }
    evidence_text = _weather_evidence_text(resolved_location, current, timezone)
    try:
        valid_time = datetime.fromisoformat(current["time"])
        if valid_time.tzinfo is None:
            valid_time = valid_time.replace(tzinfo=ZoneInfo(timezone))
        valid_at = valid_time.isoformat()
    except (KeyError, ValueError, TypeError):
        valid_at = None
    return {
        "status": "success",
        "tool": "get_current_weather",
        "data_kind": "model_current_conditions",
        "provider": "Open-Meteo",
        "location": location_data,
        "observed_at": current.get("time"),
        "current": normalized_current,
        "evidence_text": evidence_text,
        "valid_at": valid_at,
        "fetched_at": datetime.now(UTC).isoformat(),
        "source": {
            "display_name": f"Open-Meteo 当前条件：{resolved_location['name']}",
            "location": weather_url,
            "type": "weather",
            "organization": "Open-Meteo",
            "geocoding_location": geocoding_url,
        },
    }


@tool
def get_current_weather(location: str) -> dict[str, Any]:
    """查询指定地点的当前天气观测或预报。

    仅在研究问题需要具体地点的当前天气条件时使用。地点不明确时，应先向用户
    澄清，而不是猜测位置。结果是天气模型的当前条件，不是地面站实测声明。
    """

    return query_current_weather(location)
