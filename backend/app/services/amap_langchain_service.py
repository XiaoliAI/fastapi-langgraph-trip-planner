"""Amap LangChain MCP helpers."""

import asyncio
import json
import logging
import time
from typing import Any, Optional

from ..models.schemas import Location, POIInfo
from .mcp_tools import find_amap_tool


MCP_CALL_TIMEOUT_SECONDS = 15.0
POI_CACHE_TTL_SECONDS = 30 * 60
GEOCODE_CACHE_TTL_SECONDS = 30 * 60
WEATHER_CACHE_TTL_SECONDS = 10 * 60
logger = logging.getLogger(__name__)

_cache: dict[tuple[Any, ...], tuple[float, Any]] = {}
_CACHE_MISS = object()


def _get_cache_value(key: tuple[Any, ...]) -> Any:
    cached = _cache.get(key)
    if cached is None:
        return _CACHE_MISS

    expires_at, value = cached
    if expires_at < time.time():
        _cache.pop(key, None)
        return _CACHE_MISS

    return value


def _set_cache_value(key: tuple[Any, ...], value: Any, ttl_seconds: float) -> Any:
    _cache[key] = (time.time() + ttl_seconds, value)
    return value


def clear_amap_langchain_cache():
    _cache.clear()


async def _invoke_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    logger.info("Invoking Amap MCP tool: %s", tool_name)
    tool = await find_amap_tool(tool_name)
    return await asyncio.wait_for(
        tool.ainvoke(arguments),
        timeout=MCP_CALL_TIMEOUT_SECONDS,
    )


async def search_pois(
    keywords: str,
    city: str,
    citylimit: bool = True,
    limit: int = 10,
) -> list[POIInfo]:
    cache_key = ("search_pois", keywords, city, citylimit, limit)
    cached_pois = _get_cache_value(cache_key)
    if cached_pois is not _CACHE_MISS:
        logger.info("Amap POI cache hit: keywords=%s city=%s", keywords, city)
        return cached_pois

    try:
        result = await _invoke_tool(
            "maps_text_search",
            {
                "keywords": keywords,
                "city": city,
                "citylimit": "true" if citylimit else "false",
            },
        )
    except Exception:
        logger.warning(
            "Amap POI search failed or timed out: keywords=%s city=%s",
            keywords,
            city,
            exc_info=True,
        )
        return []

    raw_pois = _extract_pois(result)
    logger.info(
        "Amap POI search returned raw results: keywords=%s city=%s count=%s",
        keywords,
        city,
        len(raw_pois),
    )

    pois = await _parse_poi_items_with_concurrent_geocode(raw_pois[:limit], city)

    return _set_cache_value(cache_key, pois, POI_CACHE_TTL_SECONDS)


async def _parse_poi_items_with_concurrent_geocode(
    raw_items: list[dict[str, Any]],
    city: str,
) -> list[POIInfo]:
    parsed_pois: list[Optional[POIInfo]] = []
    geocode_indexes: list[int] = []
    geocode_tasks = []

    for item in raw_items:
        poi = _parse_poi_item(item)
        parsed_pois.append(poi)
        if poi is None:
            geocode_indexes.append(len(parsed_pois) - 1)
            geocode_tasks.append(_parse_poi_item_with_geocode(item, city))

    if geocode_tasks:
        geocoded_pois = await asyncio.gather(*geocode_tasks, return_exceptions=True)
        for index, result in zip(geocode_indexes, geocoded_pois):
            if isinstance(result, Exception):
                parsed_pois[index] = None
            else:
                parsed_pois[index] = result

    return [poi for poi in parsed_pois if poi is not None]


async def search_pois_text(
    keywords: str,
    city: str,
    citylimit: bool = True,
    limit: int = 10,
) -> str:
    pois = await search_pois(
        keywords=keywords,
        city=city,
        citylimit=citylimit,
        limit=limit,
    )
    return format_pois_for_planner(pois, city=city, keywords=keywords)


def format_pois_for_planner(
    pois: list[POIInfo],
    city: str = "",
    keywords: str = "",
) -> str:
    if not pois:
        if city and keywords:
            return f"未搜索到 {city} 与 {keywords} 相关的 POI。"
        return "未搜索到相关 POI。"

    lines = []
    for index, poi in enumerate(pois, start=1):
        lines.append(
            f"{index}. {poi.name} | 地址: {poi.address} | "
            f"坐标: {poi.location.longitude},{poi.location.latitude} | "
            f"POI ID: {poi.id} | 类型: {poi.type}"
        )
    return "\n".join(lines)


def _extract_pois(result: Any) -> list[dict[str, Any]]:
    if isinstance(result, list):
        for item in result:
            pois = _extract_pois(item)
            if pois:
                return pois
        return []

    if isinstance(result, dict):
        if "pois" in result and isinstance(result["pois"], list):
            return result["pois"]

        content = result.get("content")
        if content:
            return _extract_pois(content)

        text = result.get("text")
        if text:
            return _extract_pois(text)

        return []

    if isinstance(result, str):
        try:
            data = json.loads(result)
            return _extract_pois(data)
        except json.JSONDecodeError:
            return []

    return []


def _parse_poi_item(item: dict[str, Any]) -> Optional[POIInfo]:
    if not isinstance(item, dict):
        return None

    location = _parse_location(item.get("location"))
    if location is None:
        return None

    return POIInfo(
        id=str(item.get("id") or item.get("poi_id") or ""),
        name=str(item.get("name") or ""),
        type=str(item.get("type") or item.get("typecode") or ""),
        address=_string_value(item.get("address")),
        location=location,
        tel=_optional_string_value(item.get("tel")),
    )


def _parse_location(value: Any) -> Optional[Location]:
    if isinstance(value, dict):
        longitude = value.get("longitude") or value.get("lng")
        latitude = value.get("latitude") or value.get("lat")
    elif isinstance(value, str) and "," in value:
        longitude, latitude = value.split(",", 1)
    else:
        return None

    try:
        return Location(longitude=float(longitude), latitude=float(latitude))
    except (TypeError, ValueError):
        return None


def _string_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    if value is None:
        return ""
    return str(value)


def _optional_string_value(value: Any) -> Optional[str]:
    text = _string_value(value)
    return text or None


async def geocode_location(address: str, city: str = "") -> Optional[Location]:
    if not address:
        return None

    cache_key = ("geocode", address, city)
    cached_location = _get_cache_value(cache_key)
    if cached_location is not _CACHE_MISS:
        logger.info("Amap geocode cache hit: address=%s city=%s", address, city)
        return cached_location

    args = {"address": address}
    if city:
        args["city"] = city

    try:
        result = await _invoke_tool("maps_geo", args)
    except Exception:
        logger.warning(
            "Amap geocode failed or timed out: address=%s city=%s",
            address,
            city,
            exc_info=True,
        )
        return None

    data = _extract_json_like(result)
    if not isinstance(data, dict):
        return _set_cache_value(cache_key, None, GEOCODE_CACHE_TTL_SECONDS)

    geocodes = data.get("geocodes") or data.get("return")
    if not isinstance(geocodes, list) or not geocodes:
        return _set_cache_value(cache_key, None, GEOCODE_CACHE_TTL_SECONDS)

    first = geocodes[0]
    if not isinstance(first, dict):
        return _set_cache_value(cache_key, None, GEOCODE_CACHE_TTL_SECONDS)

    return _set_cache_value(
        cache_key,
        _parse_location(first.get("location")),
        GEOCODE_CACHE_TTL_SECONDS,
    )


def _extract_json_like(result: Any) -> Any:
    if isinstance(result, list):
        for item in result:
            data = _extract_json_like(item)
            if data is not None:
                return data
        return None

    if isinstance(result, dict):
        content = result.get("content")
        if content is not None:
            return _extract_json_like(content)

        text = result.get("text")
        if text is not None:
            return _extract_json_like(text)

        return result

    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return None

    return None


async def _parse_poi_item_with_geocode(
    item: dict[str, Any],
    city: str,
) -> Optional[POIInfo]:
    if not isinstance(item, dict):
        return None

    name = str(item.get("name") or "")
    address = _string_value(item.get("address"))
    query_address = address or name

    location = await geocode_location(query_address, city=city)
    if location is None:
        return None

    return POIInfo(
        id=str(item.get("id") or item.get("poi_id") or ""),
        name=name,
        type=str(item.get("type") or item.get("typecode") or ""),
        address=address,
        location=location,
        tel=_optional_string_value(item.get("tel")),
    )


async def get_weather_text(city: str) -> str:
    cache_key = ("weather", city)
    cached_weather = _get_cache_value(cache_key)
    if cached_weather is not _CACHE_MISS:
        logger.info("Amap weather cache hit: city=%s", city)
        return cached_weather

    try:
        result = await _invoke_tool("maps_weather", {"city": city})
    except Exception:
        logger.warning("Amap weather failed or timed out: city=%s", city, exc_info=True)
        return f"未获取到 {city} 的天气信息。"

    data = _extract_json_like(result)
    if not isinstance(data, dict):
        return _set_cache_value(
            cache_key,
            f"未获取到 {city} 的天气信息。",
            WEATHER_CACHE_TTL_SECONDS,
        )

    return _set_cache_value(
        cache_key,
        format_weather_for_planner(data, city=city),
        WEATHER_CACHE_TTL_SECONDS,
    )


def format_weather_for_planner(data: dict[str, Any], city: str = "") -> str:
    forecasts = data.get("forecasts")

    if not isinstance(forecasts, list) or not forecasts:
        if city:
            return f"未获取到 {city} 的可解析天气信息。"
        return "未获取到可解析天气信息。"

    lines = []
    for index, item in enumerate(forecasts, start=1):
        if not isinstance(item, dict):
            continue

        date = item.get("date", "")
        day_weather = item.get("dayweather", "")
        night_weather = item.get("nightweather", "")
        day_temp = item.get("daytemp", "")
        night_temp = item.get("nighttemp", "")
        day_wind = item.get("daywind", "")
        night_wind = item.get("nightwind", "")
        day_power = item.get("daypower", "")
        night_power = item.get("nightpower", "")

        lines.append(
            f"{index}. 日期: {date} | 白天: {day_weather} {day_temp}℃ {day_wind}风 {day_power}级 | "
            f"夜间: {night_weather} {night_temp}℃ {night_wind}风 {night_power}级"
        )

    if not lines:
        if city:
            return f"未获取到 {city} 的可解析天气信息。"
        return "未获取到可解析天气信息。"

    return "\n".join(lines)
