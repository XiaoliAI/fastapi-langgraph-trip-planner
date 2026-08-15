import pytest

from backend.app.models.schemas import Location
from backend.app.services import amap_langchain_service


@pytest.mark.asyncio
async def test_search_pois_uses_cache(monkeypatch):
    amap_langchain_service.clear_amap_langchain_cache()
    calls = []

    async def fake_invoke_tool(tool_name, arguments):
        calls.append((tool_name, arguments))
        return {
            "pois": [
                {
                    "id": "B001",
                    "name": "故宫博物院",
                    "type": "景点",
                    "address": "景山前街4号",
                    "location": "116.397005,39.919278",
                }
            ]
        }

    monkeypatch.setattr(
        amap_langchain_service,
        "_invoke_tool",
        fake_invoke_tool,
    )

    first = await amap_langchain_service.search_pois("历史文化", "北京", limit=1)
    second = await amap_langchain_service.search_pois("历史文化", "北京", limit=1)

    assert first == second
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_geocode_location_uses_cache(monkeypatch):
    amap_langchain_service.clear_amap_langchain_cache()
    calls = []

    async def fake_invoke_tool(tool_name, arguments):
        calls.append((tool_name, arguments))
        return {
            "return": [
                {
                    "location": "116.397005,39.919278",
                }
            ]
        }

    monkeypatch.setattr(
        amap_langchain_service,
        "_invoke_tool",
        fake_invoke_tool,
    )

    first = await amap_langchain_service.geocode_location("景山前街4号", "北京")
    second = await amap_langchain_service.geocode_location("景山前街4号", "北京")

    assert first == Location(longitude=116.397005, latitude=39.919278)
    assert second == first
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_get_weather_text_uses_cache(monkeypatch):
    amap_langchain_service.clear_amap_langchain_cache()
    calls = []

    async def fake_invoke_tool(tool_name, arguments):
        calls.append((tool_name, arguments))
        return {
            "forecasts": [
                {
                    "date": "2026-08-01",
                    "dayweather": "晴",
                    "nightweather": "多云",
                    "daytemp": "33",
                    "nighttemp": "25",
                    "daywind": "南",
                    "nightwind": "南",
                    "daypower": "1-3",
                    "nightpower": "1-3",
                }
            ]
        }

    monkeypatch.setattr(
        amap_langchain_service,
        "_invoke_tool",
        fake_invoke_tool,
    )

    first = await amap_langchain_service.get_weather_text("北京")
    second = await amap_langchain_service.get_weather_text("北京")

    assert "晴" in first
    assert second == first
    assert len(calls) == 1
