#测试mcp服务的搜索工具封装解析情况
import json
import pytest
from backend.app.services.amap_langchain_service import _extract_pois
from backend.app.services.amap_langchain_service import _parse_poi_item, format_pois_for_planner
from backend.app.services.amap_langchain_service import format_weather_for_planner
def test_extract_pois_from_langchain_tool_message_list():
    result = [
        {
            "content": json.dumps(
                {
                    "pois": [
                        {
                            "id": "B0001",
                            "name": "中国国家博物馆",
                            "address": "东长安街16号",
                            "typecode": "140100",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "id": "lc_test",
        }
    ]

    pois = _extract_pois(result)

    assert len(pois) == 1
    assert pois[0]["name"] == "中国国家博物馆"
    assert pois[0]["address"] == "东长安街16号"


def test_extract_pois_returns_empty_list_for_invalid_result():
    assert _extract_pois("not json") == []
    assert _extract_pois({"content": "not json"}) == []
    assert _extract_pois([]) == []

    from backend.app.services.amap_langchain_service import (
    _parse_poi_item,
    format_pois_for_planner,
)
from backend.app.models.schemas import Location, POIInfo


def test_parse_poi_item_with_location_string():
    item = {
        "id": "B0001",
        "name": "中国国家博物馆",
        "address": "东长安街16号",
        "type": "博物馆",
        "location": "116.397128,39.916527",
    }

    poi = _parse_poi_item(item)

    assert poi is not None
    assert poi.id == "B0001"
    assert poi.name == "中国国家博物馆"
    assert poi.address == "东长安街16号"
    assert poi.location.longitude == 116.397128
    assert poi.location.latitude == 39.916527


def test_parse_poi_item_returns_none_without_location():
    item = {
        "id": "B0001",
        "name": "中国国家博物馆",
        "address": "东长安街16号",
    }

    poi = _parse_poi_item(item)

    assert poi is None


def test_format_pois_for_planner():
    pois = [
        POIInfo(
            id="B0001",
            name="中国国家博物馆",
            type="博物馆",
            address="东长安街16号",
            location=Location(longitude=116.397128, latitude=39.916527),
            tel=None,
        )
    ]

    text = format_pois_for_planner(pois, city="北京", keywords="历史文化")

    assert "中国国家博物馆" in text
    assert "东长安街16号" in text
    assert "116.397128,39.916527" in text
    assert "B0001" in text
    import pytest

from backend.app.services import amap_langchain_service


@pytest.mark.asyncio
async def test_search_pois_geocodes_items_without_location(monkeypatch):
    async def fake_find_amap_tool(tool_name: str):
        class FakeTool:
            async def ainvoke(self, args):
                if tool_name == "maps_text_search":
                    return {
                        "pois": [
                            {
                                "id": "B0001",
                                "name": "中国国家博物馆",
                                "address": "东长安街16号",
                                "typecode": "140100",
                            }
                        ]
                    }

                if tool_name == "maps_geo":
                    return {
                        "geocodes": [
                            {
                                "location": "116.397128,39.916527"
                            }
                        ]
                    }

                raise AssertionError(f"Unexpected tool: {tool_name}")

        return FakeTool()

    monkeypatch.setattr(
        amap_langchain_service,
        "find_amap_tool",
        fake_find_amap_tool,
    )

    pois = await amap_langchain_service.search_pois(
        keywords="历史文化",
        city="北京",
        citylimit=True,
        limit=5,
    )

    assert len(pois) == 1
    assert pois[0].name == "中国国家博物馆"
    assert pois[0].location.longitude == 116.397128
    assert pois[0].location.latitude == 39.916527

def test_extract_pois_from_langchain_text_field():
    result = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "pois": [
                        {
                            "id": "B000A8UIN8",
                            "name": "故宫博物院",
                            "address": "景山前街4号",
                            "typecode": "110201|140100",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "id": "lc_test",
        }
    ]

    pois = _extract_pois(result)

    assert len(pois) == 1
    assert pois[0]["name"] == "故宫博物院"


from backend.app.services.amap_langchain_service import _extract_json_like
def test_extract_json_like_parses_langchain_text_field():
    result = [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "return": [
                        {
                            "location": "116.397029,39.917839",
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            "id": "lc_test",
        }
    ]

    data = _extract_json_like(result)

    assert data["return"][0]["location"] == "116.397029,39.917839"

    import pytest

from backend.app.services import amap_langchain_service


@pytest.mark.asyncio
async def test_geocode_location_parses_return_field(monkeypatch):
    async def fake_find_amap_tool(tool_name: str):
        assert tool_name == "maps_geo"

        class FakeTool:
            async def ainvoke(self, args):
                assert args["address"] == "故宫博物院"
                assert args["city"] == "北京"
                return [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "return": [
                                    {
                                        "location": "116.397029,39.917839",
                                    }
                                ]
                            },
                            ensure_ascii=False,
                        ),
                    }
                ]

        return FakeTool()

    monkeypatch.setattr(
        amap_langchain_service,
        "find_amap_tool",
        fake_find_amap_tool,
    )

    location = await amap_langchain_service.geocode_location(
        "故宫博物院",
        city="北京",
    )

    assert location is not None
    assert location.longitude == 116.397029
    assert location.latitude == 39.917839

def test_format_weather_for_planner_from_forecasts():
    data = {
        "city": "北京市",
        "forecasts": [
            {
                "date": "2026-07-27",
                "week": "1",
                "dayweather": "多云",
                "nightweather": "多云",
                "daytemp": "33",
                "nighttemp": "24",
                "daywind": "南",
                "nightwind": "南",
                "daypower": "1-3",
                "nightpower": "1-3",
            },
            {
                "date": "2026-07-28",
                "week": "2",
                "dayweather": "晴",
                "nightweather": "晴",
                "daytemp": "34",
                "nighttemp": "25",
                "daywind": "南",
                "nightwind": "南",
                "daypower": "1-3",
                "nightpower": "1-3",
            },
        ],
    }

    text = format_weather_for_planner(data, city="北京")

    assert "2026-07-27" in text
    assert "多云" in text
    assert "33℃" in text
    assert "24℃" in text
    assert "南风" in text
    assert "1-3级" in text
    assert "2026-07-28" in text
    assert "晴" in text

def test_format_weather_for_planner_returns_fallback_for_empty_data():
    text = format_weather_for_planner({}, city="北京")

    assert text == "未获取到 北京 的可解析天气信息。"