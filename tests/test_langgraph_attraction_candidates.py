import pytest

from backend.app.agents import langgraph_trip_planner
from backend.app.models.schemas import Location, POIInfo


def make_poi(name: str, poi_id: str) -> POIInfo:
    return POIInfo(
        id=poi_id,
        name=name,
        type="景点",
        address=f"{name}地址",
        location=Location(longitude=120.0, latitude=36.0),
    )


@pytest.mark.asyncio
async def test_collect_unique_attraction_candidates_uses_more_keywords_when_primary_is_short(monkeypatch):
    async def fake_search_pois(keywords, city, citylimit=True, limit=20):
        if keywords == "景点":
            return [
                make_poi("五四广场", "poi-1"),
                make_poi("青岛奥帆海洋文化旅游区", "poi-2"),
                make_poi("青岛第三海水浴场", "poi-3"),
            ]
        if keywords == "热门景点":
            return [
                make_poi("五四广场", "poi-1"),
                make_poi("八大关风景区", "poi-4"),
                make_poi("栈桥", "poi-5"),
                make_poi("小鱼山公园", "poi-6"),
            ]
        return []

    monkeypatch.setattr(
        langgraph_trip_planner.amap_langchain_service,
        "search_pois",
        fake_search_pois,
    )

    candidates = await langgraph_trip_planner._collect_unique_attraction_candidates(
        city="青岛",
        primary_keywords="景点",
        required_count=6,
        limit=20,
    )

    assert [candidate.name for candidate in candidates] == [
        "五四广场",
        "青岛奥帆海洋文化旅游区",
        "青岛第三海水浴场",
        "八大关风景区",
        "栈桥",
        "小鱼山公园",
    ]
