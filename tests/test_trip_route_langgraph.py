import pytest

from backend.app.api.routes import trip
from backend.app.models.schemas import TripRequest
from backend.app.services.amap_langchain_service import (
    _parse_poi_item,
    format_pois_for_planner,
    _extract_pois,
    # ... 其他需要的函数
)

@pytest.mark.asyncio
async def test_plan_trip_returns_langgraph_trip_plan():
    request = TripRequest(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )

    response = await trip.plan_trip(request)

    assert response.success is True
    assert response.data.city == "北京"
    assert response.data.start_date == "2026-08-01"
    assert response.data.end_date == "2026-08-02"
    assert len(response.data.days) == 2
    assert response.data.days[0].attractions[0].name == "北京推荐景点1"
    assert response.data.days[1].attractions[0].name == "北京推荐景点2"