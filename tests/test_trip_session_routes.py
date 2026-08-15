import pytest

from backend.app.api.routes import trip
from backend.app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    TripPlan,
    TripRequest,
)
from backend.app.services.trip_session_service import reset_trip_session_service
from backend.app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    TripPlan,
    TripRequest,
    TripSessionCreateRequest,
)

def make_request() -> TripRequest:
    return TripRequest(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )


def make_plan() -> TripPlan:
    return TripPlan(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        days=[
            DayPlan(
                date="2026-08-01",
                day_index=0,
                description="第1天",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[
                    Attraction(
                        name="故宫博物院",
                        address="景山前街4号",
                        location=Location(longitude=116.397005, latitude=39.919278),
                        visit_duration=180,
                        description="历史文化景点",
                        category="历史文化",
                        ticket_price=60,
                    )
                ],
                meals=[
                    Meal(type="breakfast", name="早餐"),
                    Meal(type="lunch", name="午餐"),
                    Meal(type="dinner", name="晚餐"),
                ],
            )
        ],
        weather_info=[],
        overall_suggestions="建议提前预约。",
        budget=None,
    )


@pytest.mark.asyncio
async def test_create_trip_session_route_returns_session():
    reset_trip_session_service()

    response = await trip.create_trip_session(
    payload=TripSessionCreateRequest(
        request=make_request(),
        plan=make_plan(),
    )
)

    assert response.success is True
    assert response.data.id
    assert response.data.request.city == "北京"
    assert response.data.current_plan.city == "北京"


@pytest.mark.asyncio
async def test_get_trip_session_route_returns_existing_session():
    reset_trip_session_service()

    created = await trip.create_trip_session(
    payload=TripSessionCreateRequest(
        request=make_request(),
        plan=make_plan(),
    )
)

    fetched = await trip.get_trip_session(created.data.id)

    assert fetched.success is True
    assert fetched.data.id == created.data.id
    assert fetched.data.current_plan.city == "北京"


@pytest.mark.asyncio
async def test_get_trip_session_route_raises_for_missing_session():
    reset_trip_session_service()

    with pytest.raises(Exception) as exc_info:
        await trip.get_trip_session("missing-id")

    assert "Trip session not found" in str(exc_info.value)