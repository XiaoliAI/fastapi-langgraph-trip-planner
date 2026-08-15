import pytest

from backend.app.api.routes import trip


@pytest.mark.asyncio
async def test_health_check_uses_langgraph_planner(monkeypatch):
    class FakePlanner:
        pass

    monkeypatch.setattr(
        trip,
        "get_langgraph_trip_planner",
        lambda: FakePlanner(),
    )

    result = await trip.health_check()

    assert result["status"] == "healthy"
    assert result["service"] == "trip-planner"
    assert result["agent_name"] == "FakePlanner"
    assert result["tools_count"] == 0
