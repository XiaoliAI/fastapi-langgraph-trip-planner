import pytest

from backend.app.api.routes import trip
from backend.app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    TripChatRequest,
    TripPlan,
    TripRequest,
    TripSessionCreateRequest,
    TripChangeIntent,
)
from backend.app.services.trip_session_service import reset_trip_session_service


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


def make_plan(suggestion: str = "原始建议") -> TripPlan:
    return TripPlan(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        days=[
            DayPlan(
                date="2026-08-01",
                day_index=1,
                description="第一天行程",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[
                    Attraction(
                        name="故宫博物院",
                        address="景山前街4号",
                        location=Location(
                            longitude=116.397005,
                            latitude=39.919278,
                        ),
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
        overall_suggestions=suggestion,
        budget=None,
    )


@pytest.mark.asyncio
async def test_major_revision_chat_saves_pending_revision_summary(monkeypatch):
    reset_trip_session_service()

    def fake_classify_trip_change_intent(
        message,
        knowledge_text="",
        use_llm=False,
    ):
        assert message == "我想少走路，整体轻松一点"
        assert use_llm is True
        return TripChangeIntent(
            change_type="major_revision",
            summary="用户希望减少步行强度，调整行程节奏使其更轻松。",
            patch_operations=[],
            clarification_question=None,
        )

    monkeypatch.setattr(
        trip,
        "classify_trip_change_intent",
        fake_classify_trip_change_intent,
    )

    created = await trip.create_trip_session(
        payload=TripSessionCreateRequest(
            request=make_request(),
            plan=make_plan(),
        )
    )

    response = await trip.chat_with_trip_session(
        session_id=created.data.id,
        payload=TripChatRequest(message="我想少走路，整体轻松一点"),
    )

    assert response.success is True
    assert response.intent.change_type == "major_revision"
    assert response.data.pending_revision_summary
    assert "减少步行强度" in response.data.pending_revision_summary
    assert "确认" in response.data.messages[-1].content


@pytest.mark.asyncio
async def test_revise_trip_session_saves_old_plan_and_updates_current_plan(monkeypatch):
    reset_trip_session_service()

    revised_plan = make_plan(suggestion="新版少走路建议")

    class FakeRevisionPlanner:
        async def revise_trip(self, session, revision_summary):
            assert session.pending_revision_summary == "少走路，整体轻松一点"
            assert revision_summary == "少走路，整体轻松一点"
            return revised_plan

    monkeypatch.setattr(
        trip,
        "get_revision_trip_planner",
        lambda: FakeRevisionPlanner(),
    )

    created = await trip.create_trip_session(
        payload=TripSessionCreateRequest(
            request=make_request(),
            plan=make_plan(),
        )
    )

    service = trip.get_trip_session_service()
    service.update_pending_revision_summary(
        session_id=created.data.id,
        revision_summary="少走路，整体轻松一点",
    )

    response = await trip.revise_trip_session(session_id=created.data.id)

    assert response.success is True
    assert response.data.current_plan.overall_suggestions == "新版少走路建议"
    assert len(response.data.plan_versions) == 1
    assert response.data.plan_versions[0].overall_suggestions == "原始建议"
    assert response.data.pending_revision_summary is None


@pytest.mark.asyncio
async def test_revise_trip_session_keeps_old_plan_when_validation_fails(monkeypatch):
    reset_trip_session_service()

    invalid_plan = make_plan(suggestion="不合法的新计划")

    class FakeRevisionPlanner:
        async def revise_trip_result(self, session, revision_summary):
            return {
                "revised_plan": invalid_plan,
                "validation_errors": ["行程天数不匹配"],
            }

    monkeypatch.setattr(
        trip,
        "get_revision_trip_planner",
        lambda: FakeRevisionPlanner(),
    )

    created = await trip.create_trip_session(
        payload=TripSessionCreateRequest(
            request=make_request(),
            plan=make_plan(),
        )
    )

    service = trip.get_trip_session_service()
    service.update_pending_revision_summary(
        session_id=created.data.id,
        revision_summary="少走路，整体轻松一点",
    )

    response = await trip.revise_trip_session(session_id=created.data.id)

    assert response.success is False
    assert "行程天数不匹配" in response.message
    assert response.data.current_plan.overall_suggestions == "原始建议"
    assert response.data.plan_versions == []
    assert response.data.pending_revision_summary == "少走路，整体轻松一点"
