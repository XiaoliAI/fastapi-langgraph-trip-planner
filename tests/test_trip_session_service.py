import pytest

from backend.app.models.schemas import (
    Attraction,
    ChatMessage,
    DayPlan,
    Location,
    Meal,
    TripPlan,
    TripRequest,
)
from backend.app.services.trip_session_service import TripSessionService
from backend.app.services.trip_session_service import (
    get_trip_session_service,
    reset_trip_session_service,
)
from backend.app.models.schemas import PendingPatchIntent

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


def test_create_session_stores_request_and_plan():
    service = TripSessionService()

    session = service.create_session(make_request(), make_plan())

    assert session.id
    assert session.request.city == "北京"
    assert session.current_plan.city == "北京"
    assert session.messages == []
    assert session.status == "draft"


def test_get_session_returns_existing_session():
    service = TripSessionService()
    created = service.create_session(make_request(), make_plan())

    fetched = service.get_session(created.id)

    assert fetched.id == created.id
    assert fetched.current_plan.city == "北京"


def test_get_session_raises_for_missing_session():
    service = TripSessionService()

    with pytest.raises(ValueError, match="Trip session not found"):
        service.get_session("missing-id")


def test_append_message_adds_message_to_session():
    service = TripSessionService()
    session = service.create_session(make_request(), make_plan())

    message = ChatMessage(role="user", content="我想少走路")
    updated = service.append_message(session.id, message)

    assert len(updated.messages) == 1
    assert updated.messages[0].role == "user"
    assert updated.messages[0].content == "我想少走路"


def test_update_plan_replaces_current_plan():
    service = TripSessionService()
    session = service.create_session(make_request(), make_plan())

    new_plan = make_plan()
    new_plan.overall_suggestions = "新版建议"

    updated = service.update_plan(session.id, new_plan)

    assert updated.current_plan.overall_suggestions == "新版建议"

def test_create_session_has_no_pending_patch_intent():
    reset_trip_session_service()
    service = get_trip_session_service()

    session = service.create_session(
        request=make_request(),
        plan=make_plan(),
    )

    assert session.pending_patch_intent is None


def test_update_pending_patch_intent():
    reset_trip_session_service()
    service = get_trip_session_service()

    session = service.create_session(
        request=make_request(),
        plan=make_plan(),
    )

    pending = PendingPatchIntent(
        operation="replace_attraction",
        known_fields={
            "new_target_preference": "适合拍照的地方",
        },
        missing_fields=["day_index", "old_attraction_name"],
        clarification_question="你想修改第几天的哪个景点？",
    )

    updated = service.update_pending_patch_intent(
        session_id=session.id,
        pending_patch_intent=pending,
    )

    assert updated.pending_patch_intent is not None
    assert updated.pending_patch_intent.operation == "replace_attraction"
    assert updated.pending_patch_intent.known_fields["new_target_preference"] == "适合拍照的地方"
    assert updated.pending_patch_intent.missing_fields == ["day_index", "old_attraction_name"]


def test_clear_pending_patch_intent():
    reset_trip_session_service()
    service = get_trip_session_service()

    session = service.create_session(
        request=make_request(),
        plan=make_plan(),
    )

    pending = PendingPatchIntent(
        operation="replace_attraction",
        known_fields={
            "new_target_preference": "适合拍照的地方",
        },
        missing_fields=["day_index", "old_attraction_name"],
        clarification_question="你想修改第几天的哪个景点？",
    )

    service.update_pending_patch_intent(
        session_id=session.id,
        pending_patch_intent=pending,
    )

    updated = service.clear_pending_patch_intent(session.id)

    assert updated.pending_patch_intent is None