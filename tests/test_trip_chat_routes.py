import pytest

from backend.app.api.routes import trip
from backend.app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    PendingPatchIntent,
    POIInfo,
    TripChatRequest,
    TripPlan,
    TripRequest,
    TripSessionCreateRequest,
    TripChangeIntent,
)
from backend.app.services.trip_session_service import reset_trip_session_service
from backend.app.services.trip_plan_patch_service import PatchBuildResult


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
                    ),
                    Attraction(
                        name="天坛公园",
                        address="天坛东路甲1号",
                        location=Location(
                            longitude=116.410886,
                            latitude=39.881949,
                        ),
                        visit_duration=120,
                        description="历史文化景点",
                        category="历史文化",
                        ticket_price=30,
                    ),
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
async def test_chat_route_appends_user_and_assistant_messages(monkeypatch):
    reset_trip_session_service()

    def fake_classify_trip_change_intent(
        message,
        knowledge_text="",
        use_llm=False,
    ):
        return TripChangeIntent(
            change_type="major_revision",
            summary="用户希望减少步行强度",
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
        payload=TripChatRequest(message="我想少走路"),
    )

    assert response.success is True
    assert response.data.id == created.data.id
    assert len(response.data.messages) == 2

    assert response.data.messages[0].role == "user"
    assert response.data.messages[0].content == "我想少走路"

    assert response.data.messages[1].role == "assistant"
    assert "整体调整需求" in response.data.messages[1].content

    assert response.intent is not None
    assert response.intent.change_type == "major_revision"
    assert response.intent.summary == "用户希望减少步行强度"


@pytest.mark.asyncio
async def test_chat_route_raises_for_missing_session():
    reset_trip_session_service()

    with pytest.raises(Exception) as exc_info:
        await trip.chat_with_trip_session(
            session_id="missing-id",
            payload=TripChatRequest(message="我想少走路"),
        )

    assert "Trip session not found" in str(exc_info.value)


@pytest.mark.asyncio
async def test_chat_route_applies_remove_attraction_patch(monkeypatch):
    reset_trip_session_service()

    async def fake_build_patch_result_from_message(message, plan):
        return PatchBuildResult(
            operations=[
                {
                    "operation": "remove_attraction",
                    "day_index": 1,
                    "attraction_name": "故宫博物院",
                }
            ],
            pending_patch_intent=None,
        )

    monkeypatch.setattr(
        trip,
        "build_patch_result_from_message",
        fake_build_patch_result_from_message,
    )

    created = await trip.create_trip_session(
        payload=TripSessionCreateRequest(
            request=make_request(),
            plan=make_plan(),
        )
    )

    response = await trip.chat_with_trip_session(
        session_id=created.data.id,
        payload=TripChatRequest(message="删除第一天的故宫"),
    )

    assert response.success is True
    assert response.intent is not None
    assert response.intent.change_type == "small_change"
    assert response.intent.patch_operations == [
        {
            "operation": "remove_attraction",
            "day_index": 1,
            "attraction_name": "故宫博物院",
        }
    ]

    day_one_names = [
        attraction.name
        for attraction in response.data.current_plan.days[0].attractions
    ]

    assert "故宫博物院" not in day_one_names
    assert "天坛公园" in day_one_names
    assert response.data.messages[-1].role == "assistant"
    assert "已完成局部修改" in response.data.messages[-1].content


@pytest.mark.asyncio
async def test_chat_route_applies_replace_attraction_patch(monkeypatch):
    reset_trip_session_service()

    async def fake_build_patch_result_from_message(message, plan):
        return PatchBuildResult(
            operations=[
                {
                    "operation": "replace_attraction",
                    "day_index": 1,
                    "old_attraction_name": "故宫博物院",
                    "new_attraction_name": "天坛",
                }
            ],
            pending_patch_intent=None,
        )
    async def fake_apply_trip_patch_operations(plan, operations):
        from backend.app.services.trip_plan_patch_service import (
            apply_trip_patch_operations,
        )

        async def fake_search_pois(keywords: str, city: str, limit: int = 5):
            return [
                POIInfo(
                    id="poi-tiantan",
                    name="天坛公园",
                    type="风景名胜",
                    address="北京市东城区天坛东路甲1号",
                    location=Location(
                        longitude=116.410886,
                        latitude=39.881949,
                    ),
                )
            ]

        return await apply_trip_patch_operations(
            plan=plan,
            operations=operations,
            search_pois_func=fake_search_pois,
        )

    monkeypatch.setattr(
        trip,
        "build_patch_result_from_message",
        fake_build_patch_result_from_message,
    )
    monkeypatch.setattr(
        trip,
        "apply_trip_patch_operations",
        fake_apply_trip_patch_operations,
    )

    created = await trip.create_trip_session(
        payload=TripSessionCreateRequest(
            request=make_request(),
            plan=make_plan(),
        )
    )

    response = await trip.chat_with_trip_session(
        session_id=created.data.id,
        payload=TripChatRequest(message="把第一天的故宫换成天坛"),
    )

    assert response.success is True
    assert response.intent is not None
    assert response.intent.change_type == "small_change"
    assert response.intent.patch_operations == [
        {
            "operation": "replace_attraction",
            "day_index": 1,
            "old_attraction_name": "故宫博物院",
            "new_attraction_name": "天坛",
        }
    ]

    day_one_attractions = response.data.current_plan.days[0].attractions
    day_one_names = [attraction.name for attraction in day_one_attractions]

    assert "故宫博物院" not in day_one_names
    assert "天坛公园" in day_one_names

    tiantan = next(
        attraction
        for attraction in day_one_attractions
        if attraction.name == "天坛公园"
    )

    assert tiantan.address == "北京市东城区天坛东路甲1号"
    assert tiantan.location.longitude == 116.410886
    assert tiantan.location.latitude == 39.881949
    assert tiantan.poi_id == "poi-tiantan"


@pytest.mark.asyncio
async def test_chat_route_uses_pending_patch_intent_for_follow_up(monkeypatch):
    reset_trip_session_service()

    async def fake_build_patch_result_from_pending_intent(
        message,
        plan,
        pending_patch_intent,
    ):
        assert message == "第一天的故宫"
        assert pending_patch_intent.known_fields == {
            "operation": "replace_attraction",
            "new_attraction_name": "天坛",
        }
        return PatchBuildResult(
            operations=[
                {
                    "operation": "replace_attraction",
                    "day_index": 1,
                    "old_attraction_name": "故宫博物院",
                    "new_attraction_name": "天坛",
                }
            ],
            pending_patch_intent=None,
        )

    async def fake_apply_trip_patch_operations(plan, operations):
        from backend.app.services.trip_plan_patch_service import (
            apply_trip_patch_operations,
        )

        async def fake_search_pois(keywords: str, city: str, limit: int = 5):
            return [
                POIInfo(
                    id="poi-tiantan",
                    name="天坛公园",
                    type="风景名胜",
                    address="北京市东城区天坛东路甲1号",
                    location=Location(
                        longitude=116.410886,
                        latitude=39.881949,
                    ),
                )
            ]

        return await apply_trip_patch_operations(
            plan=plan,
            operations=operations,
            search_pois_func=fake_search_pois,
        )

    monkeypatch.setattr(
        trip,
        "build_patch_result_from_pending_intent",
        fake_build_patch_result_from_pending_intent,
    )
    monkeypatch.setattr(
        trip,
        "apply_trip_patch_operations",
        fake_apply_trip_patch_operations,
    )

    created = await trip.create_trip_session(
        payload=TripSessionCreateRequest(
            request=make_request(),
            plan=make_plan(),
        )
    )

    service = trip.get_trip_session_service()
    service.update_pending_patch_intent(
        session_id=created.data.id,
        pending_patch_intent=PendingPatchIntent(
            operation="replace_attraction",
            known_fields={
                "operation": "replace_attraction",
                "new_attraction_name": "天坛",
            },
            missing_fields=["day_index", "old_attraction_name"],
            clarification_question="你想修改第几天的哪个景点？",
        ),
    )

    response = await trip.chat_with_trip_session(
        session_id=created.data.id,
        payload=TripChatRequest(message="第一天的故宫"),
    )

    assert response.success is True
    assert response.intent is not None
    assert response.intent.change_type == "small_change"
    assert response.intent.patch_operations == [
        {
            "operation": "replace_attraction",
            "day_index": 1,
            "old_attraction_name": "故宫博物院",
            "new_attraction_name": "天坛",
        }
    ]

    day_one_names = [
        attraction.name
        for attraction in response.data.current_plan.days[0].attractions
    ]

    assert "故宫博物院" not in day_one_names
    assert "天坛公园" in day_one_names
    assert response.data.pending_patch_intent is None
    assert "已完成局部修改" in response.data.messages[-1].content


@pytest.mark.asyncio
async def test_chat_route_passes_retrieved_knowledge_to_intent_classifier(monkeypatch):
    reset_trip_session_service()

    def fake_retrieve_default_travel_knowledge_text(query, city, limit=3):
        assert query == "我想少走路"
        assert city == "北京"
        assert limit == 3
        return "少走路建议：优先安排同一区域景点。"

    def fake_classify_trip_change_intent(message, knowledge_text="", use_llm=False):
        from backend.app.models.schemas import TripChangeIntent

        assert message == "我想少走路"
        assert "同一区域景点" in knowledge_text
        assert use_llm is True

        return TripChangeIntent(
            change_type="major_revision",
            summary="用户希望减少步行强度",
            patch_operations=[],
            clarification_question=None,
        )

    monkeypatch.setattr(
        trip,
        "retrieve_default_travel_knowledge_smart_text",
        fake_retrieve_default_travel_knowledge_text,
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
        payload=TripChatRequest(message="我想少走路"),
    )

    assert response.success is True
    assert response.intent.change_type == "major_revision"
    assert response.data.pending_revision_summary == "用户希望减少步行强度"
