import pytest

from backend.app.agents import revision_planner_graph
from backend.app.agents.revision_planner_graph import (
    build_revision_plan_prompt,
    build_revision_planner_graph,
    generate_revised_plan_from_revision_context,
)
from backend.app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    TripPlan,
    TripRequest,
    
)
from backend.app.services.trip_session_service import TripSessionService

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
                day_index=0,
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
            ),
            DayPlan(
                date="2026-08-02",
                day_index=1,
                description="第二天行程",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[
                    Attraction(
                        name="中国国家博物馆",
                        address="东长安街16号",
                        location=Location(
                            longitude=116.397755,
                            latitude=39.903182,
                        ),
                        visit_duration=180,
                        description="博物馆景点",
                        category="博物馆",
                        ticket_price=0,
                    )
                ],
                meals=[
                    Meal(type="breakfast", name="早餐"),
                    Meal(type="lunch", name="午餐"),
                    Meal(type="dinner", name="晚餐"),
                ],
            ),
        ],
        weather_info=[],
        overall_suggestions=suggestion,
        budget=None,
    )


def test_build_revision_plan_prompt_contains_original_plan_and_summary():
    service = TripSessionService()
    session = service.create_session(
        request=make_request(),
        plan=make_plan(),
    )

    prompt = build_revision_plan_prompt(
        session=session,
        revision_summary="少走路，整体轻松一点",
    )

    assert "原始旅行请求" in prompt
    assert "当前行程摘要" in prompt
    assert "故宫博物院" in prompt
    assert "中国国家博物馆" in prompt
    assert "少走路，整体轻松一点" in prompt
    assert "只返回 JSON" in prompt


@pytest.mark.asyncio
async def test_generate_revised_plan_from_revision_context_parses_llm_json(monkeypatch):
    service = TripSessionService()
    session = service.create_session(
        request=make_request(),
        plan=make_plan(),
    )

    class FakeResponse:
        content = """
        {
          "city": "北京",
          "start_date": "2026-08-01",
          "end_date": "2026-08-02",
          "days": [
            {
              "date": "2026-08-01",
              "day_index": 0,
              "description": "第一天轻松游览",
              "transportation": "公共交通",
              "accommodation": "经济型酒店",
              "attractions": [
                {
                  "name": "故宫博物院",
                  "address": "景山前街4号",
                  "location": {"longitude": 116.397005, "latitude": 39.919278},
                  "visit_duration": 120,
                  "description": "减少停留时间，轻松游览。",
                  "category": "历史文化",
                  "ticket_price": 60
                }
              ],
              "meals": [
                {"type": "breakfast", "name": "早餐"},
                {"type": "lunch", "name": "午餐"},
                {"type": "dinner", "name": "晚餐"}
              ]
            },
            {
              "date": "2026-08-02",
              "day_index": 1,
              "description": "第二天轻松游览",
              "transportation": "公共交通",
              "accommodation": "经济型酒店",
              "attractions": [
                {
                  "name": "中国国家博物馆",
                  "address": "东长安街16号",
                  "location": {"longitude": 116.397755, "latitude": 39.903182},
                  "visit_duration": 120,
                  "description": "保留室内景点，减少步行。",
                  "category": "博物馆",
                  "ticket_price": 0
                }
              ],
              "meals": [
                {"type": "breakfast", "name": "早餐"},
                {"type": "lunch", "name": "午餐"},
                {"type": "dinner", "name": "晚餐"}
              ]
            }
          ],
          "weather_info": [],
          "overall_suggestions": "新版少走路建议",
          "budget": null
        }
        """

    class FakeChatModel:
        async def ainvoke(self, messages):
            joined = "\n".join(message.content for message in messages)
            assert "当前行程摘要" in joined
            assert "少走路，整体轻松一点" in joined
            return FakeResponse()

    monkeypatch.setattr(
        revision_planner_graph,
        "get_chat_model",
        lambda: FakeChatModel(),
    )

    plan = await generate_revised_plan_from_revision_context(
        session=session,
        revision_summary="少走路，整体轻松一点",
    )

    assert plan.overall_suggestions == "新版少走路建议"
    assert len(plan.days) == 2
    assert plan.days[0].attractions[0].name == "故宫博物院"


@pytest.mark.asyncio
async def test_generate_revised_plan_from_revision_context_falls_back_for_invalid_llm_output(
    monkeypatch,
):
    service = TripSessionService()
    session = service.create_session(
        request=make_request(),
        plan=make_plan(),
    )

    class FakeResponse:
        content = "这不是 JSON"

    class FakeChatModel:
        async def ainvoke(self, messages):
            return FakeResponse()

    monkeypatch.setattr(
        revision_planner_graph,
        "get_chat_model",
        lambda: FakeChatModel(),
    )

    plan = await generate_revised_plan_from_revision_context(
        session=session,
        revision_summary="少走路，整体轻松一点",
    )

    assert plan.city == "北京"
    assert plan.start_date == "2026-08-01"
    assert plan.end_date == "2026-08-02"
    assert len(plan.days) == 2
    assert "兜底" in plan.overall_suggestions


@pytest.mark.asyncio
async def test_revision_graph_returns_revised_plan_and_keeps_original(monkeypatch):
    service = TripSessionService()
    original_plan = make_plan()
    session = service.create_session(
        request=make_request(),
        plan=original_plan,
    )
    revised_plan = make_plan(suggestion="新版少走路建议")

    async def fake_generate_revised_plan_from_revision_context(
        session,
        revision_summary,
    ):
        assert session.current_plan is original_plan
        assert revision_summary == "少走路，整体轻松一点"
        return revised_plan

    monkeypatch.setattr(
        revision_planner_graph,
        "generate_revised_plan_from_revision_context",
        fake_generate_revised_plan_from_revision_context,
    )

    graph = build_revision_planner_graph()

    result = await graph.ainvoke(
        {
            "session": session,
            "revision_summary": "少走路，整体轻松一点",
        }
    )

    assert result["original_plan"] is original_plan
    assert result["revised_plan"].overall_suggestions == "新版少走路建议"
    assert result["validation_errors"] == []


@pytest.mark.asyncio
async def test_revision_graph_records_validation_errors(monkeypatch):
    service = TripSessionService()
    session = service.create_session(
        request=make_request(),
        plan=make_plan(),
    )
    invalid_plan = make_plan(suggestion="错误计划")
    invalid_plan.days = invalid_plan.days[:1]

    async def fake_generate_revised_plan_from_revision_context(
        session,
        revision_summary,
    ):
        return invalid_plan

    monkeypatch.setattr(
        revision_planner_graph,
        "generate_revised_plan_from_revision_context",
        fake_generate_revised_plan_from_revision_context,
    )

    graph = build_revision_planner_graph()

    result = await graph.ainvoke(
        {
            "session": session,
            "revision_summary": "压缩行程",
        }
    )

    assert result["revised_plan"] is invalid_plan
    assert result["validation_errors"]
    assert "行程天数不匹配" in result["validation_errors"][0]


def test_build_revision_plan_prompt_contains_retrieved_knowledge():
    service = TripSessionService()
    session = service.create_session(
        request=make_request(),
        plan=make_plan(),
    )

    prompt = build_revision_plan_prompt(
        session=session,
        revision_summary="少走路，整体轻松一点",
        knowledge_text=(
            "少走路建议：优先安排同一区域景点，减少跨区域移动。"
        ),
    )

    assert "参考旅行知识" in prompt
    assert "优先安排同一区域景点" in prompt
    assert "少走路，整体轻松一点" in prompt