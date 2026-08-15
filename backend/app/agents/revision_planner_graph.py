from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from ..rag.travel_retriever import (
    retrieve_default_travel_knowledge_smart_text,
)
from ..models.schemas import TripPlan, TripSession
from ..services.trip_plan_validator import validate_trip_plan
from ..services.llm_service import get_chat_model
from .langgraph_trip_planner import create_fallback_plan, parse_trip_plan_from_text


class RevisionPlannerState(TypedDict, total=False):
    session: TripSession
    revision_summary: str
    original_plan: TripPlan
    revised_plan: TripPlan
    validation_errors: list[str]
    error: Optional[str]


def prepare_revision_context(
    state: RevisionPlannerState,
) -> RevisionPlannerState:
    session = state["session"]
    return {"original_plan": session.current_plan}


async def generate_revised_plan(
    state: RevisionPlannerState,
) -> RevisionPlannerState:
    session = state["session"]
    revision_summary = state["revision_summary"]

    revised_plan = await generate_revised_plan_from_revision_context(
        session=session,
        revision_summary=revision_summary,
    )

    return {"revised_plan": revised_plan}


def validate_revised_plan(
    state: RevisionPlannerState,
) -> RevisionPlannerState:
    session = state["session"]
    revised_plan = state.get("revised_plan")

    if revised_plan is None:
        return {"validation_errors": ["缺少 revised_plan"]}

    return {
        "validation_errors": validate_trip_plan(
            revised_plan,
            session.request,
        )
    }


async def generate_revised_plan_from_revision_context(
    session: TripSession,
    revision_summary: str,
) -> TripPlan:
    knowledge_text = retrieve_default_travel_knowledge_smart_text(
        query=revision_summary,
        city=session.request.city,
        limit=3,
    )

    prompt = build_revision_plan_prompt(
        session=session,
        revision_summary=revision_summary,
        knowledge_text=knowledge_text,
    )
    chat_model = get_chat_model()

    response = await chat_model.ainvoke(
        [
            SystemMessage(
                content="你是专业旅行重规划智能体。你必须严格输出合法 JSON。"
            ),
            HumanMessage(content=prompt),
        ]
    )

    content = getattr(response, "content", str(response))

    try:
        return parse_trip_plan_from_text(content)
    except Exception as exc:
        print(f"Revision TripPlan 解析失败，使用 fallback: {exc}")
        return create_fallback_plan(session.request)


def build_revision_plan_prompt(
    session: TripSession,
    revision_summary: str,
    knowledge_text: str = "",
) -> str:
    request = session.request
    current_plan_summary = _build_current_plan_summary(session.current_plan)
    preferences = "、".join(request.preferences) if request.preferences else "无"

    return f"""
请基于当前已有旅行计划，按用户确认的大改动要求重新规划，并且只返回 JSON，不要返回解释文字。

原始旅行请求：
- 城市：{request.city}
- 开始日期：{request.start_date}
- 结束日期：{request.end_date}
- 天数：{request.travel_days}
- 交通方式：{request.transportation}
- 住宿偏好：{request.accommodation}
- 旅行偏好：{preferences}
- 额外要求：{request.free_text_input or "无"}

当前行程摘要：
{current_plan_summary}

用户确认的大改动要求：
{revision_summary}

参考旅行知识：
{knowledge_text or "无相关旅行知识。"}

重规划约束：
1. 保持 city、start_date、end_date 与原始旅行请求一致。
2. days 数量必须等于 {request.travel_days}。
3. 每天必须包含 breakfast、lunch、dinner。
4. 尽量保留仍然符合新要求的景点和餐食。
5. 只调整和用户大改动要求相关的部分。
6. 如果用户要求少走路，应减少跨区域跳转，优先安排距离更近或室内休息更方便的点。
7. 不要编造天气；weather_info 没有可靠来源时返回 []。
8. 只返回 JSON，不要 Markdown，不要解释。

JSON 格式必须匹配 TripPlan：
{{
  "city": "{request.city}",
  "start_date": "{request.start_date}",
  "end_date": "{request.end_date}",
  "days": [
    {{
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "当天行程概述",
      "transportation": "{request.transportation}",
      "accommodation": "{request.accommodation}",
      "attractions": [
        {{
          "name": "景点名称",
          "address": "景点地址",
          "location": {{"longitude": 116.397128, "latitude": 39.916527}},
          "visit_duration": 120,
          "description": "景点说明",
          "category": "景点类型",
          "ticket_price": 0
        }}
      ],
      "meals": [
        {{"type": "breakfast", "name": "早餐"}},
        {{"type": "lunch", "name": "午餐"}},
        {{"type": "dinner", "name": "晚餐"}}
      ]
    }}
  ],
  "weather_info": [],
  "overall_suggestions": "整体建议",
  "budget": null
}}
""".strip()


def _build_current_plan_summary(plan: TripPlan) -> str:
    lines = [
        f"城市：{plan.city}",
        f"日期：{plan.start_date} 至 {plan.end_date}",
        "每日行程：",
    ]

    for day in plan.days:
        attraction_names = "、".join(
            attraction.name for attraction in day.attractions
        )
        meal_names = "、".join(meal.name for meal in day.meals)
        lines.append(
            f"- 第 {day.day_index + 1} 天 {day.date}："
            f"{day.description}；景点：{attraction_names or '无'}；"
            f"餐食：{meal_names or '无'}"
        )

    lines.append(f"整体建议：{plan.overall_suggestions}")
    return "\n".join(lines)


def build_revision_planner_graph():
    builder = StateGraph(RevisionPlannerState)

    builder.add_node("prepare_revision_context", prepare_revision_context)
    builder.add_node("generate_revised_plan", generate_revised_plan)
    builder.add_node("validate_revised_plan", validate_revised_plan)

    builder.set_entry_point("prepare_revision_context")
    builder.add_edge("prepare_revision_context", "generate_revised_plan")
    builder.add_edge("generate_revised_plan", "validate_revised_plan")
    builder.add_edge("validate_revised_plan", END)

    return builder.compile()
