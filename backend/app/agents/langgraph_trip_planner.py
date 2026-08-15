"""LangGraph travel planner."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Optional, TypedDict
from urllib.parse import urlencode
from urllib.request import urlopen

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from ..models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    POIInfo,
    TripPlan,
    TripRequest,
    WeatherInfo,
)
from ..services import amap_langchain_service
from ..services.llm_service import get_chat_model
from ..services.trip_plan_validator import validate_trip_plan

logger = logging.getLogger(__name__)


class TripPlannerState(TypedDict, total=False):
    request: TripRequest
    attractions_text: str
    attraction_candidates: list[POIInfo]
    weather_text: str
    hotels_text: str
    plan: TripPlan
    error: Optional[str]
    validation_errors: list[str]


async def collect_attractions(state: TripPlannerState) -> TripPlannerState:
    request = state["request"]
    keywords = request.preferences[0] if request.preferences else "景点"
    required_attraction_count = request.travel_days * 2
    candidate_limit = max(required_attraction_count + 8, 20)

    try:
        attraction_candidates = await _collect_unique_attraction_candidates(
            city=request.city,
            primary_keywords=keywords,
            required_count=required_attraction_count,
            limit=candidate_limit,
        )
        attractions_text = amap_langchain_service.format_pois_for_planner(
            attraction_candidates,
            city=request.city,
            keywords=keywords,
        )
    except Exception as exc:
        attractions_text = f"{request.city} 景点候选获取失败: {exc}"
        attraction_candidates = []

    return {
        "attractions_text": attractions_text,
        "attraction_candidates": attraction_candidates,
    }


async def _collect_unique_attraction_candidates(
    city: str,
    primary_keywords: str,
    required_count: int,
    limit: int,
) -> list[POIInfo]:
    search_keywords = [
        primary_keywords,
        "景点",
        "热门景点",
        "旅游景点",
        "风景区",
        "博物馆",
        "公园",
    ]
    unique_keywords = list(dict.fromkeys(keyword for keyword in search_keywords if keyword))
    candidates: list[POIInfo] = []
    seen_poi_ids: set[str] = set()
    seen_names: set[str] = set()

    for keyword in unique_keywords:
        if len(candidates) >= required_count:
            break

        pois = await amap_langchain_service.search_pois(
            keywords=keyword,
            city=city,
            citylimit=True,
            limit=limit,
        )
        for poi in pois:
            poi_id = str(poi.id or "").strip()
            name = _normalize_identity_name(poi.name)
            if poi_id and poi_id in seen_poi_ids:
                continue
            if name and name in seen_names:
                continue

            candidates.append(poi)
            if poi_id:
                seen_poi_ids.add(poi_id)
            if name:
                seen_names.add(name)

            if len(candidates) >= required_count:
                break

    logger.info(
        "Collected attraction candidates: city=%s required=%s final=%s keywords=%s",
        city,
        required_count,
        len(candidates),
        unique_keywords,
    )
    return candidates


async def collect_weather(state: TripPlannerState) -> TripPlannerState:
    request = state["request"]

    try:
        weather_text = await amap_langchain_service.get_weather_text(city=request.city)
    except Exception as exc:
        weather_text = f"{request.city} 天气信息获取失败: {exc}"

    return {"weather_text": weather_text}


async def collect_hotels(state: TripPlannerState) -> TripPlannerState:
    request = state["request"]
    keywords = request.accommodation or "酒店"

    try:
        hotels_text = await amap_langchain_service.search_pois_text(
            keywords=keywords,
            city=request.city,
            citylimit=True,
            limit=3,
        )
    except Exception as exc:
        hotels_text = f"{request.city} 酒店候选获取失败: {exc}"

    return {"hotels_text": hotels_text}


async def collect_travel_context(state: TripPlannerState) -> TripPlannerState:
    attractions_result, weather_result, hotels_result = await asyncio.gather(
        collect_attractions(state),
        collect_weather(state),
        collect_hotels(state),
    )

    return {
        **attractions_result,
        **weather_result,
        **hotels_result,
    }


async def generate_trip_plan_from_context(
    request: TripRequest,
    attractions_text: str,
    weather_text: str,
    hotels_text: str,
) -> TripPlan:
    prompt = build_trip_plan_prompt(
        request=request,
        attractions_text=attractions_text,
        weather_text=weather_text,
        hotels_text=hotels_text,
    )

    model = get_chat_model()
    response = await model.ainvoke(
        [
            SystemMessage(content="你是专业旅行规划智能体，必须严格输出合法 JSON。"),
            HumanMessage(content=prompt),
        ]
    )

    try:
        return parse_trip_plan_from_text(response.content)
    except Exception as exc:
        print(f"LLM TripPlan 解析失败，使用 fallback: {exc}")
        return create_fallback_plan(request)


async def generate_plan(state: TripPlannerState) -> TripPlannerState:
    request = state["request"]
    weather_text = state.get("weather_text", "")

    plan = await generate_trip_plan_from_context(
        request=request,
        attractions_text=state.get("attractions_text", ""),
        weather_text=weather_text,
        hotels_text=state.get("hotels_text", ""),
    )
    plan = backfill_plan_attractions_from_candidates(
        plan=plan,
        candidates=state.get("attraction_candidates", []),
    )
    plan.weather_info = parse_weather_info_from_text(
        weather_text=weather_text,
        request=request,
    )
    plan = await enrich_plan_with_poi_metadata(plan, request)

    return {"plan": plan}


def backfill_plan_attractions_from_candidates(
    plan: TripPlan,
    candidates: list[POIInfo],
) -> TripPlan:
    """Replace repeated LLM attractions and backfill with unused Amap candidate POIs."""
    updated_plan = plan.model_copy(deep=True)
    if not updated_plan.days:
        return updated_plan

    required_count = len(updated_plan.days) * 2
    original_count = sum(len(day.attractions) for day in updated_plan.days)
    unique_attractions = _collect_unique_attractions(updated_plan)
    seen_poi_ids, seen_names = _collect_attraction_identity_sets(unique_attractions)
    unique_before_backfill_count = len(unique_attractions)

    for candidate in candidates:
        candidate_poi_id = str(candidate.id or "").strip()
        candidate_name = _normalize_identity_name(candidate.name)
        if candidate_poi_id and candidate_poi_id in seen_poi_ids:
            continue
        if candidate_name and candidate_name in seen_names:
            continue

        unique_attractions.append(_candidate_poi_to_attraction(candidate))
        if candidate_poi_id:
            seen_poi_ids.add(candidate_poi_id)
        if candidate_name:
            seen_names.add(candidate_name)
        if len(unique_attractions) >= required_count:
            break

    _assign_attractions_to_days(updated_plan, unique_attractions[:required_count])
    logger.info(
        "Trip attraction backfill: original=%s unique_before=%s candidates=%s final=%s required=%s daily_counts=%s",
        original_count,
        unique_before_backfill_count,
        len(candidates),
        sum(len(day.attractions) for day in updated_plan.days),
        required_count,
        [len(day.attractions) for day in updated_plan.days],
    )

    return updated_plan


def _collect_unique_attractions(plan: TripPlan) -> list[Attraction]:
    unique_attractions: list[Attraction] = []
    seen_poi_ids: set[str] = set()
    seen_names: set[str] = set()

    for day in plan.days:
        for attraction in day.attractions:
            poi_id = str(attraction.poi_id or "").strip()
            name = _normalize_identity_name(attraction.name)
            if poi_id and poi_id in seen_poi_ids:
                continue
            if name and name in seen_names:
                continue

            unique_attractions.append(attraction)
            if poi_id:
                seen_poi_ids.add(poi_id)
            if name:
                seen_names.add(name)

    return unique_attractions


def _collect_attraction_identity_sets(
    attractions: list[Attraction],
) -> tuple[set[str], set[str]]:
    poi_ids: set[str] = set()
    names: set[str] = set()

    for attraction in attractions:
        poi_id = str(attraction.poi_id or "").strip()
        name = _normalize_identity_name(attraction.name)
        if poi_id:
            poi_ids.add(poi_id)
        if name:
            names.add(name)

    return poi_ids, names


def _candidate_poi_to_attraction(candidate: POIInfo) -> Attraction:
    return Attraction(
        name=candidate.name,
        address=candidate.address,
        location=candidate.location,
        visit_duration=90,
        description=f"{candidate.name} 是候选景点，可作为行程补充安排。",
        category=candidate.type or "景点",
        poi_id=candidate.id,
        ticket_price=0,
    )


def _normalize_identity_name(name: str | None) -> str:
    name_key = str(name or "").strip().lower()
    name_key = re.sub(r"[·\-\s\(\)（）【】\[\]、，。,.]+", "", name_key)
    return name_key


def _assign_attractions_to_days(plan: TripPlan, attractions: list[Attraction]) -> None:
    cursor = 0
    for day in plan.days:
        day.attractions = attractions[cursor:cursor + 2]
        cursor += 2


def validate_plan(state: TripPlannerState) -> TripPlannerState:
    request = state["request"]
    plan = state.get("plan")

    if plan is None:
        return {"validation_errors": ["缺少 TripPlan"]}

    errors = validate_trip_plan(
        plan,
        request,
        weather_text=state.get("weather_text", ""),
    )
    if errors:
        print("TripPlan validation errors:")
        for error in errors:
            print(f" - {error}")

    return {"validation_errors": errors}


def build_trip_planner_graph():
    builder = StateGraph(TripPlannerState)

    builder.add_node("collect_travel_context", collect_travel_context)
    builder.add_node("generate_plan", generate_plan)
    builder.add_node("validate_plan", validate_plan)
    builder.set_entry_point("collect_travel_context")

    builder.add_edge("collect_travel_context", "generate_plan")
    builder.add_edge("generate_plan", "validate_plan")
    builder.add_edge("validate_plan", END)

    return builder.compile()


class LangGraphTripPlanner:
    """LangGraph planner facade used by the existing FastAPI route."""

    def __init__(self):
        self.graph = build_trip_planner_graph()

    async def plan_trip(self, request: TripRequest) -> TripPlan:
        result = await self.graph.ainvoke({"request": request})
        plan = result.get("plan")
        if plan is not None:
            return plan
        return create_fallback_plan(request)


async def enrich_plan_with_poi_metadata(
    plan: TripPlan,
    request: TripRequest,
) -> TripPlan:
    updated_plan = plan.model_copy(deep=True)
    used_poi_ids: set[str] = set()

    for day in updated_plan.days:
        for attraction in day.attractions:
            if attraction.poi_id:
                used_poi_ids.add(attraction.poi_id)

    for day in updated_plan.days:
        for attraction in day.attractions:
            needs_location = attraction.location is None or (
                attraction.location.longitude == 0 and attraction.location.latitude == 0
            )
            needs_metadata = (
                not attraction.address
                or not attraction.category
                or needs_location
            )
            if attraction.poi_id and not needs_metadata:
                continue

            if attraction.poi_id and needs_location:
                poi_location = _get_poi_location_from_id(attraction.poi_id)
                if poi_location is not None:
                    attraction.location = poi_location
                    continue

            poi = await _find_matching_poi_for_attraction(
                attraction_name=attraction.name,
                city=request.city,
                used_poi_ids=used_poi_ids,
            )
            if poi is None:
                continue

            used_poi_ids.add(poi.id)
            attraction.poi_id = attraction.poi_id or poi.id
            if not attraction.address:
                attraction.address = poi.address
            if attraction.location is None or needs_location:
                attraction.location = poi.location
            if not attraction.category:
                attraction.category = poi.type or attraction.category

    return updated_plan


def _get_poi_location_from_id(poi_id: str) -> Location | None:
    from ..config import get_settings

    settings = get_settings()
    if not settings.amap_api_key or not poi_id:
        return None

    params = {
        "key": settings.amap_api_key,
        "id": poi_id,
        "extensions": "all",
    }

    try:
        query = urlencode(params)
        with urlopen(f"https://restapi.amap.com/v3/place/detail?{query}", timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception:
        return None

    pois = data.get("pois") if isinstance(data, dict) else None
    if not isinstance(pois, list) or not pois:
        return None

    first = pois[0]
    if not isinstance(first, dict):
        return None

    location_text = first.get("location")
    if not isinstance(location_text, str) or "," not in location_text:
        return None

    try:
        longitude_text, latitude_text = location_text.split(",", 1)
        return Location(longitude=float(longitude_text), latitude=float(latitude_text))
    except ValueError:
        return None


async def _find_matching_poi_for_attraction(
    attraction_name: str,
    city: str,
    used_poi_ids: set[str],
) -> POIInfo | None:
    if not attraction_name:
        return None

    try:
        pois = await amap_langchain_service.search_pois(
            keywords=attraction_name,
            city=city,
            citylimit=True,
            limit=5,
        )
    except Exception as exc:
        print(f"POI metadata enrichment failed: {exc}")
        return None

    for poi in pois:
        if poi.id in used_poi_ids:
            continue
        if _poi_name_matches(attraction_name, poi.name):
            return poi

    for poi in pois:
        if poi.id not in used_poi_ids:
            return poi

    return None


def _poi_name_matches(attraction_name: str, poi_name: str) -> bool:
    if not attraction_name or not poi_name:
        return False
    if attraction_name == poi_name:
        return True
    return attraction_name in poi_name or poi_name in attraction_name


def create_fallback_plan(request: TripRequest) -> TripPlan:
    """Create a valid TripPlan when real LLM planning fails."""
    start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
    days = []

    for index in range(request.travel_days):
        current_date = start_date + timedelta(days=index)
        days.append(
            DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=index,
                description=f"第{index + 1}天基础行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}推荐景点{index + 1}",
                        address=request.city,
                        location=Location(longitude=116.397128, latitude=39.916527),
                        visit_duration=120,
                        description="这是 LangGraph 兜底规划生成的景点，后续会替换为真实 POI。",
                        category="景点",
                        ticket_price=0,
                    )
                ],
                meals=[
                    Meal(type="breakfast", name="早餐", description="推荐当地早餐", estimated_cost=30),
                    Meal(type="lunch", name="午餐", description="推荐当地午餐", estimated_cost=60),
                    Meal(type="dinner", name="晚餐", description="推荐当地晚餐", estimated_cost=80),
                ],
            )
        )

    return TripPlan(
        city=request.city,
        start_date=request.start_date,
        end_date=request.end_date,
        days=days,
        weather_info=[],
        overall_suggestions="当前使用 LangGraph 兜底行程。后续会接入 LLM、MCP 和真实旅行数据。",
        budget=None,
    )


def parse_trip_plan_from_text(text: str) -> TripPlan:
    json_text = _extract_json_object(text)
    if not json_text:
        raise ValueError("未找到 TripPlan JSON")

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"TripPlan JSON 解析失败: {exc}") from exc

    return TripPlan(**data)


def _extract_json_object(text: str) -> str:
    if not text:
        return ""

    code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""

    return text[start : end + 1]


def parse_weather_info_from_text(
    weather_text: str,
    request: TripRequest,
) -> list[WeatherInfo]:
    if not weather_text:
        return []

    allowed_dates = {
        (datetime.strptime(request.start_date, "%Y-%m-%d") + timedelta(days=index)).strftime("%Y-%m-%d")
        for index in range(request.travel_days)
    }
    weather_infos: list[WeatherInfo] = []

    pattern = re.compile(
        r"(?:日期|日期:|日期：)?\s*(?P<date>\d{4}-\d{2}-\d{2}).*?"
        r"(?:白天|dayweather)[:：]\s*(?P<day_weather>.*?)\s+"
        r"(?P<day_temp>-?\d+(?:\.\d+)?)\s*℃?\s*"
        r"(?P<day_wind>[^\d|]+?)\s*"
        r"(?P<day_power>\d+-\d+级?)\s*[|｜].*?"
        r"(?:夜间|nightweather)[:：]\s*(?P<night_weather>.*?)\s+"
        r"(?P<night_temp>-?\d+(?:\.\d+)?)\s*℃?\s*"
        r"(?P<night_wind>[^\d|]+?)\s*"
        r"(?P<night_power>\d+-\d+级?)",
        re.IGNORECASE,
    )

    for match in pattern.finditer(weather_text):
        weather_date = match.group("date")
        if weather_date not in allowed_dates:
            continue

        weather_infos.append(
            WeatherInfo(
                date=weather_date,
                day_weather=match.group("day_weather").strip(),
                night_weather=match.group("night_weather").strip(),
                day_temp=int(float(match.group("day_temp"))),
                night_temp=int(float(match.group("night_temp"))),
                wind_direction=match.group("day_wind").strip(),
                wind_power=match.group("day_power").strip(),
            )
        )

    return weather_infos


def build_trip_plan_prompt(
    request: TripRequest,
    attractions_text: str,
    weather_text: str,
    hotels_text: str,
) -> str:
    preferences = "、".join(request.preferences) if request.preferences else "无"

    return f"""
请根据以下信息生成一份旅行规划，只返回 JSON，不要返回解释文字。
为了减少等待时间，字段内容尽量简洁；景点 description 不超过 40 字，overall_suggestions 不超过 80 字。

基本信息：
- 城市：{request.city}
- 开始日期：{request.start_date}
- 结束日期：{request.end_date}
- 天数：{request.travel_days}
- 交通方式：{request.transportation}
- 住宿偏好：{request.accommodation}
- 旅行偏好：{preferences}
- 额外要求：{request.free_text_input or "无"}

景点候选：
{attractions_text}

天气信息：
{weather_text}

酒店候选：
{hotels_text}

JSON 格式必须匹配以下结构：
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
      "hotel": {{
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {{"longitude": 116.397128, "latitude": 39.916527}},
        "price_range": "价格范围",
        "rating": "",
        "distance": "",
        "type": "酒店类型",
        "estimated_cost": 300
      }},
      "attractions": [
        {{
          "name": "景点名称",
          "address": "景点地址",
          "location": {{"longitude": 116.397128, "latitude": 39.916527}},
          "visit_duration": 120,
          "description": "景点说明",
          "category": "景点类型",
          "poi_id": "POI ID",
          "ticket_price": 0
        }},
        {{
          "name": "第二个景点名称",
          "address": "第二个景点地址",
          "location": {{"longitude": 116.397128, "latitude": 39.916527}},
          "visit_duration": 90,
          "description": "景点说明",
          "category": "景点类型",
          "poi_id": "POI ID",
          "ticket_price": 0
        }}
      ],
      "meals": [
        {{"type": "breakfast", "name": "早餐", "description": "早餐建议", "estimated_cost": 30}},
        {{"type": "lunch", "name": "午餐", "description": "午餐建议", "estimated_cost": 60}},
        {{"type": "dinner", "name": "晚餐", "description": "晚餐建议", "estimated_cost": 80}}
      ]
    }}
  ],
  "weather_info": [],
  "overall_suggestions": "整体建议",
  "budget": {{
    "total_attractions": 0,
    "total_hotels": 0,
    "total_meals": 0,
    "total_transportation": 0,
    "total": 0
  }}
}}

要求：
1. days 数量必须等于 {request.travel_days}
2. 每天必须包含 breakfast、lunch、dinner
3. 景点优先从景点候选中选择
4. 酒店优先从酒店候选中选择
5. 坐标必须使用候选信息中的真实坐标
6. 只返回 JSON，不要 Markdown，不要解释
7. weather_info 只能使用天气信息中明确出现的日期；如果天气信息不覆盖旅行日期，则返回 []
8. 不要编造天气、温度、风向、风力
9. 如果景点来自候选 POI，必须保留或填写候选信息中的 POI ID 到 attractions[].poi_id
10. 每天 attractions 必须恰好 2 个景点
11. 同一天的 2 个景点距离要尽量近，距离较远的景点不要安排在同一天
""".strip()
