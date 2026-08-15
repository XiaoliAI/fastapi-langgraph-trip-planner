"""Travel plan experience enrichment."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from math import atan2, cos, radians, sin, sqrt
from typing import Any, Iterable, Optional

from ..models.schemas import Attraction, Hotel, Location, Meal, PhotoSpot, TripPlan, TripRequest
from .amap_langchain_service import geocode_location, search_pois
from .llm_service import get_chat_model

REVIEW_SUMMARY_TIMEOUT_SECONDS = 60.0


@dataclass
class ExperienceSnippet:
    title: str
    photo_spots: list[str]
    review_summary: str
    visit_tips: list[str]


async def enrich_trip_plan_experience(plan: TripPlan, request: TripRequest) -> TripPlan:
    updated_plan = plan.model_copy(deep=True)
    attractions = [
        attraction
        for day in updated_plan.days
        for attraction in day.attractions
    ]
    review_summaries = await _build_review_summaries_with_timeout(attractions, request.city)

    previous_location: Optional[Location] = None
    previous_label = ""
    used_meal_keys: set[str] = set()

    for day in updated_plan.days:
        await _enrich_hotel(day.hotel, request.city)

        for attraction in day.attractions:
            _enrich_attraction(
                attraction,
                request.city,
                review_summaries.get(attraction.name),
            )
            attraction.route_tip = _build_route_tip(previous_location, previous_label, attraction)
            previous_location = attraction.location
            previous_label = attraction.name

        await _enrich_meals(
            day.meals,
            request.city,
            day.attractions,
            day.hotel,
            used_meal_keys,
        )

    await _apply_llm_meal_reasons_with_timeout(updated_plan, request.city)
    updated_plan.hotels = await _build_hotel_candidates(updated_plan, request)

    return updated_plan


async def _build_review_summaries_with_timeout(
    attractions: list[Attraction],
    city: str,
) -> dict[str, str]:
    try:
        return await asyncio.wait_for(
            _build_llm_review_summaries(attractions, city),
            timeout=REVIEW_SUMMARY_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print("LLM点评摘要生成超时，使用本地摘要兜底")
        return _build_local_review_summaries(attractions, city)


def _build_local_review_summaries(
    attractions: list[Attraction],
    city: str,
) -> dict[str, str]:
    summaries: dict[str, str] = {}
    for attraction in attractions:
        snippet = _build_snippet(attraction.name, city)
        summaries[attraction.name] = snippet.review_summary
    return summaries


async def _build_llm_review_summaries(
    attractions: list[Attraction],
    city: str,
) -> dict[str, str]:
    pending_attractions = [
        attraction
        for attraction in attractions
        if not attraction.review_summary
    ]
    if not pending_attractions:
        return {}

    attraction_lines = "\n".join(
        (
            f"{index}. {attraction.name} | "
            f"{attraction.category or '景点'} | "
            f"{attraction.description[:32]}"
        )
        for index, attraction in enumerate(pending_attractions, start=1)
    )
    prompt = f"""
为旅行规划页生成景点点评摘要，只返回 JSON。
规则：每条 35-55 字；说明看点、适合人群、注意点；不要假装读取了真实网友评论。
示例：{{"景点名":"适合拍照和轻松游览，主要看点集中，旺季注意错峰。"}}
城市：{city}
景点：
{attraction_lines}
""".strip()

    try:
        model = get_chat_model()
        if hasattr(model, "bind"):
            model = model.bind(max_tokens=700)
        response = await model.ainvoke(prompt)
        content = str(getattr(response, "content", response)).strip()
        summaries = _parse_llm_summary_map(content)
        return {
            attraction.name: summary
            for attraction in pending_attractions
            if (summary := _clean_llm_summary(summaries.get(attraction.name, "")))
        }
    except Exception as exc:
        print(f"LLM点评摘要生成失败，使用本地摘要兜底: {exc}")
        return {}


async def _build_llm_review_summary(attraction: Attraction, city: str) -> Optional[str]:
    summaries = await _build_llm_review_summaries([attraction], city)
    return summaries.get(attraction.name)


def _parse_llm_summary_map(content: str) -> dict[str, str]:
    text = content.strip()
    if not text:
        return {}

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}

    text = text[start:end + 1]

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, dict):
        return {}

    return {
        str(key): str(value)
        for key, value in data.items()
        if isinstance(value, str) and value.strip()
    }


def _clean_llm_summary(content: str) -> Optional[str]:
    if not content:
        return None

    text = content.strip().strip('"').strip("'")
    for prefix in ("点评摘要：", "点评摘要:", "摘要：", "摘要:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    if not text:
        return None

    return text[:120]


def _enrich_attraction(
    attraction: Attraction,
    city: str,
    review_summary: Optional[str] = None,
) -> None:
    snippet = _build_snippet(attraction.name, city)
    if not attraction.review_summary:
        attraction.review_summary = review_summary or snippet.review_summary
    if not attraction.photo_spots:
        attraction.photo_spots = _unique_photo_spots(snippet.photo_spots)
    if not attraction.visit_tips:
        attraction.visit_tips = snippet.visit_tips
    if not attraction.photo_spot_details:
        attraction.photo_spot_details = _build_photo_spot_details(snippet)


def _build_photo_spot_details(snippet: ExperienceSnippet) -> list[PhotoSpot]:
    details: list[PhotoSpot] = []
    tips = snippet.visit_tips or []

    for index, spot in enumerate(_unique_photo_spots(snippet.photo_spots)):
        tip = tips[index] if index < len(tips) else ""
        description = tip or f"{spot} 适合作为 {snippet.title} 的重点停留位置。"
        details.append(
            PhotoSpot(
                name=spot,
                description=description,
                source="generated",
            )
        )

    return details


def _unique_photo_spots(photo_spots: list[str]) -> list[str]:
    unique: list[str] = []
    for spot in photo_spots:
        normalized = str(spot or "").strip()
        if not normalized or normalized in unique:
            continue
        unique.append(normalized)

    fallback_spots = ["入口视角", "核心景观", "人像打卡位", "全景取景位"]
    for spot in fallback_spots:
        if len(unique) >= 4:
            break
        if spot not in unique:
            unique.append(spot)

    return unique[:4]


async def _enrich_meals(
    meals: Iterable[Meal],
    city: str,
    attractions: list[Attraction],
    hotel: Optional[Hotel],
    used_meal_keys: set[str],
) -> None:
    meal_list = list(meals)
    anchor_points = _build_meal_anchor_points(attractions, hotel, city)

    search_keywords = {
        _meal_search_keyword(meal.type, city, meal.name)
        for meal in meal_list
        if not (meal.address and meal.location)
    }
    poi_results_by_keyword = await _search_meal_pois_by_keyword(search_keywords, city)

    for meal in meal_list:
        anchor_location, anchor_label = _meal_anchor_for_type(meal.type, anchor_points)
        if meal.address and meal.location:
            meal.review_summary = meal.review_summary or f"{meal.name} 适合作为行程中的用餐点，建议结合前后景点距离安排。"
            meal.route_tip = meal.route_tip or _build_route_tip(anchor_location, anchor_label, None, meal.location, meal.name)
            meal.recommended_reason = meal.recommended_reason or "适合穿插在当天行程中，减少额外绕路。"
            used_meal_keys.add(_meal_dedupe_key(meal))
            continue

        keyword = _meal_search_keyword(meal.type, city, meal.name)
        pois = poi_results_by_keyword.get(keyword, [])

        if pois:
            poi = _select_nearest_unused_meal_poi(pois, anchor_location, used_meal_keys)
            if poi is not None:
                if not meal.name or meal.name in {"早餐", "午餐", "晚餐", "小吃"}:
                    meal.name = poi.name
                meal.poi_id = poi.id
                meal.address = poi.address
                meal.location = poi.location
                meal.review_summary = meal.review_summary or _meal_review_summary(meal, poi, anchor_label)
                meal.recommended_reason = meal.recommended_reason or _meal_recommended_reason(poi, anchor_location)
                meal.route_tip = meal.route_tip or _build_route_tip(anchor_location, anchor_label, None, poi.location, poi.name)
                used_meal_keys.add(_poi_dedupe_key(poi))
            else:
                meal.review_summary = meal.review_summary or f"{city} 的 {meal.type} 可优先选择靠近景点或交通站点的餐厅。"
                meal.recommended_reason = meal.recommended_reason or "当天同类餐厅候选已被使用，建议在景点附近现场选择不重复店铺。"
        else:
            meal.review_summary = meal.review_summary or f"{city} 的 {meal.type} 可优先选择靠近景点或交通站点的餐厅。"
            meal.recommended_reason = meal.recommended_reason or "优先安排在景点附近，减少绕路。"


async def _search_meal_pois_by_keyword(
    keywords: set[str],
    city: str,
) -> dict[str, list[Any]]:
    async def search_one(keyword: str) -> tuple[str, list[Any]]:
        try:
            return keyword, await search_pois(keyword, city, citylimit=True, limit=12)
        except Exception:
            return keyword, []

    if not keywords:
        return {}

    results = await asyncio.gather(
        *(search_one(keyword) for keyword in keywords)
    )
    return dict(results)


def _build_meal_anchor_points(
    attractions: list[Attraction],
    hotel: Optional[Hotel],
    city: str,
) -> dict[str, tuple[Optional[Location], str]]:
    fallback = (hotel.location, hotel.name) if hotel and hotel.location else (None, city)
    if not attractions:
        return {"breakfast": fallback, "lunch": fallback, "dinner": fallback, "snack": fallback}

    first = (attractions[0].location, attractions[0].name)
    middle = (attractions[len(attractions) // 2].location, attractions[len(attractions) // 2].name)
    last = (attractions[-1].location, attractions[-1].name)
    return {
        "breakfast": first,
        "lunch": middle,
        "dinner": last,
        "snack": middle,
    }


def _meal_anchor_for_type(
    meal_type: str,
    anchor_points: dict[str, tuple[Optional[Location], str]],
) -> tuple[Optional[Location], str]:
    return anchor_points.get(meal_type, anchor_points.get("snack", (None, "")))


def _select_nearest_unused_meal_poi(
    pois: list[Any],
    anchor_location: Optional[Location],
    used_meal_keys: set[str],
) -> Optional[Any]:
    candidates = [poi for poi in pois if _poi_dedupe_key(poi) not in used_meal_keys]
    if not candidates:
        return None
    if anchor_location is None:
        return candidates[0]
    return min(
        candidates,
        key=lambda poi: _location_distance_km(anchor_location, getattr(poi, "location", None)),
    )


def _poi_dedupe_key(poi: Any) -> str:
    poi_id = str(getattr(poi, "id", "") or "").strip().lower()
    if poi_id:
        return f"id:{poi_id}"
    name = _normalize_meal_name(str(getattr(poi, "name", "") or ""))
    address = _normalize_meal_name(str(getattr(poi, "address", "") or ""))
    return f"name:{name}|{address}"


def _meal_dedupe_key(meal: Meal) -> str:
    if meal.poi_id:
        return f"id:{meal.poi_id.strip().lower()}"
    return f"name:{_normalize_meal_name(meal.name)}|{_normalize_meal_name(meal.address or '')}"


def _normalize_meal_name(value: str) -> str:
    return "".join(str(value or "").lower().split())


def _location_distance_km(
    origin: Optional[Location],
    destination: Optional[Location],
) -> float:
    if origin is None or destination is None:
        return float("inf")
    return _haversine_km(
        origin.longitude,
        origin.latitude,
        destination.longitude,
        destination.latitude,
    )


def _meal_review_summary(meal: Meal, poi: Any, anchor_label: str) -> str:
    label = {
        "breakfast": "早餐",
        "lunch": "午餐",
        "dinner": "晚餐",
        "snack": "小吃",
    }.get(meal.type, meal.type)
    return f"{poi.name} 靠近 {anchor_label}，适合作为当天{label}安排，减少景点之间额外绕路。"


def _meal_recommended_reason(poi: Any, anchor_location: Optional[Location]) -> str:
    distance = _location_distance_km(anchor_location, getattr(poi, "location", None))
    if distance != float("inf"):
        return f"按当天景点位置筛选，距离约 {distance:.1f} 公里，适合顺路用餐。"
    return "按城市餐饮 POI 匹配得到，优先选择靠近当天景点的店铺。"


async def _apply_llm_meal_reasons_with_timeout(plan: TripPlan, city: str) -> None:
    try:
        await asyncio.wait_for(
            _apply_llm_meal_reasons(plan, city),
            timeout=35.0,
        )
    except asyncio.TimeoutError:
        print("LLM餐饮推荐理由生成超时，保留本地推荐理由")


async def _apply_llm_meal_reasons(plan: TripPlan, city: str) -> None:
    meal_items: list[tuple[str, Meal]] = []
    lines: list[str] = []

    for day in plan.days:
        for meal in day.meals:
            if not meal.name:
                continue
            key = str(len(meal_items) + 1)
            meal_items.append((key, meal))
            lines.append(
                f"{key}. {meal.type} | {meal.name} | {meal.address or '地址待补充'} | {meal.recommended_reason or '靠近当天景点，适合顺路用餐'}"
            )

    if not meal_items:
        return

    prompt = (
        "/no_think\n"
        "只输出最终答案，不要解释，不要思考过程。"
        "只返回JSON对象，格式必须是：{\"1\":\"理由\",\"2\":\"理由\"}。"
        "为旅行计划生成餐饮推荐理由，每条18-32字，保留距离信息，说明顺路或特色，不写推荐点评/路线。\n"
        f"城市：{city}\n餐饮：\n" + "\n".join(lines)
    )

    try:
        model = get_chat_model()
        if hasattr(model, "bind"):
            model = model.bind(max_tokens=min(900, max(220, len(meal_items) * 55)))
        response = await model.ainvoke(prompt)
        content = _llm_response_text(response)
        summaries = _parse_llm_meal_reason_map(content)

        updated_count = 0
        for key, meal in meal_items:
            summary = _clean_meal_reason(summaries.get(key, ""))
            if summary:
                meal.recommended_reason = summary
                updated_count += 1
        if updated_count:
            print(f"LLM餐饮推荐理由已更新: {updated_count}/{len(meal_items)}")
        else:
            print(f"LLM餐饮推荐理由未返回可用JSON，保留本地推荐理由: {content[:160]}")
    except Exception as exc:
        print(f"LLM餐饮推荐理由生成失败，保留本地推荐理由: {exc}")


def _parse_llm_meal_reason_map(content: str) -> dict[str, str]:
    text = content.strip()
    if not text:
        return {}

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return _parse_llm_meal_reason_lines(content)

    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return _parse_llm_meal_reason_lines(content)

    if isinstance(data, dict):
        direct = {
            str(key): str(value)
            for key, value in data.items()
            if isinstance(value, str) and value.strip()
        }
        if direct:
            return direct

        nested = data.get("reasons") or data.get("data") or data.get("items")
        if isinstance(nested, dict):
            return {
                str(key): str(value)
                for key, value in nested.items()
                if isinstance(value, str) and value.strip()
            }
        if isinstance(nested, list):
            return _parse_meal_reason_items(nested)

    return _parse_llm_meal_reason_lines(content)


def _parse_llm_meal_reason_lines(text: str) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("-").strip()
        if not line:
            continue
        for separator in ("=", "：", ":"):
            if separator not in line:
                continue
            key, reason = line.split(separator, 1)
            key = key.strip().lstrip("\"'").rstrip("\"'")
            reason = reason.strip().strip(",").strip().strip("\"'")
            if key.isdigit() and reason:
                reasons[key] = reason
            break
    return reasons


def _parse_meal_reason_items(items: list[Any]) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = item.get("id") or item.get("key") or item.get("index") or item.get("序号")
        reason = item.get("reason") or item.get("recommended_reason") or item.get("理由")
        if key is not None and isinstance(reason, str) and reason.strip():
            reasons[str(key)] = reason
    return reasons


def _llm_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts).strip()

    additional_kwargs = getattr(response, "additional_kwargs", {}) or {}
    reasoning_content = additional_kwargs.get("reasoning_content")
    if isinstance(reasoning_content, str):
        return reasoning_content.strip()

    return str(content or "").strip()


def _clean_meal_reason(content: str) -> Optional[str]:
    if not content:
        return None

    text = content.strip().strip('"').strip("'")
    for prefix in ("推荐理由：", "推荐理由:", "理由：", "理由:"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    if not text:
        return None

    return text[:80]


async def _enrich_hotel(hotel: Optional[Hotel], city: str) -> None:
    if hotel is None:
        return

    if hotel.location is None and hotel.address:
        location = await geocode_location(hotel.address, city=city)
        if location is not None:
            hotel.location = location

    if not hotel.review_summary:
        hotel.review_summary = f"{hotel.name} 适合落脚在行程覆盖区域内，方便第二天继续出发。"


async def _build_hotel_candidates(plan: TripPlan, request: TripRequest) -> list[Hotel]:
    attractions = [
        attraction
        for day in plan.days
        for attraction in day.attractions
        if attraction.location is not None
    ]
    keywords = [
        f"{request.city} {request.accommodation or '酒店'}",
        f"{request.city} 酒店",
    ]

    poi_candidates: list[Any] = []
    for keyword in dict.fromkeys(keywords):
        try:
            poi_candidates.extend(await search_pois(keyword, request.city, citylimit=True, limit=12))
        except Exception as exc:
            print(f"酒店候选搜索失败: {keyword} {exc}")

    seen_keys: set[str] = set()
    hotels: list[Hotel] = []
    for poi in sorted(
        poi_candidates,
        key=lambda item: _average_distance_to_attractions(getattr(item, "location", None), attractions),
    ):
        key = _hotel_dedupe_key(poi)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        hotels.append(_poi_to_hotel(poi, request, attractions))
        if len(hotels) >= 4:
            break

    if len(hotels) < 4:
        for day in plan.days:
            if day.hotel is None:
                continue
            key = _normalize_meal_name(f"{day.hotel.name}|{day.hotel.address}")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            hotels.append(day.hotel.model_copy(deep=True))
            if len(hotels) >= 4:
                break

    return hotels[:4]


def _poi_to_hotel(poi: Any, request: TripRequest, attractions: list[Attraction]) -> Hotel:
    average_distance = _average_distance_to_attractions(getattr(poi, "location", None), attractions)
    if average_distance == float("inf"):
        distance_text = "位于城市行程区域内"
    else:
        distance_text = f"距离规划景点平均约 {average_distance:.1f} 公里"

    name = str(getattr(poi, "name", "") or "酒店")
    return Hotel(
        name=name,
        address=str(getattr(poi, "address", "") or ""),
        location=getattr(poi, "location", None),
        price_range=request.accommodation or "以预订平台实时价格为准",
        rating="暂无",
        distance=distance_text,
        type=str(getattr(poi, "type", "") or "酒店"),
        review_summary=f"{name} {distance_text}，适合作为多日行程的住宿候选。",
        estimated_cost=0,
    )


def _average_distance_to_attractions(
    location: Optional[Location],
    attractions: list[Attraction],
) -> float:
    if location is None or not attractions:
        return float("inf")

    distances = [
        _location_distance_km(location, attraction.location)
        for attraction in attractions
        if attraction.location is not None
    ]
    if not distances:
        return float("inf")
    return sum(distances) / len(distances)


def _hotel_dedupe_key(poi: Any) -> str:
    poi_id = str(getattr(poi, "id", "") or "").strip().lower()
    if poi_id:
        return f"id:{poi_id}"
    return _normalize_meal_name(
        f"{getattr(poi, 'name', '')}|{getattr(poi, 'address', '')}"
    )


def _meal_search_keyword(meal_type: str, city: str, meal_name: str) -> str:
    if meal_name and meal_name not in {"早餐", "午餐", "晚餐", "小吃"}:
        return meal_name

    if meal_type == "breakfast":
        return f"{city} 早餐店"
    if meal_type == "lunch":
        return f"{city} 本地餐厅"
    if meal_type == "dinner":
        return f"{city} 特色餐厅"
    return f"{city} 小吃"


def _build_snippet(name: str, city: str) -> ExperienceSnippet:
    lower_name = name.lower()
    if "博物" in name or "museum" in lower_name:
        return ExperienceSnippet(
            title=name,
            photo_spots=["主馆入口", "核心展厅", "标志性建筑外立面", "文创商店"],
            review_summary=f"{name} 适合想了解 {city} 历史文化的游客，节奏较稳，建议提前确认开放时间。",
            visit_tips=["建议提前预约", "优先上午到访", "适合和周边同类景点串联", "留出拍照时间"],
        )
    if any(keyword in name for keyword in ("公园", "湖", "海", "岛", "山", "浴场", "海滨")):
        return ExperienceSnippet(
            title=name,
            photo_spots=["观景台", "临水步道", "日落位置", "栈道转角"],
            review_summary=f"{name} 更适合拍照和放松，整体节奏轻松，建议安排在下午或傍晚体验。",
            visit_tips=["建议傍晚到访", "注意防风", "适合和咖啡店一起安排", "预留拍照时间"],
        )
    return ExperienceSnippet(
        title=name,
        photo_spots=["入口广场", "地标建筑", "最佳取景角度", "主视觉背景墙"],
        review_summary=f"{name} 是 {city} 较典型的打卡点，适合先看图片和点评摘要再决定停留时长。",
        visit_tips=["建议提前看开放时间", "优先结合附近路线安排", "适合和餐饮酒店一起串联", "拍照时注意避开高峰"],
    )


def _build_route_tip(
    previous_location: Optional[Location],
    previous_label: str,
    current_attraction: Optional[Attraction] = None,
    current_location: Optional[Location] = None,
    current_label: str = "",
) -> Optional[str]:
    if current_attraction is not None:
        current_location = current_attraction.location
        current_label = current_attraction.name

    if previous_location is None or current_location is None:
        return None

    distance_km = _haversine_km(
        previous_location.longitude,
        previous_location.latitude,
        current_location.longitude,
        current_location.latitude,
    )

    if distance_km < 1.2:
        mode = "步行"
    elif distance_km < 4:
        mode = "打车或地铁"
    else:
        mode = "地铁/打车"

    return f"从 {previous_label} 到 {current_label} 约 {distance_km:.1f} 公里，建议 {mode}。"


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius_km * atan2(sqrt(a), sqrt(1 - a))
