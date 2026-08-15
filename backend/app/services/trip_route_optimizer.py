"""Optimize trip attraction grouping by geographic distance."""

from __future__ import annotations

import re
import logging
from math import atan2, cos, radians, sin, sqrt

from ..models.schemas import Attraction, TripPlan

logger = logging.getLogger(__name__)


def optimize_trip_attraction_routes(plan: TripPlan) -> TripPlan:
    """Assign nearby attractions to each day without reintroducing duplicates."""
    updated_plan = plan.model_copy(deep=True)
    original_attractions = [
        attraction
        for day in updated_plan.days
        for attraction in day.attractions
    ]
    attractions = _deduplicate_attractions(original_attractions)
    required_count = len(updated_plan.days) * 2

    if len(attractions) < required_count:
        _assign_two_attractions_per_day(updated_plan, _order_nearby_attractions(attractions))
        _log_optimization_result(updated_plan, len(original_attractions), len(attractions), required_count)
        return updated_plan

    if not all(_has_location(attraction) for attraction in attractions):
        selected = attractions[:required_count]
        _assign_two_attractions_per_day(updated_plan, selected)
        _log_optimization_result(updated_plan, len(original_attractions), len(attractions), required_count)
        return updated_plan

    pairs = _build_nearby_pairs(attractions, len(updated_plan.days))
    selected = [attraction for pair in pairs for attraction in pair]
    _assign_two_attractions_per_day(updated_plan, selected)
    _log_optimization_result(updated_plan, len(original_attractions), len(attractions), required_count)
    return updated_plan


def _log_optimization_result(
    plan: TripPlan,
    original_count: int,
    unique_count: int,
    required_count: int,
) -> None:
    logger.info(
        "Trip route optimization: original=%s unique=%s final=%s required=%s daily_counts=%s",
        original_count,
        unique_count,
        sum(len(day.attractions) for day in plan.days),
        required_count,
        [len(day.attractions) for day in plan.days],
    )


def _deduplicate_attractions(attractions: list[Attraction]) -> list[Attraction]:
    unique_attractions: list[Attraction] = []
    seen_poi_ids: set[str] = set()
    seen_names: set[str] = set()

    for attraction in attractions:
        poi_id = (attraction.poi_id or "").strip()
        normalized_name = _normalize_attraction_name(attraction.name)

        if poi_id and poi_id in seen_poi_ids:
            continue
        if normalized_name and normalized_name in seen_names:
            continue
        if _has_alias_name_match(attraction, unique_attractions):
            continue

        unique_attractions.append(attraction)
        if poi_id:
            seen_poi_ids.add(poi_id)
        if normalized_name:
            seen_names.add(normalized_name)

    return unique_attractions


def _has_alias_name_match(
    attraction: Attraction,
    existing_attractions: list[Attraction],
) -> bool:
    normalized_name = _normalize_attraction_name(attraction.name)
    if len(normalized_name) < 2:
        return False

    for existing in existing_attractions:
        existing_name = _normalize_attraction_name(existing.name)
        if len(existing_name) < 2:
            continue
        if normalized_name in existing_name or existing_name in normalized_name:
            return True

    return False


def _normalize_attraction_name(name: str) -> str:
    text = str(name or "").lower()
    text = re.sub(r"[·\-\s\(\)（）【】\[\]、，。,.]+", "", text)
    removable_words = (
        "青岛",
        "北京",
        "文化旅游区",
        "旅游区",
        "风景区",
        "景区",
        "公园",
        "中心",
        "广场",
        "打卡点",
        "游客中心",
        "售票处",
        "入口",
        "出口",
    )
    for word in removable_words:
        text = text.replace(word, "")
    return text


def _assign_two_attractions_per_day(plan: TripPlan, attractions: list[Attraction]) -> None:
    cursor = 0
    for day in plan.days:
        day.attractions = attractions[cursor:cursor + 2]
        cursor += 2


def _build_nearby_pairs(
    attractions: list[Attraction],
    day_count: int,
) -> list[tuple[Attraction, Attraction]]:
    remaining = attractions[:]
    pairs: list[tuple[Attraction, Attraction]] = []

    for _ in range(day_count):
        if len(remaining) < 2:
            break

        first, second = min(
            (
                (first, second)
                for index, first in enumerate(remaining)
                for second in remaining[index + 1:]
            ),
            key=lambda pair: _distance_km(pair[0], pair[1]),
        )
        pairs.append((first, second))
        remaining.remove(first)
        remaining.remove(second)

    return pairs


def _order_nearby_attractions(attractions: list[Attraction]) -> list[Attraction]:
    if len(attractions) < 3 or not all(_has_location(attraction) for attraction in attractions):
        return attractions

    remaining = attractions[:]
    current = _western_then_southern_attraction(remaining)
    ordered = [current]
    remaining.remove(current)

    while remaining:
        current = min(
            remaining,
            key=lambda attraction: _distance_km(ordered[-1], attraction),
        )
        ordered.append(current)
        remaining.remove(current)

    return ordered


def _western_then_southern_attraction(attractions: list[Attraction]) -> Attraction:
    return min(
        attractions,
        key=lambda attraction: (
            attraction.location.longitude,
            attraction.location.latitude,
        ),
    )


def _has_location(attraction: Attraction) -> bool:
    return (
        attraction.location is not None
        and attraction.location.longitude is not None
        and attraction.location.latitude is not None
    )


def _distance_km(first: Attraction, second: Attraction) -> float:
    lon1 = first.location.longitude
    lat1 = first.location.latitude
    lon2 = second.location.longitude
    lat2 = second.location.latitude

    radius_km = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * radius_km * atan2(sqrt(a), sqrt(1 - a))
