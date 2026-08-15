from backend.app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    TripPlan,
)
from backend.app.services.trip_plan_patch_service import (
    _parse_json_array,
    apply_trip_patch_operations,
    build_plan_patch_prompt,
)
import pytest
from backend.app.models.schemas import POIInfo

def make_attraction(name: str) -> Attraction:
    return Attraction(
        name=name,
        address="北京市",
        location=Location(longitude=116.397, latitude=39.916),
        visit_duration=120,
        description=f"{name}介绍",
    )


def make_meal(meal_type: str) -> Meal:
    return Meal(
        type=meal_type,
        name=f"{meal_type}餐厅",
        estimated_cost=50,
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
                    make_attraction("故宫博物院"),
                    make_attraction("天坛公园"),
                ],
                meals=[
                    make_meal("breakfast"),
                    make_meal("lunch"),
                    make_meal("dinner"),
                ],
            ),
            DayPlan(
                date="2026-08-02",
                day_index=2,
                description="第二天行程",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[
                    make_attraction("颐和园"),
                ],
                meals=[
                    make_meal("breakfast"),
                    make_meal("lunch"),
                    make_meal("dinner"),
                ],
            ),
        ],
        weather_info=[],
        overall_suggestions="整体建议",
    )


def test_build_plan_patch_prompt_contains_current_plan():
    plan = make_plan()

    prompt = build_plan_patch_prompt(
        message="把第一天的故宫换成天坛",
        plan=plan,
    )

    assert "故宫博物院" in prompt
    assert "天坛公园" in prompt
    assert "颐和园" in prompt
    assert "remove_attraction" in prompt
    assert "replace_attraction" in prompt


def test_parse_json_array_from_llm_text():
    text = """
```json
[
  {
    "operation": "remove_attraction",
    "day_index": 1,
    "attraction_name": "故宫博物院"
  }
]
"""
    operations = _parse_json_array(text)

    assert operations == [
    {
        "operation": "remove_attraction",
        "day_index": 1,
        "attraction_name": "故宫博物院",
    }
]

def test_parse_json_array_returns_empty_list_for_invalid_text():
    operations = _parse_json_array("这不是 JSON")

    assert operations == []
@pytest.mark.asyncio
async def test_apply_remove_attraction_operation():
    plan = make_plan()

    updated = await apply_trip_patch_operations(
        plan=plan,
        operations=[
            {
                "operation": "remove_attraction",
                "day_index": 1,
                "attraction_name": "故宫博物院",
            }
        ],
    )

    day_one_names = [attraction.name for attraction in updated.days[0].attractions]

    assert day_one_names == ["天坛公园"]

@pytest.mark.asyncio
async def test_apply_replace_attraction_operation():
    plan = make_plan()

    async def fake_search_pois(keywords: str, city: str, limit: int = 5):
        return []

    updated = await apply_trip_patch_operations(
        plan=plan,
        operations=[
            {
                "operation": "replace_attraction",
                "day_index": 1,
                "old_attraction_name": "故宫博物院",
                "new_attraction_name": "天坛",
            }
        ],
        search_pois_func=fake_search_pois,
    )

    day_one_names = [attraction.name for attraction in updated.days[0].attractions]

    assert "故宫博物院" not in day_one_names
    assert "天坛" in day_one_names
    assert "天坛公园" in day_one_names

@pytest.mark.asyncio
async def test_apply_replace_attraction_operation_uses_poi_search_result():
    plan = make_plan()

    async def fake_search_pois(keywords: str, city: str, limit: int = 5):
        assert keywords == "天坛"
        assert city == "北京"

        return [
            POIInfo(
                id="poi-tiantan",
                name="天坛公园",
                type="风景名胜",
                address="北京市东城区天坛东路甲1号",
                location=Location(longitude=116.410886, latitude=39.881949),
            )
        ]

    updated = await apply_trip_patch_operations(
        plan=plan,
        operations=[
            {
                "operation": "replace_attraction",
                "day_index": 1,
                "old_attraction_name": "故宫博物院",
                "new_attraction_name": "天坛",
            }
        ],
        search_pois_func=fake_search_pois,
    )

    day_one_attractions = updated.days[0].attractions
    day_one_names = [attraction.name for attraction in day_one_attractions]

    assert "故宫博物院" not in day_one_names
    assert "天坛公园" in day_one_names

    tiantan = next(
        attraction for attraction in day_one_attractions
        if attraction.name == "天坛公园"
    )

    assert tiantan.address == "北京市东城区天坛东路甲1号"
    assert tiantan.location.longitude == 116.410886
    assert tiantan.location.latitude == 39.881949
    assert tiantan.poi_id == "poi-tiantan"