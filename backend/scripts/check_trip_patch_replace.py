import asyncio
import json

from app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    TripPlan,
)
from app.services.trip_plan_patch_service import (
    apply_trip_patch_operations,
    build_patch_operations_from_message,
)


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
                    make_attraction("中国国家博物馆"),
                ],
                meals=[
                    make_meal("breakfast"),
                    make_meal("lunch"),
                    make_meal("dinner"),
                ],
            )
        ],
        weather_info=[],
        overall_suggestions="整体建议",
    )


async def main():
    plan = make_plan()
    message = "把第一天的故宫换成天坛"

    print("User message:")
    print(message)

    operations = await build_patch_operations_from_message(
        message=message,
        plan=plan,
    )

    print("\nPatch operations:")
    print(json.dumps(operations, ensure_ascii=False, indent=2))

    updated = await apply_trip_patch_operations(
        plan=plan,
        operations=operations,
    )

    print("\nUpdated attractions:")
    for attraction in updated.days[0].attractions:
        print(
            f"- {attraction.name} | 地址: {attraction.address} | "
            f"坐标: {attraction.location.longitude},{attraction.location.latitude} | "
            f"POI ID: {attraction.poi_id}"
        )


if __name__ == "__main__":
    asyncio.run(main())