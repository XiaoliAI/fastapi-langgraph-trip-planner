import asyncio
import json

from app.agents.langgraph_trip_planner import LangGraphTripPlanner
from app.models.schemas import TripRequest


async def main():
    planner = LangGraphTripPlanner()

    request = TripRequest(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
        free_text_input="少走路",
    )

    plan = await planner.plan_trip(request)

    print("repr city:")
    print(repr(plan.city))

    print("\nplain city:")
    print(plan.city)

    print("\njson ensure_ascii=True:")
    print(json.dumps(plan.model_dump(), ensure_ascii=True, indent=2))

    print("\njson ensure_ascii=False:")
    print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())