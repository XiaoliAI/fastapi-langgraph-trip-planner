import asyncio

from app.agents.langgraph_trip_planner import build_trip_planner_graph
from app.models.schemas import TripRequest


async def main():
    graph = build_trip_planner_graph()

    request = TripRequest(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
        free_text_input="",
    )

    result = await graph.ainvoke({"request": request})

    print("Attractions text:")
    print(result.get("attractions_text"))

    print("\nWeather text:")
    print(result.get("weather_text"))


if __name__ == "__main__":
    asyncio.run(main())