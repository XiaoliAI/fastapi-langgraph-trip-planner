import pytest
import asyncio
import time
from backend.app.agents.langgraph_trip_planner import (
    collect_travel_context,
    parse_weather_info_from_text,
    parse_trip_plan_from_text,
)
from backend.app.agents import langgraph_trip_planner
from backend.app.agents.langgraph_trip_planner import build_trip_planner_graph
from backend.app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    POIInfo,
    TripPlan,
    TripRequest,
)


@pytest.mark.asyncio
async def test_trip_planner_graph_generates_plan(monkeypatch):
    async def fake_search_pois_text(*args, **kwargs):
        return "1. 故宫博物院 | 地址: 景山前街4号"

    async def fake_get_weather_text(city: str):
        return "1. 日期: 2026-08-01 | 白天: 多云 33℃ 南风 1-3级 | 夜间: 多云 24℃ 南风 1-3级"

    async def fake_search_pois(keywords: str, city: str, citylimit: bool = True, limit: int = 10):
        return [
            POIInfo(
                id=f"poi-{keywords}",
                name=keywords,
                type="景点",
                address=f"{city}{keywords}地址",
                location=Location(longitude=116.397005, latitude=39.919278),
            )
        ]

    async def fake_generate_trip_plan_from_context(
        request: TripRequest,
        attractions_text: str,
        weather_text: str,
        hotels_text: str,
    ) -> TripPlan:
        assert request.city == "北京"
        assert "故宫博物院" in attractions_text
        assert "多云" in weather_text
        assert "故宫博物院" in hotels_text or hotels_text

        return TripPlan(
            city="北京",
            start_date="2026-08-01",
            end_date="2026-08-02",
            days=[
                DayPlan(
                    date="2026-08-01",
                    day_index=0,
                    description="第1天行程",
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
                            description="北京代表性历史文化景点。",
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
                    description="第2天行程",
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
                            description="综合性国家博物馆。",
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
            overall_suggestions="建议提前预约热门景点。",
            budget=None,
        )

    monkeypatch.setattr(
        langgraph_trip_planner.amap_langchain_service,
        "search_pois_text",
        fake_search_pois_text,
    )

    monkeypatch.setattr(
        langgraph_trip_planner.amap_langchain_service,
        "get_weather_text",
        fake_get_weather_text,
    )

    monkeypatch.setattr(
        langgraph_trip_planner.amap_langchain_service,
        "search_pois",
        fake_search_pois,
    )

    monkeypatch.setattr(
        langgraph_trip_planner,
        "generate_trip_plan_from_context",
        fake_generate_trip_plan_from_context,
    )

    graph = build_trip_planner_graph()

    request = TripRequest(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )

    result = await graph.ainvoke({"request": request})

    assert result["plan"].city == "北京"
    assert len(result["plan"].days) == 2
    assert result["plan"].days[0].attractions[0].name == "故宫博物院"
    assert result["plan"].days[0].attractions[0].poi_id == "poi-故宫博物院"
    assert result["plan"].weather_info[0].day_temp == 33
    assert result["validation_errors"] == []


@pytest.mark.asyncio
async def test_enrich_plan_with_poi_metadata_fills_missing_poi_id(monkeypatch):
    async def fake_search_pois(keywords: str, city: str, citylimit: bool = True, limit: int = 10):
        assert keywords == "明显陵文化旅游景区"
        assert city == "钟祥"
        return [
            POIInfo(
                id="B0TESTMINGXIANLING",
                name="明显陵文化旅游景区",
                type="风景名胜",
                address="湖北省荆门市钟祥市明显陵",
                location=Location(longitude=112.587311, latitude=31.205118),
            )
        ]

    monkeypatch.setattr(
        langgraph_trip_planner.amap_langchain_service,
        "search_pois",
        fake_search_pois,
    )

    request = TripRequest(
        city="钟祥",
        start_date="2026-08-01",
        end_date="2026-08-01",
        travel_days=1,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )
    plan = TripPlan(
        city="钟祥",
        start_date="2026-08-01",
        end_date="2026-08-01",
        days=[
            DayPlan(
                date="2026-08-01",
                day_index=0,
                description="第1天行程",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[
                    Attraction(
                        name="明显陵文化旅游景区",
                        address="",
                        location=Location(longitude=0, latitude=0),
                        visit_duration=120,
                        description="明显陵文化旅游景区。",
                        category="景点",
                    )
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

    enriched = await langgraph_trip_planner.enrich_plan_with_poi_metadata(plan, request)

    attraction = enriched.days[0].attractions[0]
    assert attraction.poi_id == "B0TESTMINGXIANLING"
    assert attraction.address == "湖北省荆门市钟祥市明显陵"


@pytest.mark.asyncio
async def test_enrich_plan_with_poi_metadata_refills_missing_location_when_poi_id_exists(monkeypatch):
    monkeypatch.setattr(
        langgraph_trip_planner,
        "_get_poi_location_from_id",
        lambda poi_id: Location(longitude=112.587311, latitude=31.205118),
    )

    request = TripRequest(
        city="钟祥",
        start_date="2026-08-01",
        end_date="2026-08-01",
        travel_days=1,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )
    plan = TripPlan(
        city="钟祥",
        start_date="2026-08-01",
        end_date="2026-08-01",
        days=[
            DayPlan(
                date="2026-08-01",
                day_index=0,
                description="第1天行程",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[
                    Attraction(
                        name="明显陵文化旅游景区",
                        address="",
                        location=Location(longitude=0, latitude=0),
                        visit_duration=120,
                        description="明显陵文化旅游景区。",
                        category="景点",
                        poi_id="B0TESTMINGXIANLING",
                    )
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

    enriched = await langgraph_trip_planner.enrich_plan_with_poi_metadata(plan, request)

    attraction = enriched.days[0].attractions[0]
    assert attraction.poi_id == "B0TESTMINGXIANLING"
    assert attraction.location.longitude == 112.587311
    assert attraction.location.latitude == 31.205118


@pytest.mark.asyncio
async def test_collect_travel_context_runs_independent_fetches_concurrently(monkeypatch):
    async def fake_search_pois_text(*args, **kwargs):
        await asyncio.sleep(0.05)
        if kwargs["keywords"] == "经济型酒店":
            return "酒店候选"
        return "景点候选"

    async def fake_get_weather_text(city: str):
        await asyncio.sleep(0.05)
        return "天气信息"

    monkeypatch.setattr(
        langgraph_trip_planner.amap_langchain_service,
        "search_pois_text",
        fake_search_pois_text,
    )
    monkeypatch.setattr(
        langgraph_trip_planner.amap_langchain_service,
        "get_weather_text",
        fake_get_weather_text,
    )

    request = TripRequest(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )

    started_at = time.perf_counter()
    result = await collect_travel_context({"request": request})
    elapsed = time.perf_counter() - started_at

    assert result["attractions_text"] == "景点候选"
    assert result["weather_text"] == "天气信息"
    assert result["hotels_text"] == "酒店候选"
    assert elapsed < 0.12
    
def test_parse_trip_plan_from_json_text():
    response_text = """
    下面是旅行计划：
    ```json
    {
      "city": "北京",
      "start_date": "2026-08-01",
      "end_date": "2026-08-02",
      "days": [
        {
          "date": "2026-08-01",
          "day_index": 0,
          "description": "第1天行程",
          "transportation": "公共交通",
          "accommodation": "经济型酒店",
          "attractions": [
            {
              "name": "故宫博物院",
              "address": "景山前街4号",
              "location": {
                "longitude": 116.397005,
                "latitude": 39.919278
              },
              "visit_duration": 180,
              "description": "北京代表性历史文化景点。",
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
          "description": "第2天行程",
          "transportation": "公共交通",
          "accommodation": "经济型酒店",
          "attractions": [
            {
              "name": "中国国家博物馆",
              "address": "东长安街16号",
              "location": {
                "longitude": 116.397755,
                "latitude": 39.903182
              },
              "visit_duration": 180,
              "description": "综合性国家博物馆。",
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
      "overall_suggestions": "建议提前预约热门景点。",
      "budget": null
    }
    ```
    """

    plan = parse_trip_plan_from_text(response_text)

    assert plan.city == "北京"
    assert len(plan.days) == 2
    assert plan.days[0].attractions[0].name == "故宫博物院"


def test_parse_weather_info_from_text_keeps_only_trip_dates():
    request = TripRequest(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )

    weather_infos = parse_weather_info_from_text(
        weather_text=(
            "1. 日期: 2026-08-01 | 白天: 晴 33℃ 南风 1-3级 | 夜间: 多云 25℃ 南风 1-3级\n"
            "2. 日期: 2026-08-03 | 白天: 雨 20℃ 北风 3-4级 | 夜间: 雨 18℃ 北风 3-4级"
        ),
        request=request,
    )

    assert len(weather_infos) == 1
    assert weather_infos[0].date == "2026-08-01"
    assert weather_infos[0].day_weather == "晴"
    assert weather_infos[0].night_weather == "多云"
    assert weather_infos[0].day_temp == 33
    assert weather_infos[0].night_temp == 25
    assert weather_infos[0].wind_direction == "南"
    assert weather_infos[0].wind_power == "1-3"


def test_parse_trip_plan_from_text_raises_for_missing_json():
    with pytest.raises(ValueError, match="未找到 TripPlan JSON"):
        parse_trip_plan_from_text("这里没有 JSON")


from backend.app.agents.langgraph_trip_planner import build_trip_plan_prompt


def test_build_trip_plan_prompt_contains_context():
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

    prompt = build_trip_plan_prompt(
        request=request,
        attractions_text="1. 故宫博物院 | 坐标: 116.397005,39.919278",
        weather_text="1. 日期: 2026-07-27 | 白天: 多云 33℃",
        hotels_text="1. 北京前门酒店 | 坐标: 116.391,39.900",
    )

    assert "北京" in prompt
    assert "2026-08-01" in prompt
    assert "故宫博物院" in prompt
    assert "北京前门酒店" in prompt
    assert "少走路" in prompt
    assert "只返回 JSON" in prompt
