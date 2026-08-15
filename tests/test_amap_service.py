import unittest
import os
from unittest.mock import patch

os.environ["DEBUG"] = "false"
from backend.app.services.amap_service import AmapService
from backend.app.agents.trip_planner_agent import MultiAgentTripPlanner
from backend.app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    Meal,
    POIInfo,
    TripPlan,
    TripRequest,
    WeatherInfo,
)


class FakeMCPTool:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def run(self, payload):
        self.calls.append(payload)
        return self.response


class FakeAmapLookupService:
    def __init__(self, pois=None, location=None):
        self.pois = pois or []
        self.location = location
        self.poi_calls = []
        self.geocode_calls = []

    def search_poi(self, keywords, city, citylimit=True):
        self.poi_calls.append((keywords, city, citylimit))
        return self.pois.pop(0) if self.pois else []

    def geocode(self, address, city=None):
        self.geocode_calls.append((address, city))
        return self.location


class AmapServiceTests(unittest.TestCase):
    def test_search_poi_parses_mcp_text_search_results(self):
        raw_response = """
工具 'maps_text_search' 执行结果:
{"status":"1","pois":[{"id":"B0FFTEST01","name":"西湖风景名胜区","type":"风景名胜","address":"杭州市西湖区龙井路1号","location":"120.143222,30.236064","tel":"0571-87977767"}]}
"""
        service = AmapService.__new__(AmapService)
        service.mcp_tool = FakeMCPTool(raw_response)

        pois = service.search_poi("风景名胜", "杭州")

        self.assertEqual(len(pois), 1)
        self.assertEqual(pois[0].name, "西湖风景名胜区")
        self.assertEqual(pois[0].address, "杭州市西湖区龙井路1号")
        self.assertEqual(pois[0].location.longitude, 120.143222)
        self.assertEqual(pois[0].location.latitude, 30.236064)
        self.assertEqual(
            service.mcp_tool.calls[0]["arguments"],
            {"keywords": "风景名胜", "city": "杭州", "citylimit": "true"},
        )


    def test_geocode_parses_city_center_from_mcp_result(self):
        raw_response = """
tool result:
{"status":"1","geocodes":[{"formatted_address":"湖北省荆门市钟祥市","location":"112.587831,31.16735"}]}
"""
        service = AmapService.__new__(AmapService)
        service.mcp_tool = FakeMCPTool(raw_response)

        location = service.geocode("\u949f\u7965")

        self.assertIsNotNone(location)
        self.assertAlmostEqual(location.longitude, 112.587831)
        self.assertAlmostEqual(location.latitude, 31.16735)

    def test_search_poi_uses_http_fallback_when_mcp_result_cannot_be_parsed(self):
        service = AmapService.__new__(AmapService)
        service.mcp_tool = FakeMCPTool("工具返回了不可解析的内容")
        expected_pois = [
            POIInfo(
                id="DL001",
                name="星海广场",
                type="风景名胜",
                address="辽宁省大连市沙河口区",
                location=Location(longitude=121.586, latitude=38.881),
            )
        ]
        calls = []

        def fake_http_search(keywords, city, citylimit):
            calls.append((keywords, city, citylimit))
            return expected_pois

        service._search_poi_http = fake_http_search

        pois = service.search_poi("大连 景点", "", False)

        self.assertEqual(pois, expected_pois)
        self.assertEqual(calls, [("大连 景点", "", False)])


class SchemaNormalizationTests(unittest.TestCase):
    def test_weather_info_accepts_numeric_wind_values(self):
        weather = WeatherInfo(
            date="2026-07-07",
            day_weather="sunny",
            night_weather="cloudy",
            day_temp=30,
            night_temp=24,
            wind_direction=90,
            wind_power=5,
        )

        self.assertEqual(weather.wind_direction, "90")
        self.assertEqual(weather.wind_power, "5")

    def test_parse_response_keeps_llm_meals_when_wind_power_is_numeric(self):
        planner = MultiAgentTripPlanner.__new__(MultiAgentTripPlanner)
        request = TripRequest(
            city="\u9752\u5c9b",
            start_date="2026-07-07",
            end_date="2026-07-07",
            travel_days=1,
            transportation="public transit",
            accommodation="hotel",
            preferences=[],
        )
        response = """
```json
{
  "city": "\u9752\u5c9b",
  "start_date": "2026-07-07",
  "end_date": "2026-07-07",
  "days": [
    {
      "date": "2026-07-07",
      "day_index": 0,
      "description": "day plan",
      "transportation": "public transit",
      "accommodation": "hotel",
      "attractions": [],
      "meals": [
        {"type": "breakfast", "name": "\u9752\u5c9b\u65e9\u9910", "description": "\u9505\u8d34", "estimated_cost": 20},
        {"type": "lunch", "name": "\u9752\u5c9b\u5348\u9910", "description": "\u6d77\u9c9c", "estimated_cost": 80},
        {"type": "dinner", "name": "\u9752\u5c9b\u665a\u9910", "description": "\u5564\u9152\u5c4b", "estimated_cost": 100}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "2026-07-07",
      "day_weather": "\u6674",
      "night_weather": "\u591a\u4e91",
      "day_temp": 30,
      "night_temp": 24,
      "wind_direction": "\u5357\u98ce",
      "wind_power": 5
    }
  ],
  "overall_suggestions": "ok"
}
```
"""

        plan = planner._parse_response(response, request, [])

        self.assertEqual(plan.days[0].meals[0].name, "\u9752\u5c9b\u65e9\u9910")
        self.assertEqual(plan.days[0].meals[1].description, "\u6d77\u9c9c")
        self.assertEqual(plan.weather_info[0].wind_power, "5")


class TripPlannerPoiTests(unittest.TestCase):
    def test_apply_poi_locations_replaces_llm_example_beijing_coordinates(self):
        planner = MultiAgentTripPlanner.__new__(MultiAgentTripPlanner)
        request = TripRequest(
            city="杭州",
            start_date="2026-07-10",
            end_date="2026-07-10",
            travel_days=1,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=["自然风光"],
        )
        plan = TripPlan(
            city="杭州",
            start_date="2026-07-10",
            end_date="2026-07-10",
            overall_suggestions="示例计划",
            days=[
                DayPlan(
                    date="2026-07-10",
                    day_index=0,
                    description="第1天",
                    transportation="公共交通",
                    accommodation="经济型酒店",
                    attractions=[
                        Attraction(
                            name="景点名称",
                            address="详细地址",
                            location=Location(longitude=116.397128, latitude=39.916527),
                            visit_duration=120,
                            description="LLM 示例景点",
                        )
                    ],
                    meals=[
                        Meal(type="breakfast", name="早餐"),
                        Meal(type="lunch", name="午餐"),
                        Meal(type="dinner", name="晚餐"),
                    ],
                )
            ],
        )
        pois = [
            POIInfo(
                id="B0FFTEST01",
                name="西湖风景名胜区",
                type="风景名胜",
                address="杭州市西湖区龙井路1号",
                location=Location(longitude=120.143222, latitude=30.236064),
                tel="0571-87977767",
            )
        ]

        enriched = planner._apply_poi_locations(plan, pois, request)

        attraction = enriched.days[0].attractions[0]
        self.assertEqual(attraction.name, "西湖风景名胜区")
        self.assertEqual(attraction.address, "杭州市西湖区龙井路1号")
        self.assertEqual(attraction.location.longitude, 120.143222)
        self.assertEqual(attraction.location.latitude, 30.236064)

    def test_apply_poi_locations_does_not_reuse_poi_across_days(self):
        planner = MultiAgentTripPlanner.__new__(MultiAgentTripPlanner)
        request = TripRequest(
            city="大连",
            start_date="2026-07-10",
            end_date="2026-07-11",
            travel_days=2,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=[],
        )
        repeated_attraction = Attraction(
            name="星海广场",
            address="旧地址",
            location=Location(longitude=0, latitude=0),
            visit_duration=120,
            description="重复景点",
        )
        plan = TripPlan(
            city="大连",
            start_date="2026-07-10",
            end_date="2026-07-11",
            overall_suggestions="测试",
            days=[
                DayPlan(
                    date="2026-07-10",
                    day_index=0,
                    description="第1天",
                    transportation="公共交通",
                    accommodation="经济型酒店",
                    attractions=[repeated_attraction.model_copy(deep=True)],
                    meals=[],
                ),
                DayPlan(
                    date="2026-07-11",
                    day_index=1,
                    description="第2天",
                    transportation="公共交通",
                    accommodation="经济型酒店",
                    attractions=[repeated_attraction.model_copy(deep=True)],
                    meals=[],
                ),
            ],
        )
        pois = [
            POIInfo(
                id="DL001",
                name="星海广场",
                type="风景名胜",
                address="中山路572号",
                location=Location(longitude=121.587922, latitude=38.882006),
            ),
            POIInfo(
                id="DL002",
                name="大连金石滩国家旅游度假区",
                type="风景名胜",
                address="金石路65号",
                location=Location(longitude=121.995346, latitude=39.096033),
            ),
        ]

        enriched = planner._apply_poi_locations(plan, pois, request)

        names = [day.attractions[0].name for day in enriched.days]
        self.assertEqual(names, ["星海广场", "大连金石滩国家旅游度假区"])
        self.assertEqual(len(names), len(set(names)))

    def test_apply_poi_locations_removes_duplicate_when_no_unused_replacement_exists(self):
        planner = MultiAgentTripPlanner.__new__(MultiAgentTripPlanner)
        request = TripRequest(
            city="大连",
            start_date="2026-07-10",
            end_date="2026-07-11",
            travel_days=2,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=[],
        )
        plan = TripPlan(
            city="大连",
            start_date="2026-07-10",
            end_date="2026-07-11",
            overall_suggestions="测试",
            days=[
                DayPlan(
                    date="2026-07-10",
                    day_index=0,
                    description="第1天",
                    transportation="公共交通",
                    accommodation="经济型酒店",
                    attractions=[
                        Attraction(
                            name="星海广场",
                            address="中山路572号",
                            location=Location(longitude=121.587922, latitude=38.882006),
                            visit_duration=120,
                            description="景点",
                        )
                    ],
                    meals=[],
                ),
                DayPlan(
                    date="2026-07-11",
                    day_index=1,
                    description="第2天",
                    transportation="公共交通",
                    accommodation="经济型酒店",
                    attractions=[
                        Attraction(
                            name="星海广场",
                            address="中山路572号",
                            location=Location(longitude=121.587922, latitude=38.882006),
                            visit_duration=120,
                            description="重复景点",
                        )
                    ],
                    meals=[],
                ),
            ],
        )
        pois = [
            POIInfo(
                id="DL001",
                name="星海广场",
                type="风景名胜",
                address="中山路572号",
                location=Location(longitude=121.587922, latitude=38.882006),
            )
        ]

        enriched = planner._apply_poi_locations(plan, pois, request)

        names = [attraction.name for day in enriched.days for attraction in day.attractions]
        self.assertEqual(names, ["星海广场"])

    def test_fallback_plan_does_not_repeat_pois_when_pois_are_insufficient(self):
        planner = MultiAgentTripPlanner.__new__(MultiAgentTripPlanner)
        request = TripRequest(
            city="钟祥",
            start_date="2026-07-10",
            end_date="2026-07-12",
            travel_days=3,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=[],
        )
        pois = [
            POIInfo(
                id="ZX001",
                name="钟祥市博物馆",
                type="博物馆",
                address="莫愁湖路28号",
                location=Location(longitude=112.61881, latitude=31.182896),
            ),
            POIInfo(
                id="ZX002",
                name="明显陵文化旅游景区",
                type="风景名胜",
                address="明显陵",
                location=Location(longitude=112.6426, latitude=31.2061),
            ),
        ]

        plan = planner._create_fallback_plan(request, pois)

        names = [attraction.name for day in plan.days for attraction in day.attractions]
        self.assertEqual(len(names), len(set(names)))

    def test_fallback_plan_uses_destination_city_center_instead_of_beijing(self):
        planner = MultiAgentTripPlanner.__new__(MultiAgentTripPlanner)
        request = TripRequest(
            city="杭州",
            start_date="2026-07-10",
            end_date="2026-07-10",
            travel_days=1,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=[],
        )

        plan = planner._create_fallback_plan(request)

        location = plan.days[0].attractions[0].location
        self.assertAlmostEqual(location.longitude, 120.1551, places=3)
        self.assertAlmostEqual(location.latitude, 30.2741, places=3)

    def test_parse_failure_fallback_keeps_existing_pois(self):
        planner = MultiAgentTripPlanner.__new__(MultiAgentTripPlanner)
        request = TripRequest(
            city="\u949f\u7965",
            start_date="2026-07-10",
            end_date="2026-07-10",
            travel_days=1,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=[],
        )
        pois = [
            POIInfo(
                id="ZX001",
                name="明显陵",
                type="风景名胜",
                address="湖北省荆门市钟祥市明显陵",
                location=Location(longitude=112.6426, latitude=31.2061),
            )
        ]

        plan = planner._parse_response("not json", request, pois)

        attraction = plan.days[0].attractions[0]
        self.assertEqual(attraction.name, "明显陵")
        self.assertIn("明显陵", attraction.description)
        self.assertIn("湖北省荆门市钟祥市明显陵", attraction.description)
        self.assertEqual(attraction.location.longitude, 112.6426)

    def test_unknown_city_fallback_geocodes_destination_center(self):
        planner = MultiAgentTripPlanner.__new__(MultiAgentTripPlanner)
        request = TripRequest(
            city="\u949f\u7965",
            start_date="2026-07-10",
            end_date="2026-07-10",
            travel_days=1,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=[],
        )
        fake_service = FakeAmapLookupService(location=Location(longitude=112.587831, latitude=31.16735))

        with patch("backend.app.agents.trip_planner_agent.get_amap_service", return_value=fake_service):
            plan = planner._create_fallback_plan(request)

        location = plan.days[0].attractions[0].location
        self.assertAlmostEqual(location.longitude, 112.587831, places=3)
        self.assertAlmostEqual(location.latitude, 31.16735, places=3)

    def test_search_real_pois_tries_city_name_when_preference_search_is_empty(self):
        planner = MultiAgentTripPlanner.__new__(MultiAgentTripPlanner)
        request = TripRequest(
            city="\u949f\u7965",
            start_date="2026-07-10",
            end_date="2026-07-10",
            travel_days=1,
            transportation="公共交通",
            accommodation="经济型酒店",
            preferences=["自然风光"],
        )
        expected_poi = POIInfo(
            id="ZX001",
            name="明显陵",
            type="风景名胜",
            address="湖北省荆门市钟祥市明显陵",
            location=Location(longitude=112.6426, latitude=31.2061),
        )
        fake_service = FakeAmapLookupService(pois=[[], [expected_poi]])

        with patch("backend.app.agents.trip_planner_agent.get_amap_service", return_value=fake_service):
            pois = planner._search_real_pois(request)

        self.assertEqual(pois, [expected_poi])
        self.assertEqual(fake_service.poi_calls[0], ("自然风光", "\u949f\u7965", True))
        self.assertEqual(fake_service.poi_calls[1], ("\u949f\u7965 景点", "", False))


if __name__ == "__main__":
    unittest.main()
