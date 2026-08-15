import pytest
from backend.app.models.schemas import WeatherInfo
from backend.app.models.schemas import (
    Attraction,
    Budget,
    DayPlan,
    Location,
    Meal,
    TripPlan,
    TripRequest,
)
from backend.app.services.trip_plan_validator import validate_trip_plan


def make_request() -> TripRequest:
    return TripRequest(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        travel_days=2,
        transportation="公共交通",
        accommodation="经济型酒店",
        preferences=["历史文化"],
    )


def make_valid_plan() -> TripPlan:
    return TripPlan(
        city="北京",
        start_date="2026-08-01",
        end_date="2026-08-02",
        days=[
            DayPlan(
                date="2026-08-01",
                day_index=0,
                description="第1天",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[
                    Attraction(
                        name="故宫博物院",
                        address="景山前街4号",
                        location=Location(longitude=116.397005, latitude=39.919278),
                        visit_duration=180,
                        description="历史文化景点",
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
                description="第2天",
                transportation="公共交通",
                accommodation="经济型酒店",
                attractions=[
                    Attraction(
                        name="中国国家博物馆",
                        address="东长安街16号",
                        location=Location(longitude=116.397755, latitude=39.903182),
                        visit_duration=150,
                        description="博物馆",
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
        overall_suggestions="建议提前预约。",
        budget=Budget(
            total_attractions=60,
            total_hotels=300,
            total_meals=300,
            total_transportation=40,
            total=700,
        ),
    )


def test_validate_trip_plan_accepts_valid_plan():
    errors = validate_trip_plan(make_valid_plan(), make_request())

    assert errors == []


def test_validate_trip_plan_detects_wrong_day_count():
    plan = make_valid_plan()
    plan.days = plan.days[:1]

    errors = validate_trip_plan(plan, make_request())

    assert "行程天数不匹配: expected 2, got 1" in errors


def test_validate_trip_plan_detects_duplicate_attractions():
    plan = make_valid_plan()
    plan.days[1].attractions[0].name = "故宫博物院"

    errors = validate_trip_plan(plan, make_request())

    assert "景点重复: 故宫博物院" in errors


def test_validate_trip_plan_detects_missing_meal_type():
    plan = make_valid_plan()
    plan.days[0].meals = [
        Meal(type="breakfast", name="早餐"),
        Meal(type="lunch", name="午餐"),
    ]

    errors = validate_trip_plan(plan, make_request())

    assert "2026-08-01 缺少餐食: dinner" in errors


def test_validate_trip_plan_detects_budget_total_mismatch():
    plan = make_valid_plan()
    plan.budget.total = 999

    errors = validate_trip_plan(plan, make_request())

    assert "预算总额不匹配: expected 700, got 999" in errors

def test_validate_trip_plan_rejects_weather_dates_not_in_source_text():
    plan = make_valid_plan()
    plan.weather_info = [
        WeatherInfo(
            date="2026-08-01",
            day_weather="雷阵雨",
            night_weather="多云",
            day_temp=33,
            night_temp=24,
            wind_direction="南",
            wind_power="1-3",
        )
    ]

    errors = validate_trip_plan(
        plan,
        make_request(),
        weather_text="1. 日期: 2026-07-27 | 白天: 多云 33℃ | 夜间: 多云 24℃",
    )

    assert "天气日期不在来源数据中: 2026-08-01" in errors