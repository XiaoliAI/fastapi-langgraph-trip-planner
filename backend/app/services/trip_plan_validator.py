from datetime import datetime, timedelta

from ..models.schemas import TripPlan, TripRequest


REQUIRED_MEAL_TYPES = {"breakfast", "lunch", "dinner"}


def validate_trip_plan(
    plan: TripPlan,
    request: TripRequest,
    weather_text: str = "",
) -> list[str]:
    errors: list[str] = []

    if plan.city != request.city:
        errors.append(f"城市不匹配: expected {request.city}, got {plan.city}")

    if plan.start_date != request.start_date:
        errors.append(f"开始日期不匹配: expected {request.start_date}, got {plan.start_date}")

    if plan.end_date != request.end_date:
        errors.append(f"结束日期不匹配: expected {request.end_date}, got {plan.end_date}")

    if len(plan.days) != request.travel_days:
        errors.append(
            f"行程天数不匹配: expected {request.travel_days}, got {len(plan.days)}"
        )

    expected_dates = _expected_dates(request.start_date, request.travel_days)

    for index, day in enumerate(plan.days):
        if index < len(expected_dates) and day.date != expected_dates[index]:
            errors.append(
                f"日期不匹配: day_index {index}, expected {expected_dates[index]}, got {day.date}"
            )

        if day.day_index != index:
            errors.append(
                f"day_index 不匹配: expected {index}, got {day.day_index}"
            )

        meal_types = {meal.type for meal in day.meals}
        missing_meals = REQUIRED_MEAL_TYPES - meal_types
        for meal_type in sorted(missing_meals):
            errors.append(f"{day.date} 缺少餐食: {meal_type}")

        for attraction in day.attractions:
            if attraction.location is None:
                errors.append(f"景点缺少坐标: {attraction.name}")

    _validate_duplicate_attractions(plan, errors)
    _validate_budget(plan, errors)
    _validate_weather_source(plan, weather_text, errors)

    return errors


def _validate_duplicate_attractions(
    plan: TripPlan,
    errors: list[str],
) -> None:
    seen_attractions = set()

    for day in plan.days:
        for attraction in day.attractions:
            name = attraction.name
            if name in seen_attractions:
                errors.append(f"景点重复: {name}")
            seen_attractions.add(name)


def _validate_budget(
    plan: TripPlan,
    errors: list[str],
) -> None:
    if plan.budget is None:
        return

    expected_total = (
        plan.budget.total_attractions
        + plan.budget.total_hotels
        + plan.budget.total_meals
        + plan.budget.total_transportation
    )

    if plan.budget.total != expected_total:
        errors.append(
            f"预算总额不匹配: expected {expected_total}, got {plan.budget.total}"
        )

#验证天气来源
def _validate_weather_source(
    plan: TripPlan,
    weather_text: str,
    errors: list[str],
) -> None:
    if not plan.weather_info or not weather_text:
        return

    for weather in plan.weather_info:
        if weather.date not in weather_text:
            errors.append(f"天气日期不在来源数据中: {weather.date}")


def _expected_dates(start_date: str, travel_days: int) -> list[str]:
    start = datetime.strptime(start_date, "%Y-%m-%d")
    return [
        (start + timedelta(days=index)).strftime("%Y-%m-%d")
        for index in range(travel_days)
    ]