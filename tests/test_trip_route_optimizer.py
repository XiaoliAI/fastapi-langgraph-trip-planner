from backend.app.models.schemas import (
    Attraction,
    DayPlan,
    Location,
    TripPlan,
)
from backend.app.services.trip_route_optimizer import optimize_trip_attraction_routes


def make_attraction(name: str, longitude: float, latitude: float) -> Attraction:
    return Attraction(
        name=name,
        address=f"{name}地址",
        location=Location(longitude=longitude, latitude=latitude),
        visit_duration=60,
        description=f"{name}描述",
    )


def make_day(index: int, attractions: list[Attraction]) -> DayPlan:
    return DayPlan(
        date=f"2026-08-{12 + index:02d}",
        day_index=index,
        description=f"第{index + 1}天",
        transportation="公共交通",
        accommodation="酒店",
        attractions=attractions,
        meals=[],
    )


def test_optimize_trip_attraction_routes_groups_nearby_attractions_by_day():
    west_a = make_attraction("西区景点A", 116.10, 39.90)
    west_b = make_attraction("西区景点B", 116.11, 39.91)
    east_a = make_attraction("东区景点A", 116.50, 39.90)
    east_b = make_attraction("东区景点B", 116.51, 39.91)

    plan = TripPlan(
        city="北京",
        start_date="2026-08-11",
        end_date="2026-08-12",
        days=[
            DayPlan(
                date="2026-08-11",
                day_index=0,
                description="第一天",
                transportation="公共交通",
                accommodation="酒店",
                attractions=[west_a, east_a],
                meals=[],
            ),
            DayPlan(
                date="2026-08-12",
                day_index=1,
                description="第二天",
                transportation="公共交通",
                accommodation="酒店",
                attractions=[west_b, east_b],
                meals=[],
            ),
        ],
        overall_suggestions="",
    )

    optimized = optimize_trip_attraction_routes(plan)

    day_one_names = {attraction.name for attraction in optimized.days[0].attractions}
    day_two_names = {attraction.name for attraction in optimized.days[1].attractions}

    assert day_one_names == {"西区景点A", "西区景点B"}
    assert day_two_names == {"东区景点A", "东区景点B"}
    assert [len(day.attractions) for day in optimized.days] == [2, 2]


def test_optimize_trip_attraction_routes_preserves_plan_when_location_missing():
    west_a = make_attraction("西区景点A", 116.10, 39.90)
    no_location = west_a.model_copy(deep=True)
    no_location.name = "无坐标景点"
    no_location.location = None

    plan = TripPlan(
        city="北京",
        start_date="2026-08-11",
        end_date="2026-08-11",
        days=[
            DayPlan(
                date="2026-08-11",
                day_index=0,
                description="第一天",
                transportation="公共交通",
                accommodation="酒店",
                attractions=[west_a, no_location],
                meals=[],
            ),
        ],
        overall_suggestions="",
    )

    optimized = optimize_trip_attraction_routes(plan)

    assert [attraction.name for attraction in optimized.days[0].attractions] == [
        "西区景点A",
        "无坐标景点",
    ]


def test_optimize_trip_attraction_routes_keeps_exactly_two_attractions_per_day():
    attractions = [
        make_attraction("西区景点A", 116.10, 39.90),
        make_attraction("西区景点B", 116.11, 39.91),
        make_attraction("东区景点A", 116.50, 39.90),
        make_attraction("东区景点B", 116.51, 39.91),
        make_attraction("远郊景点A", 117.20, 40.20),
        make_attraction("远郊景点B", 117.21, 40.21),
    ]

    plan = TripPlan(
        city="北京",
        start_date="2026-08-11",
        end_date="2026-08-12",
        days=[
            DayPlan(
                date="2026-08-11",
                day_index=0,
                description="第一天",
                transportation="公共交通",
                accommodation="酒店",
                attractions=attractions[:3],
                meals=[],
            ),
            DayPlan(
                date="2026-08-12",
                day_index=1,
                description="第二天",
                transportation="公共交通",
                accommodation="酒店",
                attractions=attractions[3:],
                meals=[],
            ),
        ],
        overall_suggestions="",
    )

    optimized = optimize_trip_attraction_routes(plan)

    assert [len(day.attractions) for day in optimized.days] == [2, 2]


def test_optimize_trip_attraction_routes_removes_duplicate_place_variants():
    olympic_sailing = make_attraction("青岛奥帆海洋文化旅游区", 120.390, 36.060)
    olympic_sailing_alias = make_attraction("奥林匹克帆船中心·奥帆海洋文化旅游区", 120.391, 36.061)
    beach = make_attraction("青岛第三海水浴场", 120.360, 36.055)
    beach_duplicate = make_attraction("青岛第三海水浴场", 120.360, 36.055)
    pier = make_attraction("栈桥", 120.320, 36.064)
    badaguan = make_attraction("八大关风景区", 120.355, 36.050)

    olympic_sailing.poi_id = "poi-1"
    olympic_sailing_alias.poi_id = "poi-2"
    beach.poi_id = "poi-3"
    beach_duplicate.poi_id = "poi-3"

    plan = TripPlan(
        city="青岛",
        start_date="2026-08-12",
        end_date="2026-08-13",
        days=[
            DayPlan(
                date="2026-08-12",
                day_index=0,
                description="第一天",
                transportation="公共交通",
                accommodation="酒店",
                attractions=[olympic_sailing, beach, olympic_sailing_alias],
                meals=[],
            ),
            DayPlan(
                date="2026-08-13",
                day_index=1,
                description="第二天",
                transportation="公共交通",
                accommodation="酒店",
                attractions=[beach_duplicate, pier, badaguan],
                meals=[],
            ),
        ],
        overall_suggestions="",
    )

    optimized = optimize_trip_attraction_routes(plan)
    names = [attraction.name for day in optimized.days for attraction in day.attractions]

    assert names.count("青岛奥帆海洋文化旅游区") == 1
    assert "奥林匹克帆船中心·奥帆海洋文化旅游区" not in names
    assert names.count("青岛第三海水浴场") == 1


def test_optimize_trip_attraction_routes_does_not_keep_duplicates_when_candidates_are_short():
    square = make_attraction("五四广场", 120.382, 36.067)
    sailing = make_attraction("青岛奥帆海洋文化旅游区", 120.390, 36.060)
    beach = make_attraction("青岛第三海水浴场", 120.360, 36.055)
    badaguan = make_attraction("八大关风景区", 120.355, 36.050)
    pier = make_attraction("栈桥", 120.320, 36.064)

    square_duplicate = square.model_copy(deep=True)
    sailing_duplicate = sailing.model_copy(deep=True)
    beach_duplicate = beach.model_copy(deep=True)

    plan = TripPlan(
        city="青岛",
        start_date="2026-08-11",
        end_date="2026-08-14",
        days=[
            make_day(0, [square, sailing]),
            make_day(1, [square_duplicate, beach]),
            make_day(2, [badaguan, sailing_duplicate]),
            make_day(3, [pier, beach_duplicate]),
        ],
        overall_suggestions="",
    )

    optimized = optimize_trip_attraction_routes(plan)
    names = [attraction.name for day in optimized.days for attraction in day.attractions]

    assert len(names) == len(set(names))
    assert set(names) == {
        "五四广场",
        "青岛奥帆海洋文化旅游区",
        "青岛第三海水浴场",
        "八大关风景区",
        "栈桥",
    }


def test_optimize_trip_attraction_routes_fills_four_day_plan_when_backfilled_candidates_exist():
    attractions = [
        make_attraction("五四广场", 120.382, 36.067),
        make_attraction("青岛奥帆海洋文化旅游区", 120.390, 36.060),
        make_attraction("五四广场", 120.382, 36.067),
        make_attraction("青岛第三海水浴场", 120.360, 36.055),
        make_attraction("八大关风景区", 120.355, 36.050),
        make_attraction("青岛奥帆海洋文化旅游区", 120.390, 36.060),
        make_attraction("栈桥", 120.320, 36.064),
        make_attraction("小鱼山公园", 120.331, 36.065),
        make_attraction("信号山公园", 120.326, 36.070),
        make_attraction("青岛啤酒博物馆", 120.343, 36.087),
    ]

    plan = TripPlan(
        city="青岛",
        start_date="2026-08-12",
        end_date="2026-08-15",
        days=[
            make_day(0, attractions[0:2]),
            make_day(1, attractions[2:4]),
            make_day(2, attractions[4:6]),
            make_day(3, attractions[6:]),
        ],
        overall_suggestions="",
    )

    optimized = optimize_trip_attraction_routes(plan)
    names = [attraction.name for day in optimized.days for attraction in day.attractions]

    assert [len(day.attractions) for day in optimized.days] == [2, 2, 2, 2]
    assert len(names) == len(set(names))


def test_optimize_trip_attraction_routes_keeps_five_day_plan_when_enough_unique_attractions_exist():
    attractions = [
        make_attraction("五四广场", 120.382, 36.067),
        make_attraction("青岛奥帆海洋文化旅游区", 120.390, 36.060),
        make_attraction("青岛第三海水浴场", 120.360, 36.055),
        make_attraction("八大关风景区", 120.355, 36.050),
        make_attraction("栈桥", 120.320, 36.064),
        make_attraction("小鱼山公园", 120.331, 36.065),
        make_attraction("信号山公园", 120.326, 36.070),
        make_attraction("青岛啤酒博物馆", 120.343, 36.087),
        make_attraction("鲁迅公园", 120.335, 36.060),
        make_attraction("太平山景区", 120.345, 36.075),
    ]

    plan = TripPlan(
        city="青岛",
        start_date="2026-08-12",
        end_date="2026-08-16",
        days=[
            make_day(0, attractions[0:2]),
            make_day(1, attractions[2:4]),
            make_day(2, attractions[4:6]),
            make_day(3, attractions[6:8]),
            make_day(4, attractions[8:10]),
        ],
        overall_suggestions="",
    )

    optimized = optimize_trip_attraction_routes(plan)
    names = [attraction.name for day in optimized.days for attraction in day.attractions]

    assert [len(day.attractions) for day in optimized.days] == [2, 2, 2, 2, 2]
    assert len(names) == len(set(names))
