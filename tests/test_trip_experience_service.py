import pytest

from backend.app.models.schemas import Attraction, Location, Meal, POIInfo
from backend.app.services import trip_experience_service


class FakeResponse:
    content = '{"明显陵文化旅游景区": "适合想了解地方文化的游客，核心看点集中，游览节奏不赶；如果时间有限，可重点看主入口和代表性展陈。"}'


class FakeChatModel:
    calls = 0

    async def ainvoke(self, prompt):
        self.calls += 1
        assert "明显陵文化旅游景区" in prompt
        assert "不要假装读取了真实网友评论" in prompt
        return FakeResponse()


@pytest.mark.asyncio
async def test_enrich_attraction_uses_llm_review_summary(monkeypatch):
    attraction = Attraction(
        name="明显陵文化旅游景区",
        address="湖北省钟祥市",
        location=Location(longitude=112.58, latitude=31.17),
        visit_duration=120,
        description="明代帝陵文化遗产景区。",
        poi_id="",
    )

    monkeypatch.setattr(
        trip_experience_service,
        "get_chat_model",
        lambda: FakeChatModel(),
    )

    summary = await trip_experience_service._build_llm_review_summary(
        attraction,
        city="钟祥",
    )

    assert summary == "适合想了解地方文化的游客，核心看点集中，游览节奏不赶；如果时间有限，可重点看主入口和代表性展陈。"
    assert "比较典型的打卡点" not in summary


@pytest.mark.asyncio
async def test_batch_review_summary_calls_llm_once(monkeypatch):
    attractions = [
        Attraction(
            name="明显陵文化旅游景区",
            address="湖北省钟祥市",
            location=Location(longitude=112.58, latitude=31.17),
            visit_duration=120,
            description="明代帝陵文化遗产景区。",
            poi_id="",
        ),
        Attraction(
            name="莫愁湖国家湿地公园",
            address="湖北省钟祥市",
            location=Location(longitude=112.60, latitude=31.18),
            visit_duration=90,
            description="湖泊湿地景观。",
            poi_id="",
        ),
    ]

    class BatchResponse:
        content = (
            '{"明显陵文化旅游景区": "适合历史文化游，核心看点清晰，建议放慢节奏参观。",'
            '"莫愁湖国家湿地公园": "适合轻松散步和看湖景，傍晚体验更好，注意预留拍照时间。"}'
        )

    class BatchChatModel:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, prompt):
            self.calls += 1
            assert "明显陵文化旅游景区" in prompt
            assert "莫愁湖国家湿地公园" in prompt
            return BatchResponse()

    fake_model = BatchChatModel()
    monkeypatch.setattr(
        trip_experience_service,
        "get_chat_model",
        lambda: fake_model,
    )

    summaries = await trip_experience_service._build_llm_review_summaries(
        attractions,
        city="钟祥",
    )

    assert fake_model.calls == 1
    assert set(summaries) == {"明显陵文化旅游景区", "莫愁湖国家湿地公园"}


def test_build_photo_spot_details_returns_four_unique_spots():
    snippet = trip_experience_service._build_snippet("莫愁湖国家湿地公园", "钟祥")

    details = trip_experience_service._build_photo_spot_details(snippet)
    names = [detail.name for detail in details]

    assert len(details) == 4
    assert len(set(names)) == 4


@pytest.mark.asyncio
async def test_enrich_meals_prefers_nearby_and_avoids_duplicates(monkeypatch):
    attraction = Attraction(
        name="Anchor",
        address="Anchor Address",
        location=Location(longitude=120.0, latitude=30.0),
        visit_duration=90,
        description="Anchor attraction",
    )
    meals = [
        Meal(type="breakfast", name="早餐"),
        Meal(type="lunch", name="午餐"),
        Meal(type="dinner", name="晚餐"),
    ]

    shared_near = POIInfo(
        id="shared",
        name="Shared Near Restaurant",
        type="餐饮服务",
        address="Near address",
        location=Location(longitude=120.001, latitude=30.001),
    )
    breakfast_unique = POIInfo(
        id="breakfast",
        name="Breakfast Unique",
        type="餐饮服务",
        address="Breakfast address",
        location=Location(longitude=120.002, latitude=30.002),
    )
    lunch_unique = POIInfo(
        id="lunch",
        name="Lunch Unique",
        type="餐饮服务",
        address="Lunch address",
        location=Location(longitude=120.003, latitude=30.003),
    )
    dinner_unique = POIInfo(
        id="dinner",
        name="Dinner Unique",
        type="餐饮服务",
        address="Dinner address",
        location=Location(longitude=120.004, latitude=30.004),
    )

    async def fake_search_pois(keyword, city, citylimit=True, limit=12):
        if "早餐" in keyword:
            return [shared_near, breakfast_unique]
        if "本地餐厅" in keyword:
            return [shared_near, lunch_unique]
        if "特色餐厅" in keyword:
            return [shared_near, dinner_unique]
        return []

    monkeypatch.setattr(trip_experience_service, "search_pois", fake_search_pois)

    await trip_experience_service._enrich_meals(
        meals,
        city="TestCity",
        attractions=[attraction],
        hotel=None,
        used_meal_keys=set(),
    )

    assert meals[0].name == "Shared Near Restaurant"
    assert meals[1].name == "Lunch Unique"
    assert meals[2].name == "Dinner Unique"
    assert len({meal.poi_id for meal in meals}) == 3
    assert all(meal.location is not None for meal in meals)
    assert all(meal.review_summary for meal in meals)
    assert all(meal.recommended_reason for meal in meals)
