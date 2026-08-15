import pytest

from backend.app.api.routes import poi


class FakeUnsplashService:
    def get_photo_url(self, query: str):
        return None

    def get_photo_urls(self, query: str, limit: int = 4):
        return []


@pytest.mark.asyncio
async def test_get_attraction_photo_returns_placeholder_without_external_photo(monkeypatch):
    monkeypatch.setattr(
        poi,
        "get_unsplash_service",
        lambda: FakeUnsplashService(),
    )
    monkeypatch.setattr(
        poi,
        "_search_amap_photo_urls_http",
        lambda name, city=None, limit=10: [],
    )

    response = await poi.get_attraction_photo(name="江汉路")

    assert response["success"] is True
    assert response["data"]["source"] == "placeholder"
    assert len(response["data"]["photo_urls"]) == 1
    assert response["data"]["photo_url"].startswith("data:image/svg+xml")


@pytest.mark.asyncio
async def test_get_attraction_photo_prefers_amap_http_photo(monkeypatch):
    monkeypatch.setattr(
        poi,
        "get_unsplash_service",
        lambda: FakeUnsplashService(),
    )
    monkeypatch.setattr(
        poi,
        "_get_amap_poi_detail_http",
        lambda poi_id: {
            "pois": [
                {
                    "name": "江汉路步行街",
                    "typecode": "110200",
                    "photos": [{"url": "https://example.com/a.jpg"}],
                }
            ]
        },
    )
    monkeypatch.setattr(
        poi,
        "_search_amap_photo_urls_http",
        lambda name, city=None, limit=10: [],
    )

    response = await poi.get_attraction_photo(name="江汉路", poi_id="B001")

    assert response["success"] is True
    assert response["data"]["source"] == "amap"
    assert response["data"]["photo_urls"] == ["https://example.com/a.jpg"]
    assert response["data"]["photo_url"] == "https://example.com/a.jpg"


@pytest.mark.asyncio
async def test_get_attraction_photo_merges_amap_and_unsplash_photos(monkeypatch):
    class MixedUnsplashService(FakeUnsplashService):
        def get_photo_urls(self, query: str, limit: int = 4):
            return [
                "https://example.com/b.jpg",
                "https://example.com/c.jpg",
            ]

    monkeypatch.setattr(
        poi,
        "get_unsplash_service",
        lambda: MixedUnsplashService(),
    )
    monkeypatch.setattr(
        poi,
        "_get_amap_poi_detail_http",
        lambda poi_id: {
            "pois": [
                {
                    "name": "江汉路步行街",
                    "typecode": "110200",
                    "photos": [{"url": "https://example.com/a.jpg"}],
                }
            ]
        },
    )
    monkeypatch.setattr(
        poi,
        "_search_amap_photo_urls_http",
        lambda name, city=None, limit=10: [],
    )

    response = await poi.get_attraction_photo(name="江汉路", poi_id="B001")

    assert response["success"] is True
    assert response["data"]["source"] == "amap+unsplash"
    assert response["data"]["photo_urls"] == [
        "https://example.com/a.jpg",
        "https://example.com/b.jpg",
        "https://example.com/c.jpg",
    ]


@pytest.mark.asyncio
async def test_get_attraction_photo_searches_amap_by_name_before_unsplash(monkeypatch):
    class UnusedUnsplashService(FakeUnsplashService):
        def get_photo_urls(self, query: str, limit: int = 4):
            return ["https://example.com/unused.jpg"]

    monkeypatch.setattr(
        poi,
        "get_unsplash_service",
        lambda: UnusedUnsplashService(),
    )
    monkeypatch.setattr(
        poi,
        "_search_amap_photo_urls_http",
        lambda name, city=None, limit=10: [
            "https://example.com/amap-search-1.jpg",
            "https://example.com/amap-search-2.jpg",
        ],
    )

    response = await poi.get_attraction_photo(name="江汉路", city="武汉")

    assert response["success"] is True
    assert response["data"]["source"] == "amap+unsplash"
    assert response["data"]["city"] == "武汉"
    assert response["data"]["photo_urls"][:2] == [
        "https://example.com/amap-search-1.jpg",
        "https://example.com/amap-search-2.jpg",
    ]


def test_extract_amap_scenic_photo_urls_filters_hotel_pois():
    detail = {
        "pois": [
            {
                "name": "江汉路步行街",
                "typecode": "110200",
                "photos": [{"url": "https://example.com/scenic.jpg", "title": "街区夜景"}],
            },
            {
                "name": "江汉路酒店",
                "typecode": "100000",
                "photos": [{"url": "https://example.com/hotel.jpg", "title": "酒店大堂"}],
            },
        ]
    }

    urls = poi._extract_amap_scenic_photo_urls(detail, target_name="江汉路")

    assert urls == ["https://example.com/scenic.jpg"]


def test_extract_amap_scenic_photo_urls_filters_low_quality_and_wrong_scene_photos():
    detail = {
        "pois": [
            {
                "name": "江汉路步行街",
                "typecode": "110200",
                "photos": [
                    {"url": "https://example.com/thumb_80.jpg", "title": "街区夜景"},
                    {"url": "https://example.com/room.jpg", "title": "酒店客房"},
                    {"url": "https://example.com/view.jpg", "title": "步行街入口"},
                ],
            },
        ]
    }

    urls = poi._extract_amap_scenic_photo_urls(detail, target_name="江汉路")

    assert urls == ["https://example.com/view.jpg"]
