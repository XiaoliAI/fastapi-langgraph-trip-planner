"""POI相关API路由"""

import json

from fastapi import APIRouter, HTTPException
from html import escape
from pydantic import BaseModel, Field
from typing import Any, List, Optional
from urllib.parse import quote, urlencode
from urllib.request import urlopen

from ...config import get_settings
from ...services.amap_service import get_amap_service
from ...services.unsplash_service import get_unsplash_service

router = APIRouter(prefix="/poi", tags=["POI"])


class POIDetailResponse(BaseModel):
    """POI详情响应"""
    success: bool
    message: str
    data: Optional[dict] = None


@router.get(
    "/detail/{poi_id}",
    response_model=POIDetailResponse,
    summary="获取POI详情",
    description="根据POI ID获取详细信息,包括图片"
)
async def get_poi_detail(poi_id: str):
    """
    获取POI详情
    
    Args:
        poi_id: POI ID
        
    Returns:
        POI详情响应
    """
    try:
        amap_service = get_amap_service()
        
        # 调用高德地图POI详情API
        result = amap_service.get_poi_detail(poi_id)
        
        return POIDetailResponse(
            success=True,
            message="获取POI详情成功",
            data=result
        )
        
    except Exception as e:
        print(f"❌ 获取POI详情失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取POI详情失败: {str(e)}"
        )


@router.get(
    "/search",
    summary="搜索POI",
    description="根据关键词搜索POI"
)
async def search_poi(keywords: str, city: str = "北京"):
    """
    搜索POI

    Args:
        keywords: 搜索关键词
        city: 城市名称

    Returns:
        搜索结果
    """
    try:
        amap_service = get_amap_service()
        result = amap_service.search_poi(keywords, city)

        return {
            "success": True,
            "message": "搜索成功",
            "data": result
        }

    except Exception as e:
        print(f"❌ 搜索POI失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"搜索POI失败: {str(e)}"
        )


@router.get(
    "/photo",
    summary="获取景点图片",
    description="优先从高德地图获取景点图片,高德不足时使用外部图片服务兜底"
)
async def get_attraction_photo(
    name: str,
    poi_id: Optional[str] = None,
    city: Optional[str] = None,
    category: Optional[str] = None,
):
    """
    获取景点图片

    Args:
        name: 景点名称

    Returns:
        图片URL
    """
    try:
        amap_photo_urls: list[str] = []
        is_meal_photo = category == "meal"
        is_hotel_photo = category == "hotel"

        if poi_id:
            detail = _get_amap_poi_detail_http(poi_id)
            if is_meal_photo:
                amap_photo_urls = _extract_amap_food_photo_urls(detail, target_name=name)
            elif is_hotel_photo:
                amap_photo_urls = _extract_amap_hotel_photo_urls(detail, target_name=name)
            else:
                amap_photo_urls = _extract_amap_scenic_photo_urls(detail, target_name=name)

        if len(amap_photo_urls) < 10:
            if is_meal_photo:
                amap_search_photo_urls = _search_amap_food_photo_urls_http(
                    name=name,
                    city=city,
                    limit=10 - len(amap_photo_urls),
                )
            elif is_hotel_photo:
                amap_search_photo_urls = _search_amap_hotel_photo_urls_http(
                    name=name,
                    city=city,
                    limit=10 - len(amap_photo_urls),
                )
            else:
                amap_search_photo_urls = _search_amap_photo_urls_http(
                    name=name,
                    city=city,
                    limit=10 - len(amap_photo_urls),
                )
            amap_photo_urls = _merge_photo_urls(
                amap_photo_urls,
                amap_search_photo_urls,
                limit=10,
            )

        unsplash_photo_urls: list[str] = []
        if not is_meal_photo and not is_hotel_photo and len(amap_photo_urls) < 10:
            unsplash_service = get_unsplash_service()
            unsplash_photo_urls = unsplash_service.get_photo_urls(
                f"{city or ''} {name} 景点",
                limit=10 - len(amap_photo_urls),
            )
            if not unsplash_photo_urls:
                unsplash_photo_urls = unsplash_service.get_photo_urls(
                    name,
                    limit=10 - len(amap_photo_urls),
                )

        photo_urls = _merge_photo_urls(amap_photo_urls, unsplash_photo_urls, limit=10)

        if not photo_urls:
            placeholder = _placeholder_photo_url(name, category=category)
            photo_urls = [placeholder]

        if amap_photo_urls and unsplash_photo_urls:
            source = "amap+unsplash"
        elif amap_photo_urls:
            source = "amap"
        elif unsplash_photo_urls:
            source = "unsplash"
        else:
            source = "placeholder"

        return {
            "success": True,
            "message": "获取图片成功",
            "data": {
                "name": name,
                "poi_id": poi_id,
                "city": city,
                "category": category,
                "photo_urls": photo_urls,
                "photo_url": photo_urls[0],
                "source": source,
            }
        }

    except Exception as e:
        print(f"❌ 获取景点图片失败: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取景点图片失败: {str(e)}"
        )


def _get_amap_poi_detail_http(poi_id: str) -> dict[str, Any]:
    settings = get_settings()
    if not settings.amap_api_key:
        return {}

    params = {
        "key": settings.amap_api_key,
        "id": poi_id,
        "extensions": "all",
    }

    try:
        query = urlencode(params)
        with urlopen(f"https://restapi.amap.com/v3/place/detail?{query}", timeout=6) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        print(f"高德HTTP POI详情获取失败: {exc}")
        return {}


def _search_amap_photo_urls_http(
    name: str,
    city: Optional[str] = None,
    limit: int = 10,
) -> list[str]:
    settings = get_settings()
    if not settings.amap_api_key or not name or limit <= 0:
        return []

    params: dict[str, Any] = {
        "key": settings.amap_api_key,
        "keywords": name,
        "offset": 20,
        "page": 1,
        "extensions": "all",
    }
    if city:
        params["city"] = city
        params["citylimit"] = "true"

    try:
        query = urlencode(params)
        with urlopen(f"https://restapi.amap.com/v3/place/text?{query}", timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _merge_photo_urls(
            _extract_amap_scenic_photo_urls(data, target_name=name),
            limit=limit,
        )
    except Exception as exc:
        print(f"高德HTTP POI图片搜索失败: {exc}")
        return []


def _search_amap_food_photo_urls_http(
    name: str,
    city: Optional[str] = None,
    limit: int = 10,
) -> list[str]:
    settings = get_settings()
    if not settings.amap_api_key or not name or limit <= 0:
        return []

    params: dict[str, Any] = {
        "key": settings.amap_api_key,
        "keywords": name,
        "types": "050000",
        "offset": 20,
        "page": 1,
        "extensions": "all",
    }
    if city:
        params["city"] = city
        params["citylimit"] = "true"

    try:
        query = urlencode(params)
        with urlopen(f"https://restapi.amap.com/v3/place/text?{query}", timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _merge_photo_urls(
            _extract_amap_food_photo_urls(data, target_name=name),
            limit=limit,
        )
    except Exception as exc:
        print(f"Amap food photo search failed: {exc}")
        return []


def _search_amap_hotel_photo_urls_http(
    name: str,
    city: Optional[str] = None,
    limit: int = 10,
) -> list[str]:
    settings = get_settings()
    if not settings.amap_api_key or not name or limit <= 0:
        return []

    params: dict[str, Any] = {
        "key": settings.amap_api_key,
        "keywords": name,
        "types": "100000",
        "offset": 20,
        "page": 1,
        "extensions": "all",
    }
    if city:
        params["city"] = city
        params["citylimit"] = "true"

    try:
        query = urlencode(params)
        with urlopen(f"https://restapi.amap.com/v3/place/text?{query}", timeout=6) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _merge_photo_urls(
            _extract_amap_hotel_photo_urls(data, target_name=name),
            limit=limit,
        )
    except Exception as exc:
        print(f"Amap hotel photo search failed: {exc}")
        return []


def _extract_amap_scenic_photo_urls(detail: Any, target_name: str = "") -> list[str]:
    pois = _extract_poi_items(detail)
    if not pois:
        return _filter_photo_urls(_extract_photo_urls(detail), target_name=target_name)

    urls: list[str] = []
    for poi_item in pois:
        if not _is_scenic_poi(poi_item):
            continue
        if not _is_relevant_poi_name(poi_item.get("name"), target_name):
            continue
        urls.extend(_extract_photo_urls(poi_item))

    return _merge_photo_urls(urls, limit=10)


def _extract_amap_food_photo_urls(detail: Any, target_name: str = "") -> list[str]:
    pois = _extract_poi_items(detail)
    if not pois:
        return _filter_food_photo_urls(_extract_food_photo_urls(detail))

    urls: list[str] = []
    for poi_item in pois:
        if not _is_food_poi(poi_item):
            continue
        if not _is_relevant_poi_name(poi_item.get("name"), target_name):
            continue
        urls.extend(_extract_food_photo_urls(poi_item))

    return _merge_photo_urls(urls, limit=10)


def _extract_amap_hotel_photo_urls(detail: Any, target_name: str = "") -> list[str]:
    pois = _extract_poi_items(detail)
    if not pois:
        return _filter_hotel_photo_urls(_extract_hotel_photo_urls(detail))

    urls: list[str] = []
    for poi_item in pois:
        if not _is_hotel_poi(poi_item):
            continue
        if not _is_relevant_poi_name(poi_item.get("name"), target_name):
            continue
        urls.extend(_extract_hotel_photo_urls(poi_item))

    return _merge_photo_urls(_filter_hotel_photo_urls(urls), limit=10)


def _extract_poi_items(detail: Any) -> list[dict[str, Any]]:
    if isinstance(detail, dict):
        raw_pois = detail.get("pois")
        if isinstance(raw_pois, list):
            return [item for item in raw_pois if isinstance(item, dict)]
        if any(key in detail for key in ("id", "name", "typecode", "photos")):
            return [detail]
        for key in ("data", "detail"):
            items = _extract_poi_items(detail.get(key))
            if items:
                return items
    if isinstance(detail, list):
        return [item for item in detail if isinstance(item, dict)]
    return []


def _is_scenic_poi(poi_item: dict[str, Any]) -> bool:
    type_text = " ".join(
        str(poi_item.get(key) or "")
        for key in ("type", "typecode", "biz_type")
    )

    excluded_tokens = (
        "100000",  # 住宿服务
        "050000",  # 餐饮服务
        "060000",  # 购物服务
        "070000",  # 生活服务
        "120000",  # 商务住宅
        "150000",  # 交通设施
        "住宿",
        "酒店",
        "宾馆",
        "民宿",
        "餐饮",
        "餐厅",
        "购物",
        "商场",
    )
    if any(token in type_text for token in excluded_tokens):
        return False

    scenic_tokens = (
        "110000",  # 风景名胜
        "110100",
        "110200",
        "110201",
        "110202",
        "110203",
        "110204",
        "110205",
        "110206",
        "110207",
        "110208",
        "110209",
        "140100",  # 博物馆
        "风景名胜",
        "景点",
        "公园",
        "广场",
        "博物馆",
        "纪念馆",
        "文物古迹",
        "寺",
        "观景",
    )
    return any(token in type_text for token in scenic_tokens)


def _is_food_poi(poi_item: dict[str, Any]) -> bool:
    type_text = " ".join(
        str(poi_item.get(key) or "")
        for key in ("type", "typecode", "biz_type")
    )
    return any(token in type_text for token in ("050000", "餐饮", "餐厅", "美食", "小吃"))


def _is_hotel_poi(poi_item: dict[str, Any]) -> bool:
    type_text = " ".join(
        str(poi_item.get(key) or "")
        for key in ("type", "typecode", "biz_type")
    )
    typecode = str(poi_item.get("typecode") or "")
    return (
        typecode.startswith("10")
        or typecode.startswith("120")
        or any(token in type_text for token in ("住宿", "酒店", "宾馆", "民宿", "客栈"))
    )


def _is_relevant_poi_name(poi_name: Any, target_name: str) -> bool:
    normalized_poi = _normalize_place_name(poi_name)
    normalized_target = _normalize_place_name(target_name)
    if not normalized_target:
        return True
    if not normalized_poi:
        return False
    return normalized_target in normalized_poi or normalized_poi in normalized_target


def _normalize_place_name(value: Any) -> str:
    text = str(value or "").strip().lower()
    removable_words = (
        "风景区",
        "景区",
        "公园",
        "博物馆",
        "纪念馆",
        "旅游区",
        "打卡点",
        "售票处",
        "游客中心",
        "停车场",
    )
    for word in removable_words:
        text = text.replace(word, "")
    return "".join(ch for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")


def _extract_photo_urls(detail: Any) -> list[str]:
    urls: list[str] = []

    def add_url(value: Any, title: Any = ""):
        if _is_usable_photo_url(value, title=title):
            urls.append(value)

    def walk(value: Any):
        if isinstance(value, dict):
            title = value.get("title") or value.get("name") or value.get("caption") or ""
            for key in ("url", "photo_url", "image_url", "src"):
                add_url(value.get(key), title=title)
            for key in ("photos", "images", "photo", "pois", "data", "detail"):
                if key in value:
                    walk(value[key])
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            add_url(value)

    walk(detail)
    return list(dict.fromkeys(urls))


def _extract_food_photo_urls(detail: Any) -> list[str]:
    urls: list[str] = []

    def add_url(value: Any, title: Any = ""):
        if _is_usable_food_photo_url(value, title=title):
            urls.append(value)

    def walk(value: Any):
        if isinstance(value, dict):
            title = value.get("title") or value.get("name") or value.get("caption") or ""
            for key in ("url", "photo_url", "image_url", "src"):
                add_url(value.get(key), title=title)
            for key in ("photos", "images", "photo", "pois", "data", "detail"):
                if key in value:
                    walk(value[key])
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            add_url(value)

    walk(detail)
    return list(dict.fromkeys(urls))


def _extract_hotel_photo_urls(detail: Any) -> list[str]:
    urls: list[str] = []

    def add_url(value: Any):
        if _is_usable_hotel_photo_url(value):
            urls.append(value)

    def walk(value: Any):
        if isinstance(value, dict):
            for key in ("url", "photo_url", "image_url", "src"):
                add_url(value.get(key))
            for key in ("photos", "images", "photo", "pois", "data", "detail"):
                if key in value:
                    walk(value[key])
        elif isinstance(value, list):
            for item in value:
                walk(item)
        else:
            add_url(value)

    walk(detail)
    return list(dict.fromkeys(urls))


def _filter_photo_urls(urls: list[str], target_name: str = "") -> list[str]:
    return [
        url
        for url in urls
        if _is_usable_photo_url(url, title=target_name)
    ]


def _filter_food_photo_urls(urls: list[str]) -> list[str]:
    return [
        url
        for url in urls
        if _is_usable_food_photo_url(url)
    ]


def _filter_hotel_photo_urls(urls: list[str]) -> list[str]:
    return [
        url
        for url in urls
        if _is_usable_hotel_photo_url(url)
    ]


def _is_usable_hotel_photo_url(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return False

    lowered_url = value.lower()
    return not any(token in lowered_url for token in ("thumbnail", "thumb", "_80.", "small"))


def _is_usable_food_photo_url(value: Any, title: Any = "") -> bool:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return False

    lowered_url = value.lower()
    if any(token in lowered_url for token in ("thumbnail", "thumb", "_80.", "small")):
        return False

    title_text = str(title or "")
    bad_title_tokens = (
        "酒店",
        "宾馆",
        "民宿",
        "客房",
        "房间",
        "大堂",
        "前台",
        "停车场",
        "卫生间",
    )
    return not any(token in title_text for token in bad_title_tokens)


def _is_usable_photo_url(value: Any, title: Any = "") -> bool:
    if not isinstance(value, str) or not value.startswith(("http://", "https://")):
        return False

    lowered_url = value.lower()
    if any(token in lowered_url for token in ("thumbnail", "thumb", "_80.", "small")):
        return False

    title_text = str(title or "")
    bad_title_tokens = (
        "酒店",
        "宾馆",
        "民宿",
        "客房",
        "房间",
        "大堂",
        "前台",
        "餐厅",
        "菜单",
        "菜品",
        "停车场",
        "卫生间",
    )
    return not any(token in title_text for token in bad_title_tokens)


def _merge_photo_urls(*groups: list[str], limit: int = 5) -> list[str]:
    merged: list[str] = []

    for group in groups:
        for url in group:
            if not isinstance(url, str) or not url:
                continue
            if url in merged:
                continue
            merged.append(url)
            if len(merged) >= limit:
                return merged

    return merged


def _placeholder_photo_url(name: str, category: Optional[str] = None) -> str:
    escaped_name = escape(name or "旅行景点")
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' width='800' height='600'>"
        "<defs>"
        "<linearGradient id='g' x1='0%' y1='0%' x2='100%' y2='100%'>"
        f"<stop offset='0%' stop-color='{'#f59e0b' if category == 'meal' else '#667eea'}'/>"
        f"<stop offset='100%' stop-color='{'#10b981' if category == 'meal' else '#764ba2'}'/>"
        "</linearGradient>"
        "</defs>"
        "<rect width='800' height='600' fill='url(#g)'/>"
        "<text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle' "
        "font-family='Arial, sans-serif' font-size='44' font-weight='700' fill='white'>"
        f"{escaped_name}"
        "</text>"
        "</svg>"
    )
    return f"data:image/svg+xml;utf8,{quote(svg)}"
