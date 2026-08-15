import asyncio
import json
import traceback

from app.services.mcp_tools import find_amap_tool
from app.services.amap_langchain_service import (
    _extract_pois,
    _parse_poi_item,
    _parse_poi_item_with_geocode,
    geocode_location,
)


async def main():
    try:
        tool = await find_amap_tool("maps_text_search")
        result = await tool.ainvoke(
            {
                "keywords": "历史文化",
                "city": "北京",
                "citylimit": "true",
            }
        )

        print("RAW RESULT TYPE:")
        print(type(result))

        print("\nRAW RESULT:")
        print(result)

        raw_pois = _extract_pois(result)
        print(f"\nRAW POI COUNT: {len(raw_pois)}")

        if raw_pois:
            first = raw_pois[0]
            print("\nFIRST RAW POI:")
            print(json.dumps(first, ensure_ascii=False, indent=2))

            parsed = _parse_poi_item(first)
            print("\nPARSED DIRECT POI:")
            print(parsed)

            geocoded = await _parse_poi_item_with_geocode(first, "北京")
            print("\nPARSED WITH GEOCODE:")
            print(geocoded)

            address = first.get("address") or first.get("name") or ""
            location = await geocode_location(address, "北京")
            print("\nGEOCODE RESULT:")
            print(location)

    except BaseException as exc:
        print("\nTop-level exception:")
        print(type(exc).__name__, exc)
        print("\nFull traceback:")
        traceback.print_exception(type(exc), exc, exc.__traceback__)


if __name__ == "__main__":
    asyncio.run(main())