import asyncio

from app.services.amap_langchain_service import search_pois, search_pois_text


async def main():
    pois = await search_pois(
        keywords="历史文化",
        city="北京",
        citylimit=True,
        limit=5,
    )

    print(f"Structured POIs: {len(pois)}")
    for poi in pois:
        print(f"- {poi.name} | {poi.address} | {poi.location.longitude},{poi.location.latitude}")

    print("\nPlanner text:")
    text = await search_pois_text(
        keywords="历史文化",
        city="北京",
        citylimit=True,
        limit=5,
    )
    print(text)


if __name__ == "__main__":
    asyncio.run(main())