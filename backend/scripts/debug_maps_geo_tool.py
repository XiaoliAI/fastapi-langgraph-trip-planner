import asyncio
import traceback

from app.services.mcp_tools import find_amap_tool
from app.services.amap_langchain_service import _extract_json_like, _parse_location


async def try_geo(tool, args):
    print("\nCALL ARGS:")
    print(args)

    try:
        result = await tool.ainvoke(args)

        print("RAW TYPE:")
        print(type(result))

        print("RAW RESULT:")
        print(result)

        data = _extract_json_like(result)

        print("EXTRACTED DATA:")
        print(data)

        if isinstance(data, dict):
            geocodes = data.get("geocodes")
            print("GEOCODES:")
            print(geocodes)

            if isinstance(geocodes, list) and geocodes:
                first = geocodes[0]
                print("FIRST GEOCODE:")
                print(first)

                if isinstance(first, dict):
                    location = _parse_location(first.get("location"))
                    print("PARSED LOCATION:")
                    print(location)

    except BaseException as exc:
        print("EXCEPTION:")
        traceback.print_exception(type(exc), exc, exc.__traceback__)


async def main():
    tool = await find_amap_tool("maps_geo")

    print("TOOL NAME:")
    print(tool.name)

    print("\nTOOL DESCRIPTION:")
    print(tool.description)

    print("\nTOOL ARGS SCHEMA:")
    print(tool.args_schema)

    await try_geo(
        tool,
        {
            "address": "景山前街4号",
            "city": "北京",
        },
    )

    await try_geo(
        tool,
        {
            "address": "北京市景山前街4号",
        },
    )

    await try_geo(
        tool,
        {
            "address": "故宫博物院",
            "city": "北京",
        },
    )

    await try_geo(
        tool,
        {
            "address": "北京市故宫博物院",
        },
    )


if __name__ == "__main__":
    asyncio.run(main())