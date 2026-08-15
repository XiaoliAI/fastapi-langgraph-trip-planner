import asyncio
import traceback

from app.services.mcp_tools import find_amap_tool
from app.services.amap_langchain_service import _extract_json_like


async def main():
    tool = await find_amap_tool("maps_weather")

    print("TOOL NAME:")
    print(tool.name)

    print("\nTOOL DESCRIPTION:")
    print(tool.description)

    print("\nTOOL ARGS SCHEMA:")
    print(tool.args_schema)

    print("\nCALLING maps_weather...")

    try:
        result = await tool.ainvoke(
            {
                "city": "北京",
            }
        )

        print("\nRAW TYPE:")
        print(type(result))

        print("\nRAW RESULT:")
        print(result)

        data = _extract_json_like(result)

        print("\nEXTRACTED DATA:")
        print(data)

    except BaseException as exc:
        print("\nEXCEPTION:")
        traceback.print_exception(type(exc), exc, exc.__traceback__)


if __name__ == "__main__":
    asyncio.run(main())