import asyncio
import json

from app.services.mcp_tools import find_amap_tool

#测试mcp工具搜索功能
async def main():
    tool = await find_amap_tool("maps_text_search")

    result = await tool.ainvoke(
        {
            "keywords": "历史文化",
            "city": "北京",
            "citylimit": "true",
        }
    )

    print(type(result))
    print(result)

    if isinstance(result, str):
        try:
            data = json.loads(result)
            print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
        except json.JSONDecodeError:
            print("Result is not direct JSON string")


if __name__ == "__main__":
    asyncio.run(main())