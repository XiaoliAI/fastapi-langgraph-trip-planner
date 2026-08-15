#测试mcp是否能返回工具

import asyncio

from app.services.mcp_tools import get_amap_langchain_tools


async def main():
    tools = await get_amap_langchain_tools()

    print(f"Loaded tools: {len(tools)}")
    for tool in tools:
        print(f"- {tool.name}: {tool.description[:120] if tool.description else ''}")


if __name__ == "__main__":
    asyncio.run(main())