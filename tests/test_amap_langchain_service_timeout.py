import asyncio

import pytest

from backend.app.services import amap_langchain_service


class SlowTool:
    async def ainvoke(self, args):
        await asyncio.sleep(0.05)
        return {"pois": [{"id": "B0001", "name": "天坛", "address": "天坛东路", "typecode": "110201"}]}


@pytest.mark.asyncio
async def test_search_pois_returns_empty_list_when_mcp_times_out(monkeypatch):
    async def fake_find_amap_tool(tool_name: str):
        assert tool_name == "maps_text_search"
        return SlowTool()

    monkeypatch.setattr(amap_langchain_service, "find_amap_tool", fake_find_amap_tool)
    monkeypatch.setattr(amap_langchain_service, "MCP_CALL_TIMEOUT_SECONDS", 0.001)

    pois = await amap_langchain_service.search_pois(
        keywords="天坛",
        city="北京",
        citylimit=True,
        limit=3,
    )

    assert pois == []
