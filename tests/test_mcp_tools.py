import pytest
from pathlib import Path

from backend.app.services import mcp_tools


class FakeTool:
    def __init__(self, name):
        self.name = name


class FakeClient:
    def __init__(self, tools):
        self._tools = tools

    async def get_tools(self, server_name=None):
        assert server_name == "amap"
        return self._tools


def test_create_amap_mcp_client_uses_stdio_config(monkeypatch):
    captured = {}

    class CapturingClient:
        def __init__(self, connections):
            captured.update(connections)

    monkeypatch.setenv("AMAP_API_KEY", "test-amap-key")
    monkeypatch.setattr(mcp_tools, "MultiServerMCPClient", CapturingClient)

    mcp_tools.create_amap_mcp_client()

    assert "amap" in captured
    assert captured["amap"]["transport"] == "stdio"
    assert Path(captured["amap"]["command"]).name == "uvx.exe"
    assert captured["amap"]["args"] == ["amap-mcp-server"]
    assert captured["amap"]["env"]["AMAP_MAPS_API_KEY"] == "test-amap-key"


@pytest.mark.asyncio
async def test_find_amap_tool_returns_matching_tool(monkeypatch):
    mcp_tools.reset_mcp_tools()

    fake_tools = [
        FakeTool("maps_text_search"),
        FakeTool("maps_weather"),
    ]

    monkeypatch.setattr(
        mcp_tools,
        "get_amap_mcp_client",
        lambda: FakeClient(fake_tools),
    )

    tool = await mcp_tools.find_amap_tool("maps_weather")

    assert tool.name == "maps_weather"


@pytest.mark.asyncio
async def test_find_amap_tool_raises_for_missing_tool(monkeypatch):
    mcp_tools.reset_mcp_tools()

    fake_tools = [
        FakeTool("maps_text_search"),
    ]

    monkeypatch.setattr(
        mcp_tools,
        "get_amap_mcp_client",
        lambda: FakeClient(fake_tools),
    )

    with pytest.raises(ValueError, match="Amap MCP tool not found"):
        await mcp_tools.find_amap_tool("maps_weather")
