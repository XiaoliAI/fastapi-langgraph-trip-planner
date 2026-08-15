"""LangChain MCP tool loading utilities."""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from ..config import get_settings


_amap_mcp_client = None
_amap_tools = None


def _backend_env_path() -> Path:
    return Path(__file__).resolve().parents[2] / ".env"


def _read_env_value(key: str) -> Optional[str]:
    value = os.getenv(key)
    if value:
        return value.strip().strip('"').strip("'")

    env_path = _backend_env_path()
    if not env_path.exists():
        return None

    file_values = dotenv_values(env_path)
    file_value = file_values.get(key)
    if file_value:
        return str(file_value).strip().strip('"').strip("'")

    return None


def get_amap_api_key() -> str:
    settings = get_settings()
    api_key = _read_env_value("AMAP_API_KEY") or settings.amap_api_key

    if not api_key:
        raise ValueError("AMAP_API_KEY is not configured")

    return api_key


def _resolve_uvx_command() -> str:
    candidate_paths = [
        Path(sys.executable).parent / "uvx.exe",
        Path(sys.executable).parent / "Scripts" / "uvx.exe",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    uvx_in_path = shutil.which("uvx")
    if uvx_in_path:
        return uvx_in_path

    return "uvx"


def create_amap_mcp_client() -> MultiServerMCPClient:
    api_key = get_amap_api_key()
    uvx_command = _resolve_uvx_command()

    return MultiServerMCPClient(
        {
            "amap": {
                "transport": "stdio",
                "command": uvx_command,
                "args": ["amap-mcp-server"],
                "env": {
                    "AMAP_MAPS_API_KEY": api_key,
                },
            }
        }
    )


def get_amap_mcp_client() -> MultiServerMCPClient:
    global _amap_mcp_client

    if _amap_mcp_client is None:
        _amap_mcp_client = create_amap_mcp_client()

    return _amap_mcp_client


async def get_amap_langchain_tools() -> list[BaseTool]:
    global _amap_tools

    if _amap_tools is None:
        client = get_amap_mcp_client()
        _amap_tools = await client.get_tools(server_name="amap")

    return _amap_tools


async def find_amap_tool(tool_name: str) -> BaseTool:
    tools = await get_amap_langchain_tools()

    for tool in tools:
        if tool.name == tool_name:
            return tool

    available_names = ", ".join(tool.name for tool in tools)
    raise ValueError(f"Amap MCP tool not found: {tool_name}. Available tools: {available_names}")


def reset_mcp_tools():
    global _amap_mcp_client, _amap_tools

    _amap_mcp_client = None
    _amap_tools = None
