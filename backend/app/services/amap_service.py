"""高德地图MCP服务封装"""

import json
import re
from urllib.parse import urlencode
from urllib.request import urlopen
from typing import List, Dict, Any, Optional
from hello_agents.tools import MCPTool
from ..config import get_settings
from ..models.schemas import Location, POIInfo, WeatherInfo

# 全局MCP工具实例
_amap_mcp_tool = None


def get_amap_mcp_tool() -> MCPTool:
    """
    获取高德地图MCP工具实例(单例模式)
    
    Returns:
        MCPTool实例
    """
    global _amap_mcp_tool
    
    if _amap_mcp_tool is None:
        settings = get_settings()
        
        if not settings.amap_api_key:
            raise ValueError("高德地图API Key未配置,请在.env文件中设置AMAP_API_KEY")
        
        # 创建MCP工具
        _amap_mcp_tool = MCPTool(
            name="amap",
            description="高德地图服务,支持POI搜索、路线规划、天气查询等功能",
            server_command=["uvx", "amap-mcp-server"],
            env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
            auto_expand=True  # 自动展开为独立工具
        )
        
        print(f"✅ 高德地图MCP工具初始化成功")
        print(f"   工具数量: {len(_amap_mcp_tool._available_tools)}")
        
        # 打印可用工具列表
        if _amap_mcp_tool._available_tools:
            print("   可用工具:")
            for tool in _amap_mcp_tool._available_tools[:5]:  # 只打印前5个
                print(f"     - {tool.get('name', 'unknown')}")
            if len(_amap_mcp_tool._available_tools) > 5:
                print(f"     ... 还有 {len(_amap_mcp_tool._available_tools) - 5} 个工具")
    
    return _amap_mcp_tool


class AmapService:
    """高德地图服务封装类"""
    
    def __init__(self):
        """初始化服务"""
        self.mcp_tool = get_amap_mcp_tool()
    
    def search_poi(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        """
        搜索POI
        
        Args:
            keywords: 搜索关键词
            city: 城市
            citylimit: 是否限制在城市范围内
            
        Returns:
            POI信息列表
        """
        try:
            # 调用MCP工具
            result = self.mcp_tool.run({
                "action": "call_tool",
                "tool_name": "maps_text_search",
                "arguments": {
                    "keywords": keywords,
                    "city": city,
                    "citylimit": str(citylimit).lower()
                }
            })
            
            print(f"POI搜索结果: {result[:200]}...")  # 打印前200字符

            pois = self._parse_poi_result(result)
            if pois:
                return pois

            print("MCP POI结果解析为空, 尝试高德HTTP接口兜底...")
            return self._search_poi_http(keywords, city, citylimit)
            
        except Exception as e:
            print(f"❌ POI搜索失败: {str(e)}")
            return []

    def _parse_poi_result(self, result: str) -> List[POIInfo]:
        """从高德 MCP 文本响应中解析 POI 列表。"""
        data = self._extract_json(result)
        if not data:
            return []

        raw_pois = data.get("pois") if isinstance(data, dict) else data
        if not isinstance(raw_pois, list):
            return []

        pois: List[POIInfo] = []
        for item in raw_pois:
            poi = self._parse_poi_item(item)
            if poi:
                pois.append(poi)
        return pois

    def _parse_poi_item(self, item: Dict[str, Any]) -> Optional[POIInfo]:
        if not isinstance(item, dict):
            return None

        location = self._parse_location(item.get("location"))
        if location is None:
            return None

        return POIInfo(
            id=str(item.get("id") or item.get("poi_id") or ""),
            name=str(item.get("name") or ""),
            type=str(item.get("type") or ""),
            address=self._string_value(item.get("address")),
            location=location,
            tel=self._optional_string_value(item.get("tel")),
        )

    def _extract_json(self, result: str) -> Optional[Any]:
        if not result:
            return None

        candidates = [
            match.group(0)
            for match in re.finditer(r"(\{.*\}|\[.*\])", result, re.DOTALL)
        ]
        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def _parse_location(self, value: Any) -> Optional[Location]:
        if isinstance(value, dict):
            longitude = value.get("longitude") or value.get("lng")
            latitude = value.get("latitude") or value.get("lat")
        elif isinstance(value, str) and "," in value:
            longitude, latitude = value.split(",", 1)
        else:
            return None

        try:
            return Location(longitude=float(longitude), latitude=float(latitude))
        except (TypeError, ValueError):
            return None

    def _string_value(self, value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(item) for item in value if item)
        if value is None:
            return ""
        return str(value)

    def _optional_string_value(self, value: Any) -> Optional[str]:
        text = self._string_value(value)
        return text or None

    def _search_poi_http(self, keywords: str, city: str, citylimit: bool = True) -> List[POIInfo]:
        settings = get_settings()
        params = {
            "key": settings.amap_api_key,
            "keywords": keywords,
            "offset": 20,
            "page": 1,
            "extensions": "base",
        }
        if city:
            params["city"] = city
            params["citylimit"] = "true" if citylimit else "false"

        data = self._request_amap_json("https://restapi.amap.com/v3/place/text", params)
        pois = self._parse_poi_data(data)
        print(f"高德HTTP POI兜底结果数量: {len(pois)}")
        return pois

    def _request_amap_json(self, url: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            query = urlencode(params)
            with urlopen(f"{url}?{query}", timeout=10) as response:
                raw = response.read().decode("utf-8")
            return json.loads(raw)
        except Exception as e:
            print(f"高德HTTP请求失败: {str(e)}")
            return None

    def _parse_poi_data(self, data: Any) -> List[POIInfo]:
        if not isinstance(data, dict):
            return []

        raw_pois = data.get("pois")
        if not isinstance(raw_pois, list):
            return []

        pois: List[POIInfo] = []
        for item in raw_pois:
            poi = self._parse_poi_item(item)
            if poi:
                pois.append(poi)
        return pois
    
    def get_weather(self, city: str) -> List[WeatherInfo]:
        """
        查询天气
        
        Args:
            city: 城市名称
            
        Returns:
            天气信息列表
        """
        try:
            # 调用MCP工具
            result = self.mcp_tool.run({
                "action": "call_tool",
                "tool_name": "maps_weather",
                "arguments": {
                    "city": city
                }
            })
            
            print(f"天气查询结果: {result[:200]}...")
            
            # TODO: 解析实际的天气数据
            return []
            
        except Exception as e:
            print(f"❌ 天气查询失败: {str(e)}")
            return []
    
    def plan_route(
        self,
        origin_address: str,
        destination_address: str,
        origin_city: Optional[str] = None,
        destination_city: Optional[str] = None,
        route_type: str = "walking"
    ) -> Dict[str, Any]:
        """
        规划路线
        
        Args:
            origin_address: 起点地址
            destination_address: 终点地址
            origin_city: 起点城市
            destination_city: 终点城市
            route_type: 路线类型 (walking/driving/transit)
            
        Returns:
            路线信息
        """
        try:
            # 根据路线类型选择工具
            tool_map = {
                "walking": "maps_direction_walking_by_address",
                "driving": "maps_direction_driving_by_address",
                "transit": "maps_direction_transit_integrated_by_address"
            }
            
            tool_name = tool_map.get(route_type, "maps_direction_walking_by_address")
            
            # 构建参数
            arguments = {
                "origin_address": origin_address,
                "destination_address": destination_address
            }
            
            # 公共交通需要城市参数
            if route_type == "transit":
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city
            else:
                # 其他路线类型也可以提供城市参数提高准确性
                if origin_city:
                    arguments["origin_city"] = origin_city
                if destination_city:
                    arguments["destination_city"] = destination_city
            
            # 调用MCP工具
            result = self.mcp_tool.run({
                "action": "call_tool",
                "tool_name": tool_name,
                "arguments": arguments
            })
            
            print(f"路线规划结果: {result[:200]}...")
            
            # TODO: 解析实际的路线数据
            return {}
            
        except Exception as e:
            print(f"❌ 路线规划失败: {str(e)}")
            return {}
    
    def geocode(self, address: str, city: Optional[str] = None) -> Optional[Location]:
        """
        地理编码(地址转坐标)

        Args:
            address: 地址
            city: 城市

        Returns:
            经纬度坐标
        """
        try:
            arguments = {"address": address}
            if city:
                arguments["city"] = city

            result = self.mcp_tool.run({
                "action": "call_tool",
                "tool_name": "maps_geo",
                "arguments": arguments
            })

            print(f"地理编码结果: {result[:200]}...")

            data = self._extract_json(result)
            if not isinstance(data, dict):
                return None

            geocodes = data.get("geocodes")
            if not isinstance(geocodes, list) or not geocodes:
                return None

            first = geocodes[0]
            if not isinstance(first, dict):
                return None

            return self._parse_location(first.get("location"))

        except Exception as e:
            print(f"❌ 地理编码失败: {str(e)}")
            return None

    def get_poi_detail(self, poi_id: str) -> Dict[str, Any]:
        """
        获取POI详情

        Args:
            poi_id: POI ID

        Returns:
            POI详情信息
        """
        try:
            result = self.mcp_tool.run({
                "action": "call_tool",
                "tool_name": "maps_search_detail",
                "arguments": {
                    "id": poi_id
                }
            })

            print(f"POI详情结果: {result[:200]}...")

            # 解析结果并提取图片
            import json
            import re

            # 尝试从结果中提取JSON
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return data

            return {"raw": result}

        except Exception as e:
            print(f"❌ 获取POI详情失败: {str(e)}")
            return {}


# 创建全局服务实例
    def extract_photo_urls(self, detail: Any) -> List[str]:
        """Extract photo URLs from common Amap POI detail response shapes."""
        urls: List[str] = []

        def add_url(value: Any):
            if isinstance(value, str) and value.startswith(("http://", "https://")):
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


_amap_service = None


def get_amap_service() -> AmapService:
    """获取高德地图服务实例(单例模式)"""
    global _amap_service
    
    if _amap_service is None:
        _amap_service = AmapService()
    
    return _amap_service
