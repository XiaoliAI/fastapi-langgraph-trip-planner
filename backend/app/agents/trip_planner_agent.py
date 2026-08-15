"""多智能体旅行规划系统"""

import json
from typing import Dict, Any, List, Optional
from hello_agents import SimpleAgent
from hello_agents.tools import MCPTool
from ..services.llm_service import get_llm
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel, POIInfo
from ..config import get_settings
from ..services.amap_service import get_amap_service

# ============ Agent提示词 ============

ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。你的任务是根据城市和用户偏好搜索合适的景点。

**重要提示:**
你必须使用工具来搜索景点!不要自己编造景点信息!

**工具调用格式:**
使用maps_text_search工具时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_text_search:keywords=景点关键词,city=城市名]`

**示例:**
用户: "搜索北京的历史文化景点"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=历史文化,city=北京]

用户: "搜索上海的公园"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=公园,city=上海]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
3. 参数用逗号分隔
"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。你的任务是查询指定城市的天气信息。

**重要提示:**
你必须使用工具来查询天气!不要自己编造天气信息!

**工具调用格式:**
使用maps_weather工具时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_weather:city=城市名]`

**示例:**
用户: "查询北京天气"
你的回复: [TOOL_CALL:amap_maps_weather:city=北京]

用户: "上海的天气怎么样"
你的回复: [TOOL_CALL:amap_maps_weather:city=上海]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
"""

HOTEL_AGENT_PROMPT = """你是酒店推荐专家。你的任务是根据城市和景点位置推荐合适的酒店。

**重要提示:**
你必须使用工具来搜索酒店!不要自己编造酒店信息!

**工具调用格式:**
使用maps_text_search工具搜索酒店时,必须严格按照以下格式:
`[TOOL_CALL:amap_maps_text_search:keywords=酒店,city=城市名]`

**示例:**
用户: "搜索北京的酒店"
你的回复: [TOOL_CALL:amap_maps_text_search:keywords=酒店,city=北京]

**注意:**
1. 必须使用工具,不要直接回答
2. 格式必须完全正确,包括方括号和冒号
3. 关键词使用"酒店"或"宾馆"
"""

PLANNER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息,生成详细的旅行计划。

请严格按照以下JSON格式返回旅行计划:
```json
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397128, "latitude": 39.916527},
        "price_range": "300-500元",
        "rating": "4.5",
        "distance": "距离景点2公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "详细地址",
          "location": {"longitude": 116.397128, "latitude": 39.916527},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50},
        {"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "总体建议",
  "budget": {
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }
}
```

**重要提示:**
1. weather_info数组必须包含每一天的天气信息
2. 温度必须是纯数字(不要带°C等单位)
3. 每天安排2-3个景点
4. 考虑景点之间的距离和游览时间
5. 每天必须包含早中晚三餐
6. 提供实用的旅行建议
7. **必须包含预算信息**:
   - 景点门票价格(ticket_price)
   - 餐饮预估费用(estimated_cost)
   - 酒店预估费用(estimated_cost)
   - 预算汇总(budget)包含各项总费用
"""


CITY_CENTER_COORDINATES = {
    "北京": (116.397128, 39.916527),
    "上海": (121.473701, 31.230416),
    "广州": (113.264385, 23.129112),
    "深圳": (114.057868, 22.543099),
    "杭州": (120.1551, 30.2741),
    "成都": (104.066541, 30.572269),
    "西安": (108.940174, 34.341568),
    "南京": (118.796877, 32.060255),
    "重庆": (106.551556, 29.563009),
    "武汉": (114.305392, 30.593098),
    "苏州": (120.585315, 31.298886),
    "厦门": (118.089425, 24.479834),
    "青岛": (120.382665, 36.066938),
    "天津": (117.200983, 39.084158),
    "长沙": (112.938814, 28.228209),
    "昆明": (102.832891, 24.880095),
    "大连": (121.614682, 38.914003),
    "三亚": (109.511909, 18.252847),
    "拉萨": (91.140856, 29.645554),
    "乌鲁木齐": (87.616848, 43.825592),
    "哈尔滨": (126.534967, 45.803775),
}


class MultiAgentTripPlanner:
    """多智能体旅行规划系统"""

    def __init__(self):
        """初始化多智能体系统"""
        print("🔄 开始初始化多智能体旅行规划系统...")

        try:
            settings = get_settings()
            self.llm = get_llm()

            # 创建共享的MCP工具(只创建一次)
            print("  - 创建共享MCP工具...")
            self.amap_tool = MCPTool(
                name="amap",
                description="高德地图服务",
                server_command=["uvx", "amap-mcp-server"],
                env={"AMAP_MAPS_API_KEY": settings.amap_api_key},
                auto_expand=True
            )
            self.amap_tool.expandable=True

            # 创建景点搜索Agent
            print("  - 创建景点搜索Agent...")
            self.attraction_agent = SimpleAgent(
                name="景点搜索专家",
                llm=self.llm,
                system_prompt=ATTRACTION_AGENT_PROMPT
            )
            self.attraction_agent.add_tool(self.amap_tool)

            # 创建天气查询Agent
            print("  - 创建天气查询Agent...")
            self.weather_agent = SimpleAgent(
                name="天气查询专家",
                llm=self.llm,
                system_prompt=WEATHER_AGENT_PROMPT
            )
            self.weather_agent.add_tool(self.amap_tool)

            # 创建酒店推荐Agent
            print("  - 创建酒店推荐Agent...")
            self.hotel_agent = SimpleAgent(
                name="酒店推荐专家",
                llm=self.llm,
                system_prompt=HOTEL_AGENT_PROMPT
            )
            self.hotel_agent.add_tool(self.amap_tool)

            # 创建行程规划Agent(不需要工具)
            print("  - 创建行程规划Agent...")
            self.planner_agent = SimpleAgent(
                name="行程规划专家",
                llm=self.llm,
                system_prompt=PLANNER_AGENT_PROMPT
            )

            print(f"✅ 多智能体系统初始化成功")
            print(f"   景点搜索Agent: {len(self.attraction_agent.list_tools())} 个工具")
            print(f"   天气查询Agent: {len(self.weather_agent.list_tools())} 个工具")
            print(f"   酒店推荐Agent: {len(self.hotel_agent.list_tools())} 个工具")

        except Exception as e:
            print(f"❌ 多智能体系统初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def plan_trip(self, request: TripRequest) -> TripPlan:
        """
        使用多智能体协作生成旅行计划

        Args:
            request: 旅行请求

        Returns:
            旅行计划
        """
        try:
            poi_results: List[POIInfo] = []
            print(f"\n{'='*60}")
            print(f"🚀 开始多智能体协作规划旅行...")
            print(f"目的地: {request.city}")
            print(f"日期: {request.start_date} 至 {request.end_date}")
            print(f"天数: {request.travel_days}天")
            print(f"偏好: {', '.join(request.preferences) if request.preferences else '无'}")
            print(f"{'='*60}\n")

            # 步骤1: 景点搜索Agent搜索景点
            print("📍 步骤1: 搜索景点...")
            attraction_query = self._build_attraction_query(request)
            poi_results = self._search_real_pois(request)
            if poi_results:
                attraction_response = self._format_poi_results(poi_results)
            else:
                attraction_response = self.attraction_agent.run(attraction_query)
            print(f"景点搜索结果: {attraction_response[:200]}...\n")

            # 步骤2: 天气查询Agent查询天气
            print("🌤️  步骤2: 查询天气...")
            weather_query = f"请查询{request.city}的天气信息"
            weather_response = self.weather_agent.run(weather_query)
            print(f"天气查询结果: {weather_response[:200]}...\n")

            # 步骤3: 酒店推荐Agent搜索酒店
            print("🏨 步骤3: 搜索酒店...")
            hotel_query = f"请搜索{request.city}的{request.accommodation}酒店"
            hotel_response = self.hotel_agent.run(hotel_query)
            print(f"酒店搜索结果: {hotel_response[:200]}...\n")

            # 步骤4: 行程规划Agent整合信息生成计划
            print("📋 步骤4: 生成行程计划...")
            planner_query = self._build_planner_query(request, attraction_response, weather_response, hotel_response)
            planner_response = self.planner_agent.run(planner_query)
            print(f"行程规划结果: {planner_response[:300]}...\n")

            # 解析最终计划
            trip_plan = self._parse_response(planner_response, request, poi_results)
            trip_plan = self._apply_poi_locations(trip_plan, poi_results, request)

            print(f"{'='*60}")
            print(f"✅ 旅行计划生成完成!")
            print(f"{'='*60}\n")

            return trip_plan

        except Exception as e:
            print(f"❌ 生成旅行计划失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request, locals().get("poi_results", []))
    
    def _build_attraction_query(self, request: TripRequest) -> str:
        """构建景点搜索查询 - 直接包含工具调用"""
        keywords = []
        if request.preferences:
            # 只取第一个偏好作为关键词
            keywords = request.preferences[0]
        else:
            keywords = "景点"

        # 直接返回工具调用格式
        query = f"请使用amap_maps_text_search工具搜索{request.city}的{keywords}相关景点。\n[TOOL_CALL:amap_maps_text_search:keywords={keywords},city={request.city}]"
        return query

    def _search_real_pois(self, request: TripRequest) -> List[POIInfo]:
        """优先通过服务层直接查询真实 POI,避免 LLM 示例坐标进入最终结果。"""
        primary_keywords = request.preferences[0] if request.preferences else "景点"
        attempts = [
            (primary_keywords, request.city, True),
            (f"{request.city} 景点", "", False),
            (request.city, "", False),
        ]
        if primary_keywords != "景点":
            attempts.append(("景点", request.city, True))

        try:
            service = get_amap_service()
            for keywords, city, citylimit in attempts:
                pois = service.search_poi(keywords, city, citylimit)
                if pois:
                    return pois
            return []
        except Exception as e:
            print(f"[WARN] 直接查询POI失败,将回退到Agent工具调用: {str(e)}")
            return []

    def _format_poi_results(self, pois: List[POIInfo]) -> str:
        """将真实 POI 转成规划 Agent 容易稳定使用的文本。"""
        lines = []
        for index, poi in enumerate(pois[:12], start=1):
            lines.append(
                f"{index}. {poi.name} | 地址: {poi.address} | 类型: {poi.type} | "
                f"坐标: {poi.location.longitude},{poi.location.latitude} | POI ID: {poi.id}"
            )
        return "\n".join(lines)

    def _apply_poi_locations(self, trip_plan: TripPlan, pois: List[POIInfo], request: TripRequest) -> TripPlan:
        """用真实 POI 修正最终计划中的景点名称、地址和坐标。"""
        if not pois:
            return trip_plan

        used_poi_ids = set()
        poi_index = 0
        for day in trip_plan.days:
            for attraction in day.attractions:
                poi = self._find_matching_poi(attraction.name, pois, used_poi_ids)
                while poi is None and poi_index < len(pois):
                    candidate = pois[poi_index]
                    poi_index += 1
                    if self._poi_key(candidate) not in used_poi_ids:
                        poi = candidate

                if poi is None:
                    continue

                used_poi_ids.add(self._poi_key(poi))
                attraction.name = poi.name
                attraction.address = poi.address or f"{request.city}市"
                attraction.location = poi.location
                attraction.category = poi.type or attraction.category
                attraction.poi_id = poi.id

        trip_plan.city = request.city
        return self._remove_duplicate_attractions(trip_plan)

    def _find_matching_poi(
        self,
        attraction_name: str,
        pois: List[POIInfo],
        used_poi_ids: Optional[set] = None
    ) -> Optional[POIInfo]:
        used_poi_ids = used_poi_ids or set()
        for poi in pois:
            if self._poi_key(poi) in used_poi_ids:
                continue
            if attraction_name and (attraction_name in poi.name or poi.name in attraction_name):
                return poi
        return None

    def _poi_key(self, poi: POIInfo) -> str:
        return poi.id or poi.name

    def _remove_duplicate_attractions(self, trip_plan: TripPlan) -> TripPlan:
        seen_ids = set()
        seen_names = set()
        for day in trip_plan.days:
            unique_attractions = []
            for attraction in day.attractions:
                poi_id = attraction.poi_id or ""
                name = attraction.name or ""
                if (poi_id and poi_id in seen_ids) or (name and name in seen_names):
                    continue
                if poi_id:
                    seen_ids.add(poi_id)
                if name:
                    seen_names.add(name)
                unique_attractions.append(attraction)
            day.attractions = unique_attractions
        return trip_plan

    def _fallback_location(self, city: str, day_index: int, attraction_index: int) -> Location:
        coordinates = CITY_CENTER_COORDINATES.get(city)
        if coordinates is None:
            try:
                location = get_amap_service().geocode(city)
                if location:
                    coordinates = (location.longitude, location.latitude)
            except Exception as e:
                print(f"[WARN] 地理编码{city}失败,使用默认中心点: {str(e)}")

        longitude, latitude = coordinates or (104.195397, 35.86166)
        return Location(
            longitude=longitude + day_index * 0.01 + attraction_index * 0.005,
            latitude=latitude + day_index * 0.01 + attraction_index * 0.005,
        )

    def _build_poi_description(self, poi: POIInfo, city: str) -> str:
        category = poi.type or "景点"
        address = poi.address or f"{city}市"
        return f"{poi.name}位于{address}, 是{city}值得安排的{category}, 适合结合周边路线安排游览。"

    def _build_planner_query(self, request: TripRequest, attractions: str, weather: str, hotels: str = "") -> str:
        """构建行程规划查询"""
        query = f"""请根据以下信息生成{request.city}的{request.travel_days}天旅行计划:

**基本信息:**
- 城市: {request.city}
- 日期: {request.start_date} 至 {request.end_date}
- 天数: {request.travel_days}天
- 交通方式: {request.transportation}
- 住宿: {request.accommodation}
- 偏好: {', '.join(request.preferences) if request.preferences else '无'}

**景点信息:**
{attractions}

**天气信息:**
{weather}

**酒店信息:**
{hotels}

**去重约束:**
全程景点不得重复。已经安排过的景点, 后续日期不能再次安排。

**要求:**
1. 每天安排2-3个景点
2. 每天必须包含早中晚三餐
3. 每天推荐一个具体的酒店(从酒店信息中选择)
3. 考虑景点之间的距离和交通方式
4. 返回完整的JSON格式数据
5. 景点的经纬度坐标要真实准确
"""
        if request.free_text_input:
            query += f"\n**额外要求:** {request.free_text_input}"

        return query
    
    def _parse_response(self, response: str, request: TripRequest, pois: Optional[List[POIInfo]] = None) -> TripPlan:
        """
        解析Agent响应
        
        Args:
            response: Agent响应文本
            request: 原始请求
            
        Returns:
            旅行计划
        """
        try:
            # 尝试从响应中提取JSON
            # 查找JSON代码块
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
            elif "{" in response and "}" in response:
                # 直接查找JSON对象
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_str = response[json_start:json_end]
            else:
                raise ValueError("响应中未找到JSON数据")
            
            # 解析JSON
            data = json.loads(json_str)
            
            # 转换为TripPlan对象
            trip_plan = TripPlan(**data)
            
            return trip_plan
            
        except Exception as e:
            print(f"[WARN] 解析响应失败: {str(e)}")
            print(f"   将使用备用方案生成计划")
            return self._create_fallback_plan(request, pois)
    
    def _create_fallback_plan(self, request: TripRequest, pois: Optional[List[POIInfo]] = None) -> TripPlan:
        """创建备用计划(当Agent失败时)"""
        from datetime import datetime, timedelta
        
        pois = pois or []
        # 解析日期
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        
        # 创建每日行程
        days = []
        used_poi_ids = set()
        poi_index = 0
        for i in range(request.travel_days):
            current_date = start_date + timedelta(days=i)
            
            attractions = []
            for j in range(2):
                poi = None
                while poi_index < len(pois):
                    candidate = pois[poi_index]
                    poi_index += 1
                    if self._poi_key(candidate) not in used_poi_ids:
                        poi = candidate
                        break

                if poi:
                    used_poi_ids.add(self._poi_key(poi))
                    attractions.append(
                        Attraction(
                            name=poi.name,
                            address=poi.address,
                            location=poi.location,
                            visit_duration=120,
                            description=self._build_poi_description(poi, request.city),
                            category=poi.type or "景点",
                            poi_id=poi.id,
                        )
                    )
                else:
                    placeholder_index = i * 2 + j + 1
                    attractions.append(
                        Attraction(
                            name=f"{request.city}城市探索{placeholder_index}",
                            address=f"{request.city}市",
                            location=self._fallback_location(request.city, i, j),
                            visit_duration=120,
                            description=f"结合{request.city}当地交通和体力情况, 安排周边街区、公园或特色公共空间作为弹性游览点。",
                            category="景点"
                        )
                    )

            day_plan = DayPlan(
                date=current_date.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=attractions,
                meals=[
                    Meal(type="breakfast", name=f"第{i+1}天早餐", description="当地特色早餐"),
                    Meal(type="lunch", name=f"第{i+1}天午餐", description="午餐推荐"),
                    Meal(type="dinner", name=f"第{i+1}天晚餐", description="晚餐推荐")
                ]
            )
            days.append(day_plan)
        
        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions=f"这是为您规划的{request.city}{request.travel_days}日游行程,建议提前查看各景点的开放时间。"
        )


# 全局多智能体系统实例
_multi_agent_planner = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取多智能体旅行规划系统实例(单例模式)"""
    global _multi_agent_planner

    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()

    return _multi_agent_planner
