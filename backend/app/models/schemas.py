"""Data models for the trip planner."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class TripRequest(BaseModel):
    """Trip planning request."""

    city: str = Field(..., description="Destination city", examples=["北京"])
    start_date: str = Field(..., description="Start date YYYY-MM-DD", examples=["2025-06-01"])
    end_date: str = Field(..., description="End date YYYY-MM-DD", examples=["2025-06-03"])
    travel_days: int = Field(..., description="Travel days", ge=1, le=30, examples=[3])
    transportation: str = Field(..., description="Transportation preference", examples=["公共交通"])
    accommodation: str = Field(..., description="Accommodation preference", examples=["经济型酒店"])
    preferences: List[str] = Field(default_factory=list, description="Travel preferences")
    free_text_input: Optional[str] = Field(default="", description="Additional requirements")

    model_config = {
        "json_schema_extra": {
            "example": {
                "city": "北京",
                "start_date": "2025-06-01",
                "end_date": "2025-06-03",
                "travel_days": 3,
                "transportation": "公共交通",
                "accommodation": "经济型酒店",
                "preferences": ["历史文化", "美食"],
                "free_text_input": "希望多安排一些博物馆",
            }
        }
    }


class POISearchRequest(BaseModel):
    """POI search request."""

    keywords: str = Field(..., description="Search keywords", examples=["故宫"])
    city: str = Field(..., description="City", examples=["北京"])
    citylimit: bool = Field(default=True, description="Limit search to city")


class RouteRequest(BaseModel):
    """Route planning request."""

    origin_address: str = Field(..., description="Origin address")
    destination_address: str = Field(..., description="Destination address")
    origin_city: Optional[str] = Field(default=None, description="Origin city")
    destination_city: Optional[str] = Field(default=None, description="Destination city")
    route_type: str = Field(default="walking", description="walking/driving/transit")


class Location(BaseModel):
    """Geographic location."""

    longitude: float = Field(..., description="Longitude")
    latitude: float = Field(..., description="Latitude")


class PhotoSpot(BaseModel):
    """Photo spot recommendation."""

    name: str = Field(..., description="Photo spot name")
    description: Optional[str] = Field(default=None, description="Description")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    source: str = Field(default="generated", description="Source")


class Attraction(BaseModel):
    """Attraction information."""

    name: str = Field(..., description="Attraction name")
    address: str = Field(..., description="Address")
    location: Location = Field(..., description="Location")
    visit_duration: int = Field(..., description="Suggested visit duration in minutes")
    description: str = Field(..., description="Description")
    category: Optional[str] = Field(default="景点", description="Category")
    rating: Optional[float] = Field(default=None, description="Rating")
    photos: Optional[List[str]] = Field(default_factory=list, description="Photo URLs")
    poi_id: Optional[str] = Field(default="", description="POI ID")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    review_summary: Optional[str] = Field(default=None, description="Review summary")
    photo_spots: List[str] = Field(default_factory=list, description="Photo spots")
    photo_spot_details: List[PhotoSpot] = Field(default_factory=list, description="Photo spot details")
    visit_tips: List[str] = Field(default_factory=list, description="Visit tips")
    route_tip: Optional[str] = Field(default=None, description="Route tip")
    ticket_price: int = Field(default=0, description="Ticket price")


class Meal(BaseModel):
    """Meal information."""

    type: str = Field(..., description="breakfast/lunch/dinner/snack")
    name: str = Field(..., description="Meal or restaurant name")
    poi_id: Optional[str] = Field(default="", description="POI ID")
    address: Optional[str] = Field(default=None, description="Address")
    location: Optional[Location] = Field(default=None, description="Location")
    description: Optional[str] = Field(default=None, description="Description")
    photos: Optional[List[str]] = Field(default_factory=list, description="Photo URLs")
    image_url: Optional[str] = Field(default=None, description="Image URL")
    review_summary: Optional[str] = Field(default=None, description="Review summary")
    route_tip: Optional[str] = Field(default=None, description="Route tip")
    recommended_reason: Optional[str] = Field(default=None, description="Recommended reason")
    estimated_cost: int = Field(default=0, description="Estimated cost")


class Hotel(BaseModel):
    """Hotel information."""

    name: str = Field(..., description="Hotel name")
    address: str = Field(default="", description="Address")
    location: Optional[Location] = Field(default=None, description="Location")
    price_range: str = Field(default="", description="Price range")
    rating: str = Field(default="", description="Rating")
    distance: str = Field(default="", description="Distance description")
    type: str = Field(default="", description="Hotel type")
    review_summary: Optional[str] = Field(default=None, description="Review summary")
    estimated_cost: int = Field(default=0, description="Estimated cost per night")


class DayPlan(BaseModel):
    """Single-day plan."""

    date: str = Field(..., description="Date YYYY-MM-DD")
    day_index: int = Field(..., description="Zero-based day index")
    description: str = Field(..., description="Day summary")
    transportation: str = Field(..., description="Transportation")
    accommodation: str = Field(..., description="Accommodation")
    hotel: Optional[Hotel] = Field(default=None, description="Recommended hotel")
    attractions: List[Attraction] = Field(default_factory=list, description="Attractions")
    meals: List[Meal] = Field(default_factory=list, description="Meals")


class WeatherInfo(BaseModel):
    """Weather information."""

    date: str = Field(..., description="Date")
    day_weather: str = Field(default="", description="Day weather")
    night_weather: str = Field(default="", description="Night weather")
    day_temp: Union[int, str] = Field(default=0, description="Day temperature")
    night_temp: Union[int, str] = Field(default=0, description="Night temperature")
    wind_direction: str = Field(default="", description="Wind direction")
    wind_power: str = Field(default="", description="Wind power")

    @field_validator("day_temp", "night_temp", mode="before")
    @classmethod
    def parse_temperature(cls, value):
        if isinstance(value, str):
            text = value.replace("°C", "").replace("℃", "").replace("°", "").strip()
            try:
                return int(float(text))
            except ValueError:
                return 0
        return value

    @field_validator("wind_direction", "wind_power", mode="before")
    @classmethod
    def parse_wind_text(cls, value):
        if value is None:
            return ""
        return str(value)


class Budget(BaseModel):
    """Budget information."""

    total_attractions: int = Field(default=0, description="Attraction cost")
    total_hotels: int = Field(default=0, description="Hotel cost")
    total_meals: int = Field(default=0, description="Meal cost")
    total_transportation: int = Field(default=0, description="Transportation cost")
    total: int = Field(default=0, description="Total cost")


class TripPlan(BaseModel):
    """Trip plan."""

    city: str = Field(..., description="Destination city")
    start_date: str = Field(..., description="Start date")
    end_date: str = Field(..., description="End date")
    days: List[DayPlan] = Field(..., description="Daily plans")
    hotels: List[Hotel] = Field(default_factory=list, description="Hotel candidates")
    weather_info: List[WeatherInfo] = Field(default_factory=list, description="Weather")
    overall_suggestions: str = Field(..., description="Overall suggestions")
    budget: Optional[Budget] = Field(default=None, description="Budget")


class TripPlanResponse(BaseModel):
    """Trip plan response."""

    success: bool = Field(..., description="Success")
    message: str = Field(default="", description="Message")
    data: Optional[TripPlan] = Field(default=None, description="Trip plan")


class ChatMessage(BaseModel):
    """Chat message."""

    role: str = Field(..., description="user/assistant/system")
    content: str = Field(..., description="Message content")


class PendingPatchIntent(BaseModel):
    """Pending small-change intent awaiting clarification."""

    operation: str = Field(..., description="Patch operation")
    known_fields: Dict[str, Any] = Field(default_factory=dict, description="Known fields")
    missing_fields: List[str] = Field(default_factory=list, description="Missing fields")
    clarification_question: str = Field(default="", description="Clarification question")


class TripSession(BaseModel):
    """Trip editing session."""

    id: str = Field(..., description="Session ID")
    request: TripRequest = Field(..., description="Original request")
    current_plan: TripPlan = Field(..., description="Current plan")
    messages: List[ChatMessage] = Field(default_factory=list, description="Messages")
    status: str = Field(default="draft", description="Session status")
    pending_patch_intent: Optional[PendingPatchIntent] = Field(default=None, description="Pending patch intent")
    pending_revision_summary: Optional[str] = Field(default=None, description="Pending revision summary")
    plan_versions: List[TripPlan] = Field(default_factory=list, description="Plan versions")


class TripSessionCreateRequest(BaseModel):
    """Create trip session request."""

    request: TripRequest = Field(..., description="Original request")
    plan: TripPlan = Field(..., description="Current plan")


class TripSessionResponse(BaseModel):
    """Trip session response."""

    success: bool = Field(..., description="Success")
    message: str = Field(default="", description="Message")
    data: Optional[TripSession] = Field(default=None, description="Session")


class TripChatRequest(BaseModel):
    """Trip edit chat request."""

    message: str = Field(..., description="User message")


class TripChangeIntent(BaseModel):
    """Trip change intent."""

    change_type: str = Field(..., description="small_change / major_revision / clarification_needed")
    summary: str = Field(..., description="Intent summary")
    patch_operations: List[Dict[str, Any]] = Field(default_factory=list, description="Patch operations")
    clarification_question: Optional[str] = Field(default=None, description="Clarification question")


class TripChatResponse(BaseModel):
    """Trip edit chat response."""

    success: bool = Field(..., description="Success")
    message: str = Field(default="", description="Message")
    data: Optional[TripSession] = Field(default=None, description="Updated session")
    intent: Optional[TripChangeIntent] = Field(default=None, description="Change intent")


class POIInfo(BaseModel):
    """POI information."""

    id: str = Field(..., description="POI ID")
    name: str = Field(..., description="Name")
    type: str = Field(..., description="Type")
    address: str = Field(..., description="Address")
    location: Location = Field(..., description="Location")
    tel: Optional[str] = Field(default=None, description="Phone")


class POISearchResponse(BaseModel):
    """POI search response."""

    success: bool = Field(..., description="Success")
    message: str = Field(default="", description="Message")
    data: List[POIInfo] = Field(default_factory=list, description="POI list")


class RouteInfo(BaseModel):
    """Route information."""

    distance: float = Field(..., description="Distance in meters")
    duration: int = Field(..., description="Duration in seconds")
    route_type: str = Field(..., description="Route type")
    description: str = Field(..., description="Description")


class RouteResponse(BaseModel):
    """Route response."""

    success: bool = Field(..., description="Success")
    message: str = Field(default="", description="Message")
    data: Optional[RouteInfo] = Field(default=None, description="Route")


class WeatherResponse(BaseModel):
    """Weather response."""

    success: bool = Field(..., description="Success")
    message: str = Field(default="", description="Message")
    data: List[WeatherInfo] = Field(default_factory=list, description="Weather list")


class ErrorResponse(BaseModel):
    """Error response."""

    success: bool = Field(default=False, description="Success")
    message: str = Field(..., description="Message")
    error_code: Optional[str] = Field(default=None, description="Error code")
