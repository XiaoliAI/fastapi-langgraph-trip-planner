// 类型定义

export interface Location {
  longitude: number
  latitude: number
}

export interface Attraction {
  name: string
  address: string
  location: Location
  visit_duration: number
  description: string
  category?: string
  rating?: number
  photos?: string[]
  image_url?: string
  poi_id?: string
  review_summary?: string
  photo_spots?: string[]
  photo_spot_details?: PhotoSpot[]
  visit_tips?: string[]
  route_tip?: string
  ticket_price?: number
}

export interface PhotoSpot {
  name: string
  description?: string
  image_url?: string
  source?: string
}

export interface Meal {
  type: 'breakfast' | 'lunch' | 'dinner' | 'snack'
  name: string
  poi_id?: string
  address?: string
  location?: Location
  description?: string
  photos?: string[]
  image_url?: string
  review_summary?: string
  route_tip?: string
  recommended_reason?: string
  estimated_cost?: number
}

export interface Hotel {
  name: string
  address: string
  location?: Location
  price_range: string
  rating: string
  distance: string
  type: string
  review_summary?: string
  estimated_cost?: number
}

export interface Budget {
  total_attractions: number
  total_hotels: number
  total_meals: number
  total_transportation: number
  total: number
}

export interface DayPlan {
  date: string
  day_index: number
  description: string
  transportation: string
  accommodation: string
  hotel?: Hotel
  attractions: Attraction[]
  meals: Meal[]
}

export interface WeatherInfo {
  date: string
  day_weather: string
  night_weather: string
  day_temp: number
  night_temp: number
  wind_direction: string
  wind_power: string
}

export interface TripPlan {
  city: string
  start_date: string
  end_date: string
  days: DayPlan[]
  hotels?: Hotel[]
  weather_info: WeatherInfo[]
  overall_suggestions: string
  budget?: Budget
}

export interface TripFormData {
  city: string
  start_date: string
  end_date: string
  travel_days: number
  transportation: string
  accommodation: string
  preferences: string[]
  free_text_input: string
}

export interface TripPlanResponse {
  success: boolean
  message: string
  data?: TripPlan
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
}

export interface PendingPatchIntent {
  operation: string
  known_fields: Record<string, any>
  missing_fields: string[]
  clarification_question: string
}

export interface TripSession {
  id: string
  request: TripFormData
  current_plan: TripPlan
  messages: ChatMessage[]
  status: string
  pending_patch_intent?: PendingPatchIntent | null
  pending_revision_summary?: string | null
  plan_versions: TripPlan[]
}

export interface TripSessionCreateRequest {
  request: TripFormData
  plan: TripPlan
}

export interface TripSessionResponse {
  success: boolean
  message: string
  data?: TripSession
}

export interface TripChatRequest {
  message: string
}

export interface TripChangeIntent {
  change_type: 'small_change' | 'major_revision' | 'clarification_needed'
  summary: string
  patch_operations: Record<string, any>[]
  clarification_question?: string | null
}

export interface TripChatResponse {
  success: boolean
  message: string
  data?: TripSession
  intent?: TripChangeIntent
}
