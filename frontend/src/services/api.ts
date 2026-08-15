import axios from 'axios'
import type {
  TripChatResponse,
  TripFormData,
  TripPlanResponse,
  TripSessionCreateRequest,
  TripSessionResponse
} from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000, // 5分钟超时
  headers: {
    'Content-Type': 'application/json'
  }
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.config.url)
    return response
  },
  (error) => {
    console.error('响应错误:', error.response?.status, error.message)
    return Promise.reject(error)
  }
)

/**
 * 生成旅行计划
 */
export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error: any) {
    console.error('生成旅行计划失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '生成旅行计划失败')
  }
}

/**
 * 创建旅行规划会话
 */
export async function createTripSession(payload: TripSessionCreateRequest): Promise<TripSessionResponse> {
  try {
    const response = await apiClient.post<TripSessionResponse>('/api/trip/sessions', payload)
    return response.data
  } catch (error: any) {
    console.error('创建旅行规划会话失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '创建旅行规划会话失败')
  }
}

/**
 * 发送旅行规划编辑对话
 */
export async function chatWithTripSession(sessionId: string, message: string): Promise<TripChatResponse> {
  try {
    const response = await apiClient.post<TripChatResponse>(
      `/api/trip/sessions/${sessionId}/chat`,
      { message }
    )
    return response.data
  } catch (error: any) {
    console.error('旅行规划编辑对话失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '旅行规划编辑对话失败')
  }
}

/**
 * 确认并重新规划旅行会话
 */
export async function reviseTripSession(sessionId: string): Promise<TripSessionResponse> {
  try {
    const response = await apiClient.post<TripSessionResponse>(
      `/api/trip/sessions/${sessionId}/revise`,
      {}
    )
    return response.data
  } catch (error: any) {
    console.error('重新规划旅行会话失败:', error)
    throw new Error(error.response?.data?.detail || error.message || '重新规划旅行会话失败')
  }
}

/**
 * 健康检查
 */
export async function healthCheck(): Promise<any> {
  try {
    const response = await apiClient.get('/health')
    return response.data
  } catch (error: any) {
    console.error('健康检查失败:', error)
    throw new Error(error.message || '健康检查失败')
  }
}

export default apiClient
