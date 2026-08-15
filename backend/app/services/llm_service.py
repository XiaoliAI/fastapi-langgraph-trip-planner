"""LLM服务模块"""

from hello_agents import HelloAgentsLLM
from ..config import get_settings
from langchain_openai import ChatOpenAI
import os
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values
# 全局LLM实例
_llm_instance = None
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

def get_llm() -> HelloAgentsLLM:
    """
    获取LLM实例(单例模式)
    
    Returns:
        HelloAgentsLLM实例
    """
    global _llm_instance
    
    if _llm_instance is None:
        settings = get_settings()
        
        # HelloAgentsLLM会自动从环境变量读取配置
        # 包括OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL等
        _llm_instance = HelloAgentsLLM()
        
        print(f"✅ LLM服务初始化成功")
        print(f"   提供商: {_llm_instance.provider}")
        print(f"   模型: {_llm_instance.model}")
    
    return _llm_instance


def reset_llm():
    """
    重置 LLM 实例，用于测试或重新加载配置。
    """
    global _llm_instance, _chat_model_instance

    _llm_instance = None
    _chat_model_instance = None


#采用langchian
_chat_model_instance = None

def get_chat_model() -> ChatOpenAI:
    global _chat_model_instance

    if _chat_model_instance is None:
        settings = get_settings()

        api_key = (
            _read_env_value("LLM_API_KEY")
            or _read_env_value("OPENAI_API_KEY")
            or settings.openai_api_key
        )
        base_url = (
            _read_env_value("LLM_BASE_URL")
            or _read_env_value("OPENAI_BASE_URL")
            or settings.openai_base_url
        )
        model = (
            _read_env_value("LLM_MODEL_ID")
            or _read_env_value("OPENAI_MODEL")
            or settings.openai_model
        )
        timeout_text = _read_env_value("LLM_TIMEOUT")
        timeout = float(timeout_text) if timeout_text else None

        _chat_model_instance = ChatOpenAI(
            api_key=api_key or None,
            base_url=base_url,
            model=model,
            temperature=0.3,
            timeout=timeout,
        )

        print("LangChain Chat Model 初始化成功")
        print(f"   Base URL: {base_url}")
        print(f"   Model: {model}")

    return _chat_model_instance