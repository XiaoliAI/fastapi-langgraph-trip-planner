import os

from app.services import llm_service


def mask(value: str | None) -> str:
    if not value:
        return "EMPTY"
    if len(value) <= 8:
        return "***"
    return value[:4] + "..." + value[-4:]


def main():
    env_path = llm_service._backend_env_path()

    print("backend env path:")
    print(env_path)

    print("\nexists:")
    print(env_path.exists())

    print("\nprocess env:")
    print("LLM_API_KEY =", mask(os.getenv("LLM_API_KEY")))
    print("OPENAI_API_KEY =", mask(os.getenv("OPENAI_API_KEY")))
    print("LLM_BASE_URL =", os.getenv("LLM_BASE_URL"))
    print("OPENAI_BASE_URL =", os.getenv("OPENAI_BASE_URL"))
    print("LLM_MODEL_ID =", os.getenv("LLM_MODEL_ID"))
    print("OPENAI_MODEL =", os.getenv("OPENAI_MODEL"))

    print("\nread env values:")
    print("LLM_API_KEY =", mask(llm_service._read_env_value("LLM_API_KEY")))
    print("OPENAI_API_KEY =", mask(llm_service._read_env_value("OPENAI_API_KEY")))
    print("LLM_BASE_URL =", llm_service._read_env_value("LLM_BASE_URL"))
    print("OPENAI_BASE_URL =", llm_service._read_env_value("OPENAI_BASE_URL"))
    print("LLM_MODEL_ID =", llm_service._read_env_value("LLM_MODEL_ID"))
    print("OPENAI_MODEL =", llm_service._read_env_value("OPENAI_MODEL"))


if __name__ == "__main__":
    main()