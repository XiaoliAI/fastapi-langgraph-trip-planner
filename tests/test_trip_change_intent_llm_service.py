from backend.app.services.trip_change_intent_service import (
    SMALL_CHANGE_KEYWORDS,
    build_intent_classification_prompt,
    classify_trip_change_intent,
)


class FakeLLMResponse:
    def __init__(self, content):
        self.content = content


class FakeChatModel:
    def __init__(self, content):
        self.content = content
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeLLMResponse(self.content)


def test_classify_with_llm_parses_json_intent(monkeypatch):
    fake_model = FakeChatModel(
        """
        {
            "change_type": "major_revision",
            "summary": "User wants a lower-walking itinerary.",
            "patch_operations": [],
            "clarification_question": null
        }
        """
    )

    monkeypatch.setattr(
        "backend.app.services.trip_change_intent_service.get_chat_model",
        lambda: fake_model,
    )

    intent = classify_trip_change_intent("make the trip easier", use_llm=True)

    assert intent.change_type == "major_revision"
    assert intent.summary == "User wants a lower-walking itinerary."


def test_classify_with_llm_puts_knowledge_text_in_prompt(monkeypatch):
    fake_model = FakeChatModel(
        """
        {
            "change_type": "clarification_needed",
            "summary": "Need more detail.",
            "patch_operations": [],
            "clarification_question": "Which part do you want to adjust?"
        }
        """
    )

    monkeypatch.setattr(
        "backend.app.services.trip_change_intent_service.get_chat_model",
        lambda: fake_model,
    )

    classify_trip_change_intent(
        message="make it easier",
        knowledge_text="Low walking trips should group nearby attractions.",
        use_llm=True,
    )

    assert fake_model.prompts
    assert "Low walking trips should group nearby attractions." in fake_model.prompts[0]


def test_intent_classification_prompt_requires_chinese_summary():
    prompt = build_intent_classification_prompt(
        message="make the trip easier",
        knowledge_text="Low walking trips should group nearby attractions.",
    )

    assert "summary 必须使用中文" in prompt
    assert "clarification_question 必须使用中文" in prompt
    assert "即使用户使用英文表达" in prompt


def test_classify_with_llm_falls_back_to_rules_for_invalid_json(monkeypatch):
    fake_model = FakeChatModel("not json")

    monkeypatch.setattr(
        "backend.app.services.trip_change_intent_service.get_chat_model",
        lambda: fake_model,
    )

    intent = classify_trip_change_intent(
        f"{SMALL_CHANGE_KEYWORDS[0]} first day palace",
        use_llm=True,
    )

    assert intent.change_type == "small_change"
