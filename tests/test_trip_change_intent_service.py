from backend.app.services.trip_change_intent_service import classify_trip_change_intent


def test_classify_major_revision():
    intent = classify_trip_change_intent("我想少走路，第二天轻松一点")

    assert intent.change_type == "major_revision"
    assert "少走路" in intent.summary
    assert intent.clarification_question is None


def test_classify_small_change():
    intent = classify_trip_change_intent("删除第一天的故宫")

    assert intent.change_type == "small_change"
    assert intent.patch_operations == []
    assert intent.clarification_question is None


def test_classify_clarification_needed():
    intent = classify_trip_change_intent("不太满意")

    assert intent.change_type == "clarification_needed"
    assert intent.clarification_question is not None


def test_classify_with_llm_falls_back_to_rules_for_now():
    intent = classify_trip_change_intent("删除第一天的故宫", use_llm=True)

    assert intent.change_type == "small_change"


def test_classify_trip_change_intent_accepts_knowledge_text():
    intent = classify_trip_change_intent(
        message="我想少走路，整体轻松一点",
        knowledge_text="少走路建议：优先安排同一区域景点，减少跨区域移动。",
    )

    assert intent.change_type == "major_revision"
    assert "少走路" in intent.summary
