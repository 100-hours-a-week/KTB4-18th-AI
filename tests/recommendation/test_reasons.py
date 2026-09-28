from types import SimpleNamespace

from backend.recommendation import reasons


def test_llm_model_and_invalid_reason_json(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "openai/test-model")

    def create(**kwargs):
        assert kwargs["model"] == "openai/test-model"
        return SimpleNamespace(output_text="[]")

    track = SimpleNamespace(track_id="1", title="곡", artist="가수", reason="기존 이유")
    reasons.assign_reasons(SimpleNamespace(responses=SimpleNamespace(create=create)),
                                  "음악", [track], {})
    assert track.reason == "기존 이유"
