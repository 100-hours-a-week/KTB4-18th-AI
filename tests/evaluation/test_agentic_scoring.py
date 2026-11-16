"""에이전틱 대화 기대값(질문 여부·에피소드·답 반영·양자택일)의 코드 판정을 검증한다."""

from evaluation.grading.scoring import route_of, score_case, score_turn

TRACK = {"track_id": "1", "title": "t", "artist": "a"}


def actual(text="", tracks=(TRACK,), **state):
    return dict(done=True, text=text, tracks=list(tracks), state=state)


def scores(expected, now, before=None):
    return {k: v["score"] for k, v in score_turn(expected, now, before=before).items()}


def test_asks_counts_only_delivered_questions_in_same_episode():
    before = {"episode": 1, "question_count": 1}
    assert scores({"asks": True}, actual(episode=1, question_count=2), before)["asks"] == 1
    assert scores({"asks": False}, actual(episode=1, question_count=1), before)["asks"] == 1
    # 새 에피소드면 이전 횟수와 비교하지 않고 0부터 센다.
    assert scores({"asks": True}, actual(episode=2, question_count=1), before)["asks"] == 1
    assert scores({"asks": True}, actual(episode=2, question_count=0), before)["asks"] == 0


def test_episode_new_or_same():
    assert scores({"episode": "new"}, actual(episode=1))["episode"] == 1
    assert scores({"episode": "same"}, actual(episode=1), {"episode": 1})["episode"] == 1
    assert scores({"episode": "new"}, actual(episode=1), {"episode": 1})["episode"] == 0
    assert scores({"episode": "new"}, actual())["episode"] == 0


def test_missing_dialogue_state_fails():
    before = {"episode": 1, "question_count": 1}
    assert scores({"episode": "same"}, actual(), before)["episode"] == 0
    assert scores({"asks": False}, actual(episode=1), before)["asks"] == 0
    assert scores({"asks": False}, actual(episode=1, question_count=0), before)["asks"] == 0


def test_query_includes_any_context_word():
    now = actual(recommend_query="버스 정류장에서 비 올 때 듣는 노래")
    assert scores({"query_includes_any": ["정류장", "버스"]}, now)["query_includes_any"] == 1
    assert scores({"query_includes_any": ["카페"]}, now)["query_includes_any"] == 0


def test_no_binary_choice():
    open_question = "어디에 계신지 알려주시면 더 맞춰 드릴게요."
    choice = "차분하게 즐기고 싶으세요, 아니면 신나게 듣고 싶으세요?"
    assert scores({"no_binary_choice": True}, actual(open_question))["no_binary_choice"] == 1
    assert scores({"no_binary_choice": True}, actual(choice))["no_binary_choice"] == 0
    for choice in ("차분한 게 좋아요, 신나는 게 좋아요?", "잔잔한 곡 또는 신나는 곡 중 무엇이 좋을까요?",
                   "잔잔한/신나는 곡 중 골라 주세요.", "잔잔하게 들을지 신나게 들을지 골라주세요.",
                   "잔잔한 곡과 신나는 곡, 어느 쪽을 원하세요?", "실내인지 실외인지 알려주세요.",
                   "조용한 카페에 계신지 사람이 많은 야외에 계신지 알려주세요.",
                   "잔잔한 곡이 좋으신가요, 신나는 곡이 좋으신가요?", "실내와 야외 중에서 골라 주세요.",
                   "잔잔한 곡과 신나는 곡 중에 어떤 곡을 원하세요?"):
        assert scores({"no_binary_choice": True}, actual(choice))["no_binary_choice"] == 0, choice


def test_open_question_examples_are_not_binary():
    for text in ("현재 어디에 있으신지, 어떤 분위기인지 등 조금 더 자세한 설명을 해주시면 더 좋은 노래를 "
                 "추천해드릴 수 있어요.", "장소나 하고 계신 일을 알려주시면 더 맞춰 드릴게요.",
                 "현재 어디에 있으신지 어떤 분위기를 원하시는지 알려주시면 더 맞춰 드릴게요.",
                 "무엇인지 어디인지 알려주세요."):
        assert scores({"no_binary_choice": True}, actual(text))["no_binary_choice"] == 1, text


def test_insufficient_info_with_tracks_routes_to_tracks():
    state = {"intent": "recommend", "recommend_has_enough_info": False}
    assert route_of({**state, "tracks": [TRACK]}) == ["recommend", "tracks"]
    assert route_of(state) == ["recommend", "insufficient"]


def test_score_case_passes_previous_state():
    case = {"metadata": {}, "turns": [
        {"expected": {"episode": "new"}, "actual": actual(episode=1, question_count=1)},
        {"expected": {"episode": "same", "asks": False}, "actual": actual(episode=1, question_count=1)},
    ]}
    second = score_case(case)["turns"][1]["scores"]
    assert second["episode"]["score"] == 1 and second["asks"]["score"] == 1


def test_episode_new_requires_exactly_next_number():
    assert scores({"episode": "new"}, actual(episode=0))["episode"] == 0
    assert scores({"episode": "new"}, actual(episode=1), {"episode": 2})["episode"] == 0
    assert scores({"episode": "new"}, actual(episode=4), {"episode": 2})["episode"] == 0
    assert scores({"episode": "new"}, actual(episode=3), {"episode": 2})["episode"] == 1
