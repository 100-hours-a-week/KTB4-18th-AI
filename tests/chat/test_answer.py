"""안내문의 근거와 코드 소유 개수 정책을 검증한다."""

import pytest

from backend.chat.answer import AnswerHints, answer_input, count_notice
from db.types import Track


def test_answer_distinguishes_original_query_and_applied_filters():
    hints = AnswerHints.from_classification({
        "intent": "recommend", "message": "그보다 차분한 재즈 2곡",
        "recommend_min_year": 2020, "recommend_max_year": 2024, "recommend_genres": ["Jazz"],
        "requested_count": 2,
    })
    track = Track(track_id="1", title="a song", artist="an artist", artwork_url="https://art.invalid",
                  preview_url="https://preview.invalid", store_url="https://store.invalid")
    text = answer_input("퇴근길에 듣기 좋은 차분한 재즈", [track], None, hints)
    assert "사용자 원문: 그보다 차분한 재즈 2곡" in text
    assert "해석한 음악 요청·검색 문장: 퇴근길에 듣기 좋은 차분한 재즈" in text
    assert '적용된 장르 필터: ["Jazz"]' in text
    assert "2020~2024년 발매곡만 검색함" in text
    assert '"title": "a song", "artist": "an artist"' in text
    assert "https://" not in text


def test_lookup_does_not_present_unapplied_recommendation_filters():
    hints = AnswerHints.from_classification({"intent": "lookup", "recommend_genres": ["Jazz"],
                                            "recommend_min_year": 2020})
    assert hints.genres is None
    assert hints.year_condition() == "없음"


@pytest.mark.parametrize(("requested", "actual", "notice"), [
    (10, 5, "한 번에 최대 5곡까지 보여드릴 수 있어요."),
    (6, 2, "한 번에 최대 5곡까지 보여드릴 수 있어요. 찾은 곡은 2곡이에요."),
    (2, 1, "찾은 곡은 1곡이에요."),
    (2, 2, ""), (None, 1, ""), (None, 5, ""), (10, 0, "한 번에 최대 5곡까지 보여드릴 수 있어요."),
])
def test_count_notice_uses_actual_results(requested, actual, notice):
    assert count_notice(AnswerHints(requested_count=requested), actual) == notice
