import pytest
from backend.chat.answer import AnswerHints, count_notice

@pytest.mark.parametrize(("requested", "actual", "notice"), [
    (10, 5, "한 번에 최대 5곡까지 보여드릴 수 있어요."),
    (6, 2, "한 번에 최대 5곡까지 보여드릴 수 있어요. 찾은 곡은 2곡이에요."),
    (2, 1, "찾은 곡은 1곡이에요."),
    (2, 2, ""), (None, 1, ""), (None, 5, ""), (10, 0, "한 번에 최대 5곡까지 보여드릴 수 있어요."),
])
def test_count_notice_uses_actual_results(requested, actual, notice):
    assert count_notice(AnswerHints(requested_count=requested), actual) == notice
