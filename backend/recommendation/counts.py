"""추천·조회에 공통으로 적용하는 곡 수 정책."""

DEFAULT_TRACK_COUNT = 5
MAX_TRACK_COUNT = 5


def track_limit(requested_count: int | None) -> int:
    if requested_count is None:
        return DEFAULT_TRACK_COUNT
    if type(requested_count) is not int or requested_count < 1:
        raise ValueError("곡 수는 1 이상의 정수여야 합니다.")
    return min(requested_count, MAX_TRACK_COUNT)
