"""LLM 출력의 장소 단어가 정해진 보기 안에서만 나오는지 검증한다."""

import pytest
from pydantic import ValidationError

from backend.location.query import LocationQuery


def test_place_must_be_one_of_words():
    """보기에 없는 자유 문장(이름이 섞인 표현 포함)은 스키마 위반이다"""
    with pytest.raises(ValidationError):
        LocationQuery(place="해운대 바닷가", message="해운대예요.")


def test_schema_sends_words_as_enum():
    """json_schema strict로 보내는 스키마에 보기가 enum으로 들어가야 모델이 그 안에서 고른다"""
    schema = LocationQuery.model_json_schema()
    assert "해수욕장" in schema["properties"]["place"]["enum"]
