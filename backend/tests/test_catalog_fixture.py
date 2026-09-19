"""합성 카탈로그의 DB 형태와 고정 기대값 검사. 실제 추천 품질 검증은 아니다."""

import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.schemas import Track

FIXTURES = Path(__file__).resolve().parents[2] / "docs" / "DB" / "fixtures"
CATALOG = json.loads((FIXTURES / "catalog.json").read_text())
CASES = json.loads((FIXTURES / "cases.json").read_text())
TRACK_COLUMNS = {
    "track_id", "title", "artist", "album", "genre", "release_date",
    "preview_url", "artwork_url", "store_url", "duration_ms", "seed_artist",
    "bucket", "mood_tags", "emb_clap", "model_version", "embedded_at",
    "fail_count", "fail_reason",
}


def test_fixture_matches_documented_columns_and_boundaries():
    rows = CATALOG["tracks"]
    assert len(rows) == len({r["track_id"] for r in rows}) == 14
    for row in rows:
        assert set(row) == TRACK_COLUMNS
        assert isinstance(row["track_id"], int) and -(2**63) <= row["track_id"] < 0
        assert row["title"] and row["artist"] and row["preview_url"]
        assert row["fail_count"] >= 0
        for key in ("preview_url", "artwork_url", "store_url"):
            assert row[key] is None or row[key].startswith("https://example.invalid/")
        if row["release_date"]:
            date.fromisoformat(row["release_date"])
        if row["embedded_at"]:
            datetime.fromisoformat(row["embedded_at"])
        if row["emb_clap"] is not None:
            vector = row["emb_clap"]
            assert len(vector) == 512 and all(math.isfinite(x) for x in vector)
            assert math.isclose(math.hypot(*vector), 1, abs_tol=1e-8)
            assert row["model_version"] and row["embedded_at"]
        if row["mood_tags"] is not None:
            assert len(row["mood_tags"]) <= 10
            assert all(0.05 <= x <= 1 for x in row["mood_tags"].values())
    stats = CATALOG["corpus_stats"]
    assert len(stats) == 1
    assert set(stats[0]) == {"key", "value", "updated_at"}
    assert stats[0]["key"] == "clap_mean"
    assert stats[0]["value"] == [0] * 512


@pytest.mark.parametrize("case", CASES["retrieval_cases"], ids=lambda c: c["id"])
def test_synthetic_cosine_ranking_and_exact_duplicate_expectations(case):
    # 테스트 벡터의 기대 순위만 확인한다. 운영 검색 함수를 구현하거나 대체하지 않는다.
    query = case["query_vector"]
    assert len(query) == 512 and math.isclose(math.hypot(*query), 1, abs_tol=1e-8)
    eligible = [
        row for row in CATALOG["tracks"]
        if row["emb_clap"] is not None
        and row["model_version"] == CASES["active_model_version"]
        and ("artist_exact" not in case or row["artist"] == case["artist_exact"])
    ]
    def order(row):
        vector = row["emb_clap"]
        cosine = sum(a * b for a, b in zip(query, vector)) / (
            math.hypot(*query) * math.hypot(*vector)
        )
        return -cosine, row["track_id"]

    ranked = sorted(eligible, key=order)[:case["limit"]]
    assert [row["track_id"] for row in ranked] == case["expected_ranked_track_ids"]
    seen = set()
    unique = []
    for row in ranked:
        key = tuple(" ".join(row[field].split()).casefold() for field in ("artist", "title"))
        if key not in seen:
            seen.add(key)
            unique.append(row["track_id"])
    assert unique == case["expected_after_exact_dedup_track_ids"]


@pytest.mark.parametrize("case", CASES["lookup_cases"], ids=lambda c: c["id"])
def test_lookup_can_read_pending_failed_and_incomplete_metadata(case):
    if "track_id" in case:
        row = next(r for r in CATALOG["tracks"] if r["track_id"] == case["track_id"])
        if "expected_title" in case:
            assert row["title"] == case["expected_title"]
            assert row["emb_clap"] is None
        else:
            assert row[case["field"]] == case["expected_value"]
    else:
        assert sorted(r["track_id"] for r in CATALOG["tracks"] if r["title"] == case["title"]) == case["expected_track_ids"]


def test_db_rows_map_to_current_public_cards_without_internal_fields():
    for row in CATALOG["tracks"]:
        if row["emb_clap"] is None or row["model_version"] != CASES["active_model_version"]:
            continue
        card = Track(
            track_id=str(row["track_id"]), title=row["title"], artist=row["artist"],
            artwork_url=row["artwork_url"], preview_url=row["preview_url"],
            store_url=row["store_url"], reason="요청하신 분위기를 기준으로 검색한 곡입니다.",
        )
        assert set(card.model_dump()) == {
            "track_id", "title", "artist", "artwork_url", "preview_url", "store_url", "reason",
        }
