"""주간 배치의 곡 필터·중복 키·아티스트 확장 규칙. 네트워크와 DB 없이 검사한다.

사례는 실제 tracks 15,436곡의 제목·앨범명·duration_ms로 검증한 것이다.
"""

import pytest


@pytest.fixture
def weekly(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://test:test@localhost/test")
    from batch import weekly
    return weekly


def track(title, album="Album", ms=200_000, artist_id=1, track_id=1):
    return {"trackName": title, "collectionName": album, "trackTimeMillis": ms,
            "artistId": artist_id, "trackId": track_id, "wrapperType": "track",
            "kind": "song"}


@pytest.mark.parametrize("title, album, ms", [
    ("Ditto (Mixed)", "INS LAND: ISSA LEE at Culture Club (DJ Mix)", 187_708),
    ("ETA [Edit] [Mixed]", "Ray Volpe at EDC Orlando, 2024 (DJ Mix)", 36_757),
    ("I'm On Fire", "Diskonnected - Mixed by Lazy Rich", 362_789),
    ("Thunder / Young Dumb & Broke (Medley)", "Single", 251_284),
    ("Album (Highlight Medley)", "EP", 150_000),
    ("Song (Preview)", "EP", 200_000),
    ("SKIT", "19.99", 66_773),
    ("Interlude: The Glory Is in You", "A Seat at the Table", 17_772),
    ("Intro", "Wonder", 62_423),
])
def test_junk_is_excluded(weekly, title, album, ms):
    assert weekly.is_junk(track(title, album, ms))


@pytest.mark.parametrize("title, album, ms", [
    ("Mixed Up", "Album", 183_000),                        # ENHYPEN
    ("Highlight", "Going Seventeen", 225_302),             # 세븐틴
    ("Teaser", "Remember", 224_443),                       # 위너
    ("Preview", "Spring Memories - EP", 220_668),          # 엔플라잉
    ("HIGHLIGHT (GARBAGE TIME)", "HIGHLIGHT (GARBAGE TIME) - Single", 179_660),
    ("Two Girl Love a Man", "Fe's 10th : Preview - EP", 245_080),
    ("SKIT (feat. LeeHi & Loco)", "WEATHER REPORT", 151_787),
    ("Rebirth (Intro)", "MUSE", 144_000),                  # 지민
    ("Outro: Tear", "LOVE YOURSELF 轉 'Tear'", 284_981),
    ("Blue Side (Outro)", "Hope World", 90_540),
])
def test_real_songs_with_junk_words_are_kept(weekly, title, album, ms):
    assert not weekly.is_junk(track(title, album, ms))


@pytest.mark.parametrize("title, album", [
    ("Song (Remix)", "Single"),
    ("Song", "Song (Sped Up) - Single"),
    ("Song (Live)", "Album"),
    ("Song - Remastered 2011", "Album"),
    ("Song (Japanese Ver.)", "Album"),
    ("Song", "Song (Instrumental) - EP"),
])
def test_other_versions_are_covers(weekly, title, album):
    assert weekly.is_cover(track(title, album))


@pytest.mark.parametrize("title, album", [
    ("Alive", "Album"),                # 부분 문자열 매칭이면 live에 걸린다
    ("Live Your Life", "Album"),       # 괄호 밖 제목 자체는 보지 않는다
    ("Song", "Album (Deluxe Edition)"),  # edit가 Edition에 걸리지 않는다
])
def test_originals_are_not_covers(weekly, title, album):
    assert not weekly.is_cover(track(title, album))


def test_dedup_key_ignores_case(weekly):
    assert weekly.dedup_key("IU", "Blueming") == weekly.dedup_key("iu", "BLUEMING")


def test_artist_top_keeps_same_artist_only(weekly, monkeypatch):
    results = [track("Cover", artist_id=99), track("Hit", artist_id=1),
               track("Hit 2", artist_id=1)]
    monkeypatch.setattr(weekly, "get_json", lambda url: {"results": results})
    assert [r["trackName"] for r in weekly.artist_top(1, "Name")] == ["Hit", "Hit 2"]


def test_artist_top_falls_back_to_recent(weekly, monkeypatch):
    """흔한 이름이라 검색 결과가 전부 다른 아티스트면 최근 발매순으로 대신한다."""
    calls = []

    def fake_get_json(url):
        calls.append(url)
        if "search" in url:
            return {"results": [track("Other", artist_id=99)]}
        return {"results": [track("Recent", artist_id=1)]}

    monkeypatch.setattr(weekly, "get_json", fake_get_json)
    monkeypatch.setattr(weekly.time, "sleep", lambda s: None)
    assert [r["trackName"] for r in weekly.artist_top(1, "Shane")] == ["Recent"]
    assert "sort=recent" in calls[-1]
