from pydantic import BaseModel


class Track(BaseModel):
    """추천곡에 대한 Pydantic 모델."""

    track_id: str  # iTunes track ID
    title: str  # iTunes track title
    artist: str  # iTunes track artist
    artwork_url: str | None  # iTunes album artwork URL (600x600)
    preview_url: str | None  # iTunes track preview URL (30초 미리듣기)
    store_url: str | None = None  # iTunes Store 링크
    reason: str  # 현재 입력과 해당 곡이 어울리는 이유
