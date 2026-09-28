from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    """사용자가 입력창에서 확인·수정할 음성 전사 초안."""

    transcript: str
