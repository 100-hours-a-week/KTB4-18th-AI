"""음성 파일 검증과 STT 호출을 담당하는 전사 처리 모듈."""

from fastapi import HTTPException, UploadFile


def transcribe_audio(audio: UploadFile) -> str:
    """음성 파일을 전사문으로 변환한다. 
    
    STT 연결 전에는 미구현 오류를 반환한다.
    """
    
    # TODO: STT 제공자·모델 확정 후 파일 전사 호출과 텍스트 반환을 연결한다.
    # TODO: 연동 확인 후 UploadFile 검사 함수로 수신 크기 상한과 실제 재생 길이(최대 60초)를 검증한다.
    # TODO: 실제 컨테이너·코덱과 디코딩 가능 여부를 검사하고, 검증·STT 실패를 오류 응답에 매핑한다.
    raise HTTPException(status_code=501, detail="Not Implemented")
