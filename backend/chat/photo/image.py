"""사진 입력 검증과 정규화."""

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from backend.core.errors import _error

MAX_BYTES = 5 * 1024 * 1024
# NOTE: 48MP 촬영 원본을 허용하되, 비정상적으로 큰 이미지의 디코딩 메모리는 제한한다.
MAX_PIXELS = 60_000_000


def normalize_image(data: bytes) -> bytes:
    """손상·과대 이미지를 차단하고 메타데이터 없는 JPEG로 정규화한다.

    NOTE: 중계 과정에서 바뀔 수 있는 MIME 대신 실제 바이트·디코딩 결과로 JPEG 여부를 판단한다.
    """
    if len(data) > MAX_BYTES:
        raise _error(413, "IMAGE_TOO_LARGE", "사진은 5 MiB 이하여야 합니다.")
    if not data.startswith(b"\xff\xd8\xff"):
        raise _error(415, "UNSUPPORTED_IMAGE_FORMAT", "JPEG 사진만 지원합니다.")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG":
                raise ValueError("Not JPEG")
            # NOTE: draft()는 크기 정보를 축소 값으로 바꾸므로 원본 픽셀 상한을 먼저 검사한다.
            if image.width * image.height > MAX_PIXELS:
                raise _error(413, "IMAGE_TOO_LARGE", "사진은 6,000만 픽셀 이하여야 합니다.")
            # NOTE: 원본 전체를 펼치지 않고 1024px 이상인 가장 작은 JPEG 축척으로 디코딩한다.
            image.draft("RGB", (1024, 1024))
            image.load()
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            normalized.thumbnail((1024, 1024))
            # NOTE: EXIF·GPS·주석이 공급자에 전송되지 않도록 새 이미지에 픽셀만 복사한다.
            clean = Image.new("RGB", normalized.size)
            clean.paste(normalized)
            output = io.BytesIO()
            clean.save(output, "JPEG", quality=85)
            return output.getvalue()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise _error(415, "INVALID_IMAGE", "손상되었거나 처리할 수 없는 사진입니다.") from error
