"""오픈웨이트 VLM 엔드포인트에 사진 한 장을 실제 사진 추천 경로 그대로 보내 보는 확인용 스크립트.

backend/chat/photo의 정규화·VLM 호출 함수를 그대로 불러, 서비스와 같은 프롬프트·스키마로 요청한다.
.env의 PHOTO_FALLBACK_URL·PHOTO_FALLBACK_MODEL·PHOTO_FALLBACK_API_KEY를 읽는다.

사용: .venv/bin/python serving/smoke_test.py <사진 경로>
"""

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from backend.chat.photo import image, vlm  # noqa: E402


def main() -> None:
    normalized = image.normalize_image(Path(sys.argv[1]).read_bytes())

    start = time.perf_counter()
    scene = asyncio.run(vlm.call_vlm(normalized, fallback=True))
    elapsed = time.perf_counter() - start

    print(json.dumps(scene.model_dump(), ensure_ascii=False, indent=2))
    print(f"  {elapsed:.2f}초, 정규화된 사진 {len(normalized) / 1024:.0f}KB")


if __name__ == "__main__":
    main()
