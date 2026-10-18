"""Modal GPU에서 사진 분석 폴백 VLM을 vLLM OpenAI 호환 서버로 띄운다.

    VLM_PROFILE=e4b-l40s uvx modal deploy --env main infra/modal/vlm_server.py

모델·GPU·버전은 PROFILES의 항목 하나로 바꾼다. 백엔드는 별칭(photo-vlm)만 알므로 교체해도 AI 서버 설정은 그대로다.
"""

import os
import subprocess
import time

import modal

# NOTE: 버전은 vLLM 공식 이미지 태그와 transformers를 함께 고정한다. 바꿀 때는 새 항목을 만들고 28장 회귀 측정을 거친다.
PROFILES = {
    "e4b-l40s": {"model": "google/gemma-4-E4B-it", "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
                 "gpu": "L40S", "vllm": "v0.30.0", "transformers": "5.17.0"},
    "e4b-l4": {"model": "google/gemma-4-E4B-it", "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
               "gpu": "L4", "vllm": "v0.30.0", "transformers": "5.17.0"},
}
PROFILE_NAME = os.environ.get("VLM_PROFILE", "e4b-l40s")
PROFILE = PROFILES[PROFILE_NAME]
SERVED_NAME = "photo-vlm"
VLLM_PORT = 8000
MINUTES = 60

image = (
    modal.Image.from_registry(f"vllm/vllm-openai:{PROFILE['vllm']}")
    .entrypoint([])
    .run_commands("ln -s $(which python3) /usr/bin/python")
    .pip_install(f"transformers=={PROFILE['transformers']}")
    # NOTE: 컨테이너 안에서도 배포 때와 같은 프로필을 읽도록 이미지에 고정한다.
    # VLLM_SERVER_DEV_MODE는 스냅샷 전후의 /sleep·/wake_up을 켠다. 외부 접근은 Modal 프록시 인증으로 막는다.
    # NOTE: 스냅샷이 참조하는 Volume 파일이 사라지면 복원이 실패한다. 다운로드 임시 파일(xet)은 Volume 밖에 둔다.
    .env({"VLM_PROFILE": PROFILE_NAME, "HF_XET_HIGH_PERFORMANCE": "1", "HF_XET_CACHE": "/tmp/hf-xet",
          "VLLM_SERVER_DEV_MODE": "1", "TORCH_CPP_LOG_LEVEL": "FATAL"})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("photo-vlm-fallback")

with image.imports():
    import base64
    import io

    import requests
    from PIL import Image


def wait_ready(process: subprocess.Popen) -> None:
    """vLLM이 /health에 200을 줄 때까지 기다린다."""
    deadline = time.time() + 15 * MINUTES
    while time.time() < deadline:
        if (code := process.poll()) is not None:
            raise RuntimeError(f"vLLM exited with {code}")
        try:
            requests.get(f"http://127.0.0.1:{VLLM_PORT}/health", timeout=5).raise_for_status()
            return
        except requests.RequestException:
            time.sleep(5)
    raise TimeoutError("vLLM not ready in 15 minutes")


def warmup() -> None:
    """스냅샷 전에 서비스와 같은 크기의 사진·구조화 출력으로 첫 요청 컴파일을 끝낸다."""
    buffer = io.BytesIO()
    Image.new("RGB", (1024, 768), (90, 140, 200)).save(buffer, "JPEG", quality=85)
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    payload = {
        "model": SERVED_NAME, "max_tokens": 32,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Describe the colors."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "warmup", "strict": True, "schema": {
            "type": "object", "properties": {"colors": {"type": "string"}}, "required": ["colors"],
            "additionalProperties": False}}},
    }
    for _ in range(2):
        requests.post(f"http://127.0.0.1:{VLLM_PORT}/v1/chat/completions", json=payload, timeout=5 * MINUTES).raise_for_status()


@app.server(
    image=image,
    gpu=PROFILE["gpu"],
    port=VLLM_PORT,
    # NOTE: 폴백은 주 경로가 실패할 때만 깨어난다. 요청 없이 20분(Modal 최대값)이 지나야 꺼지게 해 장애 중 이어지는 요청을 받는다.
    # 실행 중 변경: modal.Server.from_name(...).update_autoscaler(scaledown_window=...), 다음 배포 때 이 값으로 돌아온다.
    scaledown_window=20 * MINUTES,
    startup_timeout=15 * MINUTES,
    # NOTE: 비용 상한을 위해 1대로 묶는다. 재배포 중에는 잠깐 응답하지 않으며, 폴백 특성상 감수한다.
    max_containers=1,
    # NOTE: 스냅샷 복원은 4번 중 1번만 일어났다(알파 기능, 2026-10-18 측정). 나머지는 일반 기동이라 컴파일 캐시로 그 시간을 줄인다
    # (컴파일 약 60초 → 13초). 캐시 임시 파일이 바뀌어 복원이 실패하면 Modal이 복원을 다시 시도한다.
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    # NOTE: GPU 메모리 스냅샷은 Modal 알파 기능이다. 첫 기동에서 초기화·컴파일·예열을 마친 상태를 저장하고 이후 복원한다.
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": True},
    unauthenticated=False,
)
class VlmServer:
    """vLLM 프로세스 하나를 컨테이너 수명에 맞춰 띄운다."""

    @modal.enter(snap=True)
    def start(self):
        """초기화·예열을 마친 뒤 재워서(KV 캐시 비움) 그 상태를 스냅샷으로 남긴다."""
        print(f"profile={PROFILE_NAME} {PROFILE}")
        self.process = subprocess.Popen([
            "vllm", "serve", PROFILE["model"],
            "--revision", PROFILE["revision"],
            # NOTE: 첫 이름이 응답의 model 값이다. 실제 모델 ID도 함께 받아 측정 도구가 그대로 쓸 수 있게 한다.
            "--served-model-name", SERVED_NAME, PROFILE["model"],
            "--host", "0.0.0.0", "--port", str(VLLM_PORT),
            "--max-model-len", "8192",
            "--limit-mm-per-prompt", '{"image": 1, "video": 0, "audio": 0}',
            "--enable-sleep-mode",
        ])
        wait_ready(self.process)
        warmup()
        requests.post(f"http://127.0.0.1:{VLLM_PORT}/sleep?level=1", timeout=5 * MINUTES).raise_for_status()

    @modal.enter(snap=False)
    def wake(self):
        """스냅샷 복원 뒤 가중치를 GPU로 되돌린다."""
        requests.post(f"http://127.0.0.1:{VLLM_PORT}/wake_up", timeout=5 * MINUTES).raise_for_status()

    @modal.exit()
    def stop(self):
        """컨테이너 종료 시 vLLM 프로세스를 정리한다."""
        self.process.terminate()
