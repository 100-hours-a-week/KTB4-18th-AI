"""오픈웨이트 VLM을 Modal GPU 위에 vLLM으로 띄운다 (OpenAI 호환 API).

사진 추천 VLM(backend/chat/photo/vlm.py)의 폴백 서버다. 우리 서버는 GPU를 갖지 않고,
이 엔드포인트를 HTTPS로 호출만 한다.

처음 한 번:
  uvx modal secret create llm-serving VLLM_API_KEY=<임의의 긴 문자열>
배포:
  uvx modal deploy serving/modal_vllm.py
  MODEL_NAME=Qwen/Qwen3-VL-4B-Instruct-FP8 uvx modal deploy serving/modal_vllm.py  # 다른 후보
내리기 (켜둔 채 잊으면 scaledown 전까지 과금):
  uvx modal app stop <앱 이름>

배포가 끝나면 출력되는 URL 뒤에 /v1 을 붙인 것이 OpenAI 호환 base_url이다.
"""

import os
import re
import subprocess

import modal

# NOTE: 이 파일은 배포할 때(로컬)와 컨테이너 안에서 각각 한 번씩 import된다.
# 로컬에서 고른 값이 컨테이너에서도 같도록 아래 env로 컨테이너에 그대로 넘긴다.
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct-FP8")
GPU = os.environ.get("GPU", "L4")
# 자주 0대에서 깨어나는 폴백 용도라 부팅 속도를 우선한다(CUDA graph·컴파일 생략).
# 부하 테스트로 최대 처리량을 볼 때는 FAST_BOOT=0으로 배포해 비교한다.
FAST_BOOT = os.environ.get("FAST_BOOT", "1") == "1"
# 사진은 긴 변 1024px로 정규화돼 들어오고 프롬프트도 짧아, 문맥 길이를 줄여 KV cache 메모리를 아낀다.
MAX_MODEL_LEN = 8192
# 요청 하나에 사진 한 장만 온다. 동영상 입력을 막아 vLLM이 그만큼의 메모리를 미리 잡지 않게 한다.
MM_LIMIT = '{"image": 1, "video": 0}'
VLLM_PORT = 8000
MINUTES = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

# 가중치·컴파일 결과를 볼륨에 캐시해 컨테이너가 새로 뜰 때마다 다시 받지 않게 한다.
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# 모델마다 앱을 따로 두어 후보 여러 개를 동시에 띄워 비교할 수 있게 한다.
app = modal.App("llm-" + re.sub(r"[^a-z0-9]+", "-", MODEL_NAME.lower()).strip("-"))


@app.server(
    image=image,
    gpu=GPU,
    env={"MODEL_NAME": MODEL_NAME, "GPU": GPU, "FAST_BOOT": "1" if FAST_BOOT else "0"},
    secrets=[modal.Secret.from_name("llm-serving")],
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    port=VLLM_PORT,
    startup_timeout=15 * MINUTES,  # 첫 배포는 가중치 다운로드까지 기다려야 한다
    scaledown_window=5 * MINUTES,  # 요청이 끊기고 이만큼 지나면 0대로 내려가 과금이 멈춘다
    max_containers=1,  # 실험 단계 비용 상한. 부하가 몰려도 GPU를 더 늘리지 않는다
    target_concurrency=32,  # 한 컨테이너가 동시에 받을 요청 수. 부하 테스트로 조정한다
    # Modal 프록시 인증은 끄고, 대신 vLLM의 --api-key로 막는다(OpenAI client가 그대로 붙도록).
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--served-model-name", MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--max-model-len", str(MAX_MODEL_LEN),
            "--limit-mm-per-prompt", MM_LIMIT,
            "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
        ]
        # NOTE: API 키는 --api-key 인자로 넘기면 vLLM이 시작 로그(non-default args)에
        # 평문으로 찍는다. secret이 넣어준 VLLM_API_KEY 환경변수를 vLLM이 직접 읽게 둔다.
        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
