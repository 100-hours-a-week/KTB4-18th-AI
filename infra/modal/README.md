# Modal 사진 분석 폴백 서버

사진 분석 주 경로(OpenRouter Gemini)가 실패했을 때 대신 쓰는 오픈웨이트 VLM 서버입니다. Modal GPU에서 vLLM OpenAI 호환 서버로 실행합니다.

## 구성

서버 정의는 `vlm_server.py` 하나입니다. 모델·GPU·버전은 `PROFILES` 항목으로 정하고, 배포할 때 `VLM_PROFILE`로 고릅니다.

| 프로필 | 모델 | GPU | vLLM | transformers |
| --- | --- | --- | --- | --- |
| `e4b-l40s` (기본) | `google/gemma-4-E4B-it` | L40S 48GB | 0.30.0 | 5.17.0 |
| `e4b-l4` | `google/gemma-4-E4B-it` | L4 24GB | 0.30.0 | 5.17.0 |

- 백엔드는 별칭 `photo-vlm`만 사용합니다. 프로필을 바꿔도 AI 서버 설정은 그대로입니다.
- Modal Environment `main`은 운영, `dev`는 실험입니다. `dev` URL에는 `-dev`가 붙습니다.

## 측정값

2026-10-18, `e4b-l40s`, 합성 사진 28장 기준입니다. 원본 측정 기록은 저장소에 포함하지 않았습니다.

| 항목 | 값 |
| --- | --- |
| 응답 시간 중앙값 | 3.15초 (최대 5.7초) |
| 정상 응답 | 28/28 |
| 배포 후 첫 기동 (스냅샷 생성 포함) | 약 7~9분 |
| 꺼진 뒤 깨우기: 스냅샷 복원 | 약 23초 + 첫 응답 약 7초 |
| 꺼진 뒤 깨우기: 일반 기동 | 약 4분 (측정 4번 중 3번) |

## 사전 준비 (1회)

1. Modal CLI 로그인: `uvx modal token new`
2. Modal 대시보드에서 Proxy Auth Token을 발급합니다. AI 서버의 `PHOTO_FALLBACK_API_KEY`에는 토큰 ID(`wk-…`)와 시크릿(`ws-…`)을 점(`.`)으로 이은 값을 넣습니다.

## 배포

```bash
# 실험
VLM_PROFILE=e4b-l40s uvx modal deploy --env dev infra/modal/vlm_server.py
# 운영
VLM_PROFILE=e4b-l40s uvx modal deploy --env main infra/modal/vlm_server.py
```

배포 직후 서버는 꺼져 있습니다. 꺼진 서버는 첫 요청에 503을 돌려주고 그때 기동을 시작합니다. 첫 기동은 스냅샷을 만들며 7~9분 걸리므로, 배포 뒤 아래 명령으로 **준비가 끝날 때까지** 확인합니다(최대 15분).

```bash
for i in $(seq 1 90); do
  curl -sf -H "Authorization: Bearer $PHOTO_FALLBACK_API_KEY" "$PHOTO_FALLBACK_URL/models" | grep -q photo-vlm \
    && echo "준비 완료" && break
  sleep 10
done
```

"준비 완료"가 나오지 않으면 Modal 대시보드의 앱 로그를 확인합니다. 장애 테스트에서는 AI 서버의 사진 API 응답 중 `scene` 이벤트의 `provider`가 `fallback`인지까지 확인합니다.

## AI 서버 환경변수

| 변수 | 값 |
| --- | --- |
| `PHOTO_FALLBACK_URL` | 배포 출력의 URL 뒤에 `/v1` |
| `PHOTO_FALLBACK_MODEL` | `photo-vlm` |
| `PHOTO_FALLBACK_API_KEY` | Proxy Auth Token |
| `PHOTO_VLM_API_KEY` | 선택. 사진 주 경로 전용 OpenRouter 키 |

세 개의 `PHOTO_FALLBACK_*` 중 하나라도 비어 있으면 폴백을 호출하지 않습니다.

## 예열과 유지

평소에는 주 경로가 실패할 때만 깨어나고, **요청 없이 20분이 지나면** 다시 꺼집니다. 시연이나 장애 테스트 전에는 미리 켜 두고, 위의 준비 확인 명령으로 준비 완료를 확인합니다.

```bash
# 켜 두기 (L40S 시간당 약 $1.95)
uvx --with modal python -c "import modal; modal.Server.from_name('photo-vlm-fallback', 'VlmServer', environment_name='main').update_autoscaler(min_containers=1)"
# 끝나면 되돌리기
uvx --with modal python -c "import modal; modal.Server.from_name('photo-vlm-fallback', 'VlmServer', environment_name='main').update_autoscaler(min_containers=0)"
```

같은 방식으로 `scaledown_window`(초)도 바꿀 수 있습니다. 실행 중 바꾼 값은 다음 배포 때 코드의 값으로 돌아갑니다.

## 모델·GPU 교체와 되돌리기

1. `PROFILES`에 항목을 추가합니다. 모델 revision을 함께 고정합니다.
2. `dev`에 배포하고 같은 평가 데이터로 회귀를 측정합니다.
3. 통과하면 `main`에 같은 프로필로 배포합니다. AI 서버 설정은 바꾸지 않습니다.

되돌릴 때는 이전 프로필로 다시 배포합니다. 폴백을 끄려면 AI 서버의 `PHOTO_FALLBACK_*` 중 하나를 비웁니다.

## 비용 확인

```bash
uvx modal environment billing summary main
uvx modal environment billing summary dev
uvx modal app stop --yes --env dev photo-vlm-fallback   # 실험이 끝나면
```

Starter 요금제에는 Environment별 예산이 없어 워크스페이스 한도(Usage limit, Spend limit)만 적용됩니다. 한도에 도달하면 폴백 서버만 멈추고 주 경로는 정상 동작합니다.

## 알려진 한계

- 서버가 꺼져 있으면 첫 폴백 요청은 즉시 실패합니다. Modal이 503을 돌려주고, 그 요청으로 서버가 켜집니다.
- GPU 메모리 스냅샷은 Modal 알파 기능입니다. 복원되지 않으면 일반 기동(약 4분)으로 동작합니다.
- 장애 시 사용자 사진이 미국(Modal us-east)으로 전송됩니다. 평소 주 경로도 해외 공급자를 사용합니다.
- 스냅샷 전후 처리를 위해 `VLLM_SERVER_DEV_MODE`로 vLLM 관리용 경로가 열립니다. 외부 접근은 Modal 프록시 인증으로 막습니다.
- 동시 요청 시의 지연은 측정하지 않았습니다. AI 서버가 사진 요청을 동시에 2건까지만 처리합니다.
