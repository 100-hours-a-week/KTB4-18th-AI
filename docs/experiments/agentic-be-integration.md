# BE 연동 수정안

2026-11-05 검토. BE `dev`의 `AiImageRecommendationProvider.java`와
`AiRecommendationProvider.java` 기준. 기존 BE는 합의된 이벤트를 엄격히 검증하고 있다.
AI가 새 이벤트를 추가한 데 따른 계약 확장이며 기존 파서 자체의 잘못은 아니다.
아래는 BE 담당자가 적용할 수정안이며 BE 저장소에 적용한 변경은 아니다.

## 필수: 사진 진행 단계 허용

파일: `src/main/java/com/muse/meomuneum/recommendation/provider/AiImageRecommendationProvider.java`
메서드: `dispatch`, `status` 처리 내부. 기존 `analyzing`, `searching` 분기 다음,
`throw invalidResponse()` 바로 앞에 다음 분기를 추가한다. `List`는 이미 import되어 있다.

```java
if (Boolean.TRUE.equals(progress.verdict)
        && progress.sceneSeen && progress.textSeen && progress.searchingSeen
        && !progress.tracksSeen
        && List.of("embedding_start", "embedding", "search_start", "tool_result").contains(stage)) {
    return;
}
```

이 이벤트는 진단용으로 소비하며 FE로 전달할 필요는 없다. 추천 텍스트·곡 카드·완료 이벤트는
기존 listener로 계속 전달한다. 상위 `progress.done` 검증은 유지한다. 알 수 없는 단계는 계속 거부한다.

정상 순서:
`analyzing → verdict(true) → scene → text → searching → embedding_start → embedding → search_start → tool_result → tracks → done`.
`tool_result`의 `count`는 진단 정보이며 결과 확정은 기존 `tracks`에서만 수행한다.

필수 회귀 테스트: 위 전체 순서 성공, 0곡 성공, 검색 중 error 실패,
verdict 이전/검색 시작 이전/tracks 이후의 새 단계 거부, 알 수 없는 단계 거부.
현재 AI 테스트는 BE 파서까지 실행하지 않아 Python 테스트 통과만으로 이 계약을 검증할 수 없다.

## 텍스트: 배포 시 인스펙터 끄기

현재 AI는 `LOCAL_INSPECTOR=1`에서만 `status`와 `agent_step`을 보낸다.
서버는 `AGENTIC_CHAT=1`, `LOCAL_INSPECTOR=0`으로 설정하면 기존 텍스트
`text → tracks → done` 계약을 유지하므로 BE 텍스트 파서 수정은 필수가 아니다.

진행 표시를 메인 서비스에서도 제공하려면 별도 계약 변경이 필요하다.
`AiRecommendationProvider.dispatchEvent`에서 `status`/`agent_step`을 검증해 소비하고,
실제 표시할 경우 listener → RecommendationService → RecommendationEventStream →
FE `src/api/recommendations.ts` handlers → ChatbotPage 상태/표시까지 연결한다.
상세 검색어·판단 자료인 `agent_step`을 그대로 FE에 중계하지 않는다.
현재는 이 전체 진행 표시 확장을 구현하지 않았다.

## 배포 순서

1. BE 사진 단계 허용과 회귀 테스트 적용.
2. AI 실험 브랜치의 테스트 및 BE 경유 텍스트·사진 통합 검증.
3. AI 배포 환경의 `AGENTIC_CHAT=1`, `LOCAL_INSPECTOR=0` 설정 후 배포.

플래그를 설정하지 않으면 새 텍스트 그래프가 활성화되지 않는다.
사진 경로 변경은 AGENTIC_CHAT과 무관하므로 BE 사진 수정은 먼저 필요하다.
