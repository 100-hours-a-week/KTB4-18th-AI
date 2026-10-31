# 로컬 실험 UI 디자인 스냅샷

참고 서비스: https://dev.meomuneum.com/chatbot
참고 소스: 로컬 meomuneum-frontend, 커밋 4975ec3. 배포본과 동일 커밋이라고 보장하지 않는다.

- `ChatbotPage.tsx`, `RecommendationCards.tsx`, `App.css`, semantic/typography tokens를 참고했다.
- SUIT Variable 글꼴과 챗봇 SVG 아이콘을 복사했다. 글꼴 라이선스는 assets/SUIT-LICENSE.txt에 보존했다.
- React 런타임과 서비스 인증·추천 기록 API를 가져오지 않고 기존 FastAPI 테스트 콘솔에 디자인을 이식했다.
  따라서 FE 저장소의 빌드나 의존성 없이 `bash run-agentic.sh` 하나로 실행된다.
- 스냅샷 이후의 FE 변경은 자동 반영되지 않는다. 갱신 시 말풍선·카드·입력창·사진 화면을 비교한다.
- 카드 미리듣기, 음성 전사, 사진 업로드는 기존 로컬 AI API를 사용한다.
  서비스의 추천 기록 저장·로그인은 이 화면에 연결하지 않았다.

## 파일 책임

- index.html: 서비스 스타일의 채팅 레이아웃 + 세 탭의 인스펙터.
- console.css: 기존 사진/음성 스타일 + 서비스 디자인 스냅샷 + 진행 패널·인스펙터 반응형 스타일.
- console.js: 기존 텍스트·사진·음성 API, SSE 수신, 카드·미리듣기.
- inspector.js: 실제 이벤트의 진행 요약·타이밍·중복 표시 병합·원문 복사. 모델 호출을 추가하지 않는다.

## 확인 기준

1. 요청 접수부터 답변 전까지 채팅 안의 진행 패널이 갱신된다. 완료하면 접히고 다시 펼칠 수 있다.
2. 요약에는 첫 이벤트·첫 텍스트·전체 시간·텍스트 조각 수, 검색 조건·시도 횟수·종료 이유가 보인다.
3. 이벤트 탭은 status/agent_step 중복을 접고 텍스트 조각을 합친다. 기록 복사에는 모든 원문 이벤트가 남는다.
4. 예시 버튼은 입력만 채우며 자동 호출하지 않는다. 새 대화는 화면과 인스펙터를 초기화하고 새 thread_id를 발급한다.
5. 모바일에서는 상단 인스펙터 버튼으로 패널을 열고 닫는다. Escape로 닫을 수 있다.

UI 자동 검증은 설치된 Chrome의 헤드리스 모드에서 합성 SSE 응답으로 수행했다.
재시도 표시, 텍스트 조각·카드 렌더링, 탭·필터·초기화, 모바일 너비와 JS 오류 유무를 검증했다.
합성 이벤트의 곡 발견 여부는 실제 모델·DB 품질을 의미하지 않는다.

추가로 응답을 열린 채 유지하는 브라우저 테스트에서 답변 전 단계 표시 → 첫 텍스트 즉시 반영 →
error 이벤트 후 실패 표시와 입력 재활성화를 확인했다. 375px·390px·가로 화면의 가로 넘침과
reduced-motion 설정도 확인했다. 실제 마이크·사진 모델·미리듣기 음원 재생은 이번 UI 검사에서 호출하지 않았다.

브라우저 테스트: `tests/ui/console.browser.cjs`. 실행 중인 로컬 서버와 Playwright가 필요하다.
Playwright가 다른 프로젝트에 설치되어 있으면 NODE_PATH로 해당 node_modules를 지정할 수 있다.
BROWSER_EXECUTABLE로 설치된 Chrome 경로, CONSOLE_URL로 테스트 주소를 지정한다.

```sh
NODE_PATH=/path/to/node_modules \
BROWSER_EXECUTABLE='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' \
node tests/ui/console.browser.cjs
```
