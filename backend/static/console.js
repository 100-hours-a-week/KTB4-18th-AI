
    "use strict";
    const CHAT_URL = "/v1/chat/messages";
    const STT_URL = "/v1/transcriptions";
    const PHOTO_URL = "/v1/chat/images";
    // NOTE: scene 이벤트의 provider로 사진 분석을 주 경로와 폴백 중 어디서 처리했는지 표시한다.
    const PHOTO_PATHS = { openrouter: ["main", "주 경로 · OpenRouter"], fallback: ["fallback", "폴백 · 자체 VLM(Modal)"] };
    // NOTE: 같은 Wi-Fi의 HTTP 테스트 주소에서도 안전한 난수로 UUID를 만든다.
    function newRequestId() {
      if (crypto.randomUUID) return crypto.randomUUID();
      const bytes = crypto.getRandomValues(new Uint8Array(16));
      bytes[6] = (bytes[6] & 0x0f) | 0x40;
      bytes[8] = (bytes[8] & 0x3f) | 0x80;
      const hex = Array.from(bytes, value => value.toString(16).padStart(2, "0")).join("");
      return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
    }
    const state = { source: "text", photoAttached: false, recording: false, busy: false, schemaOpen: true, callCount: 0, callLog: [], threadId: newRequestId() };
    const $ = (id) => document.getElementById(id);
    const scroll = $("msgScroll");
    let photoController = null;
    let photoRevision = 0;
    let photoBlob = null;
    let photoUrl = null;
    const photoUrls = new Set();

    function nowLabel() { const d = new Date(); const ap = d.getHours() < 12 ? "오전" : "오후"; return `${ap} ${d.getHours() % 12 || 12}:${String(d.getMinutes()).padStart(2, "0")}`; }
    function esc(value) { return String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;"); }
    function jsonHtml(value) { return esc(JSON.stringify(value, null, 2)).replace(/&quot;([^&]+?)&quot;(\s*:)/g, '<span class="k">&quot;$1&quot;</span>$2').replace(/:\s(&quot;.*?&quot;)/g, ': <span class="s">$1</span>').replace(/:\s(-?\d+(?:\.\d+)?)/g, ': <span class="n">$1</span>'); }
    function scrollToBottom() { scroll.scrollTop = scroll.scrollHeight; }
    function addDivider(text) { const el = document.createElement("div"); el.className = "divider"; el.innerHTML = `<span>${esc(text)}</span>`; scroll.append(el); }
    function addGreeting() { const el = document.createElement("div"); el.className = "msgrow left"; el.innerHTML = `<div class="bubble bot greeting"><span class="greeting-mark" aria-hidden="true"><span class="vinyl-disc"></span></span><div class="greeting-copy"><p>지금 어떤 순간에 머물고 있나요?</p><p>날씨나 장소, 느껴지는 기분을 편하게 알려주세요.<br>글로 적어 주시거나 음성으로 말해도 좋아요.<br>지금 딱 어울리는 음악을 골라드릴게요.</p></div><div class="greeting-save">로컬 실험의 대화와 추천은 이 화면에서 확인할 수 있어요.</div></div><div class="metaline"><span class="time">${nowLabel()}</span></div>`; scroll.append(el); }
    function addUserBubble(text, source) { const row = document.createElement("div"); row.className = "msgrow right"; const label = source === "voice" ? "음성 인식" : source === "photo" ? "사진" : "텍스트"; row.innerHTML = `<div class="bubble user pending">${esc(text)}</div><div class="metaline right"><span class="time">${nowLabel()}</span><span class="srctag ${source}">${label}</span></div>`; scroll.append(row); scrollToBottom(); return row; }
    function addBotResponse() { const row = document.createElement("div"); row.className = "msgrow left"; row.setAttribute('aria-busy','true'); row.innerHTML = `<div class="bot-result"><div class="bubble bot bot-answer"></div><div class="track-list"></div></div><div class="metaline"><span class="time">${nowLabel()}</span></div>`; scroll.append(row); scrollToBottom(); return row; }
    function addTrackCards(container, tracks) { container.innerHTML = tracks.map((track) => `<article class="track-card">${track.artwork_url ? `<img class="track-art" src="${esc(track.artwork_url)}" alt="${esc(track.title)} 앨범 표지" loading="lazy">` : '<div class="track-art empty" aria-hidden="true">♪</div>'}<div class="track-copy"><div class="track-title" title="${esc(track.title)}">${esc(track.title)}</div><div class="track-artist">${esc(track.artist)}</div>${track.store_url ? `<a class="source-link" href="${esc(track.store_url)}" target="_blank" rel="noopener">iTunes에서 보기</a>` : ""}</div>${track.preview_url ? `<button class="preview-btn" type="button" aria-label="${esc(track.title)} 미리듣기" aria-pressed="false" data-preview="${esc(track.preview_url)}"><img src="./assets/Playbutton.svg" alt=""></button>` : ""}</article>`).join(""); scrollToBottom(); }
    function setConnection(kind, label) { const cls = kind === "ok" ? "dot ok" : kind === "err" ? "dot err" : "dot"; $("connDot").className = cls; $("connDot2").className = cls; $("connText").textContent = label; }
    function setStatus(kind, code, text, latency) { const badge = $("statusBadge"); if (kind === "idle") { badge.className = "statusbadge"; badge.textContent = "대기 중"; $("latText").textContent = "—"; $("respMini").textContent = ""; return; } badge.className = kind === "ok" ? "statusbadge ok" : "statusbadge err"; badge.textContent = `${code} ${text}`; $("latText").textContent = latency == null ? "—" : `${latency}ms`; $("respMini").textContent = code; }
    function setReqBox(value) { inspector.setRequest(value); $("reqBox").innerHTML = jsonHtml(value); }
    function setResBox(value, error) { inspector.setResponse(value); const box = $("resBox"); box.className = error ? "jsonpre err" : "jsonpre"; box.innerHTML = jsonHtml(value); }
    function setPhotoPath(kind, label) { $("photoPath").innerHTML = kind ? `<span class="pathtag ${kind}">${esc(label)}</span>` : "—"; }
    function seconds(ms) { return `${(ms / 1000).toFixed(1)}초`; }
    fetch('/local/experiment').then(r => r.ok ? r.json() : null).then(config => {
      const box = $('experimentConfig'); box.replaceChildren();
      for (const label of config ? [config.workflow, `검색 ${config.search_space}`, `재시도 최대 ${Math.max(0,config.max_tool_stages-1)}회`, config.jev ? 'Jev 켜짐' : 'Jev 꺼짐'] : ['일반 실행 · 상세 설정 없음']) {
        const chip=document.createElement('span');chip.textContent=label;box.append(chip);
      }
    }).catch(() => {$('experimentConfig').textContent='실행 설정을 불러오지 못했습니다.';});
    function appendResponseStep(event, data) { inspector.record(event, data); }
    function pushLog(status, path = CHAT_URL, tag = null) { state.callCount += 1; $("callCount").textContent = `${state.callCount}회`; state.callLog.unshift({ status, path, tag, time: nowLabel() }); state.callLog = state.callLog.slice(0, 6); $("logBody").innerHTML = state.callLog.map((item) => `<div class="logrow"><span class="lm mono">POST</span><span class="lp mono">${esc(item.path)}${item.tag ? ` <span class="pathtag ${item.tag[0]}">${esc(item.tag[1])}</span>` : ""}</span><span class="lst ${item.status === 200 ? "ok" : "err"} mono">${item.status}</span><span class="lt mono">${item.time}</span></div>`).join(""); }
    function renderChips() { let html = ""; if (state.recording) html += '<span class="ctxchip rec"><span class="blink"></span>녹음 중 · 최대 60초 · 종료하면 전사 요청<span class="x" data-act="stopRec">×</span></span>'; else if (state.source === "voice") html += '<span class="ctxchip voice">전사문을 확인·수정한 뒤 전송하세요<span class="x" data-act="cancelVoice">×</span></span>'; if (state.photoAttached) html += '<span class="ctxchip photo">사진으로 추천받기 · 아래 방식 확인 <button type="button" class="x" data-act="removePhoto" aria-label="사진 제거">×</button></span>'; $("chips").innerHTML = html; }
    function refreshControls() {
      document.querySelectorAll("[data-example]").forEach(button=>button.disabled=state.busy || state.recording);
      for (const id of ["sendBtn", "input", "photoBtn", "resetBtn"]) $(id).disabled = state.busy || state.recording;
      $("input").disabled = state.busy || state.recording || state.photoAttached;
      $("capturePhoto").disabled = state.busy;
      $("recommendPhoto").disabled = state.busy || !state.photoAttached || !navigator.onLine;
      $("photoPreview").hidden = !state.photoAttached;
      $("photoPlaceholder").toggleAttribute("hidden", state.photoAttached);
      $("capturePhoto").textContent = state.photoAttached ? "다시 촬영" : "카메라로 촬영";
      $("capturePhoto").classList.toggle("secondary", state.photoAttached);
      $("photoTitle").textContent = state.photoAttached ? "촬영한 풍경을 확인해 주세요" : "눈앞의 풍경을 담아주세요";
      $("photoDescription").textContent = state.photoAttached ? "추천에 사용할 사진이 선명하게 담겼는지 확인해 주세요." : "이 순간에 어울리는 음악을 찾아드릴게요.";
      $("voiceBtn").disabled = state.busy || state.photoAttached;
      $("voiceBtn").innerHTML = state.recording ? '<span aria-hidden="true">■</span>' : '<img src="./assets/Microphone.svg" alt="">';
      $("voiceBtn").title = state.recording ? "녹음 종료 후 전사" : "마이크 녹음 시작";
      $("voiceBtn").setAttribute("aria-label", $("voiceBtn").title);
      $("voiceBtn").setAttribute("aria-pressed", String(state.recording)); $("voiceBtn").className = "iconbtn" + (state.recording ? " rec" : state.source === "voice" ? " on" : ""); $("photoBtn").className = "iconbtn ghost" + (state.photoAttached ? " on" : ""); }

    function showEndpoint(path) {
      inspector.begin(path);
      $("responseSteps").innerHTML = '<div class="empty">아직 수신한 이벤트가 없습니다.</div>';
      $("resBox").className = "jsonpre";
      $("resBox").innerHTML = '<span class="placeholder">응답 대기 중입니다.</span>';
      $("lastEndpoint").textContent = "POST " + path;
      $("endpointName").textContent = path;
      $("endpointPath").textContent = path;
      $("responseType").textContent = path === STT_URL ? "application/json" : "text/event-stream";
      setPhotoPath(null); $("photoStages").textContent = "—";
    }
    async function send() {
      if (state.busy || state.recording) return;
      if (!state.photoAttached && !$("input").value.trim()) return;
      if (!state.photoAttached && [...$("input").value.trim()].length > 200) {
        $("sttStatus").textContent = "전사문을 200자 이하로 수정한 뒤 전송하세요.";
        return;
      }
      state.busy = true; refreshControls();
      try { await sendChat(); } finally { state.busy = false; refreshControls(); }
    }
    async function sendChat() {
      const isPhoto = state.photoAttached;
      const controller = isPhoto ? new AbortController() : null;
      if (isPhoto) { photoController = controller; $("photoNotice").textContent = "사진을 확인하고 있어요…"; }
      const path = isPhoto ? PHOTO_URL : CHAT_URL;
      showEndpoint(path);
      const input = $("input");
      const raw = input.value.trim();
      if (!raw && !state.photoAttached) return;
      const userRow = addUserBubble(state.photoAttached ? "사진으로 음악 추천" : raw, state.photoAttached ? "photo" : state.source);
      if (state.photoAttached) {
        const preview = document.createElement("img");
        preview.src = photoUrl; preview.alt = "추천 요청 사진";
        preview.style.cssText = "display:block;max-width:200px;max-height:160px;border-radius:8px";
        userRow.querySelector(".bubble").append(preview);
      }
      if (isPhoto) userRow.remove();
      const body = { thread_id: state.threadId, request_id: newRequestId(), message: raw };
      let options = { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
      if (state.photoAttached) {
        const form = new FormData();
        form.append("thread_id", body.thread_id); form.append("request_id", body.request_id);
        form.append("image", photoBlob, "photo.jpg");
        options = { method: "POST", body: form };
        setReqBox({ thread_id: body.thread_id, request_id: body.request_id,
          image: { content_type: photoBlob.type, bytes: photoBlob.size }, encoding: "multipart/form-data" });
      } else { setReqBox(body); input.value = ""; $("inputCount").textContent="0 / 200"; }
      if (controller) options.signal = controller.signal;
      setConnection("", "음악 검색 중");
      const started = performance.now();
      let response;
      try {
        response = await fetch(path, options);
      } catch (error) {
        if (controller?.signal.aborted) { inspector.finish(true, performance.now()-started); return; }
        if (isPhoto) $("photoNotice").textContent = "사진을 확인하지 못했어요. 서버 연결을 확인하고 다시 시도해 주세요.";
        inspector.finish(true, Math.round(performance.now()-started)); setConnection("err", "연결 실패"); setStatus("err", "ERR", "No response", null); setResBox({ error: "서버에 연결하지 못했습니다. uvicorn 실행 상태를 확인하세요.", detail: String(error) }, true); userRow.querySelector(".bubble").classList.remove("pending"); const note = document.createElement("div"); note.className = "errnote"; note.textContent = "전송 실패 — 서버 미응답"; userRow.append(note); pushLog("ERR", path); return;
      }
      if (controller?.signal.aborted) { inspector.finish(true, performance.now()-started); return; }
      userRow.querySelector(".bubble").classList.remove("pending");
      if (!response.ok) {
        let data;
        try { data = await response.json(); } catch { data = { detail: "응답이 JSON 형식이 아닙니다." }; }
        if (isPhoto) $("photoNotice").textContent = `사진을 확인하지 못했어요 (${response.status}). 다시 시도해 주세요.`;
        const latency = Math.round(performance.now() - started);
        inspector.finish(true, latency); setConnection("err", "요청 실패"); setStatus("err", response.status, response.statusText || "Error", latency); setResBox(data, true); const note = document.createElement("div"); note.className = "errnote"; note.textContent = `요청 거부됨 (${response.status})`; userRow.append(note); pushLog(response.status, path); return;
      }

      const botRow = addBotResponse();
      if (isPhoto) botRow.remove();
      const progress = createProgress(botRow);
      const answerBox = botRow.querySelector(".bot-answer");
      const trackList = botRow.querySelector(".track-list");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let answer = "";
      let textChunks = 0;
      let firstTextMs = null;
      let tracks = [];
      let streamError = null;
      let completed = false;
      let scene = null;
      let rejected = false;
      let accepted = false;
      let sceneMs = null;
      let tracksMs = null;
      let pathTag = null;

      function consume(block) {
        if (controller?.signal.aborted) return;
        let eventName = "message";
        let dataText = "";
        for (const line of block.split("\n")) {
          if (line.startsWith("event:")) eventName = line.slice(6).trim();
          if (line.startsWith("data:")) dataText += line.slice(5).trim();
        }
        if (!dataText) return;
        const data = JSON.parse(dataText);
        const followProgress = scroll.scrollHeight - scroll.clientHeight - scroll.scrollTop < 120;
        appendResponseStep(eventName, data);
        progress.event(eventName, data);
        if (followProgress && (eventName === 'status' || eventName === 'agent_step')) scrollToBottom();
        if (eventName === "status") {
          const label = AGENT_STAGES[data.stage] ? stageLabel(data) : (data.stage === 'analyzing' ? '사진 확인 중' : '음악 검색 중');
          setConnection('', label);
        }
        if (eventName === "scene") {
          scene = data; sceneMs = Math.round(performance.now() - started);
          const [kind, label] = PHOTO_PATHS[data.provider] || ["main", String(data.provider)];
          setPhotoPath(kind, label); pathTag = [kind, kind === "fallback" ? "폴백" : "주 경로"];
        }
        // NOTE: 판정 전에는 채팅을 바꾸지 않고, 적합 판정에서만 기존 대화에 사진을 합류시킨다.
        if (eventName === "verdict" && isPhoto) {
          if (data.usable && !accepted) {
            accepted = true;
            scroll.append(userRow, botRow);
            $("photoScreen").hidden = true;
            setPhotoChatInert(false);
            removePhoto();
            scrollToBottom();
          } else if (!data.usable) { rejected = true; openPhotoSheet(); }
        }
        if (eventName === "rejected") { rejected = true; if (isPhoto && !$("photoScreen").hidden) openPhotoSheet(); answerBox.textContent = data.message; setConnection("err", "재촬영 안내"); }
        if (eventName === "done") completed = true;
        if (eventName === "text") {
          if (data.delta) {
            textChunks += 1;
            if (firstTextMs == null) firstTextMs = Math.round(performance.now() - started);
          }
          answer += data.delta || ""; answerBox.textContent = answer;

          scrollToBottom();
        }
        if (eventName === "tracks") { tracksMs = Math.round(performance.now() - started); tracks = data.tracks || []; addTrackCards(trackList, tracks); }
        if (eventName === "error") {
          streamError = data.detail || "응답 스트리밍에 실패했습니다.";
          // NOTE: 분석 결과 전에 실패하면 주 경로·폴백 중 어디서 실패했는지 화면에서는 알 수 없어 서버 사유만 표시한다.
          if (isPhoto && sceneMs == null) { setPhotoPath("fail", `분석 실패 · ${data.reason || "오류"}`); pathTag = ["fail", "분석 실패"]; }
        }
      }

      try {
        while (true) {
          const { value, done } = await reader.read();
          buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replaceAll("\r\n", "\n");
          const blocks = buffer.split("\n\n");
          buffer = blocks.pop();
          blocks.filter(Boolean).forEach(consume);
          if (done) break;
        }
        if (buffer.trim()) consume(buffer);
      } catch (error) {
        streamError = String(error);
      }

      if (controller?.signal.aborted) { progress.finish(true, performance.now()-started); inspector.finish(true, performance.now()-started); return; }
      const latency = Math.round(performance.now() - started);
      if (isPhoto) $("photoStages").textContent = [sceneMs != null && `장면 분석까지 ${seconds(sceneMs)}`, sceneMs != null && tracksMs != null && `곡 검색 ${seconds(tracksMs - sceneMs)}`, `총 ${seconds(latency)}`].filter(Boolean).join(" · ");
      if (isPhoto && completed && !accepted && !rejected && !streamError) streamError = "사진 판정 이벤트가 없습니다. 서버를 재시작해 주세요.";
      if (isPhoto) $("photoNotice").textContent = "";
      if (!completed && !streamError) streamError = "완료 이벤트 전에 연결이 종료됐습니다.";
      progress.finish(streamError, latency);
      inspector.finish(streamError, latency);
      botRow.setAttribute('aria-busy','false');
      if (streamError) { if (isPhoto && !accepted) $("photoNotice").textContent = "사진을 확인하지 못했어요. 다시 시도해 주세요."; else answerBox.textContent = answer + "\n응답을 완료하지 못했어요. 다시 요청해 주세요."; setConnection("err", "스트림 실패"); setStatus("err", "ERR", "Stream", latency); setResBox({ error: streamError, answer, tracks }, true); }
      else { setConnection(rejected ? "err" : "ok", rejected ? "재촬영 안내" : "연결됨"); setStatus("ok", response.status, rejected ? "재촬영" : "OK", latency); setResBox({ message: rejected ? answerBox.textContent : answer, tracks, ...(scene ? { analysis: scene } : {}) }, false); }
      if (path === PHOTO_URL && completed && !rejected && !streamError) removePhoto();
      pushLog(response.status, path, pathTag);
    }

    let playingAudio = null;
    let playingButton = null;
    function stopPreview() {
      playingAudio?.pause(); playingAudio = null;
      if(playingButton) { playingButton.innerHTML='<img src="./assets/Playbutton.svg" alt="">'; playingButton.setAttribute('aria-pressed','false'); playingButton.classList.remove('playing'); }
      playingButton = null;
    }
    scroll.addEventListener('click', event=>{
      const button=event.target.closest('[data-preview]'); if(!button) return;
      const wasPlaying=playingButton===button; stopPreview(); if(wasPlaying) return;
      const audio=new Audio(button.dataset.preview); playingAudio=audio; playingButton=button;
      button.innerHTML='<img src="./assets/Play-stop.svg" alt="">'; button.setAttribute('aria-pressed','true');button.classList.add('playing');
      audio.addEventListener('ended',()=>{if(playingAudio===audio) stopPreview();});
      audio.play().catch(()=>{if(playingAudio!==audio)return;stopPreview();button.setAttribute('aria-label','미리듣기 재생 실패 · 다시 시도');});
    });

    let recorder = null;
    let microphone = null;
    let recordingTimer = null;
    let leaving = false;
    function releaseMicrophone() {
      clearTimeout(recordingTimer);
      microphone?.getTracks().forEach(track => track.stop());
      microphone = null;
      recorder = null;
      state.recording = false;
      renderChips(); refreshControls();
    }
    async function transcribe(blob, filename) {
      if (state.busy || state.recording) return;
      if (!blob?.size || blob.size > 10 * 1024 * 1024) {
        $("sttStatus").textContent = "녹음 데이터가 없거나 10 MiB를 초과했습니다. 다시 녹음해 주세요.";
        return;
      }
      state.busy = true; refreshControls();
      showEndpoint(STT_URL);
      const form = new FormData();
      form.append("audio", blob, filename);
      setReqBox({ audio: { filename, content_type: blob.type, bytes: blob.size }, encoding: "multipart/form-data" });
      $("sttStatus").textContent = "전사 중…";
      setConnection("", "전사 중"); setStatus("idle");
      const started = performance.now();
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), 60000);
      try {
        const response = await fetch(STT_URL, { method: "POST", body: form, signal: controller.signal });
        let data;
        try { data = await response.json(); } catch { data = { message: "서버 응답이 JSON이 아닙니다." }; }
        const valid = response.ok && typeof data.transcript === "string" && data.transcript.trim();
        setStatus(valid ? "ok" : "err", response.status, valid ? "OK" : "Error", Math.round(performance.now() - started));
        setResBox(data, !valid); pushLog(response.status, STT_URL);
        if (!valid) {
          const reason = data.details?.reason;
          $("sttStatus").textContent = `${response.status}: ${data.message || "유효한 전사문이 없습니다."}${reason ? " (" + reason + ")" : ""}` +
            (response.status === 503 ? " 서버의 OPENROUTER_STT_API_KEY, 사용 한도 및 제공자 상태를 확인하세요." : "");
          setConnection("err", "전사 실패");
          return;
        }
        $("input").value = data.transcript; $("input").dispatchEvent(new Event("input"));
        state.source = "voice"; renderChips();
        $("sttStatus").textContent = [...data.transcript].length > 200
          ? "전사 완료 · 200자 이하로 수정한 뒤 전송하세요."
          : "전사 완료 · 입력창에서 확인·수정한 뒤 전송하세요.";
        setConnection("ok", "전사 완료");
      } catch (error) {
        const message = error.name === "AbortError" ? "전사 대기 시간이 초과됐습니다. 서버 처리는 계속될 수 있습니다." : "서버에 연결하지 못했습니다.";
        $("sttStatus").textContent = message;
        setConnection("err", "전사 실패"); setStatus("err", "ERR", "Network", Math.round(performance.now() - started));
        setResBox({ message }, true); pushLog("ERR", STT_URL);
      } finally {
        inspector.finish(!inspector.snapshot().response?.transcript, Math.round(performance.now()-started));
        clearTimeout(timer); state.busy = false; refreshControls(); $("input").focus();
      }
    }
    function stopRecording() {
      if (recorder?.state === "recording") recorder.stop();
    }
    async function startRecording() {
      if (state.busy || state.recording) return;
      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === "undefined") {
        $("sttStatus").textContent = "마이크 녹음은 HTTPS 또는 localhost의 지원 브라우저에서 사용하세요.";
        return;
      }
      state.busy = true; refreshControls();
      $("sttStatus").textContent = "마이크 권한 확인 중…";
      try {
        microphone = await navigator.mediaDevices.getUserMedia({ audio: true });
        if (leaving) { releaseMicrophone(); return; }
        const mime = ["audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus"].find(type => MediaRecorder.isTypeSupported(type));
        const active = new MediaRecorder(microphone, mime ? { mimeType: mime } : {});
        recorder = active;
        const chunks = [];
        let failed = false;
        active.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
        active.onerror = () => { failed = true; releaseMicrophone(); $("sttStatus").textContent = "녹음에 실패했습니다. 다시 시도하세요."; };
        active.onstop = () => {
          const type = active.mimeType || mime || "";
          const blob = new Blob(chunks, { type });
          releaseMicrophone();
          if (failed || leaving) return;
          const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
          transcribe(blob, "recording." + ext);
        };
        active.start(); state.recording = true;
        $("sttStatus").textContent = "녹음 중 · 마이크 버튼을 다시 누르면 종료합니다. 최대 60초.";
        recordingTimer = setTimeout(stopRecording, 60000);
      } catch (error) {
        releaseMicrophone();
        $("sttStatus").textContent = error.name === "NotAllowedError" ? "마이크 권한이 거부되었습니다. 브라우저 권한을 확인하세요." : "마이크를 사용할 수 없습니다.";
      } finally { state.busy = false; renderChips(); refreshControls(); }
    }
    window.addEventListener("pagehide", () => { leaving = true; photoRevision++; photoController?.abort(); stopRecording(); releaseMicrophone(); photoUrls.forEach(url => URL.revokeObjectURL(url)); photoUrls.clear(); });
    window.addEventListener("pageshow", () => { leaving = false; });
    $("sendBtn").addEventListener("click", () => { if (!state.recording) send(); });
    $("input").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.isComposing) { event.preventDefault(); if (!state.recording) send(); } });
    $("voiceBtn").addEventListener("click", () => { if (state.recording) stopRecording(); else startRecording(); });
    function removePhoto() {
      if (photoUrl && !scroll.querySelector(`img[src="${photoUrl}"]`)) {
        URL.revokeObjectURL(photoUrl); photoUrls.delete(photoUrl); photoUrl = null;
      }
      state.photoAttached = false; photoBlob = null; $("photoFile").value = "";
      $("photoPreview").removeAttribute("src"); renderChips(); refreshControls();
    }
    function setPhotoChatInert(value) {
      for (const element of document.querySelectorAll(".chathead, .msgscroll, .composer")) element.inert = value;
    }
    function closePhotoSheet() {
      $("photoSheet").hidden = true; $("photoScreen").inert = false; $("recommendPhoto").focus();
    }
    function openPhotoSheet() {
      $("photoNotice").textContent = "";
      $("photoSheet").hidden = false; $("photoScreen").inert = true; $("closePhotoSheet").focus();
    }
    function pickPhoto() { $("photoFile").value = ""; $("photoFile").click(); }
    $("photoBtn").addEventListener("click", () => {
      removePhoto(); $("photoNotice").textContent = "";
      $("photoScreen").hidden = false; $("photoScreen").scrollTop = 0; setPhotoChatInert(true); $("photoBack").focus();
    });
    $("photoBack").addEventListener("click", () => {
      photoRevision++; photoController?.abort(); closePhotoSheet(); removePhoto();
      $("photoScreen").hidden = true; setPhotoChatInert(false); $("photoBtn").focus();
      setConnection("", "대기 중");
    });
    $("capturePhoto").addEventListener("click", pickPhoto);
    $("recommendPhoto").addEventListener("click", send);
    $("closePhotoSheet").addEventListener("click", closePhotoSheet);
    $("retakePhoto").addEventListener("click", () => { closePhotoSheet(); pickPhoto(); });
    $("photoSheet").addEventListener("keydown", event => {
      if (event.key === "Escape") closePhotoSheet();
      if (event.key === "Tab") { event.preventDefault(); (document.activeElement === $("closePhotoSheet") ? $("retakePhoto") : $("closePhotoSheet")).focus(); }
    });
    window.addEventListener("online", refreshControls);
    window.addEventListener("offline", refreshControls);
    $("photoFile").addEventListener("change", async () => {
      const file = $("photoFile").files[0];
      if (!file) return;
      const revision = ++photoRevision;
      state.busy = true; refreshControls();
      $("photoNotice").textContent = "사진을 불러오고 있어요…";
      let bitmap;
      try {
        bitmap = await createImageBitmap(file);
        if (bitmap.width * bitmap.height > 60000000) throw new Error("사진은 6,000만 픽셀 이하여야 합니다.");
        const canvas = document.createElement("canvas");
        // NOTE: 고해상도 원본을 전송하지 않고 분석에 필요한 크기로 먼저 축소한다.
        const scale = Math.min(1, 1024 / Math.max(bitmap.width, bitmap.height));
        canvas.width = Math.max(1, Math.round(bitmap.width * scale));
        canvas.height = Math.max(1, Math.round(bitmap.height * scale));
        canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
        const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.9));
        if (!blob || blob.size > 5 * 1024 * 1024) throw new Error("JPEG 변환 후 5 MiB를 초과합니다.");
        if (revision !== photoRevision) return;
        if (photoUrl && !scroll.querySelector(`img[src="${photoUrl}"]`)) { URL.revokeObjectURL(photoUrl); photoUrls.delete(photoUrl); }
        photoBlob = blob; photoUrl = URL.createObjectURL(blob); photoUrls.add(photoUrl);
        $("photoPreview").src = photoUrl; state.photoAttached = true;
        $("photoNotice").textContent = "";
      } catch (error) { if (revision === photoRevision) $("photoNotice").textContent = "사진을 불러오지 못했어요. " + error.message; }
      finally { bitmap?.close(); state.busy = false; renderChips(); refreshControls(); }
    });
    $("chips").addEventListener("click", (event) => { if (state.busy) return; const action = event.target.closest("[data-act]")?.dataset.act; if (action === "cancelVoice") { state.source = "text"; $("input").placeholder = "지금 분위기는?"; } if (action === "removePhoto") removePhoto(); if (action === "stopRec") stopRecording(false); renderChips(); refreshControls(); });
    $("schemaHead").addEventListener("click", () => { state.schemaOpen = !state.schemaOpen; $("schemaBody").style.display = state.schemaOpen ? "" : "none"; $("schemaChev").className = "chev" + (state.schemaOpen ? " open" : ""); });
    function resetAll() {
      if (state.busy || state.recording) return;
      stopPreview();
      removePhoto(); photoUrls.forEach(url => URL.revokeObjectURL(url)); photoUrls.clear(); photoUrl = null;
      if (state.busy || state.recording) return;
      $("input").value = ""; $("inputCount").textContent="0 / 200";
      $("sttStatus").textContent = "";
      showEndpoint(CHAT_URL); inspector.reset(CHAT_URL); setConnection("", "대기 중");
      scroll.innerHTML = ""; addDivider(new Intl.DateTimeFormat("ko-KR", { year: "numeric", month: "long", day: "numeric", weekday: "long" }).format(new Date())); addGreeting(); scrollToBottom(); state.source = "text"; state.photoAttached = false; state.recording = false; state.threadId = newRequestId(); renderChips(); refreshControls(); $("input").placeholder = "지금 분위기는?"; $("reqBox").innerHTML = '<span class="placeholder">아직 보낸 요청이 없습니다.</span>'; $("resBox").className = "jsonpre"; $("resBox").innerHTML = '<span class="placeholder">메시지를 보내면 여기에 표시됩니다.</span>'; setStatus("idle"); }
    document.querySelectorAll('[data-example]').forEach(button=>button.addEventListener('click',()=>{
      if(state.busy || state.recording) return;
      $('input').value=button.dataset.example; $('input').dispatchEvent(new Event('input')); $('input').focus();
    }));
    $('input').addEventListener('input',()=>{$('inputCount').textContent=`${[...$('input').value].length} / 200`;});
    $("resetBtn").addEventListener("click", resetAll);
    resetAll();
  
