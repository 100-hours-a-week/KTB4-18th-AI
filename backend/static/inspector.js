/* Shared presentation for real SSE events. No simulated reasoning or model calls. */
const AGENT_STAGES = {
  accepted: '요청 접수', classification_start: '요청 의도 분석 중',
  decision: '요청 해석 완료', embedding_start: '검색 임베딩 생성 중',
  embedding: '임베딩 생성 완료', search_start: '음악 검색 중', lookup_start: '곡·아티스트 조회 중',
  tool_result: '검색 결과 확인', recovery_start: '다시 검색할 방법 확인 중',
  recovery_decision: '다른 표기로 재검색 결정', finish: '검색 단계 종료', answer_start: '답변 작성 중',
  analyzing: '사진 확인 중', searching: '음악 검색 중'
};
const FINISH_REASONS = {
  results_available: '검색 결과를 찾았습니다.', attempt_limit: '최초 실행과 재시도 1회를 마쳤습니다.',
  no_results_with_required_filters: '지정한 조건을 유지하고 검색을 종료했습니다.',
  clarify_or_static: '검색 없이 안내 또는 추가 질문을 드립니다.',
  duplicate_action_blocked: '같은 검색을 반복하지 않고 종료했습니다.',
  no_supported_recovery: '지원할 수 있는 다른 검색 방법을 찾지 못했습니다.',
  artist_change_blocked: '아티스트를 임의로 바꾸지 않고 종료했습니다.',
  recovery_model_failed: '재검색 판단을 완료하지 못해 종료했습니다.',
  execution_deadline: '다음 검색을 시작할 수 있는 시간을 초과했습니다.'
};
function stageLabel(data) {
  return `${AGENT_STAGES[data.stage] || data.stage}${data.attempt ? ` · ${data.attempt}차 시도` : ''}${data.count != null ? ` · ${data.count}곡` : ''}`;
}
function eventExplanation(data) {
  if (data.stage === 'decision') return `요청을 ${ {recommend:'분위기·상황 추천',lookup:'곡·아티스트 조회',guide:'사용법 안내',clarify:'추가 정보 확인',out_of_scope:'지원 범위 안내'}[data.intent] || data.intent || '음악 요청'}로 해석했습니다.`;
  if (data.stage === 'embedding') return data.query ? `검색 문장: ${data.query}` : '';
  if (data.stage === 'tool_result') return data.count ? `${data.count}곡을 찾았습니다.` : '이번 조회에서 결과를 찾지 못했습니다.';
  if (data.stage === 'recovery_start') return '같은 목적과 조건을 유지하며 다른 검색 방법을 확인합니다.';
  if (data.stage === 'recovery_decision') return `다시 조회할 값: ${data.value || ''}${data.reason ? ` · 모델의 결정 요약: ${data.reason}` : ''}`;
  if (data.stage === 'finish') return FINISH_REASONS[data.reason] || data.reason || '';
  return '';
}
function createProgress(row) {
  const panel = document.createElement('details'); panel.className = 'run-panel'; panel.open = true;
  panel.innerHTML = '<summary><span class="run-indicator" aria-hidden="true"></span><span class="run-title" role="status" aria-live="polite">요청 처리 중</span><span class="run-time">0.0초</span></summary><ol class="run-steps"></ol>';
  row.prepend(panel);
  const title = panel.querySelector('.run-title'), time = panel.querySelector('.run-time'), steps = panel.querySelector('ol');
  const started = performance.now(); let attempt = 0; const items = new Map();
  const timer = setInterval(() => { time.textContent = `${((performance.now()-started)/1000).toFixed(1)}초`; }, 100);
  return {
    event(name, data) {
      if (name !== 'status' && name !== 'agent_step') return;
      attempt = Math.max(attempt, data.attempt || 0);
      title.textContent = stageLabel(data);
      // A public status and its detailed event describe the same step.
      const key = `${data.stage}:${data.attempt || 0}:${data.elapsed_ms}`;
      let item = items.get(key);
      if (!item) { item = document.createElement('li'); items.set(key,item); steps.append(item); }
      item.replaceChildren();
      const label = document.createElement('strong'); label.textContent = stageLabel(data); item.append(label);
      const detail = eventExplanation(data);
      if (detail) { const p = document.createElement('p'); p.textContent = detail; item.append(p); }
    },
    finish(error, duration) {
      clearInterval(timer); panel.classList.add(error ? 'failed' : 'complete');
      title.textContent = error ? '처리를 완료하지 못했어요' : `처리 완료${attempt > 1 ? ' · 재시도 1회' : ''}`;
      time.textContent = `${(duration/1000).toFixed(1)}초`; panel.open = !!error;
    }
  };
}
const inspector = (() => {
  const get = id => document.getElementById(id);
  let trace, timer, textItem, stageRows;
  function reset(path) {
    clearInterval(timer); stageRows = new Map(); textItem = null;
    trace = { endpoint:path, started_at:new Date().toISOString(), events:[], request:null, response:null, metrics:{first_event_ms:null,first_text_ms:null,text_chunks:0,total_ms:null,tool_attempts:0} };
    trace.started = performance.now();
    get('responseSteps').replaceChildren(); get('eventCount').textContent='0';
    get('currentStage').textContent='요청 대기'; get('runFacts').textContent='단계가 실행되면 검색 조건과 결과가 표시됩니다.';
    for (const id of ['metricEvent','metricText','metricTotal']) get(id).textContent='—';
    get('metricChunks').textContent='0'; get('liveBadge').textContent='대기'; get('liveBadge').classList.remove('active');
    get('copyNotice').textContent='';
  }
  function row(name,data,received) {
    if (name === 'text' && textItem) {
      textItem.data.delta += data.delta || ''; textItem.count++;
      textItem.pre.textContent=JSON.stringify(textItem.data,null,2);
      textItem.heading.textContent=`답변 텍스트 · ${textItem.count}개 조각 · 수신 +${received}ms`;
      return;
    }
    const item=document.createElement('details'); item.className='sect';
    const heading=document.createElement('summary'); heading.className='secthead';
    const labels={text:'답변 텍스트',tracks:'추천곡',done:'완료',error:'오류',scene:'사진 분석',verdict:'사진 적합 판정',rejected:'재촬영 안내'};
    heading.textContent=`${data.stage ? stageLabel(data) : labels[name] || name} · 수신 +${received}ms${data.elapsed_ms != null ? ` / 서버 +${data.elapsed_ms}ms` : ''}`;
    const description=document.createElement('p'); description.className='event-summary'; description.textContent=eventExplanation(data);
    const pre=document.createElement('pre'); pre.className='jsonpre'; pre.textContent=JSON.stringify(data,null,2);
    item.append(heading); if(description.textContent) item.append(description); item.append(pre);
    if(name==='error') item.open=true;
    get('responseSteps').append(item);
    const key=`${data.stage}:${data.attempt||0}:${data.elapsed_ms}`;
    if(name==='status') stageRows.set(key,item);
    if(name==='agent_step') stageRows.get(key)?.classList.add('public-status');
    if(name==='text') textItem={data:{...data},count:1,heading,pre};
  }
  function record(name,data) {
    const received=Math.round(performance.now()-trace.started);
    // The record keeps every event; the visible timeline merges text and paired status details.
    trace.events.push({event:name,received_ms:received,data});
    const m=trace.metrics;
    if(m.first_event_ms==null) { m.first_event_ms=received; get('metricEvent').textContent=`${(received/1000).toFixed(2)}초`; }
    if(name==='text' && data.delta) { m.text_chunks++; if(m.first_text_ms==null) m.first_text_ms=received; get('metricText').textContent=`${(m.first_text_ms/1000).toFixed(2)}초`; get('metricChunks').textContent=m.text_chunks; }
    if(data.stage) { get('currentStage').textContent=stageLabel(data); m.tool_attempts=Math.max(m.tool_attempts,data.attempt||0); }
    row(name,data,received); get('eventCount').textContent=trace.events.length;
    if(name==='agent_step') {
      trace.facts = {...trace.facts};
      if(data.intent) trace.facts['의도']=data.intent;
      if(data.search_space) trace.facts['검색 공간']=data.search_space;
      if(data.query) trace.facts['검색 문장']=data.query;
      if(data.tool) trace.facts['실행 도구']=data.tool;
      if(data.attempt) trace.facts['검색 시도']=`${data.attempt} / 2 (재시도 최대 1회)`;
      if(data.count != null) trace.facts['결과']=`${data.count}곡`;
      if(data.stage==='tool_result' && data.tool==='search_music') trace.facts['필수 조건']=`장르 ${data.genres?.join(', ') || '미지정'} · 연도 ${data.min_year || '제한 없음'} ~ ${data.max_year || '제한 없음'}`;
      if(data.value) trace.facts['재검색 값']=data.value;
      if(data.reason && data.stage==='finish') trace.facts['종료 이유']=FINISH_REASONS[data.reason] || data.reason;
      get('runFacts').replaceChildren(...Object.entries(trace.facts).map(([label,value])=>{
        const div=document.createElement('div');div.className='fact';
        for(const text of [label,value]){const span=document.createElement('span');span.textContent=text;div.append(span);}return div;
      }));
    }
    if(name==='error') trace.error = data.detail || '처리 오류';
    if(name==='done' || name==='error') finish(name==='error',received);
  }
  function begin(path) { reset(path); get('liveBadge').textContent='실행 중'; get('liveBadge').classList.add('active'); timer=setInterval(()=>get('metricTotal').textContent=`${((performance.now()-trace.started)/1000).toFixed(1)}초`,100); }
  function finish(error,duration) { clearInterval(timer); trace.metrics.total_ms=duration; get('metricTotal').textContent=`${(duration/1000).toFixed(2)}초`; get('liveBadge').textContent=error?'실패':'완료'; get('liveBadge').classList.remove('active'); get('currentStage').textContent=error?'요청 실패':'처리 완료'; }
  function setPanel(panel) {
    document.querySelectorAll('[data-view]').forEach(el=>el.hidden=el.dataset.view!==panel);
    document.querySelectorAll('[data-panel]').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.panel===panel)));
  }
  document.querySelectorAll('[data-panel]').forEach(el=>el.addEventListener('click',()=>setPanel(el.dataset.panel)));
  get('rawEvents').addEventListener('change',e=>get('responseSteps').classList.toggle('show-raw',e.target.checked));
  get('copyTrace').addEventListener('click',async()=>{
    const {started,...exported}=trace;
    try { await navigator.clipboard.writeText(JSON.stringify(exported,null,2)); get('copyNotice').textContent='현재 요청의 실행 기록을 복사했습니다.'; }
    catch { get('copyNotice').textContent='복사 권한이 없습니다. 요청·응답 탭에서 원문을 선택해 복사하세요.'; }
  });
  const toggle=get('toggleInspector');
  function syncToggle() { const mobile=matchMedia('(max-width: 900px)').matches; toggle.setAttribute('aria-expanded',String(mobile?document.querySelector('.body').classList.contains('mobile-inspector'):!document.querySelector('.app').classList.contains('inspector-hidden'))); }
  toggle.addEventListener('click',()=>{ if(matchMedia('(max-width: 900px)').matches) document.querySelector('.body').classList.toggle('mobile-inspector');else document.querySelector('.app').classList.toggle('inspector-hidden');syncToggle(); });
  matchMedia('(max-width: 900px)').addEventListener('change',syncToggle);syncToggle();
  document.addEventListener('keydown',e=>{if(e.key==='Escape'){document.querySelector('.body').classList.remove('mobile-inspector');syncToggle();}});
  reset('/v1/chat/messages');
  return {begin,record,finish,reset,setPanel,setRequest(value){trace.request=value;},setResponse(value){trace.response=value;}, snapshot(){return trace;}};
})();
