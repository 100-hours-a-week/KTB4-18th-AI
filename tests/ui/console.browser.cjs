const {chromium}=require('playwright');
const assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,...(process.env.BROWSER_EXECUTABLE ? {executablePath:process.env.BROWSER_EXECUTABLE} : {})});
 const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.route('**/v1/chat/messages',async route=>{
  const ev=(event,data)=>`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  const steps=[{stage:'accepted',elapsed_ms:0},{stage:'classification_start',elapsed_ms:10},{stage:'decision',intent:'lookup',search_space:'v4',elapsed_ms:800},{stage:'lookup_start',tool:'lookup_music',attempt:1,elapsed_ms:810},{stage:'tool_result',tool:'lookup_music',attempt:1,count:0,elapsed_ms:1100},{stage:'recovery_start',attempt:1,elapsed_ms:1101},{stage:'recovery_decision',action:'retry',value:'The Day Before',reason:'다른 제목 표기로 조회',elapsed_ms:1500},{stage:'lookup_start',tool:'lookup_music',attempt:2,elapsed_ms:1501},{stage:'tool_result',tool:'lookup_music',attempt:2,count:1,elapsed_ms:1800},{stage:'finish',reason:'results_available',elapsed_ms:1801},{stage:'answer_start',elapsed_ms:1802}];
  let body='';for(const data of steps){body+=ev('status',Object.fromEntries(Object.entries(data).filter(([k])=>['stage','elapsed_ms','attempt','count'].includes(k))));if(!['accepted','answer_start'].includes(data.stage))body+=ev('agent_step',data);}
  body+=ev('text',{delta:'말씀하신 곡을 '})+ev('text',{delta:'찾았어요.'})+ev('tracks',{tracks:[{title:'The Day Before',artist:'NELL',track_id:'1',artwork_url:null,preview_url:'https://example.invalid/preview',store_url:'https://music.apple.com/'}]})+ev('done',{});
  await route.fulfill({status:200,contentType:'text/event-stream',body});
 });
 await page.goto(process.env.CONSOLE_URL || 'http://127.0.0.1:8002/app/');
 await page.getByText('검색 v4',{exact:true}).waitFor();
 await page.screenshot({path:'/tmp/agent-lab-initial.png',fullPage:true});
 assert.equal(await page.locator('#liveBadge').innerText(),'대기');
 await page.locator('#sendBtn').click();
 assert.equal(await page.locator('#liveBadge').innerText(),'대기');
 await page.locator('[data-example]').nth(1).click();
 assert.match(await page.locator('#input').inputValue(),/Bfore/);
 await page.locator('#sendBtn').click();
 await page.locator('.run-panel.complete').waitFor();
 assert.match(await page.locator('.bot-answer').innerText(),/찾았어요/);
 assert.equal(await page.locator('.track-card').count(),1);
 assert.equal(await page.locator('#metricChunks').innerText(),'2');
 await page.screenshot({path:'/tmp/agent-lab-result.png',fullPage:true});
 await page.locator('.run-panel summary').click();
 assert.equal(await page.locator('.run-steps li').count(),11);
 assert.match(await page.locator('#runFacts').innerText(),/재시도 최대 1회/);
 await page.screenshot({path:'/tmp/agent-lab-overview.png',fullPage:true});
 await page.locator('[data-panel="events"]').click();
 assert.equal(await page.locator('#responseSteps .public-status').count(),9);
 await page.locator('#rawEvents').check();
 assert.ok(await page.locator('#responseSteps').evaluate(el=>el.classList.contains('show-raw')));
 await page.locator('[data-panel="payload"]').click();
 assert.match(await page.locator('#reqBox').innerText(),/thread_id/);
 await page.locator('#resetBtn').click();
 assert.equal(await page.locator('.track-card').count(),0);
 await page.setViewportSize({width:390,height:844});
 assert.equal(await page.locator('.inspane').isVisible(),false);
 await page.locator('#toggleInspector').click();
 assert.equal(await page.locator('.inspane').isVisible(),true);
 await page.keyboard.press('Escape');
 assert.equal(await page.locator('.inspane').isVisible(),false);
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 await page.screenshot({path:'/tmp/agent-lab-mobile.png',fullPage:true});
 // Keep a stream open: the UI must show a step before any answer or completion.
 await page.evaluate(()=>{
   const original=window.fetch;
   window.fetch=(url,options)=>String(url).endsWith('/v1/chat/messages') ? Promise.resolve(new Response(new ReadableStream({start(controller){
     window.testStream=controller;
     controller.enqueue(new TextEncoder().encode('event: status\ndata: {"stage":"classification_start","elapsed_ms":10}\n\n'));
   }}),{headers:{'Content-Type':'text/event-stream'}})) : original(url,options);
 });
 await page.locator('#input').fill('음악 추천해줘');
 await page.locator('#sendBtn').click();
 await page.locator('.run-title').filter({hasText:'요청 의도 분석 중'}).waitFor();
 assert.equal(await page.locator('#liveBadge').innerText(),'실행 중');
 assert.equal(await page.locator('.bot-answer').innerText(),'');
 assert.equal(await page.locator('#sendBtn').isDisabled(),true);
 await page.screenshot({path:'/tmp/agent-lab-running.png',fullPage:true});
 await page.evaluate(()=>{
   window.testStream.enqueue(new TextEncoder().encode('event: text\ndata: {"delta":"답변 첫 조각"}\n\n'));
 });
 await page.locator('.bot-answer').filter({hasText:'답변 첫 조각'}).waitFor();
 assert.equal(await page.locator('#liveBadge').innerText(),'실행 중');
 await page.evaluate(()=>{
   window.testStream.enqueue(new TextEncoder().encode('event: error\ndata: {"detail":"테스트용 연결 실패"}\n\n'));
   window.testStream.close();
 });
 await page.locator('.run-panel.failed').waitFor();
 assert.equal(await page.locator('#liveBadge').innerText(),'실패');
 assert.equal(await page.locator('#sendBtn').isEnabled(),true);
 for(const size of [{width:375,height:812},{width:844,height:390}]) {
   await page.setViewportSize(size);
   assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
 }
 await page.emulateMedia({reducedMotion:'reduce'});
 assert.equal(await page.locator('.run-indicator').evaluate(el=>getComputedStyle(el).animationName),'none');
 assert.deepEqual(errors,[]);
 console.log('PASS: layout, retry, open-stream progress, incremental text, errors, tabs, reset, mobile, reduced motion, no JS errors');
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
