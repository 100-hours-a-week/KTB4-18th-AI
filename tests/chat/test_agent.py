from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from backend.chat.graph import agent
from backend.chat.graph.workflow import _checkpointer, build_graph, record_shown_tracks


def row(title='The Day Before', artist='NELL'):
    return SimpleNamespace(track_id=1, title=title, artist=artist, artwork_url=None,
                           preview_url='https://example.test/p', store_url='https://example.test/s')


class Client:
    """분류를 대체한 테스트에서 추가 LLM 호출이 생기면 세어 실패시킨다."""

    def __init__(self):
        self.calls = 0
        self.responses = SimpleNamespace(create=self.create)

    def create(self, **kwargs):
        self.calls += 1
        raise AssertionError('unexpected LLM call')

    def with_options(self, **kwargs):
        return self


def intent(**fields):
    return {'intent': 'recommend', 'recommend_query': 'calm', 'recommend_has_enough_info': True,
            'recommend_unsupported_condition': False, 'recommend_genres': None, 'recommend_min_year': None,
            'recommend_max_year': None, 'lookup_song': None, 'lookup_song_alt': None, 'lookup_artist': None,
            'requested_count': None, **fields}


def run(monkeypatch, fields, message='m', thread=None, deadline_s=30):
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.setattr(agent, 'classify_node', lambda s, c: intent(**fields))
    client = Client()
    thread = thread or str(uuid4())
    config = {'configurable': {'thread_id': thread, 'client': client,
                               'deadline': agent.time.monotonic() + deadline_s}}
    return build_graph().invoke({'message': message, 'messages': [message]}, config), client, thread


@pytest.fixture
def lookup(monkeypatch):
    monkeypatch.setattr(agent, 'lookup_node', lambda s: {'lookup_status': 'not_found'})
    resolved = []
    def resolve(title, artist):
        resolved.append((title, artist))
        return row() if title == 'The Day Bfore' else None
    monkeypatch.setattr(agent, 'resolve_title_db', resolve)
    return resolved


def test_graph_exposes_real_steps(monkeypatch):
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    assert {'classify', 'embed', 'search', 'lookup', 'resolve_title', 'finish'} <= set(build_graph().get_graph().nodes)


def test_typo_is_resolved_without_any_recovery_llm_call(monkeypatch, lookup):
    result, client, _ = run(monkeypatch, {'intent': 'lookup', 'lookup_song': 'The Day Bfore', 'lookup_artist': 'NELL'})
    assert lookup == [('The Day Bfore', 'NELL')]
    assert client.calls == 0
    assert result['lookup_status'] == 'found'
    assert result['lookup_tracks'][0].title == 'The Day Before'
    assert result['matched_title'] == 'The Day Before'


def test_uncertain_title_stays_not_found(monkeypatch, lookup):
    result, client, _ = run(monkeypatch, {'intent': 'lookup', 'lookup_song': '없는 곡', 'lookup_artist': 'NELL'})
    assert result['lookup_status'] == 'not_found'
    assert result.get('matched_title') is None
    assert client.calls == 0


@pytest.mark.parametrize('fields', [{'lookup_song': 'x'}, {'lookup_artist': 'NELL'}])
def test_title_resolution_requires_title_and_artist(monkeypatch, lookup, fields):
    result, _, _ = run(monkeypatch, {'intent': 'lookup', **fields})
    assert lookup == []
    assert result['lookup_status'] == 'not_found'


def test_found_lookup_skips_title_resolution(monkeypatch, lookup):
    lookup_found(monkeypatch, 'Exact')
    run(monkeypatch, {'intent': 'lookup', 'lookup_song': 'Exact', 'lookup_artist': 'NELL'})
    assert lookup == []


def test_empty_recommendation_is_not_rewritten_or_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(agent, 'embed_node', lambda s, c: {'query_vector': [1]})
    monkeypatch.setattr(agent, 'search_node', lambda s, c: calls.append(s['recommend_query']) or {'tracks': []})
    result, client, _ = run(monkeypatch, {})
    assert calls == ['calm'] and client.calls == 0
    assert result['tracks'] == []


def test_matched_title_does_not_leak_into_next_turn(monkeypatch, lookup):
    fields = {'intent': 'lookup', 'lookup_song': 'The Day Bfore', 'lookup_artist': 'NELL'}
    _, _, thread = run(monkeypatch, fields)
    monkeypatch.setattr(agent, 'lookup_node', lambda s: {'lookup_status': 'found', 'lookup_tracks': []})
    try:
        result, _, _ = run(monkeypatch, {**fields, 'lookup_song': 'Exact'}, thread=thread)
        assert result['matched_title'] is None
    finally:
        _checkpointer.delete_thread(thread)


def test_history_is_not_duplicated(monkeypatch):
    monkeypatch.setattr(agent, 'embed_node', lambda s, c: {'query_vector': [1]})
    monkeypatch.setattr(agent, 'search_node', lambda s, c: {'tracks': [{'track_id': '1'}]})
    _, _, thread = run(monkeypatch, {}, 'first')
    try:
        result, _, _ = run(monkeypatch, {}, 'second', thread)
        assert result['messages'] == ['first', 'second']
    finally:
        _checkpointer.delete_thread(thread)


def test_recording_cards_schedules_no_followup_step(monkeypatch):
    monkeypatch.setattr(agent, 'embed_node', lambda s, c: {'query_vector': [1]})
    monkeypatch.setattr(agent, 'search_node', lambda s, c: {'tracks': []})
    _, _, thread = run(monkeypatch, {})
    try:
        record_shown_tracks(thread, [{'track_id': '1', 'title': 't', 'artist': 'a'}])
        snapshot = build_graph().get_state({'configurable': {'thread_id': thread}})
        assert snapshot.next == ()
        assert snapshot.values['shown_track_ids'] == ['1']
    finally:
        _checkpointer.delete_thread(thread)


def test_expired_turn_budget_stops_before_next_step(monkeypatch):
    searched = []
    monkeypatch.setattr(agent, 'embed_node', lambda s, c: {'query_vector': [1]})
    monkeypatch.setattr(agent, 'search_node', lambda s, c: searched.append(1) or {'tracks': []})
    with pytest.raises(HTTPException):
        run(monkeypatch, {}, deadline_s=0)
    assert searched == []


def test_events_follow_execution_order(monkeypatch, lookup):
    events = []
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.setattr(agent, 'classify_node', lambda s, c: intent(
        intent='lookup', lookup_song='The Day Bfore', lookup_artist='NELL'))
    config = {'configurable': {'thread_id': str(uuid4()), 'client': Client(), 'deadline': agent.time.monotonic() + 30,
                               'on_event': lambda e: events.append(e)}}
    build_graph().invoke({'message': 'm', 'messages': ['m']}, config)
    assert [e['stage'] for e in events] == ['classification_start', 'decision', 'lookup_start',
                                            'title_resolution_start', 'tool_result', 'finish']
    assert events[-1]['reason'] == 'title_resolved'


def test_live_steps_arrive_before_search_finishes(monkeypatch):
    import asyncio
    import threading
    from backend.chat import router
    from backend.chat.schemas import ChatRequest
    from backend.chat.service import ChatOutcome
    release = threading.Event()
    finished = threading.Event()
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.setenv('LOCAL_INSPECTOR', '1')

    def prepare(*args, on_event):
        on_event({'stage': 'search_start', 'attempt': 1})
        assert release.wait(3)
        finished.set()
        return ChatOutcome(kind='static', text='안내')

    monkeypatch.setattr(router, 'prepare_recommendation', prepare)
    response = router.chat(ChatRequest(thread_id=uuid4(), request_id=uuid4(), message='안내'))

    async def consume():
        iterator = response.body_iterator
        assert 'accepted' in await anext(iterator)
        assert not finished.is_set()
        try:
            step = await asyncio.wait_for(anext(iterator), 2)
            assert 'search_start' in step and 'event: status' in step
            assert not finished.is_set()
            detail = await anext(iterator)
            assert 'event: agent_step' in detail
        finally:
            release.set()
        output = ''.join([s async for s in iterator])
        assert 'answer_start' in output and 'event: text' in output and 'event: done' in output

    asyncio.run(consume())


def test_live_failure_is_sse_error_without_exception_details(monkeypatch):
    import asyncio
    from backend.chat import router
    from backend.chat.schemas import ChatRequest
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.delenv('LOCAL_INSPECTOR', raising=False)
    def fail(*a, **kw):
        raise RuntimeError('private-provider-details')
    monkeypatch.setattr(router, 'prepare_recommendation', fail)
    response = router.chat(ChatRequest(thread_id=uuid4(), request_id=uuid4(), message='안내'))
    async def consume():
        return ''.join([s async for s in response.body_iterator])
    output = asyncio.run(consume())
    assert 'event: error' in output
    assert 'private-provider-details' not in output and 'event: done' not in output


def test_live_details_require_local_inspector(monkeypatch):
    import asyncio
    from backend.chat import router
    from backend.chat.schemas import ChatRequest
    from backend.chat.service import ChatOutcome
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.delenv('LOCAL_INSPECTOR', raising=False)
    def prepare(*a, on_event):
        on_event({'stage': 'recovery_decision', 'value': 'private query'})
        return ChatOutcome(kind='static', text='안내')
    monkeypatch.setattr(router, 'prepare_recommendation', prepare)
    response = router.chat(ChatRequest(thread_id=uuid4(), request_id=uuid4(), message='안내'))
    async def consume():
        return ''.join([s async for s in response.body_iterator])
    output = asyncio.run(consume())
    # Spring 텍스트 파서는 모르는 이벤트를 502로 처리하므로 계약 밖 이벤트를 보내지 않는다.
    assert 'private query' not in output and 'agent_step' not in output
    events = [line.removeprefix('event: ') for line in output.splitlines() if line.startswith('event: ')]
    assert set(events) <= {'text', 'tracks', 'done', 'error'} and events[-1] == 'done'


def test_text_deltas_are_not_buffered_until_answer_completion(monkeypatch):
    import asyncio
    import threading
    from backend.chat import router
    from backend.chat.schemas import ChatRequest
    from backend.chat.service import ChatOutcome
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    release = threading.Event()
    monkeypatch.setattr(router, 'prepare_recommendation',
                        lambda *a, **kw: ChatOutcome(kind='recommend', tracks=[]))
    def stream(*a, **kw):
        yield router.sse('text', {'delta': '첫'})
        assert release.wait(3)
        yield router.sse('text', {'delta': ' 조각'})
        yield router.sse('done', {})
    monkeypatch.setattr(router, 'stream_answer', stream)
    response = router.chat(ChatRequest(thread_id=uuid4(), request_id=uuid4(), message='추천'))
    async def consume():
        iterator = response.body_iterator
        try:
            while True:
                chunk = await asyncio.wait_for(anext(iterator), 2)
                if 'event: text' in chunk:
                    assert '첫' in chunk and '조각' not in chunk
                    break
        finally:
            release.set()
        tail = ''.join([s async for s in iterator])
        assert '조각' in tail and 'event: done' in tail
    asyncio.run(consume())


def test_disconnect_stops_next_agent_stage(monkeypatch):
    import asyncio
    import threading
    from backend.chat import router
    from backend.chat.schemas import ChatRequest
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.setenv('LOCAL_INSPECTOR', '1')
    release = threading.Event()
    stopped = threading.Event()
    def prepare(*a, on_event):
        on_event({'stage': 'search_start'})
        assert release.wait(3)
        try:
            on_event({'stage': 'tool_result', 'count': 0})
        except agent.AgentCancelled:
            stopped.set()
            raise
        raise AssertionError('disconnected execution continued')
    monkeypatch.setattr(router, 'prepare_recommendation', prepare)
    response = router.chat(ChatRequest(thread_id=uuid4(), request_id=uuid4(), message='추천'))
    async def consume():
        iterator = response.body_iterator
        await anext(iterator)
        await asyncio.wait_for(anext(iterator), 2)
        await iterator.aclose()
        release.set()
        assert await asyncio.to_thread(stopped.wait, 2)
    asyncio.run(consume())


def test_disconnect_stops_silent_stream_without_inspector(monkeypatch):
    import asyncio
    import threading
    from backend.chat import router
    from backend.chat.schemas import ChatRequest
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.delenv('LOCAL_INSPECTOR', raising=False)
    started, release, stopped = threading.Event(), threading.Event(), threading.Event()
    def prepare(*a, on_event):
        started.set()
        assert release.wait(3)
        try:
            on_event({'stage': 'tool_result', 'count': 0})
        except agent.AgentCancelled:
            stopped.set()
            raise
    monkeypatch.setattr(router, 'prepare_recommendation', prepare)
    response = router.chat(ChatRequest(thread_id=uuid4(), request_id=uuid4(), message='추천'))
    async def consume():
        reading = asyncio.ensure_future(anext(response.body_iterator))
        assert await asyncio.to_thread(started.wait, 2)
        reading.cancel()
        await asyncio.gather(reading, return_exceptions=True)
        release.set()
        assert await asyncio.to_thread(stopped.wait, 2)
    asyncio.run(consume())


def test_live_failure_is_logged_on_the_server(monkeypatch, caplog):
    import asyncio
    from backend.chat import router
    from backend.chat.schemas import ChatRequest
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    def fail(*a, **kw):
        raise RuntimeError('boom')
    monkeypatch.setattr(router, 'prepare_recommendation', fail)
    response = router.chat(ChatRequest(thread_id=uuid4(), request_id=uuid4(), message='안내'))
    async def consume():
        return ''.join([s async for s in response.body_iterator])
    asyncio.run(consume())
    assert 'Live agent response failed' in caplog.text and 'boom' in caplog.text


def lookup_found(monkeypatch, *titles):
    from db.types import Track
    tracks = [Track(track_id='1', title=title, artist='A', artwork_url=None, preview_url='https://example.test/p',
                    store_url='https://example.test/s') for title in titles]
    monkeypatch.setattr(agent, 'lookup_node', lambda s: {'lookup_status': 'found', 'lookup_tracks': tracks})


def test_resolved_title_is_answered_by_code_without_answer_llm(monkeypatch):
    from backend.chat import service
    from backend.chat.graph import workflow
    from db.types import Track
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.setenv('OPENROUTER_LLM_API_KEY', 'test')
    monkeypatch.setenv('LLM_MODEL', 'test')
    track = Track(track_id='1', title='The Day Before', artist='NELL', artwork_url=None,
                  preview_url='https://example.test/p', store_url='https://example.test/s')
    graph = SimpleNamespace(invoke=lambda *a, **k: {
        'intent': 'lookup', 'lookup_status': 'found', 'lookup_tracks': [track], 'matched_title': 'The Day Before',
        'recommend_query': 'q', 'requested_count': None, 'message': 'm'})
    monkeypatch.setattr(workflow, 'build_graph', lambda: graph)
    outcome = service.prepare_recommendation('m', str(uuid4()))
    assert outcome.kind == 'fast' and outcome.tracks == [track]
    assert "'The Day Before'" in outcome.text


def test_partial_only_hit_is_not_sent_as_the_requested_song(monkeypatch, lookup):
    lookup_found(monkeypatch, 'Right On Time')
    result, _, _ = run(monkeypatch, {'intent': 'lookup', 'lookup_song': 'Right On', 'lookup_artist': 'Skrillex'})
    assert result['lookup_status'] == 'not_found'
    assert lookup == [('Right On', 'Skrillex')]


def test_exact_title_drops_partial_fillers(monkeypatch):
    lookup_found(monkeypatch, 'Butter', 'Butterflies')
    result, _, _ = run(monkeypatch, {'intent': 'lookup', 'lookup_song': 'Butter', 'lookup_artist': 'BTS'})
    assert [t.title for t in result['lookup_tracks']] == ['Butter']


@pytest.mark.parametrize('song,title', [('Hype Boy', 'Hype Boy (feat. X)'), ('스물셋', '스물셋 Twenty-Three'),
                                        ('Say It', 'Say It - Remastered'), ('love dive', 'LOVE DIVE')])
def test_same_title_with_suffix_stays_found(monkeypatch, lookup, song, title):
    lookup_found(monkeypatch, title)
    result, _, _ = run(monkeypatch, {'intent': 'lookup', 'lookup_song': song, 'lookup_artist': 'A'})
    assert result['lookup_status'] == 'found' and lookup == []


def test_llm_alias_hit_stays_found(monkeypatch, lookup):
    lookup_found(monkeypatch, 'Through the Night')
    result, _, _ = run(monkeypatch, {'intent': 'lookup', 'lookup_song': '밤편지', 'lookup_artist': 'IU',
                                     'lookup_song_alt': 'Through the Night'})
    assert result['lookup_status'] == 'found'


def test_finished_fixed_answer_survives_expired_deadline(monkeypatch):
    from backend.chat import service
    from backend.chat.graph import workflow
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.setenv('OPENROUTER_LLM_API_KEY', 'test')
    monkeypatch.setenv('LLM_MODEL', 'test')
    monkeypatch.setattr(agent, 'TURN_BUDGET_S', -1)
    graph = SimpleNamespace(invoke=lambda *a, **k: {'intent': 'guide', 'requested_count': None})
    monkeypatch.setattr(workflow, 'build_graph', lambda: graph)
    assert service.prepare_recommendation('m', str(uuid4())).kind == 'static'


def test_answer_llm_is_not_started_after_deadline(monkeypatch):
    from backend.chat import service
    from backend.chat.graph import workflow
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.setenv('OPENROUTER_LLM_API_KEY', 'test')
    monkeypatch.setenv('LLM_MODEL', 'test')
    monkeypatch.setattr(agent, 'TURN_BUDGET_S', -1)
    graph = SimpleNamespace(invoke=lambda *a, **k: {
        'intent': 'recommend', 'requested_count': None, 'recommend_unsupported_condition': False,
        'recommend_has_enough_info': True, 'tracks': [{'track_id': '1'}], 'recommend_query': 'q'})
    monkeypatch.setattr(workflow, 'build_graph', lambda: graph)
    outcome = service.prepare_recommendation('m', str(uuid4()))
    assert outcome.kind == 'fast' and outcome.tracks == [{'track_id': '1'}] and outcome.text


def recommend_turns(monkeypatch, *turns, tracks=({'track_id': '1'},), deliver=True):
    """턴마다 분류 결과를 넣어 실행하고 검색 조건과 마지막 결과를 돌려준다."""
    searched = []
    monkeypatch.setattr(agent, 'embed_node', lambda s, c: {'query_vector': [1]})
    def search(state, config):
        searched.append({k: state.get(k) for k in ('recommend_genres', 'recommend_min_year', 'recommend_max_year')})
        return {'tracks': list(tracks)}
    monkeypatch.setattr(agent, 'search_node', search)
    thread, result = str(uuid4()), None
    try:
        results = []
        for fields in turns:
            result, _, _ = run(monkeypatch, fields, thread=thread)
            results.append(result)
            if deliver and result.get('tracks') and not result.get('dialogue_reply'):
                record_shown_tracks(thread, [{'track_id': '1', 'title': 't', 'artist': 'a'}])
    finally:
        _checkpointer.delete_thread(thread)
    return results, searched


JAZZ_2010S = {'recommend_genres': ['Jazz'], 'recommend_min_year': 2010, 'recommend_max_year': 2019}


def test_unmentioned_conditions_survive_following_turns(monkeypatch):
    _, searched = recommend_turns(monkeypatch, JAZZ_2010S, {'request_relation': 'continue'},
                                  {'request_relation': 'continue'})
    assert searched[-1] == {'recommend_genres': ['Jazz'], 'recommend_min_year': 2010, 'recommend_max_year': 2019}


def test_released_year_is_not_restored(monkeypatch):
    _, searched = recommend_turns(monkeypatch, JAZZ_2010S, {'request_relation': 'continue', 'released_filters': ['year']})
    assert searched[-1] == {'recommend_genres': ['Jazz'], 'recommend_min_year': None, 'recommend_max_year': None}


def test_released_genre_is_not_restored(monkeypatch):
    _, searched = recommend_turns(monkeypatch, JAZZ_2010S, {'request_relation': 'continue', 'released_filters': ['genre']})
    assert searched[-1] == {'recommend_genres': None, 'recommend_min_year': 2010, 'recommend_max_year': 2019}


def test_changed_year_replaces_the_whole_range(monkeypatch):
    _, searched = recommend_turns(monkeypatch, JAZZ_2010S, {'request_relation': 'continue', 'recommend_min_year': 2020})
    assert searched[-1]['recommend_min_year'] == 2020 and searched[-1]['recommend_max_year'] is None


def test_answer_to_question_recommends_without_second_question(monkeypatch):
    results, searched = recommend_turns(
        monkeypatch, {'followup_question': '어떤 느낌이 좋으세요?'},
        {'request_relation': 'continue', 'followup_question': '또 물을까요?'})
    assert results[0]['dialogue_reply'] == '어떤 느낌이 좋으세요?'
    assert results[1]['dialogue_reply'] is None and len(searched) == 1


def test_feedback_after_delivered_tracks_can_ask_again(monkeypatch):
    results, searched = recommend_turns(
        monkeypatch, {'followup_question': '어떤 느낌이 좋으세요?'}, {'request_relation': 'continue'},
        {'request_relation': 'continue', 'followup_question': '어떤 점이 아쉬우셨어요?'})
    assert results[2]['dialogue_reply'] == '어떤 점이 아쉬우셨어요?' and len(searched) == 1


def test_missing_information_counts_as_the_one_question(monkeypatch):
    results, searched = recommend_turns(
        monkeypatch, {'recommend_has_enough_info': False},
        {'request_relation': 'continue', 'recommend_has_enough_info': False})
    assert results[0]['dialogue_reply'] and results[1]['dialogue_reply'] is None and len(searched) == 1


def test_empty_result_asks_once_then_stops(monkeypatch):
    results, _ = recommend_turns(monkeypatch, JAZZ_2010S, {'request_relation': 'continue'}, tracks=())
    assert '바꾸고 싶은 조건' in results[0]['dialogue_reply']
    assert results[1]['dialogue_reply'] is None


def test_new_request_drops_previous_conditions_and_question_count(monkeypatch):
    results, searched = recommend_turns(
        monkeypatch, {**JAZZ_2010S, 'followup_question': '어떤 느낌이 좋으세요?'},
        {'request_relation': 'new', 'followup_question': '새 요청 질문'})
    assert results[1]['dialogue_reply'] == '새 요청 질문' and searched == []


def test_lookup_turn_keeps_the_pending_context(monkeypatch):
    monkeypatch.setattr(agent, 'lookup_node', lambda s: {'lookup_status': 'found', 'lookup_tracks': []})
    results, _ = recommend_turns(
        monkeypatch, {**JAZZ_2010S, 'followup_question': '어떤 느낌이 좋으세요?'},
        {'intent': 'lookup', 'lookup_artist': 'IU'}, {'request_relation': 'continue'})
    assert results[2]['recommend_genres'] == ['Jazz'] and results[2]['dialogue_reply'] is None


def outcome(monkeypatch, fields):
    from backend.chat import service
    monkeypatch.setenv('OPENROUTER_LLM_API_KEY', 'test-key')
    monkeypatch.setenv('LLM_MODEL', 'test-model')
    classified = lambda state, config: intent(**fields)
    monkeypatch.setattr(agent, 'classify_node', classified)
    from backend.chat.graph import workflow
    monkeypatch.setattr(workflow, 'classify_node', classified)
    return service.prepare_recommendation('ㅁㄴㅇㄹ', str(uuid4()))


def test_out_of_scope_tells_how_to_ask_in_dialogue_path(monkeypatch):
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    assert '있는 곳' in outcome(monkeypatch, {'intent': 'out_of_scope'}).text


def test_default_path_keeps_out_of_scope_text(monkeypatch):
    from backend.chat import service
    monkeypatch.delenv('AGENTIC_CHAT', raising=False)
    assert outcome(monkeypatch, {'intent': 'out_of_scope'}).text == service.OUT_OF_SCOPE_TEXT


def test_dialogue_rules_stay_out_of_the_default_path(monkeypatch):
    from backend.recommendation import intent as module
    seen = []
    response = SimpleNamespace(output_text='{"intent": "guide", "requested_count": null}')
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: seen.append(kw) or response))
    monkeypatch.setattr(module, 'known_genres', lambda: [])
    monkeypatch.setenv('LLM_MODEL', 'test-model')
    module.classify(client, ['m'])
    module.classify(client, ['m'], dialogue={})
    plain, dialogue = [call['text']['format']['schema']['properties'] for call in seen]
    assert 'released_filters' not in plain and 'released_filters' in dialogue
    assert '장소·풍경' not in seen[0]['instructions'] and '장소·풍경' in seen[1]['instructions']


def test_blank_query_keeps_context_instead_of_failing_embedding(monkeypatch):
    results, searched = recommend_turns(monkeypatch, {'recommend_query': '  '}, {'request_relation': 'continue', 'recommend_query': '바다 노래'},
                                        {'request_relation': 'continue', 'recommend_query': ''})
    assert results[0]['recommend_query'].strip() and results[2]['recommend_query'] == '바다 노래'
    assert len(searched) == 3


def test_released_condition_ignores_a_repeated_old_value(monkeypatch):
    _, searched = recommend_turns(monkeypatch, JAZZ_2010S, {
        'request_relation': 'continue', 'released_filters': ['year'], 'recommend_min_year': 2010})
    assert searched[-1]['recommend_min_year'] is None and searched[-1]['recommend_max_year'] is None


def test_released_condition_keeps_a_new_value(monkeypatch):
    _, searched = recommend_turns(monkeypatch, JAZZ_2010S, {
        'request_relation': 'continue', 'released_filters': ['year'], 'recommend_min_year': 2020})
    assert searched[-1]['recommend_min_year'] == 2020


def test_question_count_survives_until_cards_are_sent(monkeypatch):
    results, _ = recommend_turns(
        monkeypatch, {'followup_question': '어떤 느낌이 좋으세요?'}, {'request_relation': 'continue'},
        {'request_relation': 'continue', 'followup_question': '또 물을까요?'}, deliver=False)
    assert results[2]['dialogue_reply'] is None


def test_same_thread_turns_run_one_at_a_time(monkeypatch):
    import threading
    from backend.chat import service
    from backend.chat.graph import workflow
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    monkeypatch.setenv('OPENROUTER_LLM_API_KEY', 'test')
    monkeypatch.setenv('LLM_MODEL', 'test')
    entered, release = threading.Event(), threading.Event()
    running, peak = [], []
    def invoke(*a, **k):
        running.append(1)
        peak.append(len(running))
        entered.set()
        assert release.wait(3)
        running.pop()
        return {'intent': 'guide', 'requested_count': None}
    monkeypatch.setattr(workflow, 'build_graph', lambda: SimpleNamespace(invoke=invoke))
    thread = str(uuid4())
    workers = [threading.Thread(target=service.prepare_recommendation, args=('m', thread)) for _ in range(2)]
    workers[0].start()
    assert entered.wait(2)
    workers[1].start()
    workers[1].join(0.2)
    release.set()
    for worker in workers:
        worker.join(3)
    assert peak == [1, 1]
