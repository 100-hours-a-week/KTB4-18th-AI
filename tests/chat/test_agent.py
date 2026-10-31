from types import SimpleNamespace
from backend.chat.graph import agent


def setup(monkeypatch, tracks, filters=None):
    monkeypatch.setenv('LLM_MODEL', 'test')
    monkeypatch.setattr(agent, 'search_space', lambda: 'v4')
    monkeypatch.setattr(agent, 'classify_node', lambda s,c: dict(intent='recommend', recommend_query='calm',
        recommend_has_enough_info=True, recommend_unsupported_condition=False, recommend_genres=filters))
    monkeypatch.setattr(agent, 'embed_node', lambda s: {'query_vector':[1]})
    calls=[]
    def search(s):
        calls.append(s['recommend_query'])
        return {'tracks':tracks}
    monkeypatch.setattr(agent, 'search_node', search)
    client=SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        output_text='{"action":"retry","value":"quiet","reason":"alternate wording"}')))
    client.with_options=lambda **kw: client
    return calls, {'configurable':{'client':client}}


def test_success_does_not_repeat(monkeypatch):
    calls,config=setup(monkeypatch,[{'track_id':'1'}])
    result=agent.run_agent({'message':'calm','messages':['calm'],'shown_track_ids':['2']},config)
    assert calls==['calm']
    assert 'messages' not in result and 'shown_track_ids' not in result
    assert result['agent_events'][-1]['reason']=='results_available'


def test_empty_recovery_is_bounded(monkeypatch):
    calls,config=setup(monkeypatch,[])
    result=agent.run_agent({'message':'calm','messages':['calm']},config)
    assert calls==['calm','quiet']
    assert result['agent_events'][-1]['reason']=='attempt_limit'


def test_required_filters_are_not_relaxed(monkeypatch):
    calls,config=setup(monkeypatch,[],['Jazz'])
    result=agent.run_agent({'message':'Jazz','messages':['Jazz']},config)
    assert calls==['calm'] and result['recommend_genres']==['Jazz']
    assert result['agent_events'][-1]['reason']=='no_results_with_required_filters'


def test_repeated_action_is_blocked(monkeypatch):
    calls,config=setup(monkeypatch,[])
    config['configurable']['client'].responses.create=lambda **kw: SimpleNamespace(
        output_text='{"action":"retry","value":"calm","reason":"same"}')
    result=agent.run_agent({'message':'calm','messages':['calm']},config)
    assert calls==['calm']
    assert result['agent_events'][-1]['reason']=='duplicate_action_blocked'


def test_invalid_recovery_output_ends_cleanly(monkeypatch):
    calls,config=setup(monkeypatch,[])
    config['configurable']['client'].responses.create=lambda **kw: SimpleNamespace(output_text='invalid')
    result=agent.run_agent({'message':'calm','messages':['calm']},config)
    assert calls==['calm']
    assert result['agent_events'][-1]['reason']=='recovery_model_failed'


def test_graph_does_not_duplicate_history(monkeypatch):
    from uuid import uuid4
    from backend.chat.graph.workflow import build_graph, _checkpointer
    monkeypatch.setenv('AGENTIC_CHAT','1')
    _,config=setup(monkeypatch,[{'track_id':'1'}])
    thread=str(uuid4()); config['configurable']['thread_id']=thread
    try:
        graph=build_graph()
        graph.invoke({'message':'first','messages':['first']},config)
        result=graph.invoke({'message':'second','messages':['second']},config)
        assert result['messages']==['first','second']
        assert result['agent_events'][-1]['reason']=='results_available'
    finally:
        _checkpointer.delete_thread(thread)


def test_live_steps_arrive_before_search_finishes(monkeypatch):
    import asyncio
    import threading
    from uuid import uuid4
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
    from uuid import uuid4
    from backend.chat import router
    from backend.chat.schemas import ChatRequest
    monkeypatch.setenv('AGENTIC_CHAT', '1')
    def fail(*a, **kw):
        raise RuntimeError('private-provider-details')
    monkeypatch.setattr(router, 'prepare_recommendation', fail)
    response = router.chat(ChatRequest(thread_id=uuid4(), request_id=uuid4(), message='안내'))
    async def consume():
        return ''.join([s async for s in response.body_iterator])
    output = asyncio.run(consume())
    assert 'accepted' in output and 'event: error' in output
    assert 'private-provider-details' not in output and 'event: done' not in output


def test_live_details_require_local_inspector(monkeypatch):
    import asyncio
    from uuid import uuid4
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
    assert 'recovery_decision' in output
    assert 'private query' not in output and 'agent_step' not in output


def test_agent_emits_start_before_tool_and_retry_in_order(monkeypatch):
    calls, config = setup(monkeypatch, [])
    events = []
    config['configurable']['on_event'] = lambda e: events.append((e['stage'], list(calls)))
    agent.run_agent({'message': 'calm'}, config)
    assert events[0] == ('classification_start', [])
    starts = [calls for stage, calls in events if stage == 'search_start']
    assert starts == [[], ['calm']]
    stages = [stage for stage, _ in events]
    assert stages.index('recovery_start') < stages.index('recovery_decision')
    assert stages[-1] == 'finish'


def test_text_deltas_are_not_buffered_until_answer_completion(monkeypatch):
    import asyncio
    import threading
    from uuid import uuid4
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
    from uuid import uuid4
    from backend.chat import router
    from backend.chat.schemas import ChatRequest
    monkeypatch.setenv('AGENTIC_CHAT', '1')
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
