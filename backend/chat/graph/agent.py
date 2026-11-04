"""로컬 실험: 일반 모드를 실제 단계별 노드로 나눈 그래프.

오타 복구에 LLM을 호출하지 않는다. 표기 해석은 카탈로그 후보를 코드로 비교하고,
불확실하면 곡을 고르지 않고 끝낸다. 의도가 바뀌는 변경은 사용자 확인 없이 하지 않는다.
"""
import threading
import time
from contextlib import contextmanager

from langgraph.graph import END, START, StateGraph

from backend.chat.graph.nodes import classify_node, lookup_node
from backend.chat.graph.nodes import search_node as _search_node
from backend.chat.graph.state import RecommendationState
from backend.chat.graph.workflow import route_by_intent
from backend.core.errors import _model_unavailable
from backend.providers.openrouter import get_embedding_client
from backend.recommendation.adapter import search_space
from backend.recommendation.embedding import embed_query
from backend.recommendation.retrieval import db_time_limit, resolve_title_db
from db.search import same_title
from db.types import Track
from backend.chat.graph.dialogue import no_results_dialogue, prepare_dialogue

# 답변 스트림 시작 전까지의 예산이다. 시작된 답변은 끊지 않고 조각 간 대기 시간만 제한한다.
TURN_BUDGET_S = 30
_turn_locks = {}


class AgentCancelled(Exception):
    """연결 종료 후 다음 작업을 시작하지 않도록 중단한다."""


@contextmanager
def exclusive_turn(thread_id):
    """같은 대화의 턴을 순서대로 처리한다. 겹치면 체크포인트가 갈라져 조건이 사라진다."""
    # ponytail: 프로세스 안에서만 유효하고 InMemorySaver처럼 대화 수만큼 늘어난다. 다중 워커는 공유 체크포인터가 필요하다.
    lock = _turn_locks.setdefault(thread_id, threading.Lock())
    if not lock.acquire(timeout=TURN_BUDGET_S):
        raise _model_unavailable()
    try:
        yield
    finally:
        lock.release()


def record(config, stage, **details):
    callback = config.get('configurable', {}).get('on_event')
    if callback:
        callback({'stage': stage, **details})


def remaining(config) -> float:
    """턴에 남은 시간을 돌려준다. 다 썼으면 다음 호출을 시작하지 않는다."""
    left = config['configurable']['deadline'] - time.monotonic()
    if left <= 0:
        raise _model_unavailable()
    return left


def deadline(config) -> float:
    remaining(config)
    return config['configurable']['deadline']


def classify_step(state, config):
    record(config, 'classification_start')
    configurable = {**config['configurable'], 'dialogue_enabled': True}
    client = configurable['client'].with_options(timeout=remaining(config))
    # 첫 호출의 장르 목록 조회도 같은 턴 시간 안에서 끝나야 한다.
    with db_time_limit(deadline(config)):
        result = classify_node(state, {**config, 'configurable': {**configurable, 'client': client}})
    # 실험은 한 워크플로우만 사용하며 기존 세션의 고속 상태도 이어받지 않는다.
    if result['intent'] == 'fast_mode':
        result['intent'] = 'guide'
    result |= {'fast_mode': False, 'matched_title': None}
    result = prepare_dialogue(state, result)
    if result.get('dialogue_reply'):
        record(config, 'clarification', question=result['dialogue_reply'])
    record(config, 'decision', intent=result['intent'], action=str(route_by_intent({**state, **result})),
           search_space=search_space())
    return result


def embed_node(state, config):
    client = get_embedding_client().with_options(timeout=remaining(config))
    return {'query_vector': embed_query(client, state['recommend_query'])}


def embed_step(state, config):
    record(config, 'embedding_start', attempt=1)
    result = embed_node(state, config)
    record(config, 'embedding', attempt=1, query=state['recommend_query'])
    return result


def search_node(state, config):
    with db_time_limit(deadline(config)):
        return _search_node(state)


def search_step(state, config):
    record(config, 'search_start', attempt=1, tool='search_music')
    return search_node(state, config)


def lookup_step(state, config):
    record(config, 'lookup_start', attempt=1, tool='lookup_music')
    with db_time_limit(deadline(config)):
        result = lookup_node(state)
    song, artist = state.get('lookup_song'), state.get('lookup_artist')
    if result['lookup_status'] == 'found' and song and artist:
        # 제목은 부분 일치라 요청과 다른 곡이 섞인다. 같은 제목만 남기고, 없으면 찾은 것으로 보지 않는다.
        names = [name for name in (song, state.get('lookup_song_alt')) if name]
        exact = [track for track in result['lookup_tracks'] if any(same_title(n, track.title) for n in names)]
        return {'lookup_status': 'found', 'lookup_tracks': exact} if exact else {'lookup_status': 'not_found'}
    return result


def route_lookup(state):
    """제목과 가수가 모두 있을 때만 오타를 해석한다. 가수가 없으면 후보를 확정할 근거가 없다."""
    if state['lookup_status'] == 'not_found' and state.get('lookup_song') and state.get('lookup_artist'):
        return 'resolve_title'
    return 'finish'


def resolve_title_step(state, config):
    record(config, 'title_resolution_start')
    with db_time_limit(deadline(config)):
        row = resolve_title_db(state['lookup_song'], state['lookup_artist'])
    if row is None:
        return {'matched_title': None}
    track = Track(track_id=str(row.track_id), title=row.title, artist=row.artist, artwork_url=row.artwork_url,
                  preview_url=row.preview_url, store_url=row.store_url)
    return {'lookup_status': 'found', 'lookup_tracks': [track], 'matched_title': row.title}


def finish_step(state, config):
    update = {}
    route = route_by_intent(state)
    if state.get('dialogue_reply'):
        record(config, 'finish', reason='awaiting_context')
        return {}
    if route == 'embed':
        tracks = state['tracks']
        if not tracks:
            update = no_results_dialogue(state)
            if update:
                record(config, 'clarification', question=update['dialogue_reply'])
        filtered = any(state.get(k) is not None for k in
                       ('recommend_genres', 'recommend_min_year', 'recommend_max_year'))
        reason = ('results_available' if tracks else
                  'no_results_with_required_filters' if filtered else 'no_results')
        record(config, 'tool_result', tool='search_music', attempt=1, count=len(tracks),
               genres=state.get('recommend_genres'), min_year=state.get('recommend_min_year'),
               max_year=state.get('recommend_max_year'))
    elif route == 'lookup':
        tracks = state.get('lookup_tracks', []) if state['lookup_status'] == 'found' else []
        reason = ('title_resolved' if state.get('matched_title') else
                  'results_available' if tracks else 'no_results')
        record(config, 'tool_result', tool='lookup_music', attempt=1, count=len(tracks))
    else:
        reason = 'clarify_or_static'
    record(config, 'finish', reason=reason)
    return {'matched_title': state.get('matched_title'), **update}


def build_agent_graph(checkpointer):
    graph = StateGraph(RecommendationState)
    graph.add_node('classify', classify_step)
    graph.add_node('embed', embed_step)
    graph.add_node('search', search_step)
    graph.add_node('lookup', lookup_step)
    graph.add_node('resolve_title', resolve_title_step)
    graph.add_node('finish', finish_step)
    graph.add_edge(START, 'classify')
    graph.add_conditional_edges('classify', lambda state: END if state.get('dialogue_reply') else route_by_intent(state), {'embed': 'embed', 'lookup': 'lookup', END: 'finish'})
    graph.add_edge('embed', 'search')
    graph.add_edge('search', 'finish')
    graph.add_conditional_edges('lookup', route_lookup, {'resolve_title': 'resolve_title', 'finish': 'finish'})
    graph.add_edge('resolve_title', 'finish')
    graph.add_edge('finish', END)
    return graph.compile(checkpointer=checkpointer)
