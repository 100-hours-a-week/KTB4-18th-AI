"""요청당 질문 한 번과 세션 내 추천 맥락을 관리하는 코드 정책."""

CONDITIONS = {
    'genre': ('recommend_genres',),
    'year': ('recommend_min_year', 'recommend_max_year')
    }
FILTERS = tuple(field for fields in CONDITIONS.values() for field in fields)
DEFAULT_QUESTION = '지금 어디서 뭐 하고 계세요? 창가 자리, 퇴근길처럼 알려주시면 어울리는 곡을 찾아드릴게요.'
DEFAULT_QUERY = '편하게 듣기 좋은 노래'


def prepare_dialogue(state, result):
    previous = state.get('dialogue_context') or {}
    result = {**result, 'dialogue_reply': None}
    if result['intent'] != 'recommend':
        return result
    current = dict(previous) if result.get('request_relation') == 'continue' else {}
    # 곡 선택을 맡기면 모델이 검색 문장을 비울 수 있고, 빈 문장은 임베딩에서 실패한다.
    if not (result.get('recommend_query') or '').strip():
        result['recommend_query'] = current.get('query') or DEFAULT_QUERY
    # 이어지는 응답에서 모델이 필수 조건을 생략해도 조용히 해제하지 않는다.
    # 연도는 범위 하나로 다룬다. 일부만 되살리면 2020년 이후 + 2009년까지처럼 불가능한 범위가 된다.
    released = result.get('released_filters') or []
    for name, fields in CONDITIONS.items():
        values = [result.get(field) for field in fields]
        old = [current.get(field) for field in fields]
        if name in released:
            # 해제를 표시하고 이전 값을 되풀이해도 사용자의 해제를 따른다. 새 값은 변경으로 본다.
            if all(value is None or value in old for value in values):
                result.update(dict.fromkeys(fields))
        elif all(value is None for value in values):
            result.update(zip(fields, old))
    invalid = result.get('requested_count') is not None and result['requested_count'] < 1
    supported = not result.get('recommend_unsupported_condition') and not invalid
    question = result.get('followup_question')
    if not (isinstance(question, str) and question.strip()):
        question = None if result.get('recommend_has_enough_info', True) else DEFAULT_QUESTION
    if supported and not current.get('asked') and question:
        result['dialogue_reply'] = question.strip()[:300]
        current.update(asked=True, pending_question=result['dialogue_reply'])
    elif supported and current.get('asked'):
        result['recommend_has_enough_info'] = True
        current['pending_question'] = None
    current.update({field: result.get(field) for field in FILTERS})
    current['query'] = result.get('recommend_query')
    result['dialogue_context'] = current
    return result


def delivered_dialogue(state):
    """곡을 전달하면 이 요청은 끝난다. 이후 피드백은 새 질문을 쓸 수 있다."""
    return {'dialogue_context': {**(state.get('dialogue_context') or {}), 'asked': False, 'pending_question': None}}


def no_results_dialogue(state):
    context = dict(state.get('dialogue_context') or {})
    if context.get('asked'):
        return {}
    # 필터가 없으면 벡터 검색은 곡을 돌려주므로 0곡의 주된 원인은 필수 조건이다. 느낌만 물으면 결과가 바뀌지 않는다.
    question = '지금 있는 곳이나 원하는 느낌을 알려주시거나, 바꾸고 싶은 조건을 말씀해 주세요.'
    context.update(asked=True, pending_question=question)
    return {'dialogue_reply': '말씀하신 조건에 맞는 곡을 찾지 못했어요. ' + question, 'dialogue_context': context}
