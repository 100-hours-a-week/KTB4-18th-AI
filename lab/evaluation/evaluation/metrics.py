"""사례 단위 지표. 해당 없음·판정 불가는 0점으로 발행하지 않는다."""
from collections import Counter
import math

# 턴별 코드 채점 키를 대시보드 열 단위로 묶는다. 키 이름은 실험 간 비교를 위해 바꾸지 않는다.
GROUPS = {
    'route_acc': ('routes',),
    'slot_acc': ('min_year', 'max_year', 'requested_count', 'has_non_music_request', 'genres_allowed', 'response_style_present'),
    'lookup_acc': ('lookup_track_ids_any', 'lookup_track_ids_if_found', 'lookup_artist', 'cards_title_contains'),
    'cards_ok': ('cards',),
    'exclusion_ok': ('previous_tracks_excluded',),
    'format_ok': ('plain_text',),
    'claim_ok': ('genre_claim',),
    'constraint_ok': ('returned_min_year', 'returned_max_year', 'returned_genres_allowed'),
    'photo_verdict_acc': ('usable', 'lighting_allowed', 'reject_reason_allowed'),
    'verdict_in_deadline': ('verdict_deadline_s',),
}
LIST_KEYS = ('unique_track_ratio', 'artist_diversity')
CONTRACT_KEYS = {n for key, names in GROUPS.items() if key != 'verdict_in_deadline' for n in names}
LATENCY_KEYS = ('first_text_ms', 'cards_ms', 'total_ms', 'verdict_ms')


def mean(values):
    return sum(values) / len(values) if values else None


def case_metrics(case, service_cost=None):
    """run_case 결과(+judge)를 고정 키 {key: score}로 줄인다. 값이 None인 키는 발행하지 않는다."""
    turns = case.get('turns', [])
    ok = bool(turns) and len(turns) == case['expected_turns'] and all(
        t['actual'].get('done') and not t['actual'].get('error') for t in turns)
    out = {'service_success': mean([t['scores']['service_success']['score'] for t in turns]) if turns else 0,
           'case_completion': int(ok)}
    scored = lambda names: [s['score'] for t in turns for n, s in t['scores'].items() if n in names and s['status'] == 'scored']
    for key, names in GROUPS.items():
        out[key] = mean(scored(names))
    contract = scored(CONTRACT_KEYS)
    incomplete = any(s['status'] in ('insufficient_evidence', 'execution_error')
                     for t in turns for n, s in t['scores'].items()
                     if n in CONTRACT_KEYS)
    out['contract_pass'] = (0 if not ok or any(v == 0 for v in contract) else
                            None if incomplete or not contract else 1)
    for key in LIST_KEYS:
        out[key] = mean(scored({key}))
    for key in LATENCY_KEYS:
        out[key] = mean([t['actual'][key] for t in turns if t['actual'].get(key) is not None])
    out['service_cost_usd'] = service_cost
    return {k: v for k, v in out.items() if v is not None}


def percentile(values, q):
    values = sorted(values)
    return values[math.ceil(len(values) * q) - 1] if values else None


def top_track_case_share(cases):
    """가장 자주 나온 곡이 몇 개 사례에 나왔는지 본다. 사례 안 중복은 제외한다."""
    with_cards = [c for c in cases if any(t['actual'].get('tracks') for t in c.get('turns', []))]
    counts = Counter(tid for c in with_cards for tid in {x['track_id'] for t in c['turns'] for x in t['actual'].get('tracks', [])})
    top = counts.most_common(1)
    return top[0][1] / len(with_cards) if top else None
