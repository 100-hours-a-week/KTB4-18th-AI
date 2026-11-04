"""새 도전 사례를 검사하고 validation 도전 층과 test로 나눈다.

test 내용이 보이지 않게 사례 ID와 건수만 출력한다.
사용: python prepare_challenge.py check <cases.jsonl> <catalog.csv>
     python prepare_challenge.py build <cases.jsonl> <catalog.csv> <datasets 폴더> <도전 폴더> <test 폴더> <catalog_sha>
"""
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from evaluation.metrics import CONTRACT_KEYS
from evaluation.scoring import score_turn
from prepare_validation import trim_provenance

SEED = 20261102
CHALLENGE_VERSION, TEST_VERSION = 'validation-challenge-20261102', 'test-20261102'
TOTALS = {'typo': 15, 'missing_title': 9, 'notation': 9, 'artist_mood': 9, 'catalog_gap': 9, 'multi_step': 7, 'dialogue': 4}
TEST_COUNTS = {'typo': 2, 'missing_title': 2, 'notation': 2, 'artist_mood': 2, 'catalog_gap': 2, 'multi_step': 1, 'dialogue': 4}
ROUTES = (['recommend', 'tracks'], ['recommend', 'no_tracks'], ['recommend', 'insufficient'],
          ['lookup', 'lookup_found'], ['lookup', 'lookup_not_found'], ['lookup', 'lookup_ambiguous'])
MIN_GENRE_TRACKS = 20
TEST_PHOTOS = {'normal': 3, 'difficult': 2, 'injection': 2, 'unreadable': 1}
# 이번 조사에서 내용을 본 후보는 test 후보에서 뺀다.
SEEN = {'T-002', 'T-020', 'T-043', 'T-058', 'T-072', 'D-007', 'P-002', 'F-001'}
PHOTO_CHECKS = [
    '장면 분석의 시각 요소와 설명이 원본 사진에서 확인되며, 정확한 장소·인물 감정을 추측하지 않는다.',
    'usable=true이면 music_query와 recommendation_context가 사진 및 장면 분석과 일관된다. usable=false이면 검색 문장이 비어 있고 재촬영을 안내한다.',
]


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def normal(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def distance(a, b, limit=2):
    """편집 거리. limit를 넘으면 limit+1을 돌려준다."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        now = [i]
        for j, y in enumerate(b, 1):
            now.append(min(now[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = now
    return row[-1]


def read_catalog(path):
    with open(path, newline='') as source:
        return list(csv.DictReader(source))


def passes_with_ideal_answer(expected, metrics):
    """기대값대로 응답했을 때 통과하는지 본다. 실패하면 어떤 응답도 통과할 수 없다."""
    route = expected['routes'][0]
    genres = expected.get('genres_allowed')
    count = expected.get('cards', 0 if route[1] in ('no_tracks', 'lookup_not_found') else 1)
    ids = [str(i) for i in expected.get('lookup_track_ids_any', [])] or ['1']
    cards = [{'track_id': ids[0] if i == 0 else f'other-{i}', 'title': expected.get('cards_title_contains', 'title'),
              'artist': expected.get('lookup_artist', 'artist'), 'genre': (genres or ['Pop'])[0],
              'release_year': expected.get('min_year') or expected.get('max_year') or 2000} for i in range(count)]
    state = {'recommend_min_year': expected.get('min_year'), 'recommend_max_year': expected.get('max_year'), 'recommend_genres': genres,
             'requested_count': expected.get('requested_count'), 'lookup_song': 'song', 'lookup_artist': 'artist'}
    scores = score_turn(expected, {'done': True, 'state': state, 'tracks': cards, 'route': route, 'text': ''}, metrics=metrics)
    return not any(key in CONTRACT_KEYS and score['status'] == 'scored' and score['score'] == 0 for key, score in scores.items())


def problems(cases, tracks):
    """사례가 형식과 카탈로그에 맞는지 본다. 내용을 남기지 않게 ID와 사유만 돌려준다."""
    found = []
    ids = Counter(c['metadata']['id'] for c in cases)
    families = Counter(c['metadata'].get('family_id') for c in cases)
    counts = Counter(c['metadata'].get('category') for c in cases)
    found += [f'{k}: 건수 {counts[k]}, 필요 {n}' for k, n in TOTALS.items() if counts[k] != n]
    found += [f'{k}: 중복 ID' for k, n in ids.items() if n > 1]
    found += [f'{k}: 중복 family_id' for k, n in families.items() if n > 1 or not k]
    by_id = {t['track_id']: t for t in tracks}
    titles = defaultdict(list)
    for track in tracks:
        titles[normal(track['artist'])].append(normal(track['title']))
    genre_size = Counter(t['genre'] for t in tracks)
    for case in cases:
        name, category = case['metadata']['id'], case['metadata'].get('category')
        turns = case['inputs'].get('turns', [])
        gold = case['outputs'].get('turns', [])
        if not turns or len(turns) != len(gold) or not all(t.get('message') for t in turns):
            found.append(f'{name}: inputs와 outputs의 턴 수')
            continue
        if category == 'dialogue' and len(turns) != 3 or category != 'dialogue' and len(turns) != 1:
            found.append(f'{name}: 턴 수')
        for expected in gold:
            if not expected.get('routes') or not expected.get('checks'):
                found.append(f'{name}: routes 또는 checks 없음')
                continue
            if any(route not in ROUTES for route in expected['routes']):
                found.append(f'{name}: 알 수 없는 경로')
            try:
                if not passes_with_ideal_answer(expected, case['metadata'].get('metrics', [])):
                    found.append(f'{name}: 어떤 응답도 통과할 수 없는 기대값')
            except (ValueError, TypeError) as error:
                found.append(f'{name}: 기대값 형식 {type(error).__name__}')
            for track_id in expected.get('lookup_track_ids_any', []):
                track = by_id.get(str(track_id))
                if not track or track['usable'] != '1':
                    found.append(f'{name}: 쓸 수 없는 track_id')
                elif expected.get('lookup_artist') and normal(expected['lookup_artist']) not in normal(track['artist']):
                    found.append(f'{name}: lookup_artist와 곡의 가수가 다름')
            if expected.get('lookup_artist') and normal(expected['lookup_artist']) not in titles:
                found.append(f'{name}: 카탈로그에 없는 가수')
        expected = gold[0]
        if category in ('typo', 'notation', 'multi_step') and not expected.get('lookup_track_ids_any'):
            found.append(f'{name}: lookup_track_ids_any 없음')
        if category == 'typo':
            real = normal(by_id.get(str(expected.get('lookup_track_ids_any', [''])[0]), {}).get('title', ''))
            if real and normal(turns[0]['message']).find(real) >= 0:
                found.append(f'{name}: 문장에 정확한 제목이 들어 있음')
        if category == 'missing_title':
            title = normal(case['metadata'].get('claimed_title', ''))
            known = titles.get(normal(case['metadata'].get('claimed_artist', '')))
            if known is None:
                found.append(f'{name}: 카탈로그에 없는 가수')
                known = []
            if not title or title not in normal(turns[0]['message']):
                found.append(f'{name}: claimed_title이 문장에 없음')
            if any(distance(title, t) <= 2 for t in known):
                found.append(f'{name}: 같은 가수의 실제 제목과 너무 가까움')
        if category == 'multi_step' and expected.get('lookup_artist'):
            found.append(f'{name}: 다른 가수의 추천 곡을 막는 기대값')
        if category == 'catalog_gap':
            genres = expected.get('genres_allowed') or []
            if re.search('초반|중반|후반', turns[0]['message']):
                found.append(f'{name}: 연대 경계가 애매함')
            if any(genre_size[g] < MIN_GENRE_TRACKS for g in genres):
                found.append(f'{name}: 장르 곡 수 부족')
            low, high = expected.get('min_year'), expected.get('max_year')
            if not genres or low is None or high is None:
                found.append(f'{name}: 장르·연도 조건 없음')
            elif any(t['genre'] in genres and t['release_year'] and low <= int(t['release_year']) <= high for t in tracks):
                found.append(f'{name}: 조건에 맞는 곡이 카탈로그에 있음')
    return found


def split_cases(cases):
    """유형마다 고정 시드로 test 몫을 뽑는다."""
    rng = random.Random(SEED)
    grouped = defaultdict(list)
    for case in sorted(cases, key=lambda c: c['metadata']['id']):
        grouped[case['metadata']['category']].append(case)
    test = []
    for category, count in TEST_COUNTS.items():
        test += rng.sample(grouped[category], count)
    chosen = {c['metadata']['id'] for c in test}
    return [c for c in cases if c['metadata']['id'] not in chosen], test


def pick_photos(datasets, counts=TEST_PHOTOS, skip=()):
    """미사용 test 후보에서 고른다. 평가에 쓴 사진과 같은 묶음, 조사에서 본 사례는 뺀다.

    시드를 쓰지 않는다. 코드는 Git에 올라가므로 고정 시드면 같은 선택을 재현할 수 있다.
    """
    used = set()
    for kind in ('text', 'photo', 'dialogue'):
        for row in load(datasets / f'initial-20261027/{kind}.jsonl'):
            used |= {row['metadata']['id'], row['metadata'].get('family_id')}
    pool = defaultdict(list)
    for row in load(datasets / 'baseline/photo.jsonl'):
        meta = row['metadata']
        if row['split'] == 'test' and meta['id'] not in used | SEEN | set(skip) and meta.get('family_id') not in used:
            pool[meta['category']].append(row)
    rng = random.SystemRandom()
    picked = [row for category, count in counts.items() for row in rng.sample(pool[category], count)]
    for row in picked:
        row['outputs']['checks'] = PHOTO_CHECKS
    return picked


def stamp(cases, split, version):
    for case in cases:
        case['split'] = split
        case['metadata'].update(dataset_version=version, human_reviewed=False)
    return cases


def build(cases, datasets, challenge, test, catalog_sha):
    challenge.mkdir(parents=True)
    test.mkdir(parents=True)
    for folder in (challenge, test):
        (folder / 'catalog_sha.txt').write_text(catalog_sha + '\n')
    rest, hidden = split_cases(cases)
    write(challenge / 'text.jsonl', stamp(rest, 'validation', CHALLENGE_VERSION))
    stamp(hidden, 'test', TEST_VERSION)
    write(test / 'text.jsonl', [c for c in hidden if c['metadata']['category'] != 'dialogue'])
    write(test / 'dialogue.jsonl', [c for c in hidden if c['metadata']['category'] == 'dialogue'])
    photos = stamp(pick_photos(datasets), 'test', TEST_VERSION)
    write(test / 'photo.jsonl', photos)
    (test / 'photo/images').mkdir(parents=True)
    manifest = {}
    for row in photos:
        for relative in row['attachments'].values():
            shutil.copyfile(datasets / 'baseline' / relative, test / relative)
            manifest[relative] = hashlib.sha256((test / relative).read_bytes()).hexdigest()
    (test / 'photo/manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copyfile(datasets / 'baseline/photo/provenance.json', test / 'photo/provenance.json')
    trim_provenance(test / 'photo/provenance.json', {row['metadata']['id'] for row in photos})
    assert len(rest) == 47 and len(hidden) == 15 and len(photos) == 8
    assert not {c['metadata']['family_id'] for c in rest} & {c['metadata']['family_id'] for c in hidden}


def main():
    command, cases, tracks = sys.argv[1], load(sys.argv[2]), read_catalog(sys.argv[3])
    found = problems(cases, tracks)
    for line in found:
        print(line)
    print(f'사례 {len(cases)}건, 문제 {len(found)}건')
    if command == 'build':
        assert not found, 'fix cases first'
        build(cases, Path(sys.argv[4]), Path(sys.argv[5]), Path(sys.argv[6]), sys.argv[7])
        print('도전 47건, test 23건 생성')


if __name__ == '__main__':
    main()
