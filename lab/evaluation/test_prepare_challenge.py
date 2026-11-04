import prepare_challenge as p


TRACK = {'track_id': '1', 'title': 'Real Song', 'artist': 'Alpha', 'genre': 'Jazz', 'release_year': '2001', 'usable': '1'}


def case(number, category, expected=None, message='문장', **meta):
    turns = 3 if category == 'dialogue' else 1
    gold = {'routes': [['lookup', 'lookup_found']], 'checks': ['기준'], **(expected or {})}
    return {'inputs': {'turns': [{'message': message}] * turns}, 'outputs': {'turns': [gold] * turns},
            'metadata': {'id': f'C-{number}', 'family_id': f'family-{number}', 'category': category, **meta}}


def full_set():
    numbers = iter(range(1000))
    return [case(next(numbers), category) for category, count in p.TOTALS.items() for _ in range(count)]


def reasons(cases, tracks=(TRACK,)):
    return [line.split(': ', 1)[1] for line in p.problems(cases, list(tracks))
            if not line.split(':')[0] in p.TOTALS]


def test_split_is_repeatable_and_keeps_the_agreed_counts():
    rest, hidden = p.split_cases(full_set())
    again_rest, again_hidden = p.split_cases(full_set())
    assert hidden == again_hidden and rest == again_rest
    assert len(rest) == 47 and len(hidden) == 15
    assert sum(c['metadata']['category'] == 'dialogue' for c in hidden) == 4
    assert not any(c['metadata']['category'] == 'dialogue' for c in rest)


def test_typo_must_not_contain_the_real_title():
    found = reasons([case(1, 'typo', {'lookup_track_ids_any': [1], 'lookup_artist': 'Alpha'}, message='Alpha의 Real Song 찾아줘')])
    assert '문장에 정확한 제목이 들어 있음' in found


def test_track_must_be_usable_and_belong_to_the_artist():
    assert '쓸 수 없는 track_id' in reasons([case(1, 'typo', {'lookup_track_ids_any': [2]})])
    closed = dict(TRACK, usable='0')
    assert '쓸 수 없는 track_id' in reasons([case(1, 'typo', {'lookup_track_ids_any': [1]})], [closed])
    other = dict(TRACK, track_id='2', artist='Beta')
    found = reasons([case(1, 'typo', {'lookup_track_ids_any': [2], 'lookup_artist': 'Alpha'})], [TRACK, other])
    assert 'lookup_artist와 곡의 가수가 다름' in found


def test_missing_title_must_be_far_from_the_real_titles():
    near = case(1, 'missing_title', message='Alpha의 Real Sung 찾아줘', claimed_title='Real Sung', claimed_artist='Alpha')
    far = case(2, 'missing_title', message='Alpha의 Paper River 찾아줘', claimed_title='Paper River', claimed_artist='Alpha')
    assert '같은 가수의 실제 제목과 너무 가까움' in reasons([near])
    assert '같은 가수의 실제 제목과 너무 가까움' not in reasons([far])


def test_catalog_gap_must_really_be_empty():
    gap = {'genres_allowed': ['Jazz'], 'min_year': 2000, 'max_year': 2009}
    assert '조건에 맞는 곡이 카탈로그에 있음' in reasons([case(1, 'catalog_gap', gap)])
    assert '조건에 맞는 곡이 카탈로그에 있음' not in reasons([case(1, 'catalog_gap', dict(gap, min_year=2010, max_year=2019))])


def test_unknown_expectation_key_is_reported():
    assert '기대값 형식 ValueError' in reasons([case(1, 'typo', {'lookup_artists': 'Alpha'})])


def test_photo_pick_skips_used_families_and_seen_cases(tmp_path):
    import json
    (tmp_path / 'initial-20261027').mkdir()
    (tmp_path / 'baseline').mkdir()
    used = {'inputs': {}, 'outputs': {}, 'metadata': {'id': 'P-1', 'family_id': 'shared'}}
    for kind in ('text', 'photo', 'dialogue'):
        (tmp_path / f'initial-20261027/{kind}.jsonl').write_text(json.dumps(used) + '\n' if kind == 'photo' else '')
    rows = [{'outputs': {}, 'split': 'test', 'metadata': {'id': f'P-{category}-{i}', 'family_id': f'{category}-{i}', 'category': category}}
            for category, count in p.TEST_PHOTOS.items() for i in range(count)]
    rows += [{'outputs': {}, 'split': 'test', 'metadata': {'id': 'P-same', 'family_id': 'shared', 'category': 'normal'}},
             {'outputs': {}, 'split': 'dev', 'metadata': {'id': 'P-dev', 'family_id': 'dev', 'category': 'normal'}},
             {'outputs': {}, 'split': 'test', 'metadata': {'id': 'P-002', 'family_id': 'seen', 'category': 'normal'}}]
    (tmp_path / 'baseline/photo.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    picked = {r['metadata']['id'] for r in p.pick_photos(tmp_path)}
    assert len(picked) == 8 and not picked & {'P-same', 'P-dev', 'P-002'}


def test_expectation_that_no_answer_can_pass_is_reported():
    impossible = {'routes': [['lookup', 'lookup_not_found']], 'lookup_artist': 'Alpha', 'cards': 0}
    assert '어떤 응답도 통과할 수 없는 기대값' in reasons([case(1, 'typo', impossible)])
    possible = {'routes': [['lookup', 'lookup_found']], 'lookup_artist': 'Alpha', 'lookup_track_ids_any': [1], 'cards': 1}
    assert '어떤 응답도 통과할 수 없는 기대값' not in reasons([case(1, 'typo', possible)])


def test_route_outside_the_agreed_list_is_reported():
    assert '알 수 없는 경로' in reasons([case(1, 'typo', {'routes': [['lookup', 'found']]})])


def test_vague_year_range_and_thin_genre_are_reported_for_catalog_gap():
    gap = {'genres_allowed': ['Jazz'], 'min_year': 2010, 'max_year': 2019}
    assert '연대 경계가 애매함' in reasons([case(1, 'catalog_gap', gap, message='2010년대 초반 재즈')])
    assert '연대 경계가 애매함' not in reasons([case(1, 'catalog_gap', gap, message='2010년대 재즈')])
    assert '장르 곡 수 부족' in reasons([case(1, 'catalog_gap', gap)])


def test_multi_step_must_allow_other_artists():
    found = reasons([case(1, 'multi_step', {'lookup_track_ids_any': [1], 'lookup_artist': 'Alpha', 'cards': 2})])
    assert '다른 가수의 추천 곡을 막는 기대값' in found


def test_missing_title_needs_a_claimed_artist():
    bad = case(1, 'missing_title', {'routes': [['lookup', 'lookup_not_found']], 'cards': 0}, message='Paper River', claimed_title='Paper River')
    assert '카탈로그에 없는 가수' in reasons([bad])
    good = case(2, 'missing_title', {'routes': [['lookup', 'lookup_not_found']], 'cards': 0}, message='Paper River',
                claimed_title='Paper River', claimed_artist='Alpha')
    assert '카탈로그에 없는 가수' not in reasons([good])


def test_provenance_keeps_only_the_photos_in_the_folder(tmp_path):
    import json
    from prepare_validation import trim_provenance
    path = tmp_path / 'provenance.json'
    path.write_text(json.dumps([{'id': 'P-1', 'url': 'a'}, {'id': 'P-2', 'url': 'b'}]))
    trim_provenance(path, {'P-1'})
    assert [row['id'] for row in json.loads(path.read_text())] == ['P-1']


def photo_candidates(tmp_path):
    import json
    (tmp_path / 'initial-20261027').mkdir()
    (tmp_path / 'baseline').mkdir()
    for kind in ('text', 'photo', 'dialogue'):
        (tmp_path / f'initial-20261027/{kind}.jsonl').write_text('')
    rows = [{'outputs': {}, 'split': 'test', 'metadata': {'id': f'P-{category}-{i}', 'family_id': f'{category}-{i}', 'category': category}}
            for category in p.TEST_PHOTOS for i in range(6)]
    (tmp_path / 'baseline/photo.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_photo_pick_cannot_be_rebuilt_from_the_code(tmp_path, monkeypatch):
    photo_candidates(tmp_path)
    def fixed(*_):
        raise AssertionError('고정 시드를 쓰면 공개 코드로 선택을 재현할 수 있다')
    monkeypatch.setattr(p.random, 'Random', fixed)
    assert len(p.pick_photos(tmp_path)) == 8


def test_photo_pick_can_replace_some_and_skip_retired_ones(tmp_path):
    photo_candidates(tmp_path)
    skip = {f'P-normal-{i}' for i in range(3)}
    picked = p.pick_photos(tmp_path, counts={'normal': 3}, skip=skip)
    assert len(picked) == 3 and not {r['metadata']['id'] for r in picked} & skip
