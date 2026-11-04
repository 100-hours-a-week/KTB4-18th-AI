"""기존 후보를 선별하고, 모델 출력과 독립적인 초기 회귀 평가셋을 고정한다."""
import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'lab/datasets/baseline'
TARGET = ROOT / 'lab/datasets/initial-20261027'


def load(kind):
    return {r['metadata']['id']: r for r in map(json.loads, (SOURCE / f'{kind}.jsonl').read_text().splitlines())}


def write(kind, rows):
    (TARGET / f'{kind}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def reviewed(row):
    row = copy.deepcopy(row)
    row['split'] = 'dev'
    row['metadata'].update(human_reviewed=False, labels_by='code-and-image-reviewed-20261027',
                           review_status='ai_reviewed_not_human_gold', dataset_version='initial-20261027')
    return row


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    texts = load('text')
    ids = [1,3,7,10,11,14,17,21,23,24,25,26,28,30,31,39,40,41,46,51,56,61,71,78,81,82,89,91,94,99]
    rows = [reviewed(texts[f'T-{n:03}']) for n in ids]
    for row in rows:
        gold = row['outputs']['turns'][0]
        gold['checks'] = ['사용자의 음악 요청에 맞는 응답이며, 지원하지 않는 기능을 수행했다고 주장하지 않는다.']
        if 'requested_count' in gold:
            gold['cards'] = min(gold['requested_count'], 5)
        if row['metadata']['id'] == 'T-030':
            row['inputs']['turns'][0] = {'message': 'Jazz 장르에서 느긋한 아침에 들을 음악을 추천해줘.'}
            gold['genres_allowed'] = ['Jazz']
        if row['metadata']['id'] == 'T-007':
            row['inputs']['turns'][0] = {'message': 'Classical 장르에서 시험 공부에 집중할 차분한 음악을 추천해줘.'}
            gold['genres_allowed'] = ['Classical']
    assert len(rows) == 30
    write('text', rows)

    photos = load('photo')
    ids = [1,5,9,13,17,21,25,28,31,33,35,40,41,42,43,50]
    rows = [reviewed(photos[f'P-{n:03}']) for n in ids]
    manifest = {}
    for row in rows:
        for rel in row['attachments'].values():
            path = TARGET / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE / rel, path)
            manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        row['outputs']['checks'] = [
            '장면 분석의 시각 요소와 설명이 원본 사진에서 확인되며, 정확한 장소·인물 감정을 추측하지 않는다.',
            'usable=true이면 music_query와 recommendation_context가 사진 및 장면 분석과 일관된다. usable=false이면 검색 문장이 비어 있고 재촬영을 안내한다.',
        ]
    (TARGET / 'photo/manifest.json').write_text(json.dumps(manifest, indent=2))
    shutil.copyfile(SOURCE / 'photo/provenance.json', TARGET / 'photo/provenance.json')
    assert len(rows) == 16
    write('photo', rows)

    dialogues = load('dialogue')
    thirds = [
        ('경쾌한 느낌은 유지하고 두 곡만 추천해줘.', {'routes': [['recommend','tracks']], 'requested_count':2, 'cards':2}),
        ('이번에는 2000년대 곡으로 바꿔줘.', {'routes': [['recommend','tracks']], 'min_year':2000, 'max_year':2009}),
        ('이번에도 앞에서 보여준 곡은 전부 빼고 더 찾아줘.', {'routes': [['recommend','tracks']]}),
        ('같은 분위기로 이번에는 한 곡만 부탁해.', {'routes': [['recommend','tracks']], 'requested_count':1, 'cards':1}),
        ('그 조건은 취소할게. 친구를 만날 때 들을 밝은 음악을 추천해줘.', {'routes': [['recommend','tracks']]}),
    ]
    rows = []
    for n, (message, gold) in enumerate(thirds, 1):
        row = reviewed(dialogues[f'D-{n:03}'])
        row['inputs']['turns'].append({'message': message})
        row['outputs']['turns'].append(gold)
        row['metadata']['metrics'] = ['exclusion']
        for turn in row['outputs']['turns']:
            if 'requested_count' in turn: turn['cards'] = min(turn['requested_count'], 5)
            turn['checks'] = ['대화에서 유지할 조건과 이번에 바꾼 조건을 구분하여 현재 요청을 수행한다.']
        rows.append(row)
    row = reviewed(dialogues['D-001'])
    row['metadata'].update(id='D-006', family_id='D-006', metrics=[])
    row['inputs'] = {'turns':[{'message':'아이유 곡을 찾아줘.'}, {'message':'이번에는 BTS 곡으로 바꿔줘.'}, {'message':'처음에 말한 가수 곡을 다시 찾아줘.'}]}
    row['outputs'] = {'turns':[{'routes':[['lookup','lookup_found']], 'lookup_artist':artist,
                              'checks':['현재 지칭하는 가수를 이전 대화에 근거해 올바르게 해석한다.']} for artist in ('IU','BTS','IU')]}
    rows.append(row)
    assert len(rows) == 6 and all(len(r['inputs']['turns']) == 3 for r in rows)
    write('dialogue', rows)
    faults = [reviewed(r) for r in load('faults').values()]
    assert len(faults) == 20
    write('faults', faults)
    prepare_audio()
    print(TARGET)


def prepare_audio():
    """합성 음성임을 명시한다. 실제 사용자·다양한 억양 성능으로 일반화하지 않는다."""
    import numpy as np
    from imageio_ffmpeg import get_ffmpeg_exe
    phrases = [
        '비 오는 날 집에서 들을 차분한 음악을 추천해줘.',
        '아이유 노래를 두 곡만 찾아줘.',
        '퇴근길에 들을 신나는 음악이 필요해.',
        '이천십 년부터 이천십구 년까지 나온 음악을 추천해줘.',
        '클래식 장르에서 공부할 때 들을 곡을 찾아줘.',
        '방금 추천한 곡은 빼고 다른 음악을 들려줘.',
        '잠들기 전에 듣기 좋은 조용한 음악을 골라줘.',
        '음악 추천을 받을 때 어떤 조건을 말하면 되나요?',
    ]
    rows = []
    (TARGET / 'audio').mkdir(exist_ok=True)
    for index, phrase in enumerate(phrases, 1):
        rel = f'audio/S-{index:03}.wav'
        path = TARGET / rel
        if not path.exists():
            with tempfile.TemporaryDirectory() as temp:
                aiff = Path(temp) / 'speech.aiff'
                subprocess.run(['/usr/bin/say','-v','Yuna','-r',str(210 if index == 7 else 160),'-o',str(aiff),phrase],check=True)
                subprocess.run([get_ffmpeg_exe(),'-nostdin','-loglevel','error','-i',str(aiff),'-ac','1','-ar','16000','-c:a','pcm_s16le',str(path)],check=True)
            if index in (5,6):
                with wave.open(str(path),'rb') as wav:
                    pcm=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(float)
                rng=np.random.default_rng(index)
                pcm=np.clip(pcm*.3+rng.normal(0,90,len(pcm)),-32768,32767).astype('<i2')
                with wave.open(str(path),'wb') as wav:
                    wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(16000);wav.writeframes(pcm.tobytes())
        rows.append({'inputs':{'audio':rel}, 'outputs':{'transcript':phrase},
                     'metadata':{'id':f'S-{index:03}', 'family_id':f'S-{index:03}', 'category':'synthetic_speech',
                                 'purpose':'stt_smoke_baseline', 'source':'macOS say Yuna; seeded noise for S-005/S-006',
                                 'human_reviewed':False, 'dataset_version':'initial-20261027',
                                 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}, 'split':'dev',
                     'attachments':{'audio':rel}})
    assert len(rows)==8
    write('stt',rows)


if __name__ == '__main__':
    main()
