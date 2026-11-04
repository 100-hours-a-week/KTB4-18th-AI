"""초기 회귀 평가셋을 validation으로 옮긴다. 입력과 기대값은 바꾸지 않는다.

사용: python prepare_validation.py <initial-20261027 폴더> <새 validation 폴더>
"""
import json
import shutil
import sys
from pathlib import Path

VERSION = 'validation-20261102'
KINDS = ('text', 'photo', 'dialogue', 'stt', 'faults')
# 작업 브랜치에서 빠르게 돌릴 사례다. 기능과 경로가 한 번씩 지나가게 골랐다.
SMOKE = {'T-011', 'T-023', 'T-051', 'T-061', 'T-081', 'T-094', 'P-001', 'P-031', 'P-041',
         'D-001', 'D-006', 'S-001', 'F-001', 'F-009', 'F-020'}


def trim_provenance(path, ids):
    """폴더에 있는 사진의 출처만 남긴다. 다른 사진의 출처가 들어가면 test 후보가 드러난다."""
    path.write_text(json.dumps([r for r in json.loads(path.read_text()) if r['id'] in ids], ensure_ascii=False, indent=2))


def main(source, target):
    target.mkdir(parents=True)
    seen = set()
    for kind in KINDS:
        rows = [json.loads(line) for line in (source / f'{kind}.jsonl').read_text().splitlines()]
        for row in rows:
            seen.add(row['metadata']['id'])
            row['split'] = ['validation', 'smoke'] if row['metadata']['id'] in SMOKE else 'validation'
            row['metadata']['dataset_version'] = VERSION
        (target / f'{kind}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    assert SMOKE <= seen and len(seen) == 80
    for folder in ('photo', 'audio'):
        shutil.copytree(source / folder, target / folder)
    photos = (target / 'photo.jsonl').read_text().splitlines()
    trim_provenance(target / 'photo/provenance.json', {json.loads(line)['metadata']['id'] for line in photos})


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
