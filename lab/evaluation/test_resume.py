import asyncio
import json
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

import baseline_suite
import evaluation.runner
import langsmith


def row(case_id='T-1', split='validation'):
    return {'inputs': {'message': case_id}, 'outputs': {}, 'metadata': {'id': case_id}, 'split': split}


def actual(text='안내', error=None, tracks=({'track_id': '1'},)):
    return {'input': {}, 'text': text, 'tracks': list(tracks), 'done': error is None, 'route': ['recommend', 'tracks'],
            'state': {}, 'total_ms': 100, 'first_text_ms': 10, 'cards_ms': 20, **({'error': error} if error else {})}


def service(*actuals):
    return {'expected_turns': len(actuals), 'metadata': {'metrics': []},
            'turns': [{'expected': {}, 'actual': a, 'scores': {'service_success': {'status': 'scored', 'score': 1}}}
                      for a in actuals]}


VERDICT = {'status': 'ok', 'response_compliance': 1, 'faithfulness': 1, 'explanation': 2, 'scene_accuracy': None,
           'scene_consistency': None, 'reasons': ['근거'], 'usage': {'cost': .01}}


class Harness:
    """외부 호출 없이 run_kind를 돌리고 서비스·채점 호출 수를 센다."""

    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path, self.monkeypatch = tmp_path, monkeypatch
        self.service_calls, self.judge_calls = [], []
        self.outcomes, self.verdicts = {}, []
        self.on_data = lambda data: None
        self.client = MagicMock()
        self.client.has_dataset.return_value = True
        self.client.read_dataset.return_value = SimpleNamespace(id=uuid4())
        self.client.create_project.return_value = SimpleNamespace(id=uuid4(), name='test')
        self.client.read_project.return_value = SimpleNamespace(id=uuid4(), name='test')

    def run(self, rows, *, resume=False, split=None, no_judge=False, kind='text', name='개선'):
        examples = [SimpleNamespace(id=uuid4(), inputs=r['inputs'], outputs=r['outputs'], metadata=r['metadata']) for r in rows]
        self.client.list_examples.return_value = examples

        async def run_case(found, _):
            case_id = found['metadata']['id']
            self.service_calls.append(case_id)
            return self.outcomes.get(case_id, service(actual()))

        async def evaluate(target, **kw):
            self.on_data(kw['data'])
            for example in kw['data']:
                case = await target(example.inputs, example.metadata)
                await kw['evaluators'][0](SimpleNamespace(id=uuid4(), outputs=case), example)

        def judge(*args):
            self.judge_calls.append(args[0]['actual']['text'])
            verdict = self.verdicts.pop(0) if self.verdicts else VERDICT
            if isinstance(verdict, Exception):
                raise verdict
            return verdict

        self.monkeypatch.setattr(evaluation.runner, 'run_case', run_case)
        self.monkeypatch.setattr(langsmith, 'aevaluate', evaluate)
        self.monkeypatch.setattr(langsmith, 'get_current_run_tree', lambda: None)
        self.monkeypatch.setattr(langsmith, 'trace', MagicMock())
        self.monkeypatch.setattr(baseline_suite, 'judge', judge)
        args = SimpleNamespace(name=name, output=self.tmp_path, concurrency=2, no_judge=no_judge, service_root=self.tmp_path,
                               judge_budget_usd=Decimal('5'), judge_spent_usd=Decimal(0), judge_reservation_usd=Decimal('.1'),
                               resume=resume, split=split)
        return asyncio.run(baseline_suite.run_kind(kind, rows, self.client, 'tag', args, {}, SimpleNamespace(records=[])))

    def lines(self, name):
        path = self.tmp_path / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.fixture
def harness(tmp_path, monkeypatch):
    return Harness(tmp_path, monkeypatch)


def test_service_output_survives_a_crash_during_judging(harness):
    harness.verdicts = [RuntimeError('채점 중 종료')]
    with pytest.raises(RuntimeError):
        harness.run([row()])
    assert [c['id'] for c in harness.lines('text.service.jsonl')] == ['T-1']
    assert harness.lines('text.jsonl') == []


def test_resume_judges_saved_output_without_calling_the_service_again(harness):
    harness.verdicts = [RuntimeError('채점 중 종료')]
    with pytest.raises(RuntimeError):
        harness.run([row()])
    result = harness.run([row()], resume=True)
    assert harness.service_calls == ['T-1']
    assert len(harness.judge_calls) == 2
    assert result['status'] == 'complete'


def test_resume_skips_finished_cases(harness):
    harness.run([row('T-1')])
    result = harness.run([row('T-1'), row('T-2')], resume=True)
    assert harness.service_calls == ['T-1', 'T-2']
    assert harness.judge_calls == ['안내', '안내']
    assert result['cases'] == 2
    assert [c['id'] for c in harness.lines('text.jsonl')] == ['T-1', 'T-2']


def test_resume_builds_a_new_experiment_with_one_final_run_per_example(harness):
    harness.outcomes['T-1'] = service(actual(error='BoomError'))
    harness.run([row('T-1'), row('T-2')])
    harness.outcomes['T-1'] = service(actual())
    harness.client.reset_mock(return_value=False)
    seen = []
    harness.on_data = seen.append
    result = harness.run([row('T-1'), row('T-2')], resume=True)
    assert [e.metadata['id'] for e in seen[0]] == ['T-1', 'T-2']
    assert harness.service_calls == ['T-1', 'T-2', 'T-1']
    assert harness.judge_calls.count('안내') == 2
    name = harness.client.create_project.call_args.args[0]
    assert '시도 2' in name and result['status'] == 'complete'


def test_previous_attempt_is_labelled_after_the_resume_finishes(harness):
    harness.outcomes['T-1'] = service(actual(error='BoomError'))
    harness.run([row()])
    harness.outcomes['T-1'] = service(actual())
    harness.client.read_project.return_value = SimpleNamespace(id=uuid4(), name='첫 시도')
    harness.run([row()], resume=True)
    names = [c.kwargs.get('name') for c in harness.client.update_project.call_args_list]
    assert '[이전 시도] 첫 시도' in names


def test_resume_with_nothing_left_creates_no_experiment(harness):
    harness.run([row()])
    harness.client.create_project.reset_mock()
    result = harness.run([row()], resume=True)
    assert harness.client.create_project.call_count == 0
    assert result['status'] == 'complete' and result['cases'] == 1


def test_failed_service_is_rerun_and_the_failure_stays_in_history(harness):
    harness.outcomes['T-1'] = service(actual(error='BoomError'))
    result = harness.run([row()])
    assert result['status'] == 'incomplete'
    harness.outcomes['T-1'] = service(actual())
    result = harness.run([row()], resume=True)
    assert harness.service_calls == ['T-1', 'T-1']
    assert result['status'] == 'complete'
    assert len(harness.lines('text.service.jsonl')) == 2


def test_only_the_failed_turn_is_judged_again(harness):
    harness.outcomes['T-1'] = service(actual('첫 턴'), actual('둘째 턴'))
    harness.verdicts = [VERDICT, {'status': 'evaluation_error', 'error': 'Timeout', 'usage': {'cost': 0}}]
    harness.run([row()])
    harness.run([row()], resume=True)
    assert harness.judge_calls == ['첫 턴', '둘째 턴', '둘째 턴']
    final = baseline_suite.read_saved(harness.tmp_path / 'text.jsonl')['T-1']
    assert all(t['initial_judge']['status'] == 'ok' for t in final['turns'])
    assert final['judge_cost_usd'] == pytest.approx(.02)


def test_paid_failed_judging_stays_in_the_total_after_a_resume(harness):
    harness.verdicts = [{'status': 'evaluation_error', 'error': 'Timeout', 'usage': {'cost': .03}}]
    harness.run([row()])
    harness.run([row()], resume=True)
    assert baseline_suite.judge_ledger(harness.tmp_path) == {'spent': Decimal('.04'), 'missing': 0}
    final = baseline_suite.read_saved(harness.tmp_path / 'text.jsonl')['T-1']
    assert final['judge_cost_usd'] == pytest.approx(.04)


def test_judge_charge_survives_a_crash_before_the_result_is_saved(harness):
    harness.outcomes['T-1'] = service(actual('첫 턴'), actual('둘째 턴'))
    harness.verdicts = [VERDICT, RuntimeError('채점 중 종료')]
    with pytest.raises(RuntimeError):
        harness.run([row()])
    assert harness.lines('text.jsonl') == []
    assert baseline_suite.judge_ledger(harness.tmp_path)['spent'] == Decimal('.01')


def test_judge_call_without_a_known_charge_is_counted_as_missing(harness):
    harness.verdicts = [{'status': 'evaluation_error', 'error': 'Timeout', 'usage': None}]
    with pytest.raises(AssertionError):
        harness.run([row()])
    assert baseline_suite.judge_ledger(harness.tmp_path) == {'spent': Decimal(0), 'missing': 1}


def test_torn_last_line_is_treated_as_not_run(harness):
    harness.run([row('T-1')])
    with (harness.tmp_path / 'text.service.jsonl').open('a') as out:
        out.write('{"id": "T-2", "tur')
    harness.run([row('T-1'), row('T-2')], resume=True)
    assert harness.service_calls == ['T-1', 'T-2']
    assert 'T-2' in baseline_suite.read_saved(harness.tmp_path / 'text.service.jsonl')


def test_code_check_run_can_be_judged_later_from_saved_output(harness):
    harness.run([row()], no_judge=True)
    assert harness.judge_calls == []
    harness.run([row()], resume=True)
    assert harness.service_calls == ['T-1']
    assert len(harness.judge_calls) == 1


def test_split_selects_cases_and_smoke_can_sit_inside_validation(harness):
    rows = [row('T-1', ['validation', 'smoke']), row('T-2', 'validation'), row('T-3', 'test')]
    result = harness.run(rows, split='smoke')
    assert harness.service_calls == ['T-1']
    assert result['planned'] == 1
    assert [r['metadata']['id'] for r in baseline_suite.select_split(rows, 'validation')] == ['T-1', 'T-2']


def test_status_is_incomplete_when_a_planned_case_has_no_result(harness):
    harness.verdicts = [RuntimeError('채점 중 종료')]
    harness.client.create_project.return_value = SimpleNamespace(id=uuid4(), name='test')
    with pytest.raises(RuntimeError):
        harness.run([row('T-1'), row('T-2')])
    harness.verdicts = [VERDICT]
    result = harness.run([row('T-1')], resume=True, split=None)
    assert result['planned'] == 1 and result['status'] == 'complete'


def test_voice_and_fault_runs_save_output_too(harness, monkeypatch):
    rows = [{'inputs': {'pytest_node': 'tests/x.py::y'}, 'outputs': {'passed': True}, 'metadata': {'id': 'F-1'}, 'split': 'validation'}]
    monkeypatch.setattr(baseline_suite.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='', stderr=''))
    result = harness.run(rows, kind='faults')
    assert result['status'] == 'complete'
    assert harness.lines('faults.service.jsonl')[0]['success'] is True


def test_dataset_choice_changes_remote_names_and_keeps_the_old_one(tmp_path, monkeypatch):
    for name in ('DATA', 'DATASET_VERSION', 'PREFIX'):
        monkeypatch.setattr(baseline_suite, name, getattr(baseline_suite, name))
    baseline_suite.use_dataset(tmp_path, 'v2-20261110')
    assert (baseline_suite.DATA, baseline_suite.PREFIX) == (tmp_path.resolve(), 'meomuneum-v2-20261110')
    baseline_suite.use_dataset(tmp_path, 'initial-20261027')
    assert baseline_suite.PREFIX == 'meomuneum-initial'


def test_resume_refuses_changed_conditions():
    stored = {'service_commit': 'a', 'rubric_version': 'r', 'db_before': {'sha': 'x'}}
    baseline_suite.assert_resumable(stored, dict(stored, started='later'))
    with pytest.raises(AssertionError, match='service_commit'):
        baseline_suite.assert_resumable(stored, dict(stored, service_commit='b'))
    with pytest.raises(AssertionError, match='db_before'):
        baseline_suite.assert_resumable(stored, dict(stored, db_before={'sha': 'y'}))


def test_evaluator_digest_covers_scoring_code():
    assert baseline_suite.evaluator_digest() != baseline_suite.digest(baseline_suite.Path(baseline_suite.__file__))


def test_assets_digest_changes_with_attachment_bytes(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_suite, 'DATA', tmp_path)
    (tmp_path / 'a.wav').write_bytes(b'one')
    rows = [{'attachments': {'audio': 'a.wav'}, 'metadata': {'id': 'S-1'}}]
    first = baseline_suite.assets_digest('stt', rows)
    (tmp_path / 'a.wav').write_bytes(b'two')
    assert baseline_suite.assets_digest('stt', rows) != first


def test_attachment_that_differs_from_its_recorded_hash_stops_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_suite, 'DATA', tmp_path)
    (tmp_path / 'a.wav').write_bytes(b'one')
    rows = [{'attachments': {'audio': 'a.wav'}, 'metadata': {'id': 'S-1', 'sha256': 'wrong'}}]
    with pytest.raises(AssertionError, match='a.wav'):
        baseline_suite.assets_digest('stt', rows)


def test_photo_manifest_hash_is_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(baseline_suite, 'DATA', tmp_path)
    (tmp_path / 'photo').mkdir()
    (tmp_path / 'photo/a.jpg').write_bytes(b'one')
    (tmp_path / 'photo/manifest.json').write_text(json.dumps({'photo/a.jpg': 'wrong'}))
    with pytest.raises(AssertionError, match='a.jpg'):
        baseline_suite.assets_digest('photo', [{'attachments': {'image': 'photo/a.jpg'}, 'metadata': {'id': 'P-1'}}])


def test_resume_refuses_changed_attachments():
    stored = {'assets_sha256': {'photo': 'a'}}
    with pytest.raises(AssertionError, match='assets_sha256'):
        baseline_suite.assert_resumable(stored, {'assets_sha256': {'photo': 'b'}})


def test_a_folder_holds_one_run_only(tmp_path):
    baseline_suite.check_output(tmp_path, resume=False)
    (tmp_path / 'text.jsonl').write_text('')
    with pytest.raises(AssertionError, match='fresh'):
        baseline_suite.check_output(tmp_path, resume=False)
    (tmp_path / 'text.jsonl').unlink()
    with pytest.raises(AssertionError, match='nothing to resume'):
        baseline_suite.check_output(tmp_path, resume=True)
    (tmp_path / 'manifest.json').write_text('{}')
    baseline_suite.check_output(tmp_path, resume=True)
    with pytest.raises(AssertionError, match='fresh'):
        baseline_suite.check_output(tmp_path, resume=False)


@pytest.mark.parametrize('value,expected', [(None, 'gemini'), ('', 'gemini'), (' V4 ', 'v4')])
def test_search_embedding_is_recorded_as_the_service_reads_it(monkeypatch, value, expected):
    monkeypatch.delenv('SEARCH_EMBEDDING', raising=False)
    if value is not None:
        monkeypatch.setenv('SEARCH_EMBEDDING', value)
    assert baseline_suite.search_embedding() == expected


def test_only_a_full_run_becomes_the_dataset_baseline(harness):
    rows = [row('T-1', ['validation', 'smoke']), row('T-2', 'validation')]
    harness.run(rows, split='smoke', name='baseline-새 기준')
    assert not harness.client.request_with_retries.called
    (harness.tmp_path / 'text.jsonl').unlink()
    (harness.tmp_path / 'text.service.jsonl').unlink()
    harness.run(rows, split='validation', name='baseline-새 기준')
    assert harness.client.request_with_retries.called


def test_attachment_outside_the_dataset_stops_the_run(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    (tmp_path / 'secret').write_text('key')
    monkeypatch.setattr(baseline_suite, 'DATA', data)
    with pytest.raises(AssertionError, match='attachment_outside_dataset'):
        baseline_suite.assets_digest('stt', [{'attachments': {'audio': '../secret'}, 'metadata': {'id': 'S-1'}}])


def test_link_to_a_file_outside_the_dataset_stops_the_run(tmp_path, monkeypatch):
    data = tmp_path / 'data'
    data.mkdir()
    (tmp_path / 'secret').write_text('key')
    (data / 'a.wav').symlink_to(tmp_path / 'secret')
    monkeypatch.setattr(baseline_suite, 'DATA', data)
    with pytest.raises(AssertionError, match='attachment_outside_dataset'):
        baseline_suite.assets_digest('stt', [{'attachments': {'audio': 'a.wav'}, 'metadata': {'id': 'S-1'}}])


@pytest.mark.parametrize('value,expected', [(None, False), ('0', False), ('1', True)])
def test_workflow_switch_is_recorded_as_the_service_reads_it(monkeypatch, value, expected):
    monkeypatch.delenv('AGENTIC_CHAT', raising=False)
    if value is not None:
        monkeypatch.setenv('AGENTIC_CHAT', value)
    assert baseline_suite.agentic_chat() is expected


def test_resume_refuses_a_changed_workflow():
    with pytest.raises(AssertionError, match='agentic_chat'):
        baseline_suite.assert_resumable({'agentic_chat': False}, {'agentic_chat': True})
