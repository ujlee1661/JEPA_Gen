"""Orchestration tests use stubs; they are not research runs."""
from pathlib import Path

import pytest

from experiments import multidata
from experiments.data import read_json, sha256, write_json


def test_full_matrix_counts_and_resume_without_training(tmp_path, monkeypatch):
    raw = tmp_path / 'raw'
    for dataset in multidata.DATASETS:
        folder = raw / dataset / 'raw'
        folder.mkdir(parents=True)
        (folder / 'account.csv').write_text('account_id\n0\n')
        (folder / 'trans.csv').write_text('account_id,t_offset\n0,1\n')
    scorer = tmp_path / 'score.py'
    scorer.write_text('# test scorer')
    config = tmp_path / 'config.json'
    write_json(config, {'generator_steps': 20000, 'batch_size': 16,
                        'target_encoder': {'steps': 5000, 'batch_size': 32}})
    monkeypatch.setattr(multidata, 'SCORER_SHA256', sha256(scorer))
    monkeypatch.setattr(multidata, 'source_hashes', lambda: {'test': 'stable'})
    calls = []

    def run_stub(argv, logfile, cwd):
        values = [str(v) for v in argv]
        if 'experiments.pipeline' in values:
            calls.append('pipeline')
            out = Path(values[values.index('--output') + 1]); out.mkdir(parents=True, exist_ok=True)
            assets = {}
            for name in ('bank', 'preprocessor', 'parents'):
                path = out / f'{name}.pt'; path.write_text(name)
                assets[name] = {'path': str(path)}
            write_json(out/'complete.json', {'exploratory_budget': False, 'target_steps': 5000,
                'artifacts': assets, 'generator_train_command': ['stub_train', str(out)]})
        elif values[0] == 'stub_train':
            calls.append('train')
            folder = Path(values[1])/'generator'; folder.mkdir(parents=True, exist_ok=True)
            model = folder/'final.pt'; model.write_bytes(b'model')
            write_json(folder/'complete.json', {'updates': 20000, 'checkpoint_sha256': sha256(model)})
        elif 'generate' in values:
            calls.append('generate')
            folder = Path(values[values.index('--output')+1]); folder.mkdir(parents=True, exist_ok=True)
            child = folder/'child.csv'; child.write_text('entity,t_offset\n0,1\n')
            checkpoint = Path(values[values.index('--checkpoint')+1])
            seed = int(values[values.index('--seed')+1])
            assert '--count' not in values and '--time' not in values
            write_json(folder/'generation.json', {'child_sha256': sha256(child),
                'checkpoint_sha256': sha256(checkpoint), 'generation_seed': seed,
                'count_or_face_or_teacher_oracle': False})
        else:
            raise AssertionError(values)

    def score_stub(real, generated, evaluator, dataset, dest, cwd):
        calls.append('score')
        multidata._immutable_json(dest, {'dataset': dataset, 'child_sha256': sha256(generated)})

    monkeypatch.setattr(multidata, '_run', run_stub)
    monkeypatch.setattr(multidata, '_score', score_stub)
    out = tmp_path/'study'
    multidata.run(raw, scorer, out, config, device='cpu')
    assert calls.count('pipeline') == 12
    assert calls.count('train') == 12
    assert calls.count('generate') == calls.count('score') == 36
    assert read_json(out/'complete.json')['native_generations'] == 36
    calls.clear()
    multidata.run(raw, scorer, out, config, device='cpu')
    assert calls.count('pipeline') == 12
    assert calls.count('train') == calls.count('generate') == 0
    assert calls.count('score') == 36
    (raw/'berka/raw/trans.csv').write_text('changed')
    with pytest.raises(ValueError, match='Existing file differs'):
        multidata.run(raw, scorer, out, config, device='cpu')


def test_existing_user_output_is_never_overwritten(tmp_path, monkeypatch):
    raw = tmp_path/'raw'
    for dataset in multidata.DATASETS:
        folder = raw/dataset/'raw'; folder.mkdir(parents=True)
        (folder/'account.csv').write_text('account_id\n0\n')
        (folder/'trans.csv').write_text('account_id,t_offset\n0,1\n')
    scorer = tmp_path/'score.py'; scorer.write_text('x=1')
    monkeypatch.setattr(multidata, 'SCORER_SHA256', sha256(scorer))
    monkeypatch.setattr(multidata, 'source_hashes', lambda: {})
    config = tmp_path/'config.json'
    write_json(config, {'generator_steps': 20000, 'batch_size': 16,
                        'target_encoder': {'steps': 5000, 'batch_size': 32}})
    output = tmp_path/'result'; output.mkdir(); marker = output/'keep.txt'; marker.write_text('safe')
    with pytest.raises(FileExistsError, match='Non-empty output'):
        multidata.run(raw, scorer, output, config, device='cpu')
    assert marker.read_text() == 'safe'
