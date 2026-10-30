"""Small end-to-end tests; budgets here are not research results."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest
import torch

from experiments.data import load_bank, load_cache
from experiments import pipeline
from experiments import target_encoders
from experiments.data_prep import prepare_raw
from experiments.pipeline import _initial_checkpoint, _settings, _snapshot_config, run
from jets.models import JETSGenerator


def _inputs(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    pd.DataFrame({'account_id': [11, 22], 'Country': [0, 1]}).to_csv(raw / 'account.csv', index=False)
    pd.DataFrame({'trans_id': [0, 1, 2, 3], 'account_id': [11, 11, 22, 22],
                  't_offset': [1., 3., 2., 4.], 'log_amount': [1., 0., 2., 3.],
                  'log_cumamt': [1., 1., 2., 4.]}).to_csv(raw / 'trans.csv', index=False)
    config = {'training_seed': 9, 'generator_steps': 2, 'batch_size': 2,
              'learning_rate': .0003, 'weight_decay': .01, 'gradient_clip': 1.,
              'code_noise_std': .03, 'log_interval': 1,
              'model': {'code_dim': 8, 'width': 16, 'layers': 1, 'heads': 2,
                        'hidden': 16, 'components': 2, 'diffusion_steps': 10},
              'target_encoder': {'steps': 2, 'batch_size': 2, 'cache_batch_size': 4,
                                 'log_interval': 1, 'width': 16, 'layers': 1,
                                 'heads': 2, 'decoder_layers': 1}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    return raw, path, config


def test_pipeline_builds_reusable_targets_and_rejects_changes(tmp_path):
    raw, config_path, config = _inputs(tmp_path)
    out = tmp_path / 'result'
    result = run(raw, out, config_path, windows=4)
    assert result['target_steps'] == 2 and result['exploratory_budget']
    assert not result['generation_future_target_inputs']
    snapshot = Path(result['artifacts']['config_snapshot']['path'])
    assert snapshot.read_bytes() == config_path.read_bytes()
    args = result['generator_train_command']
    assert args[args.index('--config') + 1] == str(snapshot)
    bank_path, bank = load_bank(Path(result['artifacts']['bank']['path']))
    assert len(bank['windows']) == 4
    for family in ('time', 'feature'):
        entry = result['artifacts']
        checkpoint = Path(entry[f'{family}_target_encoder']['path'])
        cache = Path(entry[f'{family}_cache']['path'])
        codes = load_cache(cache, bank_path, bank['windows'], config['model']['code_dim'], checkpoint)
        assert all(code.shape == (int(window['count']) + 1, 8)
                   for code, window in zip(codes, bank['windows']))
    saved = torch.load(result['artifacts']['initial_generator']['path'], map_location='cpu', weights_only=False)
    state = bank['state']
    generator = JETSGenerator(state['token_dim'], bank['attrs'].shape[1],
                              state['mark_token_state']['observation_end'],
                              time_scale=state['time_scale'], **config['model'])
    generator.load_state_dict(saved['model'], strict=True)
    with pytest.raises(TypeError):
        generator.sample_paths(bank['attrs'][:1], torch.zeros(1), torch.ones(1),
                               teacher_codes=torch.zeros(1, 8))
    assert run(raw, out, config_path, windows=4) == result
    # The documented hand-off really starts a new generator run.  One update is
    # explicitly a functional smoke test, not the configured research budget.
    trained = subprocess.run(result['generator_train_command'] + ['--steps', '1'],
                             capture_output=True, text=True, check=True)
    assert '"step": 1' in trained.stdout
    assert (out / 'generator' / 'final.pt').is_file()
    with (raw / 'account.csv').open('a', encoding='utf-8') as stream:
        stream.write('\n')
    with pytest.raises(ValueError, match='different inputs'):
        run(raw, out, config_path, windows=4)
    assert (out / 'complete.json').exists()


def test_pipeline_rejects_nonempty_or_raw_output(tmp_path):
    raw, config_path, _ = _inputs(tmp_path)
    out = tmp_path / 'occupied'
    out.mkdir()
    (out / 'existing.txt').write_text('preserve', encoding='utf-8')
    with pytest.raises(FileExistsError, match='not empty'):
        run(raw, out, config_path, windows=4)
    assert (out / 'existing.txt').read_text(encoding='utf-8') == 'preserve'
    assert not (out / '.jets-preparation.lock').exists()
    with pytest.raises(ValueError, match='outside'):
        run(raw, raw / 'out', config_path, windows=4)
    with pytest.raises(ValueError, match='must not contain the source'):
        run(raw, tmp_path, config_path, windows=4)
    assert not (raw / '.jets-preparation.lock').exists()
    assert not (tmp_path / '.jets-preparation.lock').exists()


def test_pipeline_lock_rejects_concurrent_owner_and_releases_on_interruption(tmp_path, monkeypatch):
    raw, config_path, _ = _inputs(tmp_path)
    out = tmp_path / 'locked'
    code = ('from pathlib import Path\n'
            'from experiments.data_prep import _prepare_lock\n'
            'import sys\n'
            'with _prepare_lock(Path(sys.argv[1])):\n'
            ' print("locked", flush=True)\n'
            ' sys.stdin.readline()\n')
    owner = subprocess.Popen([sys.executable, '-u', '-c', code, str(out)],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True)
    try:
        assert owner.stdout is not None
        assert owner.stdout.readline().strip() == 'locked'
        with pytest.raises(RuntimeError, match='Another preparation'):
            run(raw, out, config_path, windows=4, target_steps=1)
        assert not (out / 'pipeline.json').exists()
    finally:
        owner.terminate()
        owner.wait(timeout=20)
        if owner.stdin is not None:
            owner.stdin.close()
        if owner.stdout is not None:
            owner.stdout.close()
        if owner.stderr is not None:
            owner.stderr.close()
    # An abruptly killed owner must not require manual stale-lock deletion.
    # Stub long-running stages so this checks lock release independently of
    # target training and any other process editing the source tree.
    monkeypatch.setattr(pipeline, '_run_locked', lambda *args, **kwargs: {'lock_reacquired': True})
    assert run(raw, out, config_path, windows=4, target_steps=1) == {'lock_reacquired': True}


def test_pipeline_module_exposes_cli():
    result = subprocess.run([sys.executable, '-m', 'experiments.pipeline', '--help'],
                            capture_output=True, text=True, check=True)
    assert '--raw-dir' in result.stdout and '--target-steps' in result.stdout


def test_target_budget_and_dimensions_are_explicit(tmp_path):
    _, _, config = _inputs(tmp_path)
    settings = _settings(config, target_steps=1)
    assert settings['teacher_steps'] == 1
    assert settings['teacher']['code_dim'] == config['model']['code_dim']
    config['target_encoder']['typo'] = 1
    with pytest.raises(ValueError, match='Unknown target_encoder'):
        _settings(config)


def test_config_snapshot_recovers_after_process_interruption_and_refuses_tampering(tmp_path):
    source = tmp_path / 'source.json'
    content = b'{"seed": 9, "note": "exact bytes"}\n'
    source.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    snapshot = tmp_path / 'snapshot.json'
    # Terminate after the fully flushed candidate is ready but before the
    # exclusive link; retry must not overwrite or remove its orphaned temp.
    code = ('import os,sys\n'
            'from pathlib import Path\n'
            'from experiments.pipeline import _snapshot_config\n'
            'os.link=lambda *args: os._exit(73)\n'
            '_snapshot_config(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])\n')
    interrupted = subprocess.run([sys.executable, '-c', code, str(source), str(snapshot), digest])
    assert interrupted.returncode == 73 and not snapshot.exists()
    orphans = list(tmp_path.glob('.snapshot.json.*.tmp'))
    assert len(orphans) == 1 and orphans[0].read_bytes() == content
    assert _snapshot_config(source, snapshot, digest) == snapshot
    assert snapshot.read_bytes() == content and orphans[0].read_bytes() == content
    snapshot.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='snapshot does not match'):
        _snapshot_config(source, snapshot, digest)
    assert snapshot.read_bytes() == b'tampered' and source.read_bytes() == content


def test_pipeline_detects_config_changed_mid_run(tmp_path, monkeypatch):
    raw, config_path, _ = _inputs(tmp_path)
    original = config_path.read_bytes()
    initialize = pipeline._initial_checkpoint

    def change_config_after_initial(*args, **kwargs):
        result = initialize(*args, **kwargs)
        config_path.write_bytes(original + b'\n')
        return result

    monkeypatch.setattr(pipeline, '_initial_checkpoint', change_config_after_initial)
    out = tmp_path / 'config_changed'
    with pytest.raises(ValueError, match='Config or config snapshot changed'):
        run(raw, out, config_path, windows=4, target_steps=1)
    assert config_path.read_bytes() == original + b'\n'
    assert (out / 'config.snapshot.json').read_bytes() == original
    assert not (out / 'complete.json').exists()


@pytest.mark.parametrize('tamper', [False, True])
def test_initial_generator_interrupted_publication_recovery(tmp_path, monkeypatch, tamper):
    raw, _, config = _inputs(tmp_path)
    out = tmp_path / 'artifacts'
    prepare_raw(raw, out / 'input', windows=4)
    bank_path, bank = load_bank(out / 'input' / 'bank')
    checkpoint = out / 'initial_generator.pt'
    publish = target_encoders._publish_json_exclusive

    def interrupt_receipt(path, value):
        if path.name == 'initial_generator.json':
            raise OSError('simulated interruption')
        return publish(path, value)

    monkeypatch.setattr(target_encoders, '_publish_json_exclusive', interrupt_receipt)
    with pytest.raises(OSError, match='simulated interruption'):
        _initial_checkpoint(bank, bank_path, config, checkpoint)
    assert checkpoint.is_file() and not checkpoint.with_suffix('.json').exists()
    assert (out / '.initial_generator.pt.journal.json').is_file()
    if tamper:
        with checkpoint.open('ab') as stream:
            stream.write(b'tampered')
    existing_bytes = checkpoint.read_bytes()
    monkeypatch.setattr(target_encoders, '_publish_json_exclusive', publish)
    if tamper:
        with pytest.raises(ValueError, match='differs|digest mismatch'):
            _initial_checkpoint(bank, bank_path, config, checkpoint)
        assert checkpoint.read_bytes() == existing_bytes
        assert not checkpoint.with_suffix('.json').exists()
    else:
        assert _initial_checkpoint(bank, bank_path, config, checkpoint) == checkpoint
        assert checkpoint.read_bytes() == existing_bytes
        assert checkpoint.with_suffix('.json').is_file()
        assert not (out / '.initial_generator.pt.journal.json').exists()


def test_explicit_retail_workbook_pipeline_is_resumable(tmp_path):
    pytest.importorskip('openpyxl')
    _, config_path, _ = _inputs(tmp_path)
    rows = pd.DataFrame([{'Invoice': str(i), 'InvoiceDate': pd.Timestamp('2010-01-01') + pd.Timedelta(days=i),
                          'Quantity': -1 if i == 0 else 1, 'Price': 2.,
                          'Customer ID': 123, 'Country': 'UK'} for i in range(21)])
    workbook = tmp_path / 'retail_ii.xlsx'
    with pd.ExcelWriter(workbook, engine='openpyxl') as writer:
        rows.iloc[:11].to_excel(writer, sheet_name='Year 2009-2010', index=False)
        rows.iloc[11:].to_excel(writer, sheet_name='Year 2010-2011', index=False)
    out = tmp_path / 'workbook_result'
    result = run(None, out, config_path, windows=2, target_steps=1, retail_ii_xlsx=workbook)
    assert 'negative total clipping' in result['input_kind']
    assert result['target_steps'] == 1 and result['exploratory_budget']
    assert (out / 'source_tables/source_manifest.json').is_file()
    assert run(None, out, config_path, windows=2, target_steps=1, retail_ii_xlsx=workbook) == result
    with pytest.raises(ValueError, match='exactly one'):
        run(tmp_path / 'raw', tmp_path / 'another', config_path, retail_ii_xlsx=workbook)
    blocked = tmp_path / 'unsupported_split'
    with pytest.raises(ValueError, match='pre-split account.csv/trans.csv'):
        run(None, blocked, config_path, windows=2, target_steps=1,
            train_ids=[0], retail_ii_xlsx=workbook)
    assert not blocked.exists()
    split = tmp_path / 'train_ids.json'
    split.write_text('[0]', encoding='utf-8')
    cli = subprocess.run([sys.executable, '-m', 'experiments.pipeline',
                          '--retail-ii-xlsx', str(workbook), '--train-ids', str(split),
                          '--config', str(config_path), '--output', str(blocked)],
                         capture_output=True, text=True)
    assert cli.returncode != 0 and 'requires pre-split' in cli.stderr
    assert not blocked.exists()
