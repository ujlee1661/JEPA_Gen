"""Standalone tests for prepared-artifact boundaries and training recovery."""
from argparse import Namespace
import json
import math

import pytest
import torch

from experiments import data as artifact
from experiments import retail
from jets.models import JETSGenerator


def test_category_export_supports_numeric_and_string_vocabularies():
    assert retail.decode_category(1, ['12']) == 12.0
    assert retail.decode_category(2, ['12', 'purchase']) == 'purchase'
    assert math.isnan(retail.decode_category(0, ['12']))
    assert math.isnan(retail.decode_category(1, ['12'], observed=False))


def test_evaluation_preserves_string_identifiers(tmp_path):
    real = tmp_path / 'real.csv'
    generated = tmp_path / 'generated.csv'
    real.write_text('account_id,value\n001,2\nNA,3\n', encoding='utf-8')
    generated.write_text('entity,value\n001,2\nNA,3\n', encoding='utf-8')
    assert retail.read_evaluation_table(real, 'account_id', 'string')['account_id'].tolist() == ['001', 'NA']
    assert retail.read_evaluation_table(generated, 'entity', 'string')['entity'].tolist() == ['001', 'NA']


def test_generation_artifact_links_and_legacy_status(tmp_path):
    bank = tmp_path / 'data.pt'
    bank.write_bytes(b'prepared bank')
    preprocessor, parents = {'kind': 'example'}, [1, 2]
    checkpoint = {'identity': {'input_sha256': {str(bank): artifact.sha256(bank)}}}
    legacy = retail.validate_generation_inputs(bank, checkpoint, preprocessor, parents)
    assert legacy == {'checkpoint_bank_verified': True, 'bank_preprocessor_verified': False,
                      'bank_parents_verified': False}
    artifact.write_json(tmp_path / 'manifest.json', {'preprocessor_hash': artifact.json_digest(preprocessor),
                                                      'parent_hash': artifact.json_digest(parents)})
    checked = retail.validate_generation_inputs(bank, checkpoint, preprocessor, parents)
    assert all(checked.values())
    with pytest.raises(ValueError, match='training inputs'):
        retail.validate_generation_inputs(bank, {'sources': {'old': 'bad'}}, preprocessor, parents)
    with pytest.raises(ValueError, match='preprocessor_hash'):
        retail.validate_generation_inputs(bank, checkpoint, {'kind': 'other'}, parents)
    with pytest.raises(ValueError, match='parent_hash'):
        retail.validate_generation_inputs(bank, checkpoint, preprocessor, [2, 1])


@pytest.fixture
def prepared(tmp_path):
    model_cfg = {'code_dim': 8, 'width': 16, 'layers': 1, 'heads': 2,
                 'hidden': 16, 'components': 2, 'diffusion_steps': 10}
    config = {'training_seed': 9, 'generator_steps': 2, 'batch_size': 2,
              'learning_rate': .0003, 'weight_decay': .01, 'gradient_clip': 1.,
              'code_noise_std': .03, 'log_interval': 1, 'model': model_cfg}
    window = {'tokens': torch.tensor([[.1, .2, .3, 1., 1.], [-.1, 0., 0., 0., 0.]]),
              'count': 1, 'event_valid': torch.tensor([True, False]),
              'active': torch.tensor([True, True]), 'attrs': torch.ones(2),
              'anchor': 0., 'delta': 1., 'past_tokens': torch.empty(0, 5)}
    bank_data = {'windows': [window], 'attrs': torch.ones(1, 2),
                 'state': {'token_dim': 5, 'time_scale': 1.,
                           'mark_token_state': {'observation_end': 10.}}}
    bank = tmp_path/'data.pt'; torch.save(bank_data, bank)
    cfg = tmp_path/'config.json'; cfg.write_text(json.dumps(config), encoding='utf-8')
    teacher = tmp_path/'teacher.pt'; torch.save({'model': {}}, teacher)
    cache = tmp_path/'cache.pt'
    torch.save({'data_sha256': artifact.sha256(bank),
                'teacher_checkpoint_sha256': artifact.sha256(teacher),
                'codes': [torch.randn(2, 8)]}, cache)
    initial = tmp_path/'initial.pt'
    torch.manual_seed(19)
    torch.save({'model': JETSGenerator(5, 2, 10., **model_cfg).state_dict()}, initial)
    return Namespace(bank=bank, config=cfg, time_cache=cache, time_teacher=teacher,
                     feature_cache=cache, feature_teacher=teacher, initial=initial,
                     output=tmp_path/'result', steps=None, feature_weight=1., device='cpu')


def test_cache_rejects_wrong_bank_teacher_and_nonfinite(prepared, tmp_path):
    _, bank = artifact.load_bank(prepared.bank)
    original = torch.load(prepared.time_cache, weights_only=False)
    for field, value, message in [
        ('data_sha256', 'bad', 'does not match bank'),
        ('teacher_checkpoint_sha256', 'bad', 'does not match teacher'),
        ('codes', [torch.full((2, 8), float('nan'))], 'incorrect suffix'),
        ('codes', [torch.zeros(1, 8)], 'incorrect suffix'),
    ]:
        value_path = tmp_path/f'{field}_{len(message)}.pt'
        torch.save({**original, field: value}, value_path)
        with pytest.raises(ValueError, match=message):
            artifact.load_cache(value_path, prepared.bank, bank['windows'], 8, prepared.time_teacher)


def test_two_step_interrupted_resume_matches_continuous(prepared, tmp_path, monkeypatch):
    original_save = retail.atomic_torch_save
    once = {'interrupted': False}
    def save_then_interrupt(value, path):
        original_save(value, path)
        if path.name == 'latest.pt' and value['step'] == 1 and not once['interrupted']:
            once['interrupted'] = True
            raise RuntimeError('simulated interruption after atomic save')
    monkeypatch.setattr(retail, 'atomic_torch_save', save_then_interrupt)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        retail.train(prepared)
    assert (prepared.output/'latest.pt').is_file()
    monkeypatch.setattr(retail, 'atomic_torch_save', original_save)
    retail.train(prepared)
    resumed = torch.load(prepared.output/'final.pt', weights_only=False)
    prepared.output = tmp_path/'continuous'
    retail.train(prepared)
    continuous = torch.load(prepared.output/'final.pt', weights_only=False)
    assert resumed['step'] == continuous['step'] == 2
    assert resumed['model'].keys() == continuous['model'].keys()
    assert all(torch.equal(resumed['model'][key], continuous['model'][key]) for key in resumed['model'])
    with pytest.raises(ValueError, match='Completed output exists'):
        retail.train(prepared)


def test_generate_refuses_to_overwrite(prepared, tmp_path, monkeypatch):
    config = artifact.read_json(prepared.config)
    model = JETSGenerator(5, 2, 10., **config['model'])
    checkpoint = tmp_path/'model.pt'
    torch.save({'model': model.state_dict(), 'config': config, 'step': 2}, checkpoint)
    preprocessor = tmp_path/'preprocessor.json'
    preprocessor.write_text(json.dumps({'kind':'tabsyn_gaussian_quantile_v1'}), encoding='utf-8')
    parents = tmp_path/'parents.json'; parents.write_text('[7]', encoding='utf-8')
    out = tmp_path/'existing'; out.mkdir(); existing = out/'child.csv'; existing.write_text('keep\n', encoding='utf-8')
    monkeypatch.setattr(retail.QuantilePreprocessor, 'from_state_dict', lambda state: object())
    args = Namespace(bank=prepared.bank, checkpoint=checkpoint, preprocessor=preprocessor,
                     parents=parents, output=out, device='cpu')
    with pytest.raises(ValueError, match='Output files already exist'):
        retail.generate(args)
    assert existing.read_text(encoding='utf-8') == 'keep\n'
