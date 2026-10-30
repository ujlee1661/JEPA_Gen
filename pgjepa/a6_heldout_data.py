"""Entity-heldout A6 data with every fitted calibration restricted to train.

The benchmark's administrative observation end is known to both splits. Held-out
marks, histories, and future targets are never passed to any fit operation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .a6_data import prepare_a6, encode_window
from .experiment import digest, read_json, write_json
from .paper_data import load_paper_raw, sha256
from .quantile import QuantilePreprocessor


def _plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    if isinstance(value,dict):
        return {str(key):_plain(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):
        return [_plain(item) for item in value]
    return value


def calibration_digest(value):
    return digest(_plain(value))


def _transform(prep, entities):
    result=[]
    for entity in entities:
        record=prep.transform(entity)
        record['calendar_times']=torch.as_tensor(entity['calendar_times'],dtype=torch.float64)
        result.append(record)
    return result


def prepare_a6_heldout(raw_root, output, *, split_seed=73, bank_seed=9,
                       windows=4096, train_entities=335, enforce_paper_counts=True):
    """Prepare335/82 split and738 test queries without held-out calibration.

    For small test fixtures pass explicit train_entities and disable the original
    paper population count assertion; scientific defaults retain that assertion.
    """
    output=Path(output)
    entities,provenance=load_paper_raw(raw_root,'online_retail',
                                     enforce_paper_counts=enforce_paper_counts)
    source_paths={name:Path(raw_root)/'online_retail/raw'/name
                  for name in provenance['source_files']}
    if not 1 <= train_entities < len(entities):
        raise ValueError('Both train and held-out entity sets must be nonempty')
    permutation=np.random.default_rng(split_seed).permutation(len(entities))
    train_indices=sorted(int(i) for i in permutation[:train_entities])
    heldout_indices=sorted(int(i) for i in permutation[train_entities:])
    train_raw=[entities[i] for i in train_indices]
    test_raw=[entities[i] for i in heldout_indices]
    train_ids=[int(entity['entity_id']) for entity in train_raw]
    test_ids=[int(entity['entity_id']) for entity in test_raw]
    assert set(train_ids).isdisjoint(test_ids)
    identity={'version':'a6_entity_heldout_v1','split_seed':int(split_seed),'bank_seed':int(bank_seed),
        'train_entity_ids':train_ids,'heldout_entity_ids':test_ids,'window_bank_size':int(windows),
        'source_files':provenance['source_files'],
        'administrative_observation_end':provenance['observation_end']}
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/'split_manifest.json'
    if manifest_path.exists():
        saved=read_json(manifest_path)
        if saved['identity'] != identity:
            raise ValueError('Held-out split identity changed; use a new output directory')
        for relative,expected in saved['artifact_sha256'].items():
            if sha256(output/relative) != expected:
                raise ValueError(f'Held-out preparation artifact changed: {relative}')
        return saved
    prep=QuantilePreprocessor(max_categories=4096,seed=bank_seed,
                             quantile_grid='all_observed').fit(train_raw)
    fitted_preprocessor_hash=digest(prep.state_dict())
    train_records=_transform(prep,train_raw)
    test_records=_transform(prep,test_raw)
    if digest(prep.state_dict()) != fitted_preprocessor_hash:
        raise RuntimeError('Transforming held-out entities changed fitted preprocessing')
    train_dir=output/'train_prepared'
    train_dir.mkdir(exist_ok=True)
    torch.save(train_records,train_dir/'population.pt')
    write_json(train_dir/'preprocessor.json',prep.state_dict())
    write_json(train_dir/'parents.json',train_ids)
    train_source={**provenance,'entities':len(train_raw),
        'events':sum(len(entity['delta']) for entity in train_raw),
        'scope':'entity-heldout study: this population contains training entities only',
        'split':'train','entity_ids_digest':digest(train_ids)}
    train_manifest={'source':train_source,'population_sha256':sha256(train_dir/'population.pt'),
        'preprocessor_hash':fitted_preprocessor_hash,'parent_hash':digest(train_ids),
        'calibration_fit_entity_ids_digest':digest(train_ids),
        'static_types':{'numeric':provenance['schema']['parent_numeric'],
                        'categorical':provenance['schema']['parent_categorical']}}
    write_json(train_dir/'manifest.json',train_manifest)
    train_a6=prepare_a6(train_dir,output/'train_a6',windows=windows,seed=bank_seed)
    token_hash=calibration_digest(train_a6['token_state'])
    profile_hash=calibration_digest(train_a6['profile_state'])
    queries=[]
    for test_index,(entity,record) in enumerate(zip(test_raw,test_records)):
        count=len(record['delta'])
        for prefix_fraction in (.25,.5,.75):
            cut=int(np.floor(count*prefix_fraction))
            anchor=float(record['calendar_times'][cut-1]) if cut else 0.
            remaining=float(record['exposure_end'])-anchor
            for horizon_fraction in (1.,.25,.5):
                item=encode_window(record,cut,remaining*horizon_fraction,
                    train_a6['token_state'],train_a6['profile_state'])
                item.update(entity=test_index,entity_id=int(entity['entity_id']),
                    prefix_fraction=prefix_fraction,horizon_fraction=horizon_fraction,
                    split='heldout',target_use='evaluation only')
                # Explicit raw evaluation target, never supplied to the predictor.
                times=record['calendar_times'].numpy()
                finish=max(cut,int(np.searchsorted(times,anchor+item['delta'],side='right')))
                item['target_calendar_times']=record['calendar_times'][cut:finish].clone()
                item['target_raw_numeric']=torch.as_tensor(entity['x_num'][cut:finish],dtype=torch.float64)
                queries.append(item)
    if token_hash != calibration_digest(train_a6['token_state']) or profile_hash != calibration_digest(train_a6['profile_state']):
        raise RuntimeError('Held-out encoding changed fitted token/profile calibration')
    test_bundle={'version':'a6_heldout_queries_v1','records':test_records,
        'entity_ids':test_ids,'attrs':torch.stack([r['attrs'] for r in test_records]),
        'queries':queries,'token_state_digest':token_hash,'profile_state_digest':profile_hash,
        'preprocessor_hash':fitted_preprocessor_hash,'fit_split':'train only',
        'evaluation_unit':'entity, not independent overlapping query windows'}
    torch.save(test_bundle,output/'heldout_queries.pt')
    if any(sha256(path)!=provenance['source_files'][name] for name,path in source_paths.items()):
        raise RuntimeError('A protected raw source file changed during preparation')
    artifact_names=['train_prepared/population.pt','train_prepared/preprocessor.json',
        'train_prepared/parents.json','train_prepared/manifest.json','train_a6/data.pt',
        'train_a6/manifest.json','heldout_queries.pt']
    manifest:dict[str,Any]={'identity':identity,'train_entities':len(train_ids),
        'heldout_entities':len(test_ids),'heldout_queries':len(queries),
        'calibration_fit_split':'train entities only','preprocessor_hash':fitted_preprocessor_hash,
        'token_state_digest':token_hash,'profile_state_digest':profile_hash,
        'protected_raw_source_sha256':provenance['source_files'],
        'entity_split_digest':digest({'train':train_ids,'heldout':test_ids}),
        'artifact_sha256':{name:sha256(output/name) for name in artifact_names},
        'scope':'same-domain entity-heldout conditional future generation; not full-population fidelity',
        'administrative_end_policy':'same known source benchmark exposure end for both splits; not a learned feature calibration'}
    write_json(manifest_path,manifest)
    return manifest
