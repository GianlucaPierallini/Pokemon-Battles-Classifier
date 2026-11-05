"""Predict Kaggle's 5,000 test battles using SAVED native model artifacts only.

No training, cross-validation, test-based model selection, external Pokémon
information, or training JSONL file is used in this script.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.metadata
import json
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import lightgbm as lgb

from pokemon_features import read_jsonl, load_type_chart, make_features


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_saved_models(artifacts_dir):
    artifacts_dir = Path(artifacts_dir)
    cat_path = artifacts_dir / 'catboost_model.cbm'
    lgb_path = artifacts_dir / 'lightgbm_model.txt'
    metadata_path = artifacts_dir / 'preprocessing.json'
    for path in (cat_path, lgb_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(
                f'Missing saved artifact: {path}. Run the ONE-TIME export '
                'notebook using Kaggle train data, then upload its three '
                'artifact files into CATBoostLightGBM/artifacts/ on GitHub.'
            )
    with metadata_path.open(encoding='utf-8') as stream:
        metadata = json.load(stream)
    if metadata.get('schema_version') != 1:
        raise ValueError('Unrecognized preprocessing.json schema version.')
    if _sha256(cat_path) != metadata['catboost_model_sha256']:
        raise ValueError('CatBoost artifact hash mismatch: do not run outdated/changed artifacts.')
    if _sha256(lgb_path) != metadata['lightgbm_model_sha256']:
        raise ValueError('LightGBM artifact hash mismatch: do not run outdated/changed artifacts.')
    feature_source = Path(__file__).with_name('pokemon_features.py')
    if _sha256(feature_source) != metadata['original_feature_module_sha256']:
        raise ValueError('pokemon_features.py differs from the code used to train these saved models.')

    for pkg in ('catboost', 'lightgbm'):
        current = importlib.metadata.version(pkg)
        expected = metadata.get('versions', {}).get(pkg)
        if expected and current != expected:
            warnings.warn(f'{pkg} version {current} differs from trained-model version {expected}; '
                          'confirm compatibility and compare outputs.', stacklevel=2)

    cat = CatBoostClassifier()
    cat.load_model(str(cat_path))
    lgb_model = lgb.Booster(model_file=str(lgb_path))
    return cat, lgb_model, metadata


def generate_submission(
    competition_dir='/kaggle/input/competitions/fds-pokemon-battles-prediction-2026',
    artifacts_dir=None,
    output_path='/kaggle/working/submission_blend_6848.csv',
):
    """Reproduce the original 65/35 blend WITHOUT loading train.jsonl.

    All train-derived features and LightGBM category levels are restored from
    preprocessing.json. Test data and the supplied type chart remain on Kaggle.
    """
    competition_dir = Path(competition_dir)
    if artifacts_dir is None:
        artifacts_dir = Path(__file__).parent / 'artifacts'
    artifacts_dir = Path(artifacts_dir)
    cat, lgb_model, metadata = load_saved_models(artifacts_dir)
    test_path = competition_dir / 'test.jsonl'
    chart_path = competition_dir / 'type_chart_gen2.csv'
    sample_path = competition_dir / 'sample_submission.csv'
    for path in (test_path, chart_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(f'Competition file not found: {path}')
    if _sha256(chart_path) != metadata['official_type_chart_sha256']:
        raise ValueError('Gen 2 type chart differs from the training-time chart.')

    test_records = read_jsonl(test_path)
    if len(test_records) != 5000:
        raise ValueError(f'Expected 5000 test battles; got {len(test_records)}')
    if any('player_won' in r for r in test_records):
        raise ValueError('Test records unexpectedly contain training labels.')
    chart = load_type_chart(chart_path)
    if not chart:
        raise ValueError('Unable to parse the supplied Generation 2 type chart.')

    # No reconstruction of vocabulary or species stats from the test set.
    test_features = make_features(
        test_records, metadata['species_lookup'], metadata['vocab'], chart,
    )
    feature_cols = metadata['feature_columns']
    if len(feature_cols) != 714 or len(metadata['categorical_columns']) != 13:
        raise ValueError('Unexpected feature schema; refusing to run inference.')
    # Original train_features.align(test_features, join="left", fill_value=0)
    # has precisely the same column order and fill behavior as reindex here.
    X_test = test_features.reindex(columns=feature_cols, fill_value=0).copy()
    cat_cols = metadata['categorical_columns']
    num_cols = X_test.columns.difference(cat_cols)
    for col in cat_cols:
        X_test[col] = X_test[col].fillna('unknown').astype(str)
    for col in num_cols:
        X_test[col] = X_test[col].astype('float32')
    assert X_test.columns.tolist() == feature_cols

    X_test_lgb = X_test.copy()
    for col in cat_cols:
        X_test_lgb[col] = pd.Categorical(
            X_test[col], categories=metadata['lightgbm_categories'][col]
        )

    p_cat = cat.predict_proba(X_test)[:, 1]
    p_lgb = lgb_model.predict(X_test_lgb)
    weights = metadata['blend_weights']
    p_blend = weights['catboost'] * p_cat + weights['lightgbm'] * p_lgb
    binary_prediction = (p_blend >= metadata['classification_threshold']).astype(int)

    test_ids = pd.Series([r['battle_id'] for r in test_records])
    sample = pd.read_csv(sample_path)
    if (test_ids.duplicated().any() or sample['battle_id'].duplicated().any()
            or sample.columns.tolist() != ['battle_id', 'player_won']):
        raise ValueError('Duplicate battle_id or unexpected submission CSV header.')
    by_id = pd.Series(binary_prediction, index=test_ids.to_numpy())
    submission = sample.copy()
    submission['player_won'] = submission['battle_id'].map(by_id)
    if submission['player_won'].isna().any():
        raise ValueError('Some template battle IDs are missing from test prediction IDs.')
    submission['player_won'] = submission['player_won'].astype(int)
    assert submission.shape == (5000, 2)
    assert submission['player_won'].isin([0, 1]).all()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    print('Inference-only mode: no train.jsonl was loaded, and NO models were fitted.')
    print('Loaded saved rounds:', metadata['final_cat_rounds'], metadata['final_lgb_rounds'])
    print('Feature matrix:', X_test.shape)
    print('Submission saved to:', output_path)
    return submission
