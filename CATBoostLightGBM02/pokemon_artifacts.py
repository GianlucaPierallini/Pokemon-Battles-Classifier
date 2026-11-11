"""One-time TRAIN / VALIDATE / EXPORT workflow for the FDS Pokémon competition.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import zipfile

import numpy as np
import pandas as pd

from pokemon_pipeline import prepare_competition_data
from pokemon_models import (
    CAT_PARAMS,
    LGB_PARAMS,
    CAT_WEIGHT,
    LGB_WEIGHT,
    THRESHOLD,
    fit_final_models,
    predict_fitted_models,
    run_cross_validation,
    save_submission,
    validation_summary,
)

ARTIFACT_NAMES = ('catboost_model.cbm', 'lightgbm_model.txt', 'preprocessing.json')


def file_sha256(path):
    """Streaming SHA-256 fingerprint for reproducibility checks."""
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def export_fitted_models(data, fitted, output_dir, train_jsonl_path=None, validation=None):
    """Persist the EXACT in-memory final models and training-derived features.

    Saves native CatBoost/LightGBM model files (no pickle), and JSON with the
    complete training-derived species lookup, vocabulary, feature column order,
    categorical levels, chart checksum, hyperparameters, and software versions.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cat_path = output_dir / 'catboost_model.cbm'
    lgb_path = output_dir / 'lightgbm_model.txt'
    meta_path = output_dir / 'preprocessing.json'

    fitted.cat_model.save_model(str(cat_path), format='cbm')
    fitted.lgb_model.booster_.save_model(str(lgb_path))

    module_file = Path(__file__).with_name('pokemon_features.py')
    chart_file = Path(data.chart_path)
    metadata = {
        'schema_version': 1,
        'description': 'FDS Gen 2 original CatBoost (65%) + LightGBM (35%) blend',
        'original_feature_module_sha256': file_sha256(module_file),
        'official_type_chart_sha256': file_sha256(chart_file),
        'train_jsonl_sha256': (file_sha256(train_jsonl_path) if train_jsonl_path else None),
        'catboost_model_sha256': file_sha256(cat_path),
        'lightgbm_model_sha256': file_sha256(lgb_path),
        'feature_columns': list(data.X.columns),
        'categorical_columns': list(data.cat_cols),
        'lightgbm_categories': {
            col: data.X_lgb[col].cat.categories.tolist() for col in data.cat_cols
        },
        'species_lookup': data.species_lookup,
        'vocab': data.vocab,
        'final_cat_rounds': fitted.final_cat_rounds,
        'final_lgb_rounds': fitted.final_lgb_rounds,
        'blend_weights': {'catboost': CAT_WEIGHT, 'lightgbm': LGB_WEIGHT},
        'classification_threshold': THRESHOLD,
        'catboost_params': CAT_PARAMS,
        'lightgbm_params': LGB_PARAMS,
        'versions': {
            'python': platform.python_version(),
            **{p: importlib.metadata.version(p)
               for p in ['catboost', 'lightgbm', 'numpy', 'pandas', 'scikit-learn']},
        },
    }
    if validation is not None:
        metadata['cv_scores_mean'] = {
            col: float(validation.cv_results[col].mean())
            for col in ('catboost_accuracy', 'lightgbm_accuracy', 'blend_accuracy')
        }
        metadata['cv_best_cat_rounds'] = list(validation.cat_best_rounds)
        metadata['cv_best_lgb_rounds'] = list(validation.lgb_best_rounds)
    with open(meta_path, 'w', encoding='utf-8') as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False, allow_nan=False)
    print('Saved trained model artifacts to', output_dir)
    for name in ARTIFACT_NAMES:
        path = output_dir / name
        print(f'  {name}: {path.stat().st_size / 1024 / 1024:.2f} MiB')
    return metadata


def make_artifacts_zip(artifact_dir, output_zip):
    """Convenience ZIP; upload its contents under GitHub/artifacts/, not the ZIP."""
    artifact_dir = Path(artifact_dir)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name in ARTIFACT_NAMES:
            file = artifact_dir / name
            if not file.is_file():
                raise FileNotFoundError(file)
            zf.write(file, arcname=f'artifacts/{name}')
    print('Upload these three extracted artifact files to GitHub:', output_zip)
    return output_zip


def train_validate_export(
    competition_dir='/kaggle/input/competitions/fds-pokemon-battles-prediction-2026',
    module_dir=None,
    artifact_dir='/kaggle/working/artifacts',
    output_dir='/kaggle/working',
):
    """One-time end-to-end experiment, then exact native-model round-trip check.

    The *old* cross-validation and training are called unmodified in meaning.
    """
    from pokemon_inference import generate_submission, load_saved_models

    competition_dir = Path(competition_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = prepare_competition_data(
        data_dir=competition_dir,
        input_root='/kaggle/input',
        features_dir=module_dir,
        working_dir=output_dir,
    )
    validation = run_cross_validation(data)
    fold_summary, mean_scores, overall_scores = validation_summary(data, validation)
    print('\nMean fold accuracy:')
    print(mean_scores.to_string())
    validation.cv_results.to_csv(output_dir / 'cross_validation_folds.csv', index=False)

    fitted = fit_final_models(data, validation)
    original_predictions = predict_fitted_models(data, fitted)
    original_submission = save_submission(
        data, original_predictions,
        output=output_dir / 'submission_blend_6848.csv',
    )
    original_submission_path = output_dir / 'submission_blend_6848.csv'

    artifact_dir = Path(artifact_dir)
    export_fitted_models(
        data, fitted, output_dir=artifact_dir,
        train_jsonl_path=competition_dir / 'train.jsonl', validation=validation,
    )

    # Verify the serialization itself: saved models vs just-trained models on
    # the SAME matrices, BEFORE validating the separate test-only pipeline.
    saved_cat, saved_lgb, meta = load_saved_models(artifact_dir)
    restored_cat = saved_cat.predict_proba(data.X_test)[:, 1]
    restored_lgb = saved_lgb.predict(data.X_test_lgb)
    np.testing.assert_allclose(restored_cat, original_predictions.p_test_cat, rtol=0, atol=1e-10)
    np.testing.assert_allclose(restored_lgb, original_predictions.p_test_lgb, rtol=0, atol=1e-10)
    print('PASS: reloaded native model probabilities equal the in-memory model predictions.')

    # Independently redo test feature engineering using ONLY stored training
    # metadata + the provided Kaggle test data and chart. This is precisely
    # the code the inference-only notebook will run after cloning GitHub.
    reloaded_submission_path = output_dir / 'submission_inference_check.csv'
    reloaded = generate_submission(
        competition_dir=competition_dir,
        artifacts_dir=artifact_dir,
        output_path=reloaded_submission_path,
    )
    if not original_submission.equals(reloaded):
        changed = (original_submission['player_won'] != reloaded['player_won']).sum()
        raise AssertionError(f'Saved-model inference changed {changed} test labels; do not upload these artifacts.')
    if original_submission_path.read_bytes() != reloaded_submission_path.read_bytes():
        raise AssertionError('Submission files are not byte-identical after save and reload.')
    print('PASS: inference-only CSV is BYTE-FOR-BYTE identical to original final submission.')

    output_zip = make_artifacts_zip(
        artifact_dir, output_dir / 'trained_models_for_github.zip',
    )
    return {
        'cross_validation': validation.cv_results,
        'cv_means': mean_scores,
        'saved_artifacts': list(ARTIFACT_NAMES),
        'artifacts_zip': str(output_zip),
        'submission_csv': str(original_submission_path),
    }
