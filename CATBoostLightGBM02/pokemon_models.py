"""CatBoost + LightGBM experiment moved from kaggle_submission(1).ipynb.

This module deliberately preserves the exact model hyperparameters, validation
split, early stopping, iteration selection, feature handling, 0.65/0.35 blend,
and binary threshold from the original 68.48% LOCAL cross-validation run.
The local score is not guaranteed on a different Kaggle environment.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

RANDOM_STATE = 42
N_SPLITS = 3
CAT_WEIGHT = 0.65
LGB_WEIGHT = 0.35
THRESHOLD = 0.5

# Unchanged from the original submission notebook.
CAT_PARAMS = dict(
    iterations=1700, depth=5, learning_rate=0.035, l2_leaf_reg=7.0,
    random_strength=0.7, loss_function='Logloss', eval_metric='Logloss',
    random_seed=RANDOM_STATE, thread_count=4,
    allow_writing_files=False, verbose=False
)
LGB_PARAMS = dict(
    n_estimators=1700, learning_rate=0.025, num_leaves=17,
    min_child_samples=55, colsample_bytree=0.85,
    subsample=0.85, subsample_freq=1, reg_lambda=5.0,
    objective='binary', random_state=RANDOM_STATE, n_jobs=4, verbosity=-1
)


@dataclass
class ValidationResult:
    cv_results: pd.DataFrame
    oof_cat: np.ndarray
    oof_lgb: np.ndarray
    oof_blend: np.ndarray
    cat_best_rounds: list
    lgb_best_rounds: list


@dataclass
class FittedModels:
    cat_model: CatBoostClassifier
    lgb_model: LGBMClassifier
    final_cat_rounds: int
    final_lgb_rounds: int


@dataclass
class TestPredictions:
    p_test_cat: np.ndarray
    p_test_lgb: np.ndarray
    p_test_blend: np.ndarray
    final_cat_rounds: int
    final_lgb_rounds: int


def run_cross_validation(data):
    """Exactly the previous three-fold CatBoost / LightGBM validation."""
    X, X_lgb, y, cat_cols = data.X, data.X_lgb, data.y, data.cat_cols

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_cat = np.zeros(len(y), dtype=float)
    oof_lgb = np.zeros(len(y), dtype=float)
    cv_rows = []
    cat_best_rounds = []
    lgb_best_rounds = []

    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        print(f'\nFold {fold}/{N_SPLITS}')
        cat = CatBoostClassifier(**CAT_PARAMS)
        cat.fit(X.iloc[tr], y.iloc[tr], cat_features=cat_cols,
                eval_set=(X.iloc[va], y.iloc[va]), early_stopping_rounds=150)
        oof_cat[va] = cat.predict_proba(X.iloc[va])[:, 1]
        cat_best_rounds.append(max(50, cat.get_best_iteration() + 1))

        lgb = LGBMClassifier(**LGB_PARAMS)
        lgb.fit(X_lgb.iloc[tr], y.iloc[tr],
                eval_set=[(X_lgb.iloc[va], y.iloc[va])],
                eval_metric='binary_logloss',
                callbacks=[early_stopping(150, verbose=False), log_evaluation(0)])
        oof_lgb[va] = lgb.predict_proba(X_lgb.iloc[va])[:, 1]
        lgb_best_rounds.append(max(50, lgb.best_iteration_))

        p_cat = (oof_cat[va] >= 0.5).astype(int)
        p_lgb = (oof_lgb[va] >= 0.5).astype(int)
        p_blend = (0.65 * oof_cat[va] + 0.35 * oof_lgb[va] >= 0.5).astype(int)
        row = {'fold': fold,
               'catboost_accuracy': accuracy_score(y.iloc[va], p_cat),
               'lightgbm_accuracy': accuracy_score(y.iloc[va], p_lgb),
               'blend_accuracy': accuracy_score(y.iloc[va], p_blend),
               'catboost_rounds': cat_best_rounds[-1],
               'lightgbm_rounds': lgb_best_rounds[-1]}
        cv_rows.append(row)
        print(row)

    cv_results = pd.DataFrame(cv_rows)
    oof_blend = 0.65 * oof_cat + 0.35 * oof_lgb
    return ValidationResult(
        cv_results=cv_results,
        oof_cat=oof_cat, oof_lgb=oof_lgb, oof_blend=oof_blend,
        cat_best_rounds=cat_best_rounds, lgb_best_rounds=lgb_best_rounds,
    )


def validation_summary(data, validation):
    """Return the same mean/std fold table and overall OOF accuracies."""
    cv_results = validation.cv_results
    print('OOF overall (may differ slightly from mean fold accuracy):')
    overall = {}
    for name, probs in [('CatBoost', validation.oof_cat),
                        ('LightGBM', validation.oof_lgb),
                        ('Blend', validation.oof_blend)]:
        score = accuracy_score(data.y, (probs >= 0.5).astype(int))
        overall[name] = score
        print(f'{name:12s}: {score:.4%}')
    means = cv_results[['catboost_accuracy', 'lightgbm_accuracy', 'blend_accuracy']].mean()
    return cv_results[['catboost_accuracy', 'lightgbm_accuracy', 'blend_accuracy']].agg(['mean', 'std']), means, overall


def fit_final_models(data, validation):
    """Refit the ORIGINAL CatBoost and LightGBM models on all labeled battles.

    Identical training parameters and median-CV-iteration selection as the
    former train_final_and_predict implementation; return model objects so
    they can be saved without fitting them a second time.
    """
    X, y = data.X, data.y
    X_lgb, cat_cols = data.X_lgb, data.cat_cols
    cat_best_rounds = validation.cat_best_rounds
    lgb_best_rounds = validation.lgb_best_rounds

    final_cat_rounds = int(np.median(cat_best_rounds))
    final_lgb_rounds = int(np.median(lgb_best_rounds))
    print('Final CatBoost rounds:', final_cat_rounds)
    print('Final LightGBM rounds:', final_lgb_rounds)

    cat_final = CatBoostClassifier(**{**CAT_PARAMS, 'iterations': final_cat_rounds})
    cat_final.fit(X, y, cat_features=cat_cols)

    lgb_final = LGBMClassifier(**{**LGB_PARAMS, 'n_estimators': final_lgb_rounds})
    lgb_final.fit(X_lgb, y)

    return FittedModels(cat_model=cat_final, lgb_model=lgb_final,
                        final_cat_rounds=final_cat_rounds,
                        final_lgb_rounds=final_lgb_rounds)


def predict_fitted_models(data, fitted):
    """Same final-model prediction and blending calculation as before."""
    p_test_cat = fitted.cat_model.predict_proba(data.X_test)[:, 1]
    p_test_lgb = fitted.lgb_model.predict_proba(data.X_test_lgb)[:, 1]
    p_test_blend = 0.65 * p_test_cat + 0.35 * p_test_lgb
    print('Full-data fitting complete.')

    return TestPredictions(
        p_test_cat=p_test_cat, p_test_lgb=p_test_lgb, p_test_blend=p_test_blend,
        final_cat_rounds=fitted.final_cat_rounds,
        final_lgb_rounds=fitted.final_lgb_rounds,
    )


def train_final_and_predict(data, validation):
    """Backward-compatible entry point for the original full Kaggle pipeline."""
    fitted = fit_final_models(data, validation)
    return predict_fitted_models(data, fitted)


def save_submission(data, predictions, output='/kaggle/working/submission_blend_6848.csv'):
    """Write the original 0/1 blend with sample_submission battle_id order."""
    p_test_blend = predictions.p_test_blend
    test_ids = data.test_ids
    sample_submission = data.sample_submission

    # Keep battle IDs aligned with the sample submission (do not sort or assume equal row order).
    by_id = pd.Series((p_test_blend >= 0.5).astype(int), index=test_ids.to_numpy())
    if not by_id.index.is_unique or not sample_submission['battle_id'].is_unique:
        raise ValueError('Duplicate battle_id detected')
    submission = sample_submission.copy()
    submission['player_won'] = submission['battle_id'].map(by_id)
    if submission['player_won'].isna().any():
        raise ValueError('Some template battle_ids are missing from test predictions')
    submission['player_won'] = submission['player_won'].astype(int)
    assert submission.shape == (5000, 2)
    assert submission.columns.tolist() == ['battle_id', 'player_won']
    assert submission['player_won'].isin([0, 1]).all()
    OUTPUT = Path(output)
    submission.to_csv(OUTPUT, index=False)
    print('Saved:', OUTPUT)
    return submission
