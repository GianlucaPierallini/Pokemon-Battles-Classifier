"""Competition input and feature-matrix preparation.

The computations and column ordering here follow kaggle_submission(1).ipynb,
unchanged in purpose and parameters. All battle data, targets, submission template,
and the official Gen 2 type chart remain on Kaggle.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from pokemon_features import (
    read_jsonl,
    build_species_lookup,
    build_vocab,
    load_type_chart,
    make_features,
)


@dataclass
class PreparedBattles:
    X: pd.DataFrame
    X_test: pd.DataFrame
    X_lgb: pd.DataFrame
    X_test_lgb: pd.DataFrame
    y: pd.Series
    train_ids: pd.Series
    test_ids: pd.Series
    sample_submission: pd.DataFrame
    cat_cols: list
    num_cols: list
    chart_path: Path
    species_count: int
    vocab: dict
    species_lookup: dict


def load_competition_data(data_dir):
    """Load the exact Kaggle competition input paths and perform original checks."""
    DATA_DIR = Path(data_dir)
    train_file_path = DATA_DIR / 'train.jsonl'
    test_file_path = DATA_DIR / 'test.jsonl'
    sample_submission_path = DATA_DIR / 'sample_submission.csv'

    for path in (train_file_path, test_file_path, sample_submission_path):
        print(f'{path.name}: {path.is_file()} — {path}')

    missing_files = [p.name for p in (train_file_path, test_file_path, sample_submission_path)
                     if not p.is_file()]
    if missing_files:
        raise FileNotFoundError(
            f'Missing competition files: {missing_files}.\n'
            'Open this notebook from the competition Code tab or attach its competition '
            'dataset via Add Input. The official starter notebook expects these files at '
            f'{DATA_DIR}.'
        )

    train_data = read_jsonl(train_file_path)
    test_data = read_jsonl(test_file_path)
    sample_submission = pd.read_csv(sample_submission_path)

    assert len(train_data) == 10000, 'Unexpected training dataset size'
    assert len(test_data) == 5000, 'Unexpected test dataset size'
    assert all('player_won' in battle for battle in train_data)
    assert all('player_won' not in battle for battle in test_data)
    assert sample_submission.columns.tolist() == ['battle_id', 'player_won']

    print('Dataset directory:', DATA_DIR)
    print('Battles:', len(train_data), 'train /', len(test_data), 'test')
    print('Submission template:', sample_submission.shape)
    return train_data, test_data, sample_submission


def prepare_competition_data(
    data_dir='/kaggle/input/competitions/fds-pokemon-battles-prediction-2026',
    input_root='/kaggle/input',
    features_dir=None,
    working_dir='/kaggle/working',
):
    """Reproduce all original metadata, type chart and matrix preparation.

    Vocabulary and species lookup are built from TRAIN records only. The
    test set is only transformed with the same columns. An optional chart
    in the repository is a fallback; the intended chart is on Kaggle.
    """
    DATA_DIR = Path(data_dir)
    DATA_ROOT = Path(input_root)
    FEATURES_DIR = Path(features_dir) if features_dir is not None else Path('/nonexistent')
    working_dir = Path(working_dir)

    train_data, test_data, sample_submission = load_competition_data(DATA_DIR)

    # Original metadata and vocabulary sizes.
    species_lookup = build_species_lookup(train_data)
    vocab = build_vocab(train_data, max_species=55, max_moves=35)

    # Original candidate order: competition data first, then optional GitHub,
    # then Kaggle working and any separately attached Kaggle datasets.
    chart_candidates = [
        DATA_DIR / 'type_chart_gen2.csv',
        FEATURES_DIR / 'type_chart_gen2.csv',
        working_dir / 'type_chart_gen2.csv',
    ]
    chart_candidates.extend(sorted(DATA_ROOT.rglob('type_chart_gen2.csv')))
    chart_paths = list(dict.fromkeys(p for p in chart_candidates if p.is_file()))

    if not chart_paths:
        raise FileNotFoundError(
            'The competition-supplied type_chart_gen2.csv was not found. '
            'Attach it using Kaggle Add Input. Checked the competition directory, '
            'the cloned GitHub folder, and Kaggle input/working directories.'
        )

    chart_path = chart_paths[0]
    type_chart = load_type_chart(chart_path)
    if not type_chart:
        raise ValueError(f'Could not read the Gen 2 type chart from {chart_path}.')

    print('Type chart:', chart_path)
    print('Type matchups loaded:', len(type_chart))
    print('Species metadata:', len(species_lookup))
    print('Common species:', len(vocab['species']))
    print('Common moves:', len(vocab['moves']))

    train_features = make_features(train_data, species_lookup, vocab, type_chart)
    test_features = make_features(test_data, species_lookup, vocab, type_chart)

    # Force precisely the same design matrix, in the same order.
    train_features, test_features = train_features.align(test_features, join='left', axis=1, fill_value=0)
    assert 'player_won' not in train_features.columns
    assert 'battle_id' not in train_features.columns
    assert train_features.columns.equals(test_features.columns)

    X = train_features.copy()
    X_test = test_features.copy()
    y = pd.Series([int(b['player_won']) for b in train_data], name='player_won')
    train_ids = pd.Series([b['battle_id'] for b in train_data])
    test_ids = pd.Series([b['battle_id'] for b in test_data])

    if train_ids.duplicated().any() or test_ids.duplicated().any():
        raise ValueError('Duplicate battle_id found; investigate before training')

    cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
    num_cols = X.columns.difference(cat_cols).tolist()
    for col in cat_cols:
        X[col] = X[col].fillna('unknown').astype(str)
        X_test[col] = X_test[col].fillna('unknown').astype(str)
    for col in num_cols:
        X[col] = X[col].astype('float32')
        X_test[col] = X_test[col].astype('float32')

    print(f'Feature matrix: train {X.shape}, test {X_test.shape}')
    print(f'{len(num_cols)} numeric features, {len(cat_cols)} categorical features')
    print('Class distribution:', y.value_counts(normalize=True).round(3).to_dict())
    assert X.shape == (10000, 714), f'Expected the previous feature shape, got {X.shape}'
    assert len(cat_cols) == 13, 'Categorical column count differs from the local run'

    # Same LightGBM categorical conversion as the original notebook.
    X_lgb = X.copy()
    X_test_lgb = X_test.copy()
    for col in cat_cols:
        # Category levels learned from train only; unknown test categories become NaN.
        categories = pd.Index(X[col].unique())
        X_lgb[col] = pd.Categorical(X[col], categories=categories)
        X_test_lgb[col] = pd.Categorical(X_test[col], categories=categories)

    return PreparedBattles(
        X=X, X_test=X_test, X_lgb=X_lgb, X_test_lgb=X_test_lgb,
        y=y, train_ids=train_ids, test_ids=test_ids,
        sample_submission=sample_submission,
        cat_cols=cat_cols, num_cols=num_cols,
        chart_path=chart_path, species_count=len(species_lookup), vocab=vocab,
        species_lookup=species_lookup,
    )
