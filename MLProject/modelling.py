"""
Simple modelling pipeline for demand forecasting.
Reads `amazon_preprocessing/daily_demand_forecasting.csv`, builds per-category time series,
trains a RandomForestRegressor, evaluates against a naive baseline, and exports feature
importance as CSV.
"""

from pathlib import Path
import joblib
import math

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from mlflow import sklearn as mlflow_sklearn


FEATURE_COLUMNS = [
    'category_encoded',
    'lag_1', 'lag_7',
    'rolling_mean_7', 'rolling_std_7',
    'day', 'month', 'weekday', 'is_weekend'
]

DATA_PATH = Path(__file__).resolve().parent / 'amazon_preprocessing' / 'daily_demand_forecasting.csv'
MODEL_OUTPUT = Path(__file__).resolve().parent / 'models'
TRAINING_DISTRIBUTION_COLUMNS = [
    'category_encoded',
    'lag_1', 'lag_7',
    'rolling_mean_7', 'rolling_std_7',
    'Daily_Demand',
]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f'Forecasting dataset not found: {path}')
    return pd.read_csv(path)


def _prepare_category_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    if 'Date' not in df.columns or 'Daily_Demand' not in df.columns:
        raise KeyError("Columns 'Date' and 'Daily_Demand' are required in the forecasting dataset.")
    if 'Category' not in df.columns:
        raise KeyError("Column 'Category' is required for per-category forecasting.")

    prepared = df.copy()
    prepared['Date'] = pd.to_datetime(prepared['Date'], errors='coerce')
    prepared['Daily_Demand'] = pd.to_numeric(prepared['Daily_Demand'], errors='coerce')
    prepared['Category'] = prepared['Category'].astype(str).str.strip()
    prepared = prepared.dropna(subset=['Date', 'Daily_Demand', 'Category'])
    prepared = prepared[prepared['Category'] != '']

    if prepared['Category'].nunique(dropna=True) > 1:
        print('NOTE: Category detected. Building per-category daily demand series.')

    category_df = (
        prepared.groupby(['Category', 'Date'], as_index=False)['Daily_Demand']
        .sum()
        .sort_values(['Category', 'Date'])
        .reset_index(drop=True)
    )
    return category_df


def _add_time_series_features(df: pd.DataFrame) -> pd.DataFrame:
    series_df = df.copy().sort_values(['Category', 'Date']).reset_index(drop=True)

    label_encoder = LabelEncoder()
    series_df['category_encoded'] = label_encoder.fit_transform(series_df['Category'])

    series_df['lag_1'] = series_df.groupby('Category')['Daily_Demand'].shift(1)
    series_df['lag_7'] = series_df.groupby('Category')['Daily_Demand'].shift(7)
    series_df['rolling_mean_7'] = series_df.groupby('Category')['Daily_Demand'].transform(
        lambda values: values.shift(1).rolling(window=7).mean()
    )
    series_df['rolling_std_7'] = series_df.groupby('Category')['Daily_Demand'].transform(
        lambda values: values.shift(1).rolling(window=7).std()
    )

    series_df['day'] = series_df['Date'].dt.day
    series_df['month'] = series_df['Date'].dt.month
    series_df['weekday'] = series_df['Date'].dt.weekday
    series_df['is_weekend'] = (series_df['weekday'] >= 5).astype(int)

    return series_df.dropna().reset_index(drop=True)


def _time_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15):
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)

    n_rows = len(df)
    if n_rows < 3:
        raise ValueError('Not enough rows for time-based split.')

    train_end = max(1, int(n_rows * train_ratio))
    val_end = max(train_end + 1, int(n_rows * (train_ratio + val_ratio)))
    val_end = min(val_end, n_rows - 1)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def _evaluate(y_true, y_pred, label: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'{label}')
    print(f'  MAE : {mae:.4f}')
    print(f'  RMSE: {rmse:.4f}')
    print(f'  R2  : {r2:.4f}')
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def _baseline_naive(df: pd.DataFrame):
    if 'lag_1' not in df.columns:
        raise KeyError('lag_1 is required for the naive baseline.')
    return pd.to_numeric(df['lag_1'], errors='coerce').fillna(0)


def _export_feature_importance(model, feature_names, output_dir: Path):
    importances = getattr(model, 'feature_importances_', None)
    if importances is None:
        return None

    importance_df = pd.DataFrame({
        'feature': list(feature_names)[:len(importances)],
        'importance': importances,
    }).sort_values('importance', ascending=False)

    csv_path = output_dir / 'feature_importance.csv'
    importance_df.to_csv(csv_path, index=False)
    return csv_path


def _print_training_distribution(df: pd.DataFrame):
    available_columns = [c for c in TRAINING_DISTRIBUTION_COLUMNS if c in df.columns]
    if not available_columns:
        return

    print('Training feature distribution snapshot')
    print(df[available_columns].describe())


def _validate_formulation(df: pd.DataFrame):
    if 'Category' not in df.columns:
        raise KeyError("Column 'Category' is required for per-category forecasting.")

    if 'Category' in df.columns and df['Category'].nunique(dropna=True) <= 1:
        print('NOTE: Only one category detected. Per-category setup still works, but category signal is limited.')


def train_evaluate(df: pd.DataFrame, random_state: int = 42):
    daily_df = _prepare_category_daily_demand(df)
    daily_df = _add_time_series_features(daily_df)

    _validate_formulation(df)

    if len(daily_df) < 10:
        raise ValueError('Not enough category-day observations after aggregation and feature creation.')

    missing = [c for c in FEATURE_COLUMNS if c not in daily_df.columns]
    if missing:
        raise KeyError(f'Missing feature columns after feature engineering: {missing}')

    _print_training_distribution(daily_df)

    train_df, val_df, test_df = _time_split(daily_df)
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df['Daily_Demand']
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df['Daily_Demand']
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df['Daily_Demand']

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    baseline_val = _baseline_naive(val_df)
    baseline_test = _baseline_naive(test_df)

    print('Evaluation Results')
    val_metrics = _evaluate(y_val, y_val_pred, 'Validation - RandomForest')
    test_metrics = _evaluate(y_test, y_test_pred, 'Test - RandomForest')
    baseline_val_metrics = _evaluate(y_val, baseline_val, 'Validation - Naive Baseline')
    baseline_test_metrics = _evaluate(y_test, baseline_test, 'Test - Naive Baseline')

    if test_metrics['rmse'] > baseline_test_metrics['rmse']:
        print('WARNING: RandomForest is worse than the naive baseline on the test split.')

    out_dir = MODEL_OUTPUT
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / 'rf_model.joblib')
    feature_csv = _export_feature_importance(model, FEATURE_COLUMNS, out_dir)
    print(f'Model saved to {out_dir / "rf_model.joblib"}')
    if feature_csv is not None:
        print(f'Feature importance exported to {feature_csv}')

    try:
        import mlflow

        mlflow.set_experiment('demand_forecasting')
        with mlflow.start_run():
            mlflow.log_params({'model': 'RandomForestRegressor', 'n_estimators': 100, 'max_depth': 10})
            mlflow.log_metrics({
                'val_mae': float(val_metrics['mae']),
                'val_rmse': float(val_metrics['rmse']),
                'val_r2': float(val_metrics['r2']),
                'test_mae': float(test_metrics['mae']),
                'test_rmse': float(test_metrics['rmse']),
                'test_r2': float(test_metrics['r2']),
                'baseline_val_rmse': float(baseline_val_metrics['rmse']),
                'baseline_test_rmse': float(baseline_test_metrics['rmse']),
            })
            mlflow_sklearn.log_model(model, 'model')
            if feature_csv is not None:
                mlflow.log_artifact(str(feature_csv))
        print('Logged run to MLflow')
    except Exception:
        print('MLflow not available or failed to log - skipping MLflow step')


if __name__ == '__main__':
    df = load_data()
    train_evaluate(df)
