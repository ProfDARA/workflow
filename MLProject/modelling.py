"""
MLProject-compatible modelling script for CI
Usage:
  python modelling.py --data namadataset_preprocessing/daily_sales_forecasting.csv --output artifacts
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_preprocessed_data(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found at {p}")
    df = pd.read_csv(p, parse_dates=['Date']) if 'Date' in pd.read_csv(p, nrows=1).columns else pd.read_csv(p)
    # Minimal feature expectation for MLProject demo
    if 'Daily_Revenue' not in df.columns:
        raise ValueError('CSV must contain Daily_Revenue column')
    # simple features: use lag1 if present else create rolling mean
    if 'lag_1' in df.columns:
        features = ['lag_1']
    else:
        df['lag_1'] = df['Daily_Revenue'].shift(1).fillna(0)
        features = ['lag_1']
    X = df[features].fillna(0)
    y = df['Daily_Revenue']
    n = len(df)
    n_train = int(n*0.7)
    n_val = int(n*0.15)
    X_train = X.iloc[:n_train].reset_index(drop=True)
    X_val = X.iloc[n_train:n_train+n_val].reset_index(drop=True)
    X_test = X.iloc[n_train+n_val:].reset_index(drop=True)
    y_train = y.iloc[:n_train].reset_index(drop=True)
    y_val = y.iloc[n_train:n_train+n_val].reset_index(drop=True)
    y_test = y.iloc[n_train+n_val:].reset_index(drop=True)
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    # compute RMSE in a backwards-compatible way
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='namadataset_preprocessing/daily_sales_forecasting.csv')
    parser.add_argument('--output', default='artifacts')
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_preprocessed_data(args.data)

    Path(args.output).mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment('MLProject_CI')

    def _run_training():
        mlflow.sklearn.autolog()
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        y_test_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_test_pred)
        mlflow.log_metrics(metrics)

        model_path = Path(args.output) / 'rf_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        mlflow.log_artifact(str(model_path))

        # save a small sample predictions CSV as additional artifact
        sample = X_test.head(50).copy()
        sample['y_true'] = y_test.reset_index(drop=True).head(50)
        sample['y_pred'] = y_test_pred[:50]
        sample_path = Path(args.output) / 'predictions_sample.csv'
        sample.to_csv(sample_path, index=False)
        mlflow.log_artifact(str(sample_path))

    # If mlflow run already active (invoked via `mlflow run`), do not start a new run
    import os as _os
    # Only start a new run if there is no active run and MLflow didn't set a run id in the environment
    if mlflow.active_run() is None and _os.environ.get('MLFLOW_RUN_ID') is None:
        with mlflow.start_run(run_name='mlproject_basic'):
            _run_training()
    else:
        _run_training()

    print('MLProject run complete. Artifacts saved to', args.output)


if __name__ == '__main__':
    main()
