"""
MLProject-compatible modelling script for CI (Demand Forecasting)
Usage:
    python modelling.py --data namadataset_preprocessing/daily_sales_forecasting.csv --output artifacts
    python modelling.py --data cleaned_amazon_sales.csv --group-col Category --group-value Kurta
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_demand_data(
    path: str,
    group_col: str,
    group_value: str | None,
    target_col: str,
    min_group_size: int
):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Preprocessed CSV not found at {p}")

    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]

    if 'Date' not in df.columns:
        raise ValueError("CSV must contain Date column")

    if target_col not in df.columns:
        if 'Daily_Revenue' in df.columns:
            print(f"Target column '{target_col}' not found. Using Daily_Revenue as proxy target.")
            target_col = 'Daily_Revenue'
        else:
            raise ValueError(f"CSV must contain target column '{target_col}' or Daily_Revenue")

    group_col = (group_col or '').strip()
    if group_col.lower() in ('', 'none', 'all', 'global'):
        group_col = ''

    if group_col and group_col not in df.columns:
        print(f"Group column '{group_col}' not found. Using global demand forecast.")
        group_col = ''

    if not group_col:
        group_col = '__group__'
        df[group_col] = 'ALL'
        min_group_size = 1

    df = df[['Date', group_col, target_col]].copy()
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df[group_col] = df[group_col].astype(str).str.strip()
    df = df.dropna(subset=['Date', group_col, target_col])

    if group_value:
        target_norm = str(group_value).strip().lower()
        group_norm = df[group_col].str.strip().str.lower()
        df = df[group_norm == target_norm]

    daily = (
        df.groupby(['Date', group_col], as_index=False)[target_col]
        .sum()
        .rename(columns={target_col: 'Daily_Demand'})
    )

    if not group_value and daily[group_col].nunique() > 1 and min_group_size > 1:
        group_counts = daily.groupby(group_col).size()
        keep_groups = group_counts[group_counts >= min_group_size].index
        daily = daily[daily[group_col].isin(keep_groups)]

    if daily.empty:
        raise ValueError("No rows available after aggregation. Check group filters or min_group_size.")

    daily = daily.sort_values([group_col, 'Date'])
    daily['lag_1'] = daily.groupby(group_col)['Daily_Demand'].shift(1)
    daily['lag_7'] = daily.groupby(group_col)['Daily_Demand'].shift(7)
    daily['rolling_mean_7'] = daily.groupby(group_col)['Daily_Demand'].transform(
        lambda s: s.shift(1).rolling(window=7).mean()
    )
    daily['rolling_std_7'] = daily.groupby(group_col)['Daily_Demand'].transform(
        lambda s: s.shift(1).rolling(window=7).std()
    )
    daily['day'] = daily['Date'].dt.day
    daily['month'] = daily['Date'].dt.month
    daily['year'] = daily['Date'].dt.year
    daily['weekday'] = daily['Date'].dt.weekday
    daily['weekofyear'] = daily['Date'].dt.isocalendar().week.astype(int)
    daily['is_weekend'] = (daily['weekday'] >= 5).astype(int)

    daily = daily.fillna(0).reset_index(drop=True)

    cat = daily[group_col].astype('category')
    daily['group_id'] = cat.cat.codes

    unique_dates = sorted(daily['Date'].unique())
    if len(unique_dates) >= 3:
        n_dates = len(unique_dates)
        n_train = max(1, int(n_dates * 0.7))
        n_val = max(1, int(n_dates * 0.15))
        train_end = unique_dates[n_train - 1]
        val_end = unique_dates[min(n_train + n_val - 1, n_dates - 1)]
        train_df = daily[daily['Date'] <= train_end]
        val_df = daily[(daily['Date'] > train_end) & (daily['Date'] <= val_end)]
        test_df = daily[daily['Date'] > val_end]
        if len(test_df) == 0:
            train_df, val_df, test_df = None, None, None
    else:
        train_df, val_df, test_df = None, None, None

    if train_df is None or val_df is None or test_df is None:
        n = len(daily)
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))
        train_df = daily.iloc[:n_train]
        val_df = daily.iloc[n_train:n_train + n_val]
        test_df = daily.iloc[n_train + n_val:]

    feature_cols = ['lag_1', 'lag_7', 'rolling_mean_7', 'rolling_std_7', 'day', 'month', 'year', 'weekday', 'weekofyear', 'is_weekend']
    if daily[group_col].nunique() > 1:
        feature_cols = ['group_id'] + feature_cols

    X_train = train_df[feature_cols].reset_index(drop=True)
    X_val = val_df[feature_cols].reset_index(drop=True)
    X_test = test_df[feature_cols].reset_index(drop=True)
    y_train = train_df['Daily_Demand'].reset_index(drop=True)
    y_val = val_df['Daily_Demand'].reset_index(drop=True)
    y_test = test_df['Daily_Demand'].reset_index(drop=True)

    meta = {
        'group_col': group_col,
        'group_value': group_value,
        'target_name': 'Daily_Demand',
        'n_groups': int(daily[group_col].nunique())
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, meta


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
    parser.add_argument('--group-col', default='Category')
    parser.add_argument('--group-value', default='')
    parser.add_argument('--target-col', default='Qty')
    parser.add_argument('--min-group-size', type=int, default=30)
    args = parser.parse_args()

    group_value = args.group_value.strip() if args.group_value else None
    X_train, X_val, X_test, y_train, y_val, y_test, meta = load_demand_data(
        args.data,
        group_col=args.group_col,
        group_value=group_value,
        target_col=args.target_col,
        min_group_size=args.min_group_size
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _run_training():
        mlflow.sklearn.autolog()

        mlflow.log_params({
            'group_col': meta.get('group_col'),
            'group_value': meta.get('group_value'),
            'target_name': meta.get('target_name'),
            'n_groups': meta.get('n_groups')
        })

        model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
        )

        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)

        metrics = evaluate_regression(y_test, y_test_pred)

        mlflow.log_metrics(metrics)

        # Save sklearn model in MLflow format
        mlflow.sklearn.log_model(model, "model")

        # Optional local pickle artifact
        model_path = output_dir / "rf_model.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mlflow.log_artifact(str(model_path))

        # Sample predictions artifact
        sample = X_test.head(50).copy()

        sample["y_true"] = y_test.reset_index(drop=True).head(50)
        sample["y_pred"] = y_test_pred[:50]

        sample_path = output_dir / "predictions_sample.csv"

        sample.to_csv(sample_path, index=False)

        mlflow.log_artifact(str(sample_path))

    _run_training()


if __name__ == '__main__':
    main()
