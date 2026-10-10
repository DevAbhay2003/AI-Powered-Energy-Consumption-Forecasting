"""
src/model.py
Model Training and Selection Engine

Trains baseline and ensemble machine learning models for time-series load forecasting.
Enforces chronological time-based splitting to prevent future data leakage.
Persists the champion model and training artifact metadata.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


FEATURE_COLUMNS = [
    "hour", "day_of_week", "day_of_month", "month", "quarter", "is_weekend",
    "sin_hour", "cos_hour", "sin_month", "cos_month",
    "temperature_celsius", "humidity_percent", "occupancy_rate",
    "cdd", "hdd",
    "lag_1", "lag_2", "lag_24", "lag_168",
    "rolling_mean_24", "rolling_std_24", "rolling_max_24", "rolling_min_24"
]

TARGET_COLUMN = "energy_consumption_kwh"


def prepare_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Splits time-series chronologically into train and test partitions.
    DO NOT shuffle time-series data to prevent future-data lookahead bias.
    """
    split_index = int(len(df) * (1 - test_ratio))
    
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]
    
    print(f"[INFO] Chronological Split: Train ({len(train_df):,} rows) | Test ({len(test_df):,} rows)")
    print(f"[INFO] Train Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"[INFO] Test Period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    return X_train, X_test, y_train, y_test, train_df, test_df


def train_and_compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, Any], str]:
    """
    Trains multiple candidate models and identifies the top-performing model on test RMSE.
    """
    candidate_models = {
        "Ridge_Regression": Ridge(alpha=1.0),
        "Random_Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        ),
        "Hist_Gradient_Boosting": HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.08,
            max_depth=8,
            random_state=42
        )
    }
    
    results = {}
    best_rmse = float("inf")
    best_model_name = ""
    
    print("\n--- Model Training & Benchmarking ---")
    for name, model in candidate_models.items():
        print(f"[TRAINING] {name}...")
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        mape = np.mean(np.abs((y_test - preds) / y_test)) * 100.0
        
        results[name] = {
            "model_obj": model,
            "predictions": preds,
            "metrics": {
                "RMSE": round(float(rmse), 3),
                "MAE": round(float(mae), 3),
                "MAPE_percent": round(float(mape), 3),
                "R2_Score": round(float(r2), 4)
            }
        }
        
        print(f"  -> {name} | RMSE: {rmse:.3f} kWh | MAE: {mae:.3f} kWh | R²: {r2:.4f} | MAPE: {mape:.2f}%")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            
    print(f"\n[WINNER] Champion Model Selected: {best_model_name} (Lowest RMSE: {best_rmse:.3f} kWh)")
    return results, best_model_name


def save_champion_model(
    model: Any,
    model_name: str,
    feature_list: list,
    model_dir: str = "models"
) -> str:
    """Saves trained champion model and feature metadata."""
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "best_energy_model.joblib")
    features_path = os.path.join(model_dir, "feature_names.json")
    
    joblib.dump(model, model_path)
    with open(features_path, "w") as f:
        json.dump({
            "model_name": model_name,
            "feature_names": feature_list
        }, f, indent=4)
        
    print(f"[INFO] Persisted champion model to: {model_path}")
    print(f"[INFO] Saved feature metadata to: {features_path}")
    return model_path


if __name__ == "__main__":
    from src.preprocessing import run_pipeline
    df = run_pipeline()
    X_train, X_test, y_train, y_test, tr_df, te_df = prepare_train_test_split(df)
    results, champion_name = train_and_compare_models(X_train, y_train, X_test, y_test)
    save_champion_model(results[champion_name]["model_obj"], champion_name, FEATURE_COLUMNS)
