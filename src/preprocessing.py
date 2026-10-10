"""
src/preprocessing.py
Data Cleaning & Feature Engineering Pipeline

Transforms raw time-series smart-meter data into an ML-ready feature matrix
using cyclical encodings, historical lag features, and rolling window aggregations
while strictly guarding against lookahead data leakage.
"""

import os
import numpy as np
import pandas as pd


def load_raw_data(filepath: str = "data/raw/energy_consumption_raw.csv") -> pd.DataFrame:
    """Loads raw energy consumption data from CSV."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}")
    
    df = pd.read_csv(filepath)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs calendar, cyclical, lag, and rolling features.

    Parameters:
        df: Input DataFrame with 'timestamp' and 'energy_consumption_kwh'

    Returns:
        pd.DataFrame with engineered feature columns (NaNs dropped)
    """
    data = df.copy()
    
    # 1. Calendar Features
    data["hour"] = data["timestamp"].dt.hour
    data["day_of_week"] = data["timestamp"].dt.dayofweek
    data["day_of_month"] = data["timestamp"].dt.day
    data["month"] = data["timestamp"].dt.month
    data["quarter"] = data["timestamp"].dt.quarter
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    
    # 2. Cyclical Encodings (Harmonic Analysis)
    # Allows machine learning models to understand that 23:00 and 00:00 are adjacent
    data["sin_hour"] = np.sin(2 * np.pi * data["hour"] / 24.0)
    data["cos_hour"] = np.cos(2 * np.pi * data["hour"] / 24.0)
    data["sin_month"] = np.sin(2 * np.pi * data["month"] / 12.0)
    data["cos_month"] = np.cos(2 * np.pi * data["month"] / 12.0)
    
    # 3. Weather / Thermal Metrics
    if "temperature_celsius" in data.columns:
        data["cdd"] = np.maximum(0, data["temperature_celsius"] - 22.0)  # Cooling Degree
        data["hdd"] = np.maximum(0, 15.0 - data["temperature_celsius"])  # Heating Degree
    
    # 4. Lag Features (Shifted to prevent lookahead leak)
    # lag_1: Prior hour demand
    # lag_2: 2 hours prior
    # lag_24: Same hour yesterday
    # lag_168: Same hour last week (7 * 24 = 168)
    target = data["energy_consumption_kwh"]
    data["lag_1"] = target.shift(1)
    data["lag_2"] = target.shift(2)
    data["lag_24"] = target.shift(24)
    data["lag_168"] = target.shift(168)
    
    # 5. Rolling Statistics (Calculated on past shifted data to prevent leakage)
    shifted_target = target.shift(1)
    data["rolling_mean_24"] = shifted_target.rolling(window=24).mean()
    data["rolling_std_24"] = shifted_target.rolling(window=24).std()
    data["rolling_max_24"] = shifted_target.rolling(window=24).max()
    data["rolling_min_24"] = shifted_target.rolling(window=24).min()
    
    # 6. Drop initial NaN rows created by the 168-hour lag
    clean_data = data.dropna().reset_index(drop=True)
    return clean_data


def save_processed_data(df: pd.DataFrame, output_path: str = "data/processed/energy_features.csv") -> None:
    """Saves engineered feature DataFrame to disk."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[INFO] Engineered feature dataset saved to: {output_path}")
    print(f"[INFO] Processed records count: {len(df):,}, Columns: {len(df.columns)}")


def run_pipeline() -> pd.DataFrame:
    """Executes the end-to-end preprocessing pipeline."""
    raw = load_raw_data()
    features = engineer_features(raw)
    save_processed_data(features)
    return features


if __name__ == "__main__":
    run_pipeline()
