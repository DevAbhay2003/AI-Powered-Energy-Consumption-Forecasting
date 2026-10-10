"""
src/forecaster.py
Recursive Multi-Step Future Load Forecaster

Generates multi-hour (e.g., 48-hour or 7-day) forward-looking energy demand predictions
by recursively rolling forward autoregressive lag features and calendar dynamics.
"""

import os
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_champion_artifacts(model_dir: str = "models"):
    """Loads champion model and feature metadata."""
    model_path = os.path.join(model_dir, "best_energy_model.joblib")
    feat_path = os.path.join(model_dir, "feature_names.json")
    
    model = joblib.load(model_path)
    with open(feat_path, "r") as f:
        metadata = json.load(f)
        
    return model, metadata["feature_names"]


def generate_future_forecast(
    historical_df: pd.DataFrame,
    horizon_hours: int = 48,
    output_csv: str = "outputs/forecast_results.csv",
    output_plot: str = "images/05_future_forecast_simulation.png"
) -> pd.DataFrame:
    """
    Simulates recursive multi-step forecasting for the upcoming horizon_hours.
    """
    model, feature_names = load_champion_artifacts()
    
    last_timestamp = historical_df["timestamp"].max()
    future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(hours=1), periods=horizon_hours, freq="h")
    
    # Keep working copy of recent history to compute rolling & lags
    # Need at least 168 hours of history for lag_168
    working_df = historical_df.tail(200).copy()
    
    predictions = []
    
    print(f"\n[FORECAST] Generating {horizon_hours}-Hour Recursive Future Load Forecast...")
    
    for current_time in future_dates:
        # 1. Calendar Features
        hour = current_time.hour
        day_of_week = current_time.dayofweek
        day_of_month = current_time.day
        month = current_time.month
        quarter = current_time.quarter
        is_weekend = int(day_of_week >= 5)
        
        sin_hour = np.sin(2 * np.pi * hour / 24.0)
        cos_hour = np.cos(2 * np.pi * hour / 24.0)
        sin_month = np.sin(2 * np.pi * month / 12.0)
        cos_month = np.cos(2 * np.pi * month / 12.0)
        
        # 2. Simulated Future Weather & Occupancy
        # Day of year temperature curve + daily diurnal cycle
        doy = current_time.dayofyear
        seasonal_t = 20.0 + 12.0 * np.sin(2 * np.pi * (doy - 110) / 365.25)
        diurnal_t = 4.5 * np.sin(2 * np.pi * (hour - 9) / 24)
        temp_c = seasonal_t + diurnal_t + np.random.normal(0, 0.8)
        humidity = np.clip(65 - 0.7 * (temp_c - 20), 25, 90)
        
        if is_weekend:
            occupancy = 0.18 + 0.08 * np.sin(np.pi * np.clip(hour - 10, 0, 10) / 10)
        else:
            occupancy = 0.82 if 8 <= hour <= 18 else 0.18
            
        cdd = max(0.0, temp_c - 22.0)
        hdd = max(0.0, 15.0 - temp_c)
        
        # 3. Lags from working_df
        history_series = working_df["energy_consumption_kwh"].values
        lag_1 = history_series[-1]
        lag_2 = history_series[-2]
        lag_24 = history_series[-24]
        lag_168 = history_series[-168] if len(history_series) >= 168 else history_series[-24]
        
        # 4. Rolling stats on last 24 values
        last_24 = history_series[-24:]
        rolling_mean_24 = np.mean(last_24)
        rolling_std_24 = np.std(last_24)
        rolling_max_24 = np.max(last_24)
        rolling_min_24 = np.min(last_24)
        
        # Construct single feature row
        row_dict = {
            "hour": hour, "day_of_week": day_of_week, "day_of_month": day_of_month,
            "month": month, "quarter": quarter, "is_weekend": is_weekend,
            "sin_hour": sin_hour, "cos_hour": cos_hour,
            "sin_month": sin_month, "cos_month": cos_month,
            "temperature_celsius": temp_c, "humidity_percent": humidity, "occupancy_rate": occupancy,
            "cdd": cdd, "hdd": hdd,
            "lag_1": lag_1, "lag_2": lag_2, "lag_24": lag_24, "lag_168": lag_168,
            "rolling_mean_24": rolling_mean_24, "rolling_std_24": rolling_std_24,
            "rolling_max_24": rolling_max_24, "rolling_min_24": rolling_min_24
        }
        
        X_step = pd.DataFrame([row_dict])[feature_names]
        pred_val = float(model.predict(X_step)[0])
        pred_val = max(50.0, pred_val)  # Physical positive lower bound
        
        predictions.append({
            "timestamp": current_time,
            "forecasted_energy_kwh": round(pred_val, 2),
            "forecasted_temp_celsius": round(temp_c, 2),
            "occupancy_rate": round(occupancy, 2)
        })
        
        # Append predicted value to working_df to feed future lags recursively
        new_row = pd.DataFrame([{
            "timestamp": current_time,
            "energy_consumption_kwh": pred_val
        }])
        working_df = pd.concat([working_df, new_row], ignore_index=True)
        
    df_forecast = pd.DataFrame(predictions)
    
    # Save CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_forecast.to_csv(output_csv, index=False)
    print(f"[INFO] Future forecast exported to: {output_csv}")
    
    # Plot Future Forecast
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.figure(figsize=(14, 5))
    
    # Plot recent history + forecast
    hist_sample = historical_df.tail(48)
    plt.plot(hist_sample["timestamp"], hist_sample["energy_consumption_kwh"], label="Historical Observed (Past 48h)", color="#1f77b4", linewidth=2.0)
    plt.plot(df_forecast["timestamp"], df_forecast["forecasted_energy_kwh"], label="AI Forecast (Next 48h)", color="#d62728", linestyle="--", marker="o", markersize=4, linewidth=2.0)
    
    plt.axvline(last_timestamp, color="black", linestyle=":", linewidth=1.5, label="Forecast Horizon Start")
    plt.title(f"48-Hour Ahead Forward Energy Load Forecasting Simulation")
    plt.xlabel("Timeline")
    plt.ylabel("Electricity Demand (kWh)")
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    print(f"[INFO] Future forecast plot saved to: {output_plot}")
    return df_forecast


if __name__ == "__main__":
    from src.preprocessing import load_raw_data
    raw = load_raw_data()
    generate_future_forecast(raw, horizon_hours=48)
