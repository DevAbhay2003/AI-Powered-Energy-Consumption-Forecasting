"""
src/evaluate.py
Performance Evaluation & Chart Generation Suite

Generates industry-grade model assessment metrics (RMSE, MAE, MAPE, R², MBE)
and exports publication-quality diagnostic plots for portfolios and technical reports.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any


# Apply clean visualization style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8
})


def compute_metrics_table(results_dict: Dict[str, Any], output_dir: str = "outputs") -> pd.DataFrame:
    """
    Summarizes all candidate model metrics into a DataFrame and persists to CSV/JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    rows = []
    for model_name, data in results_dict.items():
        metrics = data["metrics"].copy()
        metrics["Model"] = model_name
        rows.append(metrics)
        
    df_metrics = pd.DataFrame(rows).set_index("Model")
    df_metrics = df_metrics[["RMSE", "MAE", "MAPE_percent", "R2_Score"]]
    
    # Save files
    csv_path = os.path.join(output_dir, "metrics_summary.csv")
    json_path = os.path.join(output_dir, "metrics_report.json")
    
    df_metrics.to_csv(csv_path)
    df_metrics.to_json(json_path, indent=4, orient="index")
    
    print("\n================ MODEL PERFORMANCE BENCHMARK ================")
    print(df_metrics.to_string())
    print("=============================================================")
    print(f"[INFO] Exported metrics to: {csv_path} and {json_path}")
    
    return df_metrics


def plot_raw_exploration(df_raw: pd.DataFrame, images_dir: str = "images") -> str:
    """Plots initial time series trend and daily seasonality patterns."""
    os.makedirs(images_dir, exist_ok=True)
    
    # Figure 1: 30-Day Trend (Hourly Energy and Ambient Temp)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    
    sample_df = df_raw.iloc[2000:2720].copy()  # ~30 days
    
    ax1.plot(sample_df["timestamp"], sample_df["energy_consumption_kwh"], color="#1f77b4", label="Energy Demand (kWh)")
    ax1.set_ylabel("Power Demand (kWh)")
    ax1.set_title("Facility Energy Demand & Temperature Profile (30-Day Sample Window)")
    ax1.legend(loc="upper left")
    
    ax2.plot(sample_df["timestamp"], sample_df["temperature_celsius"], color="#d62728", label="Ambient Temperature (°C)")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_xlabel("Timestamp")
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    fig1_path = os.path.join(images_dir, "01_raw_hourly_trend.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    
    # Figure 2: Diurnal Seasonality (Weekday vs Weekend Profile)
    fig, ax = plt.subplots(figsize=(10, 5))
    df_raw_copy = df_raw.copy()
    df_raw_copy["hour"] = df_raw_copy["timestamp"].dt.hour
    df_raw_copy["day_type"] = np.where(df_raw_copy["timestamp"].dt.dayofweek >= 5, "Weekend", "Weekday")
    
    profile = df_raw_copy.groupby(["hour", "day_type"])["energy_consumption_kwh"].mean().reset_index()
    sns.lineplot(data=profile, x="hour", y="energy_consumption_kwh", hue="day_type", marker="o", ax=ax, palette=["#2ca02c", "#ff7f0e"])
    
    ax.set_title("Average Diurnal Load Curve: Weekday vs Weekend")
    ax.set_xlabel("Hour of Day (00:00 - 23:00)")
    ax.set_ylabel("Mean Energy Consumption (kWh)")
    ax.set_xticks(range(0, 24, 2))
    
    fig2_path = os.path.join(images_dir, "02_daily_seasonality.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    
    print(f"[INFO] Saved EDA plots: {fig1_path}, {fig2_path}")
    return fig1_path


def plot_model_diagnostics(
    test_df: pd.DataFrame,
    y_test: pd.Series,
    y_pred: np.ndarray,
    champion_name: str,
    feature_names: list,
    model_obj: Any,
    images_dir: str = "images"
) -> None:
    """
    Plots Actual vs Predicted comparison over test slice, residuals, and feature importance.
    """
    os.makedirs(images_dir, exist_ok=True)
    
    # Figure 3: Actual vs Predicted (7-Day Slice = 168 hours)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})
    
    sample_window = 168  # 1 week
    test_slice = test_df.iloc[:sample_window].copy()
    slice_actual = y_test.iloc[:sample_window].values
    slice_pred = y_pred[:sample_window]
    residuals = slice_actual - slice_pred
    
    ax1.plot(test_slice["timestamp"], slice_actual, label="Actual Consumption", color="#1f77b4", linewidth=2.0)
    ax1.plot(test_slice["timestamp"], slice_pred, label=f"Predicted ({champion_name})", color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax1.set_title(f"Model Forecasting Performance (7-Day Horizon: 168 Hours) - {champion_name}")
    ax1.set_ylabel("Load (kWh)")
    ax1.legend(loc="upper right")
    
    ax2.bar(test_slice["timestamp"], residuals, color="#7f7f7f", width=0.03, label="Residual (Actual - Pred)")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Residual (kWh)")
    ax2.set_xlabel("Timestamp")
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    fig3_path = os.path.join(images_dir, "03_actual_vs_predicted.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    
    # Figure 4: Feature Importance
    fig, ax = plt.subplots(figsize=(10, 6))
    if hasattr(model_obj, "feature_importances_"):
        importances = model_obj.feature_importances_
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        feat_df = feat_df.sort_values("Importance", ascending=False).head(12)
        
        sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False, ax=ax)
        ax.set_title(f"Top 12 Most Predictive Energy Features ({champion_name})")
        ax.set_xlabel("Relative Feature Importance")
    else:
        # Fallback for linear models: absolute coefficients
        coefs = np.abs(model_obj.coef_)
        feat_df = pd.DataFrame({"Feature": feature_names, "Importance": coefs})
        feat_df = feat_df.sort_values("Importance", ascending=False).head(12)
        sns.barplot(data=feat_df, x="Importance", y="Feature", hue="Feature", palette="Blues_r", legend=False, ax=ax)
        ax.set_title(f"Top 12 Regression Coefficients ({champion_name})")
        ax.set_xlabel("Absolute Coefficient Magnitude")
        
    plt.tight_layout()
    fig4_path = os.path.join(images_dir, "04_feature_importance.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    
    print(f"[INFO] Saved diagnostic plots: {fig3_path}, {fig4_path}")
