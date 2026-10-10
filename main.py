"""
main.py
AI-Powered Energy Consumption Forecasting System
Master Execution Pipeline

Executes the entire end-to-end data science lifecycle:
1. Data Generation / Loading
2. Exploratory Data Analysis & Plots
3. Feature Engineering & Temporal Preprocessing
4. Model Training & Benchmark Selection
5. Model Evaluation & Metric Serialization
6. 48-Hour Ahead Recursive Forward Forecasting
7. Virtual Energy Management System (EMS) Peak Shaving Simulation
"""

import os
import sys
import time

from src.data_generator import generate_energy_dataset
from src.preprocessing import load_raw_data, engineer_features, save_processed_data
from src.model import prepare_train_test_split, train_and_compare_models, save_champion_model, FEATURE_COLUMNS
from src.evaluate import compute_metrics_table, plot_raw_exploration, plot_model_diagnostics
from src.forecaster import generate_future_forecast
from src.simulate_ems import run_virtual_ems_simulation


def main():
    start_time = time.time()
    
    print("=" * 75)
    print("  AI-POWERED ENERGY CONSUMPTION FORECASTING & EMS SIMULATION SYSTEM  ")
    print("=" * 75)
    
    # -------------------------------------------------------------
    # Phase 1: Data Verification / Generation
    # -------------------------------------------------------------
    raw_path = "data/raw/energy_consumption_raw.csv"
    if not os.path.exists(raw_path):
        print("\n[PHASE 1] Raw dataset not found. Generating realistic smart-meter dataset...")
        raw_df = generate_energy_dataset(output_path=raw_path)
    else:
        print(f"\n[PHASE 1] Loading existing raw dataset from: {raw_path}")
        raw_df = load_raw_data(raw_path)
        
    # Generate exploratory time series plots
    print("[PHASE 1] Generating EDA time-series diagnostic plots...")
    plot_raw_exploration(raw_df)
    
    # -------------------------------------------------------------
    # Phase 2: Feature Engineering & Preprocessing
    # -------------------------------------------------------------
    print("\n[PHASE 2] Constructing temporal, cyclical, lag, and rolling features...")
    processed_df = engineer_features(raw_df)
    save_processed_data(processed_df, "data/processed/energy_features.csv")
    
    # -------------------------------------------------------------
    # Phase 3: Chronological Train-Test Split & Model Training
    # -------------------------------------------------------------
    print("\n[PHASE 3] Partitioning time-series dataset & benchmarking models...")
    X_train, X_test, y_train, y_test, train_df, test_df = prepare_train_test_split(processed_df, test_ratio=0.20)
    
    results, champion_name = train_and_compare_models(X_train, y_train, X_test, y_test)
    champion_obj = results[champion_name]["model_obj"]
    champion_preds = results[champion_name]["predictions"]
    
    # Save the winner model
    save_champion_model(champion_obj, champion_name, FEATURE_COLUMNS, model_dir="models")
    
    # -------------------------------------------------------------
    # Phase 4: Model Evaluation & Reporting
    # -------------------------------------------------------------
    print("\n[PHASE 4] Calculating industry error metrics and generating diagnostic visuals...")
    compute_metrics_table(results, output_dir="outputs")
    plot_model_diagnostics(
        test_df=test_df,
        y_test=y_test,
        y_pred=champion_preds,
        champion_name=champion_name,
        feature_names=FEATURE_COLUMNS,
        model_obj=champion_obj,
        images_dir="images"
    )
    
    # -------------------------------------------------------------
    # Phase 5: Future Forecasting (48-Hour Horizon)
    # -------------------------------------------------------------
    print("\n[PHASE 5] Executing 48-Hour forward load forecasting...")
    forecast_df = generate_future_forecast(
        historical_df=raw_df,
        horizon_hours=48,
        output_csv="outputs/forecast_results.csv",
        output_plot="images/05_future_forecast_simulation.png"
    )
    
    # -------------------------------------------------------------
    # Phase 6: Virtual EMS Business Simulation
    # -------------------------------------------------------------
    print("\n[PHASE 6] Running Virtual Energy Management System (EMS) Peak Shaving Simulation...")
    simulation_metrics = run_virtual_ems_simulation(
        forecast_csv="outputs/forecast_results.csv",
        output_summary_json="outputs/ems_simulation_summary.json",
        output_plot="images/06_peak_shaving_cost_analysis.png"
    )
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print("=" * 75)
    print("Generated Artifacts:")
    print("  - Models:   models/best_energy_model.joblib, models/feature_names.json")
    print("  - Outputs:  outputs/metrics_summary.csv, outputs/metrics_report.json, outputs/forecast_results.csv, outputs/ems_simulation_summary.json")
    print("  - Visuals:  images/01_raw_hourly_trend.png, images/02_daily_seasonality.png, images/03_actual_vs_predicted.png,")
    print("              images/04_feature_importance.png, images/05_future_forecast_simulation.png, images/06_peak_shaving_cost_analysis.png")
    print("=" * 75)


if __name__ == "__main__":
    main()
