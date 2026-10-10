"""
src/simulate_ems.py
Virtual Energy Management System (EMS) Business Simulation

Simulates automated enterprise energy management actions based on AI load forecasts:
- Peak Demand Shaving (Battery Energy Storage System - BESS dispatch)
- Time-of-Use (TOU) Dynamic Tariff Cost Optimization
- Carbon Emission Reduction Tracking
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_virtual_ems_simulation(
    forecast_csv: str = "outputs/forecast_results.csv",
    output_summary_json: str = "outputs/ems_simulation_summary.json",
    output_plot: str = "images/06_peak_shaving_cost_analysis.png"
) -> dict:
    """
    Executes a virtual corporate demand response and cost reduction simulation.
    """
    if not os.path.exists(forecast_csv):
        raise FileNotFoundError(f"Forecast file not found: {forecast_csv}")
        
    df = pd.read_csv(forecast_csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    
    # 1. Tariff Modeling (Time-Of-Use Tariff)
    # Peak Hours: 13:00 to 19:00 ($0.38/kWh)
    # Mid-Peak: 07:00 to 13:00, 19:00 to 22:00 ($0.22/kWh)
    # Off-Peak: 22:00 to 07:00 ($0.11/kWh)
    conditions = [
        (df["hour"] >= 13) & (df["hour"] < 19),
        ((df["hour"] >= 7) & (df["hour"] < 13)) | ((df["hour"] >= 19) & (df["hour"] < 22)),
    ]
    choices = [0.38, 0.22]
    df["tariff_rate_usd"] = np.select(conditions, choices, default=0.11)
    
    # Baseline Cost without optimization
    df["baseline_cost_usd"] = df["forecasted_energy_kwh"] * df["tariff_rate_usd"]
    
    # 2. Virtual Peak Shaving Threshold (Target building peak cap: 60th percentile)
    peak_threshold = float(df["forecasted_energy_kwh"].quantile(0.60))
    
    # 3. Simulated BESS (Battery Energy Storage System: 50 kWh / 25 kW inverter capacity)
    # Discharges battery when demand > threshold during Peak hours (curbing max peak by up to 25 kWh/h)
    max_battery_discharge = 25.0  # kWh available per hour
    
    # Peak alert active during peak tariff windows when load exceeds target cap
    df["is_peak_alert"] = (df["forecasted_energy_kwh"] > peak_threshold) & (df["tariff_rate_usd"] == 0.38)
    
    df["peak_shaved_kwh"] = np.where(
        df["is_peak_alert"],
        np.minimum(df["forecasted_energy_kwh"] - peak_threshold, max_battery_discharge),
        0.0
    )
    
    df["optimized_energy_kwh"] = df["forecasted_energy_kwh"] - df["peak_shaved_kwh"]
    df["optimized_cost_usd"] = df["optimized_energy_kwh"] * df["tariff_rate_usd"]
    
    # 4. Impact Metrics Calculation
    total_baseline_kwh = float(df["forecasted_energy_kwh"].sum())
    total_baseline_cost = float(df["baseline_cost_usd"].sum())
    
    total_shaved_kwh = float(df["peak_shaved_kwh"].sum())
    total_optimized_cost = float(df["optimized_cost_usd"].sum())
    
    # Volumetric Energy Charge Savings
    energy_savings_usd = total_baseline_cost - total_optimized_cost
    
    # Industrial Demand Charge Savings ($15.00 per kW reduced from monthly peak)
    max_unmanaged_peak = float(df["forecasted_energy_kwh"].max())
    max_managed_peak = float(df["optimized_energy_kwh"].max())
    peak_demand_reduction_kw = max_unmanaged_peak - max_managed_peak
    demand_charge_savings_usd = peak_demand_reduction_kw * 15.00
    
    total_cost_savings = energy_savings_usd + demand_charge_savings_usd
    pct_savings = (total_cost_savings / (total_baseline_cost + max_unmanaged_peak * 15.00)) * 100.0
    
    # Carbon footprint reduction (Grid emissions factor: ~0.42 kg CO2 per peak kWh)
    carbon_avoided_kg = total_shaved_kwh * 0.42
    
    summary = {
        "simulation_horizon_hours": len(df),
        "peak_threshold_kwh": round(peak_threshold, 2),
        "unmanaged_peak_load_kwh": round(max_unmanaged_peak, 2),
        "managed_peak_load_kwh": round(max_managed_peak, 2),
        "peak_shaved_kw": round(peak_demand_reduction_kw, 2),
        "total_baseline_cost_usd": round(total_baseline_cost, 2),
        "total_optimized_cost_usd": round(total_optimized_cost, 2),
        "volumetric_energy_savings_usd": round(energy_savings_usd, 2),
        "demand_charge_savings_usd": round(demand_charge_savings_usd, 2),
        "net_total_cost_savings_usd": round(total_cost_savings, 2),
        "cost_savings_percentage": round(pct_savings, 2),
        "peak_energy_dispatched_bess_kwh": round(total_shaved_kwh, 2),
        "co2_emissions_avoided_kg": round(carbon_avoided_kg, 2)
    }
    
    os.makedirs(os.path.dirname(output_summary_json), exist_ok=True)
    with open(output_summary_json, "w") as f:
        json.dump(summary, f, indent=4)
        
    print("\n================ VIRTUAL EMS SIMULATION RESULTS ================")
    print(f"Simulation Horizon:          {summary['simulation_horizon_hours']} Hours")
    print(f"Unmanaged Peak Load:         {summary['unmanaged_peak_load_kwh']} kWh")
    print(f"Managed (Shaved) Peak Load:  {summary['managed_peak_load_kwh']} kWh")
    print(f"Peak Demand Clipped:         {summary['peak_shaved_kw']} kWh ({summary['peak_energy_dispatched_bess_kwh']} total BESS kWh)")
    print(f"Baseline Energy Bill:        ${summary['total_baseline_cost_usd']:.2f}")
    print(f"AI-Optimized Energy Bill:    ${summary['total_optimized_cost_usd']:.2f}")
    print(f"Net Tariff Cost Savings:     ${summary['net_total_cost_savings_usd']:.2f} ({summary['cost_savings_percentage']}%)")
    print(f"Carbon Reduction (CO2):      {summary['co2_emissions_avoided_kg']} kg CO2")
    print("================================================================")
    
    # 5. Plotting Business Simulation
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Load profile comparison
    ax1.plot(df["timestamp"], df["forecasted_energy_kwh"], label="Forecasted Unmanaged Load", color="#d62728", linestyle="--", linewidth=2.0)
    ax1.plot(df["timestamp"], df["optimized_energy_kwh"], label="AI Shaved Load (BESS Active)", color="#2ca02c", linewidth=2.0)
    ax1.axhline(peak_threshold, color="#ff7f0e", linestyle=":", label=f"Peak Shaving Trigger ({peak_threshold:.1f} kWh)")
    ax1.fill_between(df["timestamp"], df["optimized_energy_kwh"], df["forecasted_energy_kwh"], color="#2ca02c", alpha=0.3, label="Peak Energy Shaved")
    ax1.set_ylabel("Power Demand (kWh)")
    ax1.set_title("Virtual EMS Peak Shaving Simulation: Battery Dispatch & Demand Response")
    ax1.legend(loc="upper left")
    
    # Hourly Cost Comparison
    ax2.bar(df["timestamp"], df["baseline_cost_usd"], width=0.03, color="#7f7f7f", alpha=0.6, label="Baseline Cost ($/h)")
    ax2.plot(df["timestamp"], df["optimized_cost_usd"], color="#1f77b4", linewidth=2.0, label="Optimized Cost ($/h)")
    ax2.set_ylabel("Hourly Cost ($)")
    ax2.set_xlabel("Forecast Horizon Timeline")
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    print(f"[INFO] Simulation diagnostic chart saved to: {output_plot}")
    print(f"[INFO] Simulation metrics saved to: {output_summary_json}")
    return summary


if __name__ == "__main__":
    run_virtual_ems_simulation()
