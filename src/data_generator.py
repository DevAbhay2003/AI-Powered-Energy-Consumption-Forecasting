"""
src/data_generator.py
Realistic Smart-Meter Energy Dataset Generator

Simulates hourly energy consumption for a smart commercial/residential facility
incorporating diurnal load patterns, weekly cycles, seasonal weather effects,
and thermodynamic cooling/heating degree responses.
"""

import os
import numpy as np
import pandas as pd


def generate_energy_dataset(
    start_date: str = "2024-01-01 00:00:00",
    end_date: str = "2025-12-31 23:00:00",
    freq: str = "h",
    random_seed: int = 42,
    output_path: str = "data/raw/energy_consumption_raw.csv"
) -> pd.DataFrame:
    """
    Generates a realistic time-series energy consumption dataset.

    Parameters:
        start_date: Start timestamp (YYYY-MM-DD HH:MM:SS)
        end_date: End timestamp (YYYY-MM-DD HH:MM:SS)
        freq: Sampling frequency ('h' for hourly)
        random_seed: Seed for reproducibility
        output_path: CSV destination path

    Returns:
        pd.DataFrame containing synthetic hourly energy records
    """
    np.random.seed(random_seed)
    
    # Generate timestamp index
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_samples = len(date_range)
    
    df = pd.DataFrame({"timestamp": date_range})
    
    # Time-based components
    day_of_year = df["timestamp"].dt.dayofyear
    hour = df["timestamp"].dt.hour
    day_of_week = df["timestamp"].dt.dayofweek
    is_weekend = (day_of_week >= 5).astype(int)
    
    # 1. Realistic Temperature Modeling (°C)
    # Seasonal curve: peaks around day 200 (July), coldest around day 20 (January)
    seasonal_temp = 20.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 110) / 365.25)
    
    # Daily temperature cycle: cooler at 5 AM, warmest at 3 PM (hour 15)
    diurnal_temp = 4.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    
    # Weather stochastic noise
    temp_noise = np.random.normal(0, 2.0, n_samples)
    temperature = seasonal_temp + diurnal_temp + temp_noise
    df["temperature_celsius"] = np.round(temperature, 2)
    
    # 2. Relative Humidity (%)
    # Inversely correlated with temperature with noise
    humidity = 65 - 0.7 * (temperature - 20) + np.random.normal(0, 5.0, n_samples)
    df["humidity_percent"] = np.round(np.clip(humidity, 20.0, 95.0), 1)
    
    # 3. Occupancy Index (0.0 to 1.0)
    # High during weekday business hours (08:00 - 18:00), lower on weekends/nights
    base_occupancy = np.where(
        is_weekend == 1,
        0.15 + 0.10 * np.sin(np.pi * np.clip(hour - 10, 0, 10) / 10),
        np.where(
            (hour >= 8) & (hour <= 18),
            0.80 + 0.10 * np.sin(np.pi * (hour - 8) / 10),
            0.15 + 0.05 * np.sin(np.pi * hour / 24)
        )
    )
    df["occupancy_rate"] = np.round(np.clip(base_occupancy + np.random.normal(0, 0.03, n_samples), 0.05, 1.0), 3)
    
    # 4. Thermodynamic HVAC Loads
    # Cooling load when temp > 22°C (Air Conditioning)
    cooling_demand = np.maximum(0, df["temperature_celsius"] - 22.0) * 8.5
    # Heating load when temp < 15°C (Heat Pumps/Resistance)
    heating_demand = np.maximum(0, 15.0 - df["temperature_celsius"]) * 6.0
    
    # 5. Base Plug Loads & Lighting
    # Peak activity around 10:00 - 12:00 and 14:00 - 17:00
    plug_load = 120.0 + df["occupancy_rate"] * 95.0
    
    # 6. Total Energy Consumption (kWh)
    uncontrollable_noise = np.random.normal(0, 7.0, n_samples)
    energy_kwh = plug_load + cooling_demand + heating_demand + uncontrollable_noise
    
    # Ensure physical constraint: positive energy
    df["energy_consumption_kwh"] = np.round(np.maximum(45.0, energy_kwh), 2)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[INFO] Successfully generated {n_samples:,} records.")
    print(f"[INFO] Date span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"[INFO] Raw dataset saved to: {output_path}")
    print(f"[INFO] Energy Consumption Statistics (kWh):")
    print(df["energy_consumption_kwh"].describe().to_string())
    
    return df


if __name__ == "__main__":
    generate_energy_dataset()
