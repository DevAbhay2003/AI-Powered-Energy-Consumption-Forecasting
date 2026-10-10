# System Architecture & Technical Specifications

## 1. High-Level Architecture Overview

The **AI-Powered Energy Consumption Forecasting System** is architected to ingest smart-meter time-series telemetry, process temporal patterns with zero lookahead bias, train multiple machine learning forecasters, and feed predictions into a virtual Energy Management System (EMS) for peak load shaving and automated demand response.

```
+-------------------------------------------------------------------------+
|                          1. INGESTION LAYER                             |
|  - Smart Meter Interval Logs (Hourly kWh)                               |
|  - Ambient Weather Station Data (Dry-Bulb Temp °C, Humidity %)         |
|  - Facility Operational Schedules (Occupancy Rates, Weekday/Weekend)    |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|                      2. PREPROCESSING & FEATURE ENG                     |
|  - Datetime Parsing & Chronological Sorting                             |
|  - Cyclical Encoding: sin(2π*t/T), cos(2π*t/T) [Diurnal & Annual]       |
|  - Thermal Degree Metrics: Cooling Degree (CDD), Heating Degree (HDD)   |
|  - Autoregressive Lags: t-1, t-2, t-24 (yesterday), t-168 (last week)   |
|  - Past Shifted Rolling Stats: 24h Mean, Std, Min, Max (Zero Leakage)   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|                        3. MODELING & BENCHMARK                          |
|  - Chronological Partition: 80% Train | 20% Out-of-Sample Test          |
|  - Model A: Ridge Regression (L2 Regularized Parametric Baseline)       |
|  - Model B: Random Forest Regressor (Non-Linear Bagged Ensembles)       |
|  - Model C: HistGradientBoosting (Histogram-based Tree Boosting)        |
|  - Model Selection Criteria: Min Out-of-Sample RMSE                     |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|                     4. MULTI-STEP RECURSIVE FORECASTER                  |
|  - 48-Hour Ahead Dynamic Roll-Forward Engine                            |
|  - Autoregressive lag updates using newly predicted timesteps           |
|  - Forecast export to outputs/forecast_results.csv                      |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|                  5. VIRTUAL EMS & DEMAND RESPONSE ENGINE                |
|  - Time-of-Use (TOU) Tariff Mapping (Peak / Mid-Peak / Off-Peak)        |
|  - Peak Alert Threshold Monitoring (Demand Cap at 60th Percentile)      |
|  - Battery Energy Storage System (BESS) Simulated Peak Shaving (25 kWh) |
|  - Financial Impact: Volumetric Energy Savings + Peak Demand Reduction  |
|  - Environmental Impact: Avoided Scope 2 Carbon Emissions (kg CO2)      |
+-------------------------------------------------------------------------+
```

## 2. Feature Engineering Logic

| Feature | Formula / Origin | Physical Justification |
| :--- | :--- | :--- |
| `sin_hour`, `cos_hour` | $\sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$ | Preserves the continuous circular distance between 23:00 and 00:00 |
| `sin_month`, `cos_month` | $\sin\left(\frac{2\pi \cdot \text{month}}{12}\right), \cos\left(\frac{2\pi \cdot \text{month}}{12}\right)$ | Captures annual climate seasonality (Summer vs Winter) |
| `cdd` | $\max(0, T - 22.0)$ | Cooling Degree Days: HVAC chillers turn on above 22°C |
| `hdd` | $\max(0, 15.0 - T)$ | Heating Degree Days: Building heating switches on below 15°C |
| `lag_1`, `lag_2` | $y_{t-1}, y_{t-2}$ | High short-term temporal autocorrelation |
| `lag_24` | $y_{t-24}$ | Strong 24-hour diurnal rhythm (same hour yesterday) |
| `lag_168` | $y_{t-168}$ | Weekly seasonality (e.g., Monday 10:00 vs prior Monday 10:00) |
| `rolling_mean_24` | $\frac{1}{24}\sum_{i=1}^{24} y_{t-i}$ | Captures daily baseline demand trend without lookahead leakage |

## 3. Evaluation Metrics

- **Root Mean Squared Error (RMSE):**
  $$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
  Penalizes large demand prediction misses, crucial for avoiding grid blackouts or costly peak tariff spikes.

- **Mean Absolute Error (MAE):**
  $$\text{MAE} = \frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$$
  Measures average absolute magnitude of forecasting error in kWh.

- **Mean Absolute Percentage Error (MAPE):**
  $$\text{MAPE} = \frac{100\%}{N}\sum_{i=1}^N \left|\frac{y_i - \hat{y}_i}{y_i}\right|$$
  Provides an intuitive percentage metric easily understood by business executives.

- **Coefficient of Determination ($R^2$):**
  $$R^2 = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
  Represents proportion of total demand variance explained by the model.
