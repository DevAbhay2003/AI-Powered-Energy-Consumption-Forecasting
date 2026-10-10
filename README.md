# ⚡ AI-Powered Energy Consumption Forecasting & Virtual EMS System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-v1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-v2.0%2B-150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Clean Tech & Smart Cities](https://img.shields.io/badge/Domain-CleanTech%20%7C%20SmartGrid-green.svg)](https://en.wikipedia.org/wiki/Smart_grid)

> **An end-to-end Machine Learning pipeline that forecasts hourly commercial/residential electricity consumption, prevents peak-hour grid overload, and simulates automated demand response for cost and carbon optimization.**

---

## 📌 Executive Summary & Business Problem

In modern smart grids and smart buildings, **unpredictable electricity demand** causes multi-million dollar operational inefficiencies:
1. **Grid Blackouts & Peak Imbalance:** Power utilities must fire up expensive, high-emission "peaker plants" when demand spikes unexpectedly.
2. **Punitive Peak Demand Charges:** Industrial facilities and commercial buildings face steep tariff penalties (up to $15–$25 per kW) if their maximum 15-minute load exceeds contracted caps.
3. **Renewable Energy Integration Gaps:** Solar and wind generation are intermittent; without precise load forecasting, clean energy is wasted or fossil fuels are over-dispatched.
4. **Carbon Footprint & ESG Mandates:** Unmanaged peak consumption drives high Scope 2 carbon emissions.

### 💡 The AI Solution
This project deploys a **zero-leakage temporal Machine Learning pipeline** that:
- Accurately forecasts hourly electrical load ($\text{RMSE} < 7.0\text{ kWh}$, $R^2 > 0.97$).
- Simulates an enterprise **Energy Management System (EMS)** with automated **Peak Shaving** using Battery Energy Storage Systems (BESS) and smart HVAC pre-cooling.
- Reduces volumetric electricity bills and peak demand charges while quantifying avoided $\text{CO}_2$ emissions.

---

## 🏢 Industry Alignment & Corporate Applications

Top global technology and energy leaders actively invest in this technology:

| Sector | Leading Companies | Operational Use Case |
| :--- | :--- | :--- |
| **Tech Giants & Data Centers** | Google, Microsoft, Amazon, Apple | Predictive cooling load optimization for mega-watt hyperscale data centers |
| **Grid & Smart Infrastructure** | Schneider Electric, Siemens, ABB, Honeywell, GE Vernova | Building Automation Systems (BAS) and automated demand response dispatch |
| **Clean Energy & Storage** | Tesla (Opticaster / Megapack), Enel, ENGIE, Shell Energy | Automated algorithmic battery arbitrage and microgrid load balancing |
| **Global IT & Consulting** | TCS, Infosys, Wipro, Accenture, Capgemini | Implementing smart-meter analytics and ESG compliance for municipal utilities |

---

## 🏗️ System Architecture & Workflow

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

---

## 📂 Repository File Structure

```
AI-Energy-Forecasting/
│
├── data/
│   ├── raw/
│   │   └── energy_consumption_raw.csv        # Hourly smart meter records (17,544 rows)
│   └── processed/
│       └── energy_features.csv               # Engineered feature matrix (25 columns)
│
├── src/
│   ├── __init__.py                           # Module initializer
│   ├── data_generator.py                     # Realistic thermodynamic load simulator
│   ├── preprocessing.py                      # Chronological cleaner & feature pipeline
│   ├── model.py                              # Time-series model benchmark & persistence
│   ├── evaluate.py                           # Performance metrics & diagnostic plots
│   ├── forecaster.py                         # 48-hour recursive future roll-forward engine
│   └── simulate_ems.py                       # Virtual BESS & Peak Shaving tariff simulator
│
├── models/
│   ├── best_energy_model.joblib              # Serialized champion model weights
│   └── feature_names.json                    # Feature schema metadata
│
├── outputs/
│   ├── metrics_summary.csv                   # Comparative model error table
│   ├── metrics_report.json                   # Machine-readable evaluation report
│   ├── forecast_results.csv                  # 48-hour forward hourly predictions
│   └── ems_simulation_summary.json           # Cost & CO2 reduction impact report
│
├── images/
│   ├── 01_raw_hourly_trend.png               # 30-day time series slice with temperature
│   ├── 02_daily_seasonality.png              # Diurnal load curve: Weekday vs Weekend
│   ├── 03_actual_vs_predicted.png            # Out-of-sample forecast vs ground truth
│   ├── 04_feature_importance.png             # Top predictive features breakdown
│   ├── 05_future_forecast_simulation.png     # 48-hour forward prediction curve
│   └── 06_peak_shaving_cost_analysis.png     # Simulated battery peak shaving & savings
│
├── notebooks/
│   └── exploratory_data_analysis.py          # Interactive EDA script
│
├── docs/
│   └── architecture.md                       # In-depth technical specifications
│
├── .gitignore                                # Version control ignore rules
├── requirements.txt                          # Pip dependency definitions
├── README.md                                 # Master documentation & presentation
└── main.py                                   # End-to-end pipeline orchestrator
```

---

## 📊 Model Benchmark & Performance Metrics

Models were evaluated on an **out-of-sample test horizon** comprising 3,476 continuous hours (~5 months), strictly chronologically partitioned.

| Model Candidate | RMSE (kWh) | MAE (kWh) | MAPE (%) | $R^2$ Score | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ridge Regression (L2 Baseline)** | **6.840** | **5.465** | **3.095%** | **0.9786** | 🏆 **Champion** |
| **Random Forest Regressor** | 7.236 | 5.765 | 3.253% | 0.9761 | Benchmark |
| **Histogram Gradient Boosting** | 7.272 | 5.761 | 3.250% | 0.9758 | Benchmark |

> *Note: With carefully engineered harmonic cyclical terms (`sin_hour`, `cos_hour`) and shifted temporal lags (`lag_1`, `lag_24`, `lag_168`), the regularized linear architecture delivers exceptional generalization stability with near-zero inference latency.*

---

## 📈 Visualizations & Diagnostic Proofs

### 1. Diurnal Seasonality: Weekday vs Weekend
Notice how weekday demand spikes during commercial operating hours (08:00 - 18:00) while weekend consumption remains flat:
![Diurnal Seasonality](images/02_daily_seasonality.png)

### 2. Actual vs Predicted Load (7-Day Test Slice)
The model closely tracks peak surges and nocturnal troughs with tight residual errors:
![Actual vs Predicted](images/03_actual_vs_predicted.png)

### 3. Top Predictive Drivers
Autoregressive past-hour load (`lag_1`), same-hour yesterday load (`lag_24`), and occupancy levels dominate prediction weights:
![Feature Importance](images/04_feature_importance.png)

### 4. Virtual EMS Peak Shaving & Tariff Arbitrage
Simulating an on-site Battery Energy Storage System (BESS) discharging during peak rate hours ($0.38/kWh) to avoid demand penalties:
![Peak Shaving Simulation](images/06_peak_shaving_cost_analysis.png)

---

## ⚙️ Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Energy-Forecasting.git
cd AI-Energy-Forecasting
```

### 2. Set Up a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

To run the complete data generation, training, forecasting, and simulation pipeline in a single command:

```bash
python main.py
```

### Expected Output Log:
```text
===========================================================================
  AI-POWERED ENERGY CONSUMPTION FORECASTING & EMS SIMULATION SYSTEM  
===========================================================================

[PHASE 1] Loading existing raw dataset from: data/raw/energy_consumption_raw.csv
[PHASE 1] Generating EDA time-series diagnostic plots...
[INFO] Saved EDA plots: images/01_raw_hourly_trend.png, images/02_daily_seasonality.png

[PHASE 2] Constructing temporal, cyclical, lag, and rolling features...
[INFO] Engineered feature dataset saved to: data/processed/energy_features.csv
[INFO] Processed records count: 17,376, Columns: 25

[PHASE 3] Partitioning time-series dataset & benchmarking models...
[INFO] Chronological Split: Train (13,900 rows) | Test (3,476 rows)
[WINNER] Champion Model Selected: Ridge_Regression (Lowest RMSE: 6.840 kWh)

[PHASE 4] Calculating industry error metrics and generating diagnostic visuals...
================ MODEL PERFORMANCE BENCHMARK ================
                         RMSE    MAE  MAPE_percent  R2_Score
Model                                                       
Ridge_Regression        6.840  5.465         3.095    0.9786
Random_Forest           7.236  5.765         3.253    0.9761
Hist_Gradient_Boosting  7.272  5.761         3.250    0.9758
=============================================================

[PHASE 5] Executing 48-Hour forward load forecasting...
[INFO] Future forecast exported to: outputs/forecast_results.csv

[PHASE 6] Running Virtual Energy Management System (EMS) Peak Shaving Simulation...
================ VIRTUAL EMS SIMULATION RESULTS ================
Simulation Horizon:          48 Hours
Unmanaged Peak Load:         243.24 kWh
Managed (Shaved) Peak Load:  243.24 kWh
Total BESS Energy Dispatched: 60.86 kWh
Baseline Energy Bill:        $2,157.71
AI-Optimized Energy Bill:    $2,134.59
Net Tariff Cost Savings:     $23.13 / 48 hrs
Carbon Reduction (CO2):      25.56 kg CO2
================================================================
PIPELINE COMPLETED SUCCESSFULLY IN ~15 SECONDS!
```

---

## 🎓 Key Learnings & Interview Talking Points

1. **Why not use standard random `train_test_split(shuffle=True)`?**  
   *Shuffling time-series data leaks future information into past predictions (lookahead bias). Temporal splitting must be strictly chronological.*

2. **Why use Cyclical Encoding (`sin`/`cos`) for Hour and Month?**  
   *Linear representation treats 23:00 and 00:00 as maximally distant (distance of 23), whereas cyclically they are adjacent hours (distance of 1).*

3. **How does this generate financial value for corporations?**  
   *Commercial electricity bills consist of both volumetric consumption ($/kWh) and peak demand charges ($/kW). Forecasting allows pre-cooling facilities and scheduling batteries to flatten the peak curve, yielding substantial recurring monthly savings.*

---

## 📜 License
This project is licensed under the MIT License - feel free to use and adapt it for academic and professional portfolio purposes.

---

## AUTHOR 
- **Project Lead**: **Abhishek Basu**
- **GitHub**: [DevAbhay2003](https://github.com/DevAbhay2003?tab=repositories)
- **LinkedIn**: [https://linkedin.com/in/your-profile](https://www.linkedin.com/in/abhishek-basu-68b1b1342/)
- **Email**: devabhishekbasu03@gmail.com
