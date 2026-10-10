"""
notebooks/exploratory_data_analysis.py
Interactive Exploratory Data Analysis & Diagnostic Script

Can be executed in VS Code Interactive Window or exported to .ipynb.
"""

# %% [markdown]
# # AI-Powered Energy Consumption Forecasting - Exploratory Data Analysis
# This notebook inspects the raw energy time series, analyzes diurnal and seasonal patterns,
# evaluates correlation with weather and occupancy, and prepares features for ML modeling.

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Load Raw Energy Consumption Data
df = pd.read_csv("../data/raw/energy_consumption_raw.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
print("Dataset Shape:", df.shape)
print("Date Range:", df["timestamp"].min(), "to", df["timestamp"].max())
df.head()

# %%
# Statistical Summary
df.describe().T

# %%
# Correlation Matrix
corr = df.drop(columns=["timestamp"]).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix: Energy vs Ambient Features")
plt.show()

# %%
# Diurnal Profile by Day of Week
df["hour"] = df["timestamp"].dt.hour
df["day_name"] = df["timestamp"].dt.day_name()
pivot = df.pivot_table(index="hour", columns="day_name", values="energy_consumption_kwh", aggfunc="mean")
days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
pivot = pivot[days_order]

plt.figure(figsize=(12, 6))
sns.heatmap(pivot.T, cmap="viridis", annot=False)
plt.title("Hourly Energy Intensity Heatmap by Day of Week")
plt.xlabel("Hour of Day")
plt.ylabel("Day")
plt.show()
