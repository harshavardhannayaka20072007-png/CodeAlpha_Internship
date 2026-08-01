import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 1. Load the Dataset
dataset_path = "task2/Unemployment_Rate_upto_11_2020.csv"
try:
  df = pd.read_csv(dataset_path)
  print("[INFO] Dataset successfully loaded!")
except FileNotFoundError:
  print(f"[ERROR] Could not find '{dataset_path}'. Please check the filename.")
  exit()

# 2. Data Cleaning & Preprocessing
print("\n--- Cleaning Data ---")
df.columns = df.columns.str.strip()

df = df.dropna().drop_duplicates()
df["Region"] = df["Region"].str.strip()

# Clean date formatting with dayfirst=True to avoid warnings
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

print(f"Cleaned dataset shape: {df.shape}")

# 3. Summary Statistics
print("\n--- Summary Statistics ---")
print(
    df[
        [
            "Estimated Unemployment Rate (%)",
            "Estimated Employed",
            "Estimated Labour Participation Rate (%)",
        ]
    ].describe()
)

# 4. Visualizations
sns.set_theme(style="whitegrid")

# Plot 1: Unemployment Trends Over Time (Covid-19 Impact)
plt.figure(figsize=(14, 6))
time_trend = (
    df.groupby("Date")["Estimated Unemployment Rate (%)"].mean().reset_index()
)
sns.lineplot(
    data=time_trend,
    x="Date",
    y="Estimated Unemployment Rate (%)",
    marker="o",
    color="r",
)
plt.title(
    "Overall Unemployment Rate Trend Over Time (Covid-19 Impact)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Date")
plt.ylabel("Estimated Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("covid_unemployment_trend.png")
plt.close()

# Plot 2: Average Unemployment Rate by Region
plt.figure(figsize=(12, 6))
regional_trend = (
    df.groupby("Region")["Estimated Unemployment Rate (%)"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
sns.barplot(
    data=regional_trend,
    x="Estimated Unemployment Rate (%)",
    y="Region",
    palette="viridis",
)
plt.title(
    "Average Unemployment Rate (%) by Region", fontsize=14, fontweight="bold"
)
plt.xlabel("Average Unemployment Rate (%)")
plt.ylabel("Region")
plt.tight_layout()
plt.savefig("regional_unemployment.png")
plt.close()

print(
    "\n[INFO] Analysis complete! Charts saved as 'covid_unemployment_trend.png'"
    " and 'regional_unemployment.png'."
)