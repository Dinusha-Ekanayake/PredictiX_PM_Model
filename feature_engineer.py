import pandas as pd
import numpy as np

df = pd.read_csv("raw_datasets/logistics_dataset_industrial_100k.csv")

# Drop leakage columns directly from df
df_clean = df.drop(columns=[
    "Failure_Flag",
    "Failure_History",
    "Anomaly_Flag",
    "Anomalies_Detected",
    "Downtime_Maintenance",
    "Predictive_Score",
    "Impact_on_Efficiency",
    "Maintenance_Cost"
], errors="ignore")

print("Remaining Columns:")
print(df_clean.columns)

print("\nUpdated Correlation:")
print(df_clean.corr(numeric_only=True)["Maintenance_Required"].sort_values(ascending=False))

print("\nEngine Temp Info:")
print(df["Engine_Temperature"].dtype)
print("Null ratio:", df["Engine_Temperature"].isna().mean())
print("Unique values:", df["Engine_Temperature"].nunique())


