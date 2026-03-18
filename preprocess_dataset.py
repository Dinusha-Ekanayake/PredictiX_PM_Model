import pandas as pd
import numpy as np

# =========================
# 0) CONFIG
# =========================
INPUT_PATH  = "raw_datasets/logistics_dataset_industrial_100k_v4.csv"
OUTPUT_PATH = "processed_datasets/logistics_clean_model_ready.csv"

TARGET_COL = "Maintenance_Required"

# Leakage / post-event / reactive columns to REMOVE from model features
LEAKAGE_COLS = [
    "Maintenance_Type",
    "Maintenance_Cost",
    "Impact_on_Efficiency",
    "Predictive_Score",
    "Downtime_Maintenance",
    "Failure_History",
    "Failure_Flag",
    "Anomalies_Detected",
    "Anomaly_Flag",
]

# Columns not useful for prediction (identifiers / raw dates)
DROP_NON_FEATURE_COLS = [
    "Vehicle_ID",             # identifier, causes leakage by memorization
    "Last_Maintenance_Date",  # keep derived values like Days_Since_Last_Maintenance instead
]

DATE_COL_FOR_SPLIT = "Snapshot_Date"  # we will keep this ONLY to do time-split, then drop it from features


# =========================
# 1) LOAD
# =========================
df = pd.read_csv(INPUT_PATH)
print("Loaded:", df.shape)

# =========================
# 2) TARGET CHECK
# =========================
if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

# =========================
# 3) DROP LEAKAGE + NON-FEATURE COLS
# =========================
df = df.drop(columns=LEAKAGE_COLS, errors="ignore")
df = df.drop(columns=DROP_NON_FEATURE_COLS, errors="ignore")

# Keep Snapshot_Date temporarily for time split
df[DATE_COL_FOR_SPLIT] = pd.to_datetime(df[DATE_COL_FOR_SPLIT], errors="coerce")

# =========================
# 4) CLEAN NUMERIC VALUES (REALISM FIX)
# =========================
# Usage_Hours cannot be negative
if "Usage_Hours" in df.columns:
    df["Usage_Hours"] = pd.to_numeric(df["Usage_Hours"], errors="coerce").fillna(0)
    df["Usage_Hours"] = df["Usage_Hours"].clip(lower=0)

# Load_Ratio: should be >=0
if "Load_Ratio" in df.columns:
    df["Load_Ratio"] = pd.to_numeric(df["Load_Ratio"], errors="coerce")
    df["Load_Ratio"] = df["Load_Ratio"].clip(lower=0)

# Engine_Temperature plausible bounds
if "Engine_Temperature" in df.columns:
    df["Engine_Temperature"] = pd.to_numeric(df["Engine_Temperature"], errors="coerce")
    df["Engine_Temperature"] = df["Engine_Temperature"].clip(lower=60, upper=125)

# Tire pressure plausible bounds
if "Tire_Pressure" in df.columns:
    df["Tire_Pressure"] = pd.to_numeric(df["Tire_Pressure"], errors="coerce")
    df["Tire_Pressure"] = df["Tire_Pressure"].clip(lower=15, upper=70)

# Vibration should not be negative (your min was -0.60)
if "Vibration_Levels" in df.columns:
    df["Vibration_Levels"] = pd.to_numeric(df["Vibration_Levels"], errors="coerce")
    df["Vibration_Levels"] = df["Vibration_Levels"].clip(lower=0)

# Oil quality reasonable bounds
if "Oil_Quality" in df.columns:
    df["Oil_Quality"] = pd.to_numeric(df["Oil_Quality"], errors="coerce")
    df["Oil_Quality"] = df["Oil_Quality"].clip(lower=0, upper=120)

# Any remaining NaNs in numeric columns → fill with median (safe baseline)
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove(TARGET_COL)

for c in num_cols:
    df[c] = df[c].fillna(df[c].median())

# =========================
# 5) CATEGORICAL HANDLING
# =========================
# Identify categorical columns
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove target if mis-typed (should be int)
if TARGET_COL in cat_cols:
    cat_cols.remove(TARGET_COL)

print("Categorical columns to one-hot:", cat_cols)

# One-hot encode (drop_first reduces multicollinearity)
df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# =========================
# 6) TIME-AWARE SPLIT MARKER (NOT TRAINING SPLIT YET)
# =========================
# We'll store a boolean column to split later without leakage
split_date = df_encoded[DATE_COL_FOR_SPLIT].quantile(0.80)
df_encoded["is_test_period"] = (df_encoded[DATE_COL_FOR_SPLIT] > split_date).astype(int)

# Drop Snapshot_Date from features (it should NOT be fed into the model)
df_encoded = df_encoded.drop(columns=[DATE_COL_FOR_SPLIT], errors="ignore")

# =========================
# 7) SAVE
# =========================
import os
os.makedirs("processed_datasets", exist_ok=True)

df_encoded.to_csv(OUTPUT_PATH, index=False)
print("Saved clean dataset:", OUTPUT_PATH)
print("Final shape:", df_encoded.shape)
print("Target ratio:", df_encoded[TARGET_COL].value_counts(normalize=True).to_dict())
print("Test period ratio:", df_encoded["is_test_period"].value_counts(normalize=True).to_dict())