import pandas as pd
import numpy as np

# =========================
# 0) CONFIG
# =========================
DATA_PATH = "raw_datasets/logistics_dataset_industrial_100k_v4.csv"
TARGET_COL = "Maintenance_Required"

# Columns that are identifiers or date-like (not for correlation directly)
ID_COLS = ["Vehicle_ID"]
DATE_COLS = ["Last_Maintenance_Date", "Snapshot_Date"]

# Potential leakage / suspicious columns (keep here for detection)
POTENTIAL_LEAKAGE = [
    "Maintenance_Type",        # may encode post-event info
    "Maintenance_Cost",        # may be post-event cost
    "Impact_on_Efficiency",    # may be post-event effect
    "Predictive_Score",        # already a model output
    "Downtime_Maintenance",    # can be post-maintenance signal
    "Failure_History", "Failure_Flag",
    "Anomalies_Detected", "Anomaly_Flag",
]

# =========================
# 1) LOAD DATA
# =========================
df = pd.read_csv(DATA_PATH)

print("\n=========================")
print("1) BASIC INFO")
print("=========================")
print("Shape:", df.shape)
print("\nColumns:", list(df.columns))

# =========================
# 2) QUICK DATA QUALITY CHECKS
# =========================
print("\n=========================")
print("2) DATA QUALITY")
print("=========================")

# Missing values
missing_ratio = df.isna().mean().sort_values(ascending=False)
print("\nTop missing ratios:")
print(missing_ratio.head(15))

# Duplicate rows (full row duplicates)
dup_count = df.duplicated().sum()
print("\nDuplicate rows:", dup_count)

# Constant columns (nunique=1)
nunique = df.nunique(dropna=False)
constant_cols = nunique[nunique <= 1].index.tolist()
print("\nConstant columns (nunique<=1):", constant_cols)

# =========================
# 3) TARGET ANALYSIS
# =========================
print("\n=========================")
print("3) TARGET ANALYSIS")
print("=========================")

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

target_counts = df[TARGET_COL].value_counts(dropna=False)
target_ratio = df[TARGET_COL].value_counts(normalize=True, dropna=False)

print("\nTarget counts:")
print(target_counts)
print("\nTarget ratio:")
print(target_ratio)

# =========================
# 4) TYPE CHECKS + COERCIONS FOR ANALYSIS
# =========================
print("\n=========================")
print("4) DATA TYPES")
print("=========================")

print(df.dtypes)

# Convert date columns (if present)
for c in DATE_COLS:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")

# =========================
# 5) LEAKAGE CHECK (PRESENCE)
# =========================
print("\n=========================")
print("5) LEAKAGE PRESENCE CHECK")
print("=========================")

present_leakage = [c for c in POTENTIAL_LEAKAGE if c in df.columns]
print("Suspicious columns present:", present_leakage)

# You can decide to drop them for modeling later, but here we only detect them.

# =========================
# 6) CORRELATION WITH TARGET (NUMERIC ONLY)
# =========================
print("\n=========================")
print("6) CORRELATION WITH TARGET (NUMERIC)")
print("=========================")

# Exclude ID columns from correlation
df_corr = df.drop(columns=[c for c in ID_COLS if c in df.columns], errors="ignore")

# Correlation (numeric only)
corr_series = df_corr.corr(numeric_only=True)[TARGET_COL].sort_values(ascending=False)

print("\nCorrelation with target (sorted):")
print(corr_series)

# Flag suspiciously high correlations
suspicious = corr_series[abs(corr_series) >= 0.60]
print("\nSuspicious correlations (|corr| >= 0.60) -- potential leakage:")
print(suspicious if len(suspicious) else "None")

# =========================
# 7) ENGINE TEMPERATURE SANITY
# =========================
print("\n=========================")
print("7) ENGINE_TEMPERATURE SANITY CHECK")
print("=========================")

if "Engine_Temperature" in df.columns:
    et = pd.to_numeric(df["Engine_Temperature"], errors="coerce")
    print("Engine_Temperature dtype:", df["Engine_Temperature"].dtype)
    print("Null ratio:", et.isna().mean())
    print("Unique values:", et.nunique())
    print("Min/Max/Mean/Std:", float(et.min()), float(et.max()), float(et.mean()), float(et.std()))
else:
    print("Engine_Temperature column not found.")

# =========================
# 8) GROUP STATS BY TARGET (CHECK REALISTIC SEPARATION)
# =========================
print("\n=========================")
print("8) GROUP STATS BY TARGET (NUMERIC FEATURES)")
print("=========================")

# Pick a useful set of numeric columns to compare between classes
candidate_numeric = [
    "Engine_Temperature",
    "Vibration_Levels",
    "Fuel_Consumption",
    "Usage_Hours",
    "Load_Ratio",
    "Tire_Pressure",
    "Oil_Quality",
    "Brake_Condition_Score",
    "Battery_Status_Score",
    "Days_Since_Last_Maintenance",
    "Vehicle_Age_Years",
    "Total_Operating_Hours",
    "Lifetime_Maintenance_Count",
    "Lifetime_Failure_Count",
    "Lifetime_Downtime_Hours",
    "Maintenance_Overdue_Flag",
]

available_numeric = [c for c in candidate_numeric if c in df.columns]

for c in available_numeric:
    s = pd.to_numeric(df[c], errors="coerce")
    df_tmp = pd.DataFrame({TARGET_COL: df[TARGET_COL], c: s})
    grouped = df_tmp.groupby(TARGET_COL)[c].agg(["count", "mean", "std", "min", "max"])
    print(f"\n--- {c} (by target) ---")
    print(grouped)

# =========================
# 9) CATEGORICAL DISTRIBUTIONS (OPTIONAL BUT USEFUL)
# =========================
print("\n=========================")
print("9) CATEGORICAL DISTRIBUTIONS")
print("=========================")

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Remove IDs if they are object
cat_cols = [c for c in cat_cols if c not in ID_COLS]

print("Categorical columns:", cat_cols)

# Show top values for each categorical column
for c in cat_cols:
    print(f"\n--- {c} top values ---")
    print(df[c].value_counts(dropna=False).head(10))

# =========================
# 10) TIME RANGE CHECK (IF DATES EXIST)
# =========================
print("\n=========================")
print("10) DATE RANGE CHECK")
print("=========================")

for c in DATE_COLS:
    if c in df.columns:
        print(f"{c}: min={df[c].min()}, max={df[c].max()}, null_ratio={df[c].isna().mean()}")

# =========================
# 11) SIMPLE DATASET READINESS CHECKLIST
# =========================
print("\n=========================")
print("11) READINESS CHECKLIST")
print("=========================")

checks = []

# Check target exists
checks.append(("Target column exists", TARGET_COL in df.columns))

# Check target is binary
unique_target = df[TARGET_COL].dropna().unique()
checks.append(("Target is binary-like", set(unique_target).issubset({0, 1})))

# Check no constant key features
key_features = ["Engine_Temperature", "Vibration_Levels", "Fuel_Consumption", "Usage_Hours", "Load_Ratio"]
for k in key_features:
    if k in df.columns:
        checks.append((f"{k} has variance", df[k].nunique() > 1))

# Print checks
for name, ok in checks:
    print(f"{name}: {'✅' if ok else '❌'}")

print("\nDone. Next step: use findings to finalize preprocessing + model pipeline.")