import pandas as pd

logistics_df = pd.read_csv("raw_datasets/logistics_dataset_with_maintenance_required.csv")
maintenance_df = pd.read_csv("raw_datasets/vehicle_maintenance_data.csv")



maintenance_filtered = maintenance_df[maintenance_df["Vehicle_Model"].isin(["Truck", "Van"])].copy()



def to_bool(s):
    return s.astype(str).str.lower().map({
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0
    })

maintenance_filtered["need_maint_flag"] = to_bool(maintenance_filtered["Need_Maintenance"])
maintenance_filtered["accident_flag"] = to_bool(maintenance_filtered["Accident_History"])




numeric_cols = [
    "Mileage",
    "Vehicle_Age",
    "Odometer_Reading",
    "Insurance_Premium",
    "Fuel_Efficiency",
    "Engine_Size"
]

for col in numeric_cols:
    maintenance_filtered[col] = pd.to_numeric(
        maintenance_filtered[col], errors="coerce"
    )



agg = maintenance_filtered.groupby("Vehicle_Model").agg(
    avg_mileage=("Mileage", "mean"),
    avg_vehicle_age=("Vehicle_Age", "mean"),
    avg_odometer=("Odometer_Reading", "mean"),
    maintenance_rate=("need_maint_flag", "mean"),
    accident_rate=("accident_flag", "mean"),
    avg_insurance_premium=("Insurance_Premium", "mean"),
    avg_fuel_efficiency=("Fuel_Efficiency", "mean"),
    avg_engine_size=("Engine_Size", "mean"),
).reset_index()



agg.rename(columns={"Vehicle_Model": "Vehicle_Type"}, inplace=True)



unified_df = logistics_df.merge(
    agg,
    on="Vehicle_Type",
    how="left"
)



unified_df.to_csv("unified_dataset/predictix_unified_pdm_dataset.csv", index=False)