# Sri Lanka Single-Warehouse Vehicle Maintenance Dataset V9

Rows: 100,000
Columns: 69
Maintenance rate: 0.360
Median days until maintenance: 95
Median days when maintenance required: 6
Median days when no maintenance required in next 30d: 112

Enhancements:
- Added realistic `days_until_next_maintenance` target.
- Added `predicted_next_maintenance_date` derived from snapshot date.
- Timing is tied to service overdue pressure, engine-hours since service, subsystem health, telematics anomalies, workload, vehicle role, and priority.
- Severe-condition rows are pulled closer to maintenance windows to avoid unrealistic late dates.
