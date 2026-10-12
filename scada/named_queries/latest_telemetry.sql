-- twin/latest_telemetry — last N rows for tables/KPIs
-- Parameters: limit (Integer, default 100)
SELECT timestamp, product_id, machine_type, air_temp_k, process_temp_k,
       rpm, torque_nm, tool_wear_min, machine_failure
FROM machine_telemetry
ORDER BY timestamp DESC
LIMIT :limit
