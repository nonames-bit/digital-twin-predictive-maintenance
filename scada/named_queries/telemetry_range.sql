-- twin/telemetry_range — time window for the trend chart
-- Parameters: startTs (DateTime), endTs (DateTime)
SELECT timestamp, rpm, torque_nm, air_temp_k, process_temp_k, tool_wear_min
FROM machine_telemetry
WHERE timestamp BETWEEN :startTs AND :endTs
ORDER BY timestamp
