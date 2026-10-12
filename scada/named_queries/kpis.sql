-- twin/kpis — single latest row for the KPI cards
SELECT *
FROM machine_telemetry
ORDER BY timestamp DESC
LIMIT 1
