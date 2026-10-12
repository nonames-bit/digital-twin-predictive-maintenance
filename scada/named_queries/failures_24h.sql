-- twin/failures_24h — alarm banner counter
SELECT COUNT(*) AS failures
FROM machine_telemetry
WHERE machine_failure = 1
  AND timestamp > NOW() - INTERVAL '24 hours'
