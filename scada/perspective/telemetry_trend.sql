SELECT 
    observed_at,
    MAX(CASE WHEN signal_key = 's4' THEN value_double END) AS temp_salida_lpt_s4,
    MAX(CASE WHEN signal_key = 's3' THEN value_double END) AS temp_salida_hpc_s3
FROM telemetry
WHERE asset_id = :assetId
  AND sequence_no BETWEEN :startCycle AND :endCycle
GROUP BY observed_at, sequence_no
ORDER BY sequence_no ASC;
