WITH latest_seq AS (
    SELECT 
        COALESCE(
            MAX(CASE WHEN sequence_no <= :simCycle THEN sequence_no END),
            MAX(sequence_no)
        ) AS eval_seq,
        MAX(sequence_no) AS max_seq
    FROM telemetry
    WHERE asset_id = :assetId
),
latest_pred AS (
    SELECT 
        estimate AS end_rul,
        lower_bound AS end_lower_90,
        upper_bound AS end_upper_90
    FROM prediction
    WHERE asset_id = :assetId
    ORDER BY sequence_no DESC, prediction_id DESC
    LIMIT 1
)
SELECT 
    t.asset_id,
    t.sequence_no AS current_cycle,
    MAX(t.observed_at) AS last_observed_at,
    MAX(CASE WHEN t.signal_key = 's1' THEN t.value_double END) AS fan_inlet_temp_s1,
    MAX(CASE WHEN t.signal_key = 's2' THEN t.value_double END) AS lpc_temp_s2,
    MAX(CASE WHEN t.signal_key = 's3' THEN t.value_double END) AS hpc_temp_s3,
    MAX(CASE WHEN t.signal_key = 's4' THEN t.value_double END) AS lpt_temp_s4,
    MAX(CASE WHEN t.signal_key = 's7' THEN t.value_double END) AS hpc_press_s7,
    MAX(CASE WHEN t.signal_key = 's8' THEN t.value_double END) AS fan_speed_s8,
    MAX(CASE WHEN t.signal_key = 's9' THEN t.value_double END) AS core_speed_s9,
    MAX(CASE WHEN t.signal_key = 's11' THEN t.value_double END) AS hpc_static_press_s11,
    MAX(CASE WHEN t.signal_key = 's12' THEN t.value_double END) AS fuel_flow_s12,
    MAX(CASE WHEN t.signal_key = 's21' THEN t.value_double END) AS lpt_bleed_s21,
    CASE 
        WHEN MAX(CASE WHEN t.signal_key = 's4' THEN t.value_double END) >= 1418.0 THEN 'Critico'
        WHEN MAX(CASE WHEN t.signal_key = 's4' THEN t.value_double END) >= 1408.0 THEN 'Alerta'
        ELSE 'Normal'
    END AS health_status,
    LEAST(125.0, COALESCE(MAX(lp.end_rul), 0.0) + (MAX(ls.max_seq) - t.sequence_no)) AS rul_cycles,
    LEAST(125.0, GREATEST(0.0, COALESCE(MAX(lp.end_lower_90), 0.0) + (MAX(ls.max_seq) - t.sequence_no))) AS rul_lower_90,
    LEAST(125.0, COALESCE(MAX(lp.end_upper_90), 0.0) + (MAX(ls.max_seq) - t.sequence_no)) AS rul_upper_90,
    CASE 
        WHEN LEAST(125.0, COALESCE(MAX(lp.end_rul), 0.0) + (MAX(ls.max_seq) - t.sequence_no)) <= 15.0 THEN 'Parada Inmediata'
        WHEN LEAST(125.0, COALESCE(MAX(lp.end_rul), 0.0) + (MAX(ls.max_seq) - t.sequence_no)) <= 45.0 THEN 'Mantenimiento Requerido'
        ELSE 'Operación Segura'
    END AS prescriptive_action
FROM telemetry t
JOIN latest_seq ls ON t.sequence_no = ls.eval_seq
LEFT JOIN latest_pred lp ON TRUE
WHERE t.asset_id = :assetId
GROUP BY t.asset_id, t.sequence_no;
