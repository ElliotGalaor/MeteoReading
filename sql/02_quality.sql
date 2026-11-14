-- =============================================================================
-- 02 · QUALITY: validation rules applied to every measurement
-- -----------------------------------------------------------------------------
-- Each value gets a status, in this order of priority:
--   missing       -> no measurement
--   suspect       -> Météo-France quality code = 2 (flagged, not validated)
--   out_of_range  -> physically impossible in France (wide bounds, beyond the records)
--   inconsistent  -> TN > TX on the same day
--   valid         -> kept for the analysis
-- The bounds are deliberately wide (beyond the national records, e.g. TX 46.0 °C in 2019)
-- so that a genuine extreme value is never rejected: only gross errors are targeted.
-- =============================================================================
CREATE OR REPLACE VIEW obs_flags AS
SELECT
    station_id, dep, date, tn, tx, tm, rr, ffm, fxi,
    CASE WHEN tn IS NULL THEN 'missing'
         WHEN q_tn = 2 THEN 'suspect'
         WHEN tn NOT BETWEEN -45 AND 35 THEN 'out_of_range'
         WHEN tn > tx THEN 'inconsistent'
         ELSE 'valid' END AS tn_status,
    CASE WHEN tx IS NULL THEN 'missing'
         WHEN q_tx = 2 THEN 'suspect'
         WHEN tx NOT BETWEEN -30 AND 50 THEN 'out_of_range'
         WHEN tn > tx THEN 'inconsistent'
         ELSE 'valid' END AS tx_status,
    CASE WHEN tm IS NULL THEN 'missing'
         WHEN q_tm = 2 THEN 'suspect'
         WHEN tm NOT BETWEEN -40 AND 40 THEN 'out_of_range'
         ELSE 'valid' END AS tm_status,
    CASE WHEN rr IS NULL THEN 'missing'
         WHEN q_rr = 2 THEN 'suspect'
         WHEN rr NOT BETWEEN 0 AND 700 THEN 'out_of_range'
         ELSE 'valid' END AS rr_status,
    CASE WHEN ffm IS NULL THEN 'missing'
         WHEN q_ffm = 2 THEN 'suspect'
         WHEN ffm NOT BETWEEN 0 AND 60 THEN 'out_of_range'
         ELSE 'valid' END AS ffm_status,
    CASE WHEN fxi IS NULL THEN 'missing'
         WHEN q_fxi = 2 THEN 'suspect'
         WHEN fxi NOT BETWEEN 0 AND 90 THEN 'out_of_range'
         ELSE 'valid' END AS fxi_status
FROM stg_daily;

-- "Clean" view: only valid values are kept (the others become NULL)
CREATE OR REPLACE VIEW obs_daily AS
SELECT
    station_id, dep, date,
    CASE WHEN tn_status  = 'valid' THEN tn  END AS tn,
    CASE WHEN tx_status  = 'valid' THEN tx  END AS tx,
    CASE WHEN tm_status  = 'valid' THEN tm  END AS tm,
    CASE WHEN rr_status  = 'valid' THEN rr  END AS rr,
    CASE WHEN ffm_status = 'valid' THEN ffm END AS ffm,
    CASE WHEN fxi_status = 'valid' THEN fxi END AS fxi
FROM obs_flags;

-- Quality summary: number of values per variable and per status
CREATE OR REPLACE TABLE dq_summary AS
          SELECT 'TN'  AS variable, tn_status  AS status, count(*) AS n FROM obs_flags GROUP BY ALL
UNION ALL SELECT 'TX'  AS variable, tx_status  AS status, count(*) AS n FROM obs_flags GROUP BY ALL
UNION ALL SELECT 'TM'  AS variable, tm_status  AS status, count(*) AS n FROM obs_flags GROUP BY ALL
UNION ALL SELECT 'RR'  AS variable, rr_status  AS status, count(*) AS n FROM obs_flags GROUP BY ALL
UNION ALL SELECT 'FFM' AS variable, ffm_status AS status, count(*) AS n FROM obs_flags GROUP BY ALL
UNION ALL SELECT 'FXI' AS variable, fxi_status AS status, count(*) AS n FROM obs_flags GROUP BY ALL;

-- Duplicates (same station, same day): reported here, removed in the gold layer
CREATE OR REPLACE TABLE dq_duplicates AS
SELECT station_id, date, count(*) AS n
FROM stg_daily
GROUP BY ALL
HAVING count(*) > 1;

-- How the measurement network changes over time
CREATE OR REPLACE TABLE network_yearly AS
SELECT
    year(date)                                              AS year,
    count(DISTINCT station_id)                              AS n_stations,
    count(DISTINCT station_id) FILTER (WHERE tx IS NOT NULL) AS n_stations_temp,
    count(DISTINCT station_id) FILTER (WHERE rr IS NOT NULL) AS n_stations_rain,
    count(*)                                                AS n_rows
FROM stg_daily
WHERE date IS NOT NULL
GROUP BY ALL
ORDER BY year;
