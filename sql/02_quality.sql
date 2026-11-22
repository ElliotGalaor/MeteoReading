-- =============================================================================
-- 02 · QUALITY: validation rules applied to every measurement
-- -----------------------------------------------------------------------------
-- Each value gets a status, in this order of priority:
--   missing       -> no measurement
--   rejected      -> checked by hand and found wrong (config.yaml, rejected_values), while the
--                    raw file still holds that value
--   suspect       -> Météo-France quality code = 2 (flagged, not validated)
--   out_of_range  -> physically impossible in France (wide bounds, beyond the records)
--   inconsistent  -> TN > TX on the same day
--   valid         -> kept for the analysis
-- The bounds are deliberately wide (beyond the national records, e.g. TX 46.0 °C in 2019)
-- so that a genuine extreme value is never rejected: only gross errors are targeted.
-- =============================================================================
CREATE OR REPLACE VIEW obs_flags AS
WITH rejected AS (   -- one row per station and day, the rejected value in the column of its variable
    SELECT station_id, date,
           any_value(value) FILTER (WHERE variable = 'tn')  AS tn,
           any_value(value) FILTER (WHERE variable = 'tx')  AS tx,
           any_value(value) FILTER (WHERE variable = 'tm')  AS tm,
           any_value(value) FILTER (WHERE variable = 'rr')  AS rr,
           any_value(value) FILTER (WHERE variable = 'ffm') AS ffm,
           any_value(value) FILTER (WHERE variable = 'fxi') AS fxi
    FROM rejected_values
    GROUP BY station_id, date
)
SELECT
    s.station_id, s.dep, s.date, s.tn, s.tx, s.tm, s.rr, s.ffm, s.fxi,
    CASE WHEN s.tn IS NULL THEN 'missing'
         WHEN s.tn = r.tn THEN 'rejected'
         WHEN s.q_tn = 2 THEN 'suspect'
         WHEN s.tn NOT BETWEEN -45 AND 35 THEN 'out_of_range'
         WHEN s.tn > s.tx THEN 'inconsistent'
         ELSE 'valid' END AS tn_status,
    CASE WHEN s.tx IS NULL THEN 'missing'
         WHEN s.tx = r.tx THEN 'rejected'
         WHEN s.q_tx = 2 THEN 'suspect'
         WHEN s.tx NOT BETWEEN -30 AND 50 THEN 'out_of_range'
         WHEN s.tn > s.tx THEN 'inconsistent'
         ELSE 'valid' END AS tx_status,
    CASE WHEN s.tm IS NULL THEN 'missing'
         WHEN s.tm = r.tm THEN 'rejected'
         WHEN s.q_tm = 2 THEN 'suspect'
         WHEN s.tm NOT BETWEEN -40 AND 40 THEN 'out_of_range'
         ELSE 'valid' END AS tm_status,
    CASE WHEN s.rr IS NULL THEN 'missing'
         WHEN s.rr = r.rr THEN 'rejected'
         WHEN s.q_rr = 2 THEN 'suspect'
         WHEN s.rr NOT BETWEEN 0 AND 700 THEN 'out_of_range'
         ELSE 'valid' END AS rr_status,
    CASE WHEN s.ffm IS NULL THEN 'missing'
         WHEN s.ffm = r.ffm THEN 'rejected'
         WHEN s.q_ffm = 2 THEN 'suspect'
         WHEN s.ffm NOT BETWEEN 0 AND 60 THEN 'out_of_range'
         ELSE 'valid' END AS ffm_status,
    CASE WHEN s.fxi IS NULL THEN 'missing'
         WHEN s.fxi = r.fxi THEN 'rejected'
         WHEN s.q_fxi = 2 THEN 'suspect'
         WHEN s.fxi NOT BETWEEN 0 AND 90 THEN 'out_of_range'
         ELSE 'valid' END AS fxi_status
FROM stg_daily s
LEFT JOIN rejected r ON r.station_id = s.station_id AND r.date = s.date;

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

-- The values rejected by hand, with the raw value and whether it still matches (applied)
CREATE OR REPLACE TABLE dq_rejected AS
SELECT
    r.station_id, s.station_name, r.date, r.variable, r.value,
    CASE r.variable WHEN 'tn' THEN s.tn WHEN 'tx' THEN s.tx WHEN 'tm' THEN s.tm
                    WHEN 'rr' THEN s.rr WHEN 'ffm' THEN s.ffm WHEN 'fxi' THEN s.fxi END AS raw_value,
    coalesce(raw_value = r.value, false) AS applied,
    r.reason_en, r.reason_fr, r.source
FROM rejected_values r
LEFT JOIN stg_daily s ON s.station_id = r.station_id AND s.date = r.date
QUALIFY row_number() OVER (PARTITION BY r.station_id, r.date, r.variable) = 1
ORDER BY r.date;

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
