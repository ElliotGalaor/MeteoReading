-- =============================================================================
-- 01 · STAGING: bulk load of the Météo-France CSV files
-- -----------------------------------------------------------------------------
-- * DuckDB reads the .csv.gz files directly (parallel reads, no pandas).
-- * Everything is read as text (all_varchar) then converted with TRY_CAST: a
--   malformed value becomes NULL instead of failing the whole load.
-- * The Météo-France quality codes (Q*) are kept for the next step:
--     9 = filtered, 0 = protected, 1 = validated, 2 = suspect.
-- =============================================================================
CREATE OR REPLACE TABLE stg_daily AS
SELECT
    NUM_POSTE                                    AS station_id,  -- text: keeps the leading zero (06088001)
    regexp_extract(filename, 'Q_([0-9AB]+)_', 1) AS dep,
    NOM_USUEL                                    AS station_name,
    TRY_CAST(LAT  AS DOUBLE)                     AS lat,
    TRY_CAST(LON  AS DOUBLE)                     AS lon,
    TRY_CAST(ALTI AS INTEGER)                    AS altitude,
    TRY_STRPTIME(AAAAMMJJ, '%Y%m%d')::DATE       AS date,
    TRY_CAST(RR   AS DOUBLE)   AS rr,   TRY_CAST(QRR  AS UTINYINT) AS q_rr,   -- rainfall (mm)
    TRY_CAST(TN   AS DOUBLE)   AS tn,   TRY_CAST(QTN  AS UTINYINT) AS q_tn,   -- min temperature (°C)
    TRY_CAST(TX   AS DOUBLE)   AS tx,   TRY_CAST(QTX  AS UTINYINT) AS q_tx,   -- max temperature (°C)
    TRY_CAST(TM   AS DOUBLE)   AS tm,   TRY_CAST(QTM  AS UTINYINT) AS q_tm,   -- mean of hourly temperatures (°C)
    TRY_CAST(FFM  AS DOUBLE)   AS ffm,  TRY_CAST(QFFM AS UTINYINT) AS q_ffm,  -- mean wind speed (m/s)
    TRY_CAST(FXI  AS DOUBLE)   AS fxi,  TRY_CAST(QFXI AS UTINYINT) AS q_fxi,  -- max gust (m/s)
    parse_filename(filename)                     AS source_file
FROM read_csv(${files}, delim = ';', header = true, all_varchar = true,
              filename = true, union_by_name = true);
