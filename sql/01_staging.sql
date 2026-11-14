-- =============================================================================
-- 01 · STAGING : chargement massif des fichiers CSV Météo-France
-- -----------------------------------------------------------------------------
-- * Lecture directe des .csv.gz par DuckDB (lecture parallèle, sans pandas).
-- * Tout est lu en texte (all_varchar) puis converti avec TRY_CAST : une valeur
--   mal formée devient NULL au lieu de faire échouer tout le chargement.
-- * Les codes qualité Météo-France (Q*) sont conservés pour l'étape suivante :
--     9 = filtrée, 0 = protégée, 1 = validée, 2 = douteuse.
-- =============================================================================
CREATE OR REPLACE TABLE stg_daily AS
SELECT
    NUM_POSTE                                    AS station_id,  -- texte : garde le 0 initial (06088001)
    regexp_extract(filename, 'Q_([0-9AB]+)_', 1) AS dep,
    NOM_USUEL                                    AS station_name,
    TRY_CAST(LAT  AS DOUBLE)                     AS lat,
    TRY_CAST(LON  AS DOUBLE)                     AS lon,
    TRY_CAST(ALTI AS INTEGER)                    AS altitude,
    TRY_STRPTIME(AAAAMMJJ, '%Y%m%d')::DATE       AS date,
    TRY_CAST(RR   AS DOUBLE)   AS rr,   TRY_CAST(QRR  AS UTINYINT) AS q_rr,   -- pluie (mm)
    TRY_CAST(TN   AS DOUBLE)   AS tn,   TRY_CAST(QTN  AS UTINYINT) AS q_tn,   -- T° min (°C)
    TRY_CAST(TX   AS DOUBLE)   AS tx,   TRY_CAST(QTX  AS UTINYINT) AS q_tx,   -- T° max (°C)
    TRY_CAST(TM   AS DOUBLE)   AS tm,   TRY_CAST(QTM  AS UTINYINT) AS q_tm,   -- T° moyenne horaire (°C)
    TRY_CAST(FFM  AS DOUBLE)   AS ffm,  TRY_CAST(QFFM AS UTINYINT) AS q_ffm,  -- vent moyen (m/s)
    TRY_CAST(FXI  AS DOUBLE)   AS fxi,  TRY_CAST(QFXI AS UTINYINT) AS q_fxi,  -- rafale max (m/s)
    parse_filename(filename)                     AS source_file
FROM read_csv(${files}, delim = ';', header = true, all_varchar = true,
              filename = true, union_by_name = true);
