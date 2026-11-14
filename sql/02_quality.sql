-- =============================================================================
-- 02 · QUALITÉ : règles de validation de chaque mesure
-- -----------------------------------------------------------------------------
-- Chaque valeur reçoit un statut, dans cet ordre de priorité :
--   absente      -> pas de mesure
--   douteuse     -> code qualité Météo-France = 2 (mise en doute, non validée)
--   hors_bornes  -> physiquement impossible en France (bornes larges, au-delà des records)
--   incoherente  -> TN > TX le même jour
--   valide       -> conservée pour l'analyse
-- Bornes volontairement larges (au-delà des records nationaux, ex. TX 46,0 °C en 2019)
-- pour ne jamais rejeter une vraie valeur extrême : on ne vise que les erreurs grossières.
-- =============================================================================
CREATE OR REPLACE VIEW obs_flags AS
SELECT
    station_id, dep, date, tn, tx, tm, rr, ffm, fxi,
    CASE WHEN tn IS NULL THEN 'absente'
         WHEN q_tn = 2 THEN 'douteuse'
         WHEN tn NOT BETWEEN -45 AND 35 THEN 'hors_bornes'
         WHEN tn > tx THEN 'incoherente'
         ELSE 'valide' END AS tn_status,
    CASE WHEN tx IS NULL THEN 'absente'
         WHEN q_tx = 2 THEN 'douteuse'
         WHEN tx NOT BETWEEN -30 AND 50 THEN 'hors_bornes'
         WHEN tn > tx THEN 'incoherente'
         ELSE 'valide' END AS tx_status,
    CASE WHEN tm IS NULL THEN 'absente'
         WHEN q_tm = 2 THEN 'douteuse'
         WHEN tm NOT BETWEEN -40 AND 40 THEN 'hors_bornes'
         ELSE 'valide' END AS tm_status,
    CASE WHEN rr IS NULL THEN 'absente'
         WHEN q_rr = 2 THEN 'douteuse'
         WHEN rr NOT BETWEEN 0 AND 700 THEN 'hors_bornes'
         ELSE 'valide' END AS rr_status,
    CASE WHEN ffm IS NULL THEN 'absente'
         WHEN q_ffm = 2 THEN 'douteuse'
         WHEN ffm NOT BETWEEN 0 AND 60 THEN 'hors_bornes'
         ELSE 'valide' END AS ffm_status,
    CASE WHEN fxi IS NULL THEN 'absente'
         WHEN q_fxi = 2 THEN 'douteuse'
         WHEN fxi NOT BETWEEN 0 AND 90 THEN 'hors_bornes'
         ELSE 'valide' END AS fxi_status
FROM stg_daily;

-- Vue "propre" : seules les valeurs valides sont conservées (les autres -> NULL)
CREATE OR REPLACE VIEW obs_daily AS
SELECT
    station_id, dep, date,
    CASE WHEN tn_status  = 'valide' THEN tn  END AS tn,
    CASE WHEN tx_status  = 'valide' THEN tx  END AS tx,
    CASE WHEN tm_status  = 'valide' THEN tm  END AS tm,
    CASE WHEN rr_status  = 'valide' THEN rr  END AS rr,
    CASE WHEN ffm_status = 'valide' THEN ffm END AS ffm,
    CASE WHEN fxi_status = 'valide' THEN fxi END AS fxi
FROM obs_flags;

-- Bilan qualité : nombre de valeurs par variable et par statut
CREATE OR REPLACE TABLE dq_summary AS
          SELECT 'TN'  AS variable, tn_status  AS status, count(*) AS n FROM obs_flags GROUP BY ALL
UNION ALL SELECT 'TX'  AS variable, tx_status  AS status, count(*) AS n FROM obs_flags GROUP BY ALL
UNION ALL SELECT 'TM'  AS variable, tm_status  AS status, count(*) AS n FROM obs_flags GROUP BY ALL
UNION ALL SELECT 'RR'  AS variable, rr_status  AS status, count(*) AS n FROM obs_flags GROUP BY ALL
UNION ALL SELECT 'FFM' AS variable, ffm_status AS status, count(*) AS n FROM obs_flags GROUP BY ALL
UNION ALL SELECT 'FXI' AS variable, fxi_status AS status, count(*) AS n FROM obs_flags GROUP BY ALL;

-- Doublons (même station, même jour) : signalés ici, éliminés dans la couche gold
CREATE OR REPLACE TABLE dq_duplicates AS
SELECT station_id, date, count(*) AS n
FROM stg_daily
GROUP BY ALL
HAVING count(*) > 1;

-- Évolution du réseau de mesure dans le temps
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
