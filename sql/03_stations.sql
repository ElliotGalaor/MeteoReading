-- =============================================================================
-- 03 · STATIONS : dimension station + choix de la station de référence par ville
-- =============================================================================

-- Distance orthodromique (formule de haversine), en km
CREATE OR REPLACE MACRO haversine_km(lat1, lon1, lat2, lon2) AS
    2 * 6371 * asin(sqrt(
        pow(sin(radians(lat2 - lat1) / 2), 2)
        + cos(radians(lat1)) * cos(radians(lat2)) * pow(sin(radians(lon2 - lon1) / 2), 2)
    ));

-- Dimension station : métadonnées les plus récentes + couverture sur la période d'analyse
CREATE OR REPLACE TABLE dim_station AS
WITH meta AS (
    SELECT
        station_id,
        arg_max(station_name, date) AS station_name,
        arg_max(dep, date)          AS dep,
        arg_max(lat, date)          AS lat,
        arg_max(lon, date)          AS lon,
        arg_max(altitude, date)     AS altitude,
        min(date)                   AS first_date,
        max(date)                   AS last_date,
        count(*)                    AS n_rows
    FROM stg_daily
    GROUP BY station_id
),
coverage AS (
    SELECT
        station_id,
        count(DISTINCT date) FILTER (WHERE tn IS NOT NULL AND tx IS NOT NULL) AS n_days_temp,
        count(DISTINCT date) FILTER (WHERE rr IS NOT NULL)                    AS n_days_rain
    FROM obs_daily
    WHERE date BETWEEN make_date(${start_year}, 1, 1) AND make_date(${end_year}, 12, 31)
    GROUP BY station_id
)
SELECT
    m.*,
    coalesce(c.n_days_temp, 0) AS n_days_temp,
    coalesce(c.n_days_rain, 0) AS n_days_rain,
    round(coalesce(c.n_days_temp, 0) / ${period_days}, 4) AS completeness_temp,
    round(coalesce(c.n_days_rain, 0) / ${period_days}, 4) AS completeness_rain
FROM meta m
LEFT JOIN coverage c USING (station_id);

-- Station(s) de référence de chaque ville.
-- a) Villes avec une chaîne imposée dans config.yaml : on la reprend telle quelle.
-- b) Sinon, sélection automatique de LA meilleure station, par ordre de priorité :
--      1. complétude des températures >= seuil
--      2. complétude la plus élevée (par paliers de 2 % pour ne pas sur-pondérer 0,1 %)
--      3. complétude des précipitations, puis distance au centre-ville
--    parmi les stations proches ET d'altitude comparable (évite une station de montagne).
CREATE OR REPLACE TABLE ref_chain AS
WITH forced AS (
    SELECT city,
           unnest(station_ids)              AS station_id,
           generate_subscripts(station_ids, 1) AS priority
    FROM dim_city
    WHERE len(station_ids) > 0
),
auto AS (
    SELECT c.city, s.station_id, 1 AS priority
    FROM dim_city c
    CROSS JOIN dim_station s          -- 20 villes x quelques milliers de stations : négligeable
    WHERE len(c.station_ids) = 0
      AND haversine_km(c.lat, c.lon, s.lat, s.lon) <= ${max_distance_km}
      AND (c.alt IS NULL OR abs(s.altitude - c.alt) <= ${max_altitude_diff_m})
    QUALIFY row_number() OVER (
        PARTITION BY c.city
        ORDER BY s.completeness_temp >= ${min_completeness} DESC,
                 floor(s.completeness_temp * 50) DESC,
                 s.completeness_rain DESC,
                 haversine_km(c.lat, c.lon, s.lat, s.lon)
    ) = 1
)
SELECT * FROM forced
UNION ALL
SELECT * FROM auto;

CREATE OR REPLACE TABLE ref_station AS
SELECT
    ch.city, ch.priority, s.station_id, s.station_name, s.dep, s.lat, s.lon, s.altitude,
    round(haversine_km(c.lat, c.lon, s.lat, s.lon), 1) AS distance_km,
    s.first_date, s.last_date, s.completeness_temp, s.completeness_rain
FROM ref_chain ch
JOIN dim_city c USING (city)
JOIN dim_station s USING (station_id)
ORDER BY city, priority;

-- Raccordement des séries : biais de chaque station secondaire par rapport à la
-- station principale, mesuré MOIS PAR MOIS sur les jours où les deux ont une mesure
-- (le biais varie avec la saison : inversions thermiques en hiver, altitude, etc.).
-- Il est ajouté aux températures de la station secondaire. La pluie n'est pas corrigée :
-- trop variable d'un point à l'autre pour qu'un simple décalage ait un sens.
CREATE OR REPLACE TABLE station_offsets AS
WITH obs AS (
    SELECT r.city, r.priority, r.station_id, o.date, o.tn, o.tx
    FROM ref_station r
    JOIN obs_daily o USING (station_id)
    WHERE r.city IN (SELECT city FROM ref_station WHERE priority > 1)
      AND o.tn IS NOT NULL AND o.tx IS NOT NULL
)
SELECT
    s.city, s.station_id, s.priority,
    month(s.date)                  AS month,
    count(*)                       AS n_overlap_days,
    round(avg(p.tn - s.tn), 2)     AS offset_tn,
    round(avg(p.tx - s.tx), 2)     AS offset_tx,
    round(corr(p.tx, s.tx), 3)     AS corr_tx
FROM obs s
JOIN obs p ON p.city = s.city AND p.priority = 1 AND p.date = s.date
WHERE s.priority > 1
GROUP BY ALL
ORDER BY s.city, s.priority, month;
