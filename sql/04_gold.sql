-- =============================================================================
-- 04 · GOLD: analytical tables per city (ready for analysis and BI)
-- =============================================================================

-- Daily series per city on a complete calendar: a day without a measurement
-- still exists (NULL values), which makes completeness rates exact.
-- tmean = (TN + TX) / 2: climatological convention, consistent over the whole period
-- (the hourly mean TM only exists since stations were automated).
CREATE OR REPLACE TABLE city_daily AS
WITH calendar AS (
    SELECT CAST(d AS DATE) AS date
    FROM generate_series(make_date(${start_year}, 1, 1), DATE '${last_date}', INTERVAL 1 DAY) AS t(d)
),
obs AS (
    SELECT r.city, r.priority, o.date,
           o.tn + coalesce(off.offset_tn, 0) AS tn,   -- secondary station brought to the level
           o.tx + coalesce(off.offset_tx, 0) AS tx,   -- of the main station
           o.rr, o.ffm, o.fxi
    FROM ref_station r
    JOIN obs_daily o USING (station_id)
    LEFT JOIN station_offsets off
           ON off.city = r.city AND off.station_id = r.station_id AND off.month = month(o.date)
),
-- For each day: the highest-priority station that has a measurement.
-- Temperature and rainfall are chosen independently (a station can have one without the other).
-- The row_number also removes any duplicates.
temp AS (
    SELECT city, date, tn, tx FROM obs
    WHERE tn IS NOT NULL AND tx IS NOT NULL
    QUALIFY row_number() OVER (PARTITION BY city, date ORDER BY priority) = 1
),
rain AS (
    SELECT city, date, rr, ffm, fxi FROM obs
    WHERE rr IS NOT NULL
    QUALIFY row_number() OVER (PARTITION BY city, date ORDER BY priority) = 1
)
SELECT
    c.city, cal.date,
    round(t.tn, 1) AS tn, round(t.tx, 1) AS tx, round((t.tn + t.tx) / 2, 2) AS tmean,
    r.rr, r.ffm, r.fxi
FROM dim_city c
CROSS JOIN calendar cal
LEFT JOIN temp t ON t.city = c.city AND t.date = cal.date
LEFT JOIN rain r ON r.city = c.city AND r.date = cal.date
ORDER BY c.city, cal.date;

-- Runs of dry days ("gaps and islands" problem):
-- two consecutive dry days share the same value of (date - rank).
CREATE OR REPLACE TABLE city_dry_spells AS
WITH dry AS (
    SELECT city, date,
           date - CAST(row_number() OVER (PARTITION BY city ORDER BY date) AS INTEGER) AS island
    FROM city_daily
    WHERE rr < ${rain_day}
)
SELECT city, min(date) AS start_date, max(date) AS end_date, count(*) AS n_days
FROM dry
GROUP BY city, island;

-- Heat episodes: at least N consecutive days with TX >= threshold
CREATE OR REPLACE TABLE city_heatwaves AS
WITH hot AS (
    SELECT city, date, tx,
           date - CAST(row_number() OVER (PARTITION BY city ORDER BY date) AS INTEGER) AS island
    FROM city_daily
    WHERE tx >= ${hot_day}
)
SELECT city, min(date) AS start_date, max(date) AS end_date, count(*) AS n_days,
       max(tx) AS tx_peak, round(avg(tx), 1) AS tx_mean
FROM hot
GROUP BY city, island
HAVING count(*) >= ${heatwave_min_days};

-- Monthly aggregates, with a validity flag (month completeness)
CREATE OR REPLACE TABLE city_monthly AS
SELECT
    city,
    year(date)  AS year,
    month(date) AS month,
    day(last_day(min(date))) AS days_in_month,
    count(tmean) AS n_days_temp,
    count(rr)    AS n_days_rain,
    round(avg(tn), 2)    AS tn,
    round(avg(tx), 2)    AS tx,
    round(avg(tmean), 2) AS tmean,
    round(sum(rr), 1)    AS rr,
    count(*) FILTER (WHERE rr >= ${rain_day}) AS rain_days,
    count(tmean) >= ${month_min_completeness} * day(last_day(min(date))) AS valid_temp,
    count(rr)    >= ${month_min_completeness} * day(last_day(min(date))) AS valid_rain
FROM city_daily
GROUP BY ALL
ORDER BY ALL;

-- Annual aggregates + extreme-weather indicators (complete years only)
CREATE OR REPLACE TABLE city_yearly AS
WITH base AS (
    SELECT
        city,
        year(date) AS year,
        count(*)     AS days_in_year,
        count(tmean) AS n_days_temp,
        count(rr)    AS n_days_rain,
        round(avg(tn), 2)    AS tn,
        round(avg(tx), 2)    AS tx,
        round(avg(tmean), 2) AS tmean,
        round(sum(rr), 1)    AS rr,
        max(tx) AS tx_max,
        min(tn) AS tn_min,
        max(rr) AS rr_max,
        count(*) FILTER (WHERE tx >= ${hot_day})        AS hot_days,
        count(*) FILTER (WHERE tx >= ${very_hot_day})   AS very_hot_days,
        count(*) FILTER (WHERE tn >= ${tropical_night}) AS tropical_nights,
        count(*) FILTER (WHERE tn <  ${frost_day})      AS frost_days,
        count(*) FILTER (WHERE rr >= ${rain_day})       AS rain_days,
        count(*) FILTER (WHERE rr >= ${heavy_rain})     AS heavy_rain_days
    FROM city_daily
    WHERE year(date) <= ${end_year}
    GROUP BY ALL
),
dry AS (
    SELECT city, year(start_date) AS year, max(n_days) AS longest_dry_spell
    FROM city_dry_spells GROUP BY ALL
),
heat AS (
    SELECT city, year(start_date) AS year, count(*) AS heatwaves, sum(n_days) AS heatwave_days
    FROM city_heatwaves GROUP BY ALL
)
SELECT
    b.*,
    coalesce(d.longest_dry_spell, 0) AS longest_dry_spell,
    coalesce(h.heatwaves, 0)         AS heatwaves,
    coalesce(h.heatwave_days, 0)     AS heatwave_days,
    b.n_days_temp >= ${year_min_completeness} * b.days_in_year AS valid_temp,
    b.n_days_rain >= ${year_min_completeness} * b.days_in_year AS valid_rain
FROM base b
LEFT JOIN dry  d USING (city, year)
LEFT JOIN heat h USING (city, year)
ORDER BY city, year;
