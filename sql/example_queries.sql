-- Top 10 hottest days, all cities combined
SELECT city, date, tx AS tmax_c
FROM city_daily
WHERE tx IS NOT NULL AND year(date) <= 2025
ORDER BY tx DESC
LIMIT 10;

-- Warming by city: 2011-2025 vs 1961-1990 (filtered aggregates)
SELECT
    city,
    round(avg(tmean) FILTER (WHERE year BETWEEN 1961 AND 1990), 2) AS t_1961_1990,
    round(avg(tmean) FILTER (WHERE year BETWEEN 2011 AND 2025), 2) AS t_2011_2025,
    round(t_2011_2025 - t_1961_1990, 2)                           AS change_c
FROM city_yearly
WHERE valid_temp
GROUP BY city
ORDER BY change_c DESC;

-- Warmest year of each city (window function + QUALIFY)
SELECT city, year, tmean AS tmean_c
FROM city_yearly
WHERE valid_temp AND year <= 2025
QUALIFY rank() OVER (PARTITION BY city ORDER BY tmean DESC) = 1
ORDER BY tmean DESC;

-- Rank of 2022 in each city's history (DENSE_RANK)
WITH ranking AS (
    SELECT city, year, tmean,
           dense_rank() OVER (PARTITION BY city ORDER BY tmean DESC) AS rank_
    FROM city_yearly
    WHERE valid_temp AND year <= 2025
)
SELECT city, tmean AS tmean_2022_c, rank_ AS rank_out_of_75_years
FROM ranking
WHERE year = 2022
ORDER BY rank_, city;

-- Paris: 10-year moving average (ROWS window frame)
SELECT
    year,
    tmean AS tmean_c,
    round(avg(tmean) OVER (ORDER BY year ROWS BETWEEN 9 PRECEDING AND CURRENT ROW), 2) AS moving_avg_10y
FROM city_yearly
WHERE city = 'Paris' AND year BETWEEN 1951 AND 2025
ORDER BY year DESC
LIMIT 15;

-- Tropical nights per decade and per city (PIVOT)
PIVOT (
    SELECT city, (year // 10 * 10)::VARCHAR AS decade, tropical_nights
    FROM city_yearly
    WHERE year BETWEEN 1951 AND 2025
)
ON decade
USING round(avg(tropical_nights), 1)
GROUP BY city
ORDER BY city;

-- Longest periods without rain (output of a "gaps and islands" calculation)
SELECT city, start_date, end_date, n_days AS dry_days
FROM city_dry_spells
WHERE year(start_date) <= 2025
ORDER BY n_days DESC
LIMIT 10;

-- Query over the 25 million rows: wettest départements (1991-2020 normal)
SELECT
    dep AS departement,
    count(DISTINCT station_id)     AS stations,
    round(avg(rr) * 365.25)        AS mean_annual_rainfall_mm
FROM obs_daily
WHERE date BETWEEN DATE '1991-01-01' AND DATE '2020-12-31'
GROUP BY dep
ORDER BY mean_annual_rainfall_mm DESC;
