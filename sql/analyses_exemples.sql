-- Top 10 des journées les plus chaudes, toutes villes confondues
SELECT city AS ville, date, tx AS tmax_c
FROM city_daily
WHERE tx IS NOT NULL AND year(date) <= 2025
ORDER BY tx DESC
LIMIT 10;

-- Réchauffement par ville : 2011-2025 contre 1961-1990 (agrégats filtrés)
SELECT
    city AS ville,
    round(avg(tmean) FILTER (WHERE year BETWEEN 1961 AND 1990), 2) AS t_1961_1990,
    round(avg(tmean) FILTER (WHERE year BETWEEN 2011 AND 2025), 2) AS t_2011_2025,
    round(t_2011_2025 - t_1961_1990, 2)                           AS ecart_c
FROM city_yearly
WHERE valid_temp
GROUP BY city
ORDER BY ecart_c DESC;

-- Année la plus chaude de chaque ville (fonction de fenêtre + QUALIFY)
SELECT city AS ville, year AS annee, tmean AS tmoy_c
FROM city_yearly
WHERE valid_temp AND year <= 2025
QUALIFY rank() OVER (PARTITION BY city ORDER BY tmean DESC) = 1
ORDER BY tmean DESC;

-- Rang de l'année 2022 dans l'historique de chaque ville (DENSE_RANK)
WITH classement AS (
    SELECT city, year, tmean,
           dense_rank() OVER (PARTITION BY city ORDER BY tmean DESC) AS rang
    FROM city_yearly
    WHERE valid_temp AND year <= 2025
)
SELECT city AS ville, tmean AS tmoy_2022_c, rang AS rang_sur_75_ans
FROM classement
WHERE year = 2022
ORDER BY rang, ville;

-- Paris : moyenne glissante sur 10 ans (cadre de fenêtre ROWS)
SELECT
    year AS annee,
    tmean AS tmoy_c,
    round(avg(tmean) OVER (ORDER BY year ROWS BETWEEN 9 PRECEDING AND CURRENT ROW), 2) AS moyenne_10_ans
FROM city_yearly
WHERE city = 'Paris' AND year BETWEEN 1951 AND 2025
ORDER BY year DESC
LIMIT 15;

-- Nuits tropicales par décennie et par ville (PIVOT)
PIVOT (
    SELECT city AS ville, (year // 10 * 10)::VARCHAR AS decennie, tropical_nights
    FROM city_yearly
    WHERE year BETWEEN 1951 AND 2025
)
ON decennie
USING round(avg(tropical_nights), 1)
GROUP BY ville
ORDER BY ville;

-- Plus longues périodes sans pluie (résultat d'un calcul "gaps and islands")
SELECT city AS ville, start_date AS debut, end_date AS fin, n_days AS jours_secs
FROM city_dry_spells
WHERE year(start_date) <= 2025
ORDER BY n_days DESC
LIMIT 10;

-- Requête sur les 25 millions de lignes : départements les plus arrosés (normale 1991-2020)
SELECT
    dep AS departement,
    count(DISTINCT station_id)     AS stations,
    round(avg(rr) * 365.25)        AS cumul_annuel_moyen_mm
FROM obs_daily
WHERE date BETWEEN DATE '1991-01-01' AND DATE '2020-12-31'
GROUP BY dep
ORDER BY cumul_annuel_moyen_mm DESC;
