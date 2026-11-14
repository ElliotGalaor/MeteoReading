# Presenting the project in an interview

## The pitch (30 seconds)

"I built a pipeline that automatically collects the 25 million daily records published by Météo-France for
20 large cities since 1950. I clean them and store them in a DuckDB warehouse, compute statistically tested
indicators, and present them in a dashboard. Main result: +0.33 °C per decade since 1951, a pace that is
speeding up, days at 35 °C five times more frequent, but no clear trend in rainfall. The most interesting part
technically was data quality: stations move, and I had to splice their series."

## The demo (5 minutes)

1. **Overview**: the 4 key figures, then the warming stripes. Hover over 2022.
2. **Explore a city**: Paris vs Marseille, then switch to "Anomaly": the two curves overlap, the warming is the
   same despite very different climates. Point to the p-value under the chart.
3. **A year in detail**: Paris 2022 against the normal; then 2003 for the August heatwave.
4. **Data and quality**: the pipeline diagram, the series-splicing table (Reims, Grenoble, Limoges), the ERA5
   validation.
5. **SQL queries**: run "Rank of 2022 in each city's history" (window function), then the query over the
   25 million rows: answer in ~0.1 s.
6. **Route weather**: pick "Paris to Versailles by bike" for tomorrow 9:00, switch the map to "Headwind", then
   open the Mont Ventoux sample (recorded on 15 July 2025): the temperature drops along the climb.

## Likely questions

**Why DuckDB rather than pandas or PostgreSQL?**
25 million rows do not fit comfortably in pandas. DuckDB is a serverless columnar analytical database: it reads
the compressed CSV files directly and aggregates 25 M rows in a fraction of a second. PostgreSQL would have meant
installing and running a server for a purely analytical use.

**How did you choose the stations?**
Automatically, in SQL: the most complete station over 1951-2025, within 40 km and 200 m of altitude of the city
centre. I added the altitude safeguard after seeing that Grenoble was matched to a mountain station at 806 m.
Three cities have a chain of stations forced in the configuration.

**What is series splicing?**
When a station closes (Reims-Courcy in 2014), another one takes over (Reims-Prunay, opened in 2012). Over the two
years when both measure, I compute their average difference, month by month, and correct the old series by that
difference. Without it, I would create a fake temperature jump at the switch.

**How do you know a trend is real?**
Two tests: linear regression (slope and confidence interval) and Mann-Kendall, a non-parametric test that does
not assume a normal distribution. I only call a trend significant when both are. For rainfall they are not, and
I say so: a non-significant result is still a result.

**Why anomalies rather than raw temperatures?**
To compare cities with different climates, and so that a city missing a year does not move the national average.
Each city is compared with its own 1991-2020 normal.

**What are the limitations?**
The series are not homogenised the way climatologists do it: urbanisation around a station can amplify the
trend. One station represents a city, not its whole area. And the tests assume independent years, so the
p-values are slightly optimistic.

**Why the comparison with ERA5?**
To validate my series against an independent source, and because many companies use reanalyses when they have
no stations. Conclusion: ERA5 tracks day-to-day weather very well (0.987 correlation), but smooths out local
effects (sea, city); for local extremes, you need stations.

**How does the route weather work?**
The track is sampled every kilometre or more (40 points at most), each point gets a passage time (recorded or
estimated from a speed), and the hourly weather of all points is fetched in one batch from Open-Meteo: forecast
for recent or upcoming trips, ERA5 archive for older ones. Values are interpolated at the exact passage time; the
wind is interpolated as a vector (otherwise 350° and 10° would average to 180°) and projected on the direction
of travel to get the headwind. For past trips, nearby Météo-France stations give a ground truth.

**What would you do with more time?**
Extend to the whole of France (it is one configuration setting), add sunshine and humidity, schedule a daily
refresh, and deploy the dashboard online.

**How do you make sure the code works?**
33 automated tests, including an end-to-end test on a small synthetic dataset that reproduces every pitfall
found in the real data: suspect quality code, impossible value, TN > TX, duplicate, ID with a leading zero,
station that moves.
