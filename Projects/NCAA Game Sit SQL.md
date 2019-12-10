---
layout: page
title: NCAA Game Situation SQL
permalink: /ncaasql/
---
This is a SQL query on the public BigQuery Database located here:
https://console.cloud.google.com/marketplace/details/ncaa-bb-public/ncaa-basketball

It allows you to find the number of games since 2013 where the game situation (game differential and time elapsed in the game) defined in the WHERE statement of the main query (found after the common table expressions) has happened. It also tells you how often the home team won in those games.

```
WITH game_clock AS(
SELECT
  games.game_id,
  FIRST_VALUE(elapsed_time_sec)
    OVER(
      PARTITION BY game_id
      ORDER BY timestamp
      ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
      ) AS previous_time_sec,
  elapsed_time_sec AS current_time_Sec,
  SUM(CASE WHEN team_id = h_id THEN points_scored END)
    OVER(
      PARTITION BY game_id
      ORDER BY timestamp
      ) AS home_points,
  SUM(CASE WHEN team_id = a_id THEN points_scored END)
    OVER(
      PARTITION BY game_id
      ORDER BY timestamp
      ) AS away_points,
  CASE WHEN h_points > a_points THEN 1
    ELSE 0 END AS home_win
FROM `bigquery-public-data.ncaa_basketball.mbb_pbp_sr` AS pbp
  INNER JOIN `bigquery-public-data.ncaa_basketball.mbb_games_sr` AS games
  USING(game_id)),
game_sit AS(
SELECT
  game_id,
  previous_time_sec,
  current_time_sec,
  home_points - away_points AS home_pts_diff,
  home_win
FROM game_clock)
SELECT
  COUNT(game_id) AS num_games,
  ROUND(AVG(home_win)*100,1) AS home_win_perc
FROM game_sit
  JOIN `bigquery-public-data.ncaa_basketball.mbb_games_sr` as games
  USING(game_id)
WHERE
  previous_time_sec <= 1200 AND
  current_time_sec >= 1200 AND
  home_pts_diff = -10
```
