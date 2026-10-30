-- Example training query for:  train.sh --bq_query_file query_example.sql
-- Must return `text` and `label` columns; `split` ('train'/'test') is optional.
-- Labels can be 0/1 or negative/positive (any capitalization).
-- Replace the table with your own. Select only the columns you need: BigQuery bills by data scanned.
SELECT
  review_text AS text,
  sentiment   AS label,
  IF(MOD(ABS(FARM_FINGERPRINT(review_id)), 5) = 0, 'test', 'train') AS split  -- stable 80/20 split
FROM `my-project.reviews.movie_reviews`
WHERE review_text IS NOT NULL
