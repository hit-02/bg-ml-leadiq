-- Lead affinity analysis queries for EMAIL_LEADS
-- Co-authored with CoCo
SELECT COLUMN_NAME, DATA_TYPE
FROM AWS_DP_PROD.INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_SCHEMA = 'CONNECTIONS' AND TABLE_NAME = 'EMAIL_LEADS'
  AND (COLUMN_NAME ILIKE '%MAKE%' OR COLUMN_NAME ILIKE '%CLASS%' OR COLUMN_NAME ILIKE '%LISTING_ID%')
ORDER BY COLUMN_NAME;


WITH hist AS (
  SELECT LEAD_ID,
         COUNT(*) OVER (PARTITION BY LOWER(TRIM(SENDER_EMAIL))
                        ORDER BY CREATED_TIME
                        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior_leads
  FROM AWS_DP_PROD.CONNECTIONS.EMAIL_LEADS
  WHERE IS_SPAM_FINAL = 0
    AND SENDER_EMAIL IS NOT NULL AND TRIM(SENDER_EMAIL) <> ''
)
SELECT
  COUNT(*) AS total_leads,
  SUM(CASE WHEN n_prior_leads >= 2 THEN 1 ELSE 0 END) AS leads_with_2plus_prior,
  ROUND(100.0*SUM(CASE WHEN n_prior_leads >= 2 THEN 1 ELSE 0 END)/COUNT(*), 1) AS pct_coverage
FROM hist;

WITH lead_with_listing AS (
  SELECT l.LEAD_ID,
         l.CREATED_TIME,
         LOWER(TRIM(l.SENDER_EMAIL)) AS sender_email,
         l.PRODUCT_MAKE,
         l.PRODUCT_CLASS
  FROM AWS_DP_PROD.CONNECTIONS.EMAIL_LEADS l
  WHERE l.IS_SPAM_FINAL = 0
    AND l.SENDER_EMAIL IS NOT NULL AND TRIM(l.SENDER_EMAIL) <> ''
)
SELECT
  cur.LEAD_ID,
  COUNT(prior.LEAD_ID) AS n_prior_used,
  ROUND(AVG(CASE WHEN prior.PRODUCT_MAKE  = cur.PRODUCT_MAKE  THEN 1 ELSE 0 END), 3) AS make_affinity_match,
  ROUND(AVG(CASE WHEN prior.PRODUCT_CLASS = cur.PRODUCT_CLASS THEN 1 ELSE 0 END), 3) AS class_affinity_match
FROM lead_with_listing cur
JOIN lead_with_listing prior
  ON prior.sender_email = cur.sender_email
 AND prior.CREATED_TIME < cur.CREATED_TIME
GROUP BY cur.LEAD_ID;

USE DATABASE AWS_DP_PROD;
with base1 as (
  select
    s.LEAD_ID,
    l.SENDER_EMAIL,
    s."score_1_100" as LEADIQ_SCORE,
    s."tier" as LEADIQ_TIER,
    s."raw_score" as LEADIQ_RAW_SCORE,
    f.* exclude (LEAD_ID, LABEL)
  from AWS_DP_PROD.ML_OPS.LEADIQ_SCORED_V1 s
  join AWS_DP_PROD.ML_OPS.LEADIQ_TRAINING_SET_V4 f
    on s.LEAD_ID = f.LEAD_ID
  join AWS_DP_PROD.CONNECTIONS.EMAIL_LEADS l
    on s.LEAD_ID = l.LEAD_ID
),
base2 as (
  select
    *,
    PERCENT_RANK() OVER (ORDER BY LEADIQ_SCORE) as score_percentile
  from base1
)
select
  LEAD_ID,
  SENDER_EMAIL,
  LEADIQ_SCORE,
  LEADIQ_TIER,
  AI_COMPLETE(
    'openai-gpt-5',
    '
CONTEXT:
- You are generating dealer-facing "score breakdown" and "summary sentence" text for Boat Wizard Lead IQ, a lead scoring system used by boat dealers on boats.com, Boat Trader, and YachtWorld.
- Input data: lead-level features from the V4 feature set (XGBoost model) used to produce a 1-100 score and a Hot/Warm/Cold tier.
- Feature groups and what they mean:
  * N_PRIOR_LEADS = how many leads this person has submitted before
  * N_DISTINCT_LISTINGS_PRIOR = how many different boats they''ve inquired about previously
  * DAYS_SINCE_LAST_LEAD = days since their last inquiry (very short gaps, e.g. same-day, are a weaker signal than an 8-30 day gap)
  * PRIOR_BOAT_OWNER = whether this person has owned a boat before
  * VIEWED_THIS_BOAT, CLICKED_GALLERY, SAVED_BOAT = on-listing engagement before the lead was sent
  * HAS_PHONE, MENTIONS_SURVEY, MENTIONS_SEA_TRIAL = signals from the lead message itself
  * SRC_* one-hots = which channel the lead came from (SRC_SOCIAL converts notably below average; other sources are neutral-to-positive by comparison)
  * SAME_STATE, CROSS_COUNTRY, USER_LISTING_DISTANCE = buyer/listing geographic alignment - this group has shown inconsistent, low-reliability contribution in past model evaluations, so treat it as a minor and uncertain signal, never a headline driver
  * IS_NEW_VISITOR, HAS_TIER, BOAT_LEVEL_POS = account/listing context fields
- Features as JSON:
    '||TO_JSON(OBJECT_CONSTRUCT(*))||'
- Model score: '||LEADIQ_SCORE||' (1-100 scale)
- Model tier: '||LEADIQ_TIER||'
- Score percentile among all scored leads: '||ROUND(100*score_percentile,1)||'

TASK:
1. Generate 3-5 short, plain-English bullet points identifying the most influential signals behind this score, per PRD section 5.1. Explicitly separate:
   - Positive signals (factors that raised the score)
   - Negative signals (factors that lowered the score)
2. Then generate ONE plain-language summary sentence (PRD section 5.2).
3. Calibration requirement (critical): this is a non-linear XGBoost model, so feature contributions are NOT strictly additive or precisely attributable. Use phrasing like "appears to reflect," "is consistent with," or "suggests" rather than "caused" or "is the reason."
4. Overall confidence framing: write as if you are about 60% confident in the direction and relative importance of each point.
5. Things to avoid:
   - Do not reference raw column names
   - Do not lead with geographic alignment as a primary driver
   - Do not fabricate specifics not present in the feature JSON
   - Avoid absolute claims like "this buyer will convert"

FORMATTING:
  Positive Signals
  Negative Signals
  Summary Sentence

OUTPUT:
- Succinct, dealer-readable, no jargon, no raw feature names.
- Double check that all directional claims match the actual feature values in the JSON.
'
  ) as leadiq_talking_points
from base2
limit 5



SELECT PARSE_JSON(
  SYSTEM$GET_FEATURE_STORE_ONLINE_SERVICE_STATUS('AWS_DP_PROD.BOATS_FEATURE_STORE')
):endpoints AS endpoints;

from snowflake.ml.feature_store.feature_view import StoreType

sample_lead_id = session.table("AWS_DP_PROD.ML_OPS.LEADIQ_TRAINING_SET_V5") \
    .select("LEAD_ID").limit(1).collect()[0][0]

fs.read_feature_view(fv_email_history, store_type=StoreType.ONLINE, keys=[[sample_lead_id]])


ALTER USER "JUSTIN.XUE@BOATS.COM" ADD PROGRAMMATIC ACCESS TOKEN LEADIQ_PAT;



SHOW GIT REPOSITORIES in ACCOUNT