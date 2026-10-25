

CREATE OR REPLACE TABLE AWS_DP_PROD.ML_OPS.LEADIQ_TRAINING_V5_WITH_TARGET AS
WITH
-- V5 base, de-duplicated to one row per lead --------------------------------
v5d AS (
  SELECT *
  FROM AWS_DP_PROD.ML_OPS.LEADIQ_TRAINING_SET_V5
  QUALIFY ROW_NUMBER() OVER (PARTITION BY LEAD_ID ORDER BY LEAD_TS) = 1
),

-- SOLD BOATS, clean scan, 2015+ per the label spec (Path 1 / Path 2 window) --
sold_lbl AS (
  SELECT DISTINCT LOWER(TRIM(BUYER_EMAIL)) AS buyer_email, SOLD_DATE
  FROM AWS_DP_PROD.LISTINGS.SOLD_BOATS_HIN
  WHERE VALID_HIN = 1 AND IS_CANONICAL_RECORD = TRUE
    AND SOLD_DATE >= '2015-01-01'
    AND BUYER_EMAIL IS NOT NULL AND TRIM(BUYER_EMAIL) <> ''
),
email_freq AS (
  SELECT buyer_email, COUNT(*) AS n_purchases FROM sold_lbl GROUP BY buyer_email
),
clean_buyer_emails AS (
  SELECT s.buyer_email, s.SOLD_DATE
  FROM sold_lbl s
  JOIN email_freq f USING (buyer_email)
  WHERE f.n_purchases <= 5
    AND SPLIT_PART(s.buyer_email,'@',1) NOT IN
        ('info','sales','office','admin','contact','service','support',
         'frontdesk','reception','boats','yachts','broker','brokerage',
         'inquiry','inquiries','hello','mail','team')
),

-- Label universe: USA non-spam leads (mirrors the label build) --------------
leads AS (
  SELECT LEAD_ID,
         CREATED_TIME AS lead_ts,
         LOWER(TRIM(SENDER_EMAIL)) AS email,
         UPPER(TRIM(SPLIT_PART(TRIM(CONTACT_NAME),' ',-1))) AS lastname,
         CASE WHEN LISTING_HIN IS NOT NULL AND TRIM(LISTING_HIN) <> ''
               AND LENGTH(TRIM(LISTING_HIN)) >= 10
               AND LENGTH(REPLACE(UPPER(TRIM(LISTING_HIN)),
                                  LEFT(UPPER(TRIM(LISTING_HIN)),1),'')) > 0
               AND UPPER(TRIM(LISTING_HIN)) NOT ILIKE '%123456789%'
               AND UPPER(TRIM(LISTING_HIN)) NOT ILIKE '%000000000%'
              THEN UPPER(TRIM(LISTING_HIN)) END AS hin
  FROM AWS_DP_PROD.CONNECTIONS.EMAIL_LEADS
  WHERE IS_SPAM_FINAL = 0
    AND LISTING_COUNTRY_CODE = 'USA'
    AND SENDER_EMAIL IS NOT NULL AND TRIM(SENDER_EMAIL) <> ''
),
email_path AS (                -- Path 1: email person-level, sale 0-3mo after lead
  SELECT l.LEAD_ID, MAX(1) AS person_email_pos
  FROM leads l
  JOIN clean_buyer_emails s
    ON  s.buyer_email = l.email
    AND s.SOLD_DATE >  l.lead_ts
    AND s.SOLD_DATE <= DATEADD('month', 3, l.lead_ts)
  GROUP BY l.LEAD_ID
),
boat_path AS (                 -- Path 2: same-boat (HIN) sold 0-3mo, identity check
  SELECT l.LEAD_ID,
         MAX(1) AS boat_level_pos,
         MAX(CASE WHEN (l.lastname <> ''
                        AND EDITDISTANCE(l.lastname, UPPER(TRIM(s.BUYER_LASTNAME))) <= 1)
                    OR (l.email <> ''
                        AND l.email = LOWER(TRIM(s.BUYER_EMAIL)))
                  THEN 1 ELSE 0 END) AS same_boat_confirmed
  FROM leads l
  JOIN AWS_DP_PROD.LISTINGS.SOLD_BOATS_HIN s
    ON  UPPER(TRIM(s.HIN)) = l.hin
    AND s.SOLD_DATE >  l.lead_ts
    AND s.SOLD_DATE <= DATEADD('month', 3, l.lead_ts)
    AND s.VALID_HIN = 1
    AND s.IS_CANONICAL_RECORD = TRUE
    AND s.SOLD_DATE >= '2015-01-01'
  GROUP BY l.LEAD_ID
)

SELECT
  v5.LEAD_ID,
  v5.LEAD_TS,

  -- ---- labels (person-level; computed fresh, not joined from a label table) ----
  COALESCE(b.boat_level_pos, 0)                          AS BOAT_LEVEL_POS,
  COALESCE(e.person_email_pos, 0)                        AS PERSON_EMAIL_POS,
  COALESCE(b.same_boat_confirmed, 0)                     AS SAME_BOAT_CONFIRMED,
  GREATEST(COALESCE(e.person_email_pos, 0),
           COALESCE(b.same_boat_confirmed, 0))           AS LABEL,          -- model target
  v5.LABEL                                               AS LABEL_V5_OLD,   -- old sparse label; drop if unwanted

  -- ---- every other column, untouched, straight from V5 ----
  v5.VIEWED_THIS_BOAT, v5.CLICKED_GALLERY, v5.SAVED_BOAT, v5.IS_NEW_VISITOR,
  v5.LISTING_PRICE, v5.HAS_PHONE, v5.HAS_TIER, v5.MENTIONS_SURVEY, v5.MENTIONS_SEA_TRIAL,
  v5.SRC_BT_PORTAL, v5.SRC_BT_APP, v5.SRC_YW_WEB, v5.SRC_YW_APP,
  v5.SRC_BOATS_COM, v5.SRC_SOCIAL, v5.SRC_MLS, v5.SRC_OTHER,
  v5.N_PRIOR_LEADS, v5.N_DISTINCT_LISTINGS_PRIOR, v5.DAYS_SINCE_LAST_LEAD, v5.PRIOR_BOAT_OWNER

FROM v5d v5
LEFT JOIN email_path e USING (LEAD_ID)
LEFT JOIN boat_path  b USING (LEAD_ID);