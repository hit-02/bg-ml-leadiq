"""
Lead IQ inference class -- predict() for a single incoming lead.

Adapted from the recommendation system's model_code.py (class shape, session
handling, online feature-store reads), right-sized for Lead IQ:
  - one entity (LEAD_ID), not two keyed types (DEVICE_ID / LISTING_ID)
  - one score per lead, not a ranked list of many candidates
  - XGBoost binary classifier, not a LightGBM ranker
  - no bias boosts, no seller boost, no DynamoDB/CloudWatch logging --
    none of that is part of what Lead IQ's PRD asks for

NOT YET RUN. Built against:
  - fs.read_feature_group(), confirmed working earlier this session (~226ms,
    one real LEAD_ID, all 17 features returned)
  - the model + reference-score artifacts model_train_deploy.ipynb now saves
    to LEADIQ_MODEL_STAGE

Deliberately NOT using the recommendation system's direct-REST optimization
(bypassing the SDK for a <50ms p50 budget) -- that path is unverified against
Lead IQ's actual online service. If Lead IQ's real-time SLA later requires
it, model_code.py in the recommendation system is the reference for exactly
how to build it; this file uses the proven SDK call instead.

Model persistence uses joblib (joblib.dump/joblib.load), matching the
recommendation system's convention, rather than XGBoost's native save_model
format -- joblib restores the exact fitted sklearn object as-is, so
predict_proba() behaves identically to right after training; no
reconstruction-from-booster ambiguity to worry about here.
"""

"""
Lead IQ inference class -- predict() for a single incoming lead.

Adapted from the recommendation system's model_code.py (class shape, session
handling, online feature-store reads), right-sized for Lead IQ:
  - one entity (LEAD_ID), not two keyed types (DEVICE_ID / LISTING_ID)
  - one score per lead, not a ranked list of many candidates
  - XGBoost binary classifier, not a LightGBM ranker
  - no bias boosts, no seller boost, no DynamoDB/CloudWatch logging --
    none of that is part of what Lead IQ's PRD asks for

NOT YET RUN. Built against:
  - fs.read_feature_group(), confirmed working earlier this session (~226ms,
    one real LEAD_ID, all 17 features returned)
  - the model + reference-score artifacts model_train_deploy.ipynb now saves
    to LEADIQ_MODEL_STAGE

Deliberately NOT using the recommendation system's direct-REST optimization
(bypassing the SDK for a <50ms p50 budget) -- that path is unverified against
Lead IQ's actual online service. If Lead IQ's real-time SLA later requires
it, model_code.py in the recommendation system is the reference for exactly
how to build it; this file uses the proven SDK call instead.

Model persistence uses joblib (joblib.dump/joblib.load), matching the
recommendation system's convention, rather than XGBoost's native save_model
format -- joblib restores the exact fitted sklearn object as-is, so
predict_proba() behaves identically to right after training; no
reconstruction-from-booster ambiguity to worry about here.
"""
import time
import logging
import numpy as np
import pandas as pd
import mlflow
from datetime import datetime, timezone

from snowflake.snowpark.context import get_active_session
from snowflake.ml.feature_store import FeatureStore, CreationMode
from snowflake.ml.feature_store.feature_view import StoreType


def create_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s - %(name)s - %(message)s"
        ))
        logger.addHandler(handler)
    return logger


logger = create_logger("LeadIQ-Predict")

# The exact 17 feature columns LEADIQ_TRAINING_SET_V6_FS produces (confirmed
# via sanity_checks in fs_code.ipynb) -- must stay in sync with the 3
# registered feature views. XGBoost matches columns by NAME against what the
# booster was trained with (not position), and raises if given an unexpected
# set -- that's a deliberate safety net, not something this list needs to
# work around.
FEATURE_ORDER = [
    "N_PRIOR_LEADS", "N_DISTINCT_LISTINGS_PRIOR", "DAYS_SINCE_LAST_LEAD", "PRIOR_BOAT_OWNER",
    "SRC_BT_PORTAL", "SRC_BT_APP", "SRC_YW_WEB", "SRC_YW_APP",
    "SRC_BOATS_COM", "SRC_SOCIAL", "SRC_MLS", "SRC_OTHER",
    "LISTING_PRICE", "HAS_PHONE", "HAS_TIER", "MENTIONS_SURVEY", "MENTIONS_SEA_TRIAL",
]

# PRD 3.2 tier boundaries -- fixed, independent of model version.
TIER_BINS = [0, 39, 74, 100]
TIER_LABELS = ["Cold", "Warm", "Hot"]

MODEL_STAGE = "@AWS_DP_PROD.ML_OPS.LEADIQ_MODEL_STAGE"

FEATURE_VIEWS = [
    ("LEADIQ_EMAIL_HISTORY_FEATURES", "V1"),
    ("LEADIQ_PIPELINE_FEATURES", "V1"),
    ("LEADIQ_LISTING_FORM_FEATURES", "V1"),
]
FEATURE_GROUP = ("LEADIQ_FG", "V1")


class LeadIQScorer(mlflow.pyfunc.PythonModel):
    """Loads the model, reference-score distribution, and feature views once
    via load_context(). Call predict(lead_id) as many times as needed after
    that -- each call is one online lookup + one model score."""

    def __init__(self, model_version=None):
        # "latest", or an exact snapshot timestamp string matching
        # model_train_deploy.ipynb's ts format, e.g. "20260727_153312".
        # None here -- code-based logging instantiates with no args; the
        # real value arrives via model_config in load_context().
        self.model_version = model_version
        self.session = None
        self.model = None
        self.model_file = None
        self.ref_scores = None
        self.fs = None
        self.feature_group = None
        self.feature_views = None

    def load_context(self, context):
        # model_version arrives via model_config when instantiated with no
        # args (code-based logging path) -- fall back to it if __init__
        # didn't get one.
        if self.model_version is None:
            cfg = getattr(context, "model_config", None) or {}
            self.model_version = cfg.get("model_version", "latest")

        try:
            self.session = get_active_session()
            logger.info("using active notebook session")
        except Exception as e:
            raise RuntimeError(
                "No active Snowflake session available in load_context()."
            ) from e

        self.session.sql("USE DATABASE AWS_DP_PROD").collect()

        import joblib
        self.model = joblib.load(context.artifacts["leadiq_model"])
        self.model_file = f"leadiq_persontarget_v1_{self.model_version}.joblib"  # carried into predict()'s output, PRD 7.3

        self.ref_scores = np.load(context.artifacts["leadiq_refscores"])  # sorted ascending
        
        self.fs = FeatureStore(
            session=self.session,
            database="AWS_DP_PROD",
            name="BOATS_FEATURE_STORE",
            default_warehouse=self.session.get_current_warehouse(),
            creation_mode=CreationMode.FAIL_IF_NOT_EXIST,
        )
        self.feature_views = [self.fs.get_feature_view(n, v) for n, v in FEATURE_VIEWS]

        # Prefer the Feature Group -- one call instead of three. Falls back
        # to the 3 individual feature views if it isn't registered in this
        # environment (e.g. an older snowflake-ml-python without FeatureGroup).
        try:
            self.feature_group = self.fs.get_feature_group(*FEATURE_GROUP)
        except Exception as e:
            logger.warning(f"{FEATURE_GROUP[0]} not available ({e}); using 3 separate feature-view reads")
            self.feature_group = None

        logger.info(
            f"loaded model={self.model_file} ref_scores_n={len(self.ref_scores):,} "
            f"feature_group={'yes' if self.feature_group else 'no (3 separate reads)'}"
        )

    def _fetch_features(self, lead_id):
        """One online lookup keyed by LEAD_ID. Returns a 1-row DataFrame with
        all 17 features, column-named to match training."""
        if self.feature_group is not None:
            return self.fs.read_feature_group(self.feature_group, keys=[[str(lead_id)]])
        frames = [
            self.fs.read_feature_view(fv, store_type=StoreType.ONLINE, keys=[[str(lead_id)]]).to_pandas()
            for fv in self.feature_views
        ]
        pdf = frames[0]
        for f in frames[1:]:
            pdf = pdf.merge(f, on="LEAD_ID", how="left")
        return pdf

    def _score_to_tier(self, raw_score):
        """raw_score (one XGBoost predict_proba value) -> (score_1_100, tier).
        Percentile is against the frozen reference distribution saved at
        training time -- approximates build_scored_output's
        rank(pct=True)*99+1 formula (searchsorted vs pandas' tie-averaged
        rank can differ by at most 1 at tie boundaries; close enough for a
        1-100 integer score, not claimed to be byte-identical)."""
        pct = np.searchsorted(self.ref_scores, raw_score, side="left") / len(self.ref_scores)
        score_1_100 = int(round(pct * 99 + 1))
        score_1_100 = min(max(score_1_100, 1), 100)
        tier_idx = min(max(np.digitize(score_1_100, TIER_BINS) - 1, 0), 2)
        return score_1_100, TIER_LABELS[tier_idx]

    def predict(self, context, model_input):
        """mlflow.pyfunc entrypoint -- model_input is a DataFrame with a
        LEAD_ID column (one row per lead to score). Scores each row via
        _predict_one() and returns a DataFrame of results."""
        lead_ids = model_input["LEAD_ID"].tolist()
        return pd.DataFrame([self._predict_one(lid) for lid in lead_ids])

    def _predict_one(self, lead_id):
        """Original single-lead scoring logic, unchanged."""
        t0 = time.perf_counter()

        feats = self._fetch_features(lead_id)
        io_ms = (time.perf_counter() - t0) * 1000

        missing = [c for c in FEATURE_ORDER if c not in feats.columns]
        if missing:
            logger.warning(f"lead_id={lead_id}: missing from online store: {missing}")
        X = feats.reindex(columns=FEATURE_ORDER)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').astype('float64')

        t1 = time.perf_counter()
        raw_score = float(self.model.predict_proba(X)[:, 1][0])
        predict_ms = (time.perf_counter() - t1) * 1000

        score_1_100, tier = self._score_to_tier(raw_score)
        total_ms = (time.perf_counter() - t0) * 1000

        logger.info(
            f"lead_id={lead_id} score={score_1_100} tier={tier} "
            f"io={io_ms:.1f}ms predict={predict_ms:.1f}ms total={total_ms:.1f}ms"
        )

        return {
            "lead_id": lead_id,
            "raw_score": raw_score,
            "score_1_100": score_1_100,
            "tier": tier,
            "model_version": self.model_file,
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "latency_ms": round(total_ms, 1),
        }
mlflow.models.set_model(LeadIQScorer())
