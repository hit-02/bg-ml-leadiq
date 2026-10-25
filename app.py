import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from snowflake.snowpark import Session

from model_inference import LeadIQScorer
from snowflake_session import create_local_session


# Load the .env file from the same directory as app.py.
load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env",
    override=True,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LeadIQ-API")

snowflake_session: Optional[Session] = None
inference_service: Optional[LeadIQScorer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global snowflake_session, inference_service

    try:
        if not os.getenv("SNOWFLAKE_PAT"):
            raise RuntimeError(
                "SNOWFLAKE_PAT is missing from .env. "
                "Online Feature Store access requires PAT authentication."
            )

        logger.info("Creating PAT-authenticated Snowflake session")
        snowflake_session = create_local_session()

        current_context = snowflake_session.sql(
            """
            SELECT
                CURRENT_USER() AS CURRENT_USER,
                CURRENT_ROLE() AS CURRENT_ROLE,
                CURRENT_WAREHOUSE() AS CURRENT_WAREHOUSE
            """
        ).collect()[0]

        logger.info(
            "Connected to Snowflake: user=%s role=%s warehouse=%s",
            current_context["CURRENT_USER"],
            current_context["CURRENT_ROLE"],
            current_context["CURRENT_WAREHOUSE"],
        )

        if current_context["CURRENT_ROLE"] != "DATA_SCIENTIST":
            raise RuntimeError(
                "Snowflake session started with role "
                f"{current_context['CURRENT_ROLE']} instead of DATA_SCIENTIST."
            )

        logger.info("Loading Lead IQ model and Online Feature Store context")
        inference_service = LeadIQScorer(model_version="latest")
        inference_service.load_context(session=snowflake_session)

        logger.info("Lead IQ API initialized successfully")
        yield

    except Exception:
        logger.exception("Lead IQ API failed during startup")
        raise

    finally:
        inference_service = None

        if snowflake_session is not None:
            logger.info("Closing Snowflake session")
            snowflake_session.close()
            snowflake_session = None


app = FastAPI(
    title="Lead IQ Local Inference API",
    description="Local FastAPI wrapper for Lead IQ single-lead scoring.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> Dict[str, str]:
    if snowflake_session is None or inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Service is not initialized",
        )

    return {
        "status": "ok",
        "service": "lead-iq",
        "model_version": inference_service.model_file or "not-loaded",
        "feature_store_mode": "online",
        "authentication": "programmatic-access-token",
    }


@app.get("/predict/{lead_id}")
def predict(lead_id: int) -> Dict[str, Any]:
    if inference_service is None:
        raise HTTPException(
            status_code=503,
            detail="Inference service is not initialized",
        )

    if lead_id <= 0:
        raise HTTPException(
            status_code=400,
            detail="lead_id must be a positive integer",
        )

    try:
        return inference_service.predict(lead_id)

    except (IndexError, KeyError) as exc:
        logger.warning(
            "No feature-store row found for lead_id=%s",
            lead_id,
        )
        raise HTTPException(
            status_code=404,
            detail=f"No feature data found for lead_id {lead_id}",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Prediction failed for lead_id=%s",
            lead_id,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                f"Prediction failed: {type(exc).__name__}: {str(exc)}"
            ),
        ) from exc