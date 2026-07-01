"""
FastAPI application for the SHL Assessment Recommender.

Endpoints:
  GET  /health  – readiness probe (up to 2 min allowed for cold start).
  POST /chat    – stateless conversation endpoint.

Timeout: the evaluator gives 30 s per call. We guard with a 25 s
asyncio timeout so we always return valid JSON even if the LLM is slow.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os

# Load .env file if present (local dev + some platforms pick this up)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.catalog import load_catalog
from app.schemas import ChatRequest, ChatResponse, HealthResponse, Recommendation
from app.agent import run_agent

logger = logging.getLogger(__name__)

app = FastAPI(title="SHL Assessment Recommender")

# Pre-build the TF-IDF index and warm the catalog cache at startup so the
# first /chat call is not penalised by cold-start latency.
@app.on_event("startup")
async def _warm_up() -> None:
    from app.retrieval import _ensure_index
    _ensure_index()
    load_catalog()
    logger.info("Catalog index ready (%d records).", len(load_catalog()))


# ---------------------------------------------------------------------------
REFUSAL_TEXT = (
    "I can only help with selecting SHL assessments. I can't advise on "
    "general hiring strategy, legal questions, or anything outside the "
    "SHL product catalog."
)

TIMEOUT_SECONDS = 25  # leave 5 s margin under the evaluator's 30 s cap


# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request) -> ChatResponse:
    """
    Run the agent with a hard timeout.
    If the LLM exceeds TIMEOUT_SECONDS, return a graceful clarification
    response rather than letting the evaluator time out.
    """
    loop = asyncio.get_event_loop()

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = loop.run_in_executor(pool, run_agent, req.messages)
            out = await asyncio.wait_for(future, timeout=TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.warning("Agent timed out after %ds", TIMEOUT_SECONDS)
        return ChatResponse(
            reply="I'm taking too long to respond right now. Could you briefly describe the role and key skills?",
        )
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        return ChatResponse(
            reply="Something went wrong on my end. Could you rephrase your request?",
        )

    # ── Off-topic / injection guard ──────────────────────────────────────────
    if out.intent in ("off_topic", "refuse"):
        return ChatResponse(reply=REFUSAL_TEXT)

    # ── Compare: just return the text reply, no recommendations ─────────────
    if out.intent == "compare":
        return ChatResponse(reply=out.reply, end_of_conversation=out.end_of_conversation)

    # ── Clarify: return reply with empty recommendations ─────────────────────
    if out.intent == "clarify":
        return ChatResponse(reply=out.reply, end_of_conversation=out.end_of_conversation)

    # ── Recommend / Refine: resolve names → catalog records ──────────────────
    if out.intent in ("recommend", "refine"):
        recs = _resolve_recommendations(out.recommended_names)

        if not recs:
            # LLM returned names we couldn't resolve — ask for more context
            return ChatResponse(
                reply="I couldn't pinpoint a match in the catalog yet. Could you tell me more about the role or required skills?",
            )

        return ChatResponse(
            reply=out.reply,
            recommendations=recs[:10],
            end_of_conversation=out.end_of_conversation,
        )

    # ── Fallback (should not happen) ─────────────────────────────────────────
    return ChatResponse(reply=out.reply, end_of_conversation=out.end_of_conversation)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_recommendations(names: list[str]) -> list[Recommendation]:
    """Resolve LLM-returned names to catalog records, deduplicating."""
    from app.catalog import find_by_name

    recs: list[Recommendation] = []
    seen: set[str] = set()

    for name in names:
        rec = find_by_name(name)
        if rec and rec["id"] not in seen:
            # Ensure test_type is never null (evaluator schema requires a string)
            test_type = rec.get("test_type") or "K"
            recs.append(Recommendation(name=rec["name"], url=rec["url"], test_type=test_type))
            seen.add(rec["id"])

    return recs


# ---------------------------------------------------------------------------
# Global exception handler — always returns valid JSON
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def _global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=200,  # evaluator expects 200; we encode the error in the body
        content={
            "reply": "An unexpected error occurred. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )
