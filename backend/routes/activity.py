"""
routes/activity.py — POST /activity

The extension's service worker posts every browser event here.
Pipeline: raw event → anonymize → classify intent → log to episodic memory → update working memory.
"""

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
from typing import Any

from backend.privacy.anonymizer import anonymize
from backend.engine.intent import classify_intent
from backend.memory.episodic import EpisodicMemory

router = APIRouter(prefix="/activity", tags=["activity"])
_episodic = EpisodicMemory()


class ActivityPayload(BaseModel):
    type: str                   # navigation | search | form_submit | tab_switch
    url: str | None = None
    text: str | None = None
    timestamp: int | None = None  # epoch ms from extension
    meta: dict[str, Any] | None = None


@router.post("")
async def receive_activity(payload: ActivityPayload, request: Request):
    """
    Receive a raw browser event, strip PII, classify intent, and log to memory.
    Returns the classified semantic event so the extension can echo it in the side panel.
    """
    raw = payload.model_dump(exclude_none=True)

    # 1. Strip PII — never store raw data
    clean = anonymize(raw)

    # 2. Classify intent from the clean event
    intent_data = classify_intent({**raw, **clean})
    clean.update(intent_data)

    # 3. Persist to episodic memory
    event_id = await _episodic.log({
        "event_type": payload.type,
        "intent": intent_data.get("intent"),
        "category": intent_data.get("category"),
        "time_bucket": intent_data.get("time_bucket"),
        "metadata": {k: v for k, v in clean.items() if k not in ("event_type",)},
    })

    # 4. Update working memory context (injected via app state)
    working = request.app.state.working_memory
    working.set_context(
        inferred_intent=intent_data.get("intent"),
        inferred_category=intent_data.get("category"),
        last_url=payload.url,
    )

    return {"status": "ok", "event_id": event_id, "intent": intent_data}
