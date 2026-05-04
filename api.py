"""FastAPI server — multi-user Telekom customer support.

Endpoints:
  POST /sessions                    — create a new support session
  POST /sessions/{session_id}/chat  — send a message, get AI reply
  GET  /sessions/{session_id}       — get conversation history
  GET  /health                      — liveness check

Each session gets its own thread_id in LangGraph. All sessions share one
SqliteSaver so state persists across server restarts.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Interactive docs:
    http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import logging
import smtplib
import ssl
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

logger = logging.getLogger(__name__)

from config import settings
from observability import get_langfuse_client, get_langfuse_handler
from support_graph import build_support_graph
from user_memory import (
    finish_session,
    format_chat_transcript,
    get_or_create_active_user_id,
    get_or_create_user_by_name,
    save_feedback,
    save_message,
    start_new_session,
)

# ---------------------------------------------------------------------------
# SQLite DB path for checkpoints
# ---------------------------------------------------------------------------

_DB_PATH = str(Path(__file__).resolve().parent / "data" / "checkpoints.sqlite")
Path(_DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared graph — created once on startup via lifespan
# ---------------------------------------------------------------------------

_app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the shared LangGraph once on server startup."""
    _app_state["graph"] = build_support_graph(session_id="multi-user", db_path=_DB_PATH)
    yield
    # cleanup (nothing needed for SqliteSaver — connection stays open)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------
_limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

app = FastAPI(
    title="Telekom Support API",
    description="Multi-user customer support powered by LangGraph + LLM routing",
    version="1.0.0",
    lifespan=lifespan,
)

app.state.limiter = _limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — allow ngrok domain + localhost for development
_ALLOWED_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "https://parole-amperage-enlighten.ngrok-free.dev",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)

# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

# Rate limit strings pulled from config (override via .env)
_RATE_SESSIONS = settings.rate_limit_sessions
_RATE_CHAT     = settings.rate_limit_chat


# ---------------------------------------------------------------------------
# In-memory session registry  {session_id → {thread_id, user_id, db_session_id, created_at}}
# ---------------------------------------------------------------------------
# NOTE: For production replace with Redis or a DB table.
_sessions: dict[str, dict] = {}
_SESSION_TTL_SECONDS = 4 * 60 * 60  # 4 hours


def _is_session_expired(sess: dict) -> bool:
    created_at: datetime = sess.get("created_at", datetime.now(timezone.utc))
    age = (datetime.now(timezone.utc) - created_at).total_seconds()
    return age > _SESSION_TTL_SECONDS


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    user_name: str | None = None  # optional: identify the user for demo / group testing


class CreateSessionResponse(BaseModel):
    session_id: str
    thread_id: str
    user_id: str
    message: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    classification: dict | None = None


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]


class FeedbackRequest(BaseModel):
    rating: int | None = None   # 1–5
    comment: str | None = None


class FeedbackResponse(BaseModel):
    saved: bool
    emailed: bool
    message: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_last_ai_reply(graph_output: dict) -> str:
    """Pull the last AI text message from a graph stream chunk or final state."""
    all_messages = graph_output.get("messages", [])
    for msg in reversed(all_messages):
        role = getattr(msg, "type", getattr(msg, "role", ""))
        content = getattr(msg, "content", "")
        if role in {"ai", "assistant"} and content:
            if isinstance(content, list):
                parts = [
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                    if not (isinstance(item, dict) and item.get("type") == "tool_use")
                ]
                text = "".join(parts).strip()
                if text:
                    return text
            else:
                return str(content).strip()
    return ""


# ---------------------------------------------------------------------------
# Feedback email helper
# ---------------------------------------------------------------------------

def _send_feedback_email(user_name: str, rating: int | None, comment: str, transcript: str) -> None:
    """Send feedback email to the fixed support address."""
    smtp_host     = settings.escalation_smtp_host
    smtp_port     = settings.escalation_smtp_port
    smtp_user     = settings.escalation_smtp_sender
    smtp_password = settings.escalation_smtp_password.get_secret_value()
    recipient     = "support.telekom.bot@gmail.com"

    if not smtp_user or not smtp_password:
        raise RuntimeError("SMTP not configured")

    today    = date.today().isoformat()
    stars    = ("\u2605" * rating + "\u2606" * (5 - rating)) if rating else "n/a"
    subject  = f"[FEEDBACK] {today} | {stars} | {(user_name or 'anonymous')[:40]}"

    body = (
        f"Дата: {today}\n"
        f"Користувач: {user_name or 'anonymous'}\n"
        f"Оцінка: {stars}\n\n"
        f"Коментар:\n{comment or '(без коментаря)'}\n\n"
        "Повна транскрипція розмови додана як вкладення."
    )

    msg = MIMEMultipart()
    msg["From"]    = smtp_user
    msg["To"]      = recipient
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    attachment = MIMEBase("application", "octet-stream")
    attachment.set_payload(transcript.encode("utf-8"))
    encoders.encode_base64(attachment)
    attachment.add_header(
        "Content-Disposition", "attachment",
        filename=f"chat_transcript_{today}.txt",
    )
    msg.attach(attachment)

    ctx = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.ehlo()
        server.starttls(context=ctx)
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, recipient, msg.as_string())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
@_limiter.limit(_RATE_SESSIONS)
async def create_session(body: CreateSessionRequest = CreateSessionRequest(), *, request: Request):
    """Start a new support chat session.

    Pass an optional user_name to identify yourself during group demos:
      {"user_name": "Anna"}

    Returns a session_id that must be included in all subsequent requests.
    Rate limited to 30 sessions/minute per IP.
    """
    session_id = f"s-{uuid4().hex[:12]}"
    thread_id  = f"t-{uuid4().hex[:12]}"

    if body.user_name and body.user_name.strip():
        user_id = await asyncio.to_thread(get_or_create_user_by_name, body.user_name)
    else:
        user_id = await asyncio.to_thread(get_or_create_active_user_id)

    db_sid = await asyncio.to_thread(start_new_session, user_id)

    _sessions[session_id] = {
        "thread_id":  thread_id,
        "user_id":    user_id,
        "db_session": db_sid,
        "user_name":  (body.user_name or "").strip(),
        "created_at": datetime.now(timezone.utc),
    }

    return CreateSessionResponse(
        session_id=session_id,
        thread_id=thread_id,
        user_id=user_id,
        message="Session created. Send your first message to /sessions/{session_id}/chat",
    )


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse)
@_limiter.limit(_RATE_CHAT)
async def chat(session_id: str, body: ChatRequest, request: Request):
    """Send a message and get the AI response.

    The graph routes the message through:
      Router → Docs Agent / Web Search Agent / Escalation Agent

    Uses graph.ainvoke() so the event loop stays free while the LLM
    is generating — multiple clients are served concurrently.
    Rate limited to 30 messages/minute per IP.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found. Create one at POST /sessions")
    if _is_session_expired(_sessions[session_id]):
        _sessions.pop(session_id, None)
        raise HTTPException(status_code=404, detail="Session expired. Create a new one at POST /sessions")

    sess = _sessions[session_id]
    thread_id = sess["thread_id"]
    db_sid    = sess["db_session"]
    user_id   = sess["user_id"]

    await asyncio.to_thread(save_message, db_sid, "user", body.message)

    # Pass db_session_id via configurable so send_escalation_email can read it
    # without relying on a global variable (safe for concurrent multi-user requests).
    # callbacks + metadata ensure Langfuse traces and sessions are recorded.
    config = {
        "configurable": {"thread_id": thread_id, "db_session_id": db_sid},
        "callbacks": [get_langfuse_handler()],
        "metadata": {"langfuse_session_id": db_sid, "langfuse_user_id": user_id},
    }
    input_state = {
        "messages": [HumanMessage(content=body.message)],
        "session_id": db_sid,
        "classification": {},
    }

    try:
        graph = _app_state["graph"]
        final_state = await graph.ainvoke(input_state, config=config)
    except Exception as exc:
        logger.error("Agent error for session %s: %s", session_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Внутрішня помилка сервера. Спробуйте ще раз.") from exc

    reply = _extract_last_ai_reply(final_state)
    if not reply:
        reply = "Виникла технічна помилка. Спробуйте ще раз."

    await asyncio.to_thread(save_message, db_sid, "assistant", reply)

    # Flush Langfuse buffer without blocking the event loop.
    try:
        await asyncio.to_thread(get_langfuse_client().flush)
    except Exception:
        pass

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        classification=final_state.get("classification"),
    )


@app.get("/sessions/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Return the current conversation state from the LangGraph checkpointer."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    thread_id = _sessions[session_id]["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}

    try:
        graph = _app_state["graph"]
        state = await asyncio.to_thread(graph.get_state, config)
        messages = [
            {
                "role": getattr(m, "type", getattr(m, "role", "unknown")),
                "content": str(getattr(m, "content", "")),
            }
            for m in (state.values.get("messages") or [])
        ]
    except Exception:
        messages = []

    return HistoryResponse(session_id=session_id, messages=messages)


@app.delete("/sessions/{session_id}", status_code=204)
async def close_session(session_id: str):
    """Close a session and mark it as finished in the DB."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    db_sid = _sessions.pop(session_id)["db_session"]
    await asyncio.to_thread(finish_session, db_sid)


@app.post("/sessions/{session_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(session_id: str, body: FeedbackRequest):
    """Save user feedback (rating + comment) to DB and send to support email.

    Feedback email is always sent to support.telekom.bot@gmail.com.
    The chat transcript is attached automatically.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    sess      = _sessions[session_id]
    db_sid    = sess["db_session"]
    user_name = sess.get("user_name", "")

    rating  = body.rating
    comment = (body.comment or "").strip()

    if rating is not None and not (1 <= rating <= 5):
        raise HTTPException(status_code=422, detail="rating must be 1–5")

    # Persist to DB
    await asyncio.to_thread(save_feedback, db_sid, user_name, rating, comment)

    # Send email (non-blocking)
    emailed = False
    try:
        transcript = await asyncio.to_thread(format_chat_transcript, db_sid)
        await asyncio.to_thread(
            _send_feedback_email, user_name or "anonymous", rating, comment, transcript
        )
        emailed = True
    except Exception as exc:
        logger.warning("Feedback email failed for session %s: %s", session_id, exc, exc_info=True)

    return FeedbackResponse(
        saved=True,
        emailed=emailed,
        message="Feedback saved" + (". Email sent." if emailed else " (email not sent — check SMTP config)."),
    )


@app.get("/")
async def root():
    """Serve the chat UI."""
    chat_file = Path(__file__).parent / "chat.html"
    if chat_file.exists():
        return FileResponse(chat_file, media_type="text/html")
    return {"service": "Telekom Support API", "version": "1.0", "docs": "/docs"}


@app.get("/health")
async def health():
    # Clean up expired sessions opportunistically on health check
    expired = [sid for sid, s in list(_sessions.items()) if _is_session_expired(s)]
    for sid in expired:
        _sessions.pop(sid, None)
    return {"status": "ok", "sessions_active": len(_sessions)}
