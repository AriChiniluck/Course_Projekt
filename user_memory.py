from __future__ import annotations

"""Local persistence helpers for the future course project.

Why we store this now:
- Lecture 12 needs session and user tracking for Langfuse.
- The later support project will need returning-user context and ticket history.

Privacy choice:
- We deliberately DO NOT identify people by IP or MAC address.
- IP can change and multiple users can share one network.
- MAC is sensitive device data and should not be collected casually.
- Instead, we create a pseudonymous local user id and persist it on the device.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "support_memory.db"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_schema() -> None:
    """Create the minimal tables we want as a strong base for the course project."""
    with _get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                preferred_language TEXT,
                profile_summary TEXT,
                consent_scope TEXT DEFAULT 'local_device'
            );

            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                topic_summary TEXT,
                resolution_status TEXT DEFAULT 'open',
                escalated INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS identified_customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                account_number TEXT,
                client_name TEXT,
                identifier TEXT,
                phone TEXT,
                tariff_name TEXT,
                source_file TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS escalations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                customer_info TEXT,
                target TEXT NOT NULL,
                email_subject TEXT,
                email_summary TEXT,
                email_full_text TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );

            CREATE TABLE IF NOT EXISTS feedbacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_name TEXT,
                rating INTEGER,
                comment TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            """
        )


def get_or_create_active_user_id() -> str:
    """Return a stable pseudonymous user id stored only on the local machine."""
    ensure_schema()
    with _get_connection() as conn:
        row = conn.execute("SELECT user_id FROM users ORDER BY created_at LIMIT 1").fetchone()
        if row:
            user_id = str(row["user_id"])
            conn.execute(
                "UPDATE users SET updated_at = ? WHERE user_id = ?",
                (_utc_now(), user_id),
            )
            return user_id

        user_id = f"local_user_{uuid4().hex[:12]}"
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO users (user_id, created_at, updated_at, preferred_language, profile_summary, consent_scope)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, now, now, "uk", "Local returning user profile", "local_device"),
        )
        return user_id


def get_or_create_user_by_name(name: str) -> str:
    """Return a stable user_id for a given display name (for demo / group testing).

    Looks up by profile_summary field that stores the name. Creates a new user
    if no match found. Same name across devices → same user_id in the DB.
    """
    ensure_schema()
    safe_name = name.strip()[:64]
    user_id = f"demo_{safe_name.lower().replace(' ', '_')}"
    now = _utc_now()
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT user_id FROM users WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            conn.execute("UPDATE users SET updated_at = ? WHERE user_id = ?", (now, user_id))
            return user_id
        conn.execute(
            """
            INSERT INTO users (user_id, created_at, updated_at, preferred_language, profile_summary, consent_scope)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, now, now, "uk", safe_name, "demo"),
        )
    return user_id


def start_new_session(user_id: str) -> str:
    """Create a new session row and return the session id."""
    ensure_schema()
    session_id = f"support_session_{uuid4().hex[:12]}"
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (session_id, user_id, started_at) VALUES (?, ?, ?)",
            (session_id, user_id, _utc_now()),
        )
    return session_id


def save_message(session_id: str, role: str, content: str) -> None:
    """Persist one chat message locally for future follow-up and support history."""
    if not str(content or "").strip():
        return
    ensure_schema()
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, str(content), _utc_now()),
        )


def finish_session(session_id: str, topic_summary: str = "", resolution_status: str = "open", escalated: bool = False) -> None:
    """Mark session outcome so later analytics can count solved vs escalated cases."""
    ensure_schema()
    with _get_connection() as conn:
        conn.execute(
            """
            UPDATE sessions
            SET ended_at = ?, topic_summary = ?, resolution_status = ?, escalated = ?
            WHERE session_id = ?
            """,
            (_utc_now(), topic_summary[:500], resolution_status, 1 if escalated else 0, session_id),
        )


def save_identified_customer(
    session_id: str,
    account_number: str = "",
    client_name: str = "",
    identifier: str = "",
    phone: str = "",
    tariff_name: str = "",
    source_file: str = "",
) -> None:
    """Save identified customer info for this support session."""
    ensure_schema()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO identified_customers
                (session_id, account_number, client_name, identifier, phone, tariff_name, source_file, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                account_number[:200],
                client_name[:500],
                identifier[:100],
                phone[:50],
                tariff_name[:200],
                source_file[:200],
                _utc_now(),
            ),
        )


def save_escalation(
    session_id: str,
    customer_info: str = "",
    target: str = "",
    email_subject: str = "",
    email_summary: str = "",
    email_full_text: str = "",
) -> None:
    """Persist escalation record and mark the session as escalated."""
    ensure_schema()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO escalations
                (session_id, customer_info, target, email_subject, email_summary, email_full_text, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                customer_info[:1000],
                target[:50],
                email_subject[:500],
                email_summary[:2000],
                email_full_text,
                _utc_now(),
            ),
        )
        conn.execute(
            "UPDATE sessions SET escalated = 1 WHERE session_id = ?",
            (session_id,),
        )


def get_session_messages(session_id: str) -> list[dict]:
    """Return all messages for a session as a list of dicts."""
    ensure_schema()
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [
        {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
        for row in rows
    ]


def format_chat_transcript(session_id: str) -> str:
    """Return the full chat transcript as a formatted string (for email attachments)."""
    messages = get_session_messages(session_id)
    if not messages:
        return "(No messages recorded for this session)"
    lines = [f"Chat Transcript — Session: {session_id}", "=" * 60, ""]
    for msg in messages:
        label = "Клієнт" if msg["role"] == "user" else "Агент"
        ts = str(msg["created_at"])[:19]
        lines.append(f"[{ts}] {label}:")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)

def save_feedback(
    session_id: str,
    user_name: str = "",
    rating: int | None = None,
    comment: str = "",
) -> None:
    """Persist user feedback for a session."""
    ensure_schema()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO feedbacks (session_id, user_name, rating, comment, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, (user_name or "")[:200], rating, (comment or "")[:2000], _utc_now()),
        )