"""Entry point for the Telekom customer support multi-agent chat system.

Run:
    python support_main.py

Type 'exit', 'quit', or 'вийти' to end the session.
"""
from __future__ import annotations

import sys
from uuid import uuid4

from config import settings
from observability import get_langfuse_client, get_langfuse_handler
from user_memory import (
    finish_session,
    get_or_create_active_user_id,
    save_message,
    start_new_session,
)
from agents.support_supervisor import build_support_supervisor

# ---------------------------------------------------------------------------
# Session setup — one session per process run
# ---------------------------------------------------------------------------

THREAD_ID  = f"support-{uuid4().hex[:8]}"
USER_ID    = get_or_create_active_user_id()
SESSION_ID = start_new_session(USER_ID)

# Build the single-agent supervisor (REPL mode, backward-compatible)
_supervisor = build_support_supervisor(session_id=SESSION_ID)
_CONFIG     = {"configurable": {"thread_id": THREAD_ID, "db_session_id": SESSION_ID}}

_QUIT_COMMANDS = {"exit", "quit", "вийти", "вихід", "стоп", "stop", "q"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _console_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        print(safe)


def _extract_ai_reply(chunk: dict) -> str:
    """Pull the last AI text message from a streamed chunk."""
    # LangGraph streams chunks as dicts with node-name keys, e.g. {"agent": {"messages": [...]}}
    # Try all top-level values that look like dicts with messages
    sources = []
    if "messages" in chunk:
        sources.append(chunk)
    for v in chunk.values():
        if isinstance(v, dict) and "messages" in v:
            sources.append(v)

    for source in sources:
        for msg in reversed(source.get("messages", [])):
            content = getattr(msg, "content", "")
            msg_type = getattr(msg, "type", getattr(msg, "role", ""))
            if msg_type in {"ai", "assistant"} and content:
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


def _stream_response(user_input: str) -> str:
    """Stream supervisor response and return the final AI text."""
    last_reply = ""
    lf_handler = get_langfuse_handler()
    stream_config = {
        **_CONFIG,
        "callbacks": [lf_handler],
        "metadata": {"langfuse_session_id": SESSION_ID},
    }
    for chunk in _supervisor.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config=stream_config,
    ):
        if settings.debug:
            print(f"[DEBUG chunk keys]: {list(chunk.keys())}", flush=True)
            for k, v in chunk.items():
                if isinstance(v, dict):
                    msgs = v.get("messages", [])
                    for m in msgs:
                        print(f"  [{k}] type={getattr(m,'type','?')} role={getattr(m,'role','?')} content={str(getattr(m,'content',''))[:80]}", flush=True)
        reply = _extract_ai_reply(chunk)
        if reply:
            last_reply = reply

    return last_reply


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------

def run_support_chat() -> None:
    _console_print("")
    _console_print("=" * 62)
    _console_print("  Telekom Support — мультиагентна система")
    _console_print(f"  Session: {SESSION_ID}")
    _console_print("  Введіть 'exit' або 'вийти' для завершення розмови")
    _console_print("=" * 62)
    _console_print("")

    # Send a silent greeting trigger so the agent opens the conversation
    greeting_reply = _stream_response(
        "Привіт! Починаємо нову сесію підтримки. Привітайся з клієнтом."
    )
    if greeting_reply:
        _console_print(f"Агент: {greeting_reply}\n")
        save_message(SESSION_ID, "assistant", greeting_reply)

    while True:
        try:
            user_input = input("Ви: ").strip()
        except (EOFError, KeyboardInterrupt):
            _console_print("\n[Сесію перервано]")
            break

        if not user_input:
            continue

        if user_input.lower() in _QUIT_COMMANDS:
            break

        # Persist user message
        save_message(SESSION_ID, "user", user_input)

        try:
            reply = _stream_response(user_input)
        except Exception as exc:
            reply = (
                f"[Системна помилка: {exc}] "
                "Вибачте, сталася технічна помилка. Спробуйте ще раз."
            )

        if reply:
            _console_print(f"\nАгент: {reply}\n")
            save_message(SESSION_ID, "assistant", reply)
        else:
            _console_print("\n[Агент не повернув відповідь]\n")

    # Close session
    finish_session(
        SESSION_ID,
        topic_summary="Support conversation",
        resolution_status="completed",
    )
    # Flush Langfuse buffer once — send all traces collected during the session.
    try:
        get_langfuse_client().flush()
    except Exception:
        pass
    _console_print("\nДякуємо за звернення. Гарного дня!")


if __name__ == "__main__":
    run_support_chat()
