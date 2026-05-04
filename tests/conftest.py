"""
Shared pytest configuration and fixtures for the multi-agent test suite.

Debug mode (enables verbose step-by-step agent output, same as 'debug on' in the REPL):
  DEBUG=1 deepeval test run tests/           → via environment variable
  deepeval test run tests/ --agent-debug     → via custom option (avoids clashing with
                                               deepeval's own --debug flag)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ── make project root importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from config import settings


def pytest_addoption(parser: pytest.Parser) -> None:
    # NOTE: deepeval 3.x already registers --debug; use --agent-debug to avoid conflict.
    parser.addoption(
        "--agent-debug",
        action="store_true",
        default=False,
        help=(
            "Enable agent debug mode: print every tool call, tool result, and verdict "
            "the same way as 'debug on' in the interactive REPL."
        ),
    )


@pytest.fixture(autouse=True, scope="session")
def configure_debug(request: pytest.FixtureRequest) -> None:
    """Enable debug output for the whole test session when --agent-debug / DEBUG=1 is set."""
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    if request.config.getoption("--agent-debug", default=False) or os.getenv("DEBUG") == "1":
        settings.debug = True

    # deepeval creates its own OpenAI client — make sure the key is available
    os.environ.setdefault(
        "OPENAI_API_KEY",
        settings.openai_api_key.get_secret_value(),
    )


@pytest.fixture(scope="session")
def eval_model() -> str:
    """Model name used for deepeval GEval / metric evaluation.

    Override with env var DEEPEVAL_MODEL, otherwise reuse the project's own model.
    """
    return os.getenv("DEEPEVAL_MODEL", settings.eval_model)
