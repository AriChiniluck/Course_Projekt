"""
test_support.py — Tests for the Telekom customer support multi-agent system.

Structure:
  1. Router classification     — correct category + urgency (LLM-as-a-Judge)
  2. Tool correctness          — search_customer, get_customer_sims (deterministic)
  3. Docs Agent quality        — KB answers without hallucination (LLM-as-a-Judge)
  4. WebSearch Agent           — tariff lookup (LLM-as-a-Judge)
  5. Router escalation signal  — critical messages route correctly (LLM-as-a-Judge)
  6. E2E support pipeline      — full graph: message in -> reply out (LLM-as-a-Judge)

Run:
    deepeval test run tests/test_support.py
    DEBUG=1 deepeval test run tests/test_support.py   <- verbose
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import deepeval
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import HumanMessage

from config import settings
from agents.router import classify_message
from agents.docs_agent import get_docs_agent
from agents.websearch_agent import get_websearch_agent

EVAL_MODEL = os.getenv("DEEPEVAL_MODEL", settings.eval_model)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_ai_message(invoke_result: dict) -> str:
    """Extract the last AI text from an agent invoke() result."""
    for msg in reversed(invoke_result.get("messages", [])):
        role = getattr(msg, "type", getattr(msg, "role", ""))
        if role in {"ai", "assistant"}:
            content = getattr(msg, "content", "")
            if isinstance(content, list):
                return "".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ).strip()
            return str(content).strip()
    return ""


def _run_support_graph(message: str) -> str:
    """Run the full support StateGraph and return the last AI reply."""
    from support_graph import build_support_graph

    db_path = str(Path(__file__).parent.parent / "data" / "checkpoints_test.sqlite")
    graph = build_support_graph(session_id="test_session", db_path=db_path)
    thread_id = f"test-{uuid4().hex[:8]}"
    config = {
        "configurable": {
            "thread_id": thread_id,
            "db_session_id": "test_session",
        }
    }
    state = {
        "messages": [HumanMessage(content=message)],
        "session_id": "test_session",
        "classification": {},
    }
    result = graph.invoke(state, config=config)
    return _last_ai_message(result)


# ---------------------------------------------------------------------------
# Shared judge metrics
# ---------------------------------------------------------------------------

router_judge = GEval(
    name="Router Classification Accuracy",
    evaluation_steps=[
        "The 'input' is a raw customer message sent to the Router.",
        "The 'actual output' is a JSON object with 'category', 'urgency', 'language'.",
        "The 'expected output' states which category is correct.",
        "Award score 1.0 when 'category' in actual output EXACTLY matches the expected category.",
        "Award score 0.0 when the category is wrong.",
        "Ignore urgency or language mismatches — only category matters.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=EVAL_MODEL,
    threshold=0.8,
)

kb_answer_judge = GEval(
    name="KB Answer Quality",
    evaluation_steps=[
        "The 'input' is a customer question answerable from the internal knowledge base.",
        "The 'actual output' is the agent's response.",
        "The 'expected output' describes what a correct answer should contain.",
        "Score 1.0 if the answer is specific and does NOT contain 'NEEDS_ESCALATION'.",
        "Score 0.5 if the answer is vague but does not hallucinate.",
        "Score 0.0 if the answer contradicts the expected output or hallucinates.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=EVAL_MODEL,
    threshold=0.6,
)

escalation_judge = GEval(
    name="Escalation Signal Correctness",
    evaluation_steps=[
        "The 'input' is a critical customer query.",
        "The 'actual output' is the Router's classification JSON.",
        "Score 1.0 if category is 'critical', 0.0 otherwise.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=EVAL_MODEL,
    threshold=0.8,
)

web_answer_judge = GEval(
    name="Web Search Answer Quality",
    evaluation_steps=[
        "The 'input' is a customer question about a tariff.",
        "The 'actual output' is the Web Search Agent's response.",
        "Score 1.0 if the response provides tariff info or states conditions are contractual.",
        "Score 0.5 if some tariff info is present but no source.",
        "Score 0.0 if the response is empty, off-topic, or hallucinates.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=EVAL_MODEL,
    threshold=0.5,
)

e2e_judge = GEval(
    name="E2E Support Response Quality",
    evaluation_steps=[
        "The 'input' is a customer message to the support system.",
        "The 'actual output' is the agent's final reply.",
        "The 'expected output' describes what a good reply should contain.",
        "Score 1.0 if the reply directly addresses the input with relevant facts.",
        "Score 0.5 if the reply is partially relevant or asks a clarifying question.",
        "Score 0.0 if the reply is empty, off-topic, or is a raw error message.",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model=EVAL_MODEL,
    threshold=0.6,
)


# ---------------------------------------------------------------------------
# 1. Router classification
# ---------------------------------------------------------------------------

ROUTER_CASES = [
    ("774466377",                                     "product",  "Account number -> identify customer"),
    ("Чому така висока сума у рахунку цього місяця?", "product",  "Billing question"),
    ("Що входить у тариф IoT 17?",                   "product",  "Tariff FAQ"),
    ("Всі наші номери заблоковані! ЄДРПОУ 00991887",  "critical", "Blocked numbers"),
    ("Хочу перейти з IoT 17 на IoT 45",               "critical", "Tariff change request"),
    ("Чи підтримує ваш API інтеграцію з Zapier?",     "general",  "Technical / external"),
    ("Як налаштувати IoT SIM для роботи через VPN?",  "general",  "Technical question"),
    ("Потрібен живий оператор!",                      "critical", "Human escalation request"),
    ("Скільки коштує IoT 25?",                        "product",  "Tariff price inquiry"),
    ("Мій менеджер не відповідає вже тиждень!",       "critical", "Angry complaint"),
]


@pytest.mark.parametrize("message,expected_category,description", ROUTER_CASES)
def test_router_classification(message: str, expected_category: str, description: str) -> None:
    """Router correctly classifies customer messages into product/general/critical."""
    result = classify_message([HumanMessage(content=message)])
    actual_json = json.dumps(result.model_dump(), ensure_ascii=False)
    test_case = LLMTestCase(
        input=message,
        actual_output=actual_json,
        expected_output=f"category must be '{expected_category}' — {description}",
    )
    assert_test(test_case, [router_judge])


# ---------------------------------------------------------------------------
# 2. Tool correctness — search_customer / get_customer_sims (no LLM judge)
# ---------------------------------------------------------------------------

def test_search_customer_by_account() -> None:
    """search_customer finds a client by account number (account 774466377 = Client 2)."""
    from support_tools import search_customer

    result = search_customer.invoke({"query": "774466377"})
    assert "774466377" in result or "33333994444" in result, (
        f"Expected account or EDRPOU in result, got: {result[:300]}"
    )


def test_search_customer_by_edrpou() -> None:
    """search_customer finds client by EDRPOU (EDRPOU 11113554881 = Client 1)."""
    from support_tools import search_customer

    result = search_customer.invoke({"query": "11113554881"})
    assert "111222555" in result or "11113554881" in result, (
        f"Expected client 1 data, got: {result[:300]}"
    )


def test_search_customer_not_found() -> None:
    """search_customer returns a not-found message for unknown identifiers."""
    from support_tools import search_customer

    result = search_customer.invoke({"query": "0000000000"})
    result_lower = result.lower()
    assert (
        "не знайдено" in result_lower
        or "не знайд" in result_lower
        or "нічого" in result_lower
    ), f"Expected not-found message, got: {result[:300]}"


def test_search_customer_too_short() -> None:
    """search_customer rejects queries that are too short."""
    from support_tools import search_customer

    result = search_customer.invoke({"query": "12"})
    assert "короткий" in result.lower() or "занадто" in result.lower(), (
        f"Expected too-short rejection, got: {result[:200]}"
    )


def test_get_customer_sims_returns_data() -> None:
    """get_customer_sims returns SIM list for a known account number."""
    from support_tools import get_customer_sims

    result = get_customer_sims.invoke({"account_number": "774466377"})
    assert len(result) > 20, f"Expected non-trivial response, got: {result[:200]}"
    assert "директорія" not in result.lower(), f"Unexpected directory error: {result[:200]}"


def test_get_customer_sims_unknown_account() -> None:
    """get_customer_sims returns not-found for unknown account."""
    from support_tools import get_customer_sims

    result = get_customer_sims.invoke({"account_number": "999999999"})
    result_lower = result.lower()
    assert (
        "не знайдено" in result_lower
        or "нічого" in result_lower
        or "відсутні" in result_lower
    ), f"Expected not-found message, got: {result[:300]}"


# ---------------------------------------------------------------------------
# 3. Docs Agent — KB quality (LLM-as-a-Judge)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question,expected_content", [
    (
        "Що входить в тариф IoT 17?",
        "The answer should mention the IoT 17 tariff, its monthly fee or included services. "
        "Should not hallucinate services not in KB.",
    ),
    (
        "Чому у клієнта можуть бути заблоковані номери?",
        "The answer should mention at least one real cause: debt/overdue payment, "
        "credit limit exceeded, or blocking by the customer's own request.",
    ),
    (
        "Що таке кредитний ліміт і як він впливає на обслуговування?",
        "The answer should explain that credit limit is a threshold after which "
        "services may be suspended.",
    ),
])
def test_docs_agent_kb_answer(question: str, expected_content: str) -> None:
    """Docs Agent answers KB questions without hallucinating."""
    agent = get_docs_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    last_ai = _last_ai_message(result)
    test_case = LLMTestCase(
        input=question,
        actual_output=last_ai,
        expected_output=expected_content,
    )
    assert_test(test_case, [kb_answer_judge])


# ---------------------------------------------------------------------------
# 4. WebSearch Agent — tariff lookup (LLM-as-a-Judge)
# ---------------------------------------------------------------------------

def test_websearch_agent_tariff_lookup() -> None:
    """Web Search Agent fetches tariff info from the operator website."""
    question = "Які умови тарифу IoT 17?"
    agent = get_websearch_agent()
    result = agent.invoke({"messages": [HumanMessage(content=question)]})
    last_ai = _last_ai_message(result)
    test_case = LLMTestCase(
        input=question,
        actual_output=last_ai,
        expected_output=(
            "The answer should include tariff conditions (monthly fee, included services) "
            "or clearly state that conditions are individual/contractual. "
            "A source URL is a bonus but not required."
        ),
    )
    assert_test(test_case, [web_answer_judge])


# ---------------------------------------------------------------------------
# 5. Router escalation signal (LLM-as-a-Judge)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message,expected_category", [
    ("Всі SIM-картки заблоковані! Терміново!",                "critical"),
    ("Хочу змінити договір і перейти на інший тарифний план", "critical"),
    ("Мені потрібен живий менеджер зараз же!",                "critical"),
])
def test_router_routes_critical_to_escalation(message: str, expected_category: str) -> None:
    """Router classifies critical messages correctly so they reach Escalation Agent."""
    result = classify_message([HumanMessage(content=message)])
    actual_json = json.dumps(result.model_dump(), ensure_ascii=False)
    test_case = LLMTestCase(
        input=message,
        actual_output=actual_json,
        expected_output=f"category must be '{expected_category}'",
    )
    assert_test(test_case, [escalation_judge])


# ---------------------------------------------------------------------------
# 6. E2E support pipeline (full StateGraph: router -> agent -> reply)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("message,expected", [
    (
        "111222555",
        "Reply should mention the client name or account 111222555 or tariff IoT, "
        "confirming the customer was identified from billing data.",
    ),
    (
        "Що таке кредитний ліміт?",
        "Reply should explain what a credit limit is in the context of Telekom services. "
        "Should be informative and not a raw error.",
    ),
    (
        "Hello, can you help me with my account?",
        "Reply should be in English (matching the client language) and ask for "
        "an account identifier to proceed.",
    ),
])
def test_e2e_support_pipeline(message: str, expected: str) -> None:
    """Full graph (router -> docs/websearch/escalation) returns a relevant reply."""
    reply = _run_support_graph(message)
    assert reply, "Support graph returned an empty reply"
    test_case = LLMTestCase(
        input=message,
        actual_output=reply,
        expected_output=expected,
    )
    assert_test(test_case, [e2e_judge])