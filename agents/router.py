"""Router — classifies incoming customer messages with structured output.

This is the entry-point node of the SupportGraph (Routing pattern).
It does NOT call tools; it only classifies the latest user message and
returns a ClassificationOutput that the conditional edge uses to pick
the next agent (Docs / WebSearch / Escalation).
"""
from __future__ import annotations

from langchain_core.messages import BaseMessage, SystemMessage

from config import build_chat_model, settings
from schemas import ClassificationOutput

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are a router for the Telekom customer support system.
Your ONLY job: classify the customer's LAST message into exactly one category
and return a structured JSON object. Do NOT answer the question.

CATEGORIES:
- "product"    — account/SIM lookup, tariff/billing questions, FAQ about the
                 operator's own services, customer identification
- "general"    — technical questions about Telekom services, third-party integrations
                 relevant to the operator (e.g. API compatibility, roaming, device settings),
                 questions whose answers are NOT in the internal knowledge base
- "critical"   — ALL of the following trigger "critical":
                 • blocked numbers or SIMs
                 • urgent complaints (angry tone, all-caps, escalation demands)
                 • requests to speak with a human operator
                 • tariff-change or contract-review requests
- "off_topic"  — ANYTHING unrelated to Telekom services:
                 • weather, news, sports, entertainment, geography, history
                 • general knowledge or trivia not about telecom
                 • questions about other companies, products, or services
                 • coding help, math, personal advice, or any non-telecom topic

URGENCY:
- "low"      — routine informational request
- "medium"   — billing question, minor complaint
- "critical" — blocked numbers / data loss / angry customer / tariff change

LANGUAGE: detect the customer's language and return a short ISO code
          ("uk", "en", ...).

IMPORTANT EDGE CASES:
• Greeting + only a phone/account number → "product" (customer identification)
• "Привіт / Hello" alone → "product" (will ask for credentials)
• Closing / farewell messages ("ні, дякую", "все, дякую", "до побачення",
  "дякую, все", "no thanks", "goodbye", "that's all") → "product"
  (agent will say goodbye politely)
• "Хочу змінити тариф" / "Перейти на інший план" → "critical" (sales escalation)
• "Всі номери заблоковані" → "critical", urgency "critical"
• Weather / news / general knowledge / off-telecom topics → "off_topic"
• "What else can you help with?" after a non-telecom topic → "off_topic"
• Questions about connecting, configuring, or using the Telekom service
  ("як підключити провайдера?", "як налаштувати інтернет?", "як увімкнути роумінг?",
   "how to connect?", "how to set up?") → "general"
  These are technical how-to questions about the operator's own services.
• Questions about the bot's capabilities, what it can do, how it can help,
  requests to test it, or meta-questions about the support system → "product"
  Examples: "що ти вмієш?", "які твої можливості?", "хочу потестити тебе",
  "чим ти можеш бути корисний?", "розкажи про себе", "what can you do?"
• Questions about Telekom service CONCEPTS without personal context
  (e.g. "Що таке кредитний ліміт?", "Що входить у тариф IoT 17?",
   "Скільки коштує IoT 25?", "What is a credit limit?") → "product"
  The Docs Agent will answer from the knowledge base and ask for credentials
  only if needed to look up personal data.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_message(messages: list[BaseMessage], config=None) -> ClassificationOutput:
    """Return a ClassificationOutput for the latest user message.

    Passes the last 6 messages as context so the router has enough history
    (e.g. to know the customer was already identified two turns ago).
    """
    llm = build_chat_model(temperature=0.0, model=settings.planner_model)
    structured_llm = llm.with_structured_output(ClassificationOutput)

    context = messages[-6:] if len(messages) > 6 else list(messages)
    invoke_kwargs = {"config": config} if config else {}
    return structured_llm.invoke([SystemMessage(content=_SYSTEM)] + context, **invoke_kwargs)
