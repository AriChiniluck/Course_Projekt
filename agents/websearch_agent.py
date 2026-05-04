"""Web Search Agent — handles 'general' queries via live web lookup.

Tools available:
  • lookup_tariff_telekom — fetches the operator tariff archive + DuckDuckGo fallback

Escalation signal: prepend "NEEDS_ESCALATION: <reason>" when nothing useful
was found, so the SupportGraph routes to the Escalation Agent.
"""
from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent

from config import build_chat_model, settings

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
Ти — технічний консультант Telekom. Відповідай тією мовою, якою пише клієнт. Якщо клієнт пише російською — відповідай виключно українською.

═══ ТВІЙ ІНСТРУМЕНТ ═══

• lookup_tariff_telekom(tariff_name)
  Перевіряє умови тарифного плану на офіційному сайті оператора.
  Якщо тариф не знайдено — умови є індивідуальними (договірними).

═══ ПРАВИЛА ═══

1. Відповідай ТІЛЬКИ на питання про тарифи, послуги та умови оператора Telekom.
   Якщо питання не стосується телекомунікаційних послуг Telekom — ввічливо
   відмов і запропонуй допомогу в межах підтримки Telekom.
   Формулювання: «Це виходить за межі моїх можливостей як агента підтримки
   Telekom. Чим можу допомогти щодо ваших послуг зв'язку?»

2. Відповідай на технічні або загальні питання про Telekom, відповіді на які
   відсутні у внутрішній базі знань (інтеграції, сумісність, зовнішні сервіси).

3. Якщо відповідь не вдалося знайти — починай ТОЧНО так:
   "NEEDS_ESCALATION: " і далі коротко причина.

4. Не вигадуй — спирайся тільки на дані з інструменту.

5. НЕ ПРОЩАЙСЯ після кожної відповіді. Прощайся лише якщо клієнт явно
   завершує розмову. Якщо клієнт, схоже, отримав відповідь — запитай:
   «Чи є ще щось, з чим я можу вам допомогти?»
   Тільки після «ні, дякую» — побажай гарного дня і попрощайся.
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_websearch_agent():
    """Return a cached, stateless Web Search Agent (no checkpointer)."""
    from support_tools import lookup_tariff_telekom  # noqa: PLC0415

    return create_agent(
        model=build_chat_model(temperature=0.2, model=settings.supervisor_model),
        tools=[lookup_tariff_telekom],
        system_prompt=_SYSTEM,
    )
