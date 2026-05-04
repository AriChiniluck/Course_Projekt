# Telekom Support — мультиагентна система підтримки

Курсовий проект з дисципліни **Multi-Agent Systems** (RobotDreams, 2026).

Система реалізує автоматизованого агента підтримки корпоративних клієнтів IoT-оператора.

---

## Можливості системи

- **Ідентифікація клієнта** — за номером телефону, особовим рахунком або ЄДРПОУ
- **Аналіз рахунків** — читає `.xlsx` файли з нарахуваннями, деталізує послуги
- **База знань (RAG)** — відповіді на FAQ по тарифах, блокуванні, кредитних лімітах
- **Пошук тарифів** — перевірка умов тарифів на сайті оператора (trafilatura + DuckDuckGo)
- **Ескалація** — надсилає email до служби підтримки або відділу продажів з транскриптом розмови
- **Персистентність** — всі сесії, повідомлення та ескалації зберігаються у SQLite; при ескалації до листа автоматично додається транскрипція поточної розмови
- **Багатомовність** — відповідає тією мовою, якою пише клієнт; якщо клієнт пише російською — відповідь виключно українською- **Чат-інтерфейс** — повноцінний Web UI (ук/ен/де), роздається FastAPI; меню з тестовими клієнтами + зворотний зв'язок
- **Off-topic захист** — агент одразу відмовляється відповідати на питання не по темі Telekom
---

## Архітектура

Система має **дві окремі точки входу** з різними архітектурами, що поділяють спільний шар інструментів та даних.

```
support_main.py (REPL)          api.py (FastAPI, multi-user)
        │                               │
        ▼                               ▼
agents/support_supervisor.py    support_graph.py
  (один ReAct-агент,            (LangGraph StateGraph:
   всі 5 інструментів)           Router → Docs/WebSearch/Escalation)
        │                               │
        └───────────────┬───────────────┘
                        │ СПІЛЬНЕ
                        ▼
        support_tools.py      ← 5 інструментів (search_customer,
        │                       get_customer_sims, search_telekom_kb,
        │                       lookup_tariff_telekom, send_escalation_email)
        user_memory.py        ← SQLite: sessions, messages, escalations
        observability.py      ← Langfuse tracing
        config.py             ← pydantic-settings, .env
        data/support_memory.db
        data/customers/*.xlsx
```

**Чому дві точки входу?**

| | `support_main.py` | `api.py` (FastAPI) |
|---|---|---|
| Призначення | Локальне тестування в терміналі | Продакшн, демонстрація, UI |
| Архітектура агента | Один ReAct-агент | StateGraph: Router + 3 спеціалізовані агенти |
| Кількість користувачів | 1 (один процес = одна сесія) | N паралельних HTTP-запитів |
| Стан розмови | `InMemorySaver` (губиться при exit) | `SqliteSaver` (зберігається між рестартами) |
| Langfuse tracing | ✅ | ✅ |

> **Мультиагентність** реалізована у FastAPI-режимі: 5 спеціалізованих агентів з LangGraph StateGraph + Router pattern.

```
Запит клієнта
      │
      │
 router.py       ← Агент 1: класифікує (product / general / critical / off_topic)
      │
      ├─ product   ▶  docs_agent.py        ← Агент 2: RAG + Excel
      ├─ general   ▶  websearch_agent.py   ← Агент 3: веб-пошук тарифів
      ├─ critical  ▶  escalation_agent.py  ← Агент 4: email ескалація
      └─ off_topic ▶  [вбудована відмова, без LLM] ← Агент 5: off-topic guard
```

> REPL-режим — single-agent для локального тестування.

> Обидва режими пишуть в одну `support_memory.db` — сесії не конфліктують, у кожної свій `session_id`.

```
# Режим 1: REPL (один користувач, один ReAct-агент)
support_main.py              ← точка входу, REPL-чат
│
agents/
└── support_supervisor.py    ← LangGraph ReAct-агент (всі 5 інструментів)

# Режим 2: FastAPI (багатокористувацька HTTP API, LangGraph-граф з 4 агентами)
run_api.py                   ← uvicorn запуск, :8000
api.py                       ← FastAPI: /sessions, /chat, /history, /health
support_graph.py             ← LangGraph StateGraph: Router → Docs/WebSearch/Escalation

agents/
  ├── router.py              ← класифікує запит по темі
  ├── docs_agent.py          ← рахунки та KB (search_customer + get_customer_sims + RAG)
  ├── websearch_agent.py     ← пошук тарифів в інтернеті
  └── escalation_agent.py    ← ескалація через email

# Спільне для обох режимів
support_tools.py             ← 5 інструментів:
  ├── search_customer          → пошук клієнта (ідентифікація по рахунку, ЄДРПОУ, телефону)
  ├── get_customer_sims        → перелік SIM з позатарифними витратами
  ├── search_telekom_kb        → RAG пошук по FAQ (FAISS + BM25)
  ├── lookup_tariff_telekom    → scraping сайту оператора
  └── send_escalation_email    → SMTP + збереження в БД

retriever.py             ← HybridRetriever (FAISS + BM25 + CrossEncoder reranker)
ingest.py                ← індексація документів з data/
config.py                ← pydantic-settings, читає .env
user_memory.py           ← SQLite: sessions, messages, escalations
```

### Дані клієнтів

```
data/
  ├── customers/           ← Excel-файли з рахунками (*.xlsx)
  ├── telekom_faq.md       ← Knowledge Base: FAQ
  ├── telekom_tariffs.md   ← Knowledge Base: тарифи
  └── telekom_test_scenarios.md
index/                   ← FAISS-індекс (генерується через ingest.py)
  └── chunks.json
```

---

## Швидкий старт для викладача / тестера

Система містить 6 тестових клієнтів (Excel-файли в `data/customers/`). Нижче — реальні ідентифікатори з файлів для перевірки пошуку та відповідей агента.

### Тестові клієнти

| Клієнт | Особовий рахунок | ЄДРПОУ | Тарифи | Кількість SIM | Діапазон телефонів |
|--------|-----------------|--------|--------|:-----------:|-------------------|
| **Клієнт 1** | `111222555` | `11113554881` | IoT 17/25/45/70/125/250 | 40 | 1239648 – 1239687 |
| **Клієнт 2** | `774466377` | `33333994444` | IoT 17/25/45 | 119 | 1230363 – 3340737 |
| **Клієнт 3** | `111333999` | `99999917798` | IoT 70 | 5 | 1235409 – 1235413 |
| **Клієнт 4** | `999477111` | `44443236907` | IoT 17/70/125 | 5 | 3210521 – 3210525 |
| **Клієнт 5** | `888777999`,`777999777`,`222777333`  | `22220048222` | IoT 17/25/45/70/125/250 | 115 | 4444163 – 4444277 |
| **Клієнт 6** | `666444777` | `11199988777` | IoT 17/25/45 | 49 | 2220631 – 2220679 |

> Телефони — це 7-значні IoT SIM-ідентифікатори (не звичайні мобільні номери).  
> Пошук спрацьовує по будь-якому з трьох ключів: особовий рахунок, ЄДРПОУ або номер телефону.

### Приклади запитів для демонстрації

```
# Пошук за особовим рахунком
774466377

# Пошук за ЄДРПОУ
11113554881

# Пошук за номером телефону
1235409

# Запит по рахунку
Клієнт 2, чому сума 67 грн при тарифі IoT 17?

# Блокування
У нас заблоковані всі номери! ЄДРПОУ 00991887

# Зміна тарифу (ескалація до sales)
Хочу перейти з IoT 17 на IoT 45. Рахунок 999477111
```

### Примітка щодо Excel-файлів

Файли зберігаються в `data/customers/` (всередині проекту). Якщо OneDrive синхронізується — файл тимчасово блокується і пошук по ньому повертає "не знайдено". Зачекайте завершення синхронізації.

---

## Встановлення

### 1. Клонувати репозиторій та перейти в папку

```bash
cd "HT Lektion 12"
```

### 2. Встановити залежності

```bash
pip install -r requirements.txt
```

### 3. Налаштувати `.env`

```bash
copy .env.example .env
```

Відкрити `.env` та заповнити:

| Змінна | Опис |
|--------|------|
| `openai_api_key` | Ключ OpenAI API |
| `ESCALATION_SMTP_HOST` | SMTP сервер (напр. `smtp.gmail.com`) |
| `ESCALATION_SMTP_PORT` | Порт (587 для TLS) |
| `ESCALATION_SMTP_SENDER` | Email відправника |
| `ESCALATION_SMTP_PASSWORD` | App Password від Gmail |
| `ESCALATION_EMAIL_SUPPORT` | Email служби підтримки |
| `ESCALATION_EMAIL_SALES` | Email відділу продажів |
| `TELEKOM_TARIFF_URL` | URL сторінки тарифів оператора |

> Gmail: для `ESCALATION_SMTP_PASSWORD` використовуйте **App Password** (Google Account → Security → 2FA → App passwords), а не звичайний пароль.

---

## Запуск

### Перший запуск — побудувати індекс Knowledge Base

```bash
python ingest.py
```

Виводить: `Loaded docs: 3 | Created chunks: N | Saved FAISS index`

### Запустити систему підтримки

**Режим 1 — REPL** (інтерактивний чат в терміналі):

```bash
python support_main.py
```

**Режим 2 — FastAPI** (багатокористувацький HTTP сервер):

```bash
python run_api.py
```

Після запуску (2-4 хвилини холодний старт):
- **http://127.0.0.1:8000/docs** — Swagger UI, інтерактивне тестування прямо в браузері
- **http://127.0.0.1:8000/health** — статус серверу

| Ендпоінт | Метод | Опис |
|-----------|--------|-------|
| `/sessions` | POST | Створити нову сесію |
| `/sessions/{id}/chat` | POST | Надіслати повідомлення |
| `/sessions/{id}` | GET | Історія повідомлень |
| `/sessions/{id}` | DELETE | Закрити сесію |
| `/sessions/{id}/feedback` | POST | Зворотний зв'язок (зірки + коментар) |
| `/health` | GET | Перевірка стану |

> **Увага:** завжди запускати з папки `HT Lektion 12`. Якщо термінал відкрито в батьківській папці — спочатку `cd "HT Lektion 12"`.

---

## Використання

Після запуску агент привітає клієнта і попросить ідентифікуватись:

```
================================================================
  Telekom Support — мультиагентна система
  Session: support_session_...
  Введіть 'exit' або 'вийти' для завершення розмови
================================================================

Агент: Вітаємо у службі підтримки Telekom Business! ...

Ви: 111222333
Агент: Дякую! Знайшов вас: Клієнт 1, тариф IoT 45. Чим можу допомогти?

Ви: Чому такий великий рахунок?
Агент: ...
```

### Ідентифікатори для тестування

| Клієнт | Особовий рахунок | ЄДРПОУ | Тарифи | SIM |
|--------|-----------------|--------|--------|-----|
| Клієнт 1 | `111222555` | `11113554881` | IoT 17/25/45/70/125/250 | 40 |
| Клієнт 2 | `774466377` | `33333994444` | IoT 17/25/45 | 119 |
| Клієнт 3 | `111333999` | `99999917798` | IoT 70 | 5 |
| Клієнт 4 | `999477111` | `44443236907` | IoT 17/70/125 | 5 |
| Клієнт 5 | `888777999`, `777999777`, `222777333` | `22220048222` | IoT 17/25/45/70/125/250 | 115 |
| Клієнт 6 | `666444777` | `11199988777` | IoT 17/25/45 | 49 |

---

## Оновлення Knowledge Base

Якщо додали або змінили файли в `data/`:

```bash
python ingest.py
```

Підтримуються формати: `.md`, `.txt`, `.pdf`

---

## Тестові сценарії

Файл [data/telekom_test_scenarios.md](data/telekom_test_scenarios.md) містить 8 готових сценаріїв:

1. Висока сума рахунку
2. Заблоковані всі номери
3. Інформаційний запит про тариф
4. Запит на зміну тарифу → ескалація sales
5. Кредитний ліміт
6. Клієнт пише російською
7. Емоційно збуджений клієнт
8. Клієнт не знайдений у базі

---

## База даних та ескалація

### SQLite — що зберігається автоматично

Всі дані зберігаються локально у SQLite (файл `support_memory.db`):

| Таблиця | Що записується |
|---------|---------------|
| `users` | Псевдонімний ID користувача (генерується один раз на пристрої) |
| `sessions` | Сесія: час початку/завершення, статус (`open`/`resolved`), прапорець ескалації |
| `messages` | Кожне повідомлення: роль (`user`/`assistant`), текст, час |
| `identified_customers` | Ідентифікований клієнт: рахунок, ім'я, тариф, файл-джерело |
| `escalations` | Ескалація: кому надіслано, тема, резюме, повний текст листа |
| `feedbacks` | Зворотний зв'язок: оцінка 1–5 зірочок, коментар, ім'я користувача |

Збереження відбувається автоматично — вручну нічого робити не потрібно.

### Ескалація — що отримує оператор

Коли агент викликає `send_escalation_email`, на пошту служби підтримки або відділу продажів надходить лист:

- **Тіло листа** — дата, інформація про клієнта, резюме звернення (3–5 речень)
- **Вкладення** — `chat_transcript_YYYY-MM-DD.txt` з повною транскрипцією поточної сесії (витягується з БД автоматично)

```
Від: telekom.escalation.demo@gmail.com
Кому: support.telekom.bot@gmail.com  ← або sales.telekom.bot@gmail.com
Тема: [subject, що задав агент]

Дата: 2026-04-30

Інформація про клієнта:
Клієнт 2, рахунок 774466377, тариф IoT 17, 119 SIM

Резюме звернення:
[3–5 речень від агента]

Повна транскрипція розмови додана як вкладення.

📎 chat_transcript_2026-04-30.txt
```

> Ескалація також фіксується в таблиці `escalations` (включно з повним текстом листа) — для аудиту та повторного аналізу.

---

## FastAPI — багатокористувацький режим

HTTP сервер повністю реалізовано в `api.py` + `run_api.py`. Кожен HTTP-запит обробляється окремим LangGraph-графом: Router → Docs/WebSearch/Escalation агент.

```
Клієнт 1 ──┐
Клієнт 2 ──┤── POST /sessions/{id}/chat
Клієнт 3 ──┘
                │
                ▼
         support_graph.py (Router)
                │
       ┌───────┼────────┐
       ▼        ▼        ▼
   docs_agent  websearch  escalation
   (RAG+SIM)  (web)      (email)
```

### Ідентифікація користувача (user_id)

При відкритті `chat.html` з'являється екран привітання з полем для імені. Ім'я передається у `POST /sessions` як `user_name`.

| Сценарій | `user_id` у БД |
|---|---|
| Без імені | `local_user_<hex>` (спільний для всіх анонімних сесій) |
| `user_name: "Клієнт 1"` | `demo_клієнт_1` |
| Те саме ім'я з іншого пристрою | `demo_клієнт_1` (той самий!) |

**Ізоляція розмов** забезпечується `session_id` + `thread_id` — вони завжди унікальні, незалежно від `user_id`. `user_id` використовується лише для аналітики: в SQLite (`sessions.user_id`) та в Langfuse (фільтр за `langfuse_user_id`).

```sql
-- Після демо: хто скільки сесій мав і скільки повідомлень написав
SELECT u.user_id, s.session_id, s.started_at, COUNT(m.id) AS messages
FROM users u
JOIN sessions s ON s.user_id = u.user_id
LEFT JOIN messages m ON m.session_id = s.session_id
GROUP BY s.session_id;
```

### Доступ до чату

| Спосіб | URL |
|---|---|
| Локально | `http://127.0.0.1:8000` |
| З пристроїв у тій самій WiFi | `http://<IP комп'ютера>:8000` |
| Публічно (ngrok) | `https://<id>.ngrok-free.app` |

`chat.html` роздається FastAPI на `GET /` і автоматично визначає адресу API (`window.location.origin`) — жодних змін коду при зміні середовища.

### Тестування через PowerShell

```powershell
# Створити сесію (з іменем — для розрізнення користувачів)
$s = Invoke-RestMethod -Uri "http://127.0.0.1:8000/sessions" `
  -Method POST -ContentType "application/json" `
  -Body '{"user_name": "Клієнт 1"}'

# Надіслати повідомлення
Invoke-RestMethod -Uri "http://127.0.0.1:8000/sessions/$($s.session_id)/chat" `
  -Method POST -ContentType "application/json" `
  -Body '{"message":"111222555"}'
```

---

## Стек технологій

| Компонент | Технологія |
|-----------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Агент | LangGraph + LangChain |
| RAG | FAISS + BM25Okapi + CrossEncoder |
| Embeds | `text-embedding-3-small` |
| Reranker | `BAAI/bge-reranker-base` |
| Веб-скрапінг | trafilatura + DuckDuckGo |
| База даних | SQLite (user_memory.py) |
| Excel | openpyxl |
| Email | smtplib + MIME |
| Config | pydantic-settings |
| Rate Limiting | slowapi |
| Observability | Langfuse |

---

## Безпека

Система розроблена як навчальний демо-проєкт з реалізованими заходами через OWASP Top 10:

| Захист | Реалізація |
|---|---|
| **Rate limiting** | `slowapi`: до 10 нових сесій/хв + 20 повідомлень/хв на IP |
| **Input validation** | Pydantic `Field(max_length=4000)` на повідомленнях |
| **Error masking** | 500-помилки повертають загальне повідомлення; деталі логуються на сервері |
| **CORS** | Дозволені тільки localhost + ngrok домен (не `*`) |
| **Security headers** | `X-Frame-Options`, `X-Content-Type-Options`, `X-XSS-Protection`, `Referrer-Policy` |
| **Session TTL** | Сесії автоматично закриваються через 4 години |
| **Off-topic guard** | Router + вбудована відмова для нетелеком-запитів |
| **Secrets** | Через `.env` + `pydantic.SecretStr`; `.gitignore` виключає `.env`, `*.db`, `data/customers/` |

> Це навчальний демо-проєкт. Для продакшного розгортання додатково потрібні: JWT-авторизація, шифрування SQLite, Redis для сесій.
