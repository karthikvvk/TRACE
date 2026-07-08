# TRACE

> A local-first, offline browser extension agent that passively observes browser activity and proactively surfaces reminders — without waiting to be asked.

---

## What This Is

TRACE is not a chatbot. It watches your browser, infers what you're doing, and reminds you about pending tasks at the right moment — all on-device, no cloud.

---

## Project Structure

```
friday/
├── backend/                  # FastAPI local server
│   ├── main.py               # App entry point
│   ├── config.py             # Settings (pydantic-settings + .env)
│   ├── tools/                # MCP-shaped tool definitions
│   │   ├── base.py           # BaseTool ABC
│   │   ├── task_tool.py      # Task CRUD
│   │   ├── memory_tool.py    # Episodic log/query
│   │   ├── search_tool.py    # URL → semantic intent
│   │   └── calendar_tool.py  # Phase 5 stub
│   ├── memory/
│   │   ├── episodic.py       # Event log (SQLite)
│   │   ├── semantic.py       # User facts (SQLite)
│   │   └── working.py        # Active context (in-process)
│   ├── privacy/
│   │   └── anonymizer.py     # PII stripping (regex → Presidio)
│   ├── engine/
│   │   ├── intent.py         # Rule-based classifier
│   │   ├── router.py         # Tool registry + dispatch
│   │   └── notifier.py       # Proactive surfacing logic
│   ├── routes/
│   │   ├── activity.py       # POST /activity
│   │   ├── tasks.py          # CRUD /tasks
│   │   ├── memory.py         # GET /memory/events|facts
│   │   └── notify.py         # GET /notify/pending
│   └── db/
│       └── init_db.py        # SQLite schema + connection helper
├── extension/                # MV3 Chrome extension
│   ├── manifest.json
│   ├── background/
│   │   └── service-worker.js # Core loop, alarms, message routing
│   ├── content/
│   │   └── observer.js       # Page-level signals (no raw content)
│   └── utils/
│       ├── api.js            # Fetch helpers → backend
│       └── storage.js        # IndexedDB wrapper
├── tests/
│   └── test_tools.py         # pytest suite
├── requirements.txt
├── pytest.ini
└── .env                      # Local config (not committed)
```

---

## Quickstart

### 1. Backend

```bash
cd friday
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Start the server
uvicorn backend.main:app --reload --port 8000
```

Load the LLM from LMStudio

The server initialises the SQLite database on first run. No migrations needed.

### 2. Extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `extension/` directory

The extension will connect to `http://127.0.0.1:8000` automatically.

### 3. Tests

```bash
pytest tests/ -v
```

---

## API Reference

| Method | Path                   | Description                     |
| ------ | ---------------------- | ------------------------------- |
| GET    | `/`                    | Health check                    |
| GET    | `/tools`               | List all registered tools       |
| GET    | `/context`             | Current working memory snapshot |
| POST   | `/activity`            | Receive a browser event         |
| GET    | `/tasks`               | List tasks (`?status=pending`)  |
| POST   | `/tasks`               | Create a task                   |
| PATCH  | `/tasks/{id}`          | Update a task                   |
| DELETE | `/tasks/{id}`          | Delete a task                   |
| POST   | `/tasks/{id}/done`     | Mark done                       |
| POST   | `/tasks/{id}/snooze`   | Snooze                          |
| GET    | `/memory/events`       | Query episodic events           |
| GET    | `/memory/facts`        | All semantic user facts         |
| GET    | `/notify/pending`      | Items to surface right now      |
| POST   | `/notify/dismiss/{id}` | Dismiss a notification          |

---

## Build Phases

| Phase | Scope                                                         | Status            |
| ----- | ------------------------------------------------------------- | ----------------- |
| 1     | Memory + Chat Tasks — SQLite, task CRUD, proactive alarm loop | ✅ **This build** |
| 2     | Passive Observation — history, navigation, intent cross-ref   | 🔜                |
| 3     | Privacy Layer — Presidio PII stripping, pseudonymisation      | 🔜                |
| 4     | Local LLM — Ollama intent parsing + natural language tasks    | 🔜                |
| 5     | Calendar — Google Calendar OAuth + meeting briefings          | 🔜                |

---

## Privacy Model

- **Nothing leaves the device.** All data stays in SQLite on your machine.
- Raw URLs and text are anonymised _before_ any storage write.
- PII stripping uses regex at Phase 1 and upgrades to [Microsoft Presidio](https://github.com/microsoft/presidio) at Phase 3.
- Exact timestamps are never stored — only fuzzy time buckets (morning / afternoon / evening / night).
- Sensitive permissions (`scripting`, `history`) are requested with justification and only used for semantic abstraction, never for raw data export.

---

## Design Principles

> Friday is a **stateful context engine** with lightweight AI at the edges.
> The reliability comes from the memory architecture and observation loop — not the model.
> Keep the LLM at the edges: intent parsing and natural language output only.
> Routing, scheduling, matching, and storage are deterministic code.
