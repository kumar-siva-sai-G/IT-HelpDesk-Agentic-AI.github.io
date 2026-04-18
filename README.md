# 🤖 Agentic IT Helpdesk Copilot

> A fully autonomous IT support system powered by **CrewAI**, **AutoGen**, **LangGraph**, **Groq**, **ChromaDB**, and **Mem0** — built as a final-year Agentic AI project at SASTRA Deemed University.

---

## 🏗️ Architecture Overview

```
User (Telegram / Web) → Ticket Ingest (CrewAI)
                           ↓
                   LangGraph Orchestrator (state machine)
                    ↙        ↓         ↘
        Root-Cause       System         Solution
        Diagnoser        Checker        Agent
        (CrewAI+RAG)   (AutoGen+HITL)  (CrewAI+RAG)
                    ↘        ↓         ↙
                   Escalation Handler
                   (AutoGen + Mem0)
                           ↓
                   IT Staff Alert / Telegram
```

## 🧠 Agent Crew

| Agent | Framework | Role | Tools |
|-------|-----------|------|-------|
| Orchestrator | LangGraph | State machine, routing, SLA | SQLite, conditional edges |
| Ticket Ingest | CrewAI | Parse & structure tickets | Telegram API, FastAPI, Groq |
| Root-Cause Diagnoser | CrewAI | RAG-based hypothesis generation | ChromaDB, Groq LLM |
| System Checker | AutoGen | Real OS diagnostics | subprocess, log parser, HITL |
| Solution Agent | CrewAI | Playbook retrieval + user reply | ChromaDB, Groq LLM |
| Escalation Handler | AutoGen + Mem0 | Executive summary + alert | Mem0, Groq LLM |

## ⚡ Tech Stack (100% Free)

- **LLM**: Groq free tier — Llama-3-8B (sub-200ms)
- **Orchestration**: LangGraph (stateful graph)
- **Agent Roles**: CrewAI (role-based crew)
- **Conversational Agents**: AutoGen v0.4
- **Vector KB**: ChromaDB (local, no API)
- **Memory**: Mem0 (cross-session context)
- **Comms**: Telegram Bot API (free)
- **Backend**: FastAPI
- **DB**: SQLite (logging)

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Edit .env: add GROQ_API_KEY and TELEGRAM_BOT_TOKEN

# 3. Seed the knowledge base
python scripts/seed_kb.py

# 4. Start the system
python main.py
```

## 📁 Project Structure

```
it_helpdesk_copilot/
├── main.py                    # Entry point
├── config/
│   ├── settings.py            # All config (env vars, paths)
│   └── agents_config.yaml     # CrewAI agent definitions
├── orchestrator/
│   └── graph.py               # LangGraph state machine
├── agents/
│   ├── ingest_agent.py        # Ticket Ingest (CrewAI)
│   ├── diagnoser_agent.py     # Root-Cause (CrewAI + RAG)
│   ├── system_checker.py      # System Check (AutoGen)
│   ├── solution_agent.py      # Solution (CrewAI + RAG)
│   └── escalation_agent.py    # Escalation (AutoGen + Mem0)
├── tools/
│   ├── system_tools.py        # ping, systemctl, log tail
│   ├── rag_tools.py           # ChromaDB query/upsert
│   └── telegram_tools.py      # Telegram send/receive
├── memory/
│   └── mem0_client.py         # Mem0 session memory
├── kb/
│   ├── seed_data/             # IT runbooks (Markdown)
│   └── chroma_store/          # ChromaDB persistent store
├── api/
│   └── server.py              # FastAPI web form endpoint
├── ui/
│   └── dashboard.html         # Simple status dashboard
├── tests/
│   ├── test_agents.py
│   └── test_tools.py
├── scripts/
│   └── seed_kb.py             # Populate ChromaDB
├── .env.example
└── requirements.txt
```

## 🎯 Key Innovations

1. **Multi-framework orchestration** — 3 agent frameworks in one system
2. **RAG-powered diagnosis** — not rule-based; real vector retrieval
3. **HITL for dangerous ops** — AutoGen confirms before system changes
4. **Persistent memory** — Mem0 tracks cross-session context
5. **Real tool execution** — actual ping, systemctl, log parsing
6. **SLA enforcement** — LangGraph timers trigger escalation automatically

## 📊 Demo Flow

```
Student: "Lab server 101 is down, exam tomorrow!"
     ↓
System: Detects HIGH priority (keyword: "exam")
     ↓
Diagnoser: [network_down:0.82, service_stopped:0.71]
     ↓
Checker: ping → timeout | systemctl → inactive | logs → OOM kill
     ↓
Solution: "1. SSH as admin, 2. sudo systemctl restart lab-service, 3. check free memory..."
     ↓  (if unresolved in 10 min)
Escalation: "CRITICAL: lab-server-101 OOM + service down. User exam in 12h. Steps tried: restart failed. Action needed: RAM upgrade or swap increase."
```

---

*Built by Kumar Siva Sai Gourabathuni · SASTRA Deemed University · School of Computing*  
*Guided by Dr. Thaventhiran C.*
