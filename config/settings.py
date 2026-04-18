"""
config/settings.py
Central configuration — reads .env, exposes typed settings.
"""

from __future__ import annotations
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Base ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

# ── LLM ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Telegram ─────────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_CHAT_ID: str = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")

# ── Memory ───────────────────────────────────────────────────────────────────
MEM0_API_KEY: str = os.getenv("MEM0_API_KEY", "")

# ── Storage ──────────────────────────────────────────────────────────────────
CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "kb" / "chroma_store"))
SQLITE_DB_PATH: str = os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "tickets.db"))

# ── SLA ──────────────────────────────────────────────────────────────────────
SLA_MINUTES: int = int(os.getenv("SLA_MINUTES", "10"))
HIGH_PRIORITY_KEYWORDS: List[str] = [
    kw.strip().lower()
    for kw in os.getenv(
        "HIGH_PRIORITY_KEYWORDS",
        "exam,production,critical,urgent,down,emergency"
    ).split(",")
]

# ── Safety ───────────────────────────────────────────────────────────────────
ALLOW_REAL_SYSTEM_COMMANDS: bool = os.getenv(
    "ALLOW_REAL_SYSTEM_COMMANDS", "false"
).lower() == "true"

# ── Embedding model (local, free) ────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

# ── Agent names ──────────────────────────────────────────────────────────────
AGENT_NAMES = {
    "ingest": "TicketIngestAgent",
    "diagnoser": "RootCauseDiagnoser",
    "checker": "SystemCheckerAgent",
    "solution": "SolutionAgent",
    "escalation": "EscalationHandlerAgent",
}
