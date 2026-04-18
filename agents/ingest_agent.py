"""
agents/ingest_agent.py
Ticket Ingest Agent — CrewAI role-based agent.

Responsibilities:
- Receive raw user message (text from Telegram / web form)
- Use Groq LLM to extract: resource, category, urgency
- Detect priority from keywords
- Return normalized ticket JSON
"""

import re
import json
from datetime import datetime
from crewai import Agent, Task, Crew, Process
from crewai import LLM

from config.settings import GROQ_API_KEY, GROQ_MODEL, HIGH_PRIORITY_KEYWORDS
from tools.telegram_tools import send_telegram_message


# ── LLM setup ────────────────────────────────────────────────────────────────

def _get_llm():
    return LLM(
        model=f"groq/{GROQ_MODEL}",
        api_key=GROQ_API_KEY,
        temperature=0.1,
    )


# ── Agent definition ─────────────────────────────────────────────────────────

def _build_ingest_agent() -> Agent:
    return Agent(
        role="IT Ticket Analyst",
        goal=(
            "Parse raw IT support messages and extract structured information "
            "including: affected resource, problem category, urgency level, "
            "and a clean problem summary. Output ONLY valid JSON."
        ),
        backstory=(
            "You are an expert IT support analyst at a university lab. "
            "You have seen thousands of IT tickets and can instantly identify "
            "the core problem, the affected system, and how urgent it is."
        ),
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def _build_ingest_task(agent: Agent, raw_message: str) -> Task:
    return Task(
        description=f"""
Analyze this IT support message and extract structured information.

Message: "{raw_message}"

Extract and return ONLY a JSON object with these exact fields:
{{
  "problem_summary": "one clear sentence describing the problem",
  "resource": "specific server/device/service name (e.g. lab-server-101, wifi, printer-lab2)",
  "category": one of ["network", "service_down", "access_issue", "hardware", "software", "password_reset", "other"],
  "urgency_description": "brief reason for urgency if any",
  "raw_message": "{raw_message}"
}}

Return ONLY the JSON. No preamble, no explanation.
""",
        agent=agent,
        expected_output="A valid JSON object with problem_summary, resource, category, urgency_description, raw_message fields.",
    )


# ── Priority detection ────────────────────────────────────────────────────────

def _detect_priority(raw_message: str) -> str:
    msg_lower = raw_message.lower()
    for keyword in HIGH_PRIORITY_KEYWORDS:
        if keyword in msg_lower:
            return "HIGH"
    # Medium if mentions specific resource
    if any(word in msg_lower for word in ["server", "lab", "database", "cannot access", "down"]):
        return "MEDIUM"
    return "LOW"


# ── Public function ───────────────────────────────────────────────────────────

def ingest_ticket(raw_message: str, chat_id: str) -> dict:
    """
    Main entry point for ticket ingestion.

    Args:
        raw_message: Raw user text
        chat_id: Telegram chat ID or web session ID

    Returns:
        {
            "ticket": { structured ticket dict },
            "priority": "LOW" | "MEDIUM" | "HIGH"
        }
    """
    agent = _build_ingest_agent()
    task = _build_ingest_task(agent, raw_message)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    # Parse the JSON output from the LLM
    try:
        # Strip markdown fences if present
        clean = re.sub(r"```(?:json)?|```", "", str(result)).strip()
        ticket_data = json.loads(clean)
    except json.JSONDecodeError:
        # Fallback: build minimal ticket if LLM output is malformed
        ticket_data = {
            "problem_summary": raw_message[:200],
            "resource": "unknown",
            "category": "other",
            "urgency_description": "",
            "raw_message": raw_message,
        }

    priority = _detect_priority(raw_message)

    # Enrich with metadata
    ticket_data.update({
        "chat_id": chat_id,
        "priority": priority,
        "timestamp": datetime.utcnow().isoformat(),
    })

    # Acknowledge to user via Telegram
    ack_msg = (
        f"✅ Ticket received!\n"
        f"📋 Problem: {ticket_data['problem_summary']}\n"
        f"🖥️ Resource: {ticket_data['resource']}\n"
        f"🔴 Priority: {priority}\n\n"
        f"I'm diagnosing this now. Please wait..."
    )
    send_telegram_message(chat_id, ack_msg)

    return {"ticket": ticket_data, "priority": priority}
