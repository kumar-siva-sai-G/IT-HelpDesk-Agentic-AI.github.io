"""
orchestrator/graph.py
LangGraph state machine that routes tickets through the agent crew.

Flow:
  ingest → diagnose → system_check → solve → [done | escalate]
                        ↑___________|  (re-diagnose if low confidence)
"""

from __future__ import annotations
import time
import sqlite3
from typing import TypedDict, Literal, Optional, List, Dict
from langgraph.graph import StateGraph, END
from loguru import logger

from config.settings import SLA_MINUTES, SQLITE_DB_PATH
from agents.ingest_agent import ingest_ticket
from agents.diagnoser_agent import diagnose_ticket
from agents.system_checker import check_system
from agents.solution_agent import generate_solution
from agents.escalation_agent import handle_escalation


# ── State schema ─────────────────────────────────────────────────────────────

class TicketState(TypedDict):
    # Input
    raw_message: str
    chat_id: str

    # Enriched ticket
    ticket: Optional[dict]           # structured JSON from ingest
    priority: Literal["LOW", "MEDIUM", "HIGH"]

    # Diagnosis
    hypotheses: List[Dict]           # [{hypothesis, confidence, evidence}]
    diagnosis_attempts: int

    # System check
    system_results: Optional[dict]   # {ping, service_status, log_tail, ...}

    # Solution
    solution_text: Optional[str]     # human-friendly response
    admin_commands: Optional[str]    # technical commands for admin

    # Escalation
    escalated: bool
    escalation_summary: Optional[str]

    # Metadata
    ticket_id: str
    created_at: float
    resolved: bool
    error: Optional[str]


# ── Node functions ────────────────────────────────────────────────────────────

def node_ingest(state: TicketState) -> dict:
    logger.info(f"[INGEST] Processing ticket {state['ticket_id']}")
    result = ingest_ticket(state["raw_message"], state["chat_id"])
    _log_ticket(state["ticket_id"], "INGESTED", result)
    return {
        "ticket": result["ticket"],
        "priority": result["priority"],
    }


def node_diagnose(state: TicketState) -> dict:
    logger.info(f"[DIAGNOSE] Attempt {state['diagnosis_attempts'] + 1}")
    hypotheses = diagnose_ticket(state["ticket"])
    _log_ticket(state["ticket_id"], "DIAGNOSED", {"hypotheses": hypotheses})
    return {
        "hypotheses": hypotheses,
        "diagnosis_attempts": state["diagnosis_attempts"] + 1,
    }


def node_system_check(state: TicketState) -> dict:
    logger.info(f"[SYSTEM_CHECK] Top hypothesis: {state['hypotheses'][0]['hypothesis']}")
    resource = state["ticket"].get("resource", "unknown")
    results = check_system(resource, state["hypotheses"])
    _log_ticket(state["ticket_id"], "CHECKED", results)
    return {"system_results": results}


def node_solve(state: TicketState) -> dict:
    logger.info(f"[SOLVE] Generating solution for {state['ticket_id']}")
    solution = generate_solution(
        ticket=state["ticket"],
        hypotheses=state["hypotheses"],
        system_results=state["system_results"],
    )
    _log_ticket(state["ticket_id"], "SOLVED", solution)
    return {
        "solution_text": solution["user_reply"],
        "admin_commands": solution["admin_commands"],
        "resolved": True,
    }


def node_escalate(state: TicketState) -> dict:
    logger.warning(f"[ESCALATE] Ticket {state['ticket_id']} escalated")
    summary = handle_escalation(
        ticket=state["ticket"],
        hypotheses=state["hypotheses"],
        system_results=state["system_results"],
        chat_id=state["chat_id"],
        ticket_id=state["ticket_id"],
    )
    _log_ticket(state["ticket_id"], "ESCALATED", {"summary": summary})
    return {
        "escalated": True,
        "escalation_summary": summary,
    }


# ── Conditional routing ───────────────────────────────────────────────────────

def route_after_diagnose(state: TicketState) -> str:
    """
    After diagnosis:
    - If top confidence < 0.5 and we've tried < 2 times → retry diagnosis
    - Otherwise → system_check
    """
    top_conf = state["hypotheses"][0]["confidence"] if state["hypotheses"] else 0
    if top_conf < 0.5 and state["diagnosis_attempts"] < 2:
        logger.info(f"[ROUTE] Low confidence ({top_conf:.2f}), re-diagnosing")
        return "diagnose"
    return "system_check"


def route_after_solve(state: TicketState) -> str:
    """
    After solution generation:
    - If SLA exceeded → escalate
    - Otherwise → END
    """
    elapsed = (time.time() - state["created_at"]) / 60
    if elapsed > SLA_MINUTES or state["priority"] == "HIGH":
        logger.warning(f"[ROUTE] SLA={SLA_MINUTES}min, elapsed={elapsed:.1f}min → escalate")
        return "escalate"
    return END


def route_after_check(state: TicketState) -> str:
    """
    After system check:
    - If checks were inconclusive → escalate immediately
    - Otherwise → solve
    """
    results = state.get("system_results", {})
    if results.get("error") or results.get("inconclusive"):
        return "escalate"
    return "solve"


# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(TicketState)

    graph.add_node("ingest", node_ingest)
    graph.add_node("diagnose", node_diagnose)
    graph.add_node("system_check", node_system_check)
    graph.add_node("solve", node_solve)
    graph.add_node("escalate", node_escalate)

    graph.set_entry_point("ingest")

    graph.add_edge("ingest", "diagnose")

    graph.add_conditional_edges(
        "diagnose",
        route_after_diagnose,
        {"diagnose": "diagnose", "system_check": "system_check"},
    )

    graph.add_conditional_edges(
        "system_check",
        route_after_check,
        {"solve": "solve", "escalate": "escalate"},
    )

    graph.add_conditional_edges(
        "solve",
        route_after_solve,
        {"escalate": "escalate", END: END},
    )

    graph.add_edge("escalate", END)

    return graph.compile()


# ── SQLite logging helper ─────────────────────────────────────────────────────

def init_db():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ticket_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticket_id TEXT,
            stage TEXT,
            data TEXT,
            timestamp REAL
        )
    """)
    conn.commit()
    conn.close()


def _log_ticket(ticket_id: str, stage: str, data: dict):
    import json
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.execute(
        "INSERT INTO ticket_log (ticket_id, stage, data, timestamp) VALUES (?, ?, ?, ?)",
        (ticket_id, stage, json.dumps(data), time.time()),
    )
    conn.commit()
    conn.close()
