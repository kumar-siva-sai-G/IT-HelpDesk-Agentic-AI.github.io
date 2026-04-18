"""
api/server.py
FastAPI backend — REST endpoints for ticket submission and status.

Endpoints:
  POST /ticket        — Submit a new ticket (web form / API)
  GET  /ticket/{id}   — Get ticket status
  GET  /tickets       — List all tickets
  GET  /health        — Health check
  GET  /kb/stats      — ChromaDB collection stats
"""

import uuid
import time
import sqlite3
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from loguru import logger

from config.settings import SQLITE_DB_PATH
from orchestrator.graph import build_graph, init_db, TicketState
from tools.rag_tools import get_collection_stats


app = FastAPI(
    title="Agentic IT Helpdesk Copilot",
    description="Multi-agent IT support system powered by CrewAI, AutoGen, LangGraph",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pre-build the graph at startup
_graph = None


@app.on_event("startup")
async def startup():
    global _graph
    init_db()
    _graph = build_graph()
    logger.info("[API] FastAPI server started. Graph compiled.")


# ── Request/Response models ───────────────────────────────────────────────────

class TicketRequest(BaseModel):
    message: str
    session_id: str = ""
    priority: str = ""


class TicketResponse(BaseModel):
    ticket_id: str
    status: str
    message: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ticket", response_model=TicketResponse)
async def submit_ticket(req: TicketRequest):
    """Submit a new IT support ticket."""
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    chat_id = req.session_id or f"web-{ticket_id}"

    logger.info(f"[API] New ticket {ticket_id}: {req.message[:80]}")

    # Run the LangGraph pipeline (async-compatible)
    import asyncio
    loop = asyncio.get_event_loop()

    initial_state: TicketState = {
        "raw_message": req.message,
        "chat_id": chat_id,
        "ticket": None,
        "priority": req.priority if req.priority in ("LOW", "MEDIUM", "HIGH") else "LOW",
        "hypotheses": [],
        "diagnosis_attempts": 0,
        "system_results": None,
        "solution_text": None,
        "admin_commands": None,
        "escalated": False,
        "escalation_summary": None,
        "ticket_id": ticket_id,
        "created_at": time.time(),
        "resolved": False,
        "error": None,
    }

    try:
        final_state = await loop.run_in_executor(
            None, lambda: _graph.invoke(initial_state)
        )

        solution = final_state.get("solution_text", "No solution generated")
        # Normalize: LLM may return a list of steps instead of a string
        if isinstance(solution, list):
            solution = "\n".join(f"{i+1}. {s}" for i, s in enumerate(solution))
        elif not isinstance(solution, str):
            solution = str(solution) if solution else "Your ticket has been processed."

        status = "escalated" if final_state.get("escalated") else "resolved"
        escalation_note = ""
        if final_state.get("escalated"):
            esc_summary = final_state.get("escalation_summary", "")
            escalation_note = f"\n\n⚠️ This ticket has been escalated to IT admin.\n{esc_summary}"

        return TicketResponse(
            ticket_id=ticket_id,
            status=status,
            message=(solution or "Your ticket has been processed.") + escalation_note,
        )

    except Exception as e:
        logger.error(f"[API] Pipeline error for {ticket_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.get("/ticket/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Get full ticket processing log from SQLite."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    rows = conn.execute(
        "SELECT stage, data, timestamp FROM ticket_log WHERE ticket_id = ? ORDER BY timestamp",
        (ticket_id,),
    ).fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail="Ticket not found")

    return {
        "ticket_id": ticket_id,
        "stages": [
            {
                "stage": row[0],
                "data": json.loads(row[1]),
                "timestamp": row[2],
            }
            for row in rows
        ],
    }


@app.get("/tickets")
async def list_tickets(limit: int = 20):
    """List recent tickets with their latest status."""
    conn = sqlite3.connect(SQLITE_DB_PATH)
    rows = conn.execute("""
        SELECT t.ticket_id, t.stage as last_stage, t.timestamp as last_update
        FROM ticket_log t
        INNER JOIN (
            SELECT ticket_id, MAX(id) as max_id
            FROM ticket_log
            GROUP BY ticket_id
        ) latest ON t.ticket_id = latest.ticket_id AND t.id = latest.max_id
        ORDER BY t.timestamp DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()

    return {
        "tickets": [
            {"ticket_id": r[0], "last_stage": r[1], "last_update": r[2]}
            for r in rows
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    kb_stats = get_collection_stats()
    return {
        "status": "ok",
        "kb_stats": kb_stats,
        "graph_compiled": _graph is not None,
    }


@app.get("/kb/stats")
async def kb_stats():
    """ChromaDB knowledge base statistics."""
    return get_collection_stats()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the web form for ticket submission."""
    from config.settings import BASE_DIR
    html_path = BASE_DIR / "ui" / "dashboard.html"
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())
