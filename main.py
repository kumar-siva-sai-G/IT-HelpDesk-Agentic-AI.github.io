"""
main.py
Entry point for the Agentic IT Helpdesk Copilot.

Starts:
  1. FastAPI server (for web form tickets)
  2. Telegram bot (for chat-based tickets)
  3. LangGraph pipeline (invoked per ticket)

Usage:
    python main.py                  # full system
    python main.py --web-only       # FastAPI only (no Telegram)
    python main.py --demo           # run a demo ticket without Telegram/API
"""

import sys
import os
import uuid
import time
import asyncio
import argparse

# Fix Windows console encoding for emojis/Unicode
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import TELEGRAM_BOT_TOKEN
from orchestrator.graph import build_graph, init_db, TicketState

console = Console(force_terminal=True)


# ── Banner ────────────────────────────────────────────────────────────────────

def print_banner():
    console.print(Panel.fit(
        "[bold cyan]🤖 Agentic IT Helpdesk Copilot[/bold cyan]\n"
        "[dim]CrewAI · AutoGen · LangGraph · Groq · ChromaDB · Mem0[/dim]\n"
        "[dim]SASTRA Deemed University — School of Computing[/dim]",
        border_style="cyan",
    ))


# ── Demo mode ─────────────────────────────────────────────────────────────────

def run_demo():
    """Run a demo ticket through the full pipeline and print results."""
    console.print("\n[bold yellow]▶ DEMO MODE[/bold yellow]")
    console.print("[dim]Processing a sample ticket through the full agent pipeline...[/dim]\n")

    demo_messages = [
        "Lab server lab-server-101 is completely unreachable, my exam submission is tomorrow!",
        "WiFi is not working in room 301, multiple students affected.",
        "I can't SSH into compute-node-05, getting connection refused.",
    ]

    graph = build_graph()
    init_db()

    for i, message in enumerate(demo_messages, 1):
        ticket_id = f"DEMO-{i:03d}"
        console.print(f"\n[bold]Ticket {ticket_id}:[/bold] {message}")
        console.print("[dim]─" * 60 + "[/dim]")

        initial_state: TicketState = {
            "raw_message": message,
            "chat_id": f"demo-chat-{i}",
            "ticket": None,
            "priority": "LOW",
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
            final_state = graph.invoke(initial_state)

            # Print results table
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Key", style="bold cyan", width=22)
            table.add_column("Value")

            priority = final_state.get("priority", "?")
            priority_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}.get(priority, "white")

            table.add_row("Priority", f"[{priority_color}]{priority}[/{priority_color}]")

            hyps = final_state.get("hypotheses", [])
            if hyps:
                top = hyps[0]
                table.add_row(
                    "Top Hypothesis",
                    f"{top['hypothesis']} (conf: {top['confidence']:.0%})"
                )

            sys_check = final_state.get("system_results", {})
            if sys_check:
                ping_ok = sys_check.get("ping", {}).get("success", False)
                svc_status = sys_check.get("service", {}).get("status", "?")
                table.add_row("System Check", f"Ping={'✅' if ping_ok else '❌'} | Service={svc_status}")

            solution = final_state.get("solution_text", "")
            if solution:
                table.add_row("Solution", solution[:120] + "..." if len(solution) > 120 else solution)

            if final_state.get("escalated"):
                table.add_row("Status", "[red]ESCALATED to IT Admin[/red]")
            else:
                table.add_row("Status", "[green]RESOLVED[/green]")

            console.print(table)

        except Exception as e:
            console.print(f"[red]Pipeline error: {e}[/red]")

    console.print("\n[bold green]✅ Demo complete![/bold green]")


# ── Telegram mode ─────────────────────────────────────────────────────────────

async def run_telegram_mode():
    """Start Telegram bot + handle incoming tickets via LangGraph."""
    from tools.telegram_tools import start_telegram_bot

    graph = build_graph()
    init_db()

    async def on_message(chat_id: str, text: str):
        ticket_id = f"TG-{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"[MAIN] Processing Telegram ticket {ticket_id}")

        initial_state: TicketState = {
            "raw_message": text,
            "chat_id": chat_id,
            "ticket": None,
            "priority": "LOW",
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

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: graph.invoke(initial_state))

    await start_telegram_bot(on_message)


# ── Web mode ──────────────────────────────────────────────────────────────────

def run_web_mode():
    """Start FastAPI server only."""
    import uvicorn
    from orchestrator.graph import init_db
    init_db()
    console.print("[bold green]🌐 Starting FastAPI server at http://localhost:8000[/bold green]")
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)


# ── Full mode (Telegram + Web) ────────────────────────────────────────────────

async def run_full_mode():
    """Start FastAPI + Telegram bot concurrently."""
    import uvicorn

    config = uvicorn.Config("api.server:app", host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)

    console.print("[bold green]🌐 FastAPI:  http://localhost:8000[/bold green]")
    console.print("[bold cyan]📱 Telegram: bot polling started[/bold cyan]")

    await asyncio.gather(
        server.serve(),
        run_telegram_mode(),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Agentic IT Helpdesk Copilot")
    parser.add_argument("--demo", action="store_true", help="Run demo tickets")
    parser.add_argument("--web-only", action="store_true", help="FastAPI only, no Telegram")
    args = parser.parse_args()

    print_banner()

    if args.demo:
        run_demo()
    elif args.web_only or not TELEGRAM_BOT_TOKEN:
        if not TELEGRAM_BOT_TOKEN:
            console.print("[yellow]⚠️  TELEGRAM_BOT_TOKEN not set — starting web-only mode[/yellow]")
        run_web_mode()
    else:
        asyncio.run(run_full_mode())


if __name__ == "__main__":
    main()
