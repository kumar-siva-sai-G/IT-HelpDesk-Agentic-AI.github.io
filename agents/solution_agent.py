from __future__ import annotations
from typing import List, Dict
"""
agents/solution_agent.py
Solution Agent — CrewAI + ChromaDB playbook retrieval

Responsibilities:
- Retrieve matching fix playbooks from ChromaDB based on confirmed cause
- Groq LLM translates playbook into human-friendly steps for the user
- Also generates technical admin commands
- Sends solution back via Telegram
"""

import re
import json
from crewai import Agent, Task, Crew, Process
from crewai import LLM

from config.settings import GROQ_API_KEY, GROQ_MODEL
from tools.rag_tools import query_kb
from tools.telegram_tools import send_telegram_message


def _get_llm():
    return LLM(
        model=f"groq/{GROQ_MODEL}",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )


def _build_solution_agent() -> Agent:
    return Agent(
        role="IT Solutions Specialist",
        goal=(
            "Given a diagnosed IT problem and relevant fix playbooks, "
            "generate clear, actionable instructions for the user "
            "AND a separate technical command set for IT admins."
        ),
        backstory=(
            "You are an expert IT support specialist known for explaining "
            "technical solutions clearly. You write user instructions as if "
            "explaining to a non-technical student, and admin commands for an "
            "experienced sysadmin."
        ),
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def _build_solution_task(
    agent: Agent,
    ticket: dict,
    hypotheses: List[Dict],
    system_results: dict,
    playbook_context: str,
) -> Task:
    confirmed_cause = system_results.get("analysis", {}).get(
        "confirmed_cause",
        hypotheses[0]["hypothesis"] if hypotheses else "unknown",
    )
    severity = system_results.get("analysis", {}).get("severity", "MEDIUM")

    return Task(
        description=f"""
Generate a solution for this IT problem.

=== TICKET ===
Problem: {ticket.get('problem_summary', '')}
Resource: {ticket.get('resource', '')}
Priority: {ticket.get('priority', 'MEDIUM')}

=== CONFIRMED CAUSE ===
{confirmed_cause} (Severity: {severity})

=== SYSTEM CHECK RESULTS ===
Ping: {system_results.get('ping', {}).get('success', '?')}
Service status: {system_results.get('service', {}).get('status', '?')}
Recent log errors: {system_results.get('service', {}).get('recent_errors', [])}
Memory alert: {system_results.get('memory', {}).get('alert', False)}
Disk alert: {system_results.get('disk', {}).get('alert', False)}

=== RELEVANT PLAYBOOKS FROM KNOWLEDGE BASE ===
{playbook_context}

Return ONLY a JSON object with exactly these fields:
{{
  "user_reply": "A friendly, numbered list of steps the user can try themselves (max 5 steps, plain language)",
  "admin_commands": "Technical shell commands an admin should run to fully resolve this (as a numbered list)",
  "estimated_fix_time": "e.g. '5-10 minutes' or 'Requires admin intervention'",
  "follow_up_question": "One question to ask the user to confirm if issue is resolved"
}}

Return ONLY the JSON.
""",
        agent=agent,
        expected_output="JSON with user_reply, admin_commands, estimated_fix_time, follow_up_question.",
    )


def generate_solution(ticket: dict, hypotheses: List[Dict], system_results: dict) -> dict:
    """
    Generate solution using playbook RAG + Groq LLM.

    Returns:
        {
            "user_reply": str,
            "admin_commands": str,
            "estimated_fix_time": str,
            "follow_up_question": str
        }
    """
    # Step 1: Query KB for relevant fix playbooks
    confirmed_cause = system_results.get("analysis", {}).get(
        "confirmed_cause",
        hypotheses[0]["hypothesis"] if hypotheses else "unknown",
    )
    query = f"fix {confirmed_cause} {ticket.get('resource', '')} {ticket.get('category', '')}"
    playbook_results = query_kb(query, n_results=2, collection_name="playbooks")

    playbook_context = ""
    for i, doc in enumerate(playbook_results, 1):
        playbook_context += f"\n[Playbook {i}]\n{doc['document']}\n"

    if not playbook_context.strip():
        playbook_context = "No specific playbook found. Use general troubleshooting steps."

    # Step 2: CrewAI crew for solution generation
    agent = _build_solution_agent()
    task = _build_solution_task(agent, ticket, hypotheses, system_results, playbook_context)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    # Step 3: Parse result
    try:
        clean = re.sub(r"```(?:json)?|```", "", str(result)).strip()
        solution = json.loads(clean)
    except json.JSONDecodeError:
        solution = {
            "user_reply": "Please restart your device and try again. If the issue persists, contact IT admin.",
            "admin_commands": f"# Check {ticket.get('resource', 'resource')} manually",
            "estimated_fix_time": "Unknown",
            "follow_up_question": "Did that resolve your issue?",
        }

    # Step 4: Send solution to user via Telegram
    chat_id = ticket.get("chat_id", "")
    if chat_id:
        user_msg = (
            f"🔧 *Solution for your issue:*\n\n"
            f"{solution['user_reply']}\n\n"
            f"⏱️ Estimated fix time: {solution.get('estimated_fix_time', 'unknown')}\n\n"
            f"💬 {solution.get('follow_up_question', 'Did this resolve your issue?')}"
        )
        send_telegram_message(chat_id, user_msg, parse_mode="Markdown")

    return solution
