"""
agents/diagnoser_agent.py
Root-Cause Diagnoser Agent — CrewAI + ChromaDB RAG

Responsibilities:
- Embed ticket description, query ChromaDB for similar past incidents
- Use Groq LLM to rank hypotheses with confidence scores
- Return: [{hypothesis, confidence, evidence, suggested_check}]
"""

from __future__ import annotations
from typing import List, Dict
import re
import json
from crewai import Agent, Task, Crew, Process
from crewai import LLM

from config.settings import GROQ_API_KEY, GROQ_MODEL
from tools.rag_tools import query_kb


def _get_llm():
    return LLM(
        model=f"groq/{GROQ_MODEL}",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )


def _build_diagnoser_agent() -> Agent:
    return Agent(
        role="IT Root Cause Analyst",
        goal=(
            "Analyze IT support tickets using past incident knowledge to identify "
            "the most likely root causes. Rank them by confidence. "
            "Be specific — name the exact system component likely to be failing."
        ),
        backstory=(
            "You are a senior IT systems engineer with 15 years of experience "
            "diagnosing network, server, and software failures in university lab environments. "
            "You think systematically and always check the most likely causes first."
        ),
        llm=_get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def _build_diagnose_task(agent: Agent, ticket: dict, kb_context: str) -> Task:
    return Task(
        description=f"""
You are diagnosing an IT support ticket. Use the past incident knowledge below to rank likely root causes.

=== TICKET ===
Problem: {ticket.get('problem_summary', '')}
Resource: {ticket.get('resource', '')}
Category: {ticket.get('category', '')}

=== SIMILAR PAST INCIDENTS (from Knowledge Base) ===
{kb_context}

Based on the ticket and past incidents, identify the top 3 most likely root causes.

Return ONLY a JSON array like this:
[
  {{
    "hypothesis": "exact name of the likely cause (e.g. 'service_stopped', 'network_unreachable', 'disk_full')",
    "confidence": 0.85,
    "evidence": "why this is likely based on the ticket and KB",
    "suggested_check": "what command or check would confirm this"
  }},
  ...
]

Order by confidence descending. Return ONLY the JSON array.
""",
        agent=agent,
        expected_output="A JSON array of 3 hypothesis objects with hypothesis, confidence, evidence, suggested_check fields.",
    )


def diagnose_ticket(ticket: dict) -> List[Dict]:
    """
    Run RAG-based root-cause diagnosis.

    Args:
        ticket: Structured ticket dict from ingest agent

    Returns:
        List of hypothesis dicts sorted by confidence descending
    """
    # Step 1: Query ChromaDB for similar past incidents
    query_text = f"{ticket.get('problem_summary', '')} {ticket.get('resource', '')} {ticket.get('category', '')}"
    kb_results = query_kb(query_text, n_results=3)

    # Format KB context for the LLM
    kb_context = ""
    for i, doc in enumerate(kb_results, 1):
        kb_context += f"\n[Incident {i}]\n{doc['document']}\nSimilarity: {doc['distance']:.2f}\n"

    if not kb_context.strip():
        kb_context = "No similar past incidents found in the knowledge base."

    # Step 2: Run CrewAI agent with the KB context
    agent = _build_diagnoser_agent()
    task = _build_diagnose_task(agent, ticket, kb_context)
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff()

    # Step 3: Parse JSON output
    try:
        clean = re.sub(r"```(?:json)?|```", "", str(result)).strip()
        hypotheses = json.loads(clean)
        # Ensure sorted by confidence
        hypotheses.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    except (json.JSONDecodeError, AttributeError):
        # Fallback hypothesis if LLM fails
        hypotheses = [
            {
                "hypothesis": "unknown",
                "confidence": 0.3,
                "evidence": "Could not parse diagnosis. Manual investigation needed.",
                "suggested_check": "Manual investigation required",
            }
        ]

    return hypotheses
