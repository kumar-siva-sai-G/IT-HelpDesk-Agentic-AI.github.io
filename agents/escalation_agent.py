from __future__ import annotations
from typing import List, Dict
"""
agents/escalation_agent.py
Escalation Handler — AutoGen + Mem0

Responsibilities:
- Use Mem0 to retrieve cross-session context about this user/resource
- AutoGen generates a structured escalation report
- Sends alert to IT admin Telegram group
- Stores full context in Mem0 for future reference
"""

import json
import time
import re
from loguru import logger

from config.settings import GROQ_API_KEY, GROQ_MODEL, TELEGRAM_ADMIN_CHAT_ID
from memory.mem0_client import Mem0Client
from tools.telegram_tools import send_telegram_message


def _build_autogen_config():
    return {
        "config_list": [
            {
                "model": GROQ_MODEL,
                "api_key": GROQ_API_KEY,
                "base_url": "https://api.groq.com/openai/v1",
                "api_type": "openai",
            }
        ],
        "temperature": 0.1,
    }


def handle_escalation(
    ticket: dict,
    hypotheses: List[Dict],
    system_results: dict,
    chat_id: str,
    ticket_id: str,
) -> str:
    """
    Handle ticket escalation:
    1. Retrieve past context from Mem0
    2. AutoGen generates executive summary
    3. Notify IT admin
    4. Store in Mem0 for future reference

    Returns:
        escalation_summary (str)
    """
    mem0 = Mem0Client()

    # Step 1: Get past context for this user + resource
    user_id = chat_id
    resource = ticket.get("resource", "unknown")
    past_context = mem0.get_memories(user_id=user_id, query=resource)

    # Step 2: AutoGen analysis and summary generation
    summary = _generate_escalation_summary(
        ticket=ticket,
        hypotheses=hypotheses,
        system_results=system_results,
        ticket_id=ticket_id,
        past_context=past_context,
    )

    # Step 3: Send to IT admin
    priority = ticket.get("priority", "MEDIUM")
    priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "⚪")

    admin_msg = (
        f"{priority_emoji} *ESCALATION ALERT*\n"
        f"Ticket ID: `{ticket_id}`\n"
        f"Priority: {priority}\n\n"
        f"{summary}\n\n"
        f"_Source chat: {chat_id}_"
    )

    if TELEGRAM_ADMIN_CHAT_ID:
        send_telegram_message(TELEGRAM_ADMIN_CHAT_ID, admin_msg, parse_mode="Markdown")
        logger.info(f"[ESCALATE] Sent admin alert for {ticket_id}")

    # Notify user
    send_telegram_message(
        chat_id,
        "⚠️ Your issue has been escalated to the IT team. "
        "An admin will contact you shortly with a solution.",
    )

    # Step 4: Store full context in Mem0
    mem0.store_memory(
        user_id=user_id,
        text=f"Ticket {ticket_id}: {ticket.get('problem_summary')} | "
             f"Resource: {resource} | Cause: {hypotheses[0]['hypothesis'] if hypotheses else 'unknown'} | "
             f"Escalated at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        metadata={"ticket_id": ticket_id, "resource": resource, "priority": priority},
    )

    return summary


def _generate_escalation_summary(
    ticket: dict,
    hypotheses: List[Dict],
    system_results: dict,
    ticket_id: str,
    past_context: list,
) -> str:
    """Use litellm (Groq) to generate a structured escalation report."""
    try:
        import litellm

        system_msg = """You are an IT escalation specialist.
            Write concise but complete escalation reports for IT managers.
            Structure: Problem → Confirmed Cause → Steps Already Taken → 
            Impact Assessment → Recommended Actions.
            Be direct and technical. Use bullet points.
            Max 250 words."""

        analysis = system_results.get("analysis", {})
        past_summary = ""
        if past_context:
            past_summary = "\n".join([m.get("memory", "") for m in past_context[:3]])

        prompt = f"""
Write an escalation report for this IT ticket:

Ticket ID: {ticket_id}
Problem: {ticket.get('problem_summary', '')}
Resource: {ticket.get('resource', '')}
Priority: {ticket.get('priority', 'MEDIUM')}

Confirmed Cause: {analysis.get('confirmed_cause', hypotheses[0]['hypothesis'] if hypotheses else 'unknown')}
Severity: {analysis.get('severity', 'HIGH')}
Auto-fixable: {analysis.get('auto_fixable', False)}

System Check Summary:
- Ping: {'OK' if system_results.get('ping', {}).get('success') else 'FAILED'}
- Service: {system_results.get('service', {}).get('status', 'unknown')}
- Memory alert: {system_results.get('memory', {}).get('alert', False)}
- Disk alert: {system_results.get('disk', {}).get('alert', False)}

Recent errors in logs: {system_results.get('service', {}).get('recent_errors', [])}

Past incidents for this user/resource:
{past_summary if past_summary else 'No prior incidents recorded.'}

Write the escalation report now.
"""
        response = litellm.completion(
            model=f"groq/{GROQ_MODEL}",
            api_key=GROQ_API_KEY,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        return response.choices[0].message.content

    except Exception as e:
        logger.error(f"[ESCALATE] Report generation error: {e}")
        # Fallback plain text summary
        top_cause = hypotheses[0]["hypothesis"] if hypotheses else "unknown"
        return (
            f"*Problem:* {ticket.get('problem_summary', 'unknown')}\n"
            f"*Resource:* {ticket.get('resource', 'unknown')}\n"
            f"*Likely Cause:* {top_cause} (conf: {hypotheses[0].get('confidence', 0):.0%})\n"
            f"*System Status:* Service {'active' if system_results.get('service', {}).get('status') == 'active' else 'inactive'}, "
            f"Ping {'OK' if system_results.get('ping', {}).get('success') else 'FAILED'}\n"
            f"*Action Required:* Manual investigation of {ticket.get('resource', 'resource')}"
        )
