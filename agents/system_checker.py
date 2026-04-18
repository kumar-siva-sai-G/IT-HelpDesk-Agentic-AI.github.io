"""
agents/system_checker.py
System Checker Agent — AutoGen v0.4 with Human-In-The-Loop

Responsibilities:
- Execute real system diagnostic tools (ping, systemctl, log tail, disk check)
- Ask for admin confirmation before running any destructive operations
- Return structured JSON of check results

AutoGen is used here because:
1. It supports tool-calling natively with function executors
2. Its HITL (human proxy) pattern is ideal for confirming dangerous ops
3. Conversational multi-turn is natural for "run check → analyze → run next check"
"""

from __future__ import annotations
from typing import List, Dict
import json
import subprocess
import time
import re
from loguru import logger

from config.settings import ALLOW_REAL_SYSTEM_COMMANDS
from tools.system_tools import (
    tool_ping,
    tool_check_service,
    tool_tail_logs,
    tool_check_disk,
    tool_check_memory,
)


# ── AutoGen configuration ─────────────────────────────────────────────────────

def _build_autogen_config():
    """Build AutoGen LLM config using Groq."""
    from config.settings import GROQ_API_KEY, GROQ_MODEL
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


# ── Simulated checks (safe mode) ──────────────────────────────────────────────

def _simulate_checks(resource: str, hypotheses: List[Dict]) -> dict:
    """
    Return simulated check results for demo/testing when
    ALLOW_REAL_SYSTEM_COMMANDS=false.
    """
    logger.warning("[SYSTEM_CHECK] Running in SIMULATION mode")
    top_hypothesis = hypotheses[0]["hypothesis"] if hypotheses else "unknown"

    simulated = {
        "resource": resource,
        "mode": "simulated",
        "timestamp": time.time(),
        "ping": {
            "success": "network" not in top_hypothesis,
            "latency_ms": 0 if "network" in top_hypothesis else 12.4,
            "details": "Request timeout" if "network" in top_hypothesis else "4 packets transmitted, 4 received",
        },
        "service": {
            "status": "inactive" if "service" in top_hypothesis else "active",
            "since": "2024-01-15 14:32:01" if "service" not in top_hypothesis else None,
            "recent_errors": ["OOM killer terminated service"] if "service" in top_hypothesis else [],
        },
        "logs": {
            "tail": (
                "[ERROR] Out of memory: Kill process 1234 (lab-service)\n"
                "[ERROR] oom_kill_process: memory allocation failed\n"
                "[WARN]  Service exited with code 137"
            ) if "service" in top_hypothesis else (
                "[INFO] Service running normally\n"
                "[INFO] Last restart: 7 days ago"
            ),
        },
        "disk": {
            "used_pct": 87 if "disk" in top_hypothesis else 42,
            "free_gb": 8.2,
            "alert": "disk" in top_hypothesis,
        },
        "memory": {
            "used_pct": 94 if "service" in top_hypothesis else 61,
            "free_mb": 312 if "service" in top_hypothesis else 2048,
            "alert": "service" in top_hypothesis,
        },
        "inconclusive": False,
        "error": None,
    }
    return simulated


# ── Real system checks ────────────────────────────────────────────────────────

def _run_real_checks(resource: str, hypotheses: List[Dict]) -> dict:
    """
    Execute actual system checks using tools.
    AutoGen's HITL pattern handles confirmation for dangerous ops.
    """
    logger.info(f"[SYSTEM_CHECK] Running REAL checks on: {resource}")
    results = {
        "resource": resource,
        "mode": "live",
        "timestamp": time.time(),
        "inconclusive": False,
        "error": None,
    }

    try:
        results["ping"] = tool_ping(resource)
    except Exception as e:
        results["ping"] = {"success": False, "error": str(e)}

    try:
        # Extract service name from resource (e.g. "lab-server-101" → "lab-service")
        service_name = resource.replace("-101", "").replace("-", "") + "d"
        results["service"] = tool_check_service(service_name)
    except Exception as e:
        results["service"] = {"error": str(e)}

    try:
        results["logs"] = tool_tail_logs(resource)
    except Exception as e:
        results["logs"] = {"error": str(e)}

    try:
        results["disk"] = tool_check_disk("/")
    except Exception as e:
        results["disk"] = {"error": str(e)}

    try:
        results["memory"] = tool_check_memory()
    except Exception as e:
        results["memory"] = {"error": str(e)}

    return results


# ── AutoGen HITL wrapper ──────────────────────────────────────────────────────

def _autogen_analyze(resource: str, raw_results: dict, hypotheses: List[Dict]) -> dict:
    """
    Analyze raw check results and produce a structured summary.
    Uses litellm for Groq API calls (AutoGen v0.4+ API is not backward-compatible).
    """
    try:
        import litellm

        from config.settings import GROQ_API_KEY, GROQ_MODEL

        system_msg = """You are a senior IT systems analyst.
            Given raw system check results and initial hypotheses,
            analyze what is actually wrong. Be specific and concise.
            Return a JSON object with fields:
            confirmed_cause, severity (LOW/MEDIUM/HIGH/CRITICAL),
            auto_fixable (true/false), fix_commands (list of shell commands if fixable),
            needs_human (true/false), summary (one sentence).
            Return ONLY JSON."""

        prompt = f"""
Analyze these system check results for resource: {resource}

Raw Results: {json.dumps(raw_results, indent=2)}
Top Hypotheses: {json.dumps(hypotheses[:2], indent=2)}

Provide your analysis as JSON.
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

        last_msg = response.choices[0].message.content
        clean = re.sub(r"```(?:json)?|```", "", last_msg).strip()
        analysis = json.loads(clean)
        raw_results["analysis"] = analysis
        return raw_results

    except Exception as e:
        logger.warning(f"[SYSTEM_CHECK] Analysis failed: {e}. Using raw results.")
        raw_results["analysis"] = {
            "confirmed_cause": "manual_investigation_needed",
            "severity": "HIGH",
            "auto_fixable": False,
            "needs_human": True,
            "summary": "Analysis unavailable. Manual review required.",
        }
        return raw_results


# ── Public function ───────────────────────────────────────────────────────────

def check_system(resource: str, hypotheses: List[Dict]) -> dict:
    """
    Run system checks for the given resource based on hypotheses.

    Args:
        resource: Server/device name (e.g. "lab-server-101")
        hypotheses: Ranked list from diagnoser

    Returns:
        Structured dict of all check results + AutoGen analysis
    """
    if ALLOW_REAL_SYSTEM_COMMANDS:
        raw_results = _run_real_checks(resource, hypotheses)
    else:
        raw_results = _simulate_checks(resource, hypotheses)

    # Run AutoGen analysis over the raw results
    final_results = _autogen_analyze(resource, raw_results, hypotheses)
    return final_results
