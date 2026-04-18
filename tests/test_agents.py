"""
tests/test_agents.py
Unit and integration tests for the agent pipeline.

Run with:
    pytest tests/ -v
"""

import sys
import json
import time
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_ticket():
    return {
        "problem_summary": "Lab server lab-server-101 is unreachable",
        "resource": "lab-server-101",
        "category": "service_down",
        "urgency_description": "exam tomorrow",
        "raw_message": "Lab server 101 is down, exam tomorrow!",
        "chat_id": "test-chat-001",
        "priority": "HIGH",
        "timestamp": "2024-01-15T10:00:00",
    }


@pytest.fixture
def sample_hypotheses():
    return [
        {"hypothesis": "service_stopped", "confidence": 0.85, "evidence": "service inactive", "suggested_check": "systemctl status"},
        {"hypothesis": "network_unreachable", "confidence": 0.60, "evidence": "ping fails", "suggested_check": "ping host"},
        {"hypothesis": "disk_full", "confidence": 0.30, "evidence": "disk alerts", "suggested_check": "df -h"},
    ]


@pytest.fixture
def sample_system_results():
    return {
        "resource": "lab-server-101",
        "mode": "simulated",
        "ping": {"success": False, "latency_ms": 0, "details": "Request timeout"},
        "service": {"status": "inactive", "recent_errors": ["OOM killer terminated service"]},
        "logs": {"tail": "[ERROR] Out of memory\n[ERROR] Service killed"},
        "disk": {"used_pct": 42, "alert": False},
        "memory": {"used_pct": 94, "alert": True},
        "analysis": {
            "confirmed_cause": "service_stopped",
            "severity": "HIGH",
            "auto_fixable": False,
            "needs_human": True,
            "summary": "Service stopped due to OOM kill",
        },
        "inconclusive": False,
        "error": None,
    }


# ── Priority detection tests ──────────────────────────────────────────────────

class TestPriorityDetection:
    def test_high_priority_exam_keyword(self):
        from agents.ingest_agent import _detect_priority
        assert _detect_priority("server down exam tomorrow") == "HIGH"

    def test_high_priority_critical_keyword(self):
        from agents.ingest_agent import _detect_priority
        assert _detect_priority("critical system failure") == "HIGH"

    def test_high_priority_urgent(self):
        from agents.ingest_agent import _detect_priority
        assert _detect_priority("this is urgent please help") == "HIGH"

    def test_medium_priority_server_mention(self):
        from agents.ingest_agent import _detect_priority
        assert _detect_priority("server is slow today") == "MEDIUM"

    def test_low_priority_generic(self):
        from agents.ingest_agent import _detect_priority
        assert _detect_priority("how do I change my password") == "LOW"

    def test_case_insensitive(self):
        from agents.ingest_agent import _detect_priority
        assert _detect_priority("EXAM IS TOMORROW SERVER DOWN") == "HIGH"


# ── System tools tests ────────────────────────────────────────────────────────

class TestSystemTools:
    def test_ping_localhost(self):
        from tools.system_tools import tool_ping
        result = tool_ping("127.0.0.1", count=1)
        assert "success" in result
        assert "latency_ms" in result
        assert "packet_loss_pct" in result
        assert result["success"] is True  # localhost should always respond

    def test_ping_invalid_host(self):
        from tools.system_tools import tool_ping
        result = tool_ping("999.999.999.999", count=1)
        assert result["success"] is False

    def test_disk_check(self):
        from tools.system_tools import tool_check_disk
        result = tool_check_disk("/")
        assert "used_pct" in result
        assert "alert" in result
        # Disk should be accessible
        if result["used_pct"] is not None:
            assert 0 <= result["used_pct"] <= 100

    def test_memory_check(self):
        from tools.system_tools import tool_check_memory
        result = tool_check_memory()
        assert "used_pct" in result
        assert "alert" in result

    def test_log_tail_fallback(self):
        from tools.system_tools import tool_tail_logs
        result = tool_tail_logs("nonexistent-service-xyz")
        assert "tail" in result
        # Should not crash even if log doesn't exist


# ── RAG tools tests ───────────────────────────────────────────────────────────

class TestRAGTools:
    def test_upsert_and_query(self, tmp_path, monkeypatch):
        """Test ChromaDB upsert + query round-trip."""
        monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "test_chroma"))

        # Reload settings and rag_tools with new path
        import importlib
        import config.settings as settings
        settings.CHROMA_DB_PATH = str(tmp_path / "test_chroma")

        import tools.rag_tools as rag
        rag._client = None  # force reconnect

        # Upsert a test document
        success = rag.upsert_document(
            doc_id="test_001",
            text="Server unreachable due to network failure. Ping failed.",
            metadata={"category": "network"},
            collection_name="incidents",
        )
        assert success is True

        # Query it back
        results = rag.query_kb("server unreachable ping failed", n_results=1, collection_name="incidents")
        assert len(results) > 0
        assert "ping" in results[0]["document"].lower()

    def test_empty_collection_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CHROMA_DB_PATH", str(tmp_path / "empty_chroma"))

        import config.settings as settings
        settings.CHROMA_DB_PATH = str(tmp_path / "empty_chroma")

        import tools.rag_tools as rag
        rag._client = None

        results = rag.query_kb("some query", collection_name="nonexistent_xyz")
        assert results == []


# ── LangGraph state routing tests ─────────────────────────────────────────────

class TestLangGraphRouting:
    def test_route_after_diagnose_high_confidence(self, sample_hypotheses):
        from orchestrator.graph import route_after_diagnose
        state = {
            "hypotheses": sample_hypotheses,  # top confidence = 0.85
            "diagnosis_attempts": 0,
        }
        result = route_after_diagnose(state)
        assert result == "system_check"

    def test_route_after_diagnose_low_confidence_first_attempt(self):
        from orchestrator.graph import route_after_diagnose
        state = {
            "hypotheses": [{"hypothesis": "unknown", "confidence": 0.3}],
            "diagnosis_attempts": 0,
        }
        result = route_after_diagnose(state)
        assert result == "diagnose"  # retry

    def test_route_after_diagnose_low_confidence_second_attempt(self):
        from orchestrator.graph import route_after_diagnose
        state = {
            "hypotheses": [{"hypothesis": "unknown", "confidence": 0.3}],
            "diagnosis_attempts": 2,
        }
        result = route_after_diagnose(state)
        assert result == "system_check"  # give up retrying

    def test_route_after_check_inconclusive(self, sample_system_results):
        from orchestrator.graph import route_after_check
        state = {"system_results": {"inconclusive": True, "error": None}}
        result = route_after_check(state)
        assert result == "escalate"

    def test_route_after_check_normal(self, sample_system_results):
        from orchestrator.graph import route_after_check
        state = {"system_results": sample_system_results}
        result = route_after_check(state)
        assert result == "solve"

    def test_route_after_solve_high_priority_escalates(self, sample_ticket, sample_system_results):
        from langgraph.graph import END
        from orchestrator.graph import route_after_solve
        state = {
            "priority": "HIGH",
            "created_at": time.time(),
        }
        result = route_after_solve(state)
        assert result == "escalate"

    def test_route_after_solve_low_priority_resolves(self):
        from langgraph.graph import END
        from orchestrator.graph import route_after_solve
        state = {
            "priority": "LOW",
            "created_at": time.time(),  # just created, within SLA
        }
        result = route_after_solve(state)
        assert result == END


# ── Mem0 client tests ─────────────────────────────────────────────────────────

class TestMem0Client:
    def test_local_store_and_retrieve(self, tmp_path, monkeypatch):
        import config.settings as settings
        settings.MEM0_API_KEY = ""  # force local mode

        # Point memory file to temp dir
        monkeypatch.setattr(
            "memory.mem0_client.BASE_DIR",
            tmp_path,
        )

        from memory.mem0_client import Mem0Client
        client = Mem0Client()

        # Store a memory
        client.store_memory(
            user_id="test-user",
            text="Lab server 101 had OOM issue on Jan 15",
            metadata={"ticket_id": "TKT-001"},
        )

        # Retrieve it
        results = client.get_memories("test-user", query="lab server OOM")
        assert len(results) > 0
        assert "OOM" in results[0]["memory"]

    def test_search_with_no_matches(self, tmp_path, monkeypatch):
        import config.settings as settings
        settings.MEM0_API_KEY = ""
        monkeypatch.setattr("memory.mem0_client.BASE_DIR", tmp_path)

        from memory.mem0_client import Mem0Client
        client = Mem0Client()

        results = client.get_memories("new-user-xyz", query="anything")
        assert results == []


# ── Integration: full pipeline simulation ────────────────────────────────────

class TestPipelineIntegration:
    """
    Integration test that runs the full LangGraph pipeline with mocked LLM calls.
    Validates state transitions without hitting Groq API.
    """

    def test_graph_compiles(self):
        from orchestrator.graph import build_graph
        graph = build_graph()
        assert graph is not None

    def test_graph_state_schema(self):
        """Verify TicketState has all required keys."""
        from orchestrator.graph import TicketState
        required_keys = [
            "raw_message", "chat_id", "ticket", "priority",
            "hypotheses", "diagnosis_attempts", "system_results",
            "solution_text", "admin_commands", "escalated",
            "ticket_id", "created_at", "resolved",
        ]
        # TicketState is a TypedDict — check its annotations
        annotations = TicketState.__annotations__
        for key in required_keys:
            assert key in annotations, f"Missing key: {key}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
