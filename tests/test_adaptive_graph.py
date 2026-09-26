"""Regression and unit tests for AVENIQ adaptive graph (Phase 3, 4, 5)."""

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from src.core.config import MoEConfig, SecretStr
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder
from src.aveniq.policy import RulePolicy, LayaPolicy


class TestAdaptiveGraph:
    """Verify adaptive routing: deterministic fast path vs System-2 fallback."""

    @pytest.fixture
    def mock_builder(self, tmp_path):
        trace_file = str(tmp_path / "test_traces.jsonl")
        config = MoEConfig(groq_api_key=SecretStr("mock_key"))
        with patch("src.graph.builder.LLMFactory.create_provider"):
            builder = MoEGraphBuilder(config=config, trace_file=trace_file)

        # Mock the query_agent inside plan_executor
        async def mock_agent_call(expert: str, prompt: str, **kwargs):
            return f"Mock answer from {expert} on {prompt}"

        builder.plan_executor._query_agent_fn = mock_agent_call

        # Spy on OrchestratorAgent
        builder.agents["orchestrator"].execute = MagicMock(
            side_effect=lambda state: {
                **state,
                "generated_code": "async def orchestrate(): pass",
                "reasoning_steps": [{"step": "orchestrator", "detail": "system2 code generated"}],
                "selected_experts": ["technical"],
            }
        )

        # Spy on CodeExecutionAgent
        builder.agents["code_executor"].aexecute = AsyncMock(
            side_effect=lambda state: {
                **state,
                "final_answer": "System-2 sandbox execution result",
                "selected_experts": ["technical"],
                "expert_responses": {"technical": "System-2 response"},
                "code_execution_iterations": 1,
                "code_execution_error": "",
            }
        )

        return builder

    @pytest.mark.asyncio
    async def test_single_expert_bypasses_orchestrator(self, mock_builder):
        """Prove that a single-expert request can bypass OrchestratorAgent LLM entirely."""
        graph = mock_builder.build()
        state = create_initial_state("Write a Python function to compute Fibonacci numbers")

        result = await graph.ainvoke(state)

        # CRITICAL ASSERTION: OrchestratorAgent.execute was NEVER called!
        mock_builder.agents["orchestrator"].execute.assert_not_called()
        mock_builder.agents["code_executor"].aexecute.assert_not_called()

        assert result["selected_experts"] == ["technical"]
        assert "technical" in result["expert_responses"]
        assert "Mock answer from technical" in result["final_answer"]
        assert result["metadata"]["system2_invoked"] is False
        assert result["metadata"]["execution_path"] == "single_expert"
        assert result["code_execution_iterations"] == 1
        assert result["code_execution_error"] == ""

    @pytest.mark.asyncio
    async def test_direct_query_bypasses_orchestrator(self, mock_builder):
        """Prove that a direct greeting request bypasses all agents and orchestrator."""
        graph = mock_builder.build()
        state = create_initial_state("Hello")

        result = await graph.ainvoke(state)

        mock_builder.agents["orchestrator"].execute.assert_not_called()
        mock_builder.agents["code_executor"].aexecute.assert_not_called()

        assert result["selected_experts"] == []
        assert "hello" in result["final_answer"].lower()
        assert result["metadata"]["system2_invoked"] is False
        assert result["metadata"]["execution_path"] == "direct"

    @pytest.mark.asyncio
    async def test_complex_query_invokes_system2(self, mock_builder):
        """Prove that a complex workflow request correctly invokes System-2 OrchestratorAgent."""
        graph = mock_builder.build()
        state = create_initial_state("Plan and execute a multi-step workflow pipeline across all agents")

        result = await graph.ainvoke(state)

        # CRITICAL ASSERTION: OrchestratorAgent was invoked for System-2
        mock_builder.agents["orchestrator"].execute.assert_called_once()
        mock_builder.agents["code_executor"].aexecute.assert_called_once()

        assert result["metadata"]["system2_invoked"] is True
        assert result["metadata"]["execution_path"] == "system2"
        assert "System-2 sandbox execution result" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_laya_shadow_mode_records_predictions(self, mock_builder):
        """Verify that Laya shadow mode populates predictions without altering execution path."""
        graph = mock_builder.build()
        state = create_initial_state("Write a Python script to sort items")

        result = await graph.ainvoke(state)

        # Laya shadow predictions should be recorded in metadata
        assert "laya_shadow" in result["metadata"]
        laya_meta = result["metadata"]["laya_shadow"]
        assert "confidence" in laya_meta
        assert "probabilities" in laya_meta
        assert "backend" in laya_meta
        # Still executed single_expert directly
        assert result["metadata"]["system2_invoked"] is False
        assert result["metadata"]["execution_path"] == "single_expert"

    @pytest.mark.asyncio
    async def test_outcome_trace_persisted(self, mock_builder, tmp_path):
        """Verify that OutcomeTrace is built and persisted to disk."""
        graph = mock_builder.build()
        state = create_initial_state("Write a Python function")

        result = await graph.ainvoke(state)

        assert "outcome_trace" in result["metadata"]
        trace = result["metadata"]["outcome_trace"]
        assert trace["execution_path"] == "single_expert"
        assert trace["system2_invoked"] is False
        assert trace["success"] is True
        assert "latency_seconds" in trace

        # Verify persisted traces in trace store
        saved = mock_builder.trace_store.load_all()
        assert len(saved) >= 1
        assert saved[-1].execution_path == "single_expert"
