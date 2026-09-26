"""Unit tests for AVENIQ typed ExecutionPlan IR, compiler, and executor adapter (Phase 2)."""

import pytest
from pydantic import ValidationError

from src.aveniq.plans import (
    ExecutionPlan,
    PlanCompiler,
    PlanExecutorAdapter,
    PlanOp,
    PlanStep,
    compile_decision_to_plan,
)
from src.aveniq.policy import PolicyDecision


class TestExecutionPlanIR:
    """Test validation and construction of typed execution IR."""

    def test_valid_plan_step_ops(self):
        for op in ["CALL", "PARALLEL", "SEQUENCE", "VERIFY", "RETURN"]:
            step = PlanStep(op=op)  # type: ignore
            assert step.op == op

    def test_invalid_plan_step_op(self):
        with pytest.raises(ValidationError):
            PlanStep(op="WHILE_LOOP")  # type: ignore

    def test_execution_plan_flags(self):
        direct_plan = ExecutionPlan(execution_class="direct")
        assert direct_plan.is_deterministic() is True
        assert direct_plan.is_system2() is False

        s2_plan = ExecutionPlan(execution_class="system2")
        assert s2_plan.is_deterministic() is False
        assert s2_plan.is_system2() is True


class TestPlanCompiler:
    """Test compilation of PolicyDecisions into typed ExecutionPlans."""

    def setup_method(self):
        self.compiler = PlanCompiler()

    def test_compile_direct_response(self):
        decision = PolicyDecision(
            execution_class="direct",
            needs_verification=False,
            max_agent_calls=1,
            confidence=1.0,
        )
        plan = self.compiler.compile(decision, "Hello")
        assert plan.execution_class == "direct"
        assert len(plan.steps) == 1
        assert plan.steps[0].op == "RETURN"
        assert "Hello" in (plan.steps[0].value or "")

    def test_compile_single_expert(self):
        decision = PolicyDecision(
            execution_class="single_expert",
            primary_expert="technical",
            needs_verification=False,
            max_agent_calls=1,
            confidence=0.9,
        )
        plan = self.compiler.compile(decision, "Write a quicksort in Python")
        assert plan.execution_class == "single_expert"
        assert len(plan.steps) == 1
        seq = plan.steps[0]
        assert seq.op == "SEQUENCE"
        assert seq.steps is not None
        assert len(seq.steps) == 2  # CALL, RETURN
        assert seq.steps[0].op == "CALL"
        assert seq.steps[0].expert == "technical"
        assert seq.steps[1].op == "RETURN"

    def test_compile_single_expert_with_verification(self):
        decision = PolicyDecision(
            execution_class="single_expert",
            primary_expert="technical",
            needs_verification=True,
            max_agent_calls=1,
            confidence=0.85,
        )
        plan = self.compiler.compile(decision, "Write a regex and verify")
        assert plan.execution_class == "single_expert"
        seq = plan.steps[0]
        assert seq.steps is not None
        assert len(seq.steps) == 3  # CALL, VERIFY, RETURN
        assert seq.steps[0].op == "CALL"
        assert seq.steps[1].op == "VERIFY"
        assert seq.steps[2].op == "RETURN"

    def test_compile_parallel_experts(self):
        decision = PolicyDecision(
            execution_class="parallel_experts",
            primary_expert="analytical",
            parallel_experts=["technical", "creative"],
            needs_verification=False,
            max_agent_calls=2,
            confidence=0.8,
        )
        plan = self.compiler.compile(decision, "Compare databases and write a story")
        assert plan.execution_class == "parallel_experts"
        seq = plan.steps[0]
        assert seq.steps is not None
        par_step = seq.steps[0]
        assert par_step.op == "PARALLEL"
        assert len(par_step.steps or []) == 2
        assert {s.expert for s in (par_step.steps or [])} == {"technical", "creative"}

    def test_compile_system2(self):
        decision = PolicyDecision(
            execution_class="system2",
            needs_verification=True,
            max_agent_calls=5,
            confidence=0.95,
        )
        plan = self.compiler.compile(decision, "Complex orchestration query")
        assert plan.is_system2() is True
        assert plan.is_deterministic() is False


class TestPlanExecutorAdapter:
    """Test execution of ExecutionPlans via PlanExecutorAdapter."""

    @pytest.mark.asyncio
    async def test_execute_direct_plan(self):
        compiler = PlanCompiler()
        decision = PolicyDecision(
            execution_class="direct",
            needs_verification=False,
            max_agent_calls=1,
            confidence=1.0,
        )
        plan = compiler.compile(decision, "ping")
        executor = PlanExecutorAdapter()
        result = await executor.execute(plan, "ping")
        assert "operational" in result["final_answer"].lower() or "pong" in result["final_answer"].lower()
        assert result["selected_experts"] == []

    @pytest.mark.asyncio
    async def test_execute_single_expert_plan(self):
        # Mock query agent callable
        async def mock_query_agent(expert: str, prompt: str, **kwargs):
            return f"Answer from {expert} on '{prompt}'"

        compiler = PlanCompiler()
        decision = PolicyDecision(
            execution_class="single_expert",
            primary_expert="technical",
            needs_verification=False,
            max_agent_calls=1,
            confidence=0.9,
        )
        plan = compiler.compile(decision, "How do lists work?")
        executor = PlanExecutorAdapter(query_agent_fn=mock_query_agent)
        result = await executor.execute(plan, "How do lists work?")

        assert "technical" in result["selected_experts"]
        assert "technical" in result["expert_responses"]
        assert "Answer from technical" in result["final_answer"]
        assert result["metadata"]["deterministic"] is True

    @pytest.mark.asyncio
    async def test_execute_parallel_experts_plan(self):
        called = []

        async def mock_query_agent(expert: str, prompt: str, **kwargs):
            called.append(expert)
            return f"Response from {expert}"

        compiler = PlanCompiler()
        decision = PolicyDecision(
            execution_class="parallel_experts",
            primary_expert="analytical",
            parallel_experts=["technical", "creative"],
            needs_verification=False,
            max_agent_calls=2,
            confidence=0.85,
        )
        plan = compiler.compile(decision, "Query")
        executor = PlanExecutorAdapter(query_agent_fn=mock_query_agent)
        result = await executor.execute(plan, "Query")

        assert set(called) == {"technical", "creative"}
        assert set(result["selected_experts"]) == {"technical", "creative"}
        assert "Response from technical" in result["final_answer"]
        assert "Response from creative" in result["final_answer"]

    @pytest.mark.asyncio
    async def test_execute_with_verification(self):
        async def mock_query_agent(expert: str, prompt: str, **kwargs):
            if expert == "analytical":
                return "SCORE: 0.95 REASON: accurate"
            return f"Result from {expert}"

        compiler = PlanCompiler()
        decision = PolicyDecision(
            execution_class="single_expert",
            primary_expert="technical",
            needs_verification=True,
            max_agent_calls=1,
            confidence=0.9,
        )
        plan = compiler.compile(decision, "Generate prime sieve")
        executor = PlanExecutorAdapter(query_agent_fn=mock_query_agent)
        result = await executor.execute(plan, "Generate prime sieve")

        assert result["verification_result"] is not None
        assert result["verification_result"]["passed"] is True
        assert "SCORE: 0.95" in result["verification_result"]["feedback"]
