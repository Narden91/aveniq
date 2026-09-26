"""Compiler adapter converting PolicyDecision into a typed ExecutionPlan."""

from typing import Any, Dict, List, Optional
from ..policy.decision import PolicyDecision
from .ir import ExecutionPlan, PlanStep


class PlanCompiler:
    """Compiles deterministic PolicyDecisions into typed ExecutionPlans.

    System-2 decisions are preserved as non-deterministic markers for
    the legacy generative Python orchestrator.
    """

    DIRECT_RESPONSES = {
        "hi": "Hello! How can I help you today?",
        "hello": "Hello! How can I help you today?",
        "hey": "Hello! How can I help you today?",
        "ping": "Pong! System is operational.",
        "who are you": "I am AVENIQ, an adaptive compute agent runtime.",
    }

    def compile(
        self,
        decision: PolicyDecision,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Compile a PolicyDecision and user query into an ExecutionPlan.

        Parameters
        ----------
        decision:
            The PolicyDecision produced by a PolicyEngine.
        query:
            The original user request string.
        context:
            Optional context dictionary.
        """
        exec_class = decision.execution_class

        if exec_class == "direct":
            return self._compile_direct(decision, query)
        elif exec_class == "single_expert":
            return self._compile_single_expert(decision, query)
        elif exec_class == "parallel_experts":
            return self._compile_parallel_experts(decision, query)
        elif exec_class == "system2":
            return self._compile_system2(decision, query)
        else:
            raise ValueError(f"Unknown execution class: {exec_class}")

    def _compile_direct(self, decision: PolicyDecision, query: str) -> ExecutionPlan:
        lowered = (query or "").strip().lower()
        direct_msg = self.DIRECT_RESPONSES.get(lowered, "Hello! How can I assist you?")
        step = PlanStep(
            op="RETURN",
            value=direct_msg,
            metadata={"source": "direct_policy"},
        )
        return ExecutionPlan(
            execution_class="direct",
            steps=[step],
            metadata={"decision": decision.model_dump()},
        )

    def _compile_single_expert(self, decision: PolicyDecision, query: str) -> ExecutionPlan:
        expert = decision.primary_expert or "general"
        steps: List[PlanStep] = []

        # 1. Primary CALL
        call_step = PlanStep(
            op="CALL",
            expert=expert,
            prompt=query,
            output_key=f"{expert}_response",
        )
        steps.append(call_step)

        # 2. Optional VERIFY
        if decision.needs_verification:
            verify_step = PlanStep(
                op="VERIFY",
                verifier_expert="analytical",
                target_step_output=f"{expert}_response",
                verification_criteria="Ensure factual accuracy and completeness of answer.",
            )
            steps.append(verify_step)

        # 3. RETURN
        return_step = PlanStep(
            op="RETURN",
            output_key=f"{expert}_response",
        )
        steps.append(return_step)

        # Wrap in SEQUENCE
        sequence_step = PlanStep(
            op="SEQUENCE",
            steps=steps,
        )

        return ExecutionPlan(
            execution_class="single_expert",
            steps=[sequence_step],
            metadata={"expert": expert, "decision": decision.model_dump()},
        )

    def _compile_parallel_experts(self, decision: PolicyDecision, query: str) -> ExecutionPlan:
        experts = decision.parallel_experts
        if not experts:
            experts = [decision.primary_expert or "general"]

        # Child CALL steps for PARALLEL
        parallel_calls = [
            PlanStep(
                op="CALL",
                expert=exp,
                prompt=query,
                output_key=f"{exp}_response",
            )
            for exp in experts
        ]

        parallel_step = PlanStep(
            op="PARALLEL",
            steps=parallel_calls,
        )

        steps: List[PlanStep] = [parallel_step]

        # Optional verification
        if decision.needs_verification:
            verify_step = PlanStep(
                op="VERIFY",
                verifier_expert="analytical",
                verification_criteria="Verify consistency across parallel outputs.",
            )
            steps.append(verify_step)

        # RETURN step (will combine outputs)
        return_step = PlanStep(
            op="RETURN",
            metadata={"combine_keys": [f"{exp}_response" for exp in experts]},
        )
        steps.append(return_step)

        sequence_step = PlanStep(
            op="SEQUENCE",
            steps=steps,
        )

        return ExecutionPlan(
            execution_class="parallel_experts",
            steps=[sequence_step],
            metadata={"experts": experts, "decision": decision.model_dump()},
        )

    def _compile_system2(self, decision: PolicyDecision, query: str) -> ExecutionPlan:
        """Mark as System-2 for delegation to OrchestratorAgent."""
        return ExecutionPlan(
            execution_class="system2",
            steps=[],
            metadata={"delegated_to": "orchestrator", "decision": decision.model_dump()},
        )


def compile_decision_to_plan(
    decision: PolicyDecision,
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> ExecutionPlan:
    """Convenience function to compile a PolicyDecision into an ExecutionPlan."""
    return PlanCompiler().compile(decision, query, context=context)
