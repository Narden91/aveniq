"""Builder for constructing the AVENIQ adaptive compute LangGraph workflow."""

import time
from typing import Any, Dict, Optional
from langgraph.graph import END, StateGraph

from ..agents.orchestrator import CodeExecutionAgent, OrchestratorAgent
from ..agents.registry import registry
from ..core.config import MoEConfig
from ..core.sandbox import SandboxPolicy
from ..core.state import MoEState
from ..llm.providers import LLMFactory
from ..aveniq.policy import LayaPolicy, PolicyDecision, PolicyEngine, RulePolicy
from ..aveniq.plans import ExecutionPlan, PlanCompiler, PlanExecutorAdapter
from ..aveniq.tracing import OutcomeTrace, OutcomeTraceStore


class MoEGraphBuilder:
    """Builder for constructing the AVENIQ adaptive compute LangGraph workflow.

    Workflow topology:
    policy -> {direct | deterministic_plan | system2} -> executor -> outcome -> END
    """

    def __init__(
        self,
        config: MoEConfig,
        policy_engine: Optional[PolicyEngine] = None,
        trace_file: Optional[str] = None,
    ):
        """Initialize graph builder with config, policy engine, and execution adapters."""
        self.config = config
        self.policy_engine = policy_engine or LayaPolicy(
            shadow_mode=True,
            fallback_policy=RulePolicy(),
        )
        self.plan_compiler = PlanCompiler()
        self.plan_executor = PlanExecutorAdapter()
        self.trace_store = OutcomeTraceStore(trace_file=trace_file)
        self.agents: Dict[str, Any] = {}
        self._initialize_agents()

    def _initialize_agents(self) -> None:
        provider_type = self.config.get_provider_type()
        api_key = self.config.get_api_key(provider_type)

        orchestrator_llm = LLMFactory.create_provider(
            provider_type,
            api_key,
            self.config.orchestrator_config,
        )

        self.agents["orchestrator"] = OrchestratorAgent(
            orchestrator_llm,
            available_experts=list(registry.types),
            candidate_count=self.config.orchestrator_candidate_count,
            script_few_shot_count=self.config.orchestrator_script_few_shot_count,
            atom_few_shot_count=self.config.orchestrator_atom_few_shot_count,
            enable_atom_few_shot_retrieval=self.config.enable_atom_few_shot_retrieval,
            enable_metadata_selection_bias=self.config.enable_metadata_selection_bias,
            enable_compression=self.config.enable_registry_compression,
            registry_db_path=self.config.registry_db_path,
        )

        self.agents["code_executor"] = CodeExecutionAgent(
            timeout_seconds=self.config.request_timeout,
            isolate_process=self.config.sandbox_isolate_process,
            sandbox_policy=SandboxPolicy(
                max_code_chars=self.config.sandbox_max_code_chars,
                max_ast_nodes=self.config.sandbox_max_ast_nodes,
                max_statements=self.config.sandbox_max_statements,
                max_query_calls=self.config.sandbox_max_query_calls,
            ),
            enable_compression=self.config.enable_registry_compression,
            registry_db_path=self.config.registry_db_path,
        )

    # ------------------------------------------------------------------
    # Graph Nodes
    # ------------------------------------------------------------------

    async def _policy_node(self, state: MoEState) -> Dict[str, Any]:
        """Evaluate request using PolicyEngine to determine compute tier."""
        query = state.get("query", "")
        metadata = dict(state.get("metadata", {}) or {})
        t0 = metadata.get("pipeline_start_time") or time.time()
        metadata["pipeline_start_time"] = t0

        decision = await self.policy_engine.aevaluate(
            query,
            context={"metadata": metadata},
        )

        execution_path = decision.execution_class
        metadata["policy_decision"] = decision.model_dump()
        metadata["execution_path"] = execution_path
        if "laya_shadow" in decision.raw_predictions:
            metadata["laya_shadow"] = decision.raw_predictions["laya_shadow"]

        reasoning_step = {
            "step": "policy",
            "decision": execution_path,
            "primary_expert": decision.primary_expert,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning or "",
        }

        return {
            "metadata": metadata,
            "reasoning_steps": [reasoning_step],
        }

    def _route_policy(self, state: MoEState) -> str:
        """Route from policy to direct, deterministic_plan, or system2."""
        meta = state.get("metadata", {}) or {}
        path = meta.get("execution_path", "system2")
        if path == "direct":
            return "direct"
        elif path in {"single_expert", "parallel_experts"}:
            return "deterministic_plan"
        return "system2"

    async def _direct_node(self, state: MoEState) -> Dict[str, Any]:
        """Compile a direct response plan."""
        query = state.get("query", "")
        meta = dict(state.get("metadata", {}) or {})
        decision_dict = meta.get("policy_decision") or {}
        decision = (
            PolicyDecision.model_validate(decision_dict)
            if decision_dict
            else PolicyDecision(execution_class="direct")
        )
        plan = self.plan_compiler.compile(decision, query)
        meta["compiled_plan"] = plan.model_dump()
        return {"metadata": meta}

    async def _deterministic_plan_node(self, state: MoEState) -> Dict[str, Any]:
        """Compile a single-expert or parallel-expert deterministic execution plan."""
        query = state.get("query", "")
        meta = dict(state.get("metadata", {}) or {})
        decision_dict = meta.get("policy_decision") or {}
        decision = (
            PolicyDecision.model_validate(decision_dict)
            if decision_dict
            else PolicyDecision(
                execution_class="single_expert",
                primary_expert="general",
            )
        )
        plan = self.plan_compiler.compile(decision, query)
        meta["compiled_plan"] = plan.model_dump()
        return {
            "metadata": meta,
            "execution_plan": {
                "plan_id": plan.plan_id,
                "execution_class": plan.execution_class,
                "deterministic": True,
            },
        }

    async def _system2_node(self, state: MoEState) -> Dict[str, Any]:
        """Delegate to generative code OrchestratorAgent (System-2 fallback)."""
        # OrchestratorAgent.execute is synchronous
        res = self.agents["orchestrator"].execute(state)
        # Ensure metadata records system2 invocation
        meta = dict(res.get("metadata", {}) or state.get("metadata", {}) or {})
        meta["system2_invoked"] = True
        res["metadata"] = meta
        return res

    async def _executor_node(self, state: MoEState) -> Dict[str, Any]:
        """Execute either the deterministic plan or the System-2 generated Python code."""
        meta = dict(state.get("metadata", {}) or {})
        path = meta.get("execution_path", "system2")

        if path in {"direct", "single_expert", "parallel_experts"}:
            compiled_dict = meta.get("compiled_plan")
            if compiled_dict:
                plan = ExecutionPlan.model_validate(compiled_dict)
            else:
                decision = PolicyDecision(
                    execution_class=path,  # type: ignore
                    primary_expert=meta.get("policy_decision", {}).get("primary_expert", "general"),
                )
                plan = self.plan_compiler.compile(decision, state.get("query", ""))

            # Execute plan deterministically without orchestrator LLM
            plan_res = await self.plan_executor.execute(
                plan,
                state.get("query", ""),
                context=state,
            )

            meta["system2_invoked"] = False
            if plan_res.get("verification_result"):
                meta["verification_result"] = plan_res["verification_result"]

            return {
                "final_answer": plan_res["final_answer"],
                "selected_experts": plan_res["selected_experts"],
                "expert_responses": plan_res["expert_responses"],
                "execution_plan": plan_res["execution_plan"],
                "token_usage": plan_res.get("token_usage", {}),
                "trace_dna": plan_res.get("trace_dna", []),
                "code_execution_iterations": 1,
                "code_execution_error": "",
                "metadata": meta,
            }

        # Otherwise: System-2 path executes generated code in sandbox
        meta["system2_invoked"] = True
        res = await self.agents["code_executor"].aexecute(state)
        out_meta = dict(res.get("metadata", {}) or {})
        out_meta["system2_invoked"] = True
        res["metadata"] = out_meta
        return res

    def _should_retry_code(self, state: MoEState) -> str:
        """Preserve existing retry behavior for System-2 failures."""
        meta = state.get("metadata", {}) or {}
        path = meta.get("execution_path", "system2")
        error = state.get("code_execution_error")
        iterations = state.get("code_execution_iterations", 0)

        max_retries = self.config.max_retries
        if path == "system2" and error and iterations < max_retries:
            return "system2"
        return "outcome"

    async def _outcome_node(self, state: MoEState) -> Dict[str, Any]:
        """Capture and persist the OutcomeTrace for the request."""
        meta = dict(state.get("metadata", {}) or {})
        t0 = meta.get("pipeline_start_time") or time.time()
        elapsed = time.time() - t0

        token_usage = state.get("token_usage", {}) or {}
        system2_invoked = bool(meta.get("system2_invoked", False))

        # Determine verification score if available
        verification_score: Optional[float] = None
        ver_res = meta.get("verification_result")
        if isinstance(ver_res, dict):
            feedback = str(ver_res.get("feedback") or "")
            import re
            m = re.search(r"SCORE:\s*([0-9.]+)", feedback, re.IGNORECASE)
            if m:
                try:
                    verification_score = float(m.group(1))
                except ValueError:
                    pass
            elif ver_res.get("passed"):
                verification_score = 1.0

        trace = OutcomeTrace(
            policy_decision=meta.get("policy_decision", {}),
            execution_path=meta.get("execution_path", "unknown"),
            generated_plan_or_script=(
                state.get("generated_code")
                if system2_invoked
                else str((meta.get("compiled_plan") or {}).get("plan_id") or "")
            ),
            experts_called=state.get("selected_experts", []),
            system2_invoked=system2_invoked,
            latency_seconds=round(elapsed, 4),
            input_tokens=int(token_usage.get("total_input_tokens", 0) or 0),
            output_tokens=int(token_usage.get("total_output_tokens", 0) or 0),
            total_tokens=int(token_usage.get("total_tokens", 0) or 0),
            estimated_cost_usd=float(token_usage.get("estimated_cost_usd", 0.0) or 0.0),
            success=not bool(state.get("code_execution_error")),
            verification_score=verification_score,
            error=state.get("code_execution_error") or None,
            query=state.get("query", ""),
            final_answer_snippet=str(state.get("final_answer", ""))[:200],
            shadow_predictions=meta.get("laya_shadow", {}),
        )

        self.trace_store.append(trace)
        meta["outcome_trace"] = trace.model_dump()

        return {"metadata": meta}

    # ------------------------------------------------------------------
    # Graph Assembly
    # ------------------------------------------------------------------

    def build(self):
        """Build and compile the adaptive compute LangGraph."""
        workflow = StateGraph(MoEState)

        # Nodes
        workflow.add_node("policy", self._policy_node)
        workflow.add_node("direct", self._direct_node)
        workflow.add_node("deterministic_plan", self._deterministic_plan_node)
        workflow.add_node("system2", self._system2_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("outcome", self._outcome_node)

        # Entry point: policy
        workflow.set_entry_point("policy")

        # Policy conditional branches
        workflow.add_conditional_edges(
            "policy",
            self._route_policy,
            {
                "direct": "direct",
                "deterministic_plan": "deterministic_plan",
                "system2": "system2",
            },
        )

        # Convergence to executor
        workflow.add_edge("direct", "executor")
        workflow.add_edge("deterministic_plan", "executor")
        workflow.add_edge("system2", "executor")

        # Conditional retry or completion to outcome
        workflow.add_conditional_edges(
            "executor",
            self._should_retry_code,
            {
                "system2": "system2",
                "outcome": "outcome",
            },
        )

        workflow.add_edge("outcome", END)

        return workflow.compile()
