"""Executor adapter executing typed ExecutionPlans without generative orchestrator."""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional
from src.utils.tracing import TraceEvent, TraceKind, get_tracer
from .ir import ExecutionPlan, PlanStep


class PlanExecutorAdapter:
    """Executes a typed ExecutionPlan deterministically.

    Dispatches expert calls directly through query_agent (or a supplied callable)
    without invoking the generative LLM OrchestratorAgent.
    """

    def __init__(
        self,
        query_agent_fn: Optional[Callable[..., Any]] = None,
    ):
        self._query_agent_fn = query_agent_fn

    async def _get_query_agent(self) -> Callable[..., Any]:
        if self._query_agent_fn is not None:
            return self._query_agent_fn
        from src.core.agents import query_agent
        return query_agent

    async def execute(
        self,
        plan: ExecutionPlan,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute all steps in an ExecutionPlan and assemble the resulting state dict.

        Parameters
        ----------
        plan:
            The compiled ExecutionPlan.
        query:
            The user query string.
        context:
            Optional variable store / prior state.
        """
        env: Dict[str, Any] = dict(context or {})
        env["query"] = query
        expert_responses: Dict[str, str] = {}
        selected_experts: List[str] = []
        trace_dna: List[Dict[str, Any]] = []
        verification_details: Dict[str, Any] = {}
        final_answer = ""

        tracer = get_tracer()

        for step in plan.steps:
            res = await self._execute_step(step, env, tracer)
            if res.get("final_answer"):
                final_answer = res["final_answer"]
            if res.get("expert_responses"):
                expert_responses.update(res["expert_responses"])
            if res.get("selected_experts"):
                for exp in res["selected_experts"]:
                    if exp not in selected_experts:
                        selected_experts.append(exp)
            if res.get("trace_dna"):
                trace_dna.extend(res["trace_dna"])
            if res.get("verification"):
                verification_details.update(res["verification"])

        # If final answer was not set by a RETURN step, resolve from expert_responses
        if not final_answer:
            if expert_responses:
                # Use primary expert or first response
                first_key = next(iter(expert_responses))
                final_answer = expert_responses[first_key]
            else:
                final_answer = "(no response generated)"

        # Get token tracker summary if available
        token_usage: Dict[str, Any] = {}
        try:
            from src.utils.metrics import get_token_tracker
            tracker = get_token_tracker()
            token_usage = tracker.summary()
        except Exception:
            token_usage = {
                "total_tokens": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "estimated_cost_usd": 0.0,
            }

        return {
            "final_answer": final_answer,
            "expert_responses": expert_responses,
            "selected_experts": selected_experts,
            "execution_plan": {
                "plan_id": plan.plan_id,
                "execution_class": plan.execution_class,
                "steps_count": len(plan.steps),
                "experts_used": selected_experts,
            },
            "token_usage": token_usage,
            "trace_dna": trace_dna,
            "verification_result": verification_details if verification_details else None,
            "metadata": {
                "plan_id": plan.plan_id,
                "execution_class": plan.execution_class,
                "deterministic": True,
            },
        }

    async def _execute_step(
        self,
        step: PlanStep,
        env: Dict[str, Any],
        tracer: Any,
    ) -> Dict[str, Any]:
        """Recursively execute a single PlanStep."""
        q_fn = await self._get_query_agent()

        if step.op == "RETURN":
            if step.value:
                return {"final_answer": step.value}
            if step.output_key and step.output_key in env:
                return {"final_answer": str(env[step.output_key])}
            combine_keys = step.metadata.get("combine_keys") or []
            if combine_keys:
                combined = []
                for k in combine_keys:
                    if k in env:
                        combined.append(str(env[k]))
                return {"final_answer": "\n\n".join(combined)}
            return {"final_answer": str(env.get("last_output", ""))}

        elif step.op == "CALL":
            expert = step.expert or "general"
            prompt = step.prompt or env.get("query", "")
            t0 = time.time()
            tracer.emit_sync(TraceEvent(
                kind=TraceKind.EXPERT_CALL_START.value,
                agent=expert,
                data={"prompt": prompt[:200]},
            ))
            try:
                res = await q_fn(expert, prompt)
                duration_ms = (time.time() - t0) * 1000
                res_text = getattr(res, "text", getattr(res, "result", str(res)))
                output_key = step.output_key or f"{expert}_response"
                env[output_key] = res_text
                env["last_output"] = res_text
                tracer.emit_sync(TraceEvent(
                    kind=TraceKind.EXPERT_CALL_END.value,
                    agent=expert,
                    data={"duration_ms": duration_ms, "result_snippet": res_text[:200]},
                ))
                trace_entry = {
                    "expert": expert,
                    "duration_ms": duration_ms,
                    "status": "success",
                }
                return {
                    "expert_responses": {expert: res_text},
                    "selected_experts": [expert],
                    "trace_dna": [trace_entry],
                }
            except Exception as e:
                duration_ms = (time.time() - t0) * 1000
                tracer.emit_sync(TraceEvent(
                    kind=TraceKind.SANDBOX_ERROR.value,
                    agent=expert,
                    data={"error": str(e), "duration_ms": duration_ms},
                ))
                trace_entry = {
                    "expert": expert,
                    "duration_ms": duration_ms,
                    "error": str(e),
                    "status": "failed",
                }
                raise

        elif step.op == "PARALLEL":
            sub_steps = step.steps or []
            results = await asyncio.gather(
                *[self._execute_step(sub, env, tracer) for sub in sub_steps]
            )
            combined_responses: Dict[str, str] = {}
            combined_experts: List[str] = []
            combined_trace: List[Dict[str, Any]] = []
            for r in results:
                if r.get("expert_responses"):
                    combined_responses.update(r["expert_responses"])
                if r.get("selected_experts"):
                    for exp in r["selected_experts"]:
                        if exp not in combined_experts:
                            combined_experts.append(exp)
                if r.get("trace_dna"):
                    combined_trace.extend(r["trace_dna"])
            return {
                "expert_responses": combined_responses,
                "selected_experts": combined_experts,
                "trace_dna": combined_trace,
            }

        elif step.op == "SEQUENCE":
            sub_steps = step.steps or []
            combined_responses: Dict[str, str] = {}
            combined_experts: List[str] = []
            combined_trace: List[Dict[str, Any]] = []
            final_answer = ""
            verification: Dict[str, Any] = {}

            for sub in sub_steps:
                r = await self._execute_step(sub, env, tracer)
                if r.get("final_answer"):
                    final_answer = r["final_answer"]
                if r.get("expert_responses"):
                    combined_responses.update(r["expert_responses"])
                if r.get("selected_experts"):
                    for exp in r["selected_experts"]:
                        if exp not in combined_experts:
                            combined_experts.append(exp)
                if r.get("trace_dna"):
                    combined_trace.extend(r["trace_dna"])
                if r.get("verification"):
                    verification.update(r["verification"])

            out: Dict[str, Any] = {
                "expert_responses": combined_responses,
                "selected_experts": combined_experts,
                "trace_dna": combined_trace,
            }
            if final_answer:
                out["final_answer"] = final_answer
            if verification:
                out["verification"] = verification
            return out

        elif step.op == "VERIFY":
            verifier = step.verifier_expert or "analytical"
            target_output = env.get(step.target_step_output or "last_output", "")
            criteria = step.verification_criteria or "Check for accuracy and correctness."
            prompt = (
                f"Verify the following response against criteria: '{criteria}'\n\n"
                f"RESPONSE TO VERIFY:\n{target_output}\n\n"
                "Return SCORE: <0.0 to 1.0> and REASON: <brief summary>"
            )
            tracer.emit_sync(TraceEvent(
                kind=TraceKind.EXPERT_CALL_START.value,
                agent=verifier,
                data={"verifier": verifier, "criteria": criteria},
            ))
            try:
                res = await q_fn(verifier, prompt)
                res_text = getattr(res, "result", str(res))
                tracer.emit_sync(TraceEvent(
                    kind=TraceKind.EXPERT_CALL_END.value,
                    agent=verifier,
                    data={"verification": res_text[:200]},
                ))
                return {
                    "verification": {
                        "verifier": verifier,
                        "criteria": criteria,
                        "feedback": res_text,
                        "passed": True,
                    }
                }
            except Exception as e:
                tracer.emit_sync(TraceEvent(
                    kind=TraceKind.SANDBOX_ERROR.value,
                    agent=verifier,
                    data={"error": str(e)},
                ))
                return {
                    "verification": {
                        "verifier": verifier,
                        "criteria": criteria,
                        "error": str(e),
                        "passed": False,
                    }
                }

        return {}
