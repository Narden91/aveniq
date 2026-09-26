"""Always-System-2 baseline policy engine."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .decision import PolicyDecision
from .engine import PolicyEngine


class AlwaysSystem2Policy(PolicyEngine):
    """Baseline policy that unconditionally routes every request to the generative System-2 orchestrator."""

    def evaluate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        return PolicyDecision(
            execution_class="system2",
            needs_verification=True,
            max_agent_calls=5,
            confidence=1.0,
            model_tier="strong",
            reasoning="Unconditional baseline: always invoke generative System-2 orchestrator.",
            metadata={"baseline": "always_system2"},
        )

    async def aevaluate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        return self.evaluate(query, context=context)
