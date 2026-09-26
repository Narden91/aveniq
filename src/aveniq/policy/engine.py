"""PolicyEngine protocol and interface for AVENIQ compute routing."""

from typing import Any, Dict, Optional, Protocol, runtime_checkable
from .decision import PolicyDecision


@runtime_checkable
class PolicyEngine(Protocol):
    """Protocol for policy engines making compute allocation decisions.

    All downstream code must depend on this interface, never on a specific
    backend such as Laya or RulePolicy.
    """

    def evaluate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        """Synchronously evaluate a query and return a PolicyDecision.

        Parameters
        ----------
        query:
            The user prompt or task description.
        context:
            Optional context dictionary containing conversation history,
            turn metadata, or available expert lists.
        """
        ...

    async def aevaluate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        """Asynchronously evaluate a query and return a PolicyDecision.

        Parameters
        ----------
        query:
            The user prompt or task description.
        context:
            Optional context dictionary.
        """
        ...
