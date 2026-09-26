"""Deterministic rule-based PolicyEngine implementation."""

import re
from typing import Any, Dict, List, Optional
from .decision import PolicyDecision


class RulePolicy:
    """Heuristic rule-based policy engine for fast, local decisions without LLM calls.

    Categorizes queries into direct, single_expert, parallel_experts, or system2
    based on keyword matching, length, and syntactic structure.
    """

    # Direct greetings / trivial queries
    GREETINGS = {
        "hi", "hello", "hey", "greetings", "ping", "test", "who are you",
        "what can you do", "help", "good morning", "good evening", "howdy"
    }

    # Domain keyword maps
    DOMAIN_KEYWORDS = {
        "technical": [
            "code", "python", "function", "debug", "refactor", "algorithm",
            "regex", "sql", "api", "git", "class", "javascript", "bash",
            "exception", "syntax", "compile", "docker", "fastapi", "pandas",
            "numpy", "asyncio", "database", "query", "endpoint"
        ],
        "creative": [
            "story", "poem", "lyrics", "fiction", "novel", "slogan", "pitch",
            "rhyme", "metaphor", "brainstorm", "creative", "screenplay",
            "dialogue", "haiku", "narrative", "character"
        ],
        "analytical": [
            "compare", "comparison", "pros and cons", "benchmark", "metrics",
            "tradeoff", "analysis", "statistics", "evaluate", "breakdown",
            "advantages", "disadvantages", "difference between", "trend",
            "swot", "feasibility"
        ],
        "general": [
            "what is", "who is", "who was", "capital of", "define", "summary",
            "explain", "overview", "history of", "how does", "why is",
            "population of", "when was"
        ],
    }

    # Complex multi-step markers that trigger System-2
    SYSTEM2_TRIGGERS = [
        "orchestrate", "multi-step", "workflow", "pipeline", "loop",
        "step-by-step", "first then", "execute code", "run in sandbox",
        "generate a script", "complex task", "plan and execute",
        "synthesize across all experts"
    ]

    def __init__(
        self,
        default_expert: str = "general",
        system2_length_threshold: int = 600,
        require_verification_keywords: Optional[List[str]] = None,
    ):
        self.default_expert = default_expert
        self.system2_length_threshold = system2_length_threshold
        self.require_verification_keywords = require_verification_keywords or [
            "verify", "double check", "audit", "correctness", "rigor",
            "validate", "strict", "ensure"
        ]

    def evaluate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        cleaned = (query or "").strip()
        lowered = cleaned.lower()

        # Check for empty query
        if not cleaned:
            return PolicyDecision(
                execution_class="direct",
                primary_expert=None,
                model_tier="local",
                needs_verification=False,
                max_agent_calls=1,
                confidence=1.0,
                reasoning="Empty query handled directly.",
            )

        # 1. Direct check: simple greeting / status ping
        normalized_words = re.sub(r"[^\w\s]", "", lowered).split()
        normalized_str = " ".join(normalized_words)
        if normalized_str in self.GREETINGS or (len(normalized_words) <= 3 and any(w in self.GREETINGS for w in normalized_words)):
            return PolicyDecision(
                execution_class="direct",
                primary_expert=None,
                model_tier="local",
                needs_verification=False,
                max_agent_calls=1,
                confidence=1.0,
                reasoning="Simple greeting / status inquiry matched direct execution.",
            )

        # Check for verification requirement
        needs_verification = any(kw in lowered for kw in self.require_verification_keywords)

        # 2. System-2 check: explicit trigger phrases or long complex queries
        if any(trig in lowered for trig in self.SYSTEM2_TRIGGERS) or len(cleaned) > self.system2_length_threshold:
            return PolicyDecision(
                execution_class="system2",
                primary_expert=None,
                model_tier="strong",
                needs_verification=True,
                max_agent_calls=5,
                confidence=0.95,
                reasoning="Complex multi-step or lengthy query routed to System-2 orchestrator.",
            )

        # 3. Domain score calculation
        domain_scores: Dict[str, int] = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if re.search(r"\b" + re.escape(kw) + r"\b", lowered))
            if score > 0:
                domain_scores[domain] = score

        # 4. Multi-expert parallel check (distinct domains with significant scores)
        high_scoring_domains = [d for d, s in domain_scores.items() if s >= 2]
        if len(high_scoring_domains) >= 2:
            return PolicyDecision(
                execution_class="parallel_experts",
                primary_expert=high_scoring_domains[0],
                model_tier="fast",
                needs_verification=needs_verification,
                max_agent_calls=len(high_scoring_domains),
                confidence=0.85,
                parallel_experts=high_scoring_domains,
                reasoning=f"Multi-domain query matching {high_scoring_domains} routed to parallel experts.",
            )

        # 5. Single expert check
        if domain_scores:
            best_domain = max(domain_scores, key=lambda k: domain_scores[k])
            confidence = min(0.70 + 0.1 * domain_scores[best_domain], 0.98)
            return PolicyDecision(
                execution_class="single_expert",
                primary_expert=best_domain,
                model_tier="fast",
                needs_verification=needs_verification,
                max_agent_calls=1,
                confidence=confidence,
                reasoning=f"High match for domain '{best_domain}' routed to single expert.",
            )

        # 6. Default fallback: single expert with default
        return PolicyDecision(
            execution_class="single_expert",
            primary_expert=self.default_expert,
            model_tier="fast",
            needs_verification=needs_verification,
            max_agent_calls=1,
            confidence=0.60,
            reasoning=f"General domain fallback routed to expert '{self.default_expert}'.",
        )

    async def aevaluate(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PolicyDecision:
        """Async evaluation - runs the sync rule evaluation."""
        return self.evaluate(query, context=context)
