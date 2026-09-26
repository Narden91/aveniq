"""CLI entry-point for the AVENIQ adaptive compute agent runtime."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)


async def run_query(
    query: str,
    model: str | None = None,
    *,
    memory=None,
) -> None:
    """Run a single query through the programmatic orchestration pipeline."""
    from src.core.config import MoEConfig, SecretStr, apply_model_override
    from src.core.state import create_initial_state
    from src.graph.builder import MoEGraphBuilder
    from src.utils.metrics import reset_token_tracker

    config = MoEConfig(groq_api_key=SecretStr(os.getenv("GROQ_API_KEY", "")))
    if model:
        apply_model_override(config, model)
    config.validate()

    graph = MoEGraphBuilder(config).build()
    state = create_initial_state(query)

    # Inject conversation context if memory is provided
    if memory:
        state["conversation_context"] = memory.format_context()

    # Reset token tracker for this request.
    reset_token_tracker()

    result = await graph.ainvoke(state)

    # Record turn in memory
    if memory:
        memory.add(
            query=query,
            answer=result.get("final_answer", ""),
            experts_used=result.get("selected_experts", []),
        )

    print("\n--- Generated Orchestration Code ---")
    print(result.get("generated_code", "(none)"))
    print("\n--- Experts Called ---")
    print(", ".join(result.get("selected_experts", [])))
    print("\n--- Final Answer ---")
    print(result.get("final_answer", "(no answer)"))

    iterations = result.get("code_execution_iterations", 0)
    if iterations > 1:
        print(f"\n(sandbox retried {iterations} time(s))")

    # Token usage summary
    usage = result.get("token_usage", {})
    if usage.get("total_tokens"):
        print("\n--- Token Usage ---")
        print(f"  Input:  {usage['total_input_tokens']:>7,}")
        print(f"  Output: {usage['total_output_tokens']:>7,}")
        print(f"  Total:  {usage['total_tokens']:>7,}")
        print(f"  Cost:   ${usage['estimated_cost_usd']:.4f}")

    # Execution plan summary
    plan = result.get("execution_plan", {})
    if plan.get("calls"):
        mode = []
        if plan.get("has_parallel"):
            mode.append(f"{plan['gather_groups']} parallel group(s)")
        if plan.get("has_sequential"):
            mode.append("sequential")
        print("\n--- Execution Plan ---")
        print(f"  Mode:    {' + '.join(mode)}")
        print(f"  Experts: {', '.join(plan['experts_used'])}")


async def interactive_mode(model: str | None = None) -> None:
    """Multi-turn interactive conversation loop."""
    from src.utils.memory import ConversationMemory

    memory = ConversationMemory(max_turns=20)
    print("Multi-turn MoE session (type 'exit' or 'quit' to stop)\n")

    while True:
        try:
            query = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query.lower() in ("exit", "quit"):
            break
        await run_query(query, model=model, memory=memory)
        print()  # blank line between turns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AVENIQ — Adaptive Agent Runtime CLI",
    )
    parser.add_argument("query", nargs="?", help="The query to process")
    parser.add_argument(
        "--model", "-m", default=None,
        help="Override the LLM model name for all agents",
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true",
        help="Start an interactive multi-turn conversation",
    )
    args = parser.parse_args()

    if args.interactive:
        asyncio.run(interactive_mode(model=args.model))
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    asyncio.run(run_query(args.query, model=args.model))


if __name__ == "__main__":
    main()
