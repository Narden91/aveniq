import asyncio
from pathlib import Path
from research.counterfactual import Candidate, execute_route, MoEConfig

async def test_num():
    candidate = Candidate(
        id="v04_numerical_000",
        query="Compute 113 times 19 plus 29. Return only the integer.",
        split="training",
        primary_expert="analytical",
        task_check_regex=r"^\s*2176\s*$",
        provenance={"origin": "test", "task_check": "test"},
    )
    config = MoEConfig()
    print("Testing system2 on v04_numerical_000...")
    res = await execute_route(candidate, "system2", config, 60, Path("scratch"))
    print("System2 answer:", repr(res.answer))
    print("System2 success:", res.task_success, "cost:", res.estimated_cost_usd, "failure:", res.failure_category, "error:", res.error)

if __name__ == "__main__":
    asyncio.run(test_num())
