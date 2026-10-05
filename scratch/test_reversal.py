import asyncio
from pathlib import Path
from research.counterfactual import Candidate, execute_route, MoEConfig

async def test_reversal():
    candidate = Candidate(
        id="v04_transformation_000",
        query="Reverse the characters in TASK000AVENIQ. Return only the reversed text.",
        split="training",
        primary_expert="general",
        task_check_regex=r"^\s*QINEVA000KSAT\s*$",
        provenance={"origin": "test", "task_check": "test"},
    )
    config = MoEConfig()
    print("Testing single_expert on string reversal...")
    res_se = await execute_route(candidate, "single_expert", config, 60, Path("scratch"))
    print("Single expert answer:", repr(res_se.answer))
    print("Single expert success:", res_se.task_success, "error:", res_se.error)

    print("\nTesting system2 on string reversal...")
    res_s2 = await execute_route(candidate, "system2", config, 120, Path("scratch"))
    print("System2 answer:", repr(res_s2.answer))
    print("System2 success:", res_s2.task_success, "error:", res_s2.error)

if __name__ == "__main__":
    asyncio.run(test_reversal())
