import asyncio
from pathlib import Path
from research.counterfactual import Candidate, execute_route, MoEConfig

async def test_calc():
    candidate = Candidate(
        id="v03_calc_004",
        query="Compute the SHA-256 digest of the exact UTF-8 text AVENIQ v0.3 (without newline). Return only the hex string.",
        split="training",
        primary_expert="technical",
        task_check_regex=r"\b4137bdc647082f17da0bd96cf1d6108be1878420bf7034c902811b170e637df1\b",
        provenance={"origin": "test", "task_check": "test"},
    )
    config = MoEConfig()
    print("Testing single_expert on v03_calc_004...")
    res_se = await execute_route(candidate, "single_expert", config, 60, Path("scratch"))
    print("Single expert success:", res_se.task_success, "cost:", res_se.estimated_cost_usd, "failure:", res_se.failure_category)

    print("\nTesting system2 on v03_calc_004...")
    res_s2 = await execute_route(candidate, "system2", config, 120, Path("scratch"))
    print("System2 success:", res_s2.task_success, "cost:", res_s2.estimated_cost_usd, "failure:", res_s2.failure_category)

if __name__ == "__main__":
    asyncio.run(test_calc())
