import json
from collections import defaultdict

with open("research/data/v04/route_outcomes.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(l) for l in f if l.strip()]

print(f"Total v04 outcome records: {len(records)}")
by_id = defaultdict(dict)
for r in records:
    by_id[r["id"]][r["route"]] = r

for tid, routes in by_id.items():
    print(f"\nTask: {tid}")
    for route, res in routes.items():
        print(f"  {route:15}: success={res['task_success']}, cost={res.get('estimated_cost_usd')}, failure={res.get('failure_category')}")
