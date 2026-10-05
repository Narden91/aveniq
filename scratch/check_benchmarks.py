import json
from collections import Counter

with open("benchmarks/data/adaptive_routing.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(l) for l in f if l.strip()]

print(f"Total benchmark records: {len(records)}")
print("Splits:", Counter(r.get("split") for r in records))
print("Classes:", Counter(r.get("execution_class") or r.get("expected_execution_class") for r in records))
print("Sample record keys:", list(records[0].keys()))
print("Sample record:", records[0])
