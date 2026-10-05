import json

with open("benchmarks/data/adaptive_routing.jsonl", "r", encoding="utf-8") as f:
    records = [json.loads(l) for l in f if l.strip()]

s2_records = [r for r in records if (r.get("execution_class") or r.get("expected_execution_class")) == "system2"]

print(f"Total system2 benchmark records: {len(s2_records)}")
for r in s2_records[:10]:
    print(f"\n[{r['id']}] (split={r['split']})")
    print("  Query:", r["query"])
    print("  Tags:", r.get("tags"))
    print("  Provenance:", r.get("provenance"))
