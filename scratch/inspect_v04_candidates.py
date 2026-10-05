import json
from collections import Counter

with open("research/data/v04/candidates.jsonl", "r", encoding="utf-8") as f:
    candidates = [json.loads(l) for l in f if l.strip()]

print(f"Total v04 candidates: {len(candidates)}")
print("Splits:", Counter(c["split"] for c in candidates))
print("Primary experts:", Counter(c["primary_expert"] for c in candidates))
origins = Counter(c["provenance"]["origin"] for c in candidates)
print("Origins:")
for origin, count in origins.items():
    print(f"  {origin}: {count}")

print("\nSample queries:")
for c in candidates[:5]:
    print(f"  [{c['id']}] ({c['primary_expert']}) regex={c['task_check_regex']}")
    print(f"    {c['query'][:100]}")
