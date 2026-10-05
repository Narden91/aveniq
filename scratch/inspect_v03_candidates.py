import json
from collections import Counter

with open("research/v03_candidates.jsonl", "r", encoding="utf-8") as f:
    cases = [json.loads(l) for l in f if l.strip()]

print(f"Total v03 candidates: {len(cases)}")
print("Splits:", Counter(c["split"] for c in cases))
print("Primary experts:", Counter(c["primary_expert"] for c in cases))
for c in cases[:10]:
    print(f"\n[{c['id']}] (split={c['split']}, expert={c['primary_expert']})")
    print("  Query:", c["query"])
    print("  Provenance:", c["provenance"])
