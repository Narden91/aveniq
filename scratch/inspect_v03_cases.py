import json

with open("research/v03_candidates.jsonl", "r", encoding="utf-8") as f:
    cases = [json.loads(l) for l in f if l.strip()]

for c in cases:
    print(f"[{c['id']}] {c['primary_expert']} : {c['query'][:70]} (check={c['task_check_regex']})")
