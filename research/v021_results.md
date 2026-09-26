# AVENIQ v0.2.1 validity results

Run date: 2026-09-26. The scripts used [Laya 0.3.20](https://github.com/NandhaKishorM/laya) and the project environment. Exact settings are in [v021_config.json](v021_config.json). Raw samples and answers are in [laya_smoke_results.json](laya_smoke_results.json), [routing_results.json](routing_results.json), [downstream_results.jsonl](downstream_results.jsonl), and [downstream_summary.json](downstream_summary.json).

## Loaded model and hardware

The production path calls `laya.Agent.system_one` with two typed choice questions. The checkpoint is `convaiinnovations/laya`, revision `55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. Its local weights are in the ignored `.hf-cache/laya` directory. The model has 421,293,827 parameters. Both the reported execution device and parameter placement are `cuda:0` on an NVIDIA GeForce RTX 5080 Laptop GPU (compute capability 12.0). Parameters are float32; Laya uses bfloat16 autocast for inference. PyTorch was `2.10.0+cu128`.

After loading, CUDA allocated 1,685,436,928 bytes and reserved 1,799,356,416 bytes. Across the timed run, peak allocated memory was 2,441,986,560 bytes and peak reserved memory was 2,640,314,368 bytes. The smoke script stops if the reported device and parameter devices disagree.

The checkpoint emitted a warning about a temperature for choice questions with 11 or more options. AVENIQ's questions have three and four options. Laya also triggered a `reference_compile` deprecation warning in Transformers; this call is in the upstream package.

## Timing

The smoke run made one typed decision, then 20 warm-up calls and 100 measured calls. CUDA was synchronized around each timed call and around the model forward hook. Times below are milliseconds.

| Layer | Mean | p50 | p95 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Preprocessing before model forward | 9.18 | 9.32 | 12.22 | 13.56 |
| Model forward | 65.88 | 64.69 | 86.53 | 93.17 |
| Total Laya policy call | 76.29 | 74.69 | 99.71 | 107.63 |
| Outer wall clock for policy call | 76.33 | 74.72 | 99.76 | 107.67 |

End-to-end request times are reported separately below. Preprocessing includes Python and Laya work before the model forward. The forward measurement excludes language detection, tokenization, and answer decoding.

## Routing data and held-out scores

The checked corpus contains 200 authored and agent-reviewed requests: 40 direct, 100 single expert, and 60 System-2. Each row stores its origin, label basis, review status, and split. Twelve development cases contain misleading or quoted routing words. Labels were written independently of either policy's output. The split is fixed by SHA-256 of `aveniq-v0.2.1:<case id>` within each class: 100 development, 50 validation, and 50 test.

The confidence threshold of 0.4958 was chosen on validation only, minimizing class errors plus four times false bypasses. Neither test labels nor downstream answers set that threshold. The table reports the held-out test set.

| Measure | Result |
| --- | ---: |
| Execution-class accuracy | 19/50 = 38.00% |
| Raw expert-head accuracy on gold single-expert cases | 18/25 = 72.00% |
| Correct final expert route on gold single-expert cases | 1/25 = 4.00% |
| False bypass, gold System-2 sent elsewhere | 0/15 = 0.00% |
| Unnecessary escalation, simpler case sent to System-2 | 31/35 = 88.57% |
| System-2 routes | 46/50 = 92.00% |

Confusion matrix, rows gold and columns predicted:

| Gold | Direct | Single expert | System-2 |
| --- | ---: | ---: | ---: |
| Direct | 3 | 0 | 7 |
| Single expert | 0 | 1 | 24 |
| System-2 | 0 | 0 | 15 |

Direct precision was 3/3 = 100.00% and recall was 3/10 = 30.00%. Single-expert precision was 1/1 = 100.00% and recall was 1/25 = 4.00%. System-2 precision was 15/46 = 32.61% and recall was 15/15 = 100.00%. The raw Laya execution-class probabilities had multiclass Brier score 0.5287867848 and 10-bin ECE 0.212276. These probability scores are for the neural answer before threshold escalation. The JSON records coverage and class accuracy at confidence cutoffs from 0.00 through 1.00 in steps of 0.05. At 0.50, coverage was 6/50 = 12.00% and raw class accuracy among covered cases was 6/6 = 100.00%.

The routing-only rows have no task-success score.

## Real downstream requests

Thirty held-out cases were chosen before execution: 10 direct and 20 single expert. The set has no gold System-2 tasks, so its quality and cost numbers apply to this simpler subset only. The same configured provider and model, Groq `openai/gpt-oss-120b`, were used for all three arms. The 90 comparison records contain the answers, errors, token counts, timing, and task checks. No mock provider was used. Eighteen failed Laya requests were repeated because their first attempts dropped per-call model metadata after a later graph error. Those first attempts are retained separately in [downstream_superseded_attempts.jsonl](downstream_superseded_attempts.jsonl); all 30 Laya comparison records contain the model and timing metadata. The table uses the repeated attempt for those 18 cases.

`execution_success` means the runtime ended without a recorded error. `task_success` means the answer passed that case's stored regex check. The check revision is 2; seven false negatives caused by Unicode spacing or valid acknowledgment wording were corrected across all arms. Each row keeps its initial check result. These checks cover a requested fact or form but do not prove that every statement or code block in a long answer is correct.

| Policy | Execution success | Task-check success | Input tokens | Output tokens | List-price cost USD | Mean end-to-end latency | System-2 invoked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Always System-2 | 21/30 = 70.00% | 21/30 = 70.00% | 85,955 | 37,238 | $0.024558 | 25.61 s | 30/30 = 100.00% |
| RulePolicy | 30/30 = 100.00% | 30/30 = 100.00% | 2,823 | 21,233 | $0.013164 | 6.23 s | 0/30 = 0.00% |
| Laya control | 12/30 = 40.00% | 11/30 = 36.67% | 13,499 | 8,205 | $0.006949 | 6.88 s | 26/30 = 86.67% |

Nine always-System-2 requests and 18 Laya-control requests failed when the orchestrator's request exceeded the provider's model or per-minute token limit. One rate-limit failure caused a runtime attempt to use `openai/gpt-oss-20b`; that fallback was also rejected. The table shows the same configured model across arms and includes no successful fallback output. Costs use real provider token counts and [Groq's published $0.15 input / $0.60 output rates per million tokens](https://console.groq.com/docs/model/openai/gpt-oss-120b). They are list-price estimates, because the tracker does not record cached-input discounts or the final billed amount. Lower Laya cost here reflects many failed requests and must not be read as a quality-preserving saving.

For a Windows CUDA run, install the project's `laya` extra, then install the PyTorch `2.10.0+cu128` wheel from `https://download.pytorch.org/whl/cu128` into the same virtual environment. The project lockfile's ordinary PyPI PyTorch wheel is CPU-only on Windows. The CUDA 12.8 wheel used here supports the RTX 5080; the previous `2.6.0+cu124` wheel reported no `sm_120` support.
