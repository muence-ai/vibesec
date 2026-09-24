---
license: mit
pretty_name: VibeSec
language:
- en
task_categories:
- text-generation
tags:
- security
- code
- benchmark
- llm-evaluation
- vulnerability
- fastapi
size_categories:
- 1K<n<10K
---

# VibeSec V1.1

**1,000 execution-verified security patching tasks for AI coding agents.**

Each task contains a vulnerable FastAPI app, a normal-behavior spec test, an
exploit that prints `PWNED` against the vulnerable app, and a reference patch.

## Changes from v1.0.0

v1.0.0's 1,000 tasks contained generation-seed fan-out duplicates — only **755
unique scenarios**. v1.1 collapses to those 755 and adds **245 seed-unique tasks**
(238 from an under-represented-class batch + 7 for balance) → a true 1,000 unique
tasks. Every v1.0.0 score was inflated by the redundancy; v1.1 is recomputed on the
corrected set. Panel: dropped `gpt-oss-120b`/`kimi-k2.7-code`, added Nemotron 3.5
Lightning and Gemini 3.8 Flash.

## Leaderboard (single-shot, pass@1)

| Model | Pass rate | Passed / Total |
|---|---:|---:|
| Claude Opus 4.8 | 55.4% | 554/1000 |
| Gemini 3.8 Flash | 55.1% | 551/1000 |
| Claude Sonnet 4.6 | 32.8% | 328/1000 |
| GLM 5.2 | 27.9% | 279/1000 |
| Nemotron 3 Ultra | 27.4% | 274/1000 |
| Nemotron 3.5 Lightning | 25.6% | 256/1000 |

Single-shot: one attempt per task, graded 0/1. Low numbers are by design.
*Gemini 3.8 Flash was run with thinking disabled — on default settings it
truncated before completing the required file format on ~51% of tasks; all other
models use their defaults.*

## Vulnerability mix

| class | tasks | share |
|---|---:|---:|
| `idor` | 513 | 51.3% |
| `missing_auth` | 150 | 15.0% |
| `mass_assignment` | 130 | 13.0% |
| `privilege_escalation` | 80 | 8.0% |
| `path_traversal` | 57 | 5.7% |
| `sql_injection` | 58 | 5.8% |
| `other` | 12 | 1.2% |

## Evaluation protocol

Each patch is checked by two executable gates in a sandbox: (1) the original
exploit must no longer print `PWNED`, and (2) the spec test must still print
`SPEC_PASS`. Patches that omit or change `requirements.txt` are rejected before
execution.
