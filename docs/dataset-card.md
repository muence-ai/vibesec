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
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
---

# VibeSec

**Execution-verified security evals for AI coding agents.**

VibeSec measures whether a model can patch a real, exploit-proven vulnerability
without breaking the app. Each task starts from a casual founder-style prompt,
generates a small FastAPI app, proves a vulnerability with a runnable exploit,
and verifies the patch in a sandbox.

- 📊 **Code & leaderboard:** https://github.com/jenishk20/vibesec-evals
- 🧪 **Verification:** every triplet passes four executable gates — no LLM-as-judge.

## What makes it different

Most LLM security benchmarks use static snippets or judge-model grading. VibeSec
uses executable gates. A task only enters this dataset if all four pass:

1. The generated app boots and passes a golden-path spec test.
2. A real exploit script prints `PWNED`.
3. A reference patch makes that exact exploit fail.
4. The patched app still passes the original spec.

## Dataset summary

- **634 verified triplets.**
- Each is a vulnerable app, a passing spec test, a working exploit, and a
  known-good reference patch.
- Vulnerabilities are the ones AI coding assistants reintroduce when told to
  "just ship it": missing ownership checks (IDOR), unauthenticated endpoints,
  mass assignment, and privilege escalation.

### Vulnerability class distribution

| class | tasks | share |
|---|---:|---:|
| `idor` (broken object-level authorization) | 472 | 74.4% |
| `missing_auth` | 115 | 18.1% |
| `mass_assignment` | 36 | 5.7% |
| `privilege_escalation` | 5 | 0.8% |
| `other` | 5 | 0.8% |
| `path_traversal` | 1 | 0.2% |

> IDOR = Insecure Direct Object Reference; in current OWASP API terms, Broken
> Object Level Authorization — a user can access an object by ID even though
> they do not own it.

## Fields

The default config uses a viewer-friendly Parquet file with flattened columns:

| field | meaning |
|---|---|
| `id` | sha256 prefix of the exploit (dedup key) |
| `language` | source language (`python`) |
| `runtime` | execution runtime (`python_fastapi`) |
| `framework` | web framework (`fastapi`) |
| `seed_prompt` | the casual founder prompt that produced the app |
| `vuln_class` | heuristic label from the exploit code |
| `app_main_py` | vulnerable FastAPI app |
| `app_requirements_txt` | app dependencies |
| `spec_test` | golden-path test — prints `SPEC_PASS` / `SPEC_FAIL` |
| `exploit` | working exploit — prints `PWNED` / `safe` |
| `exploit_output` | captured stdout proving the attack landed |
| `patched_main_py` | reference patch |
| `patched_requirements_txt` | patched app dependencies |
| `source_file` | provenance path inside the factory run |

The repository also includes `dataset.jsonl` as the original nested JSONL file:

| field | meaning |
|---|---|
| `id` | sha256 prefix of the exploit (dedup key) |
| `seed_prompt` | the casual founder prompt that produced the app |
| `vuln_class` | heuristic label from the exploit code |
| `app_files` | the vulnerable app (`main.py` + `requirements.txt`) |
| `spec_test` | golden-path test — prints `SPEC_PASS` / `SPEC_FAIL` |
| `exploit` | working exploit — prints `PWNED` / `safe` |
| `exploit_output` | captured stdout proving the attack landed |
| `patched_files` | a reference patch that closes the bug and keeps the spec green |
| `source_file` | provenance path inside the factory run |

## Usage

```python
from datasets import load_dataset

ds = load_dataset("muence/vibesec", split="train")
ex = ds[0]
print(ex["vuln_class"])
print(ex["app_main_py"])
print(ex["exploit"])
```

To use the raw nested JSONL directly, download `dataset.jsonl` from the repo.

## Leaderboard snapshot (V0)

Six models scored on patching the 634 tasks (pass = exploit blocked **and** spec
still passes):

| model | pass rate |
|---|---:|
| Claude Opus 4.8 | 446 / 634 (70.3%) |
| Claude Sonnet 4.6 | 256 / 634 (40.4%) |
| Kimi K2.7 Code | 233 / 634 (36.8%) |
| GLM 5.2 † | 198 / 545 (36.3%) |
| Nemotron 3 Ultra † | 179 / 606 (29.5%) |
| GPT-OSS 120B | 80 / 634 (12.6%) |

> † GLM 5.2 and Nemotron 3 Ultra runs did not complete all 634 tasks (some
> requests returned no response after retries); their pass rates are over the
> tasks actually scored (545 and 606 respectively), not the full 634.

## Limitations

- The dataset is synthetic and FastAPI-focused.
- It is currently IDOR-heavy (74%); other classes are under-represented.
- Vulnerability labels are heuristic and should be refined before larger use.
- Tasks are public and inspectable, so a serious leaderboard should add a
  private held-out split for official scoring.

## License

MIT.

## Citation

```bibtex
@misc{vibesec2026,
  title  = {VibeSec: Execution-Verified Security Evals for AI Coding Agents},
  author = {Kothari, Jenish},
  year   = {2026},
  url    = {https://github.com/jenishk20/vibesec-evals}
}
```
