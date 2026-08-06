# VibeSec

VibeSec is a benchmark for measuring whether AI coding models write **secure** code, not
just working code. Every task is verified by execution: a real exploit runs against the
model's patch and either fires or it doesn't. No model judges the result.

v1.0.0 contains **1,000 execution-verified security-patching tasks** across seven
vulnerability classes, on FastAPI/Python.

## Why execution verification

A recent benchmark found AI-generated code is 61% functionally correct but only 10.5%
secure ([arXiv:2512.03262](https://arxiv.org/abs/2512.03262)). Static analysis and
LLM-as-judge both miss logic-level flaws, because an authorization bug is not a pattern —
it is a behaviour. A task counts as solved only when the exploit stops working *and* the
application still does what it was supposed to do.

## Task format

Each line of `dataset.jsonl` is one task:

```
id               Stable task identifier
seed_prompt      The feature request the vulnerable app was generated from
vuln_class       idor | missing_auth | mass_assignment | privilege_escalation |
                 path_traversal | sql_injection | other
app_files        The vulnerable application, path -> source
exploit          Runnable script that prints PWNED while the vulnerability is live
spec_test        Runnable script asserting the app's intended behaviour
patched_files    Reference patch (never read at grading time)
```

A task enters a release only if, in a sandbox: the spec suite passes on the vulnerable
app, the exploit fires on the vulnerable app, and the reference patch both blocks the
exploit and keeps the spec suite green. Candidates failing any gate are discarded rather
than shipped.

The reference patch exists so reviewers can spot-check offline. Grading never reads it —
scoring is behavioural, so any patch that defends the exploit and preserves intended
behaviour counts, whatever its internal structure.

## Grading

A model receives `app_files` and returns a patch. Two gates:

1. **Security** — the exploit is re-run against the patch. If it still prints `PWNED`,
   the task fails (`exploit_still_works`).
2. **Functional** — the spec suite is re-run. If the patch broke intended behaviour, the
   task fails (`spec_broken`). This is what stops a model from "securing" an endpoint by
   disabling it.

Patches may not add dependencies. The sandbox has no network at run time, and
`requirements.txt` is pinned back to the original before grading, so a dependency change
is recorded as a separate `requirements_changed` flag rather than being mis-scored as an
insecure patch. Models are told this rule in the prompt.

## Leaderboard

1,000 tasks, one attempt per task.

| Model | Secure patches | |
|---|---:|---|
| claude-opus-4-8 | 649 / 1000 | 64.9% |
| claude-sonnet-4.6 | 373 / 1000 | 37.3% |
| kimi-k2.7-code | 372 / 1000 | 37.2% |
| glm-5.2 | 337 / 1000 | 33.7% |
| nemotron-3-ultra | 289 / 1000 | 28.9% |
| mistral-medium-3-5 | 129 / 1000 | 12.9% |
| gpt-oss-120b | 111 / 1000 | 11.1% |

`results/eval_outcomes.jsonl` has the per-`(model, task)` outcome behind every number —
which task, which vulnerability class, pass or fail, and the failure stage.
`results/leaderboard.json` has the aggregate with per-class breakdowns. Rebuild the table
from the raw outcomes yourself:

```bash
python eval/merge_runs.py --dry-run
```

## Running the benchmark

Sandboxed execution runs on [Modal](https://modal.com).

```bash
git clone https://github.com/muence-ai/vibesec
cd vibesec
pip install -r requirements.txt
modal setup
```

Create secrets for the providers you want:

```bash
modal secret create openrouter-key OPENROUTER_API_KEY=...
modal secret create anthropic-key  ANTHROPIC_API_KEY=...
modal secret create wandb-key      WANDB_API_KEY=...
```

Evaluate:

```bash
# 1 model x 5 tasks, ~$0.10
modal run eval/eval_models.py --sanity-check

# one model over the full benchmark
modal run eval/eval_models.py --model "anthropic/claude-opus-4-8"

# resume an interrupted run
modal run --detach eval/eval_models.py --model "..." --missing-only
```

Anything not routed directly to Anthropic or W&B Inference falls through to OpenRouter, so
`--model "mistralai/mistral-medium-3-5"` works without code changes. Roughly $14 per
frontier model for all 1,000 tasks.

Fold per-run logs into the leaderboard:

```bash
python eval/merge_runs.py --dry-run   # report only
python eval/merge_runs.py             # write results/
```

## Layout

```
dataset.jsonl                  the 1,000 tasks
eval/eval_models.py            scores a model against the benchmark
eval/merge_runs.py             folds per-run logs into the leaderboard
results/eval_outcomes.jsonl    per-(model, task) outcomes behind the leaderboard
results/leaderboard.json       aggregate scores with per-class breakdowns
docs/dataset-card.md           Hugging Face dataset card
```

Scripts resolve paths relative to the repository root — run them from here.

## Known limitations

- **Single framework.** Every v1.0.0 task is FastAPI/Python.
- **Class imbalance.** 68.8% of v1.0.0 is IDOR; SQL injection is 0.6%. See
  `PROVENANCE.md` for the full distribution.
- **Short horizon.** ~2 files per task, so these are localized patches rather than
  multi-subsystem work.
- **One attempt per task** in the published leaderboard — no variance estimate yet.
- **Machine-generated applications.** Realistic in shape, but not drawn from production
  repositories. See `PROVENANCE.md`.
- **Public, therefore contaminable.** Treat this as an open inspect set. Official ranking
  should use a held-out split that has never been published.

## Terminology

**IDOR** (Insecure Direct Object Reference), in current OWASP API terms **Broken Object
Level Authorization**: a user can reach an object by ID that they do not own or control.

## License

MIT — see [LICENSE](LICENSE). Provenance and generation details in
[PROVENANCE.md](PROVENANCE.md).
