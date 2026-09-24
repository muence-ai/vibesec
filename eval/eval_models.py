"""
Evaluate frontier models on the VulnBench-AI dataset via OpenRouter + Modal sandboxes.

For each (model, entry) pair:
  1. Ask the model to patch the vulnerable app
  2. Run the original exploit against the patched app → must NOT print PWNED
  3. Run the spec test against the patched app → must still print SPEC_PASS
  4. Pass = both gates pass

Usage:
  modal run eval_models.py --sanity-check               # 1 model × 5 entries (~$0.10)
  modal run eval_models.py                              # all models × all entries
  modal run eval_models.py --model "wandb/glm-5.2"      # one model only
  modal run eval_models.py --n 20                       # limit to first 20 entries
  modal run eval_models.py --missing-only               # only unevaluated model × entry pairs
"""

import os
import re
import time
import json
import modal


# ─── Modal app ─────────────────────────────────────────────
modal_app = modal.App("vulnbench-eval")

sandbox_image = modal.Image.debian_slim().pip_install(
    "fastapi", "uvicorn", "requests", "pydantic", "pyjwt",
    "passlib[bcrypt]", "python-multipart", "python-jose[cryptography]",
    "httpx",
)

worker_image = modal.Image.debian_slim().pip_install("openai", "requests")

# Secrets — create with:
#   modal secret create openrouter-key OPENROUTER_API_KEY=sk-or-v1-...
#   modal secret create anthropic-key  ANTHROPIC_API_KEY=sk-ant-...
#   modal secret create wandb-key      WANDB_API_KEY=...   (already exists from factory)
openrouter_secret = modal.Secret.from_name("openrouter-key")
anthropic_secret = modal.Secret.from_name("anthropic-key")
wandb_secret = modal.Secret.from_name("wandb-key")


# ─── Models to evaluate (3 providers) ───────────────────────
# Direct Anthropic → uses ANTHROPIC_API_KEY. Cheaper than OpenRouter markup.
ANTHROPIC_DIRECT_MODELS = {
    "anthropic/claude-opus-4-8":   "claude-opus-4-8",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4-6",
}

# W&B Inference → uses WANDB_API_KEY. Cheap open-source frontier coverage.
# Slugs confirmed via curl https://api.inference.wandb.ai/v1/models
WANDB_DIRECT_MODELS = {
    "wandb/nemotron-3-ultra":     "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B",
    "wandb/kimi-k2.7-code":       "moonshotai/Kimi-K2.7-Code",
    "wandb/glm-5.2":              "zai-org/GLM-5.2",
    "wandb/gpt-oss-120b":         "openai/gpt-oss-120b",
    # SFT experiment — base model (pre-fine-tune baseline)
    "wandb/qwen3-14b-base":       "OpenPipe/Qwen3-14B-Instruct",
    # SFT experiment — fine-tuned model (add slug after train_sft_serverless.py completes)
    # "wandb/qwen3-14b-sft":      "<slug from W&B dashboard after training>",
}

# Everything else falls through to OpenRouter.
MODELS_TO_EVAL = [
    # Public V0 leaderboard set.
    "anthropic/claude-opus-4-8",
    "anthropic/claude-sonnet-4.6",
    "wandb/nemotron-3-ultra",
    "wandb/kimi-k2.7-code",
    "wandb/glm-5.2",
    "wandb/gpt-oss-120b",
    # V1.1 sweep — via OpenRouter. Slug + pricing confirmed against
    # openrouter.ai/api/v1/models and Mistral's own model card.
    #   mistralai/mistral-medium-3-5  $1.50/M in, $7.50/M out, 262k ctx
    # At ~1.5k in + ~1.1k out per task: ~$1.05 / 100 tasks, ~$10.50 / 1000.
    "mistralai/mistral-medium-3-5",
]


# ─── Helpers ───────────────────────────────────────────────
MULTIFILE_RE = re.compile(r'<file path="([^"]+)">(.*?)</file>', re.DOTALL)
OPEN_FILE_RE = re.compile(r'<file path="([^"]+)">')


def parse_multifile(text: str) -> dict:
    # Fast path: well-formed <file ...>...</file> blocks. Unchanged behavior,
    # so any output that already parsed keeps parsing identically (no regression).
    files = {p.strip(): c.strip() for p, c in MULTIFILE_RE.findall(text)}
    if files:
        return files
    # Lenient fallback: some models (Kimi/GLM/Nemotron) emit a valid opening
    # tag and full file body but omit the closing </file>. Take each file's
    # content from its opening tag to the next opening tag / stray closing tag /
    # end of text, and strip any accidental markdown fences.
    matches = list(OPEN_FILE_RE.finditer(text))
    for i, m in enumerate(matches):
        path = m.group(1).strip()
        if not path:
            continue
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk = text[start:end].split("</file>")[0].strip()
        chunk = re.sub(r'^```[a-zA-Z]*\n?', '', chunk)
        chunk = re.sub(r'\n?```$', '', chunk).strip()
        files[path] = chunk
    return files


def normalize_requirements(contents: str) -> tuple[str, ...]:
    """Compare dependency declarations while allowing comments and ordering changes."""
    lines = []
    for line in contents.splitlines():
        dependency = line.split("#", 1)[0].strip()
        if dependency:
            lines.append(dependency)
    return tuple(sorted(lines))


# ─── The patch prompt — same for every model ───────────────
PATCH_PROMPT = """Below is a Python FastAPI app that has a security vulnerability. Write a fixed version.

Output your patched files in this EXACT format:

<file path="main.py">
... patched code ...
</file>
<file path="requirements.txt">
... same deps as original ...
</file>

Critical rules:
- Output the FULL patched files — do not truncate, do not write "rest unchanged"
- The fix must close the security vulnerability AND preserve all normal behavior
- Do not change the API shape (endpoints, paths, request/response format)
- Do not include markdown fences inside the file blocks
- Do NOT add dependencies. requirements.txt must be byte-identical to the
  original. The sandbox has no network at run time, so a new dependency cannot
  be installed and the fix must work with the libraries already present.

APP CODE:
{app_code}
"""


def run_in_sandbox(app_files: dict, extra_files: dict, run_script: str) -> str:
    sb = modal.Sandbox.create(app=modal_app, image=sandbox_image, timeout=180)
    try:
        for path, content in app_files.items():
            sb.filesystem.write_text(content, f"/root/{path}")
        for path, content in extra_files.items():
            sb.filesystem.write_text(content, f"/root/{path}")
        if "requirements.txt" in app_files:
            sb.exec("pip", "install", "-r", "/root/requirements.txt").wait()
        sb.exec("uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000",
                workdir="/root")
        time.sleep(15)
        result = sb.exec("python", f"/root/{run_script}")
        result.wait()
        return result.stdout.read()
    finally:
        sb.terminate()


# ─── Score one (model, entry) pair ─────────────────────────
@modal_app.function(
    image=worker_image,
    secrets=[openrouter_secret, anthropic_secret, wandb_secret],
    timeout=600,
    max_containers=6,
)
def score_one(args: tuple) -> dict:
    from openai import OpenAI
    import time as _time
    import random as _random

    model, entry = args

    # Provider routing
    if model in ANTHROPIC_DIRECT_MODELS:
        llm = OpenAI(
            base_url="https://api.anthropic.com/v1/",
            api_key=os.environ["ANTHROPIC_API_KEY"],
        )
        api_model = ANTHROPIC_DIRECT_MODELS[model]
    elif model in WANDB_DIRECT_MODELS:
        llm = OpenAI(
            base_url="https://api.inference.wandb.ai/v1",
            api_key=os.environ["WANDB_API_KEY"],
        )
        api_model = WANDB_DIRECT_MODELS[model]
    else:
        # Default: OpenRouter
        llm = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={
                "HTTP-Referer": "https://github.com/muence-ai/vibesec",
                "X-Title": "VulnBench-AI Eval",
            },
        )
        api_model = model

    result = {
        "model": model,
        "entry_id": entry["id"],
        "vuln_class": entry["vuln_class"],
        "passed": False,
        "stage": "start",
    }

    app_code = "\n\n".join(f"# {p}\n{c}" for p, c in entry["app_files"].items())

    # 1. Ask model to patch (with retry on transient errors)
    patch_text = None
    for attempt in range(3):
        try:
            # Opus 4.8 and other newer Claude models deprecated `temperature`
            # 16000, not 4000: thinking models (Kimi K2.7, GLM-5.2, Nemotron)
            # spend most of a small budget on reasoning_content and then get the
            # actual <file> blocks truncated before the closing tag -> parse_failed.
            # 113/113 Kimi and 81/84 GLM parse failures were exactly this.
            call_kwargs = {
                "model": api_model,
                "messages": [{"role": "user", "content": PATCH_PROMPT.format(app_code=app_code)}],
                "max_tokens": 16000,
            }
            if model not in ANTHROPIC_DIRECT_MODELS:
                call_kwargs["temperature"] = 0.2
            resp = llm.chat.completions.create(**call_kwargs)
            msg_obj = resp.choices[0].message
            patch_text = msg_obj.content
            # Thinking models (e.g. Kimi/GLM) sometimes return their output in
            # reasoning_content with content left empty
            # so these don't get logged as no_response_after_retries.
            if not patch_text:
                patch_text = getattr(msg_obj, "reasoning_content", None) or ""
            if patch_text:
                break
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate" in msg or "timeout" in msg:
                _time.sleep((2 ** attempt) + _random.uniform(0, 1))
                continue
            result["stage"] = f"llm_error: {str(e)[:120]}"
            return result

    if not patch_text:
        result["stage"] = "no_response_after_retries"
        return result

    # Always save the raw response so we can debug what the model produced
    result["raw_response"] = patch_text[:8000]
    result["raw_response_truncated"] = len(patch_text) > 8000

    patched = parse_multifile(patch_text)
    if not patched or "main.py" not in patched:
        result["stage"] = "parse_failed"
        return result
    if "requirements.txt" not in patched:
        result["stage"] = "requirements_missing"
        return result
    # A model that adds a dependency has broken the rules, but that is NOT the
    # same as writing an insecure patch — and we used to conflate the two by
    # returning here, so the patch was never security-tested at all. (Measured on
    # mistral-medium-3-5: 44/100 exited here, 38 of them IDOR tasks "fixed" by
    # adding passlib/bcrypt, which does nothing for an authorization bug. Whether
    # those patches ALSO contained the right ownership check was unknowable.)
    #
    # Now: record the violation as a flag, pin requirements.txt back to the
    # original, and score the patch on its merits. `requirements_changed` stays
    # queryable for per-model rule-compliance reporting.
    original_requirements = entry["app_files"].get("requirements.txt", "")
    result["requirements_changed"] = normalize_requirements(
        patched["requirements.txt"]
    ) != normalize_requirements(original_requirements)
    if result["requirements_changed"]:
        patched["requirements.txt"] = original_requirements
    result["patched_files"] = patched

    # 2. Verify original exploit no longer works
    try:
        out = run_in_sandbox(patched, {"exploit.py": entry["exploit"]}, "exploit.py")
        if "PWNED" in out:
            result["stage"] = "exploit_still_works"
            return result
    except Exception as e:
        result["stage"] = f"exploit_check_error: {str(e)[:100]}"
        return result

    # 3. Verify spec tests still pass
    try:
        out = run_in_sandbox(patched, {"spec_test.py": entry["spec_test"]}, "spec_test.py")
        if "SPEC_PASS" not in out:
            result["stage"] = "spec_broken"
            return result
    except Exception as e:
        result["stage"] = f"spec_check_error: {str(e)[:100]}"
        return result

    result["passed"] = True
    result["stage"] = "passed"
    return result


# ─── Local entrypoint ──────────────────────────────────────
@modal_app.local_entrypoint()
def main(sanity_check: bool = False, model: str = None, n: int = None,
         missing_only: bool = False,
         dataset_path: str = "dataset.jsonl",
         results_path: str = "results/eval_results.jsonl",
         summary_path: str = "results/leaderboard.json"):
    # dataset_path / results_path / summary_path let you run against a staging
    # copy (e.g. --dataset-path dataset_1.jsonl --results-path eval_results_1.jsonl
    # --summary-path eval_summary_1.json) without touching the live files.
    with open(dataset_path) as f:
        dataset = [json.loads(line) for line in f]

    if sanity_check:
        dataset = dataset[:5]
        models = [MODELS_TO_EVAL[0]]
        print(f"SANITY CHECK: {models[0]} × {len(dataset)} entries (~$0.10)\n")
    else:
        if n:
            dataset = dataset[:n]
        models = [model] if model else MODELS_TO_EVAL
        print(f"FULL EVAL: {len(models)} models × {len(dataset)} entries = "
              f"{len(models) * len(dataset)} tasks\n")

    tasks = [(m, e) for m in models for e in dataset]

    if missing_only and os.path.exists(results_path):
        existing_keys = set()
        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                existing_keys.add((r["model"], r["entry_id"]))
        before = len(tasks)
        tasks = [(m, e) for (m, e) in tasks if (m, e["id"]) not in existing_keys]
        print(f"MISSING ONLY: {len(tasks)}/{before} model-entry pairs still need eval\n")

    if not tasks:
        print("No eval tasks to run.")
        return

    # CRITICAL: open the timestamped per-run file BEFORE the loop and write
    # each result as it arrives. This way a mid-run crash doesn't lose data.
    from datetime import datetime as _dt, timezone as _tz
    import os as _os
    _os.makedirs("eval_runs", exist_ok=True)
    run_id = _dt.now().strftime("%Y%m%d_%H%M%S")
    # Provenance: two people produce data in parallel, so every run file name and
    # every record carries who made it. Set MUENCE_PRODUCER in your shell
    # (e.g. `export MUENCE_PRODUCER=jenish`). Filenames stay collision-free, which
    # is what lets both of us append without ever touching a shared file.
    producer = _os.environ.get("MUENCE_PRODUCER", "unknown")
    run_path = f"eval_runs/eval_{producer}_{run_id}.jsonl"
    started_at = _dt.now(_tz.utc).isoformat()
    print(f"Streaming results to {run_path}  (producer={producer})\n")

    results = []
    run_fh = open(run_path, "w", buffering=1)  # line-buffered = flush per line
    try:
        for r in score_one.map(tasks, order_outputs=False):
            r["run_id"] = run_id
            r["producer"] = producer
            r["created_at"] = started_at
            results.append(r)
            # Write each result immediately so crashes don't lose data
            run_fh.write(json.dumps(r) + "\n")
            run_fh.flush()
            status = "✓" if r["passed"] else "✗"
            m_short = r["model"].split("/")[-1][:30]
            print(f"  {status} {m_short:30s} {r['entry_id']} {r['stage']}")
    except Exception as e:
        print(f"\n⚠️  Loop interrupted: {type(e).__name__}: {str(e)[:100]}")
        print(f"Partial results ({len(results)} so far) already saved to {run_path}")
        print(f"Continuing to merge what we have...\n")
    finally:
        run_fh.close()

    # Aggregate
    from collections import Counter, defaultdict
    by_model = defaultdict(lambda: {"passed": 0, "total": 0, "by_class": Counter(),
                                     "fail_reasons": Counter()})
    for r in results:
        m = r["model"]
        by_model[m]["total"] += 1
        if r["passed"]:
            by_model[m]["passed"] += 1
            by_model[m]["by_class"][r["vuln_class"]] += 1
        else:
            by_model[m]["fail_reasons"][r["stage"]] += 1

    print("\n" + "=" * 70)
    print("LEADERBOARD")
    print("=" * 70)
    for m, s in sorted(by_model.items(), key=lambda x: -x[1]["passed"] / max(x[1]["total"], 1)):
        pct = 100 * s["passed"] / s["total"] if s["total"] else 0
        bar = "█" * int(40 * pct / 100)
        print(f"  {m:45s}  {s['passed']:3d}/{s['total']:3d}  {pct:5.1f}%  {bar}")

    # APPEND to the rolling eval_results.jsonl (dedupe by model+entry, latest wins)
    existing: dict = {}
    if _os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                r = json.loads(line)
                existing[(r["model"], r["entry_id"])] = r
    for r in results:
        existing[(r["model"], r["entry_id"])] = r

    with open(results_path, "w") as f:
        for r in existing.values():
            f.write(json.dumps(r) + "\n")

    # Summary — rebuilt from the FULL merged set, not just this run
    merged_results = list(existing.values())
    merged_by_model: dict = {}
    for r in merged_results:
        m = r["model"]
        if m not in merged_by_model:
            merged_by_model[m] = {"passed": 0, "total": 0, "by_class": Counter(),
                                  "fail_reasons": Counter()}
        merged_by_model[m]["total"] += 1
        if r["passed"]:
            merged_by_model[m]["passed"] += 1
            merged_by_model[m]["by_class"][r["vuln_class"]] += 1
        else:
            merged_by_model[m]["fail_reasons"][r["stage"]] += 1

    summary = {m: {"passed": s["passed"], "total": s["total"],
                   "by_class": dict(s["by_class"]),
                   "fail_reasons": dict(s["fail_reasons"])}
               for m, s in merged_by_model.items()}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved this run to:        {run_path}")
    print(f"Merged {results_path} now has {len(existing)} unique (model,entry) rows")
    print(f"Updated {summary_path} with the FULL leaderboard:")
    for m, s in sorted(merged_by_model.items(), key=lambda x: -x[1]["passed"] / max(x[1]["total"], 1)):
        pct = 100 * s["passed"] / s["total"] if s["total"] else 0
        print(f"  {m:50s}  {s['passed']:3d}/{s['total']:3d}  {pct:.1f}%")
