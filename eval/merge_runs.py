"""
merge_runs.py — fold append-only per-run eval logs into canonical state.

Two people produce data in parallel. The rule that makes that safe: nobody ever
edits a shared file. Each run writes its own immutable
`eval_runs/eval_<producer>_<ts>.jsonl`, and this script is the ONLY thing that
writes `eval_results.jsonl` / `eval_summary.json`.

Three things this exists to prevent, all of which already bit us:

1. **Mixing dataset versions.** Run history contained both V0 (634 tasks) and V1
   (1000 tasks) evals, which merged blindly gave three different Opus numbers
   (70.3% / 64.9% / 61.1%). `--dataset` scopes every merge to one version.
2. **Losing rows.** `eval_runs/` is NOT complete history — per-run streaming was
   added after some evals ran, so results/eval_outcomes.jsonl has 1000 Opus rows where
   the run files have 714. `--seed` layers the existing merged artifact underneath
   the run files. Never rebuild from `eval_runs/` alone.
3. **Scoring a rule violation as a security failure.** See `requirements_changed`
   in eval_models.py; `summarize()` reports coverage separately from pass rate.

Dedupe key is (model, entry_id), matching the eval harness. Newest wins, ordered by
run_id then file mtime, because a re-run is normally a fix. Seeded rows sit at the
bottom of that order, so any real run supersedes them.

Usage
  python merge_runs.py                          # rebuild canonical files
  python merge_runs.py --dry-run                # report only, write nothing
  python merge_runs.py --producer suprav        # fold in one person's runs only
  python merge_runs.py --dataset ''             # disable version scoping
  python merge_runs.py --out releases/v1.1.0/   # freeze a release
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import sys

KEY = ("model", "entry_id")


def _run_sort_key(path: str) -> tuple:
    """Newest last. run_id (…_YYYYmmdd_HHMMSS) sorts lexicographically = chronologically."""
    stem = pathlib.Path(path).stem
    parts = stem.split("_")
    ts = "_".join(parts[-2:]) if len(parts) >= 2 else ""
    return (ts, os.path.getmtime(path))


def load_dataset_ids(path: str | None) -> set[str] | None:
    """Entry IDs of one dataset version. Runs are scoped to this so a V0 eval and a
    V1 eval never merge into the same leaderboard — the failure that produced three
    different Opus numbers (70.3% / 64.9% / 61.1%) from the same run history."""
    if not path:
        return None
    ids = set()
    with open(path) as fh:
        for line in fh:
            if line.strip():
                ids.add(json.loads(line)["id"])
    print(f"Scoping to {len(ids)} entry ids from {path}")
    return ids


def load_runs(
    pattern: str,
    producer: str | None,
    dataset_ids: set[str] | None = None,
    exclude_models: set[str] | None = None,
    seed: str | None = None,
) -> tuple[dict, collections.Counter, int]:
    paths = sorted(glob.glob(pattern), key=_run_sort_key)
    if producer:
        paths = [p for p in paths if f"_{producer}_" in pathlib.Path(p).name]
    exclude_models = exclude_models or set()
    out_of_scope = 0
    excluded = 0

    merged: dict[tuple, dict] = {}
    by_producer: collections.Counter = collections.Counter()
    total = 0
    superseded = 0

    # Seed layer. eval_runs/ is NOT complete history — per-run streaming was added
    # after some evals had already been done, so e.g. the v1.0.0 release holds
    # 1000 Opus rows while the run files hold 714. Seeding from the existing merged
    # artifact and overlaying runs on top preserves everything; rebuilding from runs
    # alone would silently drop those rows.
    if seed and os.path.exists(seed):
        n = 0
        with open(seed) as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if not all(k in rec for k in KEY):
                    continue
                if rec["model"] in (exclude_models or set()):
                    continue
                if dataset_ids is not None and rec["entry_id"] not in dataset_ids:
                    continue
                rec.setdefault("producer", "seed")
                rec.setdefault("run_id", "seed")
                merged[tuple(rec[f] for f in KEY)] = rec
                by_producer[rec["producer"]] += 1
                n += 1
        print(f"Seeded {n} rows from {seed}")

    for path in paths:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  ! skipping malformed line in {path}", file=sys.stderr)
                    continue
                if not all(k in rec for k in KEY):
                    continue
                if rec["model"] in exclude_models:
                    excluded += 1
                    continue
                # Retroactive version scoping: records predate the dataset_version
                # field, but an entry_id that isn't in this dataset came from another
                # version, which is exactly what we must not merge in.
                if dataset_ids is not None and rec["entry_id"] not in dataset_ids:
                    out_of_scope += 1
                    continue
                total += 1
                k = tuple(rec[f] for f in KEY)
                if k in merged:
                    superseded += 1
                # Backfill provenance for records written before it existed, so old
                # and new runs can be merged without special-casing.
                rec.setdefault("producer", "unknown")
                rec.setdefault("run_id", "_".join(pathlib.Path(path).stem.split("_")[-2:]))
                merged[k] = rec
                by_producer[rec["producer"]] += 1

    print(f"Read {total} in-scope records from {len(paths)} run file(s); "
          f"{superseded} superseded by a later run.")
    if out_of_scope:
        print(f"  skipped {out_of_scope} records from other dataset versions")
    if excluded:
        print(f"  skipped {excluded} records from excluded models")
    return merged, by_producer, total


def summarize(merged: dict) -> dict:
    """Per-model leaderboard. Reports coverage separately from pass rate so a model
    disqualified on a technicality is never silently scored as insecure."""
    agg: dict[str, dict] = {}
    for rec in merged.values():
        m = rec["model"]
        a = agg.setdefault(m, {
            "passed": 0, "total": 0, "requirements_changed": 0,
            "by_class": collections.Counter(), "stages": collections.Counter(),
        })
        a["total"] += 1
        a["stages"][rec.get("stage", "?")] += 1
        if rec.get("requirements_changed"):
            a["requirements_changed"] += 1
        if rec.get("passed"):
            a["passed"] += 1
            a["by_class"][rec.get("vuln_class", "?")] += 1

    out = {}
    for m, a in agg.items():
        # "Scored" = reached a real security verdict. Harness-level aborts
        # (parse_failed, requirements_missing, *_error) are not security signal.
        aborted = sum(
            c for s, c in a["stages"].items()
            if s in ("parse_failed", "requirements_missing")
            or s.endswith("_error") or s.startswith("llm_error")
        )
        scored = a["total"] - aborted
        out[m] = {
            "passed": a["passed"],
            "total": a["total"],
            "scored": scored,
            "pass_rate": round(a["passed"] / a["total"], 4) if a["total"] else 0.0,
            "pass_rate_scored": round(a["passed"] / scored, 4) if scored else 0.0,
            "requirements_changed": a["requirements_changed"],
            "by_class": dict(a["by_class"]),
            "stages": dict(a["stages"]),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="eval_runs/eval_*.jsonl")
    ap.add_argument("--producer", default=None, help="only fold in this producer's runs")
    ap.add_argument("--out", default="results", help="output dir for merged results + summary")
    ap.add_argument("--dataset", default="dataset.jsonl",
                    help="dataset file defining the version scope; '' to disable")
    ap.add_argument("--exclude-models", default="base,sft",
                    help="comma-separated models to keep off the leaderboard "
                         "(SFT experiment arms live in muence-research, not here)")
    ap.add_argument("--seed", default="results/eval_outcomes.jsonl",
                    help="existing merged file to seed from before overlaying runs; '' to disable")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dataset_ids = load_dataset_ids(args.dataset or None)
    exclude = {m.strip() for m in args.exclude_models.split(",") if m.strip()}
    merged, by_producer, _ = load_runs(args.pattern, args.producer, dataset_ids, exclude,
                                       args.seed or None)
    if not merged:
        print("No records found — nothing to merge.")
        return

    summary = summarize(merged)

    print(f"\nUnique (model, entry_id) rows: {len(merged)}")
    print("By producer: " + ", ".join(f"{p}={n}" for p, n in by_producer.most_common()))
    print("\nLEADERBOARD" + " " * 12 + "pass/total   rate   scored-rate  reqs-changed")
    print("=" * 78)
    for m, s in sorted(summary.items(), key=lambda kv: -kv[1]["pass_rate_scored"]):
        flag = "  ⚠" if s["requirements_changed"] > 0.1 * s["total"] else ""
        print(f"  {m[:38]:38s} {s['passed']:4d}/{s['total']:<5d} "
              f"{100*s['pass_rate']:5.1f}% {100*s['pass_rate_scored']:9.1f}% "
              f"{s['requirements_changed']:9d}{flag}")
    print("\n⚠ = >10% of attempts changed requirements.txt; check rule compliance "
          "before quoting this model's number.")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    results_path = outdir / "eval_results.jsonl"
    summary_path = outdir / "eval_summary.json"

    with open(results_path, "w") as fh:
        for rec in merged.values():
            fh.write(json.dumps(rec) + "\n")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nWrote {results_path} ({len(merged)} rows)")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
