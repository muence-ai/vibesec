"""
VibeSec public benchmark dashboard.

Run:
  streamlit run viewer/dashboard.py
"""

from __future__ import annotations

import difflib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st


APP_ROOT = Path(__file__).resolve().parent.parent  # viewer/ -> repo root
DATASET_PATH = APP_ROOT / "dataset.jsonl"
EVAL_RESULTS_PATH = APP_ROOT / "results" / "eval_outcomes.jsonl"
FEEDBACK_PATH = APP_ROOT / "community_feedback.jsonl"

# Writing feedback to local disk does NOT persist on ephemeral hosts like
# Streamlit Community Cloud or Hugging Face Spaces (the filesystem is wiped on
# redeploy / sleep). So writes are OFF by default — safe for hosting — and only
# enabled when VIBESEC_ENABLE_FEEDBACK is truthy (e.g. local dev or a deploy
# backed by persistent storage).
FEEDBACK_ENABLED = os.environ.get("VIBESEC_ENABLE_FEEDBACK", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REPO_URL = "https://github.com/jenishk20/vibesec-evals"
FORK_URL = f"{REPO_URL}/fork"
ISSUES_URL = f"{REPO_URL}/issues/new"
HF_DATASET_URL = "https://huggingface.co/datasets/muence/vibesec"
EXCLUDED_MODELS = {"google/gemini-2.5-pro"}
MODEL_LABELS = {
    "anthropic/claude-opus-4-8": "Claude Opus 4.8",
    "anthropic/claude-sonnet-4.6": "Claude Sonnet 4.6",
    "wandb/nemotron-3-ultra": "Nemotron 3 Ultra",
    "wandb/kimi-k2.7-code": "Kimi K2.7 Code",
    "wandb/glm-5.2": "GLM 5.2",
    "wandb/gpt-oss-120b": "GPT-OSS 120B",
}


st.set_page_config(
    page_title="VibeSec Benchmark",
    page_icon="VS",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --bg: #f7f8fa;
        --ink: #17202a;
        --muted: #687386;
        --line: #dfe4ea;
        --panel: #ffffff;
        --red: #d64545;
        --teal: #087f8c;
        --amber: #b7791f;
        --blue: #2857a3;
        --green: #1f7a4d;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 4rem;
        max-width: 1280px;
    }

    div[data-testid="stMetric"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        box-shadow: 0 1px 2px rgba(23, 32, 42, 0.04);
    }

    div[data-testid="stMetric"] label {
        color: var(--muted);
    }

    .hero {
        border-bottom: 1px solid var(--line);
        padding: 0.25rem 0 1.1rem 0;
        margin-bottom: 1rem;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2rem, 5vw, 3.8rem);
        line-height: 0.98;
        letter-spacing: 0;
        color: var(--ink);
    }

    .hero p {
        margin: 0.75rem 0 0 0;
        max-width: 900px;
        color: var(--muted);
        font-size: 1.08rem;
        line-height: 1.55;
    }

    .pill-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.28rem 0.65rem;
        background: #ffffff;
        color: #344054;
        font-size: 0.84rem;
        white-space: nowrap;
    }

    .section-title {
        margin: 1.5rem 0 0.65rem 0;
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--ink);
    }

    .callout {
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        border-radius: 8px;
        background: #ffffff;
        padding: 0.85rem 1rem;
        color: #2d3748;
        margin: 0.4rem 0 1rem 0;
    }

    .gate-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.75rem 0 1rem 0;
    }

    .gate {
        border: 1px solid var(--line);
        background: #ffffff;
        border-radius: 8px;
        padding: 0.85rem;
        min-height: 104px;
    }

    .gate strong {
        display: block;
        color: var(--ink);
        margin-bottom: 0.35rem;
    }

    .gate span {
        display: block;
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.35;
    }

    .task-header {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        background: #ffffff;
        margin-bottom: 0.8rem;
    }

    .task-id {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    .prompt {
        font-size: 1.08rem;
        line-height: 1.45;
        color: var(--ink);
    }

    .status-pass {
        color: var(--green);
        font-weight: 700;
    }

    .status-fail {
        color: var(--red);
        font-weight: 700;
    }

    .status-neutral {
        color: var(--amber);
        font-weight: 700;
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.9rem;
    }

    @media (max-width: 900px) {
        .gate-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 640px) {
        .gate-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_jsonl(path: str, mtime: float) -> list[dict]:
    # `mtime` is part of the cache key: when the file changes on disk, the key
    # changes and the file is re-read. Without it, @st.cache_data keys only on
    # `path` and a long-running server keeps serving a stale (e.g. empty) copy.
    rows = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_jsonl(path: str) -> list[dict]:
    source = Path(path)
    if not source.exists():
        return []
    return _load_jsonl(str(source), source.stat().st_mtime)


@st.cache_data(show_spinner=False)
def load_feedback(path: str) -> list[dict]:
    return load_jsonl(path)


def save_feedback(payload: dict) -> None:
    # Defense in depth: never touch disk when writes are disabled, even if a
    # caller forgets to guard the UI control.
    if not FEEDBACK_ENABLED:
        return
    payload = {
        **payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with FEEDBACK_PATH.open("a") as f:
        f.write(json.dumps(payload) + "\n")
    st.cache_data.clear()


def pct(part: int, whole: int) -> float:
    return round(100 * part / whole, 1) if whole else 0.0


def stage_label(stage: str) -> str:
    return {
        "passed": "passed",
        "exploit_still_works": "exploit still works",
        "spec_broken": "spec broken",
        "parse_failed": "format failed",
        "no_response_after_retries": "no response",
    }.get(stage, stage.replace("_", " "))


def model_label(model: str) -> str:
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    return model.split("/")[-1].replace("-", " ").title()


def class_color(vuln_class: str) -> str:
    return {
        "idor": "#d64545",
        "missing_auth": "#087f8c",
        "mass_assignment": "#b7791f",
        "privilege_escalation": "#8a4fff",
        "path_traversal": "#1f7a4d",
        "sql_injection": "#2857a3",
        "other": "#687386",
    }.get(vuln_class, "#687386")


def pill(text: str, color: str = "#687386") -> str:
    return (
        f'<span class="pill" style="border-color:{color}33;'
        f'color:{color};background:{color}10">{text}</span>'
    )


def format_files(files: dict[str, str]) -> str:
    parts = []
    for path, content in files.items():
        parts.append(f"# {path}\n{content}")
    return "\n\n".join(parts)


def patch_diff(entry: dict) -> str:
    before = entry.get("app_files", {}).get("main.py", "").splitlines()
    after = entry.get("patched_files", {}).get("main.py", "").splitlines()
    return "\n".join(
        difflib.unified_diff(
            before,
            after,
            fromfile="vulnerable/main.py",
            tofile="patched/main.py",
            lineterm="",
        )
    )


def leaderboard_rows(results: list[dict]) -> pd.DataFrame:
    by_model = defaultdict(lambda: {"passed": 0, "total": 0, "stages": Counter()})
    for result in results:
        row = by_model[result["model"]]
        row["total"] += 1
        row["stages"][result["stage"]] += 1
        if result.get("passed"):
            row["passed"] += 1

    rows = []
    for model, stats in by_model.items():
        total = stats["total"]
        rows.append(
            {
                "Model": model_label(model),
                "Pass rate": pct(stats["passed"], total),
                "Passed": stats["passed"],
                "Total": total,
                "Exploit still works": stats["stages"].get("exploit_still_works", 0),
                "Spec broken": stats["stages"].get("spec_broken", 0),
                "Format failed": stats["stages"].get("parse_failed", 0),
                "No response": stats["stages"].get("no_response_after_retries", 0),
            }
        )
    return pd.DataFrame(rows).sort_values("Pass rate", ascending=False)


def feedback_counts(feedback: list[dict]) -> dict[str, Counter]:
    counts = defaultdict(Counter)
    for item in feedback:
        entry_id = item.get("entry_id")
        action = item.get("action")
        if entry_id and action:
            counts[entry_id][action] += 1
    return counts


dataset = load_jsonl(str(DATASET_PATH))
raw_results = load_jsonl(str(EVAL_RESULTS_PATH))
results = [result for result in raw_results if result.get("model") not in EXCLUDED_MODELS]
feedback = load_feedback(str(FEEDBACK_PATH))
feedback_by_entry = feedback_counts(feedback)

if not dataset:
    st.error(f"{DATASET_PATH} was not found. Build the dataset before opening the dashboard.")
    st.stop()

entry_by_id = {entry["id"]: entry for entry in dataset}
results_by_entry = defaultdict(list)
for result in results:
    results_by_entry[result["entry_id"]].append(result)

query_task = st.query_params.get("task")
if isinstance(query_task, list):
    query_task = query_task[0] if query_task else None


st.sidebar.markdown("### VibeSec")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Leaderboard", "Task Explorer", "Methodology", "Submit"],
    label_visibility="collapsed",
)

st.sidebar.divider()
st.sidebar.caption("Dataset")
st.sidebar.write(f"{len(dataset)} verified tasks")
st.sidebar.write(f"{len({entry['seed_prompt'] for entry in dataset})} distinct prompts")
st.sidebar.write(f"{len({entry['vuln_class'] for entry in dataset})} vulnerability classes")

if results:
    st.sidebar.caption("Evaluations")
    st.sidebar.write(f"{len({result['model'] for result in results})} models")
    st.sidebar.write(f"{len(results)} model-task runs")

st.sidebar.divider()
st.sidebar.link_button("Star on GitHub", REPO_URL, width="stretch")
st.sidebar.link_button("Fork benchmark", FORK_URL, width="stretch")
st.sidebar.link_button("Dataset on Hugging Face", HF_DATASET_URL, width="stretch")


def render_github_actions() -> None:
    left, middle, right, _ = st.columns([1, 1, 1, 3])
    left.link_button("Star GitHub", REPO_URL, width="stretch")
    middle.link_button("Fork repo", FORK_URL, width="stretch")
    right.link_button("Open issue", ISSUES_URL, width="stretch")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>VibeSec Benchmark</h1>
          <p>
            Execution-verified security evals for AI coding agents. Every task
            contains a generated app, a passing product spec, a working exploit,
            and a reference patch verified in a sandbox.
          </p>
          <div class="pill-row">
            <span class="pill">No LLM-as-judge scoring</span>
            <span class="pill">Exploit-proven vulnerabilities</span>
            <span class="pill">Patch must preserve normal behavior</span>
            <span class="pill">Built for AI coding model regression tests</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_github_actions()


def render_overview() -> None:
    render_hero()

    class_dist = Counter(entry["vuln_class"] for entry in dataset)
    verified = len(dataset)
    models = len({result["model"] for result in results}) if results else 0
    passes = sum(1 for result in results if result.get("passed"))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Verified tasks", f"{verified}")
    col2.metric("Model runs", f"{len(results)}")
    col3.metric("Models evaluated", f"{models}")
    col4.metric("Passes observed", f"{passes}")

    st.markdown('<div class="section-title">Why This Exists</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="callout">
        AI coding agents can ship code that looks reasonable and still leaks data.
        VibeSec scores whether a model can patch an exploit-proven vulnerability
        without breaking the app's intended behavior.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="callout">
        VibeSec Open V1 intentionally exposes these tasks for inspection. Official
        leaderboard scores should use a separate private held-out set that model
        builders cannot tune against.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Verification Gates</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="gate-grid">
          <div class="gate"><strong>1. Spec passes</strong><span>The generated app boots and satisfies its intended golden path.</span></div>
          <div class="gate"><strong>2. Exploit lands</strong><span>A real script prints PWNED only after reading or changing data it should not access.</span></div>
          <div class="gate"><strong>3. Patch blocks exploit</strong><span>The same exploit must fail against the patched app.</span></div>
          <div class="gate"><strong>4. Spec still passes</strong><span>The patch must keep normal product behavior working.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="section-title">Vulnerability Mix</div>', unsafe_allow_html=True)
        class_df = pd.DataFrame(
            [
                {"Class": vuln_class, "Tasks": count, "Share": pct(count, verified)}
                for vuln_class, count in class_dist.most_common()
            ]
        )
        st.dataframe(class_df, hide_index=True, width="stretch")

    with right:
        st.markdown('<div class="section-title">Current Leaderboard</div>', unsafe_allow_html=True)
        if results:
            top = leaderboard_rows(results).head(7)
            st.dataframe(top, hide_index=True, width="stretch")
        else:
            st.info("The V1 evaluation has not been published yet.")

    st.markdown('<div class="section-title">Representative Tasks</div>', unsafe_allow_html=True)
    sample_cols = st.columns(3)
    for idx, entry in enumerate(dataset[:3]):
        with sample_cols[idx]:
            counts = feedback_by_entry.get(entry["id"], Counter())
            st.markdown(
                f"""
                <div class="task-header">
                  <div class="task-id">{entry["id"]}</div>
                  <div class="pill-row">{pill(entry["vuln_class"], class_color(entry["vuln_class"]))}</div>
                  <p class="small-muted">{entry["seed_prompt"][:210]}</p>
                  <p class="small-muted">{counts.get("upvote", 0)} upvotes</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_leaderboard() -> None:
    st.title("Leaderboard")
    st.caption("A pass means the model blocked the exploit and preserved the spec.")

    if not results:
        st.warning("No eval results found yet.")
        return

    board = leaderboard_rows(results)

    st.dataframe(
        board,
        hide_index=True,
        width="stretch",
        column_config={
            "Pass rate": st.column_config.ProgressColumn(
                "Pass rate",
                min_value=0,
                max_value=100,
                format="%.1f%%",
            )
        },
    )

    st.markdown('<div class="section-title">Failure Breakdown</div>', unsafe_allow_html=True)
    stage_counts = Counter(result["stage"] for result in results)
    stage_df = pd.DataFrame(
        [
            {"Stage": stage_label(stage), "Count": count, "Share": pct(count, len(results))}
            for stage, count in stage_counts.most_common()
        ]
    )
    st.dataframe(stage_df, hide_index=True, width="stretch")

    st.markdown('<div class="section-title">Per-Class Pass Rates</div>', unsafe_allow_html=True)
    by_model_class = defaultdict(lambda: defaultdict(lambda: {"pass": 0, "total": 0}))
    for result in results:
        cell = by_model_class[result["model"]][result["vuln_class"]]
        cell["total"] += 1
        if result.get("passed"):
            cell["pass"] += 1

    classes = sorted({result["vuln_class"] for result in results})
    rows = []
    for model in sorted(by_model_class):
        row = {"Model": model_label(model)}
        for vuln_class in classes:
            cell = by_model_class[model].get(vuln_class, {"pass": 0, "total": 0})
            row[vuln_class] = (
                f"{cell['pass']}/{cell['total']} ({pct(cell['pass'], cell['total']):.1f}%)"
                if cell["total"]
                else "-"
            )
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_task_feedback(entry: dict) -> None:
    counts = feedback_by_entry.get(entry["id"], Counter())
    st.markdown('<div class="section-title">Community Signals</div>', unsafe_allow_html=True)

    if not FEEDBACK_ENABLED:
        st.caption(
            "Community actions are read-only on this hosted demo — local writes "
            "would be wiped on redeploy. Open a GitHub issue to contribute, or run "
            "locally with `VIBESEC_ENABLE_FEEDBACK=1` to record feedback."
        )

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 3])
    with c1:
        if st.button(f"Upvote ({counts.get('upvote', 0)})", width="stretch", disabled=not FEEDBACK_ENABLED):
            save_feedback({"entry_id": entry["id"], "action": "upvote"})
            st.rerun()
    with c2:
        if st.button(f"Useful ({counts.get('useful', 0)})", width="stretch", disabled=not FEEDBACK_ENABLED):
            save_feedback({"entry_id": entry["id"], "action": "useful"})
            st.rerun()
    with c3:
        if st.button(f"Needs review ({counts.get('needs_review', 0)})", width="stretch", disabled=not FEEDBACK_ENABLED):
            save_feedback({"entry_id": entry["id"], "action": "needs_review"})
            st.rerun()
    with c4:
        st.link_button("GitHub", REPO_URL, width="stretch")
    with c5:
        share_url = f"?task={quote(entry['id'])}"
        st.text_input("Share link", value=share_url, label_visibility="collapsed")

    with st.form(f"comment-{entry['id']}", clear_on_submit=True):
        comment = st.text_area(
            "Comment",
            placeholder="What makes this task interesting or broken?",
            disabled=not FEEDBACK_ENABLED,
        )
        submitted = st.form_submit_button("Add comment", disabled=not FEEDBACK_ENABLED)
        if submitted and comment.strip():
            save_feedback({"entry_id": entry["id"], "action": "comment", "comment": comment.strip()})
            st.success("Comment saved locally.")
            st.rerun()

    comments = [
        item for item in feedback
        if item.get("entry_id") == entry["id"] and item.get("action") == "comment"
    ]
    if comments:
        for item in comments[-5:]:
            st.markdown(f"> {item.get('comment', '').strip()}")


def render_task_explorer() -> None:
    st.title("Task Explorer")
    st.caption("Inspect the exact prompt, vulnerable app, exploit, reference patch, and model attempts.")

    classes = sorted({entry["vuln_class"] for entry in dataset})
    col1, col2, col3 = st.columns([1, 1, 2])
    selected_classes = col1.multiselect("Vulnerability class", classes, default=classes)
    result_filter = col2.selectbox(
        "Model result",
        ["Any", "Has model failure", "Has model pass", "No eval result"],
    )
    search = col3.text_input("Search prompts or code", "")

    filtered = []
    for entry in dataset:
        if entry["vuln_class"] not in selected_classes:
            continue
        haystack = " ".join(
            [
                entry.get("seed_prompt", ""),
                format_files(entry.get("app_files", {})),
                entry.get("exploit", ""),
            ]
        ).lower()
        if search and search.lower() not in haystack:
            continue
        entry_results = results_by_entry.get(entry["id"], [])
        if result_filter == "Has model failure" and not any(not r.get("passed") for r in entry_results):
            continue
        if result_filter == "Has model pass" and not any(r.get("passed") for r in entry_results):
            continue
        if result_filter == "No eval result" and entry_results:
            continue
        filtered.append(entry)

    if not filtered:
        st.warning("No tasks match those filters.")
        return

    default_index = 0
    if query_task in {entry["id"] for entry in filtered}:
        default_index = [entry["id"] for entry in filtered].index(query_task)

    selected_id = st.selectbox(
        "Task",
        options=[entry["id"] for entry in filtered],
        index=default_index,
        format_func=lambda entry_id: (
            f"{entry_id[:8]} | {entry_by_id[entry_id]['vuln_class']} | "
            f"{entry_by_id[entry_id]['seed_prompt'][:110]}"
        ),
    )
    st.query_params["task"] = selected_id
    entry = entry_by_id[selected_id]

    st.markdown(
        f"""
        <div class="task-header">
          <div class="task-id">Task {entry["id"]}</div>
          <div class="pill-row">
            {pill(entry["vuln_class"], class_color(entry["vuln_class"]))}
            {pill(f'{len(results_by_entry.get(entry["id"], []))} model attempts', '#2857a3')}
            {pill(f'{feedback_by_entry.get(entry["id"], Counter()).get("upvote", 0)} upvotes', '#1f7a4d')}
          </div>
          <div class="prompt">{entry["seed_prompt"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Task Anatomy</div>', unsafe_allow_html=True)
    tab_app, tab_spec, tab_exploit, tab_patch, tab_diff = st.tabs(
        ["Vulnerable App", "Spec Test", "Exploit", "Reference Patch", "Patch Diff"]
    )
    with tab_app:
        for path, content in entry.get("app_files", {}).items():
            st.markdown(f"**{path}**")
            st.code(content, language="python" if path.endswith(".py") else "text")
    with tab_spec:
        st.code(entry.get("spec_test", ""), language="python")
    with tab_exploit:
        st.code(entry.get("exploit", ""), language="python")
        if entry.get("exploit_output"):
            st.markdown("**Captured exploit output**")
            st.code(entry["exploit_output"], language="text")
    with tab_patch:
        for path, content in entry.get("patched_files", {}).items():
            st.markdown(f"**{path}**")
            st.code(content, language="python" if path.endswith(".py") else "text")
    with tab_diff:
        diff = patch_diff(entry)
        st.code(diff if diff else "No diff available.", language="diff")

    st.markdown('<div class="section-title">Model Attempts On This Task</div>', unsafe_allow_html=True)
    entry_results = results_by_entry.get(entry["id"], [])
    if entry_results:
        rows = []
        for result in sorted(entry_results, key=lambda r: (not r.get("passed"), r["model"])):
            rows.append(
                {
                    "Model": model_label(result["model"]),
                    "Outcome": "pass" if result.get("passed") else "fail",
                    "Stage": stage_label(result["stage"]),
                }
            )
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

        selected_result = st.selectbox(
            "Inspect model response",
            options=[
                f"{model_label(r['model'])} | {stage_label(r['stage'])}"
                for r in entry_results
            ],
        )
        selected_index = [
            f"{model_label(r['model'])} | {stage_label(r['stage'])}"
            for r in entry_results
        ].index(selected_result)
        model_result = entry_results[selected_index]
        response_tab, parsed_tab = st.tabs(["Raw Response", "Parsed Patch"])
        with response_tab:
            st.code(model_result.get("raw_response",
                    "Raw model responses are not published in this repository "
                    "(52MB). The per-task outcome, vulnerability class and failure "
                    "stage are in results/eval_outcomes.jsonl."), language="text")
        with parsed_tab:
            patched = model_result.get("patched_files", {})
            if patched:
                for path, content in patched.items():
                    st.markdown(f"**{path}**")
                    st.code(content, language="python" if path.endswith(".py") else "text")
            else:
                st.info("No parsed patch was available for this attempt.")
    else:
        st.info("No model attempts recorded for this task yet.")

    render_task_feedback(entry)


def render_methodology() -> None:
    st.title("Methodology")
    st.caption("What the benchmark measures, how tasks enter the dataset, and where it is still limited.")

    st.markdown('<div class="section-title">Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. A founder-style prompt is used to generate a small FastAPI app.
        2. A spec test is generated to capture intended normal behavior.
        3. A sandbox confirms the app boots and the spec passes.
        4. An attacker model writes an exploit script.
        5. A sandbox confirms the exploit prints `PWNED`.
        6. A fixer model writes a reference patch.
        7. A sandbox confirms the exploit no longer works.
        8. A sandbox confirms the original spec still passes.
        """
    )

    st.markdown('<div class="section-title">Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        """
        A model passes a task only if its patch blocks the exact recorded exploit
        and the original golden-path spec still passes. Failures are separated into
        exploitable patches, behavior-breaking patches, format failures, and empty
        or timed-out model responses.
        """
    )

    st.markdown('<div class="section-title">Current Limitations</div>', unsafe_allow_html=True)
    st.markdown(
        """
        - The current dataset is synthetic and FastAPI-focused.
        - Most current tasks are IDOR-style authorization failures.
        - Vulnerability labels are heuristic and should be refined before a larger launch.
        - Some failures measure output formatting rather than pure security ability.
        - Public examples are inspectable, so a serious leaderboard should use a held-out set.
        """
    )

    st.markdown('<div class="section-title">Public vs Held-Out Tasks</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Show enough public tasks for researchers to trust the benchmark and inspect
        failure modes. Keep a separate held-out set for official leaderboard scoring
        so model builders cannot tune directly against every task.
        """
    )

    st.markdown('<div class="section-title">Reproducibility</div>', unsafe_allow_html=True)
    st.code(
        """python -m venv myenv && source myenv/bin/activate
pip install -r requirements.txt
modal run eval/eval_models.py --dataset-path dataset.jsonl --results-path results/eval_results.jsonl --sanity-check
streamlit run viewer/dashboard.py""",
        language="bash",
    )


def render_submit() -> None:
    st.title("Submit Or Share")
    st.caption("VibeSec keeps execution controlled by the benchmark owner. Public arbitrary execution can come later.")

    st.markdown('<div class="section-title">For Model Builders</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Submit patched outputs for the public dataset as JSONL, or open an issue with
        model name, date, decoding settings, and raw responses. Official scores should
        be rerun through the benchmark harness so the leaderboard stays trusted.
        """
    )

    st.markdown('<div class="section-title">For Security Researchers</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Comment on tasks that look mislabeled, too easy, unrealistic, or especially
        useful. The fastest way to improve VibeSec is to turn public review into a
        better held-out task set.
        """
    )

    render_github_actions()

    st.markdown('<div class="section-title">Suggested Submission Schema</div>', unsafe_allow_html=True)
    st.code(
        """{"model": "provider/model-name",
 "entry_id": "task id from dataset.jsonl",
 "raw_response": "<file path=\\"main.py\\">...</file>",
 "metadata": {"temperature": 0.2, "date": "2026-06-29"}}""",
        language="json",
    )

    st.markdown('<div class="section-title">Why There Is No Public Run Button Yet</div>', unsafe_allow_html=True)
    st.markdown(
        """
        Running AI-generated apps and exploits is powerful but easy to abuse. A safe
        public execution service needs authentication, queueing, spend limits, sandbox
        hardening, request logging, timeouts, and moderation. For the public release, inspectability
        plus official reruns gives you most of the trust with much less operational risk.
        """
    )


if page == "Overview":
    render_overview()
elif page == "Leaderboard":
    render_leaderboard()
elif page == "Task Explorer":
    render_task_explorer()
elif page == "Methodology":
    render_methodology()
else:
    render_submit()
