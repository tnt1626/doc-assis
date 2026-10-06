"""Agent Memory Benchmark Script.

Place this file at `evals/bench_memory.py` and run it from the repository root.

This script acts as a mock browser client, invoking the exact REST API endpoints
used by the web user interface.

Each scenario performs the following steps:
    1. Runs `evals/reset.sh` to clear existing memory state and chat history.
    2. Setup sessions: creates a session -> sends context messages -> DELETES the session
       (deleting a session triggers memory consolidation and persistence).
    3. Creates a new evaluation session -> asks the test question -> evaluates response keywords.
    4. Records structured benchmark result rows into `evals/results/<label>.jsonl`.

Usage:
    # Terminal 1: Start application server logging output:
    uv run uvicorn app.main:app 2>&1 | tee evals/logs/ollama-on.txt

    # Terminal 2: Run benchmark suite:
    uv run python evals/bench_memory.py --label ollama-on \
        --log evals/logs/ollama-on.txt --reps 3

IMPORTANT NOTES:
    - This script does NOT toggle memory or switch LLM providers dynamically;
      MEMORY_ENABLED and LLM_PROVIDER are read by the server at startup.
      The `--label` argument is a tag to identify benchmark result configurations.
    - `--log` is required to capture token counts parsed from `[TOKENS]` log lines.
    - Runs with memory disabled are expected to fail "recall" and "update" scenarios.
"""
import argparse
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
USER_PROFILE = ROOT / ".agent" / "USER.md"

# ---------------------------------------------------------------------------
# SCENARIOS
# setup:      List of setup sessions (messages sent before testing).
#             Each session is deleted at the end to trigger memory consolidation.
# question:   Test question asked in a fresh chat session.
# expect_any: Answer is graded as PASSED if it contains AT LEAST ONE keyword (case-insensitive).
# forbid:     Answer is graded as FAILED if it contains ANY forbidden keyword.
# ---------------------------------------------------------------------------
NOT_KNOW = [
    "không biết", "chưa biết", "không rõ", "chưa nói", "chưa cho mình biết",
    "chưa được cung cấp", "không có thông tin", "chưa có thông tin",
    "bạn chưa", "cho mình biết", "bạn có thể cho",
]

SCENARIOS = [
    {
        "id": "recall_exam", "kind": "recall",
        "setup": [[
            "Mình đang ôn thi môn Cơ sở dữ liệu, thi vào thứ Sáu.",
            "Phần mình thấy khó nhất là chuẩn hóa dữ liệu.",
        ]],
        "question": "Mình đang ôn thi môn gì nhỉ?",
        "expect_any": ["cơ sở dữ liệu", "csdl", "database"],
        "forbid": [],
    },
    {
        "id": "recall_name", "kind": "recall",
        "setup": [["Gọi mình là Tâm nhé, mình thích câu trả lời ngắn gọn."]],
        "question": "Bạn nên gọi mình là gì?",
        "expect_any": ["tâm"],
        "forbid": [],
    },
    {
        "id": "recall_project", "kind": "recall",
        "setup": [["Mình đang làm đồ án về ứng dụng quản lý chi tiêu cá nhân."]],
        "question": "Đồ án của mình nói về chủ đề gì?",
        "expect_any": ["chi tiêu"],
        "forbid": [],
    },
    {
        "id": "recall_year", "kind": "recall",
        "setup": [["Mình tên Tâm, đang học năm ba ngành Khoa học máy tính."]],
        "question": "Mình đang học năm mấy?",
        "expect_any": ["năm ba", "năm 3", "năm thứ ba", "năm thứ 3"],
        "forbid": [],
    },
    {
        "id": "update_exam", "kind": "update",
        "setup": [
            ["Mình đang ôn thi môn Cơ sở dữ liệu."],
            ["Mình đổi lịch rồi, giờ mình ôn thi môn Mạng máy tính."],
        ],
        "question": "Mình đang ôn thi môn gì nhỉ?",
        "expect_any": ["mạng máy tính"],
        "forbid": ["cơ sở dữ liệu", "csdl"],
    },
    {
        "id": "unknown_exam", "kind": "unknown",
        "setup": [],
        "question": "Mình đang ôn thi môn gì nhỉ?",
        "expect_any": NOT_KNOW,
        "forbid": [],
    },
]

TOKENS_RE = re.compile(
    r"\[TOKENS\][\s\S]*?purpose=(\w+)[\s\S]*?model=(\S+)[\s\S]*?prompt=(\d+)\s+completion=(\d+)"
)
MEMORY_FLAG_RE = re.compile(r"Memory enabled:\s*(\w+)")
CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")  # Chinese/Japanese/Korean characters


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def norm(text: str) -> str:
    """Normalize text for comparison: Unicode NFC + lowercase."""
    return unicodedata.normalize("NFC", text).lower()


def grade(answer: str, sc: dict) -> bool:
    """Grade response against expected and forbidden keywords."""
    a = norm(answer)
    has_expected = any(norm(k) in a for k in sc["expect_any"])
    has_forbidden = any(norm(k) in a for k in sc["forbid"])
    return has_expected and not has_forbidden


def language_ok(answer: str) -> bool:
    """Check if the answer contains unexpected CJK character pollution."""
    return CJK_RE.search(answer) is None


def log_size(path: Path | None) -> int:
    """Return the current file size of the log file."""
    return path.stat().st_size if path else 0


def read_log(path: Path | None, start: int) -> str:
    """Read log output appended since byte offset `start`."""
    if not path:
        return ""
    time.sleep(0.5)  # Wait for file system buffer sync
    with open(path, "rb") as f:
        f.seek(start)
        return f.read().decode("utf-8", errors="replace")


def parse_tokens(text: str) -> dict:
    """Aggregate token usage counts by call purpose (agent / memory_retrieve / memory_summarize)."""
    out: dict = {}
    for purpose, model, p, c in TOKENS_RE.findall(text):
        d = out.setdefault(purpose, {"calls": 0, "prompt": 0, "completion": 0})
        d["calls"] += 1
        d["prompt"] += int(p)
        d["completion"] += int(c)
    return out


def models_in(text: str) -> list[str]:
    """Extract unique model names recorded in token log entries."""
    return sorted({m for _, m, _, _ in TOKENS_RE.findall(text)})


# ---------------------------------------------------------------------------
# API Operations
# ---------------------------------------------------------------------------
def create_session(client: httpx.Client, title: str) -> str:
    """Create a new chat session via API."""
    r = client.post("/session/", json={"title": title})
    r.raise_for_status()
    return r.json()["id"]


def delete_session(client: httpx.Client, session_id: str) -> float:
    """Delete a session to trigger memory consolidation."""
    t0 = time.perf_counter()
    r = client.delete(f"/session/{session_id}/")
    r.raise_for_status()
    return time.perf_counter() - t0


def ask(client: httpx.Client, session_id: str, question: str):
    """Send question, stream SSE events to completion, return (answer, error, duration)."""
    t0 = time.perf_counter()
    answer, error, event = "", None, None
    with client.stream(
        "POST", "/agent/query",
        json={"question": question, "session_id": session_id},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = json.loads(line[len("data:"):].strip())
                if event == "answer":
                    answer += data.get("text", "")
                elif event == "error":
                    error = data.get("detail")
    return answer.strip(), error, time.perf_counter() - t0


def run_reset() -> None:
    """Run `evals/reset.sh` script to reset memory state."""
    script = ROOT / "evals" / "reset.sh"
    subprocess.run([str(script)], cwd=ROOT, check=True, capture_output=True)


# ---------------------------------------------------------------------------
# Scenario Execution
# ---------------------------------------------------------------------------
def run_scenario(client, sc, rep, args, log_path) -> dict:
    """Execute a single scenario run and return result metrics dict."""
    row = {
        "label": args.label, "scenario": sc["id"], "kind": sc["kind"], "rep": rep,
        "question": sc["question"], "answer": None, "passed": False,
        "language_ok": None, "error": None,
    }
    try:
        if not args.no_reset:
            run_reset()
        start_offset = log_size(log_path)
        consolidation_s = 0.0

        # Execute setup sessions and trigger memory consolidation on delete
        for i, messages in enumerate(sc["setup"], 1):
            sid = create_session(client, f"{sc['id']}-setup{i}")
            for msg in messages:
                ask(client, sid, msg)
            consolidation_s += delete_session(client, sid)

        # Record profile snapshot post-setup
        row["profile_after_setup"] = (
            USER_PROFILE.read_text(encoding="utf-8")[:500]
            if USER_PROFILE.exists() else None
        )

        # Run test evaluation session
        sid = create_session(client, f"{sc['id']}-test")
        final_offset = log_size(log_path)
        answer, err, final_s = ask(client, sid, sc["question"])

        row.update(
            answer=answer, error=err,
            passed=grade(answer, sc), language_ok=language_ok(answer),
            final_question_s=round(final_s, 2),
            consolidation_s=round(consolidation_s, 2),
        )
        if log_path:
            whole = read_log(log_path, start_offset)
            final_part = read_log(log_path, final_offset)
            row["tokens_total"] = parse_tokens(whole)
            row["tokens_final"] = parse_tokens(final_part)
            row["models"] = models_in(whole)
    except (httpx.HTTPError, subprocess.CalledProcessError, json.JSONDecodeError) as e:
        row["error"] = f"{type(e).__name__}: {e}"
    return row


# ---------------------------------------------------------------------------
# Summary Output
# ---------------------------------------------------------------------------
def avg(values):
    """Compute average for non-None numeric items."""
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def summarize(rows: list[dict], label: str) -> None:
    """Print formatted summary metrics to stdout."""
    n = len(rows)
    if not n:
        return
    ok = sum(r["passed"] for r in rows)
    lang = sum(1 for r in rows if r["language_ok"])
    errs = sum(1 for r in rows if r["error"])
    print(f"\n==== Benchmark Summary: {label} ({n} runs) ====")
    print(f"Passed content : {ok}/{n}")
    print(f"Valid language : {lang}/{n}")
    print(f"Errors         : {errs}/{n}")
    print(f"Test question  : {avg([r.get('final_question_s') for r in rows]):.1f}s (avg)")
    print(f"Delete session : {avg([r.get('consolidation_s') for r in rows]):.1f}s (avg, includes memory consolidation)")

    print("\nBy Scenario Type (passed/total):")
    for kind in sorted({r["kind"] for r in rows}):
        sub = [r for r in rows if r["kind"] == kind]
        print(f"  {kind:<8} {sum(r['passed'] for r in sub)}/{len(sub)}")

    purposes = sorted({p for r in rows for p in (r.get("tokens_total") or {})})
    if purposes:
        print("\nAverage Tokens per Scenario by Purpose:")
        for p in purposes:
            calls = avg([(r.get("tokens_total") or {}).get(p, {}).get("calls", 0) for r in rows])
            pr = avg([(r.get("tokens_total") or {}).get(p, {}).get("prompt", 0) for r in rows])
            co = avg([(r.get("tokens_total") or {}).get(p, {}).get("completion", 0) for r in rows])
            print(f"  {p:<18} calls={calls:.1f}  prompt={pr:.0f}  completion={co:.0f}")
        fin = avg([(r.get("tokens_final") or {}).get("agent", {}).get("prompt", 0) for r in rows])
        print(f"\nAgent Prompt Tokens for Test Question (avg): {fin:.0f} tokens")


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Agent Memory Benchmark Tool")
    ap.add_argument("--label", required=True, help="configuration label (e.g. ollama-on, groq-on)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="base URL of the application server")
    ap.add_argument("--log", type=Path, help="server log file path (for parsing token metrics)")
    ap.add_argument("--reps", type=int, default=1, help="number of repetition runs per scenario")
    ap.add_argument("--only", nargs="*", help="filter to run only specific scenario IDs")
    ap.add_argument("--pause", type=float, default=0, help="pause delay between scenarios in seconds")
    ap.add_argument("--no-reset", action="store_true", help="skip memory state reset before scenarios")
    ap.add_argument("--out", type=Path, help="output result path (default: evals/results/<label>.jsonl)")
    args = ap.parse_args()

    scenarios = [s for s in SCENARIOS if not args.only or s["id"] in args.only]
    if not scenarios:
        print("No scenarios matched --only filter.", file=sys.stderr)
        return 1

    out_path = args.out or ROOT / "evals" / "results" / f"{args.label}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.log and args.log.exists():
        flags = MEMORY_FLAG_RE.findall(args.log.read_text(encoding="utf-8", errors="replace"))
        if flags:
            print(f"[Check] Server log indicates 'Memory enabled' = {flags[-1]}. "
                  f"Specified label is '{args.label}'. Ensure configurations match.")

    rows: list[dict] = []
    total = len(scenarios) * args.reps
    done = 0
    timeout = httpx.Timeout(connect=10, read=300, write=30, pool=30)
    with httpx.Client(base_url=args.base_url, timeout=timeout) as client, \
            open(out_path, "a", encoding="utf-8") as fout:
        for rep in range(1, args.reps + 1):
            for sc in scenarios:
                done += 1
                row = run_scenario(client, sc, rep, args, args.log)
                rows.append(row)
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                status = "PASSED" if row["passed"] else "FAILED"
                note = f"  ERROR: {row['error']}" if row["error"] else ""
                print(f"[{done}/{total}] {sc['id']:<15} rep={rep} {status}{note}")
                if args.pause:
                    time.sleep(args.pause)

    summarize(rows, args.label)
    print(f"\nDetailed benchmark output written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())