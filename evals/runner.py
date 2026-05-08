"""
Eval harness for VlogAgent.

Each *case* is a JSON file under evals/cases/ describing:
  - what to pre-load into the project (synthetic videos / transcripts)
  - the user message that drives the agent
  - either an LLM script (offline mode) OR nothing (real LLM)
  - a list of checks to apply once the turn finishes

Run all cases in offline mode (deterministic, no API tokens):
    python -m evals.runner --offline

Run all cases against the real LLM in your .env:
    python -m evals.runner

Run one case by name:
    python -m evals.runner --case cooking_filler_removal --offline
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

# Make the parent package importable when running as a script:
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent as agent_mod
from evals import checks as checks_mod
from evals.checks import RunResult
from evals.scripted_client import ScriptedClient

CASES_DIR = Path(__file__).resolve().parent / "cases"


# ---------------------------------------------------------------------------
# Case loading and synthetic project setup
# ---------------------------------------------------------------------------

def _load_cases(only: str | None = None) -> list[dict]:
    cases: list[dict] = []
    for path in sorted(CASES_DIR.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        case.setdefault("name", path.stem)
        case["_path"] = str(path)
        if only and case["name"] != only:
            continue
        cases.append(case)
    return cases


def _build_project(case: dict) -> agent_mod.Project:
    """Build a Project pre-loaded with synthetic VideoEntry objects, no
    Whisper or ffmpeg required. Eval cases describe synthetic transcripts
    inline."""
    project = agent_mod.Project()
    project.dry_run_cuts = True   # never call real ffmpeg in evals
    for idx, video in enumerate(case.get("setup", {}).get("videos", []), start=1):
        vid = video.get("video_id") or f"v{idx}"
        path = Path(video.get("path", f"/tmp/eval_{vid}.mp4"))
        transcript = video.get("transcript") or {
            "language": "en", "duration": 0, "segments": []
        }
        project.videos[vid] = agent_mod.VideoEntry(
            video_id=vid, path=path, transcript=transcript,
        )
        if "visuals" in video:
            project.visuals[vid] = video["visuals"]
    if case.get("setup", {}).get("goal"):
        project.set_goal(**case["setup"]["goal"])
    return project


# ---------------------------------------------------------------------------
# Single-case execution
# ---------------------------------------------------------------------------

@dataclass
class CaseRun:
    name: str
    passed: int
    failed: int
    check_lines: list[str]
    totals: dict
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.failed == 0


def _run_case(case: dict, *, offline: bool, max_iters: int = 20) -> CaseRun:
    name = case["name"]
    project = _build_project(case)

    if offline:
        if "script" not in case:
            return CaseRun(name, 0, 1,
                           [f"  ⚠ no 'script' in case (required for --offline)"],
                           {}, error="missing script")
        client = ScriptedClient(case["script"])
        a = agent_mod.VlogAgent(
            model=case.get("model", "scripted"),
            max_tool_iters=max_iters,
            client=client,
        )
        a.project = project
    else:
        a = agent_mod.VlogAgent(max_tool_iters=max_iters)
        a.project = project

    # Drive the chat. Capture trace via the on_tool callback we already have.
    tool_log: list[dict] = []
    error: str | None = None
    final_reply = ""
    try:
        final_reply = a.chat(
            case["prompt"],
            on_tool=lambda n, ar, r: tool_log.append({
                "name": n, "arguments": ar, "result": r,
                "is_error": isinstance(r, dict) and "error" in r,
            }),
        )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    # Tally token usage from the tracer if there's one; else estimate from
    # message history.
    totals = {"total_tokens": 0, "llm_calls": 0, "tool_calls": len(tool_log)}
    if a.tracer and a.tracer._cur is not None:
        totals.update(a.tracer._cur.get("totals", {}))

    result = RunResult(
        case_name=name,
        final_reply=final_reply or "",
        project=project,
        tool_calls=tool_log,
        steps=[],   # TODO: surface from tracer
        totals=totals,
        error=error,
    )

    passed = failed = 0
    lines: list[str] = []
    for spec in case.get("checks", []):
        ok, msg = checks_mod.run_check(result, spec)
        marker = "✓" if ok else "✗"
        lines.append(f"    {marker} {spec.get('type'):<25s} {msg}")
        if ok:
            passed += 1
        else:
            failed += 1

    return CaseRun(name, passed, failed, lines, totals, error)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", help="Run a single case by name (matches filename stem)")
    ap.add_argument("--offline", action="store_true",
                    help="Use the scripted_client; no real LLM calls.")
    ap.add_argument("--cases-dir", default=None,
                    help="Override cases directory.")
    ap.add_argument("--quiet", action="store_true",
                    help="Print only the summary line.")
    args = ap.parse_args()

    if args.cases_dir:
        global CASES_DIR
        CASES_DIR = Path(args.cases_dir).resolve()

    cases = _load_cases(only=args.case)
    if not cases:
        print(f"No cases found in {CASES_DIR}"
              + (f" matching '{args.case}'" if args.case else ""))
        return 1

    runs: list[CaseRun] = []
    for i, case in enumerate(cases, start=1):
        run = _run_case(case, offline=args.offline)
        runs.append(run)
        if args.quiet:
            continue
        status = "PASS" if run.ok else "FAIL"
        print(f"[{i}/{len(cases)}] {run.name:<40s} {status} "
              f"({run.passed}/{run.passed + run.failed} checks, "
              f"{run.totals.get('total_tokens', 0)} tokens)")
        if run.error:
            print(f"    ⚠ {run.error}")
        for line in run.check_lines:
            print(line)

    total_pass = sum(1 for r in runs if r.ok)
    total_checks = sum(r.passed + r.failed for r in runs)
    passed_checks = sum(r.passed for r in runs)
    total_tokens = sum(r.totals.get("total_tokens", 0) for r in runs)
    print()
    print(f"Summary: {total_pass}/{len(runs)} cases passed, "
          f"{passed_checks}/{total_checks} checks passed, "
          f"{total_tokens} tokens total")
    return 0 if total_pass == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main())
