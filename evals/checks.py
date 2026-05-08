"""
Check primitives the eval harness applies to a finished case run.

Each function takes a Result bundle (the trace + the final reply +
the project state) and returns (passed: bool, message: str).

Cases reference checks by `type` plus parameters:

    {"type": "tool_called", "tool": "set_goal"}
    {"type": "kept_text_includes", "patterns": ["coffee"]}
    {"type": "duration_in", "min": 50, "max": 70}

Add a new check by writing a function below + listing it in CHECKS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class RunResult:
    """Everything an eval check might want to inspect after a chat() turn."""
    case_name: str
    final_reply: str
    project: Any                  # the Project instance
    tool_calls: list[dict]        # [{name, arguments, result, is_error, duration_ms}]
    steps: list[dict]             # raw LLM steps from Tracer
    totals: dict                  # tokens / latency totals
    error: str | None = None      # set if the agent crashed


# ---- check helpers ---------------------------------------------------------

def _calls_for(result: RunResult, tool: str) -> list[dict]:
    return [c for c in result.tool_calls if c["name"] == tool]


def _last_create_cut_clips(result: RunResult) -> list[dict]:
    """Clips of the last create_cut tool call (if any)."""
    cuts = _calls_for(result, "create_cut")
    if not cuts:
        return []
    return list(cuts[-1].get("arguments", {}).get("clips", []))


def _kept_transcript_text(result: RunResult) -> str:
    """Concatenate transcript text covered by the last create_cut's clips."""
    clips = _last_create_cut_clips(result)
    parts: list[str] = []
    for c in clips:
        entry = result.project.videos.get(c.get("video_id"))
        if not entry:
            continue
        s, e = float(c.get("start", 0)), float(c.get("end", 0))
        for seg in entry.transcript.get("segments", []):
            if seg["end"] > s and seg["start"] < e:
                parts.append(seg["text"])
    return " ".join(parts).lower()


# ---- individual check functions --------------------------------------------

def check_tool_called(result: RunResult, *, tool: str,
                     min_count: int = 1, max_count: int | None = None) -> tuple[bool, str]:
    n = len(_calls_for(result, tool))
    if n < min_count:
        return False, f"{tool} called {n} time(s), needed ≥{min_count}"
    if max_count is not None and n > max_count:
        return False, f"{tool} called {n} time(s), allowed ≤{max_count}"
    return True, f"{tool} called {n} time(s)"


def check_tool_not_called(result: RunResult, *, tool: str) -> tuple[bool, str]:
    n = len(_calls_for(result, tool))
    if n > 0:
        return False, f"{tool} should NOT have been called (was {n}×)"
    return True, f"{tool} not called"


def check_tool_args_match(result: RunResult, *, tool: str,
                         args_contain: dict) -> tuple[bool, str]:
    """Pass if any call to `tool` has args that contain ALL key:value pairs."""
    calls = _calls_for(result, tool)
    if not calls:
        return False, f"{tool} was never called"
    for call in calls:
        a = call.get("arguments", {})
        if all(a.get(k) == v for k, v in args_contain.items()):
            return True, f"{tool} called with {args_contain}"
    return False, f"no call to {tool} matched args {args_contain}; got {[c['arguments'] for c in calls]}"


def check_duration_in(result: RunResult, *, min: float, max: float) -> tuple[bool, str]:
    clips = _last_create_cut_clips(result)
    total = sum(float(c["end"]) - float(c["start"]) for c in clips
                if c.get("end") and c.get("start") is not None)
    if total < min or total > max:
        return False, f"final duration {total:.1f}s not in [{min}, {max}]"
    return True, f"duration {total:.1f}s ∈ [{min}, {max}]"


def check_clip_count(result: RunResult, *, min: int = 0,
                    max: int | None = None) -> tuple[bool, str]:
    n = len(_last_create_cut_clips(result))
    if n < min:
        return False, f"only {n} clip(s), need ≥{min}"
    if max is not None and n > max:
        return False, f"{n} clip(s), allowed ≤{max}"
    return True, f"{n} clip(s)"


def check_kept_text_includes(result: RunResult, *, patterns: list[str],
                             match_all: bool = True) -> tuple[bool, str]:
    text = _kept_transcript_text(result)
    misses = [p for p in patterns if p.lower() not in text]
    if match_all and misses:
        return False, f"kept text missing: {misses}"
    if not match_all and len(misses) == len(patterns):
        return False, f"kept text matches none of: {patterns}"
    return True, f"kept text covers {len(patterns) - len(misses)}/{len(patterns)} pattern(s)"


def check_kept_text_excludes(result: RunResult, *, patterns: list[str]) -> tuple[bool, str]:
    text = _kept_transcript_text(result)
    hits = [p for p in patterns if p.lower() in text]
    if hits:
        return False, f"kept text contains forbidden patterns: {hits}"
    return True, f"kept text excludes all of: {patterns}"


def check_max_total_tokens(result: RunResult, *, value: int) -> tuple[bool, str]:
    used = result.totals.get("total_tokens", 0)
    if used > value:
        return False, f"used {used} tokens, budget {value}"
    return True, f"used {used} tokens (≤ {value})"


def check_no_validation_failures(result: RunResult) -> tuple[bool, str]:
    """Every create_cut call should eventually succeed (no validation_failed
    in the final cut)."""
    cuts = _calls_for(result, "create_cut")
    if not cuts:
        return False, "create_cut never called"
    last = cuts[-1].get("result", {})
    if isinstance(last, dict) and last.get("validation_failed"):
        return False, f"final create_cut still fails validation: {last.get('issues')}"
    return True, "final create_cut passed validation"


def check_final_reply_matches(result: RunResult, *, pattern: str,
                              flags: int = re.IGNORECASE) -> tuple[bool, str]:
    if re.search(pattern, result.final_reply or "", flags):
        return True, f"reply matches /{pattern}/"
    return False, f"reply does not match /{pattern}/; got: {(result.final_reply or '')[:120]}"


def check_no_agent_error(result: RunResult) -> tuple[bool, str]:
    if result.error:
        return False, f"agent crashed: {result.error}"
    return True, "agent completed without error"


CHECKS: dict[str, Callable[..., tuple[bool, str]]] = {
    "tool_called":            check_tool_called,
    "tool_not_called":        check_tool_not_called,
    "tool_args_match":        check_tool_args_match,
    "duration_in":            check_duration_in,
    "clip_count":             check_clip_count,
    "kept_text_includes":     check_kept_text_includes,
    "kept_text_excludes":     check_kept_text_excludes,
    "max_total_tokens":       check_max_total_tokens,
    "no_validation_failures": check_no_validation_failures,
    "final_reply_matches":    check_final_reply_matches,
    "no_agent_error":         check_no_agent_error,
}


def run_check(result: RunResult, spec: dict) -> tuple[bool, str]:
    """Dispatch a check spec dict against the result."""
    spec = dict(spec)   # don't mutate caller's
    kind = spec.pop("type", None)
    fn = CHECKS.get(kind)
    if fn is None:
        return False, f"unknown check type: {kind!r}"
    try:
        return fn(result, **spec)
    except TypeError as e:
        return False, f"check {kind} bad args: {e}"
    except Exception as e:
        return False, f"check {kind} crashed: {e}"
