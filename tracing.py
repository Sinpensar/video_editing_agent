"""
Lightweight tracing: dump every conversational turn to a JSON file
under traces/<session_id>/turn_NNN.json.

Each turn captures:
- the user's input
- every LLM API call inside the tool-use loop (with tokens + latency)
- every tool dispatch (with args, result, duration, error flag)
- a totals block (tokens, latency, llm calls, tool calls)
- the final assistant reply

The traces are read by viewer.py to build a static HTML inspector.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

TRACE_DIR_DEFAULT = Path(__file__).parent / "traces"


def _new_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _truncate_for_log(value: Any, max_chars: int = 4000) -> Any:
    """Keep traces readable when a tool returns a huge transcript."""
    s = json.dumps(value, ensure_ascii=False, default=str)
    if len(s) <= max_chars:
        return value
    head = s[: max_chars // 2]
    tail = s[-max_chars // 4 :]
    return {
        "_truncated": True,
        "_original_len": len(s),
        "preview_head": head,
        "preview_tail": tail,
    }


class Tracer:
    """Records LLM + tool activity to disk, one JSON per user-input turn."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        traces_dir: Optional[Path] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.session_id = session_id or _new_session_id()
        self.dir = Path(traces_dir or TRACE_DIR_DEFAULT) / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.base_url = base_url
        self.turn_count = 0
        self._cur: Optional[dict] = None
        self._meta_path = self.dir / "session.json"
        # Persist session metadata so the viewer can label it
        self._write_meta(turns=0)

    def _write_meta(self, turns: int) -> None:
        meta = {
            "session_id": self.session_id,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "model": self.model,
            "base_url": self.base_url,
            "turns": turns,
            "cwd": os.getcwd(),
        }
        # Preserve original started_at if file already exists
        if self._meta_path.exists():
            try:
                old = json.loads(self._meta_path.read_text(encoding="utf-8"))
                meta["started_at"] = old.get("started_at", meta["started_at"])
            except Exception:
                pass
        self._meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---- per-turn lifecycle --------------------------------------------------

    def begin_turn(self, user_input: str) -> None:
        self.turn_count += 1
        self._cur = {
            "turn": self.turn_count,
            "started_at": datetime.now().isoformat(timespec="milliseconds"),
            "user_input": user_input,
            "steps": [],
            "tool_calls": [],
            "final_reply": None,
            "totals": {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "llm_latency_ms": 0,
                "tool_duration_ms": 0,
                "llm_calls": 0,
                "tool_calls": 0,
            },
        }

    def record_llm_call(
        self,
        request_messages: list[dict],
        response: Any,
        latency_ms: int,
    ) -> None:
        if not self._cur:
            return

        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tok = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tok = getattr(usage, "total_tokens", in_tok + out_tok) if usage else in_tok + out_tok

        msg = response.choices[0].message
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                # arguments comes back as a JSON-encoded string from the API
                raw_args = tc.function.arguments
                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                except Exception:
                    parsed_args = {"_raw": raw_args}
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": parsed_args,
                })

        step = {
            "step": len(self._cur["steps"]) + 1,
            "n_messages_in_request": len(request_messages),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": total_tok,
            "latency_ms": latency_ms,
            "stop_reason": getattr(response.choices[0], "finish_reason", None),
            "response": {
                "content": msg.content,
                "tool_calls": tool_calls or None,
            },
        }
        self._cur["steps"].append(step)
        t = self._cur["totals"]
        t["input_tokens"] += in_tok
        t["output_tokens"] += out_tok
        t["total_tokens"] += total_tok
        t["llm_latency_ms"] += latency_ms
        t["llm_calls"] += 1

    def record_tool_call(
        self,
        name: str,
        args: dict,
        result: Any,
        duration_ms: int,
        is_error: bool,
    ) -> None:
        if not self._cur:
            return
        # Step the tool was issued from = current last LLM step number
        step_num = len(self._cur["steps"])
        self._cur["tool_calls"].append({
            "step": step_num,
            "name": name,
            "arguments": args,
            "result": _truncate_for_log(result),
            "is_error": bool(is_error),
            "duration_ms": duration_ms,
        })
        self._cur["totals"]["tool_duration_ms"] += duration_ms
        self._cur["totals"]["tool_calls"] += 1

    def end_turn(self, final_reply: Optional[str]) -> None:
        if not self._cur:
            return
        self._cur["final_reply"] = final_reply
        self._cur["ended_at"] = datetime.now().isoformat(timespec="milliseconds")
        path = self.dir / f"turn_{self._cur['turn']:03d}.json"
        path.write_text(
            json.dumps(self._cur, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self._cur = None
        self._write_meta(turns=self.turn_count)
