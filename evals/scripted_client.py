"""
A scripted "LLM client" that replays a canned sequence of responses,
matching the OpenAI Python SDK's surface that VlogAgent.chat() touches.

Used by the eval harness in --offline mode and in unit tests, so cases
can run deterministically without burning real API tokens.

A script is a list of step dicts, each describing what one LLM API call
should return. Two shapes are supported per step:

    # Step that issues tool calls
    {
        "tool_calls": [
            {"name": "set_goal",
             "arguments": {"target_duration_sec": 30}},
            {"name": "get_transcript",
             "arguments": {"video_id": "v1"}},
        ]
    }

    # Final, plain-text step (loop ends after this)
    {"content": "Done. Picked 5 clips totaling 28.4s."}

Each call to client.chat.completions.create(...) advances one step in
the script. If the script runs out, an error response is returned so
the surrounding loop notices.
"""

from __future__ import annotations

import json
import uuid
from typing import Any


class _Fn:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = _Fn(name, arguments)


class _Message:
    def __init__(self, content: str | None, tool_calls: list[_ToolCall] | None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message: _Message, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


class _Response:
    def __init__(self, message: _Message, finish_reason: str,
                 prompt_tokens: int, completion_tokens: int) -> None:
        self.choices = [_Choice(message, finish_reason)]
        self.usage = _Usage(prompt_tokens, completion_tokens)


def _step_to_response(step: dict, recorded_messages: list[dict]) -> _Response:
    """Convert one script step into an OpenAI-shaped response object."""
    p_tokens = step.get("prompt_tokens", _approx_tokens_in(recorded_messages))
    c_tokens = step.get("completion_tokens", 50)

    if "tool_calls" in step and step["tool_calls"]:
        tcs = []
        for tc in step["tool_calls"]:
            tcs.append(_ToolCall(
                id=tc.get("id", f"call_{uuid.uuid4().hex[:10]}"),
                name=tc["name"],
                arguments=json.dumps(tc.get("arguments", {}), ensure_ascii=False),
            ))
        msg = _Message(content=step.get("content"), tool_calls=tcs)
        finish = step.get("finish_reason", "tool_calls")
    else:
        msg = _Message(content=step.get("content", ""), tool_calls=None)
        finish = step.get("finish_reason", "stop")
    return _Response(msg, finish, p_tokens, c_tokens)


def _approx_tokens_in(messages: list[dict]) -> int:
    """Rough proxy for prompt token count — for trace bookkeeping only."""
    text = json.dumps(messages, ensure_ascii=False, default=str)
    return max(1, len(text) // 4)


class _ChatCompletions:
    def __init__(self, parent: "ScriptedClient") -> None:
        self._parent = parent

    def create(self, **kwargs: Any) -> _Response:
        return self._parent._next(kwargs.get("messages", []))


class _Chat:
    def __init__(self, parent: "ScriptedClient") -> None:
        self.completions = _ChatCompletions(parent)


class ScriptedClient:
    """Drop-in replacement for openai.OpenAI that walks through a script.

    Only the .chat.completions.create surface is implemented — everything
    else raises AttributeError, intentionally.
    """

    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self._index = 0
        self.calls: list[dict] = []     # observed kwargs per call (for tests)
        self.chat = _Chat(self)

    @property
    def remaining(self) -> int:
        return len(self._script) - self._index

    def _next(self, messages: list[dict]) -> _Response:
        self.calls.append({"n_messages": len(messages)})
        if self._index >= len(self._script):
            # Script exhausted — emit a polite error so the agent loop ends
            msg = _Message(
                content="(scripted client: script exhausted)",
                tool_calls=None,
            )
            return _Response(msg, "stop", 0, 0)
        step = self._script[self._index]
        self._index += 1
        return _step_to_response(step, messages)
