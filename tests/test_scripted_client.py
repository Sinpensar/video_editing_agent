"""ScriptedClient is what makes our --offline eval mode work. Make sure
its OpenAI-shape mimicry stays correct."""

import json

from evals.scripted_client import ScriptedClient


def test_returns_text_response():
    c = ScriptedClient([{"content": "hello"}])
    resp = c.chat.completions.create(messages=[])
    assert resp.choices[0].message.content == "hello"
    assert resp.choices[0].message.tool_calls is None
    assert resp.choices[0].finish_reason == "stop"


def test_returns_tool_calls():
    c = ScriptedClient([
        {"tool_calls": [{"name": "set_goal",
                         "arguments": {"target_duration_sec": 30}}]},
    ])
    resp = c.chat.completions.create(messages=[])
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "tool_calls"
    assert msg.tool_calls is not None
    assert len(msg.tool_calls) == 1
    tc = msg.tool_calls[0]
    assert tc.function.name == "set_goal"
    assert json.loads(tc.function.arguments) == {"target_duration_sec": 30}
    assert tc.id.startswith("call_")


def test_advances_through_script():
    c = ScriptedClient([
        {"content": "step 1"},
        {"content": "step 2"},
        {"content": "step 3"},
    ])
    assert c.remaining == 3
    c.chat.completions.create(messages=[])
    assert c.remaining == 2
    c.chat.completions.create(messages=[])
    c.chat.completions.create(messages=[])
    assert c.remaining == 0


def test_records_calls():
    c = ScriptedClient([{"content": "ok"}])
    c.chat.completions.create(messages=[{"role": "user", "content": "hi"}])
    assert len(c.calls) == 1
    assert c.calls[0]["n_messages"] == 1


def test_exhausted_script_returns_polite_error():
    """If the script runs out, we return a final-text response so the
    agent loop terminates rather than silently looping."""
    c = ScriptedClient([])
    resp = c.chat.completions.create(messages=[])
    assert "exhausted" in (resp.choices[0].message.content or "")
    assert resp.choices[0].finish_reason == "stop"


def test_usage_attribute_present():
    c = ScriptedClient([{"content": "x"}])
    resp = c.chat.completions.create(messages=[])
    assert hasattr(resp, "usage")
    assert resp.usage.prompt_tokens >= 0
    assert resp.usage.completion_tokens >= 0
    assert resp.usage.total_tokens == (
        resp.usage.prompt_tokens + resp.usage.completion_tokens
    )


def test_id_uniqueness_across_calls():
    c = ScriptedClient([
        {"tool_calls": [{"name": "x", "arguments": {}}]},
        {"tool_calls": [{"name": "y", "arguments": {}}]},
    ])
    r1 = c.chat.completions.create(messages=[])
    r2 = c.chat.completions.create(messages=[])
    assert r1.choices[0].message.tool_calls[0].id != r2.choices[0].message.tool_calls[0].id
