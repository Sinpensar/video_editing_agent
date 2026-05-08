# Eval harness

Regression tests for the agent's *behavior*, not just its code. Each
case describes a scenario, runs the agent against it, and asserts that
the right tools were called with the right arguments and the resulting
cut satisfies content/length/cost constraints.

## Run

```bash
# All cases, deterministic offline mode (no API tokens spent):
python -m evals.runner --offline

# A single case:
python -m evals.runner --offline --case 01_filler_removal

# Real LLM (your .env's LLM_API_KEY is used):
python -m evals.runner
```

`--offline` walks each case's pre-recorded `script` of LLM responses
through the agent loop. The validators, the project state, the tracer,
and your check assertions are all real — only the LLM is replayed.

Real-LLM mode is what you'd actually use to compare prompt or model
changes; offline mode is for fast deterministic CI / iteration on the
harness itself.

## Anatomy of a case

```jsonc
{
  "name":         "...",
  "description":  "what we're testing",
  "setup": {
    "videos": [
      {
        "video_id": "v1",
        "transcript": { "language": "en", "duration": 30.0,
                        "segments": [ { "id": 0, "start": 0, "end": 4, "text": "..." } ] },
        "visuals":     null      // optional pre-baked visuals
      }
    ],
    "goal": null                 // optional pre-set goal
  },
  "prompt": "What the user types into the agent.",
  "script": [                    // only used in --offline; one entry per LLM API call
    { "tool_calls": [ { "name": "set_goal", "arguments": { ... } } ] },
    { "content": "Final reply text." }
  ],
  "checks": [
    { "type": "tool_called",       "tool": "set_goal" },
    { "type": "duration_in",       "min": 25, "max": 35 },
    { "type": "kept_text_includes", "patterns": ["espresso"] }
  ]
}
```

The agent runs in `dry_run_cuts=True` mode under the harness, so
ffmpeg is never invoked — only the planning, validation, and tool-call
behavior are exercised.

## Available checks

(Defined in `checks.py` as `CHECKS`.)

| `type` | Parameters | Passes when |
|---|---|---|
| `no_agent_error` | – | the chat loop returned without exception |
| `tool_called` | `tool`, `min_count=1`, `max_count=None` | `tool` was invoked the right number of times |
| `tool_not_called` | `tool` | the named tool was never invoked |
| `tool_args_match` | `tool`, `args_contain={...}` | some call to `tool` had ALL the listed key/value pairs |
| `no_validation_failures` | – | the final `create_cut` was not rejected by validators |
| `duration_in` | `min`, `max` | the total kept duration falls in [min, max] seconds |
| `clip_count` | `min=0`, `max=None` | the final cut has the right number of clips |
| `kept_text_includes` | `patterns=[..]`, `match_all=True` | every (or any) pattern appears in kept content |
| `kept_text_excludes` | `patterns=[..]` | none of the patterns appear in kept content |
| `max_total_tokens` | `value` | total tokens used didn't exceed budget |
| `final_reply_matches` | `pattern`, `flags` | the assistant's final text matches a regex |

Add a new check by writing a function in `checks.py` and registering it
in `CHECKS`. Naming convention: `check_<verb>_<noun>(result, **params)`.

## Why a separate `dry_run_cuts` mode?

Running real ffmpeg in evals would mean shipping fixture videos,
make every case take seconds, and produce nondeterministic byte-level
output that's hard to assert against. The interesting LLM behavior —
which clips it picks, with what timestamps — is fully observable
without ever encoding a frame. So the runner sets `Project.dry_run_cuts
= True`, which makes `create_cut` validate + return a plan instead of
calling `editor.assemble`.
