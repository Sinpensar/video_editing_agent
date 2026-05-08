# VlogAgent

<!-- Once you push to GitHub, replace OWNER/REPO below with your actual path. -->
[![CI](https://github.com/Sinpensar/video_editing_agent
/actions/workflows/ci.yml/badge.svg)](https://github.com/Sinpensar/video_editing_agent
/actions/workflows/ci.yml)

A conversational vlog editor.
You hand it raw footage, tell it what you want, and it transcribes the audio, decides what to keep, and runs FFmpeg to spit a finished cut into `output/`.

Stack:
- **Whisper** — turns the audio in your video into timestamped text
- **LLM (default: Zhipu GLM-4-Flash, free)** — reads the transcript, understands your goal, decides which segments to keep. Switching to Gemini / Qwen / DeepSeek is a `.env` change, no code edits.
- **FFmpeg** — actually does the cutting and concatenation

## Install

You need [`ffmpeg`](https://ffmpeg.org/download.html) on your `PATH` first (`ffmpeg -version` should print something).

```bash
cd "AI agent/vlog_agent"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The first install is slow because it pulls in PyTorch and the Whisper `base` model (~150 MB).

## Configure your API key

Default provider is **Zhipu GLM-4-Flash** — fully free, no credit card, accessible from anywhere in the world:

1. Sign up at https://open.bigmodel.cn (WeChat scan or phone number)
2. Go to https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys → "Add API Key" → copy
3. Drop it into your local `.env`:

```bash
cp .env.example .env
# open .env in your editor and paste your key into LLM_API_KEY
```

## Run it

```bash
python main.py
# pre-load specific clips:
python main.py ~/Movies/raw/clip1.mp4 ~/Movies/raw/clip2.mp4
# or point at a folder to bulk-load every video inside it:
python main.py ~/Movies/trip_japan/
```

Once you're in, just chat. For example:

```
you ▸ Here's my cooking video ~/Movies/cooking.mp4. Keep only the parts where I'm explaining the steps, ~2 min.
  ✓ add_video(path=~/Movies/cooking.mp4) → {'video_id': 'v1', ...}
  ✓ get_transcript(video_id=v1) → {...}
  ✓ create_cut(clips=[...]) → {'output_path': '.../output/vlog_xxx.mp4', ...}

agent ▸ Picked 6 clips of step-by-step cooking, total length 1:58.
        Output: output/vlog_xxx.mp4
```

You can iterate freely: "shorter", "drop clip 3", "keep more of the seasoning part" — the agent recomputes the cut and renders a new file.

### Multi-video mode

Once you've loaded 10+ clips, the agent shifts strategy: for each video it sees a short *snippet* (head + tail of the transcript) injected into its system prompt, so it can scan the whole library at a glance without reading every transcript. For theme-based goals it reaches for `search_segments` ("find every mention of coffee across all clips") instead of pulling full transcripts. Default clip ordering in the final cut is by source filename, which matches capture-time naming for most cameras and phones — say "order by topic" or "best moment first" if you want something else.

```
you ▸ I dropped 23 trip clips into ~/Movies/trip_japan. Make me a 3-minute highlight focused on food.
  ✓ list_videos() → [23 entries]
  ✓ search_segments(query=food) → 41 matches across 14 videos
  ✓ get_transcript(video_id=v07) → {...}
  ✓ get_transcript(video_id=v12) → {...}
  ✓ create_cut(clips=[18 clips]) → output/japan_food_xxx.mp4

agent ▸ Built a 3:02 cut from 18 food moments across 14 of your 23 clips.
        Ordered by capture day (filename ascending). Output: output/japan_food_xxx.mp4
```

### Built-in slash commands

| Command | What it does |
|---------|--------------|
| `/add <path>` | Register a single video |
| `/add_dir <path>` | Bulk-register every video file inside a folder (top-level only) |
| `/list` | List the videos currently registered (with snippets) |
| `/reset` | Clear conversation history (keeps loaded videos) |
| `/quit` | Exit |
| `/help` | Show help |

## Project layout

```
vlog_agent/
├── main.py              # CLI entry point
├── agent.py             # VlogAgent + LLM function-calling loop
├── transcribe.py        # Whisper transcription (with file-fingerprint cache)
├── vision.py            # Frame-level visual tagging via GLM-4V-Flash (cached)
├── editor.py            # FFmpeg cut + concat
├── validators.py        # Pre-flight checks before ffmpeg runs (reject-and-retry)
├── tracing.py           # Tracer — dumps every conversation turn to JSON
├── viewer.py            # Builds a self-contained HTML viewer for traces
├── evals/               # Behavioral regression tests (offline + real-LLM)
├── tests/               # pytest unit tests for pure functions
├── .github/workflows/   # CI: ruff + pytest + offline evals
├── pyproject.toml       # ruff / pytest config + project metadata
├── requirements.txt     # Runtime deps
├── requirements-dev.txt # Dev / CI deps
├── workspace/           # Transcript + visuals cache (auto-created)
├── output/              # Finished cuts (auto-created)
└── traces/              # Conversation traces (auto-created, one folder per session)
```

## Vision (optional)

For goals that depend on what was *shown* rather than what was *said* —
scenery, food, faces, animals, action shots — the agent can analyze frames
via GLM-4V-Flash (free) and tag each one with subject / scene / action /
interest score. Tell it in chat:

```
you ▸ Make a 1-minute travel highlight, pick the most cinematic shots.
  ✓ analyze_video_visuals(video_id=v1)        ← 30 frames analyzed, ~45s
  ✓ search_visuals(query=scenery)             ← 12 high-interest frames
  ✓ create_cut(clips=[8 cinematic moments])   ← rendered to output/
```

Vision analysis runs only when you ask for it — `add_video` stays cheap.
Results are cached to `workspace/`, so re-running on the same video is
instant. Cap how many frames are sampled with `VISION_MAX_FRAMES` in
`.env` (default 30, capping first-run time at 30–60 seconds per video).

## Development & CI

Three layers of automated checks run in CI on every push (see
`.github/workflows/ci.yml`):

```bash
pip install -r requirements-dev.txt   # ruff + pytest

ruff check .                          # lint
pytest                                # unit tests for pure functions
python -m evals.runner --offline      # behavioral regression tests
```

CI does **not** install `requirements.txt` — `agent.py`, `vision.py`,
and `transcribe.py` lazy-import `openai` and `whisper`, so the test
matrix runs in ~30 s instead of pulling PyTorch (≈500 MB) every time.
A dedicated CI step verifies the lazy-import invariant stays intact.

The Python matrix tests on 3.10 / 3.11 / 3.12.

## Eval harness

The `evals/` directory holds behavioral regression tests — each case
declares a synthetic project setup, a user prompt, and a list of
assertions the resulting agent run must satisfy.

```bash
# Deterministic offline run (no API tokens spent, uses scripted LLM responses):
python -m evals.runner --offline

# Real LLM run (uses your .env's LLM_API_KEY):
python -m evals.runner
```

Each case asserts things like *"the agent must call set_goal with
target_duration_sec=30"*, *"final clips must include the word 'espresso'
but not 'weather'"*, *"total duration in [25, 35] seconds"*, *"agent
must not call analyze_video_visuals"*. Tweak the system prompt or
switch models, run the suite, and you get an objective pass/fail
report instead of vibing it.

See `evals/README.md` for the case format and the catalog of available
checks.

## Validation & self-correction

Before each `create_cut` actually runs ffmpeg, the agent runs a pre-flight
check: clip bounds, durations, duplicates, overlaps, plus any goal you
registered (target length, must-include / must-exclude keywords, clip
count). If anything fails, the LLM gets back a list of issue descriptions
and is instructed to fix the offending clips and try again — no half-baked
mp4 ever leaves the system.

Tell the agent your goal in chat and it will call `set_goal` for you:

```
you ▸ Cut a 2-minute highlight, must include the espresso shot, no rambling about weather.
  ✓ set_goal(target_duration_sec=120, must_include_keywords=[espresso shot], must_exclude_keywords=[weather])
  ...
```

When the LLM proposes a cut that doesn't fit, you'll see the rejection
in the trace: `validation_failed: True, issues: [...]` followed by a
revised `create_cut` attempt that addresses each issue.

## Inspecting what the agent did

Every chat session writes a structured JSON file per turn under
`traces/<session_id>/turn_NNN.json`. Each turn captures the user input,
every LLM API call (with token counts and latency), every tool call
(with arguments, result, duration), and the final assistant reply.

To browse them, build the static viewer:

```bash
python viewer.py --open
```

This writes `traces/viewer.html` and opens it in your browser. No server
needed — the data is inlined into the HTML and works from `file://`.
Re-run after each session to refresh.

## Workflow

```
user chat
   │
   ▼
VlogAgent ──► LLM (decides which tools to call)
   │             │
   │             ├─► add_video / add_videos_from_dir → transcribe.py → Whisper
   │             ├─► get_transcript                  → cache
   │             ├─► search_segments                 → substring match across all
   │             ├─► list_videos                     → project state + snippets
   │             └─► create_cut                      → editor.py    → FFmpeg
   │
   ▼
output/*.mp4 + a natural-language summary
```

## Tweaks

- **Transcription too slow** — `WHISPER_MODEL=tiny` is faster but coarser; `small`/`medium` are more accurate but slower.
- **Want a stronger LLM** — default `glm-4-flash` is free but not the smartest. Try `glm-4-plus` (Zhipu's flagship) or `glm-4-air` (balanced).
- **Want a different OpenAI-compatible provider** — the code uses a vanilla OpenAI client, so just change these three variables in `.env`. **No code edits needed.**

  | Provider | `LLM_BASE_URL` | `LLM_MODEL` |
  |----------|----------------|-------------|
  | Zhipu GLM (free, default) | `https://open.bigmodel.cn/api/paas/v4/` | `glm-4-flash` |
  | Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | `gemini-2.0-flash` |
  | Alibaba Qwen | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `qwen-plus` |
  | DeepSeek | `https://api.deepseek.com/v1` | `deepseek-chat` |
  | Moonshot Kimi | `https://api.moonshot.cn/v1` | `moonshot-v1-8k` |
- **Cut points are off** — Whisper segment boundaries are routinely ±0.5s; the agent is already instructed to leave a 0.3s buffer.

## Known limitations (room for v2)

- No transitions, BGM, or burned-in subtitles yet.
- `search_segments` is plain case-insensitive substring — no fuzzy / semantic search yet.
- When source clips have different resolutions, the concatenated output keeps the per-clip resolution rather than normalizing to a single canvas. Most players handle it, some don't.
- CLI only, no GUI.
