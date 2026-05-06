# VlogAgent

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
# or pre-load some footage so the agent already knows about it:
python main.py ~/Movies/raw/clip1.mp4 ~/Movies/raw/clip2.mp4
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

### Built-in slash commands

| Command | What it does |
|---------|--------------|
| `/add <path>` | Register a video directly (you can also just tell the agent in chat) |
| `/list` | List the videos currently registered |
| `/reset` | Clear conversation history (keeps loaded videos) |
| `/quit` | Exit |
| `/help` | Show help |

## Project layout

```
vlog_agent/
├── main.py          # CLI entry point
├── agent.py         # VlogAgent + LLM function-calling loop
├── transcribe.py    # Whisper transcription (with file-fingerprint cache)
├── editor.py        # FFmpeg cut + concat
├── workspace/       # Transcript cache (auto-created)
├── output/          # Finished cuts (auto-created)
└── requirements.txt
```

## Workflow

```
user chat
   │
   ▼
VlogAgent ──► LLM (decides which tools to call)
   │             │
   │             ├─► add_video      → transcribe.py → Whisper
   │             ├─► get_transcript → cache
   │             ├─► list_videos    → project state
   │             └─► create_cut     → editor.py    → FFmpeg
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
- Multi-video order follows what the user tells the agent — there's no auto-merge by file metadata or timestamps.
- CLI only, no GUI.
