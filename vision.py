"""
Frame-level visual analysis via a vision-capable LLM (default: GLM-4V-Flash).

Produces compact structured tags per sampled frame. The output is cached
under workspace/<stem>.<fingerprint>.<model>.visuals.json so re-running
on the same file is free.

Output shape:
    {
        "video": str,
        "model": str,
        "sample_fps": float,
        "frame_count": int,
        "frames": [
            {
                "id": int,
                "time": float,                  # seconds into source
                "tags": {
                    "subject": str,             # 1-3 words
                    "scene": str,               # short enum-like
                    "action": "still"|"slow"|"fast",
                    "interest": int,            # 1-10 cinematic appeal
                    "tags": [str, ...],         # 1-4 keyword tags
                    "note": str,                # short caption ≤15 words
                }
            },
            ...
        ]
    }

Why structured tags rather than free-form descriptions?
  - Tokens. A 10-min vlog can produce 25-30 frames; per-frame structured
    output is ~30 tokens vs. ~120 for prose. The main planning LLM sees
    a 5x smaller visual context.
  - Composability. The agent can grep / filter / sort tags directly,
    no second-pass parsing.
"""

from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

from transcribe import _file_fingerprint   # reuse the same fingerprint scheme

WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

# ---- defaults --------------------------------------------------------------

DEFAULT_VISION_MODEL = os.getenv("VISION_MODEL", "glm-4v-flash")
DEFAULT_VISION_BASE_URL = os.getenv(
    "VISION_BASE_URL",
    os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"),
)
DEFAULT_MAX_FRAMES = int(os.getenv("VISION_MAX_FRAMES", "30"))
DEFAULT_MAX_WIDTH = 640        # downscale frames before base64 (saves bandwidth)


# Asking for one frame at a time keeps the prompt simple and the parser robust.
# A grid-batched version is a planned optimization; see the docstring.
PROMPT_SINGLE_FRAME = """You are tagging frames for an automatic video editor.

Examine the image and return ONE JSON object with these exact keys:
- "subject":   1-3 word noun phrase for the main subject (e.g. "person cooking", "city skyline", "cat close-up", "empty room")
- "scene":     one of "indoor", "outdoor_urban", "outdoor_nature", "indoor_kitchen", "indoor_room", "studio", "vehicle", "other"
- "action":    one of "still", "slow", "fast"   (motion intensity)
- "interest":  integer 1-10  (1 = bland filler, 10 = highly engaging / cinematic)
- "tags":      array of 1-4 short keyword tags, lowercase
- "note":      ≤15 word caption

Output ONLY the JSON object — no markdown fences, no commentary, no preamble."""


# ---- ffmpeg helpers --------------------------------------------------------

def _video_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {res.stderr}")
    return float(res.stdout.strip() or 0)


def _extract_frame(video_path: Path, time_sec: float, out_path: Path,
                   max_width: int = DEFAULT_MAX_WIDTH) -> None:
    cmd = [
        "ffmpeg",
        "-ss", f"{time_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale='min({max_width},iw)':-2",
        "-q:v", "3",
        "-y",
        str(out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"frame extraction at {time_sec}s failed:\n"
                           f"{res.stderr[-500:]}")


def _b64_jpeg(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode()


# ---- vision API helpers ----------------------------------------------------

def _build_vision_message(prompt: str, image_path: Path) -> dict:
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{_b64_jpeg(image_path)}"
                },
            },
        ],
    }


_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _parse_vision_json(text: str) -> dict:
    """Extract a JSON object from the vision model's reply, tolerating
    common formatting tics (markdown fences, leading prose)."""
    if not text:
        return {"_parse_error": "empty response"}

    # Strip markdown fences if present
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)

    text = text.strip()

    # Walk to first { and try to balance braces
    start = text.find("{")
    if start < 0:
        return {"_parse_error": "no JSON object in reply", "_raw": text[:300]}
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError as e:
                    return {"_parse_error": str(e), "_raw": candidate[:300]}
    return {"_parse_error": "unbalanced braces", "_raw": text[:300]}


# ---- main entry point ------------------------------------------------------

def analyze_visuals(
    video_path: str | Path,
    *,
    sample_fps: Optional[float] = None,
    max_frames: int = DEFAULT_MAX_FRAMES,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    force: bool = False,
    client: Any = None,                # injectable for tests
) -> dict:
    """Sample frames, tag them via the vision LLM, and cache the result."""
    video_path = Path(video_path).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    model_name = model_name or DEFAULT_VISION_MODEL
    base_url = base_url or DEFAULT_VISION_BASE_URL
    api_key = api_key or os.getenv("LLM_API_KEY")

    fp = _file_fingerprint(video_path)
    cache_path = WORKSPACE / f"{video_path.stem}.{fp}.{model_name}.visuals.json"

    if cache_path.exists() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    # Compute sampling timestamps -------------------------------------------
    duration = _video_duration(video_path)
    if duration <= 0:
        raise RuntimeError("Could not determine video duration.")

    if sample_fps is None:
        # Default: pick at most max_frames frames, evenly spaced.
        n = min(max_frames, max(1, int(duration / 2)))   # at least every ~2s
        interval = duration / n
        sample_fps = 1.0 / interval
    else:
        interval = 1.0 / sample_fps
        n = min(max_frames, max(1, int(duration / interval)))

    times = [round((i + 0.5) * interval, 2) for i in range(n)]

    # Build (or reuse) an OpenAI-compatible client --------------------------
    if client is None:
        if not api_key:
            raise RuntimeError("LLM_API_KEY not set; cannot call vision API.")
        from openai import OpenAI   # lazy
        client = OpenAI(api_key=api_key, base_url=base_url)

    frames: list[dict] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, t in enumerate(times):
            frame_path = tmp_dir / f"f_{i:04d}.jpg"
            _extract_frame(video_path, t, frame_path)
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[_build_vision_message(PROMPT_SINGLE_FRAME, frame_path)],
                    temperature=0.1,
                )
                tags = _parse_vision_json(resp.choices[0].message.content or "")
            except Exception as e:
                tags = {"_api_error": str(e)}
            frames.append({"id": i, "time": t, "tags": tags})

    output = {
        "video": str(video_path),
        "model": model_name,
        "sample_fps": round(sample_fps, 4),
        "frame_count": len(frames),
        "frames": frames,
    }
    cache_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


# ---- search helpers (used by Project.search_visuals) -----------------------

def frame_search_text(frame: dict) -> str:
    """Flatten a frame's tags into a single lower-case string for substring
    search. Stable across malformed `tags` payloads."""
    tags = frame.get("tags") or {}
    if not isinstance(tags, dict):
        return ""
    pieces: list[str] = []
    for key in ("subject", "scene", "action", "note"):
        v = tags.get(key)
        if isinstance(v, str):
            pieces.append(v)
    extra = tags.get("tags")
    if isinstance(extra, list):
        pieces.extend(str(t) for t in extra if isinstance(t, (str, int, float)))
    return " ".join(pieces).lower()
