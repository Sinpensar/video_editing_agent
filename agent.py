"""
VlogAgent — a conversational agent that edits vlogs.

Defaults to Zhipu GLM-4-Flash (free, no card required). The code uses
a plain OpenAI client, so any OpenAI-compatible endpoint works — switch
providers by changing LLM_BASE_URL / LLM_MODEL / LLM_API_KEY in .env.

Architecture:
- The user chats with the LLM.
- The LLM has access to four tools (function calling):
    add_video       — register a video & transcribe it
    list_videos     — show what's currently in the project
    get_transcript  — read back the timestamped transcript of one video
    create_cut      — produce an mp4 from a list of {video_id, start, end}
- Each tool call is dispatched to a Python function below.
- The conversation loop keeps running until the LLM returns a final
  text response (no more tool calls), at which point we hand control
  back to the user.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass   # dotenv is optional; .env will just be ignored if missing

import editor
import transcribe
import validators
import vision

if TYPE_CHECKING:
    from tracing import Tracer

DEFAULT_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
DEFAULT_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://open.bigmodel.cn/api/paas/v4/",
)

SYSTEM_PROMPT = """You are VlogAgent, a senior video editor that turns raw vlog footage into a polished short video, controlled entirely through chat.

You can handle anywhere from a single clip to dozens of clips at once. The available tools are:
- `add_video(path)` — register & transcribe one file
- `add_videos_from_dir(path)` — bulk-register every video in a folder
- `list_videos` — see what's loaded, with a short snippet of each
- `get_transcript(video_id)` — full timestamped transcript of one video
- `search_segments(query, video_ids?)` — find every mention of a keyword across all (or selected) videos
- `analyze_video_visuals(video_id)` — opt-in: sample frames and tag each one (subject, scene, interest, etc.). Slow on first run. Required before search_visuals/get_visuals can return data.
- `search_visuals(query, video_ids?)` — find every frame matching a visual keyword across analyzed videos
- `get_visuals(video_id)` — full per-frame visual tags
- `set_goal(...)` — register the user's editing target (length, keywords, clip count). Validators use this.
- `create_cut(clips, output_filename?)` — splice (video_id, start, end) tuples into one mp4

Speech vs vision — choose the right perception tool:
- If the user's goal is content-based (what was SAID) — coffee, cooking steps, jokes, story arc — use the transcript tools (get_transcript, search_segments). Free and fast.
- If the goal is visual (what was SHOWN) — scenery, action, faces, animals, outfits — call analyze_video_visuals first, then search_visuals / get_visuals. This is slow, so warn the user before kicking it off, and only do it for videos you actually need.
- For mixed goals, transcripts are usually enough; only escalate to visuals when the transcript misses something important (e.g. silent stretches with interesting footage).

Workflow for a single video:
1. If the user mentioned a target length, required topics, or things to avoid, call `set_goal(...)` first.
2. Call `add_video`, then `get_transcript`, then pick clips, then `create_cut`.

Workflow for many videos (10+):
1. The current project state — including a short snippet of each video — is injected into this system prompt. Read those snippets first; they give you a feel for what each video is about without spending tokens on full transcripts.
2. If the user gave a clear goal, call `set_goal(...)` early so validators can enforce it.
3. If the goal is theme-based ("everything about coffee", "all the cooking parts"), call `search_segments` instead of reading every transcript end-to-end.
4. Only call `get_transcript` on videos that actually look promising from snippets/searches.
5. Compose the cut. Default ordering: ascending by source filename (which usually corresponds to capture time). If the user explicitly asks for a different order, follow their lead.
6. Call `create_cut` and report the output path.

Validation & self-correction:
- Every `create_cut` runs pre-flight checks (clip bounds, durations, duplicates, overlaps, goal compliance).
- If the response contains `validation_failed: true`, READ each item in `issues`, FIX the offending clips, and call `create_cut` again. Do NOT pass `force=true` unless the user explicitly accepts the warnings.
- Common fixes: shorten / extend a clip, drop a duplicate, swap two overlapping clips, add or remove clips to hit the target duration, replace a clip whose transcript misses a required keyword.

Important rules:
- Ground every editing decision in real transcript content. Don't hallucinate.
- All start/end values are in seconds, at most two decimals.
- When picking a clip, leave a ~0.3s buffer at the start and end if possible so you don't clip word boundaries.
- Reply in the user's language, in plain prose. Mention the final length, which clips you kept, and why.
- If the user's request is too vague (e.g. just "edit this for me"), ask one clarifying question before doing heavy work.
- Before bulk-loading a big folder, briefly tell the user "this will transcribe N files, may take a few minutes" so they're not staring at a frozen screen.
"""


def _function(name: str, description: str, parameters: dict) -> dict:
    """Helper to build an OpenAI-style function tool spec."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


TOOLS: list[dict[str, Any]] = [
    _function(
        "add_video",
        "Register a video file with the project and transcribe its audio. "
        "Returns a video_id you'll use in later tool calls, plus basic stats "
        "(duration, language, segment count). Transcription is cached, so "
        "re-adding the same file is fast.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute or ~-relative path to the video file.",
                }
            },
            "required": ["path"],
        },
    ),
    _function(
        "list_videos",
        "List all videos currently registered in the project, with their "
        "video_id, filename, duration, and segment count.",
        {"type": "object", "properties": {}},
    ),
    _function(
        "get_transcript",
        "Return the full timestamped transcript of one video. Each segment "
        "has id, start (sec), end (sec), text.",
        {
            "type": "object",
            "properties": {
                "video_id": {
                    "type": "string",
                    "description": "The video_id returned by add_video.",
                }
            },
            "required": ["video_id"],
        },
    ),
    _function(
        "add_videos_from_dir",
        "Bulk-import every video file inside a folder. Each one gets "
        "transcribed (with cache, so re-runs are cheap). Returns the list "
        "of added videos and any that failed. Use this instead of calling "
        "add_video repeatedly when the user points at a folder.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Folder path (absolute or ~-relative).",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional glob (e.g. '*.MP4'). Default: "
                                   "all common video extensions.",
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Recurse into subfolders. Default false.",
                },
                "max_videos": {
                    "type": "integer",
                    "description": "Safety cap. Default 100.",
                },
            },
            "required": ["path"],
        },
    ),
    _function(
        "search_segments",
        "Substring search (case-insensitive) across the transcripts of one, "
        "several, or all loaded videos. Use this when there are many videos "
        "and you want to find every mention of a topic without reading "
        "every transcript end-to-end. Returns matching segments with "
        "video_id and timestamps.",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keyword or short phrase to look for.",
                },
                "video_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of video_ids to limit the "
                                   "search. Omit to search every video.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Cap on returned matches. Default 30.",
                },
                "context_segments": {
                    "type": "integer",
                    "description": "How many neighbouring segments to "
                                   "include around each match (0 = match "
                                   "only). Default 0.",
                },
            },
            "required": ["query"],
        },
    ),
    _function(
        "create_cut",
        "Cut and concatenate sub-clips into a single mp4. `clips` is an "
        "ordered list; each item picks a (start, end) window from one "
        "registered video. Pre-flight validation runs before ffmpeg: if it "
        "fails, the call returns {validation_failed:true, issues:[...]} "
        "instead of producing a file. Read the issues, fix the clips, and "
        "call again. Returns absolute output path and final duration on "
        "success.",
        {
            "type": "object",
            "properties": {
                "clips": {
                    "type": "array",
                    "description": "Ordered list of clips to splice together.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "video_id": {"type": "string"},
                            "start": {"type": "number", "description": "seconds"},
                            "end":   {"type": "number", "description": "seconds"},
                            "note":  {"type": "string", "description": "Optional reason for picking this clip."},
                        },
                        "required": ["video_id", "start", "end"],
                    },
                },
                "output_filename": {
                    "type": "string",
                    "description": "Desired output filename (e.g. 'travel_v2.mp4'). "
                                   "Optional — a unique name is generated if omitted.",
                },
                "force": {
                    "type": "boolean",
                    "description": "Bypass validation. Only set true if the "
                                   "user has explicitly accepted the warnings.",
                },
            },
            "required": ["clips"],
        },
    ),
    _function(
        "analyze_video_visuals",
        "Run vision analysis on one registered video — extract sampled "
        "frames and tag each with subject, scene, action, interest score, "
        "tags, and a short caption. Cached on disk, so re-running on the "
        "same file is free. Required before search_visuals or get_visuals "
        "can return data for that video. SLOW: 30-60s per video on first "
        "run; warn the user.",
        {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
                "max_frames": {
                    "type": "integer",
                    "description": "Cap on sampled frames (default 30)."
                },
            },
            "required": ["video_id"],
        },
    ),
    _function(
        "search_visuals",
        "Substring search (case-insensitive) across the visual tags of "
        "loaded videos. Use this for goals that depend on what the camera "
        "saw rather than what was said — e.g. 'find every scenic landscape "
        "shot', 'find frames with food'. Returns matching frames with "
        "video_id + timestamp + tags. Will skip videos that haven't been "
        "analyzed yet (returned in not_analyzed_video_ids).",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "video_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of video_ids to limit "
                                   "the search. Omit to search all.",
                },
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        },
    ),
    _function(
        "get_visuals",
        "Return the full per-frame visual analysis for one video "
        "(after analyze_video_visuals has been run). Each frame includes "
        "id, time (seconds), and tags {subject, scene, action, interest, "
        "tags, note}.",
        {
            "type": "object",
            "properties": {
                "video_id": {"type": "string"},
            },
            "required": ["video_id"],
        },
    ),
    _function(
        "set_goal",
        "Register or update the user's editing goal. Validators use this "
        "when checking each create_cut. Call this near the start of a "
        "session whenever the user gives a clear target (e.g. '2 minutes', "
        "'must include the cooking part', 'no rambling about the weather'). "
        "Pass any subset of fields; existing fields are preserved. Pass "
        "null to clear a single field.",
        {
            "type": "object",
            "properties": {
                "target_duration_sec": {
                    "type": "number",
                    "description": "Target final length in seconds (±25% by default).",
                },
                "min_clips": {"type": "integer"},
                "max_clips": {"type": "integer"},
                "must_include_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords/phrases that must appear in the "
                                   "kept content's transcript.",
                },
                "must_exclude_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keywords/phrases that must NOT appear in "
                                   "the kept content's transcript.",
                },
            },
        },
    ),
]


@dataclass
class VideoEntry:
    video_id: str
    path: Path
    transcript: dict  # output of transcribe.transcribe()

    @property
    def mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0

    def snippet(self, head_chars: int = 140, tail_chars: int = 80) -> str:
        """Short preview of the spoken content — head + tail of the transcript.

        Used in the dynamic system prompt so the LLM gets a sense of each
        video without us shoving every segment into context.
        """
        segs = self.transcript.get("segments", [])
        if not segs:
            return "(silent / no transcript)"
        all_text = " ".join(s["text"] for s in segs).strip()
        if len(all_text) <= head_chars + tail_chars + 5:
            return all_text
        return f"{all_text[:head_chars].strip()} … {all_text[-tail_chars:].strip()}"


@dataclass
class Project:
    videos: dict[str, VideoEntry] = field(default_factory=dict)
    # User-declared editing goal — used by validators. None until set_goal.
    goal: dict = field(default_factory=dict)
    # video_id → output of vision.analyze_visuals(). Populated lazily.
    visuals: dict[str, dict] = field(default_factory=dict)

    def next_id(self) -> str:
        return f"v{len(self.videos) + 1}"

    def set_goal(self, **fields) -> dict:
        """Register / update the user's editing goal.

        Any of: target_duration_sec, min_clips, max_clips,
        must_include_keywords, must_exclude_keywords. Pass None to clear
        an individual field. Returns the resulting goal dict.
        """
        for k, v in fields.items():
            if v is None:
                self.goal.pop(k, None)
            else:
                self.goal[k] = v
        return {"goal": dict(self.goal), "ok": True}

    def clear_goal(self) -> dict:
        self.goal = {}
        return {"goal": {}, "ok": True}

    def add(self, path: str) -> dict:
        path_obj = Path(path).expanduser().resolve()
        if not path_obj.exists():
            return {"error": f"File not found: {path_obj}"}

        # If we already have this exact path, return existing
        for entry in self.videos.values():
            if entry.path == path_obj:
                return self._summary(entry)

        try:
            tr = transcribe.transcribe(path_obj)
        except Exception as e:
            return {"error": f"Transcription failed: {e}"}

        vid = self.next_id()
        entry = VideoEntry(video_id=vid, path=path_obj, transcript=tr)
        self.videos[vid] = entry
        return self._summary(entry)

    def _summary(self, entry: VideoEntry, *, include_snippet: bool = False) -> dict:
        tr = entry.transcript
        out = {
            "video_id": entry.video_id,
            "path": str(entry.path),
            "filename": entry.path.name,
            "language": tr.get("language"),
            "duration_sec": tr.get("duration"),
            "segment_count": len(tr.get("segments", [])),
            "mtime": entry.mtime,
        }
        if include_snippet:
            out["snippet"] = entry.snippet()
        return out

    def list_videos(self) -> list[dict]:
        # Sorted by filename so the LLM sees a stable, time-friendly order.
        entries = sorted(self.videos.values(), key=lambda e: e.path.name)
        return [self._summary(e, include_snippet=True) for e in entries]

    def get_transcript(self, video_id: str) -> dict:
        entry = self.videos.get(video_id)
        if not entry:
            return {"error": f"No such video_id: {video_id}. "
                             f"Known: {list(self.videos)}"}
        return {
            "video_id": video_id,
            "filename": entry.path.name,
            "language": entry.transcript.get("language"),
            "duration_sec": entry.transcript.get("duration"),
            "segments": entry.transcript.get("segments", []),
        }

    def search_segments(
        self,
        query: str,
        video_ids: list[str] | None = None,
        max_results: int = 30,
        context_segments: int = 0,
    ) -> dict:
        """Substring search across one or more transcripts.

        Case-insensitive. If `video_ids` is None or empty, search all
        registered videos. `context_segments` controls how many neighbouring
        segments to attach for context (0 = match only).
        """
        if not query or not query.strip():
            return {"error": "Empty query."}
        q = query.strip().lower()

        targets: list[VideoEntry]
        if video_ids:
            unknown = [v for v in video_ids if v not in self.videos]
            if unknown:
                return {"error": f"Unknown video_id(s): {unknown}"}
            targets = [self.videos[v] for v in video_ids]
        else:
            targets = list(self.videos.values())

        hits: list[dict] = []
        for entry in targets:
            segs = entry.transcript.get("segments", [])
            for i, seg in enumerate(segs):
                if q in seg["text"].lower():
                    lo = max(0, i - context_segments)
                    hi = min(len(segs), i + context_segments + 1)
                    context = [
                        {"id": s["id"], "start": s["start"],
                         "end": s["end"], "text": s["text"]}
                        for s in segs[lo:hi]
                    ]
                    hits.append({
                        "video_id": entry.video_id,
                        "filename": entry.path.name,
                        "match": {
                            "id": seg["id"],
                            "start": seg["start"],
                            "end": seg["end"],
                            "text": seg["text"],
                        },
                        "context": context if context_segments else None,
                    })
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break

        return {
            "query": query,
            "total_matches": len(hits),
            "truncated": len(hits) >= max_results,
            "matches": hits,
        }

    # File extensions we treat as video by default.
    VIDEO_GLOBS: tuple[str, ...] = (
        "*.mp4", "*.MP4", "*.mov", "*.MOV", "*.mkv", "*.MKV",
        "*.m4v", "*.M4V", "*.webm", "*.WEBM", "*.avi", "*.AVI",
    )

    def add_videos_from_dir(
        self,
        path: str,
        pattern: str | None = None,
        recursive: bool = False,
        max_videos: int = 100,
        on_progress=None,
    ) -> dict:
        """Bulk-import every video in a folder.

        on_progress: optional callback(idx, total, filename, result) so the
        CLI can print per-file status while transcription runs.
        """
        dir_path = Path(path).expanduser().resolve()
        if not dir_path.exists():
            return {"error": f"Directory not found: {dir_path}"}
        if not dir_path.is_dir():
            return {"error": f"Not a directory: {dir_path}"}

        if pattern:
            patterns = [pattern]
        else:
            patterns = list(self.VIDEO_GLOBS)

        files: list[Path] = []
        seen: set[Path] = set()
        for p in patterns:
            iterator = dir_path.rglob(p) if recursive else dir_path.glob(p)
            for f in iterator:
                if f.is_file() and f not in seen:
                    seen.add(f)
                    files.append(f)
        files.sort(key=lambda f: f.name)
        if len(files) > max_videos:
            return {
                "error": f"Found {len(files)} videos in {dir_path}, exceeds "
                         f"max_videos={max_videos}. Pass a tighter pattern "
                         f"or raise the limit."
            }
        if not files:
            return {
                "error": f"No videos matched in {dir_path} "
                         f"(patterns: {patterns}, recursive={recursive})"
            }

        added: list[dict] = []
        skipped: list[dict] = []
        for i, f in enumerate(files, start=1):
            res = self.add(str(f))
            if on_progress:
                on_progress(i, len(files), f.name, res)
            if "error" in res:
                skipped.append({"filename": f.name, "error": res["error"]})
            else:
                added.append(res)

        return {
            "directory": str(dir_path),
            "scanned": len(files),
            "added": added,
            "skipped": skipped,
        }

    # -- vision -----------------------------------------------------------

    def analyze_visuals(
        self,
        video_id: str,
        *,
        max_frames: int | None = None,
        force: bool = False,
        client: Any = None,
    ) -> dict:
        """Run vision.analyze_visuals on a registered video and store the
        result in self.visuals[video_id]. Returns a compact summary."""
        entry = self.videos.get(video_id)
        if not entry:
            return {"error": f"No such video_id: {video_id}. "
                             f"Known: {list(self.videos)}"}
        try:
            kwargs: dict = {"force": force}
            if max_frames is not None:
                kwargs["max_frames"] = int(max_frames)
            if client is not None:
                kwargs["client"] = client
            result = vision.analyze_visuals(entry.path, **kwargs)
        except Exception as e:
            return {"error": f"Vision analysis failed: {e}"}

        self.visuals[video_id] = result
        return {
            "video_id": video_id,
            "filename": entry.path.name,
            "model": result.get("model"),
            "frame_count": result.get("frame_count"),
            "sample_fps": result.get("sample_fps"),
        }

    def get_visuals(self, video_id: str) -> dict:
        entry = self.videos.get(video_id)
        if not entry:
            return {"error": f"No such video_id: {video_id}"}
        v = self.visuals.get(video_id)
        if not v:
            return {
                "error": f"Visuals not analyzed yet for {video_id}. "
                         f"Call analyze_video_visuals first."
            }
        return {
            "video_id": video_id,
            "filename": entry.path.name,
            "model": v.get("model"),
            "frame_count": v.get("frame_count"),
            "frames": v.get("frames", []),
        }

    def search_visuals(
        self,
        query: str,
        video_ids: list[str] | None = None,
        max_results: int = 30,
    ) -> dict:
        """Substring search across the visual tags of one, several, or all
        analyzed videos. Returns matching frames with video_id + timestamp."""
        if not query or not query.strip():
            return {"error": "Empty query."}
        q = query.strip().lower()

        if video_ids:
            unknown = [v for v in video_ids if v not in self.videos]
            if unknown:
                return {"error": f"Unknown video_id(s): {unknown}"}
            target_ids = video_ids
        else:
            target_ids = list(self.videos.keys())

        not_analyzed = [vid for vid in target_ids if vid not in self.visuals]
        hits: list[dict] = []
        for vid in target_ids:
            v = self.visuals.get(vid)
            if not v:
                continue
            entry = self.videos[vid]
            for frame in v.get("frames", []):
                text = vision.frame_search_text(frame)
                if q in text:
                    hits.append({
                        "video_id": vid,
                        "filename": entry.path.name,
                        "frame_id": frame.get("id"),
                        "time": frame.get("time"),
                        "tags": frame.get("tags"),
                    })
                    if len(hits) >= max_results:
                        break
            if len(hits) >= max_results:
                break

        return {
            "query": query,
            "total_matches": len(hits),
            "truncated": len(hits) >= max_results,
            "matches": hits,
            "not_analyzed_video_ids": not_analyzed,
        }

    # When True, all create_cut calls validate but skip ffmpeg. Used by
    # the eval harness to check planning behavior without burning compute.
    dry_run_cuts: bool = field(default=False)

    def create_cut(
        self,
        clips: list[dict],
        output_filename: str | None,
        *,
        force: bool = False,
        dry_run: bool = False,
    ) -> dict:
        # ---- pre-flight validation -------------------------------------
        issues = validators.validate_clips(clips, self, goal=self.goal)
        if issues and not force:
            return {
                "validation_failed": True,
                "issues": issues,
                "goal": dict(self.goal) if self.goal else None,
                "hint": (
                    "Read each issue, adjust the offending clips, then call "
                    "create_cut again. Do NOT pass force=true unless the user "
                    "has explicitly accepted the warnings."
                ),
            }

        # ---- resolve video_id → real path -----------------------------
        resolved: list[editor.Clip] = []
        for c in clips:
            entry = self.videos[c["video_id"]]
            resolved.append({
                "video": str(entry.path),
                "start": float(c["start"]),
                "end": float(c["end"]),
            })

        total = sum(c["end"] - c["start"] for c in resolved)

        # If dry_run is on (per-call or project-wide), skip ffmpeg.
        if dry_run or self.dry_run_cuts:
            return {
                "dry_run": True,
                "would_output": output_filename or "vlog_<auto>.mp4",
                "duration_sec": round(total, 2),
                "clip_count": len(resolved),
                "clips": resolved,
                "warnings": issues if issues and force else None,
            }

        try:
            out_path = editor.assemble(resolved, output_filename=output_filename)
        except Exception as e:
            return {"error": f"Assembly failed: {e}"}

        return {
            "output_path": str(out_path),
            "view_link": f"computer://{out_path}",
            "duration_sec": round(total, 2),
            "clip_count": len(resolved),
            "warnings": issues if issues and force else None,
        }


class VlogAgent:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_tool_iters: int = 10,
        tracer: "Optional[Tracer]" = None,
        client: Any = None,    # injectable for tests / eval harness
    ):
        if client is not None:
            self.client = client
        else:
            api_key = os.getenv("LLM_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "LLM_API_KEY not set. Copy .env.example to .env and fill it in."
                )
            from openai import OpenAI   # lazy: keeps `import agent` openai-free
            self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_tool_iters = max_tool_iters
        self.project = Project()
        # System prompt lives at index 0; we refresh it each turn so the
        # model always sees the current project state.
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        # Optional observability layer — None means tracing is off.
        self.tracer = tracer

    # -- dynamic system prompt ---------------------------------------------
    def _build_system_prompt(self) -> str:
        videos = self.project.list_videos()
        goal = self.project.goal
        if not videos and not goal:
            return SYSTEM_PROMPT

        lines: list[str] = []

        if goal:
            lines.append("")
            lines.append("")
            lines.append("Active goal (validators will enforce this on every create_cut):")
            for k, v in goal.items():
                lines.append(f"- {k}: {v}")

        if videos:
            lines.append("")
            lines.append("")
            lines.append("Videos already loaded in the project "
                         "(reference these video_ids directly — don't ask the user for a path):")
            for v in videos:
                dur = v.get("duration_sec") or 0
                snippet = (v.get("snippet") or "").strip()
                vis = self.project.visuals.get(v["video_id"])
                vis_tag = (f"visuals: {vis.get('frame_count')} frames analyzed"
                           if vis else "visuals: not analyzed")
                lines.append(
                    f"- {v['video_id']} | {v['filename']} | "
                    f"{dur:.1f}s | {v['segment_count']} segs | lang={v.get('language')} | {vis_tag}"
                )
                if snippet:
                    lines.append(f"    snippet: {snippet}")

        return SYSTEM_PROMPT + "\n".join(lines)

    # -- tool dispatch ------------------------------------------------------
    def _dispatch(self, name: str, args: dict) -> Any:
        if name == "add_video":
            return self.project.add(args["path"])
        if name == "add_videos_from_dir":
            return self.project.add_videos_from_dir(
                args["path"],
                pattern=args.get("pattern"),
                recursive=bool(args.get("recursive", False)),
                max_videos=int(args.get("max_videos", 100)),
            )
        if name == "list_videos":
            return self.project.list_videos()
        if name == "get_transcript":
            return self.project.get_transcript(args["video_id"])
        if name == "search_segments":
            return self.project.search_segments(
                args["query"],
                video_ids=args.get("video_ids"),
                max_results=int(args.get("max_results", 30)),
                context_segments=int(args.get("context_segments", 0)),
            )
        if name == "create_cut":
            return self.project.create_cut(
                args["clips"],
                args.get("output_filename"),
                force=bool(args.get("force", False)),
            )
        if name == "set_goal":
            return self.project.set_goal(**{
                k: v for k, v in args.items()
                if k in {"target_duration_sec", "min_clips", "max_clips",
                         "must_include_keywords", "must_exclude_keywords"}
            })
        if name == "analyze_video_visuals":
            return self.project.analyze_visuals(
                args["video_id"],
                max_frames=args.get("max_frames"),
            )
        if name == "search_visuals":
            return self.project.search_visuals(
                args["query"],
                video_ids=args.get("video_ids"),
                max_results=int(args.get("max_results", 30)),
            )
        if name == "get_visuals":
            return self.project.get_visuals(args["video_id"])
        return {"error": f"Unknown tool: {name}"}

    # -- single user turn ---------------------------------------------------
    def chat(self, user_input: str, *, on_tool=None) -> str:
        """
        Send one user message, run the tool-use loop, and return the
        model's final assistant text.

        on_tool: optional callback (name, args, result) -> None for live UX.
        """
        if self.tracer:
            self.tracer.begin_turn(user_input)

        # Refresh system prompt with current project state every turn.
        self.messages[0] = {"role": "system", "content": self._build_system_prompt()}
        self.messages.append({"role": "user", "content": user_input})

        final_reply: Optional[str] = None
        try:
            for _ in range(self.max_tool_iters):
                t_llm = time.monotonic()
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOLS,
                    # tool_choice defaults to "auto" — let the model decide.
                )
                latency_ms = int((time.monotonic() - t_llm) * 1000)
                if self.tracer:
                    self.tracer.record_llm_call(self.messages, resp, latency_ms)

                msg = resp.choices[0].message

                # Append the assistant turn back into history.
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or "",
                }
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                self.messages.append(assistant_msg)

                if not msg.tool_calls:
                    # final answer
                    final_reply = (msg.content or "").strip()
                    return final_reply

                # Execute every tool call, append a "tool" message per call.
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    t_tool = time.monotonic()
                    try:
                        result = self._dispatch(name, args)
                    except Exception as e:
                        result = {"error": f"Tool crashed: {e}"}
                    duration_ms = int((time.monotonic() - t_tool) * 1000)
                    is_error = isinstance(result, dict) and "error" in result

                    if self.tracer:
                        self.tracer.record_tool_call(
                            name, args, result, duration_ms, is_error
                        )
                    if on_tool:
                        on_tool(name, args, result)

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    })

            final_reply = ("(Hit the tool-call limit without finishing. "
                           "Please rephrase or break the request into smaller pieces.)")
            return final_reply
        finally:
            if self.tracer:
                self.tracer.end_turn(final_reply)
