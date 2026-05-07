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

from openai import OpenAI
from dotenv import load_dotenv

import editor
import transcribe

if TYPE_CHECKING:
    from tracing import Tracer

load_dotenv()

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
- `create_cut(clips, output_filename?)` — splice (video_id, start, end) tuples into one mp4

Workflow for a single video:
1. User gives a file + a goal. Call `add_video`, then `get_transcript`, then pick clips, then `create_cut`.

Workflow for many videos (10+):
1. The current project state — including a short snippet of each video — is injected into this system prompt. Read those snippets first; they give you a feel for what each video is about without spending tokens on full transcripts.
2. If the goal is theme-based ("everything about coffee", "all the cooking parts"), call `search_segments` with the relevant keyword instead of reading every transcript end-to-end.
3. Only call `get_transcript` on videos that actually look promising from snippets/searches.
4. Compose the cut. Default ordering: ascending by source filename (which usually corresponds to capture time). If the user explicitly asks for a different order ("by topic", "best first"), follow their lead.
5. Call `create_cut` and report the output path.

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
        "registered video. Returns the absolute output path and the final "
        "duration.",
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
            },
            "required": ["clips"],
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

    def next_id(self) -> str:
        return f"v{len(self.videos) + 1}"

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

    def create_cut(self, clips: list[dict], output_filename: str | None) -> dict:
        if not clips:
            return {"error": "No clips provided."}

        resolved: list[editor.Clip] = []
        for i, c in enumerate(clips):
            vid = c.get("video_id")
            entry = self.videos.get(vid)
            if not entry:
                return {"error": f"clips[{i}] references unknown video_id={vid}"}
            start = float(c["start"])
            end = float(c["end"])
            if end <= start:
                return {"error": f"clips[{i}] has end<=start ({start} >= {end})"}
            resolved.append({
                "video": str(entry.path),
                "start": start,
                "end": end,
            })

        try:
            out_path = editor.assemble(resolved, output_filename=output_filename)
        except Exception as e:
            return {"error": f"Assembly failed: {e}"}

        total = sum(c["end"] - c["start"] for c in resolved)
        return {
            "output_path": str(out_path),
            "view_link": f"computer://{out_path}",
            "duration_sec": round(total, 2),
            "clip_count": len(resolved),
        }


class VlogAgent:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        max_tool_iters: int = 10,
        tracer: "Optional[Tracer]" = None,
    ):
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_API_KEY not set. Copy .env.example to .env and fill it in."
            )
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
        if not videos:
            return SYSTEM_PROMPT
        lines = ["", "",
                 "Videos already loaded in the project "
                 "(reference these video_ids directly — don't ask the user for a path):"]
        for v in videos:
            dur = v.get("duration_sec") or 0
            snippet = (v.get("snippet") or "").strip()
            lines.append(
                f"- {v['video_id']} | {v['filename']} | "
                f"{dur:.1f}s | {v['segment_count']} segs | lang={v.get('language')}"
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
                args["clips"], args.get("output_filename")
            )
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
