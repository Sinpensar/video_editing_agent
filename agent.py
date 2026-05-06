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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from dotenv import load_dotenv

import editor
import transcribe

load_dotenv()

DEFAULT_MODEL = os.getenv("LLM_MODEL", "glm-4-flash")
DEFAULT_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://open.bigmodel.cn/api/paas/v4/",
)

SYSTEM_PROMPT = """You are VlogAgent, a senior video editor that turns raw vlog footage into a polished short video, controlled entirely through chat.

Your workflow:
1. The user gives you one or more video files plus a goal (e.g. "keep only the cooking parts", "make a 2-minute travel highlight", "cut out all the rambling").
2. Call `add_video` for each file. This transcribes the audio so you can read what was said and when.
3. Call `get_transcript` to read the segments of each video.
4. Decide which sub-clips to keep and in what order, based on the transcript and the user's goal. You may pick any (start, end) window inside a segment — even a slice of a long one.
5. Call `create_cut` with the chosen clips. Report the returned output path back to the user.
6. Iterate: the user may say "shorter", "drop the second clip", "keep the part about coffee", etc. Recompute the cut and call `create_cut` again with a new filename.

Important rules:
- Ground every editing decision in the real transcript. Don't hallucinate content.
- All start/end values are in seconds, at most two decimals.
- When picking a clip, leave a ~0.3s buffer at the start and end if possible to avoid clipping word boundaries.
- Reply in the user's language, in plain prose. Tell them how long the cut ended up being, which segments you kept, and why.
- If the user's request is too vague (e.g. just "edit this for me"), ask one clarifying question before transcribing.
- If a video is long (>10 min) and not yet transcribed, warn the user that transcription may take a while before calling add_video.
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

    def _summary(self, entry: VideoEntry) -> dict:
        tr = entry.transcript
        return {
            "video_id": entry.video_id,
            "path": str(entry.path),
            "filename": entry.path.name,
            "language": tr.get("language"),
            "duration_sec": tr.get("duration"),
            "segment_count": len(tr.get("segments", [])),
        }

    def list_videos(self) -> list[dict]:
        return [self._summary(e) for e in self.videos.values()]

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

    # -- dynamic system prompt ---------------------------------------------
    def _build_system_prompt(self) -> str:
        videos = self.project.list_videos()
        if not videos:
            return SYSTEM_PROMPT
        lines = ["", "", "Videos already loaded in the project "
                         "(reference these video_ids directly — don't ask the user for a path):"]
        for v in videos:
            dur = v.get("duration_sec") or 0
            lines.append(
                f"- {v['video_id']}: {v['filename']} "
                f"({dur:.1f}s, {v['segment_count']} segments, lang={v.get('language')})"
            )
        return SYSTEM_PROMPT + "\n".join(lines)

    # -- tool dispatch ------------------------------------------------------
    def _dispatch(self, name: str, args: dict) -> Any:
        if name == "add_video":
            return self.project.add(args["path"])
        if name == "list_videos":
            return self.project.list_videos()
        if name == "get_transcript":
            return self.project.get_transcript(args["video_id"])
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
        # Refresh system prompt with current project state every turn.
        self.messages[0] = {"role": "system", "content": self._build_system_prompt()}
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(self.max_tool_iters):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                # tool_choice defaults to "auto" — let the model decide.
            )
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
                return (msg.content or "").strip()

            # Execute every tool call, append a "tool" message per call.
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = self._dispatch(name, args)
                except Exception as e:
                    result = {"error": f"Tool crashed: {e}"}

                if on_tool:
                    on_tool(name, args, result)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        return ("(Hit the tool-call limit without finishing. "
                "Please rephrase or break the request into smaller pieces.)")
