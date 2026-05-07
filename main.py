"""
CLI entry point for VlogAgent.

Run:
    python main.py
    python main.py /path/to/clip1.mp4 /path/to/clip2.mp4
       (any positional args are pre-loaded as videos before the chat starts)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from agent import VlogAgent, DEFAULT_MODEL, DEFAULT_BASE_URL
from tracing import Tracer


HELP = """\
Commands (anything else is sent to the agent as chat):
  /add <path>          register a single video
  /add_dir <path>      bulk-register every video in a folder (top-level only)
  /list                list registered videos
  /reset               clear conversation history (keeps loaded videos)
  /quit, /exit         leave
  /help                show this help
"""

BANNER = r"""
 _   _ _              _                    _
| | | | | ___   __ _ /_\   __ _  ___ _ __ | |_
| |_| | |/ _ \ / _` ////_\ / _` |/ _ \ '_ \| __|
|  _  | | (_) | (_| /  _  \ (_| |  __/ | | | |_
|_| |_|_|\___/ \__, \_/ \_/\__, |\___|_| |_|\__|
               |___/        |___/
   conversational vlog editor — LLM + Whisper + FFmpeg
"""


def _print_tool(name: str, args: dict, result):
    # Compact one-line print so the chat stays readable
    arg_preview = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
    if isinstance(result, dict) and "error" in result:
        print(f"  ⚠ {name}({arg_preview}) → ERROR: {result['error']}")
    else:
        print(f"  ✓ {name}({arg_preview}) → {_short(result)}")


def _short(value, limit: int = 80) -> str:
    s = repr(value) if not isinstance(value, str) else value
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return s


def main() -> None:
    if not os.getenv("LLM_API_KEY"):
        print("ERROR: LLM_API_KEY is not set.")
        print("       cp .env.example .env  and fill in your LLM key, then re-run.")
        print("       Free key recommendation: Zhipu GLM-4-Flash → https://open.bigmodel.cn/usercenter/proj-mgmt/apikeys")
        sys.exit(1)

    print(BANNER)
    tracer = Tracer(model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL)
    print(f"  trace session: {tracer.session_id} → traces/{tracer.session_id}/\n")

    try:
        agent = VlogAgent(tracer=tracer)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Pre-load any positional args as videos. A directory means bulk-load.
    def _prog(i, total, name, res):
        tag = "⚠" if "error" in res else "✓"
        print(f"  [{i}/{total}] {tag} {name}")

    for arg in sys.argv[1:]:
        p = Path(arg).expanduser()
        if not p.exists():
            print(f"(skipping non-existent: {p})")
            continue
        if p.is_dir():
            print(f"Pre-loading folder {p}...")
            result = agent.project.add_videos_from_dir(str(p), on_progress=_prog)
            if "error" in result:
                print(f"  ⚠ {result['error']}")
            else:
                print(f"  ✓ scanned {result['scanned']}, "
                      f"added {len(result['added'])}, "
                      f"skipped {len(result['skipped'])}")
        else:
            print(f"Pre-loading {p.name}...")
            result = agent.project.add(str(p))
            if "error" in result:
                print(f"  ⚠ {result['error']}")
            else:
                print(f"  ✓ {result['video_id']}: {result['filename']} "
                      f"({result['duration_sec']:.1f}s, {result['segment_count']} segments, "
                      f"lang={result['language']})")

    print(HELP)

    while True:
        try:
            user_input = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd, *rest = user_input.split(maxsplit=1)
            if cmd in ("/quit", "/exit"):
                break
            if cmd == "/help":
                print(HELP)
                continue
            if cmd == "/list":
                vids = agent.project.list_videos()
                if not vids:
                    print("  (no videos loaded)")
                for v in vids:
                    snippet = (v.get("snippet") or "").strip()
                    if len(snippet) > 90:
                        snippet = snippet[:87] + "..."
                    print(f"  {v['video_id']}: {v['filename']} "
                          f"({v['duration_sec']:.1f}s, {v['segment_count']} segs)")
                    if snippet:
                        print(f"      └ {snippet}")
                continue
            if cmd == "/reset":
                # keep system prompt at index 0
                agent.messages = agent.messages[:1]
                print("  (conversation cleared; videos kept)")
                continue
            if cmd == "/add":
                if not rest:
                    print("  usage: /add <path>")
                    continue
                result = agent.project.add(rest[0])
                if "error" in result:
                    print(f"  ⚠ {result['error']}")
                else:
                    print(f"  ✓ added {result['video_id']}: {result['filename']}")
                continue
            if cmd == "/add_dir":
                if not rest:
                    print("  usage: /add_dir <path>")
                    continue
                def _prog(i, total, name, res):
                    tag = "⚠" if "error" in res else "✓"
                    print(f"  [{i}/{total}] {tag} {name}")
                result = agent.project.add_videos_from_dir(
                    rest[0], on_progress=_prog
                )
                if "error" in result:
                    print(f"  ⚠ {result['error']}")
                else:
                    print(f"  ✓ scanned {result['scanned']}, "
                          f"added {len(result['added'])}, "
                          f"skipped {len(result['skipped'])}")
                continue
            print(f"  unknown command: {cmd}")
            continue

        try:
            reply = agent.chat(user_input, on_tool=_print_tool)
        except Exception as e:
            print(f"  ⚠ agent crashed: {e}")
            continue

        print(f"\nagent ▸ {reply}\n")


if __name__ == "__main__":
    main()
