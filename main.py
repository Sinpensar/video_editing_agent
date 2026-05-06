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

from agent import VlogAgent


HELP = """\
Commands (anything else is sent to the agent as chat):
  /add <path>      register a video (you can also just tell the agent in chat)
  /list            list registered videos
  /reset           clear conversation history (keeps loaded videos)
  /quit, /exit     leave
  /help            show this help
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
    try:
        agent = VlogAgent()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Pre-load any positional args as videos
    for arg in sys.argv[1:]:
        p = Path(arg).expanduser()
        if not p.exists():
            print(f"(skipping non-existent: {p})")
            continue
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
                    print(f"  {v['video_id']}: {v['filename']} "
                          f"({v['duration_sec']:.1f}s)")
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
