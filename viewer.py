"""
Build a self-contained HTML viewer for the conversation traces under
traces/<session_id>/.

Usage:
    python viewer.py                  # writes traces/viewer.html
    python viewer.py --open           # also opens it in your default browser
    python viewer.py --traces other/  # use a different traces directory

The output file is fully offline: every session's data is embedded as
inline JSON, so you can double-click it from Finder.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

DEFAULT_TRACES_DIR = Path(__file__).parent / "traces"


def collect_sessions(traces_dir: Path) -> list[dict]:
    sessions = []
    if not traces_dir.exists():
        return sessions

    for session_dir in sorted(traces_dir.iterdir()):
        if not session_dir.is_dir():
            continue
        meta_path = session_dir / "session.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {"_meta_parse_error": True}
        turns = []
        for turn_path in sorted(session_dir.glob("turn_*.json")):
            try:
                turns.append(json.loads(turn_path.read_text(encoding="utf-8")))
            except Exception as e:
                turns.append({"_parse_error": str(e), "filename": turn_path.name})
        if not turns and not meta:
            continue
        sessions.append({
            "id": session_dir.name,
            "meta": meta,
            "turns": turns,
        })
    return sessions


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>VlogAgent Trace Viewer</title>
<style>
  :root {
    --bg: #1a1a1a;
    --panel: #232323;
    --panel-2: #2a2a2a;
    --border: #3a3a3a;
    --text: #e5e5e5;
    --muted: #999;
    --accent: #ff8a3d;
    --user: #6cb6ff;
    --assistant: #a8e6a3;
    --tool: #ffcb6b;
    --error: #ff6b6b;
    --code-bg: #181818;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace;
    background: var(--bg); color: var(--text); font-size: 13px; line-height: 1.5;
  }
  .layout { display: flex; height: 100vh; }
  .sidebar {
    width: 320px; min-width: 280px; max-width: 420px;
    background: var(--panel); border-right: 1px solid var(--border);
    overflow-y: auto; padding: 12px;
  }
  .main { flex: 1; overflow-y: auto; padding: 20px 28px; }
  h1 { font-size: 14px; margin: 0 0 12px; color: var(--accent); letter-spacing: 1px; text-transform: uppercase; }
  .session {
    border: 1px solid var(--border); border-radius: 6px;
    margin-bottom: 8px; padding: 10px; cursor: pointer;
    transition: background 0.1s;
  }
  .session:hover { background: var(--panel-2); }
  .session.active { background: var(--panel-2); border-color: var(--accent); }
  .session-id { font-weight: 600; color: var(--accent); font-size: 12px; }
  .session-meta { color: var(--muted); font-size: 11px; margin-top: 4px; }
  .turn {
    background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
    margin-bottom: 16px; padding: 16px;
  }
  .turn-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
  }
  .turn-title { font-size: 14px; font-weight: 600; color: var(--accent); }
  .turn-totals { color: var(--muted); font-size: 11px; }
  .user-input {
    background: var(--code-bg); border-left: 3px solid var(--user);
    padding: 8px 12px; border-radius: 4px; margin-bottom: 12px;
    white-space: pre-wrap;
  }
  .final-reply {
    background: var(--code-bg); border-left: 3px solid var(--assistant);
    padding: 8px 12px; border-radius: 4px; margin-top: 12px;
    white-space: pre-wrap;
  }
  details {
    background: var(--panel-2); border: 1px solid var(--border); border-radius: 4px;
    margin: 6px 0; padding: 6px 10px;
  }
  details > summary {
    cursor: pointer; user-select: none;
    list-style: none; outline: none;
  }
  details > summary::before {
    content: "▸ "; color: var(--muted); display: inline-block; transition: transform 0.1s;
  }
  details[open] > summary::before { content: "▾ "; }
  .step-summary { display: flex; gap: 12px; align-items: center; }
  .step-tag {
    background: var(--panel); padding: 2px 6px; border-radius: 3px;
    font-size: 10px; color: var(--muted);
  }
  .step-tag.tool { color: var(--tool); }
  .step-tag.assistant { color: var(--assistant); }
  .step-tag.error { color: var(--error); }
  pre {
    background: var(--code-bg); padding: 10px 12px; border-radius: 4px;
    overflow-x: auto; font-size: 11px; margin: 6px 0; white-space: pre-wrap; word-break: break-word;
  }
  .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 8px; display: block; }
  .empty { color: var(--muted); padding: 40px; text-align: center; font-style: italic; }
  .stats-bar {
    display: flex; gap: 16px; padding: 8px 12px; background: var(--code-bg);
    border-radius: 4px; margin-bottom: 12px; font-size: 11px;
  }
  .stat { display: flex; flex-direction: column; gap: 2px; }
  .stat-value { color: var(--accent); font-weight: 600; font-size: 13px; }
  .stat-label { color: var(--muted); }
</style>
</head>
<body>
<div class="layout">
  <div class="sidebar">
    <h1>Trace Sessions</h1>
    <div id="session-list"></div>
  </div>
  <div class="main" id="main">
    <div class="empty">Select a session on the left.</div>
  </div>
</div>

<script>
const SESSIONS = __DATA__;

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "html") e.innerHTML = v;
    else e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c == null) continue;
    e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return e;
}

function fmtMs(ms) {
  if (ms == null) return "–";
  if (ms < 1000) return ms + " ms";
  return (ms / 1000).toFixed(2) + " s";
}

function fmtNum(n) {
  if (n == null) return "0";
  return n.toLocaleString();
}

function renderSidebar() {
  const list = document.getElementById("session-list");
  list.innerHTML = "";
  if (SESSIONS.length === 0) {
    list.appendChild(el("div", {class: "empty"}, "No traces yet. Run main.py and chat with the agent."));
    return;
  }
  SESSIONS.forEach((s, i) => {
    const node = el("div", {class: "session", "data-i": i}, [
      el("div", {class: "session-id"}, s.id),
      el("div", {class: "session-meta"}, [
        (s.meta.model || "?") + " · " + (s.turns.length) + " turn" + (s.turns.length === 1 ? "" : "s"),
        el("br"),
        s.meta.started_at || "",
      ])
    ]);
    node.addEventListener("click", () => selectSession(i));
    list.appendChild(node);
  });
}

function selectSession(i) {
  document.querySelectorAll(".session").forEach((n, j) => {
    n.classList.toggle("active", i === j);
  });
  renderMain(SESSIONS[i]);
}

function renderToolCall(tc) {
  const headerCls = tc.is_error ? "step-tag error" : "step-tag tool";
  const headerLabel = (tc.is_error ? "ERROR " : "tool ") + tc.name + " · " + fmtMs(tc.duration_ms);
  return el("details", {}, [
    el("summary", {}, [
      el("span", {class: headerCls}, headerLabel),
    ]),
    el("span", {class: "label"}, "arguments"),
    el("pre", {}, JSON.stringify(tc.arguments, null, 2)),
    el("span", {class: "label"}, "result"),
    el("pre", {}, JSON.stringify(tc.result, null, 2)),
  ]);
}

function renderStep(step, toolsForStep) {
  const headerLabel = "step " + step.step + " · "
    + fmtMs(step.latency_ms) + " · "
    + fmtNum(step.input_tokens) + " in / "
    + fmtNum(step.output_tokens) + " out"
    + (step.stop_reason ? " · " + step.stop_reason : "");
  const children = [
    el("summary", {}, [el("span", {class: "step-tag assistant"}, headerLabel)]),
  ];
  if (step.response.content) {
    children.push(el("span", {class: "label"}, "assistant text"));
    children.push(el("pre", {}, step.response.content));
  }
  if (step.response.tool_calls && step.response.tool_calls.length) {
    children.push(el("span", {class: "label"}, "issued tool calls"));
    for (const tc of step.response.tool_calls) {
      children.push(el("pre", {}, tc.name + "(" + JSON.stringify(tc.arguments) + ")"));
    }
  }
  if (toolsForStep.length) {
    children.push(el("span", {class: "label"}, "tool results"));
    for (const tc of toolsForStep) {
      children.push(renderToolCall(tc));
    }
  }
  return el("details", {open: ""}, children);
}

function renderTurn(turn) {
  const totals = turn.totals || {};
  const stats = el("div", {class: "stats-bar"}, [
    el("div", {class: "stat"}, [
      el("span", {class: "stat-value"}, fmtNum(totals.total_tokens)),
      el("span", {class: "stat-label"}, "tokens"),
    ]),
    el("div", {class: "stat"}, [
      el("span", {class: "stat-value"}, fmtMs(totals.llm_latency_ms)),
      el("span", {class: "stat-label"}, "llm latency"),
    ]),
    el("div", {class: "stat"}, [
      el("span", {class: "stat-value"}, fmtMs(totals.tool_duration_ms)),
      el("span", {class: "stat-label"}, "tool time"),
    ]),
    el("div", {class: "stat"}, [
      el("span", {class: "stat-value"}, fmtNum(totals.llm_calls)),
      el("span", {class: "stat-label"}, "llm calls"),
    ]),
    el("div", {class: "stat"}, [
      el("span", {class: "stat-value"}, fmtNum(totals.tool_calls)),
      el("span", {class: "stat-label"}, "tool calls"),
    ]),
  ]);

  const stepNodes = [];
  for (const step of turn.steps || []) {
    const toolsForStep = (turn.tool_calls || []).filter(tc => tc.step === step.step);
    stepNodes.push(renderStep(step, toolsForStep));
  }

  return el("div", {class: "turn"}, [
    el("div", {class: "turn-header"}, [
      el("div", {class: "turn-title"}, "Turn " + turn.turn),
      el("div", {class: "turn-totals"}, (turn.started_at || "") + (turn.ended_at ? " → " + turn.ended_at : "")),
    ]),
    el("span", {class: "label"}, "user"),
    el("div", {class: "user-input"}, turn.user_input || ""),
    stats,
    ...stepNodes,
    turn.final_reply
      ? el("div", {}, [
          el("span", {class: "label"}, "final assistant reply"),
          el("div", {class: "final-reply"}, turn.final_reply),
        ])
      : null,
  ]);
}

function renderMain(session) {
  const main = document.getElementById("main");
  main.innerHTML = "";
  if (!session.turns.length) {
    main.appendChild(el("div", {class: "empty"}, "No turns yet in this session."));
    return;
  }
  main.appendChild(el("h1", {}, "Session " + session.id));
  for (const turn of session.turns) {
    main.appendChild(renderTurn(turn));
  }
}

renderSidebar();
if (SESSIONS.length > 0) selectSession(SESSIONS.length - 1);
</script>
</body>
</html>
"""


def build(traces_dir: Path, out_path: Path) -> int:
    sessions = collect_sessions(traces_dir)
    payload = json.dumps(sessions, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA__", payload)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return len(sessions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default=str(DEFAULT_TRACES_DIR),
                    help="Path to traces directory (default: ./traces)")
    ap.add_argument("--out", default=None,
                    help="Output HTML path (default: <traces>/viewer.html)")
    ap.add_argument("--open", action="store_true",
                    help="Open the viewer in your default browser")
    args = ap.parse_args()

    traces_dir = Path(args.traces).expanduser().resolve()
    out_path = Path(args.out).resolve() if args.out else traces_dir / "viewer.html"

    n = build(traces_dir, out_path)
    if n == 0:
        print(f"⚠ No sessions found in {traces_dir}. Run main.py and chat first.")
    else:
        total_turns = sum(len(list((traces_dir / s).glob("turn_*.json")))
                          for s in [d.name for d in traces_dir.iterdir() if d.is_dir()])
        print(f"✓ Wrote {out_path} ({n} session(s), {total_turns} turn(s)).")
    print(f"  Open: file://{out_path}")
    if args.open:
        webbrowser.open(f"file://{out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
