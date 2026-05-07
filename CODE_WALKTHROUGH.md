# VlogAgent 代码逐段讲解

这份文档不会改源码,只解释每个 .py 文件每段代码在做什么、为什么这么写。
对照阅读建议:左边开 `transcribe.py` / `editor.py` / `agent.py` / `main.py`,右边开本文。每节标题里的行号都对应当前文件的实际行号。

---

## 0. 整体架构

```
你 ─ chat ─►  main.py  ──►  agent.py (VlogAgent)
                              │
                              ├──►  transcribe.py  ──►  ffmpeg + Whisper
                              │     (video → 带时间戳字幕)
                              │
                              ├──►  editor.py      ──►  ffmpeg
                              │     (字幕里挑出的 (start,end) → mp4)
                              │
                              └──►  OpenAI 兼容 API (默认智谱 GLM-4-Flash)
                                    (读字幕 + 决定剪哪几段)
```

四个文件的职责:
- **transcribe.py** —— 把视频转成带时间戳的字幕(是「让 LLM 看懂视频」的关键)
- **editor.py** —— 拿到 `(start, end)` 列表后,真正去切视频、拼成片
- **agent.py** —— 对话核心:管理项目状态、定义工具、跑 LLM 工具调用循环
- **main.py** —— CLI 外壳:启动、解析参数、命令行交互、把日志打给你看

数据流:
```
.mp4 文件
  │
  ▼ transcribe.transcribe()
{language, duration, segments:[{id,start,end,text},...]}   ← Whisper 输出
  │
  ▼ Project.add() 包装成 VideoEntry,塞进 self.videos[v1]
  │
  ▼ LLM 通过 get_transcript / search_segments 工具读到这些 segment
  │
  ▼ LLM 决定保留哪些区间 → 调 create_cut(clips=[{video_id,start,end},...])
  │
  ▼ Project.create_cut() 把 video_id 翻译回真实路径,交给 editor.assemble()
  │
  ▼ editor.assemble() → ffmpeg 切片 + concat
  │
  ▼ output/vlog_xxxxxxxx.mp4
```

---

## 1. transcribe.py — 视频转字幕

文件总长 ~145 行。核心函数只有一个:`transcribe(video_path)`。其它都是辅助。

### 1.1 文档字符串 + imports(1–18 行)

```python
"""
Transcribe a video file into timestamped segments using Whisper.
The result is cached to workspace/<sha1>.json so re-running on the same
video is essentially free.
"""

from __future__ import annotations

import hashlib, json, os, subprocess, tempfile
from pathlib import Path
from typing import Optional

import whisper
```

- `from __future__ import annotations` —— 让所有类型注解(像 `dict[str, ...]`)都被当字符串处理,Python 3.9 之前的版本也能跑;同时减少运行时开销。
- `hashlib` —— 算文件指纹,用 SHA1 哈希。
- `subprocess` —— 调外部命令(ffmpeg)。
- `tempfile` —— 创建临时目录存中间的 wav 文件,自动清理。
- `pathlib.Path` —— 比 `os.path` 更面向对象的路径库。
- `whisper` —— OpenAI 开源的本地语音识别库,真正干活的人。

### 1.2 全局常量与缓存(20–23 行)

```python
WORKSPACE = Path(__file__).parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

_MODEL_CACHE: dict[str, "whisper.Whisper"] = {}
```

- `Path(__file__).parent` —— 当前 .py 文件所在目录;把 `workspace/` 钉死在项目根下,无论从哪里启动 Python 都能找到。
- `_MODEL_CACHE` —— 模块级字典,**进程内**缓存已加载的 Whisper 模型。Whisper 加载一个 base 模型要 ~1 秒,一个进程里多次 `transcribe()` 就只加载一次。
- 下划线开头是 Python 约定:「这是私有的,模块外不要用」。

### 1.3 文件指纹 `_file_fingerprint`(26–36 行)

```python
def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha1()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as f:
        h.update(f.read(1024 * 1024))           # 头 1 MB
        if size > 2 * 1024 * 1024:
            f.seek(-1024 * 1024, os.SEEK_END)   # 跳到结尾前 1 MB
            h.update(f.read(1024 * 1024))       # 尾 1 MB
    return h.hexdigest()[:16]
```

为什么不直接 SHA1 整个文件?**视频可以几个 GB,全部读一遍很慢**。这里取「文件大小 + 首 1MB + 尾 1MB」组合哈希,几乎可以唯一识别一个文件,但 I/O 量是常数。`[:16]` 截前 16 位,够防碰撞,做文件名也短。

`os.SEEK_END` 是文件 seek 的「相对结尾」标记。

### 1.4 抽音频 `_extract_audio`(39–53 行)

```python
def _extract_audio(video_path: Path, audio_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # 无损 16-bit PCM
        "-ar", "16000",         # 16 kHz 采样率
        "-ac", "1",             # 单声道
        "-y",                   # 覆盖已存在的输出
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")
```

为什么转成 16 kHz / 单声道 / PCM?这是 **Whisper 模型训练时用的输入格式**。给它别的格式它内部也会重采样,但费时。预转好就快。

`subprocess.run([...], capture_output=True)` —— 同步等待 ffmpeg 跑完,把 stdout/stderr 收集起来。失败时把 stderr 抛进异常,方便调试。

注意 cmd 用列表传不用字符串拼接:**杜绝 shell 注入风险**(如果文件名里有空格、引号、特殊字符,字符串拼接会出大问题)。

### 1.5 模型加载缓存 `_load_model`(56–59 行)

```python
def _load_model(name: str) -> "whisper.Whisper":
    if name not in _MODEL_CACHE:
        _MODEL_CACHE[name] = whisper.load_model(name)
    return _MODEL_CACHE[name]
```

经典的 memoization。第一次调用真去加载,之后命中字典直接返回。`whisper.load_model("base")` 第一次还会去 OpenAI 服务器下载 ~150MB 权重(下载到 `~/.cache/whisper/`),之后是从磁盘读。

### 1.6 主函数 `transcribe`(62–124 行)

这是整个文件的入口。分四段看:

#### 1.6.1 参数处理与缓存命中(83–93 行)

```python
video_path = Path(video_path).expanduser().resolve()
if not video_path.exists():
    raise FileNotFoundError(f"Video not found: {video_path}")

model_name = model_name or os.getenv("WHISPER_MODEL", "base")
fp = _file_fingerprint(video_path)
cache_path = WORKSPACE / f"{video_path.stem}.{fp}.{model_name}.json"

if cache_path.exists() and not force:
    with cache_path.open("r", encoding="utf-8") as f:
        return json.load(f)
```

- `expanduser()` 把 `~` 展开成 `/Users/xxx`。
- `resolve()` 转成绝对路径,顺便解析符号链接。
- `model_name or os.getenv(..., "base")` —— 三段优先级:**函数参数 > 环境变量 > 默认值 "base"**。
- 缓存文件名是 `<视频名>.<指纹>.<模型>.json`。同一视频不同模型的缓存互不冲突。
- 命中缓存直接读 JSON 返回,跳过下面所有重活。

#### 1.6.2 抽音频 + 跑 Whisper(95–100 行)

```python
with tempfile.TemporaryDirectory() as tmp:
    audio_path = Path(tmp) / "audio.wav"
    _extract_audio(video_path, audio_path)

    model = _load_model(model_name)
    result = model.transcribe(str(audio_path), language=language)
```

`tempfile.TemporaryDirectory()` 创建一个临时目录,用 `with` 块退出时**自动删除**(连里面的 audio.wav 一起)。这样不用手动清理。

`model.transcribe(audio_path, language=None)` —— 把 wav 喂给 Whisper。`language=None` 让它自动检测语言(中文会输出中文,英文输出英文)。返回值长这样:

```python
{
    "text": "完整一段话拼起来...",
    "language": "zh",
    "segments": [
        {"id": 0, "start": 0.0, "end": 4.2, "text": "...", "avg_logprob": ..., "tokens":[...], ...},
        ...
    ]
}
```

我们只用 `language` 和 `segments`,其它字段不关心。

#### 1.6.3 整理 segments(102–112 行)

```python
segments = [
    {
        "id": int(seg["id"]),
        "start": round(float(seg["start"]), 2),
        "end": round(float(seg["end"]), 2),
        "text": seg["text"].strip(),
    }
    for seg in result.get("segments", [])
]

duration = segments[-1]["end"] if segments else 0.0
```

Whisper 返回的 segment 字段太多(包含 token、概率等),我们只留 4 个核心字段,顺便:
- `round(.., 2)` —— 时间戳保留两位小数,够用又干净。
- `.strip()` —— 去掉文本前后空白。
- `int(..)`/`float(..)` —— 显式转类型,避免 numpy 类型混进来导致 JSON 序列化失败。

视频时长用最后一个 segment 的 `end` 近似(差几秒不影响后续剪辑)。

#### 1.6.4 写缓存 + 返回(114–124 行)

```python
output = {
    "video": str(video_path),
    "language": result.get("language", "unknown"),
    "duration": duration,
    "segments": segments,
}

with cache_path.open("w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

return output
```

`ensure_ascii=False` —— 中文等非 ASCII 字符**直接写,不转 \uXXXX**,JSON 文件人看得懂。
`indent=2` —— pretty-print,方便人工 cat 看缓存内容。

### 1.7 `format_time` + `__main__`(127–144 行)

```python
def format_time(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins}:{secs:05.2f}"
```

`05.2f` —— 至少 5 位、保留 2 位小数(包括小数点)。所以 7.3 秒打印成 `07.30`,对齐好看。

`if __name__ == "__main__":` 让你能直接 `python transcribe.py video.mp4` 单独测这个模块。`__name__` 在被 import 时是模块名 `"transcribe"`,直接运行时是 `"__main__"`,经典模式。

---

## 2. editor.py — FFmpeg 切片 + 拼接

文件 ~150 行。一个公开函数 `assemble()`,两个私有 `_cut_one` / `_concat`,加一个工具 `probe_duration`。

### 2.1 文档字符串与策略说明(1–14 行)

文档解释了**为什么不用 `-c copy` 直接切**:
直接 copy 切的位置必须落在视频关键帧(keyframe)上,否则要么开头黑屏要么开头有马赛克。我们这里采用**先逐段重新编码,再 concat**:
- 重新编码 → 切点准到每一帧
- concat 时所有片段编码格式一致,可以用 `-c copy` 无损拼

代价是慢 2-3 倍,但效果稳定。

### 2.2 输出目录 + Clip 类型(25–32 行)

```python
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

class Clip(TypedDict):
    video: str
    start: float
    end: float
```

`TypedDict` —— 给 dict 加上「字段名 + 类型」检查。它**不是 runtime 强制**,只是给类型检查器(mypy / IDE)看的;运行时还是普通 dict。

为什么用 dict 不用 dataclass?因为 `editor.py` 跟 LLM 来回传输的是 JSON,而 JSON 自然映射到 dict。dataclass 还要 `asdict()` 转一次,没必要。

### 2.3 包装 ffmpeg 调用 `_run`(35–42 行)

```python
def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed:\n"
            f"  cmd: {' '.join(shlex.quote(c) for c in cmd)}\n"
            f"  stderr: {result.stderr[-1500:]}"
        )
```

公用错误处理:
- `shlex.quote` —— 把命令重新组成可以复制粘贴回 shell 的形式(参数里有空格也能正确加引号)。报错时方便你直接复刻命令调试。
- `[-1500:]` —— 只保留 stderr 末尾 1500 字符。ffmpeg 的 stderr 极长,大部分是初始化信息,真正错误信息一般在末尾。

### 2.4 切单段 `_cut_one`(45–63 行)

```python
def _cut_one(clip: Clip, out_path: Path) -> None:
    duration = max(0.01, float(clip["end"]) - float(clip["start"]))
    cmd = [
        "ffmpeg",
        "-ss", f"{float(clip['start']):.3f}",   # 起点(秒)
        "-i", str(clip["video"]),                # 输入
        "-t", f"{duration:.3f}",                 # 持续时长
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-y",
        str(out_path),
    ]
    _run(cmd)
```

参数挨个讲:
- `-ss <时间>` 在 `-i` **之前** —— 这叫「快速 seek」,ffmpeg 直接跳到大概位置;放在 `-i` **之后**叫「精确 seek」,会逐帧解码到目标位置(慢)。我们这里既要快又要准,放前面 + 后续重编码足够准。
- `-t <时长>` —— 切多长。比 `-to <终点>` 更直观,不容易算错。
- `-c:v libx264` —— H.264 编码,兼容性最好。
- `-preset veryfast` —— 编码速度档位。`ultrafast`/`superfast`/`veryfast`/`fast`/`medium`/`slow` 越往右画质越好但越慢。
- `-crf 20` —— 画质参数,0=无损,18=视觉无损,20≈中高画质,28=肉眼明显损失。
- `-c:a aac` + `-b:a 192k` —— 音频用 AAC,192 kbps,人声/环境音都够。
- `-pix_fmt yuv420p` —— 像素格式。手机/网页播放器都认这个,**不写有时会输出某些播放器播不出来的格式**。
- `-movflags +faststart` —— 把 mp4 的 metadata 移到文件头,**网页里播放可以边下边播**,不用整个下完才播。
- `-y` —— 覆盖已存在的输出文件,不问。

### 2.5 拼接 `_concat`(66–88 行)

```python
def _concat(parts: list[Path], out_path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        list_path = Path(f.name)
        for p in parts:
            f.write(f"file '{p.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n")
    try:
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            "-movflags", "+faststart",
            "-y",
            str(out_path),
        ]
        _run(cmd)
    finally:
        list_path.unlink(missing_ok=True)
```

**concat demuxer 的玩法:** 写一个文本文件,每行一条 `file '/path/to/part.mp4'`,然后 `ffmpeg -f concat -i list.txt -c copy out.mp4`。要求所有 part 的编码、分辨率、帧率一致(我们 `_cut_one` 里强制统一过了)。

最让人困惑的一行:

```python
p.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))
```

`chr(39) = '` (单引号), `chr(92) = \` (反斜杠)。它在做的事:**把单引号转义成 `'\''`** —— ffmpeg concat 列表的引用规则。如果文件路径里有单引号(虽然罕见),这一步保护它。

`delete=False` —— 临时文件不会在 with 块结束时自动删,需要手动 `unlink`。这里这么写是因为 ffmpeg 进程要在 with 块退出之后才能读到文件。`finally` 确保最后一定删除。

### 2.6 入口 `assemble`(91–121 行)

```python
def assemble(clips: Iterable[Clip], output_filename: str | None = None) -> Path:
    clips = [c for c in clips if float(c["end"]) > float(c["start"])]
    if not clips:
        raise ValueError("No valid clips to assemble.")

    if not output_filename:
        output_filename = f"vlog_{uuid.uuid4().hex[:8]}.mp4"
    if not output_filename.lower().endswith(".mp4"):
        output_filename += ".mp4"
    out_path = OUTPUT_DIR / output_filename

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []
        for i, clip in enumerate(clips):
            part_path = tmp_dir / f"part_{i:03d}.mp4"
            _cut_one(clip, part_path)
            parts.append(part_path)

        if len(parts) == 1:
            import shutil
            shutil.copy2(parts[0], out_path)
        else:
            _concat(parts, out_path)

    return out_path
```

流程:
1. 过滤掉无效片段(end<=start)。
2. 没起名字就用 `uuid` 自动生成 8 位随机文件名,避免覆盖之前的成片。
3. 起一个临时目录,逐段 `_cut_one` 输出 `part_000.mp4`、`part_001.mp4`、...
4. 只有 1 段就直接 `shutil.copy2`(连同元数据一起复制),省一次 concat。
5. 多段就走 `_concat`。
6. 临时目录自动清理,只留下最终 `out_path`。

`f"part_{i:03d}"` —— 数字补 0 到 3 位:`part_000`, `part_001`, ..., `part_010`,这样按文件名排序就是按顺序。

### 2.7 `probe_duration` + `__main__`(124–149 行)

```python
def probe_duration(video_path: str | Path) -> float:
    cmd = ["ffprobe", "-v", "error",
           "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1",
           str(video_path)]
```

`ffprobe` 是 ffmpeg 自带的视频信息工具。这条命令只抽出 `duration` 字段,纯数字输出。`agent.py` 当前没用到,但留作未来需要(比如在没转录之前先报个时长)。

`__main__` 这块允许你手写一份 plan.json 单独跑 editor:
```bash
echo '[{"video":"a.mp4","start":0,"end":5}]' > plan.json
python editor.py plan.json
```
方便不开 LLM 测 ffmpeg 那边。

---

## 3. agent.py — 对话核心

最长(~610 行),也是最有意思的一个文件。结构:

```
imports / 全局
SYSTEM_PROMPT
_function() 帮手
TOOLS = [6 个工具的 JSON schema]
@dataclass VideoEntry
@dataclass Project    ← 真正的业务逻辑
class VlogAgent       ← 跟 LLM 来回的循环
```

### 3.1 imports + 配置(21–41 行)

```python
import json, os
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
```

- `from openai import OpenAI` —— 用的是 OpenAI 官方 SDK,但下面会传不同的 `base_url`,所以可以指向智谱、Gemini、千问等任何「OpenAI 兼容」的服务。
- `load_dotenv()` —— 读 `.env` 文件里的变量,塞进 `os.environ`。这里是模块加载时就执行,所以下面 `os.getenv(...)` 已经能拿到值。
- `import editor / transcribe` —— 同目录下的两个模块,直接相对导入。

### 3.2 SYSTEM_PROMPT(43–70 行)

这就是给 LLM 看的「岗位说明书 + SOP」。每次发请求都会作为第一条 message 发给 LLM。

要点:
- 列出 6 个工具的名字 + 简短描述(冗余但有用,避免 LLM 漏看 `tools=` 参数)。
- 给单视频和多视频两套工作流。
- 「Important rules」是约束,比如:必须基于真实字幕、留 0.3s buffer、用用户语言回复、不清楚就先问。

**写 system prompt 的核心心得:** 多用具体场景例子,少用抽象命令。LLM 看到「'everything about coffee'」比看「theme-based search」更知道该用哪个工具。

### 3.3 工具 schema 帮手 `_function`(73–82 行)

```python
def _function(name, description, parameters):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
```

OpenAI function-calling 协议要求外面包一层 `{"type":"function","function":{...}}`。每次手写太啰嗦,封装成帮手。

### 3.4 TOOLS 列表(85–220 行)

6 个工具的 JSON schema。每一个都是用 `_function(name, description, parameters)` 定义。

`parameters` 是 **JSON Schema**(标准),关键字段:
- `type: "object"` —— 表示参数是个对象(类似 dict)。
- `properties` —— 每个字段的名字、类型、描述。
- `required` —— 必填字段名列表。

LLM 只会看到 `name + description + parameters`,看不到 Python 函数本身。所以**描述写得越准、参数 description 越详细,LLM 用得越对**。

举例 `search_segments`(154–187 行):

```python
"search_segments",
"Substring search (case-insensitive) across the transcripts of one, "
"several, or all loaded videos. Use this when there are many videos "
"and you want to find every mention of a topic without reading "
"every transcript end-to-end. Returns matching segments with "
"video_id and timestamps.",
{ "type": "object",
  "properties": {
      "query": {"type": "string", ...},
      "video_ids": {"type": "array", "items": {"type": "string"}, ...},
      "max_results": {"type": "integer", ...},
      "context_segments": {"type": "integer", ...},
  },
  "required": ["query"],
},
```

注意 description 不光说「这个工具做什么」,还说了「**什么时候用**」(many videos + theme-based)、「**返回什么结构**」(segments with video_id and timestamps)。这两条对 LLM 决策最关键。

### 3.5 `VideoEntry` dataclass(223–248 行)

```python
@dataclass
class VideoEntry:
    video_id: str
    path: Path
    transcript: dict

    @property
    def mtime(self) -> float:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return 0.0

    def snippet(self, head_chars: int = 140, tail_chars: int = 80) -> str:
        segs = self.transcript.get("segments", [])
        if not segs:
            return "(silent / no transcript)"
        all_text = " ".join(s["text"] for s in segs).strip()
        if len(all_text) <= head_chars + tail_chars + 5:
            return all_text
        return f"{all_text[:head_chars].strip()} … {all_text[-tail_chars:].strip()}"
```

`@dataclass` —— 自动生成 `__init__` / `__repr__` / `__eq__`,省得手写。
`@property` —— 让 `entry.mtime` 像属性一样访问,实际背后调函数。文件可能被删,所以包了个 try。

`snippet()` 的策略:把所有 segment 的 text 拼一起,如果总长 <= 头+尾+5(头 140 + 尾 80 + 一个 `…` 大概 5 字符),就直接返回完整文本;否则取头 140 字 + 「 … 」 + 尾 80 字。这个 snippet 会进系统提示给 LLM 看,**让它对每个视频有个 220 字左右的「印象」**,不用读全文也能选片。

### 3.6 `Project` dataclass(251–472 行)

整个文件最重要的类。是「项目状态 + 所有真实业务方法」。LLM 调用的每个工具背后都是 Project 里的一个方法。

#### 3.6.1 状态 + 工具方法(251–256 行)

```python
@dataclass
class Project:
    videos: dict[str, VideoEntry] = field(default_factory=dict)

    def next_id(self) -> str:
        return f"v{len(self.videos) + 1}"
```

`field(default_factory=dict)` —— dataclass 默认值如果是可变对象(dict / list)必须用 `default_factory`,否则所有实例会共享同一个 dict(经典 Python 坑)。

#### 3.6.2 `add()` —— 加一个视频(258–276 行)

```python
def add(self, path: str) -> dict:
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        return {"error": f"File not found: {path_obj}"}

    for entry in self.videos.values():
        if entry.path == path_obj:
            return self._summary(entry)         # 已经加过,直接返回原条目

    try:
        tr = transcribe.transcribe(path_obj)
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}

    vid = self.next_id()
    entry = VideoEntry(video_id=vid, path=path_obj, transcript=tr)
    self.videos[vid] = entry
    return self._summary(entry)
```

注意它**遇错不抛异常,返回 `{"error": ...}`**。这是因为 LLM 工具调用结果会被序列化成 JSON 喂回去,有 error 字段 LLM 就知道失败了、可以选择重试或报告给用户。

「已经加过的同路径直接返回原条目」—— 防止 LLM 重复添加导致 v1/v2/v3 都指向同一个文件。

#### 3.6.3 `_summary()` —— 摘要生成器(278–291 行)

```python
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
```

把一个 VideoEntry 转成 LLM-friendly 的 dict。`*, include_snippet` 是 keyword-only 参数(必须 `_summary(e, include_snippet=True)` 调用,不能位置传)。

#### 3.6.4 `list_videos()`(293–296 行)

```python
def list_videos(self) -> list[dict]:
    entries = sorted(self.videos.values(), key=lambda e: e.path.name)
    return [self._summary(e, include_snippet=True) for e in entries]
```

按文件名排序,把每个的摘要(带 snippet)返回。**排序很重要**:LLM 多次调用看到的顺序固定,推理稳定;而且文件名通常带日期/时间,顺序就是时间顺序。

#### 3.6.5 `get_transcript()`(298–309 行)

```python
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
```

注意 error 信息里**附上已有 video_id 列表**(`Known: ['v1','v2']`)。这样 LLM 看到错误立刻知道「噢,我应该用 v1 不是 v3」,不需要再调一次 list_videos。**给 LLM 的错误信息要主动包含足够上下文。**

#### 3.6.6 `search_segments()`(311–370 行)

最复杂的一个,但思路简单:遍历目标视频的所有 segment,文本里包含 query 就算 hit。

关键细节:
- `q = query.strip().lower()` + `q in seg["text"].lower()` —— 大小写不敏感的子串匹配。
- `targets` 选择:有传 video_ids 就只搜那些;没传就搜全部。
- `context_segments` 参数:命中后顺便把前后几段抓出来当上下文。LLM 经常用这个判断「这一句的上下文是讲什么主题」。
- `if len(hits) >= max_results: break` —— 双层 break(内外两个 for 都要 break),避免找到上限后还继续遍历。
- 返回 `truncated: bool` —— 让 LLM 知道结果是否被截断了,需要的话可以收紧 query 再搜。

#### 3.6.7 `add_videos_from_dir()`(378–439 行)

```python
def add_videos_from_dir(self, path, pattern=None, recursive=False,
                        max_videos=100, on_progress=None):
    dir_path = Path(path).expanduser().resolve()
    if not dir_path.exists(): return {"error": ...}
    if not dir_path.is_dir(): return {"error": ...}

    patterns = [pattern] if pattern else list(self.VIDEO_GLOBS)

    files = []
    seen = set()
    for p in patterns:
        iterator = dir_path.rglob(p) if recursive else dir_path.glob(p)
        for f in iterator:
            if f.is_file() and f not in seen:
                seen.add(f); files.append(f)
    files.sort(key=lambda f: f.name)
    if len(files) > max_videos:
        return {"error": ...}
    if not files:
        return {"error": ...}

    added, skipped = [], []
    for i, f in enumerate(files, start=1):
        res = self.add(str(f))
        if on_progress:
            on_progress(i, len(files), f.name, res)
        if "error" in res: skipped.append(...)
        else: added.append(res)

    return {"directory": ..., "scanned": ..., "added": ..., "skipped": ...}
```

设计点:
- **去重用 set** —— 用户可能传 `*.mp4` 和 `*.MP4` 两个 pattern,同个文件别加两次。
- `glob` vs `rglob` —— glob 只看顶层,rglob 递归。默认不递归(怕用户指着 `~/Movies` 然后被卷进 5 万个文件)。
- `max_videos` 安全帽 —— 同样防爆炸,默认 100 应该够大部分场景。
- `on_progress` 回调 —— Project 类里**不直接 print**,而是接受一个回调,让外层(CLI、Web UI、未来的 GUI)各自决定怎么显示。这是经典的「关注点分离」。

#### 3.6.8 `create_cut()`(441–472 行)

```python
def create_cut(self, clips: list[dict], output_filename: str | None) -> dict:
    if not clips: return {"error": "No clips provided."}

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
        resolved.append({"video": str(entry.path), "start": start, "end": end})

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
```

LLM 给的是 `[{video_id: "v1", start, end}, ...]`,这里要做两件事:
1. **把 video_id 翻译成真实文件路径**(LLM 不知道路径,只知道项目里给的 ID)。
2. **传给 editor.assemble()**,这是与 ffmpeg 唯一的接口。

`view_link: "computer://..."` —— 一些桌面 / IDE 能识别这个 scheme 直接打开文件,方便用户点开预览。

### 3.7 `VlogAgent` 类(475–608 行)

#### 3.7.1 `__init__`(476–493 行)

```python
def __init__(self, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, max_tool_iters=10):
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set. ...")
    self.client = OpenAI(api_key=api_key, base_url=base_url)
    self.model = model
    self.max_tool_iters = max_tool_iters
    self.project = Project()
    self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
```

- `OpenAI(api_key=, base_url=)` —— 关键就这一行,把 OpenAI SDK 指向智谱。SDK 不在乎服务器是谁,只要协议对就能跑。
- `max_tool_iters=10` —— 防止 LLM 进死循环(自己一直 call 同一个工具)。10 轮够正常任务用。
- `messages` 列表保存整段对话历史,index 0 永远是 system prompt(下面会动态更新)。

#### 3.7.2 `_build_system_prompt`(495–512 行)

```python
def _build_system_prompt(self) -> str:
    videos = self.project.list_videos()
    if not videos:
        return SYSTEM_PROMPT
    lines = ["", "",
             "Videos already loaded in the project ..."]
    for v in videos:
        dur = v.get("duration_sec") or 0
        snippet = (v.get("snippet") or "").strip()
        lines.append(f"- {v['video_id']} | {v['filename']} | "
                     f"{dur:.1f}s | {v['segment_count']} segs | lang={v.get('language')}")
        if snippet:
            lines.append(f"    snippet: {snippet}")
    return SYSTEM_PROMPT + "\n".join(lines)
```

每次对话开始,把当前项目状态(视频列表 + 摘要)拼到 system prompt 末尾。这样 LLM **每个回合都看到最新状态**,不需要先调 list_videos 才知道有什么。

为什么不用单独的 message 传?因为单独 message 会被算到对话上下文里,后续会被 LLM 当成「用户在某个时刻告诉我了这件事」;塞进 system prompt 是「这是当前事实」。语义不同。

#### 3.7.3 `_dispatch`(514–540 行)

```python
def _dispatch(self, name, args):
    if name == "add_video":          return self.project.add(args["path"])
    if name == "add_videos_from_dir":return self.project.add_videos_from_dir(...)
    if name == "list_videos":        return self.project.list_videos()
    if name == "get_transcript":     return self.project.get_transcript(args["video_id"])
    if name == "search_segments":    return self.project.search_segments(...)
    if name == "create_cut":         return self.project.create_cut(args["clips"], args.get("output_filename"))
    return {"error": f"Unknown tool: {name}"}
```

工具名 → Project 方法的映射。可以用字典+lambda 写更紧凑,但 if/elif 形式直白,加新工具不会忘改测试。

#### 3.7.4 `chat()` —— 核心循环(542–608 行)

整个程序的「灵魂」就这块。详细看:

```python
def chat(self, user_input: str, *, on_tool=None) -> str:
    self.messages[0] = {"role": "system", "content": self._build_system_prompt()}
    self.messages.append({"role": "user", "content": user_input})

    for _ in range(self.max_tool_iters):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message

        # 1) 把 LLM 这一轮的输出写回历史
        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name,
                              "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        self.messages.append(assistant_msg)

        # 2) 没有工具调用就是最终答案,返回
        if not msg.tool_calls:
            return (msg.content or "").strip()

        # 3) 执行每个工具调用,把结果作为 "tool" message 写回
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

    return "(Hit the tool-call limit ...)"
```

**乒乓球结构,反复打到 LLM 不再要工具为止:**

| 轮次 | messages 末尾 |
|------|---------------|
| 1 (开始) | `[system, user]` |
| LLM 回应 | append `{role:assistant, tool_calls:[get_transcript(v1)]}` |
| 我们执行 | append `{role:tool, tool_call_id:..., content:"<字幕JSON>"}` |
| 2 (再发) | LLM 看到字幕,可能再 call create_cut |
| LLM 回应 | append `{role:assistant, tool_calls:[create_cut(...)]}` |
| 我们执行 | append `{role:tool, tool_call_id:..., content:"<输出路径JSON>"}` |
| 3 (再发) | LLM 看到剪辑成功,生成最终回复 |
| LLM 回应 | append `{role:assistant, content:"挑了 6 段..."}` |
| 这次没 tool_calls | 退出循环,return `"挑了 6 段..."` |

注意几点:
- **每轮都重建 system prompt**(line 551),保证视频列表最新。
- `tool_call_id` 是必须的,标识哪条 tool 结果回应哪个 call(LLM 可能并行 call 多个工具)。
- 工具结果统一 `json.dumps()` 成字符串再传(API 协议要求 content 是字符串)。
- `Tool crashed` 包成 error dict 喂回去,LLM 能感知崩溃并向用户说明。
- `max_tool_iters` 兜底防死循环。

---

## 4. main.py — CLI 外壳

最简单的一个文件,~177 行,几乎都是 IO 和 UI。

### 4.1 文档 + 常量(1–37 行)

```python
HELP = """\
Commands ...
"""

BANNER = r"""
 ASCII art ...
"""
```

`r"""..."""` —— 原始字符串,反斜杠不转义。ASCII art 里有 `\_` 这种不希望被解释的。

### 4.2 工具调用日志辅助(40–53 行)

```python
def _print_tool(name, args, result):
    arg_preview = ", ".join(f"{k}={_short(v)}" for k, v in args.items())
    if isinstance(result, dict) and "error" in result:
        print(f"  ⚠ {name}({arg_preview}) → ERROR: {result['error']}")
    else:
        print(f"  ✓ {name}({arg_preview}) → {_short(result)}")

def _short(value, limit=80):
    s = repr(value) if not isinstance(value, str) else value
    if len(s) > limit:
        s = s[: limit - 3] + "..."
    return s
```

这就是上面 `agent.chat(user_input, on_tool=_print_tool)` 里那个 `on_tool` 回调。每次 LLM 调用一个工具,打印一行类似:
```
  ✓ get_transcript(video_id=v1) → {'video_id': 'v1', 'segments': [...]}
```

`_short` 截断长输出避免刷屏。

### 4.3 `main()` 函数(56–172 行)

#### 4.3.1 启动检查(57–68 行)

```python
if not os.getenv("LLM_API_KEY"):
    print("ERROR: LLM_API_KEY is not set."); ...
    sys.exit(1)

print(BANNER)
try:
    agent = VlogAgent()
except RuntimeError as e:
    print(f"ERROR: {e}")
    sys.exit(1)
```

提前检查 API key,避免后面神秘的网络错误。

#### 4.3.2 预加载位置参数(70–97 行)

```python
def _prog(i, total, name, res):
    tag = "⚠" if "error" in res else "✓"
    print(f"  [{i}/{total}] {tag} {name}")

for arg in sys.argv[1:]:
    p = Path(arg).expanduser()
    if not p.exists():
        print(f"(skipping non-existent: {p})"); continue
    if p.is_dir():
        print(f"Pre-loading folder {p}...")
        result = agent.project.add_videos_from_dir(str(p), on_progress=_prog)
        ...
    else:
        print(f"Pre-loading {p.name}...")
        result = agent.project.add(str(p))
        ...
```

两类参数自动区分:文件 → `add()`,文件夹 → `add_videos_from_dir()`。这样 `python main.py ~/Movies/trip/` 就直接批量加。

`_prog` 是 `add_videos_from_dir` 的进度回调,这里就 print 一下。

#### 4.3.3 REPL 主循环(101–172 行)

```python
while True:
    try:
        user_input = input("you ▸ ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break

    if not user_input: continue

    if user_input.startswith("/"):
        # 处理 /add /add_dir /list /reset /help /quit ...
        continue

    try:
        reply = agent.chat(user_input, on_tool=_print_tool)
    except Exception as e:
        print(f"  ⚠ agent crashed: {e}")
        continue

    print(f"\nagent ▸ {reply}\n")
```

经典 REPL(Read-Eval-Print Loop):
- `input()` 阻塞等用户输入。
- Ctrl-D (EOF) 或 Ctrl-C (KeyboardInterrupt) 退出。
- 以 `/` 开头是命令,自己处理,**不发给 LLM**。
- 否则发给 `agent.chat()`,把 LLM 最终回复打印出来。

#### 4.3.4 斜杠命令处理(111–164 行)

每个命令几行,写得直白:

- `/quit, /exit` → break 出循环
- `/help` → 打印 HELP
- `/list` → 调 `agent.project.list_videos()` 自己 print(不经过 LLM)
- `/reset` → `agent.messages = agent.messages[:1]` 只留 system prompt
- `/add <path>` → 调 `agent.project.add(rest[0])`
- `/add_dir <path>` → 调 `agent.project.add_videos_from_dir(...)`,带进度回调

**为什么这些命令不让 LLM 处理?** 
1. 省 LLM token / 调用次数。
2. 这些操作都是确定性的、不需要推理。
3. 让用户有「逃生通道」—— 万一 LLM 行为怪异,可以直接命令操作。

`cmd, *rest = user_input.split(maxsplit=1)` —— 解构赋值。`maxsplit=1` 只切一刀,所以路径里有空格也安全(只要整段当一个 token)。

### 4.4 `if __name__ == "__main__"`(175–176 行)

```python
if __name__ == "__main__":
    main()
```

`python main.py ...` 启动时 `__name__ == "__main__"`,调 `main()`。如果有人 `import main`,这里就不执行,只暴露函数。

---

## 5. 数据/控制流总图

把上面所有东西串起来,以「用户说『把 v1 精华部分留下』」为例:

```
1. main.py REPL 收到字符串,发给 agent.chat()
       │
       ▼
2. agent.chat():
   - 重建 system prompt(注入当前 v1 的 snippet)
   - 把 user 消息 append 到 messages
   - 调 OpenAI client → 请求飞向智谱 GLM-4-Flash
       │
       ▼
3. LLM 看完 system + user,决定要先读字幕,
   返回 {assistant, tool_calls:[get_transcript(v1)]}
       │
       ▼
4. agent.chat() 拆 tool_calls,_dispatch("get_transcript", {video_id:"v1"})
   → Project.get_transcript("v1")
   → 返回 v1 的 transcript dict
       │
       ▼
5. agent.chat() 把 dict json.dumps,作为 role:tool 消息 append
   → main 那边的 _print_tool 打印 "  ✓ get_transcript(...)"
       │
       ▼
6. agent.chat() 继续循环,messages 多了刚才的 assistant + tool 消息,
   再次请求 LLM
       │
       ▼
7. LLM 看到字幕,挑出精华片段,返回
   {assistant, tool_calls:[create_cut(clips=[(v1,3.2,12.8),(v1,28.4,42.1),...])]}
       │
       ▼
8. agent.chat() → Project.create_cut(clips, None)
   → Project 把 v1 翻成真实路径
   → editor.assemble([{video:..., start:3.2, end:12.8}, ...])
   → editor 逐段 _cut_one() → ffmpeg 切并重编码 → temp/part_000.mp4 / part_001.mp4 / ...
   → editor _concat() → ffmpeg concat → output/vlog_xxx.mp4
   → 返回 Path 对象
       │
       ▼
9. Project.create_cut() 包装结果 {output_path, view_link, duration_sec, clip_count}
   → agent.chat() 序列化作为 tool 消息 append
       │
       ▼
10. LLM 看到 output_path,生成自然语言回复:
    "好的,挑了 5 段精华,共 23.4 秒,文件在 output/vlog_xxx.mp4"
    没有 tool_calls 了,循环退出
       │
       ▼
11. agent.chat() return 这段文字 → main.py 打印 "agent ▸ 好的,挑了..."
```

11 步,LLM 被叫了 3 次,**没看一帧视频**。

---

## 6. 学习路径建议

如果你想吃透这套代码,建议按这个顺序:

1. **`transcribe.py`** —— 最独立、最容易跑通。先理解「视频怎么变成 segments」。
   实验:`python transcribe.py 你的某个mp4`,看输出。
2. **`editor.py`** —— 也独立可测。
   实验:手写 `plan.json`,`python editor.py plan.json`,观察生成的 mp4。
3. **`agent.py` 里的 `Project` 类** —— 不涉及 LLM,纯业务。
   实验:写个小脚本不调 LLM,直接 `Project().add(...)` + `create_cut(...)`,验证流程。
4. **`agent.py` 里的 `TOOLS` + `_dispatch` + `chat`** —— 真正的 LLM 集成。
   实验:在 `chat()` 里加 `print(self.messages[-1])`,看每轮对话历史是怎么演化的。
5. **`main.py`** —— 最简单,看它是怎么调 agent 就够了。

如果只想读不想跑,**单独看 `chat()` 这一个函数就能理解整个 agent 模式**(那个 for 循环里的 4 件事:发请求 → 写 assistant → 检查是否有 tool_calls → 跑工具并写 tool 消息),其它都是细节。

---

## 7. 几个常被问到的「为什么」

**Q: 为什么 Whisper 跑在本地,LLM 跑在云端?**
A: Whisper 是开源、单次推理在 CPU 几秒到几十秒就完;LLM 自己跑要 GPU + 几十 GB 显存,不现实。Whisper 的输出又是离散的文本,云端 LLM 处理 token 不耗带宽。这是最经济的分工。

**Q: 为什么不用 LangChain / LlamaIndex 这类框架?**
A: 这个项目的 agent 循环就 60 行,加上工具定义统共 220 行,框架带来的间接层比省的代码还多。等到你需要 RAG、向量检索、多 agent 协作,再考虑框架。

**Q: 为什么所有错误都返回 dict,不抛异常?**
A: 因为 LLM 看到的是 JSON。返回 `{"error": "..."}` LLM 能读、能解释、能改方案;抛异常会被外层 catch 成模糊的 "Tool crashed",信息丢失。

**Q: 缓存的指纹用 SHA1 不会撞吗?**
A: 16 位十六进制 = 64 bits,要撞需要约 2³² ≈ 40 亿个不同文件。个人项目里完全够用。撞了也只是缓存读到错的转录,删了重跑就行,不会数据丢失。

**Q: 重新编码不浪费吗?能不能 `-c copy`?**
A: `-c copy` 要求切点必须落在 GOP(关键帧组)边界,否则开头会有几帧损坏。LLM 给的时间是从字幕来的,不可能正好在关键帧上。重编码慢但稳。如果未来要优化,可以先 `-c copy` 切到最近关键帧,再用 `concat filter`(不是 concat demuxer)做帧级精确剪。
