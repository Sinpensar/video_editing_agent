# VlogAgent 代码逐段讲解

这份文档不会改源码,只解释每个 .py 文件每段代码在做什么、为什么这么写。
对照阅读建议:左边开 `transcribe.py` / `editor.py` / `agent.py` / `main.py`,右边开本文。每节标题里的行号都对应当前文件的实际行号。

---

## 0. 整体架构

```
你 ─ chat ─►  main.py  ──►  agent.py (VlogAgent)  ──►  tracing.py (Tracer)
                              │                          │
                              │                          └► traces/<session>/turn_NNN.json
                              │
                              ├──►  transcribe.py     ──►  ffmpeg + Whisper
                              │     (video → 带时间戳字幕)
                              │
                              ├──►  validators.py     ──►  pre-flight 检查 clips
                              │     (越界/重复/重叠/目标偏差)
                              │
                              ├──►  editor.py         ──►  ffmpeg
                              │     (字幕里挑出的 (start,end) → mp4)
                              │
                              └──►  OpenAI 兼容 API (默认智谱 GLM-4-Flash)
                                    (读字幕 + 决定剪哪几段)

                              traces/ ──► viewer.py ──► viewer.html (浏览器查看)
```

七个文件的职责:
- **transcribe.py** —— 把视频转成带时间戳的字幕(是「让 LLM 看懂视频」的关键)
- **editor.py** —— 拿到 `(start, end)` 列表后,真正去切视频、拼成片
- **agent.py** —— 对话核心:管理项目状态、定义工具、跑 LLM 工具调用循环
- **validators.py** —— pre-flight 校验层,reject-and-retry 的核心
- **tracing.py** —— observability 层:每轮对话结构化 dump 到 JSON,记录所有 LLM 和工具调用
- **viewer.py** —— 把 traces/ 里的 JSON 渲染成自包含 HTML 查看器
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

#### 3.7.1 `__init__`(476–495 行)

```python
def __init__(self, model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL,
             max_tool_iters=10, tracer: Optional[Tracer] = None):
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY not set. ...")
    self.client = OpenAI(api_key=api_key, base_url=base_url)
    self.model = model
    self.max_tool_iters = max_tool_iters
    self.project = Project()
    self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    self.tracer = tracer    # optional, can be None
```

- `OpenAI(api_key=, base_url=)` —— 关键就这一行,把 OpenAI SDK 指向智谱。SDK 不在乎服务器是谁,只要协议对就能跑。
- `max_tool_iters=10` —— 防止 LLM 进死循环(自己一直 call 同一个工具)。10 轮够正常任务用。
- `messages` 列表保存整段对话历史,index 0 永远是 system prompt(下面会动态更新)。
- `tracer` —— 可选的 observability 钩子,见第 4 节。`None` 时所有 trace 调用都是 no-op,业务逻辑完全不受影响。

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

#### 3.7.4 `chat()` —— 核心循环(542–615 行)

整个程序的「灵魂」就这块。详细看:

```python
def chat(self, user_input, *, on_tool=None):
    if self.tracer:
        self.tracer.begin_turn(user_input)

    self.messages[0] = {"role": "system", "content": self._build_system_prompt()}
    self.messages.append({"role": "user", "content": user_input})

    final_reply = None
    try:
        for _ in range(self.max_tool_iters):
            t_llm = time.monotonic()
            resp = self.client.chat.completions.create(
                model=self.model, messages=self.messages, tools=TOOLS,
            )
            latency_ms = int((time.monotonic() - t_llm) * 1000)
            if self.tracer:
                self.tracer.record_llm_call(self.messages, resp, latency_ms)

            msg = resp.choices[0].message

            # 1) 把 LLM 这一轮的输出写回历史
            assistant_msg = {"role": "assistant", "content": msg.content or ""}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [...]    # 略
            self.messages.append(assistant_msg)

            # 2) 没有工具调用就是最终答案,返回
            if not msg.tool_calls:
                final_reply = (msg.content or "").strip()
                return final_reply

            # 3) 执行每个工具调用,把结果作为 "tool" message 写回
            for tc in msg.tool_calls:
                name = tc.function.name
                try:    args = json.loads(tc.function.arguments or "{}")
                except: args = {}
                t_tool = time.monotonic()
                try:    result = self._dispatch(name, args)
                except Exception as e: result = {"error": f"Tool crashed: {e}"}
                duration_ms = int((time.monotonic() - t_tool) * 1000)
                is_error = isinstance(result, dict) and "error" in result

                if self.tracer:
                    self.tracer.record_tool_call(name, args, result, duration_ms, is_error)
                if on_tool:
                    on_tool(name, args, result)

                self.messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        final_reply = "(Hit the tool-call limit ...)"
        return final_reply
    finally:
        if self.tracer:
            self.tracer.end_turn(final_reply)
```

跟之前相比加了三处:**`try/finally` 包裹**(确保 trace 一定能 end_turn)、**LLM 调用前后 `time.monotonic()` 测延迟**、**两个 `if self.tracer` 钩子**。所有 trace 调用都是 None-safe,关掉 tracer 不影响主逻辑。

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

## 4. validators.py — pre-flight 检查层

文件 ~150 行。一个公开函数 `validate_clips(clips, project, goal)` 返回一个**字符串列表**。空列表 = 通过,非空 = 每条字符串都是一个具体的问题描述。

### 4.1 设计哲学:为什么不抛异常,返回字符串列表

传统软件里参数错通常 `raise ValueError`。这里不抛是因为:**LLM 看不到异常,只能看到 tool result**。返回结构化的「问题描述列表」给 LLM,它能逐条读、逐条修。**字符串还要写得「可读且可操作」**:
- ❌ 不好:`"Invalid clip[2]"`(LLM 不知道哪里错)
- ✅ 好:`"clip[2] end (180.50s) exceeds duration of v1 (95.30s)"`(明确告诉它怎么改)

### 4.2 两层校验

```python
# 层 1:Sanity checks(永远跑)— 抓低级错误
- 未知 video_id
- start < 0
- end <= start
- end > 视频时长 + 0.5s 容差
- 时长 < 0.3s(切了等于没切)
- 时长 > 90s(LLM 算错的概率大)
- 完全重复的 clip
- 同一视频的 clip 之间 overlap > 0.5s

# 层 2:Goal checks(只在 Project.goal 非空时跑)
- 总时长偏离目标超过 ±25%
- clip 数 < min_clips 或 > max_clips
- must_include_keywords 必须在保留内容的字幕里出现
- must_exclude_keywords 必须不出现在保留内容的字幕里
```

### 4.3 关键函数 `_clipped_text`(36–48 行)

```python
def _clipped_text(clips, project) -> str:
    parts = []
    for c in clips:
        entry = project.videos.get(c.get("video_id"))
        if not entry: continue
        s, e = float(c.get("start", 0)), float(c.get("end", 0))
        for seg in entry.transcript.get("segments", []):
            if seg["end"] > s and seg["start"] < e:   # 重叠就算
                parts.append(seg["text"])
    return " ".join(parts).lower()
```

这是关键词检查的基础:**把所有「保留下来的字幕文字」拼成一大段**,再看必含/必排关键词在不在里面。重叠判定用 `seg.end > clip.start AND seg.start < clip.end`,经典区间相交。

### 4.4 Per-clip 检查(73–122 行)

逐个 clip 看 7 件事:type-cast 不掉链子(`try float()`),video_id 存在,start ≥ 0,end > start,end 不超过视频时长,时长在 [min, max] 范围内。每发现一处问题 append 一条人话描述。**注意 `continue`**:遇到致命错误(end<=start)就跳到下一个 clip,不再继续后面对该 clip 的检查 —— 否则会输出「end 也超时长」「时长也太短」一堆派生问题让 LLM 困惑。

### 4.5 重复 + 重叠(124–157 行)

```python
# 重复:基于 (video_id, round(start,2), round(end,2)) 三元组
# 重叠:把同一 video 的 clip 按 start 排序,逐对算 prev_end - cur_start
seen_keys = set()
for i, c in enumerate(clips):
    key = (c["video_id"], round(c["start"], 2), round(c["end"], 2))
    if key in seen_keys:
        issues.append(f"clip[{i}] duplicates an earlier clip")
    seen_keys.add(key)

by_video = group_by_video(clips)
for vid, items in by_video.items():
    items.sort(key=start)
    for j in range(1, len(items)):
        overlap = items[j-1].end - items[j].start
        if overlap > overlap_tolerance:
            issues.append(...)
```

为什么 `round(.., 2)`?Whisper 的时间戳本来就是浮点数,LLM 复制时也会保留小数;但 `0.5000001` 跟 `0.5` 在内存里不相等。round 到 2 位避免误判。

### 4.6 Goal checks(159–202 行)

只在 `goal` 字段被显式设置时才跑。每一项都按「目标存在 → 检查 → 不达标 append issue」的固定模式。

`target_duration_sec` 用相对偏差(`abs(total - target) / target`)而不是绝对差 —— **2 分钟目标差 30 秒和 5 秒目标差 30 秒严重程度完全不同**。

`must_include_keywords` / `must_exclude_keywords` 的检查是 `O(n_clips × n_segments × n_keywords)`,但常量都很小(几十个 segment、几个 keyword),效率不是问题。

### 4.7 设计权衡:为什么 v1 没做严重程度分级

理想方案是把 issue 分成 `critical / warning`:critical 必须修才能跑,warning 可以 `force=true` 跳过。但:
- 实现复杂度上去了一档
- LLM 看到 warning 容易直接 `force=true` 偷懒,反而失去校验意义

V1 做法:全部当作 critical。LLM 想跳过得**显式传 `force=true`**,而 system prompt 明确告诉它「只在用户接受了警告之后才传 force」。这样 LLM 默认会去修正,而不是绕过。

### 4.8 怎么接进 `Project.create_cut`

```python
def create_cut(self, clips, output_filename, *, force=False):
    issues = validators.validate_clips(clips, self, goal=self.goal)
    if issues and not force:
        return {
            "validation_failed": True,
            "issues": issues,
            "goal": dict(self.goal) if self.goal else None,
            "hint": "Read each issue, adjust the offending clips, then call create_cut again. ..."
        }
    # ... 否则正常执行 ffmpeg
```

返回值里 `validation_failed: True` 是给 LLM 的明确信号。`hint` 是「怎么自我修复」的提示。`goal` 一并附上,提醒 LLM 当前目标是什么。

### 4.9 配套的 `set_goal` 工具

LLM 通过这个工具登记用户目标:

```python
set_goal(
    target_duration_sec=120,
    must_include_keywords=["coffee", "barista"],
    must_exclude_keywords=["um", "uh"],
)
```

存到 `Project.goal` dict。下次 `create_cut`,validators 自动读它。SYSTEM_PROMPT 里写了「会话开始如果用户说了清晰目标,先 set_goal」让 LLM 主动这么做。

`_build_system_prompt` 也会把当前 goal 注入到每一轮的 system prompt 里:

```
Active goal (validators will enforce this on every create_cut):
- target_duration_sec: 120
- must_include_keywords: ['coffee']
```

这样 LLM 每轮都看得到目标,挑 clip 时心里有数。

---

## 5. tracing.py — observability 层

文件 ~170 行。一个类 `Tracer`,记录 agent 在每轮对话里的全部活动到 `traces/<session_id>/`。

### 5.1 文件用途与目录结构(1–22 行)

```
traces/
└── 20260507_103245/              ← session_id(时间戳)
    ├── session.json              ← 元数据:模型、起始时间、cwd
    ├── turn_001.json             ← 用户第 1 句话引发的全过程
    ├── turn_002.json
    └── ...
```

**为什么按 session 分目录、按 turn 分文件?**
- 一个 session = 一次启动 main.py 到退出。
- 一个 turn = 一次用户输入到 agent 给出最终回复(中间可能跑了多轮 LLM + 多次工具)。
- 文件颗粒度按 turn 分,既不会因为太细(每个 LLM 调用一个文件)导致文件爆炸,也不会因为太粗(整个 session 一个文件)导致写盘频繁的并发问题。
- JSON 文件可读、可搜索、可 diff、可被 viewer.py 离线渲染。

### 5.2 帮手 `_truncate_for_log`(28–42 行)

```python
def _truncate_for_log(value, max_chars=4000):
    s = json.dumps(value, ensure_ascii=False, default=str)
    if len(s) <= max_chars:
        return value
    head = s[: max_chars // 2]
    tail = s[-max_chars // 4 :]
    return {
        "_truncated": True,
        "_original_len": len(s),
        "preview_head": head,
        "preview_tail": tail,
    }
```

如果一个 tool 的返回值超过 4 KB(典型例子:`get_transcript` 返回完整字幕),只留**头一半 + 尾四分之一**,标记为 truncated。
trace 文件会保持小巧,`viewer.html` 加载也快;真要看完整内容可以查 cache 或重跑。

### 5.3 `Tracer.__init__`(48–73 行)

```python
def __init__(self, session_id=None, traces_dir=None,
             model=None, base_url=None):
    self.session_id = session_id or _new_session_id()
    self.dir = Path(traces_dir or TRACE_DIR_DEFAULT) / self.session_id
    self.dir.mkdir(parents=True, exist_ok=True)
    self.model = model
    self.base_url = base_url
    self.turn_count = 0
    self._cur = None
    self._meta_path = self.dir / "session.json"
    self._write_meta(turns=0)
```

`session_id` 默认是 `YYYYMMDD_HHMMSS` 格式,做目录名 + UI 标签都直观。
`_cur` 保存当前正在记录的 turn 字典,turn 结束才 flush 到磁盘 —— 中途 crash 也能保住已经写好的 turns。

### 5.4 元数据写盘 `_write_meta`(75–93 行)

```python
def _write_meta(self, turns: int) -> None:
    meta = {
        "session_id": ...,
        "started_at": datetime.now().isoformat(...),
        "model": ..., "base_url": ..., "turns": ..., "cwd": ...,
    }
    if self._meta_path.exists():
        old = json.loads(self._meta_path.read_text(...))
        meta["started_at"] = old.get("started_at", meta["started_at"])
    self._meta_path.write_text(json.dumps(meta, ...))
```

每次 `end_turn` 后会重写 `session.json` 更新 `turns` 计数,但 `started_at` 保持第一次写入的值不被覆盖。这样 viewer 显示「会话从 X 开始,共 N 轮」一目了然。

### 5.5 `begin_turn`(97–117 行)

```python
def begin_turn(self, user_input: str) -> None:
    self.turn_count += 1
    self._cur = {
        "turn": self.turn_count,
        "started_at": ...,
        "user_input": user_input,
        "steps": [],          # 每次 LLM API 调用一个 step
        "tool_calls": [],     # 每次 tool dispatch 一个
        "final_reply": None,
        "totals": {
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
            "llm_latency_ms": 0, "tool_duration_ms": 0,
            "llm_calls": 0, "tool_calls": 0,
        },
    }
```

新一轮开始,准备一个空骨架。`steps` 跟 `tool_calls` 是平级两个数组,通过 `step` 字段相互关联(后面 viewer 用这个把 LLM 决策和它产生的工具结果对应起来)。

### 5.6 `record_llm_call`(119–169 行)

```python
def record_llm_call(self, request_messages, response, latency_ms):
    usage = getattr(response, "usage", None)
    in_tok = getattr(usage, "prompt_tokens", 0)
    out_tok = getattr(usage, "completion_tokens", 0)
    total_tok = getattr(usage, "total_tokens", in_tok + out_tok)

    msg = response.choices[0].message
    tool_calls = []
    if msg.tool_calls:
        for tc in msg.tool_calls:
            try:
                parsed_args = json.loads(tc.function.arguments or "{}")
            except Exception:
                parsed_args = {"_raw": tc.function.arguments}
            tool_calls.append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": parsed_args,
            })

    step = {
        "step": len(self._cur["steps"]) + 1,
        "n_messages_in_request": len(request_messages),
        "input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": total_tok,
        "latency_ms": latency_ms,
        "stop_reason": getattr(response.choices[0], "finish_reason", None),
        "response": {
            "content": msg.content,
            "tool_calls": tool_calls or None,
        },
    }
    self._cur["steps"].append(step)
    # 累加到 totals ...
```

要点:
- **从真实 response 拿 token 数**(`usage.prompt_tokens` 等),不靠估算。每个 OpenAI 兼容服务都会返回 usage。
- `n_messages_in_request` 而不是 messages 全文 —— **保持 trace 文件小**。完整 messages 的快照会爆炸成几十 MB,而 messages 内容的演化其实可以从 user_input + 各步 response + tool 结果重建。
- `stop_reason`(`tool_calls` / `stop` / `length`)很关键 —— 看 LLM 为什么停下来:还要调工具?给最终答了?还是 token 上限被打断?
- 工具的 `arguments` 立刻 `json.loads` 成 dict,避免 viewer 再去 parse 一次嵌套的 JSON 字符串。

### 5.7 `record_tool_call`(171–192 行)

```python
def record_tool_call(self, name, args, result, duration_ms, is_error):
    step_num = len(self._cur["steps"])  # 当前 step 号
    self._cur["tool_calls"].append({
        "step": step_num,                # 关联到哪个 LLM step 触发的
        "name": name,
        "arguments": args,
        "result": _truncate_for_log(result),
        "is_error": bool(is_error),
        "duration_ms": duration_ms,
    })
    self._cur["totals"]["tool_duration_ms"] += duration_ms
    self._cur["totals"]["tool_calls"] += 1
```

`step` 字段是关键 —— viewer 渲染时,把 `tool_calls` 里 `step == N` 的所有项,展示在第 N 个 LLM step 下方。这样用户看到的是「第 N 步 LLM 决定调 X 和 Y,X 的结果是这样,Y 的结果是这样」的层次。

### 5.8 `end_turn`(194–204 行)

```python
def end_turn(self, final_reply):
    self._cur["final_reply"] = final_reply
    self._cur["ended_at"] = ...
    path = self.dir / f"turn_{self._cur['turn']:03d}.json"
    path.write_text(json.dumps(self._cur, ensure_ascii=False, indent=2))
    self._cur = None
    self._write_meta(turns=self.turn_count)
```

`turn_{N:03d}.json` —— 数字补 3 位,排序自然。
`indent=2` —— pretty-print JSON,方便人工 cat 看。

### 5.9 容错设计

留意 Tracer 里大量 `if not self._cur: return` —— 即使 caller 顺序调错(没 `begin_turn` 就 `record_*`),也只会 silently skip,不会让真业务因为 trace 失败而崩。**Observability 一定要「壮硕」,不能让监控本身变成单点故障。**

---

## 6. viewer.py — 把 trace JSON 变成可看的 HTML

文件 ~290 行,大部分是 HTML/CSS/JS 模板。Python 部分只做两件事:扫目录、把数据塞进模板。

### 6.1 数据收集 `collect_sessions`(28–55 行)

```python
def collect_sessions(traces_dir):
    sessions = []
    for session_dir in sorted(traces_dir.iterdir()):
        if not session_dir.is_dir(): continue
        meta = json.loads((session_dir / "session.json").read_text())
        turns = []
        for turn_path in sorted(session_dir.glob("turn_*.json")):
            turns.append(json.loads(turn_path.read_text()))
        sessions.append({"id": session_dir.name, "meta": meta, "turns": turns})
    return sessions
```

直白:扫目录 → 读所有 JSON → 拼成数组。`sorted()` 按 session_id(时间戳)和 turn 文件名排序,viewer 里就是时间正序。

### 6.2 HTML 模板(58–225 行)

注意第 64 行附近:

```python
HTML_TEMPLATE = r"""<!doctype html>
...
<script>
const SESSIONS = __DATA__;
...
</script>
</html>
"""
```

`__DATA__` 是 placeholder,后面用 `replace("__DATA__", json.dumps(sessions))` 把整个数据 inline 进 HTML。这样最终的 `viewer.html` **是个完全独立文件**,双击就开,不用 web server,可以邮件发给同事看。

CSS 是手写深色主题(没引外部资源,纯 vendored),JS 是纯 vanilla DOM 操作(不引 React/Vue 这种,避免 build step)。

### 6.3 渲染逻辑(JS 部分)

JS 三个核心函数:

```javascript
renderSidebar()     // 把 SESSIONS 数组渲染成左边的 session 列表
selectSession(i)    // 切换到某个 session,把它的 turns 显示在右边
renderTurn(turn)    // 渲染单轮:user_input + stats_bar + steps + tools + final_reply
renderStep(step, toolsForStep)  // 单个 LLM step 的折叠卡片
renderToolCall(tc)  // 单个 tool 调用的折叠卡片
```

**关键的关联渲染:** `renderTurn` 里:

```javascript
for (const step of turn.steps) {
    const toolsForStep = (turn.tool_calls || []).filter(tc => tc.step === step.step);
    stepNodes.push(renderStep(step, toolsForStep));
}
```

把同一 step 触发的所有 tool_call 摆到那个 step 节点下面,形成「LLM 决定 → 触发的工具结果」的层次。

### 6.4 命令行入口 `main`(259–286 行)

```python
ap.add_argument("--traces", default=str(DEFAULT_TRACES_DIR))
ap.add_argument("--out", default=None)
ap.add_argument("--open", action="store_true")
```

简单的 argparse:
- 不传参就读 `./traces/`
- 不指定 `--out` 就写到 `<traces>/viewer.html`
- 加 `--open` 用 `webbrowser.open` 自动打开

```bash
python viewer.py --open    # 最常用
```

### 6.5 设计权衡:为什么不做实时面板

也可以起一个 Flask/FastAPI server,实时读 traces 目录,websocket 推送给前端做实时 dashboard。但:
- 开发复杂度 ×3
- 依赖一个长跑进程
- 需要处理「正在写入的文件」的并发读

**对个人 / 小团队,「跑完看回放」比「实时看」实用得多**(实时也只是看个表面热闹,真要分析还是事后慢慢翻)。所以 viewer 故意做成静态生成,简单可靠。

要做实时,**未来可以考虑** `python -m http.server` 配合 `setInterval(fetch, 2000)`,改 30 行就能实时刷,但目前 YAGNI。

---

## 7. main.py — CLI 外壳

最简单的一个文件,~177 行,几乎都是 IO 和 UI。

### 7.1 文档 + 常量(1–37 行)

```python
HELP = """\
Commands ...
"""

BANNER = r"""
 ASCII art ...
"""
```

`r"""..."""` —— 原始字符串,反斜杠不转义。ASCII art 里有 `\_` 这种不希望被解释的。

### 7.2 工具调用日志辅助(40–53 行)

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

### 7.3 `main()` 函数(56–172 行)

#### 7.3.1 启动检查 + Tracer(57–72 行)

```python
if not os.getenv("LLM_API_KEY"):
    print("ERROR: LLM_API_KEY is not set."); ...
    sys.exit(1)

print(BANNER)
tracer = Tracer(model=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL)
print(f"  trace session: {tracer.session_id} → traces/{tracer.session_id}/\n")

try:
    agent = VlogAgent(tracer=tracer)
except RuntimeError as e:
    print(f"ERROR: {e}")
    sys.exit(1)
```

启动时:
1. 先确认有 API key,没有就提前退出。
2. **建一个 Tracer**,会立刻在 `traces/<时间戳>/` 下生成 session.json。
3. 把 tracer 传给 `VlogAgent` —— 这样后面所有对话都会自动记录。

如果你想关掉 trace(比如想跑一个不留痕迹的会话),只要在这行不传 tracer 就行:`VlogAgent()` —— 主流程零改动。

#### 7.3.2 预加载位置参数(70–97 行)

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

#### 7.3.3 REPL 主循环(101–172 行)

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

#### 7.3.4 斜杠命令处理(111–164 行)

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

### 7.4 `if __name__ == "__main__"`(175–176 行)

```python
if __name__ == "__main__":
    main()
```

`python main.py ...` 启动时 `__name__ == "__main__"`,调 `main()`。如果有人 `import main`,这里就不执行,只暴露函数。

---

## 8. 数据/控制流总图

把上面所有东西串起来,以「用户说『把 v1 精华部分留下』」为例。**右边一栏标了 Tracer 在每一步记录什么**,这样你能直接对照 viewer 里看到的内容:

```
                                                            Tracer 记录的内容
                                                            ────────────────────
1. main.py REPL 收到字符串,发给 agent.chat()
       │
       ▼
2. agent.chat():                                            tracer.begin_turn(user_input)
   - tracer.begin_turn(user_input)                          → 准备一个空 turn 骨架
   - 重建 system prompt(注入当前 v1 的 snippet)
   - 把 user 消息 append 到 messages
   - 调 OpenAI client → 请求飞向智谱 GLM-4-Flash
       │
       ▼
3. LLM 看完 system + user,决定要先读字幕,                    tracer.record_llm_call(step1)
   返回 {assistant, tool_calls:[get_transcript(v1)]}        → 记 input/output tokens、
                                                              latency_ms、stop_reason、
                                                              issued tool_calls
       │
       ▼
4. agent.chat() 拆 tool_calls,_dispatch("get_transcript")
   → Project.get_transcript("v1")
   → 返回 v1 的 transcript dict
       │
       ▼
5. agent.chat() 把 dict json.dumps,作为 role:tool 消息       tracer.record_tool_call(...)
   append → main 那边的 _print_tool 打印                    → 记 name/args/result/duration
                                                              步骤号关联到 step1
       │
       ▼
6. agent.chat() 继续循环,再次请求 LLM
       │
       ▼
7. LLM 看到字幕,挑出精华片段,返回                            tracer.record_llm_call(step2)
   {assistant, tool_calls:[create_cut(clips=[...])]}
       │
       ▼
8. agent.chat() → Project.create_cut(clips, None)
   → editor.assemble() → ffmpeg 切并重编码 → concat
   → output/vlog_xxx.mp4
       │
       ▼
9. Project.create_cut() 包装结果 → tool 消息 append          tracer.record_tool_call(...)
                                                              关联到 step2
       │
       ▼
10. LLM 看到 output_path,生成自然语言回复,                   tracer.record_llm_call(step3)
    没有 tool_calls 了,循环退出                              → stop_reason="stop"
       │
       ▼
11. agent.chat() return 这段文字 → main.py 打印              tracer.end_turn(reply)
                                                            → 把 turn 骨架 dump 成
                                                              traces/<sid>/turn_001.json
```

**11 步,LLM 被叫了 3 次,没看一帧视频。** Tracer 在 finally 块里调用 `end_turn`,所以即使中途 ffmpeg 抛异常,这一轮的 trace 也会被持久化(包含失败现场)。

下次你运行 `python viewer.py --open`,这段对话会显示成一张可折叠的卡片:3 个 LLM step、2 个 tool 调用、所有 token 数、所有耗时一目了然。

---

## 9. 学习路径建议

如果你想吃透这套代码,建议按这个顺序:

1. **`transcribe.py`** —— 最独立、最容易跑通。先理解「视频怎么变成 segments」。
   实验:`python transcribe.py 你的某个mp4`,看输出。
2. **`editor.py`** —— 也独立可测。
   实验:手写 `plan.json`,`python editor.py plan.json`,观察生成的 mp4。
3. **`agent.py` 里的 `Project` 类** —— 不涉及 LLM,纯业务。
   实验:写个小脚本不调 LLM,直接 `Project().add(...)` + `create_cut(...)`,验证流程。
4. **`agent.py` 里的 `TOOLS` + `_dispatch` + `chat`** —— 真正的 LLM 集成。
   实验:在 `chat()` 里加 `print(self.messages[-1])`,看每轮对话历史是怎么演化的。
5. **`validators.py`** —— 看「怎么用纯函数把 LLM 的输出验一道」。
   实验:把 validators 里的 issue 临时改成中文,看 LLM 在 viewer 里有没有学着用中文描述自己的修复思路。
6. **`tracing.py` + `viewer.py`** —— observability。这块最不影响业务,但最影响你**理解**业务。
   实验:跑一两轮对话,然后 `python viewer.py --open`,观察 trace 里的 step 序列 ——
   特别是 reject-and-retry 时 validator 把 issues 喂回去之后,LLM 的下一个 step 会怎么改。
7. **`main.py`** —— 最简单,看它是怎么调 agent + tracer 就够了。

如果只想读不想跑,**单独看 `chat()` 这一个函数就能理解整个 agent 模式**(那个 for 循环里的 4 件事:发请求 → 写 assistant → 检查是否有 tool_calls → 跑工具并写 tool 消息),其它都是细节。

**Tracer 是怎么进来的也值得专门看一眼:** Tracer 完全是「旁路记录」—— `chat()` 里加的几行 `if self.tracer: tracer.record_*` 是钩子模式的教科书例子。业务逻辑零变化,功能可以独立开关。后续要加 logger / metrics / 远程上报,都是同一种模式扩展的。

---

## 10. 几个常被问到的「为什么」

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

**Q: Tracer 为什么不直接用 Python 的 logging?**
A: `logging` 是行式日志,擅长记「发生了什么」;Tracer 要记的是**结构化的对话状态**(嵌套的 LLM step / tool call / tokens / latency),需要 JSON。两者目的不同。理论上可以用 `logging.handlers.JsonFormatter` 写 JSONL,但 viewer 想做时间线视图就还要回头反 parse,折腾。直接每轮一个 JSON 文件简单到极致,也方便 grep / diff / 加新字段。

**Q: Tracer 加进去为什么不会拖慢 agent?**
A: 每个 record 调用就是几微秒的字典操作,turn 结束才一次性写盘。**写盘量级 < LLM 网络延迟的 1/1000**,完全察觉不到。如果未来 trace 写盘真的成为瓶颈(比如长跑 agent 一秒几十轮),可以改成 background thread 异步写,不需要动 caller。

**Q: viewer 为什么不做实时刷新?**
A: 实时刷新要么轮询(拉)要么 websocket(推),都需要起 server。对个人项目,「跑完看回放」实用度 > 「实时围观」。况且 trace JSON 已经是落盘的,真要实时,30 行 JS + `setInterval(fetch)` 就能加上,YAGNI。

**Q: validators 为什么不抛异常,要返回字符串列表?**
A: 因为 LLM 看不到异常 —— 异常会被 caller catch 成模糊的 `"Tool crashed: ..."`,LLM 没法逐条修。返回 `{validation_failed: True, issues: [...]}` 让 LLM 看到具体每一处问题,它就能逐条改。Validators 写的是给 LLM 读的「错题本」,不是给 Python 看的异常。

**Q: set_goal 为什么是 LLM 调,不是 main.py 让用户填?**
A: 一来用户说话经常隐含目标(「剪 2 分钟」「不要废话」),让 LLM 抽出来比让用户填表更自然;二来 LLM 跨会话延续目标更平滑(它在 system prompt 里看到 active goal,自然会沿用)。让 LLM 主动调 set_goal 也是「workflow 工具化」的练习 —— 状态变更全走工具,意味着每一步都进 trace,可观测、可回放。

**Q: validator 卡在死循环怎么办?**
A: 上一道防线是 `max_tool_iters=10`,LLM 改不出来就到上限退出。理论上可以加 `consecutive_validation_failures` 计数器,3 次后自动 force=True 跑掉,但 v1 没做。如果你跑下来发现 LLM 真的反复犯同一个错,改它的 system prompt(给个具体的修复模式范例)比加策略层更有效。
