# 改动记录 (Changelog)

本文档记录从上游仓库克隆后需要手动对齐的代码改动，使另一台电脑上的工程与本机一致。

> 本机当前 HEAD: a35583a — refactor(logging): simplified prints and add spinner (#37)

---

## Commit 1: a35583a — refactor(logging): simplified prints and add spinner (#37)

### 目的

抽取统一的日志模块，简化终端输出，在等待上游响应时显示旋转动画。

### 涉及文件


| 文件                                     | 操作  | 说明                               |
| -------------------------------------- | --- | -------------------------------- |
| src/deepseek_cursor_proxy/logging.py   | 新增  | 统一的日志模块和控制台 spinner              |
| src/deepseek_cursor_proxy/server.py    | 修改  | 使用新的日志模块，添加 spinner，重构 main 启动日志 |
| src/deepseek_cursor_proxy/transform.py | 修改  | 改为 from .logging import LOG      |
| src/deepseek_cursor_proxy/tunnel.py    | 修改  | 改为 from .logging import LOG      |


### 具体改动

#### a) 新增 src/deepseek_cursor_proxy/logging.py（完整新文件）

创建新文件，内容如下：

```python
from __future__ import annotations

import logging as stdlib_logging
import sys
import threading
from typing import Any


LOG = stdlib_logging.getLogger("deepseek_cursor_proxy")

DEFAULT_INFO_LOG_FORMAT = "%(message)s"
DEFAULT_WARNING_LOG_FORMAT = "%(levelname)s %(message)s"
VERBOSE_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


class ConsoleLogFormatter(stdlib_logging.Formatter):
    def __init__(self, *, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self._verbose_formatter = stdlib_logging.Formatter(VERBOSE_LOG_FORMAT)
        self._info_formatter = stdlib_logging.Formatter(DEFAULT_INFO_LOG_FORMAT)
        self._warning_formatter = stdlib_logging.Formatter(DEFAULT_WARNING_LOG_FORMAT)

    def format(self, record: stdlib_logging.LogRecord) -> str:
        if self.verbose:
            return self._verbose_formatter.format(record)
        if record.levelno <= stdlib_logging.INFO:
            return self._info_formatter.format(record)
        return self._warning_formatter.format(record)


def configure_logging(*, verbose: bool) -> None:
    handler = stdlib_logging.StreamHandler()
    handler.setFormatter(ConsoleLogFormatter(verbose=verbose))
    stdlib_logging.basicConfig(
        level=stdlib_logging.INFO,
        handlers=[handler],
        force=True,
    )


class TerminalSpinner:
    hide_cursor = "\x1b[?25l"
    show_cursor = "\x1b[?25h"
    frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

    def __init__(self, *, enabled: bool, text: str, stream=None, interval: float = 0.12) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled and bool(getattr(self.stream, "isatty", lambda: False)())
        self.text = text
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._visible = False

    def start(self) -> "TerminalSpinner":
        if not self.enabled or self._thread is not None:
            return self
        self.stream.write(self.hide_cursor)
        self.stream.flush()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        if self._visible:
            self.stream.write("\r" + (" " * self._clear_width()) + "\r")
            self.stream.flush()
            self._visible = False
        self.stream.write(self.show_cursor)
        self.stream.flush()

    def _run(self) -> None:
        index = 0
        while not self._stop.is_set():
            self.stream.write("\r" + self.text.format(frame=self.frames[index]))
            self.stream.flush()
            self._visible = True
            index = (index + 1) % len(self.frames)
            self._stop.wait(self.interval)

    def _clear_width(self) -> int:
        return max(len(self.text.format(frame=frame)) for frame in self.frames)
```

#### b) 修改 src/deepseek_cursor_proxy/server.py

**导入变更：** 删除原来的 `import logging` 和 `LOG = logging.getLogger(...)`，改为：

```python
from .logging import (
    LOG,
    TerminalSpinner,
    configure_logging,
)
```

**_upstream_request 方法中添加 spinner：** 在 `log_send_summary(prepared)` 之后添加：

```python
if self.config.verbose:
    log_send_summary(prepared)
spinner = TerminalSpinner(
    enabled=bool(prepared.payload.get("stream")) and not self.config.verbose,
    text="\u2514 {frame}",
).start()
```

在所有 return/异常路径前添加 `spinner.stop()`，并用 `try/finally` 确保 spinner 一定停止。

**main() 函数重构启动日志：** 删除原来的多条分散的 LOG.info，改为：

```python
configure_logging(verbose=config.verbose)

local_base_url = f"http://{config.host}:{config.port}/v1"
api_base_url = (
    f"{public_url.rstrip('/')}/v1" if public_url is not None else local_base_url
)

LOG.info(
    "default_model: %s (%s, %s)",
    config.upstream_model,
    "thinking" if config.thinking == "enabled" else "no thinking",
    config.reasoning_effort,
)
if config.verbose:
    LOG.info("display_reasoning: %s", ...)
    LOG.info("missing_reasoning_strategy: %s", config.missing_reasoning_strategy)
    LOG.info("reasoning_cache: %s", config.reasoning_content_path)
if public_url is None and not config.ngrok:
    LOG.info("public_tunnel: off")
LOG.info("local_base_url: %s", local_base_url)
LOG.info("api_base_url: %s", api_base_url)
```

**log_cursor_request 函数：** 日志格式改为同时输出 effort：

```
"┌ request model=%s effort=%s messages=%s"
```

**log_context_summary 函数：** 输出格式改为以 status 为中心：

```
"├ context status=ok reasoning_context=%s"          # status==ok 时
"├ context status=%s missing=%s recovered=%s dropped=%s"  # 其他
```

#### c) 修改 src/deepseek_cursor_proxy/transform.py

删除原有的 `import logging` 和 `LOG = logging.getLogger(...)`，改为：

```python
from .logging import LOG
```

#### d) 修改 src/deepseek_cursor_proxy/tunnel.py

删除原有的 `import logging` 和 `LOG = logging.getLogger(...)`，改为：

```python
from .logging import LOG
```

---

## Commit 2: 3ed8da6 — feat(config): default reasoning effort to max (#36)

### 目的

将 reasoning_effort 默认值从 high 升级为 max。

### 具体改动

**config.py 第 23 行：**

```python
# 改前
DEFAULT_REASONING_EFFORT = "high"
# 改后
DEFAULT_REASONING_EFFORT = "max"
```

**server.py — --reasoning-effort help 文本：**

```
"DeepSeek reasoning effort, default from config or max"
```

**transform.py — prepare_upstream_request 函数（约 795 行）：**

```python
# 改前 — 允许请求中传入 effort
prepared["reasoning_effort"] = normalize_reasoning_effort(
    prepared.get("reasoning_effort") or config.reasoning_effort
)
# 改后 — 始终用配置文件的值
prepared["reasoning_effort"] = normalize_reasoning_effort(
    config.reasoning_effort
)
```

---

## Commit 3: 4eebf78 — fix: prevent recovery cascade and improve Stop-scenario reasoning lookup (#25)

### 目的

修复三个问题：

1. 用户按 Stop 后流式响应被截断，tool_call.id 未到达，缓存无法匹配
2. 非正常退出 stream 时不存缓存，推理内容丢失
3. 缺少 tool_name 级别的 fallback key

### 具体改动

#### a) reasoning_store.py — 新增 tool_call_names() 函数

在 `tool_call_ids()` 之后添加：

```python
def tool_call_names(message: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool_call in message.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if isinstance(function, dict) and function.get("name"):
            names.append(str(function["name"]))
    return names
```

#### b) reasoning_store.py — scoped_reasoning_keys 结尾新增 tool_name fallback

```python
keys.extend(
    f"scope:{scope}:tool_name:{tool_name}" for tool_name in tool_call_names(message)
)
```

#### c) reasoning_store.py — portable_reasoning_keys 结尾新增

```python
keys.extend(
    f"namespace:{cache_namespace}:turn:{turn_signature}:tool_name:{tool_name}"
    for tool_name in tool_call_names(message)
)
```

#### d) server.py — _proxy_streaming_response 方法

将读取循环和存储逻辑用 try/finally 包裹，确保非正常退出也存缓存：

```python
try:
    while True:
        # 读取和转发逻辑...
finally:
    if not finalized:
        stored = sum(
            accumulator.store_reasoning(
                self.reasoning_store, ctx_scope, cache_namespace, prior_messages,
            )
            for ctx_scope, prior_messages in response_contexts
        )
```

#### e) transform.py — reasoning_lookup_keys 函数

在 scope key 之后添加 tool_name fallback：

```python
keys.extend(
    {
        "kind": "tool_name",
        "function_name": tool_name,
        "key": f"scope:{scope}:tool_name:{tool_name}",
        "portable": False,
        "hit": False,
    }
    for tool_name in tool_call_names(message)
)
```

在 portable key 之后添加 portable_tool_name：

```python
keys.extend(
    {
        "kind": "portable_tool_name",
        "function_name": tool_name,
        "key": f"namespace:{cache_namespace}:turn:{turn_signature}:tool_name:{tool_name}",
        "turn_context_signature": turn_signature,
        "portable": True,
        "hit": False,
    }
    for tool_name in tool_call_names(message)
)
```

---

## Commit 4: 7bdf177 — fix(server): honor missing-reasoning reject mode (#34)

### 目的

修复 missing_reasoning_strategy=reject 时不拒绝请求的 bug。同时改进 trace 记录。

### 具体改动

#### a) server.py — 在 log_context_summary 之后

```python
# 改前
if prepared.missing_reasoning_messages:
    LOG.warning(...)
# 改后 — 只在 reject 模式下拒绝
if (
    prepared.missing_reasoning_messages
    and self.config.missing_reasoning_strategy == "reject"
):
    LOG.warning("strict missing-reasoning mode rejected request...")
```

#### b) server.py — 在 404/401 等路径中添加 trace 记录

每个提前 return 前添加 `self._record_request_body_for_trace(trace)`

#### c) server.py — 新增 _record_request_body_for_trace 方法

```python
def _record_request_body_for_trace(self, trace: TraceRequest | None) -> None:
    if trace is None:
        return
    try:
        length = int(self.headers.get("Content-Length") or 0)
    except ValueError:
        trace.record_cursor_body_omitted(reason="invalid_content_length")
        return
    if length < 0:
        trace.record_cursor_body_omitted(reason="invalid_content_length", body_bytes=length)
        return
    if length > self.config.max_request_body_bytes:
        trace.record_cursor_body_omitted(reason="body_too_large", body_bytes=length)
        self.close_connection = True
        return
    try:
        raw_body = self.rfile.read(length)
    except OSError as exc:
        trace.record_cursor_body_omitted(reason=f"read_failed:{exc}", body_bytes=length)
        return
    trace.record_cursor_body_bytes(raw_body)
```

#### d) trace.py — TraceRequest 类新增两个方法

```python
def record_cursor_body_bytes(self, body: bytes) -> None:
    self.data["request"]["body_bytes"] = len(body)
    text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        self.data["request"]["body"] = {"text": text}
        return
    self.data["request"]["body"] = payload
    if isinstance(payload, dict):
        self.data["request"]["summary"] = payload_summary(payload)

def record_cursor_body_omitted(self, *, reason: str, body_bytes: int | None = None) -> None:
    omitted: dict[str, Any] = {"reason": reason}
    if body_bytes is not None:
        omitted["body_bytes"] = body_bytes
    self.data["request"]["body_omitted"] = omitted
```

---

## Commit 5: be03107 — refactor(proxy): audit thinking-mode protocol and refactor test suite (#33)

### 目的

移除 never-used 的 pass-through thinking 模式，审计协议实现，重构测试套件。

### 具体改动

#### a) config.py — normalize_thinking

移除 pass-through / pass_through 别名支持，只保留 enabled 和 disabled。

#### b) server.py — CLI --thinking 选项

choices 从 ["enabled", "disabled", "pass-through"] 改为 ["enabled", "disabled"]。

#### c) server.py — rewrite_response_body 调用传递新参数

```python
rewrite_response_body(
    ...,
    display_reasoning=self.config.display_reasoning,
    collapsible_reasoning=self.config.collapsible_reasoning,
)
```

#### d) streaming.py — 新增 fold_reasoning_into_content 函数

```python
def fold_reasoning_into_content(response_payload: dict[str, Any], collapsible: bool) -> None:
    block_start = COLLAPSIBLE_THINKING_BLOCK_START if collapsible else THINKING_BLOCK_START
    block_end = COLLAPSIBLE_THINKING_BLOCK_END if collapsible else THINKING_BLOCK_END
    choices = response_payload.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning:
            continue
        content = message.get("content")
        message["content"] = (
            block_start + reasoning + block_end
            + (content if isinstance(content, str) else "")
        )
```

#### e) transform.py — 多项改动

1. SUPPORTED_REQUEST_FIELDS 新增 user, seed, n, logit_bias
2. prepare_upstream_request 中移除 pass-through 判断，直接设 thinking
3. 添加丢弃不支持字段时的 warning
4. 非 DeepSeek 模型名重写时打印 warning
5. 新增 strip_recovery_notice_for_upstream 函数
6. 移除 LEGACY_RECOVERY_NOTICE_TEXT
7. rewrite_response_body 新增 display_reasoning 和 collapsible_reasoning 参数

#### f) .gitignore — 新增 .claude/

---

## 对齐检查清单

在新电脑上克隆仓库后，用以下命令确认改动已对齐：

```bash
# 检查当前 HEAD
git log --oneline -1
# 应输出: a35583a refactor(logging): simplified prints and add spinner (#37)

# 确认 logging.py 存在
ls src/deepseek_cursor_proxy/logging.py

# 确认 reasoning_effort 默认值
rg "DEFAULT_REASONING_EFFORT" src/deepseek_cursor_proxy/config.py
# 应输出: DEFAULT_REASONING_EFFORT = "max"

# 确认 thinking choices 不含 pass-through
rg "pass-through" src/deepseek_cursor_proxy/ -l
# 应无输出
```

