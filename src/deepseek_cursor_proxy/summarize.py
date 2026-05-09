from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .logging import LOG

SUMMARIZE_SYSTEM_PROMPT = """\
You are a context summarization assistant. Your task is to produce a concise \
summary of a conversation that was truncated due to context window limits.

The summary will be injected into the conversation so the AI assistant can \
maintain continuity. Focus on:
1. Key decisions made and their rationale
2. Current state of the task (what's done, what's pending)
3. Important file paths, function/class names, and architectural choices
4. Any constraints or requirements the user specified
5. Errors encountered and how they were resolved

Be concise but preserve critical context. Write in the same language the \
user was using. Output ONLY the summary, no preamble."""

SUMMARIZE_USER_TEMPLATE = """\
The following conversation ({num_messages} messages, ~{num_tokens} tokens) \
is being truncated to fit the context window. Please summarize the key \
context that the assistant needs to continue the conversation effectively:

---
{conversation}
---

Produce a concise summary (max 500 words) capturing the essential context."""

MAX_CONVERSATION_CHARS_FOR_SUMMARY = 60_000
SUMMARY_MAX_TOKENS = 1024
SUMMARIZE_TIMEOUT = 30.0


def _format_message_for_summary(msg: dict[str, Any]) -> str:
    """Format a single message for the summarization prompt."""
    role = msg.get("role", "unknown")
    content = msg.get("content")

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
        text = "\n".join(parts)
    else:
        text = ""

    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        tc_parts = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if len(args) > 200:
                    args = args[:200] + "..."
                tc_parts.append(f"  → {name}({args})")
        if tc_parts:
            text = (text + "\n" if text else "") + "\n".join(tc_parts)

    if not text:
        text = "(empty)"

    if len(text) > 2000:
        text = text[:1000] + "\n...[truncated]...\n" + text[-800:]

    return f"[{role}]: {text}"


def _build_conversation_text(
    messages: list[dict[str, Any]],
    max_chars: int = MAX_CONVERSATION_CHARS_FOR_SUMMARY,
) -> str:
    """Build a text representation of messages for the summarization prompt.
    If the full text exceeds max_chars, keep the beginning and end."""
    formatted = [_format_message_for_summary(m) for m in messages]
    full_text = "\n\n".join(formatted)

    if len(full_text) <= max_chars:
        return full_text

    # Keep beginning and end to capture both early context and recent state
    half = max_chars // 2
    return (
        full_text[:half]
        + "\n\n... [middle of conversation omitted for brevity] ...\n\n"
        + full_text[-half:]
    )


def generate_summary_via_api(
    messages_to_summarize: list[dict[str, Any]],
    estimated_tokens: int,
    *,
    base_url: str,
    authorization: str,
    model: str,
    timeout: float = SUMMARIZE_TIMEOUT,
) -> str | None:
    """Call the upstream API to generate a natural-language summary of dropped messages.
    Returns the summary text, or None if the call fails (falls back to lightweight).
    """
    conversation_text = _build_conversation_text(messages_to_summarize)

    user_content = SUMMARIZE_USER_TEMPLATE.format(
        num_messages=len(messages_to_summarize),
        num_tokens=estimated_tokens,
        conversation=conversation_text,
    )

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": SUMMARY_MAX_TOKENS,
        "temperature": 0.3,
        "stream": False,
    }

    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    url = f"{base_url.rstrip('/')}/chat/completions"

    headers = {
        "Authorization": authorization,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Accept-Encoding": "identity",
    }

    request = Request(url, data=body, method="POST", headers=headers)

    started = time.monotonic()
    try:
        LOG.info(
            "├ summarize: generating context summary via API (%d messages, ~%d tokens)...",
            len(messages_to_summarize),
            estimated_tokens,
        )
        response = urlopen(request, timeout=timeout)
        response_body = response.read()
        elapsed = time.monotonic() - started

        result = json.loads(response_body.decode("utf-8"))
        choices = result.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            summary = message.get("content", "")
            if summary:
                usage = result.get("usage", {})
                LOG.info(
                    "├ summarize: done in %.1fs (prompt=%s output=%s)",
                    elapsed,
                    usage.get("prompt_tokens", "?"),
                    usage.get("completion_tokens", "?"),
                )
                return summary.strip()

        LOG.warning("├ summarize: API returned no content, falling back to lightweight")
        return None

    except HTTPError as exc:
        elapsed = time.monotonic() - started
        LOG.warning(
            "├ summarize: API call failed status=%s elapsed=%.1fs, "
            "falling back to lightweight summary",
            exc.code,
            elapsed,
        )
        return None
    except URLError as exc:
        elapsed = time.monotonic() - started
        LOG.warning(
            "├ summarize: API call failed reason=%s elapsed=%.1fs, "
            "falling back to lightweight summary",
            exc.reason,
            elapsed,
        )
        return None
    except Exception as exc:
        elapsed = time.monotonic() - started
        LOG.warning(
            "├ summarize: unexpected error %s elapsed=%.1fs, "
            "falling back to lightweight summary",
            exc,
            elapsed,
        )
        return None
