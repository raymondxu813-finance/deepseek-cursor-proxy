from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

from .config import ProxyConfig
from .logging import LOG
from .reasoning_store import (
    ReasoningStore,
    conversation_scope,
    message_signature,
    stable_conversation_id,
    tool_call_ids,
    tool_call_names,
    tool_call_signature,
    turn_context_signature,
)
from .streaming import fold_reasoning_into_content
from .summarize import generate_summary_via_api


SUPPORTED_REQUEST_FIELDS = {
    "model",
    "messages",
    "stream",
    "stream_options",
    "max_tokens",
    "response_format",
    "stop",
    "tools",
    "tool_choice",
    "thinking",
    "reasoning_effort",
    "temperature",
    "top_p",
    "presence_penalty",
    "frequency_penalty",
    "logprobs",
    "top_logprobs",
    # Standard OpenAI Chat Completions fields that DeepSeek either honors or
    # safely ignores. Cursor and most OpenAI SDKs send these unconditionally,
    # so forwarding keeps clients happy and avoids log spam.
    "user",
    "seed",
    "n",
    "logit_bias",
}

MESSAGE_FIELDS = {
    "role",
    "content",
    "name",
    "tool_call_id",
    "tool_calls",
    "reasoning_content",
    "prefix",
}

ROLE_MESSAGE_FIELDS = {
    "system": {"role", "content", "name"},
    "user": {"role", "content", "name"},
    "assistant": {
        "role",
        "content",
        "name",
        "tool_calls",
        "reasoning_content",
        "prefix",
    },
    "tool": {"role", "content", "tool_call_id"},
}

EFFORT_ALIASES = {
    "low": "high",
    "medium": "high",
    "high": "high",
    "max": "max",
    "xhigh": "max",
}

CURSOR_THINKING_BLOCK_RE = re.compile(
    r"""
    (?:
        <(?:think|thinking)\b[^>]*>[\s\S]*?(?:</(?:think|thinking)>|\Z)
        |
        <details\b[^>]*>\s*
        <summary\b[^>]*>\s*Thinking\s*</summary>
        [\s\S]*?(?:</details>|\Z)
    )\s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

TOKENS_PER_ENGLISH_CHAR = 0.3
TOKENS_PER_CJK_CHAR = 0.6

_FILE_PATH_RE = re.compile(
    r"""(?:^|[\s"'`(])(/[\w./-]+\.[\w]+|[\w][\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|c|cpp|h|rb|swift|kt|vue|svelte|css|scss|html|json|yaml|yml|toml|sql|sh|md))""",
    re.MULTILINE,
)
_FUNC_CLASS_RE = re.compile(
    r"""(?:def|function|class|interface|struct|enum|type|impl|fn|func|val|var|const|let)\s+([\w]+)""",
)
MAX_SUMMARY_TOKENS = 800


def _is_cjk(char: str) -> bool:
    """Check if a character is CJK (Chinese/Japanese/Korean)."""
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)       # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)    # CJK Extension A
        or (0x20000 <= cp <= 0x2A6DF)  # CJK Extension B
        or (0xF900 <= cp <= 0xFAFF)    # CJK Compatibility Ideographs
        or (0x2F800 <= cp <= 0x2FA1F)  # CJK Compatibility Supplement
        or (0x3000 <= cp <= 0x303F)    # CJK Symbols and Punctuation
        or (0xFF00 <= cp <= 0xFFEF)    # Fullwidth Forms
        or (0x3040 <= cp <= 0x309F)    # Hiragana
        or (0x30A0 <= cp <= 0x30FF)    # Katakana
        or (0xAC00 <= cp <= 0xD7AF)    # Hangul Syllables
    )


def estimate_tokens_for_text(text: str) -> int:
    """Estimate token count using DeepSeek's official ratios:
    - English/ASCII: 1 char ≈ 0.3 tokens
    - CJK (Chinese/Japanese/Korean): 1 char ≈ 0.6 tokens
    """
    if not text:
        return 0
    cjk_chars = sum(1 for c in text if _is_cjk(c))
    other_chars = len(text) - cjk_chars
    return int(cjk_chars * TOKENS_PER_CJK_CHAR + other_chars * TOKENS_PER_ENGLISH_CHAR)


def estimate_message_tokens(message: dict[str, Any]) -> int:
    """Estimate token count for a single message using language-aware ratios."""
    total_tokens = 0
    content = message.get("content")
    if isinstance(content, str):
        total_tokens += estimate_tokens_for_text(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    total_tokens += estimate_tokens_for_text(text)
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str):
        total_tokens += estimate_tokens_for_text(reasoning)
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict):
                fn = tc.get("function")
                if isinstance(fn, dict):
                    total_tokens += estimate_tokens_for_text(fn.get("name") or "")
                    total_tokens += estimate_tokens_for_text(fn.get("arguments") or "")
    # ~4 tokens overhead per message for role/formatting
    return total_tokens + 4


def estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
    """Estimate total tokens for a list of messages."""
    return sum(estimate_message_tokens(m) for m in messages)


def compress_long_message(message: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    """Compress a single message if it exceeds max_tokens.

    Only compresses tool and assistant content. Keeps head (40%) + tail (40%)
    and replaces the middle with a truncation notice. Returns the original
    message unchanged if it doesn't exceed the limit or is a system/user message.
    """
    if max_tokens <= 0:
        return message

    role = message.get("role")
    if role in ("system", "user"):
        return message

    tokens = estimate_message_tokens(message)
    if tokens <= max_tokens:
        return message

    compressed = dict(message)
    content = compressed.get("content")
    if isinstance(content, str) and content:
        content_tokens = estimate_tokens_for_text(content)
        if content_tokens > max_tokens:
            keep_ratio = max_tokens / content_tokens * 0.8
            keep_chars = int(len(content) * keep_ratio)
            head_size = int(keep_chars * 0.5)
            tail_size = int(keep_chars * 0.5)
            omitted_tokens = content_tokens - max_tokens
            compressed["content"] = (
                content[:head_size]
                + f"\n\n...[truncated ~{omitted_tokens} tokens]...\n\n"
                + content[-tail_size:]
            )

    reasoning = compressed.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning:
        reasoning_tokens = estimate_tokens_for_text(reasoning)
        max_reasoning_tokens = int(max_tokens * 0.5)
        if reasoning_tokens > max_reasoning_tokens:
            keep_ratio = max_reasoning_tokens / reasoning_tokens * 0.8
            keep_chars = int(len(reasoning) * keep_ratio)
            compressed["reasoning_content"] = reasoning[:keep_chars] + "..."

    return compressed


def compress_messages(
    messages: list[dict[str, Any]], max_tokens: int
) -> list[dict[str, Any]]:
    """Compress all messages that exceed max_tokens individually."""
    if max_tokens <= 0:
        return messages
    return [compress_long_message(m, max_tokens) for m in messages]


def _extract_text(message: dict[str, Any]) -> str:
    """Extract text content from a message for summarization."""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _extract_tool_names(messages: list[dict[str, Any]]) -> list[str]:
    """Extract tool/function names that were called in dropped messages."""
    names: list[str] = []
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                if isinstance(tc, dict):
                    fn = tc.get("function")
                    if isinstance(fn, dict):
                        name = fn.get("name")
                        if name and name not in names:
                            names.append(name)
    return names


def summarize_dropped_messages(dropped_messages: list[dict[str, Any]]) -> str:
    """Generate a concise context summary from dropped messages without extra API calls.
    Extracts file paths, function/class names, and key topics mentioned.
    """
    all_text = "\n".join(_extract_text(m) for m in dropped_messages)

    file_paths: list[str] = []
    for match in _FILE_PATH_RE.finditer(all_text):
        path = match.group(1)
        if path not in file_paths:
            file_paths.append(path)

    symbols: list[str] = []
    for match in _FUNC_CLASS_RE.finditer(all_text):
        sym = match.group(1)
        if sym not in symbols and len(sym) > 2:
            symbols.append(sym)

    tool_names = _extract_tool_names(dropped_messages)

    user_topics: list[str] = []
    for msg in dropped_messages:
        if msg.get("role") == "user":
            text = _extract_text(msg)
            first_line = text.strip().split("\n")[0][:120]
            if first_line:
                user_topics.append(first_line)

    parts: list[str] = []
    parts.append(
        f"[Context summary: {len(dropped_messages)} earlier messages were "
        f"truncated to fit the context window. Key information from those messages:]"
    )

    if user_topics:
        parts.append(f"- User discussed: {'; '.join(user_topics[:8])}")

    if file_paths:
        parts.append(f"- Files mentioned: {', '.join(file_paths[:20])}")

    if symbols:
        parts.append(f"- Symbols referenced: {', '.join(symbols[:20])}")

    if tool_names:
        parts.append(f"- Tools used: {', '.join(tool_names[:10])}")

    summary = "\n".join(parts)
    if estimate_tokens_for_text(summary) > MAX_SUMMARY_TOKENS:
        # Iteratively trim until within budget
        while estimate_tokens_for_text(summary) > MAX_SUMMARY_TOKENS and len(summary) > 100:
            summary = summary[: int(len(summary) * 0.8)]
        summary += "\n..."
    return summary


def _split_system_and_conversation(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split messages into leading system prefix and conversation body."""
    system_messages: list[dict[str, Any]] = []
    conversation_messages: list[dict[str, Any]] = []
    in_system_prefix = True
    for msg in messages:
        if in_system_prefix and msg.get("role") == "system":
            system_messages.append(msg)
        else:
            in_system_prefix = False
            conversation_messages.append(msg)
    return system_messages, conversation_messages


def _first_msg_hash(conversation_messages: list[dict[str, Any]]) -> str:
    """Hash of the first conversation message for verifying truncation records."""
    if not conversation_messages:
        return ""
    first = conversation_messages[0]
    content = first.get("content", "")
    if isinstance(content, list):
        content = str(content)[:300]
    elif isinstance(content, str):
        content = content[:300]
    payload = f"{first.get('role', '')}:{content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def truncate_messages_to_fit(
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    target_ratio: float = 0.5,
    strategy: str = "summarize",
    base_url: str = "",
    authorization: str = "",
    model: str = "",
    store: ReasoningStore | None = None,
    scope: str = "",
) -> tuple[list[dict[str, Any]], int]:
    """Stateful context truncation.

    On first truncation: drops old messages, generates summary, stores state.
    On subsequent requests: restores the stored summary (replacing the same old
    messages), preserving a stable prefix for DeepSeek's prompt cache.
    Only generates a new summary when additional truncation is needed.

    Returns (truncated_messages, dropped_count).
    """
    if max_tokens <= 0:
        return messages, 0

    system_messages, conversation_messages = _split_system_and_conversation(messages)

    # --- Phase 1: Try to apply stored truncation from previous request ---
    restored_from_cache = False
    conv_id = stable_conversation_id(messages) if store else ""

    if store and conv_id:
        stored = store.get_context_truncation(conv_id)
        if stored:
            stored_hash = stored["first_msg_hash"]
            stored_count = stored["dropped_count"]
            current_hash = _first_msg_hash(conversation_messages)

            if (
                current_hash == stored_hash
                and len(conversation_messages) >= stored_count
            ):
                # The old messages are still at the front — replace with stored summary
                summary_message: dict[str, Any] = {
                    "role": "system",
                    "content": stored["summary"],
                }
                conversation_messages = conversation_messages[stored_count:]
                system_messages = system_messages + [summary_message]
                restored_from_cache = True
                LOG.info(
                    "restored stored truncation: replaced %d old messages with cached summary "
                    "(saved ~%d tokens)",
                    stored_count,
                    stored["estimated_tokens_dropped"],
                )

    # --- Phase 2: Check if result fits within limit ---
    reconstructed = system_messages + conversation_messages
    current_tokens = estimate_messages_tokens(reconstructed)

    if current_tokens <= max_tokens:
        if restored_from_cache:
            return reconstructed, 0
        return messages, 0

    # --- Phase 3: Need (additional) truncation ---
    target_tokens = int(max_tokens * max(0.1, min(target_ratio, 0.9)))
    system_tokens = estimate_messages_tokens(system_messages)
    available_for_conversation = target_tokens - system_tokens - MAX_SUMMARY_TOKENS

    if available_for_conversation <= 0:
        return [system_messages[-1]] + conversation_messages[-1:], len(messages) - 2

    newly_dropped: list[dict[str, Any]] = []
    while conversation_messages and estimate_messages_tokens(conversation_messages) > available_for_conversation:
        first = conversation_messages[0]
        if first.get("role") == "assistant" and first.get("tool_calls"):
            newly_dropped.append(conversation_messages.pop(0))
            while conversation_messages and conversation_messages[0].get("role") == "tool":
                newly_dropped.append(conversation_messages.pop(0))
        else:
            newly_dropped.append(conversation_messages.pop(0))

    if not conversation_messages:
        return system_messages + messages[-1:], len(messages) - len(system_messages) - 1

    if newly_dropped:
        new_summary_text = _generate_summary(
            newly_dropped,
            strategy=strategy,
            base_url=base_url,
            authorization=authorization,
            model=model,
        )

        # Combine with previous summary if we had one
        if restored_from_cache and store:
            prev_summary = store.get_context_truncation(conv_id)
            if prev_summary:
                combined_summary = (
                    prev_summary["summary"]
                    + "\n\n[Additional context truncated]\n"
                    + new_summary_text
                )
                total_dropped = prev_summary["dropped_count"] + len(newly_dropped)
                total_tokens = prev_summary["estimated_tokens_dropped"] + estimate_messages_tokens(newly_dropped)
            else:
                combined_summary = new_summary_text
                total_dropped = len(newly_dropped)
                total_tokens = estimate_messages_tokens(newly_dropped)
        else:
            combined_summary = new_summary_text
            # Total dropped = messages from original input that were removed
            total_dropped = len(newly_dropped)
            if restored_from_cache and store:
                prev = store.get_context_truncation(conv_id)
                if prev:
                    total_dropped += prev["dropped_count"]
                    total_tokens = prev["estimated_tokens_dropped"] + estimate_messages_tokens(newly_dropped)
                else:
                    total_tokens = estimate_messages_tokens(newly_dropped)
            else:
                total_tokens = estimate_messages_tokens(newly_dropped)

        # Persist updated truncation state for next request
        if store and conv_id:
            # Compute hash against the ORIGINAL conversation start (before any replacement)
            orig_system, orig_conv = _split_system_and_conversation(messages)
            original_first_hash = _first_msg_hash(orig_conv)
            try:
                store.save_context_truncation(
                    conversation_id=conv_id,
                    dropped_count=total_dropped,
                    first_msg_hash=original_first_hash,
                    summary=combined_summary,
                    estimated_tokens_dropped=total_tokens,
                )
                LOG.info(
                    "saved truncation state: conversation=%s, total_dropped=%d, "
                    "tokens_saved=~%d",
                    conv_id[:8],
                    total_dropped,
                    total_tokens,
                )
            except Exception as exc:
                LOG.warning("failed to persist truncation state: %s", exc)

        # Remove old summary from system_messages if present (we'll inject combined)
        system_messages = [
            m for m in system_messages
            if not (m.get("role") == "system" and "[Context summary" in (m.get("content") or ""))
        ]
        summary_msg: dict[str, Any] = {"role": "system", "content": combined_summary}
        return system_messages + [summary_msg] + conversation_messages, len(newly_dropped)

    return system_messages + conversation_messages, 0


def _generate_summary(
    dropped_messages: list[dict[str, Any]],
    *,
    strategy: str,
    base_url: str,
    authorization: str,
    model: str,
) -> str:
    """Generate a summary of dropped messages using the configured strategy.
    Falls back to lightweight if API summarization fails."""
    if strategy == "summarize" and base_url and authorization and model:
        estimated_tokens = estimate_messages_tokens(dropped_messages)
        api_summary = generate_summary_via_api(
            dropped_messages,
            estimated_tokens,
            base_url=base_url,
            authorization=authorization,
            model=model,
        )
        if api_summary:
            return (
                f"[Context summary of {len(dropped_messages)} earlier messages "
                f"(generated by AI)]\n{api_summary}"
            )
        LOG.warning("API summarization failed, falling back to lightweight extraction")

    return summarize_dropped_messages(dropped_messages)


RECOVERY_NOTICE_TEXT = "[deepseek-cursor-proxy] Refreshed reasoning_content history."
RECOVERY_NOTICE_CONTENT = f"{RECOVERY_NOTICE_TEXT}\n\n"
RECOVERY_SYSTEM_CONTENT = (
    "deepseek-cursor-proxy recovered this request because older DeepSeek "
    "thinking-mode tool-call reasoning_content was unavailable. Older "
    "unrecoverable tool-call history was omitted; continue using only the "
    "remaining recovered context."
)


@dataclass(frozen=True)
class PreparedRequest:
    payload: dict[str, Any]
    original_model: str
    upstream_model: str
    cache_namespace: str
    patched_reasoning_messages: int
    missing_reasoning_messages: int
    recovered_reasoning_messages: int = 0
    recovery_dropped_messages: int = 0
    recovery_notice: str | None = None
    record_response_scope: str | None = None
    record_response_messages: list[dict[str, Any]] = field(default_factory=list)
    record_response_contexts: list[tuple[str, list[dict[str, Any]]]] = field(
        default_factory=list
    )
    reasoning_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    recovery_steps: list[dict[str, Any]] = field(default_factory=list)
    continued_recovery_boundary: bool = False
    retired_prefix_messages: int = 0


def normalize_reasoning_effort(value: Any) -> str:
    if not isinstance(value, str):
        return "high"
    return EFFORT_ALIASES.get(value.strip().lower(), "high")


def extract_text_content(content: Any) -> str | None:
    if content is None or isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            item_type = item.get("type")
            text = item.get("text") or item.get("content")
            if item_type in {"text", "input_text"} and isinstance(text, str):
                parts.append(text)
            elif isinstance(text, str):
                parts.append(text)
            elif item_type:
                parts.append(f"[{item_type} omitted by DeepSeek text proxy]")
        return "\n".join(part for part in parts if part)
    if isinstance(content, (dict, tuple)):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def strip_cursor_thinking_blocks(content: str) -> str:
    return CURSOR_THINKING_BLOCK_RE.sub("", content).lstrip("\r\n")


def normalize_tool_call(tool_call: Any) -> dict[str, Any]:
    if not isinstance(tool_call, dict):
        tool_call = {}
    function = tool_call.get("function") or {}
    if not isinstance(function, dict):
        function = {}

    arguments = function.get("arguments", "")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True)

    normalized: dict[str, Any] = {
        "id": str(tool_call.get("id") or ""),
        "type": tool_call.get("type") or "function",
        "function": {
            "name": str(function.get("name") or ""),
            "arguments": arguments,
        },
    }
    if not normalized["id"]:
        normalized.pop("id")
    return normalized


def normalize_tool(tool: Any) -> dict[str, Any]:
    if not isinstance(tool, dict):
        return {
            "type": "function",
            "function": {"name": "", "description": "", "parameters": {}},
        }
    normalized = dict(tool)
    normalized["type"] = normalized.get("type") or "function"
    function = normalized.get("function")
    if isinstance(function, dict):
        normalized["function"] = function
    return normalized


def legacy_function_to_tool(function: Any) -> dict[str, Any]:
    if not isinstance(function, dict):
        function = {}
    return {"type": "function", "function": function}


def convert_function_call(function_call: Any) -> Any:
    if isinstance(function_call, str):
        if function_call in {"auto", "none", "required"}:
            return function_call
        return None
    if isinstance(function_call, dict) and function_call.get("name"):
        return {
            "type": "function",
            "function": {"name": str(function_call["name"])},
        }
    return None


def normalize_tool_choice(tool_choice: Any) -> Any:
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "none", "required"}:
            return tool_choice
        return None
    if isinstance(tool_choice, dict):
        if tool_choice.get("type") == "function":
            function = tool_choice.get("function")
            if isinstance(function, dict) and function.get("name"):
                return {
                    "type": "function",
                    "function": {"name": str(function["name"])},
                }
        return tool_choice
    return tool_choice


def normalize_message(
    message: Any,
    store: ReasoningStore | None,
    prior_messages: list[dict[str, Any]],
    cache_namespace: str,
    repair_reasoning: bool,
    keep_reasoning: bool,
) -> tuple[dict[str, Any], bool, bool, dict[str, Any] | None]:
    if not isinstance(message, dict):
        message = {"role": "user", "content": str(message)}
    normalized = {key: value for key, value in message.items() if key in MESSAGE_FIELDS}
    role = normalized.get("role") or "user"
    normalized["role"] = role

    if role == "function":
        normalized["role"] = "tool"

    if "content" in normalized:
        normalized["content"] = extract_text_content(normalized["content"]) or ""
    elif normalized["role"] in {"assistant", "tool", "system", "user"}:
        normalized["content"] = ""
    if normalized["role"] == "assistant" and isinstance(normalized.get("content"), str):
        normalized["content"] = strip_cursor_thinking_blocks(normalized["content"])

    if normalized.get("tool_calls"):
        normalized["tool_calls"] = [
            normalize_tool_call(tool_call)
            for tool_call in normalized.get("tool_calls") or []
        ]

    patched = False
    missing = False
    diagnostic: dict[str, Any] | None = None
    if normalized["role"] == "assistant":
        if not keep_reasoning:
            normalized.pop("reasoning_content", None)
        elif repair_reasoning:
            reasoning = normalized.get("reasoning_content")
            if not isinstance(reasoning, str):
                normalized.pop("reasoning_content", None)
                needs_reasoning = assistant_needs_reasoning_for_tool_context(
                    normalized, prior_messages
                )
                lookup_scope = conversation_scope(prior_messages, cache_namespace)
                lookup_keys = (
                    reasoning_lookup_keys(
                        normalized,
                        lookup_scope,
                        cache_namespace,
                        prior_messages,
                    )
                    if needs_reasoning
                    else []
                )
                hit_kind = None
                if needs_reasoning and store is not None:
                    for lookup_key in lookup_keys:
                        restored = store.get(str(lookup_key["key"]))
                        if restored is not None:
                            lookup_key["hit"] = True
                            hit_kind = lookup_key["kind"]
                            normalized["reasoning_content"] = restored
                            patched = True
                            if not lookup_key.get("portable"):
                                store.backfill_portable_aliases(
                                    normalized,
                                    restored,
                                    cache_namespace,
                                    prior_messages,
                                )
                            break
                if needs_reasoning and not patched:
                    missing = True
                if needs_reasoning:
                    diagnostic = {
                        "message_index": len(prior_messages),
                        "role": "assistant",
                        "needs_reasoning": True,
                        "had_reasoning_content": False,
                        "patched": patched,
                        "missing": missing,
                        "lookup_scope": lookup_scope,
                        "message_signature": message_signature(normalized),
                        "tool_call_ids": tool_call_ids(normalized),
                        "lookup_keys": lookup_keys,
                        "hit_kind": hit_kind,
                    }
            elif assistant_needs_reasoning_for_tool_context(normalized, prior_messages):
                diagnostic = {
                    "message_index": len(prior_messages),
                    "role": "assistant",
                    "needs_reasoning": True,
                    "had_reasoning_content": True,
                    "patched": False,
                    "missing": False,
                    "lookup_scope": conversation_scope(prior_messages, cache_namespace),
                    "message_signature": message_signature(normalized),
                    "tool_call_ids": tool_call_ids(normalized),
                    "lookup_keys": [],
                    "hit_kind": "request",
                }

    allowed_fields = ROLE_MESSAGE_FIELDS.get(str(normalized["role"]), MESSAGE_FIELDS)
    normalized = {
        key: value for key, value in normalized.items() if key in allowed_fields
    }
    return normalized, patched, missing, diagnostic


def reasoning_lookup_keys(
    message: dict[str, Any],
    scope: str,
    cache_namespace: str = "",
    prior_messages: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    keys = [
        {
            "kind": "message_signature",
            "key": f"scope:{scope}:signature:{message_signature(message)}",
            "portable": False,
            "hit": False,
        }
    ]
    keys.extend(
        {
            "kind": "tool_call_id",
            "tool_call_id": tool_call_id,
            "key": f"scope:{scope}:tool_call:{tool_call_id}",
            "portable": False,
            "hit": False,
        }
        for tool_call_id in tool_call_ids(message)
    )
    keys.extend(
        {
            "kind": "tool_call_signature",
            "function_name": str((tool_call.get("function") or {}).get("name") or ""),
            "key": (
                f"scope:{scope}:tool_call_signature:"
                f"{tool_call_signature(tool_call)}"
            ),
            "portable": False,
            "hit": False,
        }
        for tool_call in (message.get("tool_calls") or [])
        if isinstance(tool_call, dict)
    )
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
    if cache_namespace and prior_messages is not None:
        turn_signature = turn_context_signature(prior_messages)
        keys.append(
            {
                "kind": "portable_message_signature",
                "key": (
                    f"namespace:{cache_namespace}:turn:{turn_signature}:"
                    f"signature:{message_signature(message)}"
                ),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
        )
        keys.extend(
            {
                "kind": "portable_tool_call_id",
                "tool_call_id": tool_call_id,
                "key": (
                    f"namespace:{cache_namespace}:turn:{turn_signature}:"
                    f"tool_call:{tool_call_id}"
                ),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
            for tool_call_id in tool_call_ids(message)
        )
        keys.extend(
            {
                "kind": "portable_tool_call_signature",
                "function_name": str(
                    (tool_call.get("function") or {}).get("name") or ""
                ),
                "key": (
                    f"namespace:{cache_namespace}:turn:{turn_signature}:"
                    f"tool_call_signature:{tool_call_signature(tool_call)}"
                ),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
            for tool_call in (message.get("tool_calls") or [])
            if isinstance(tool_call, dict)
        )
        keys.extend(
            {
                "kind": "portable_tool_name",
                "function_name": tool_name,
                "key": (
                    f"namespace:{cache_namespace}:turn:{turn_signature}:"
                    f"tool_name:{tool_name}"
                ),
                "turn_context_signature": turn_signature,
                "portable": True,
                "hit": False,
            }
            for tool_name in tool_call_names(message)
        )
    return keys


def normalize_messages(
    messages: Any,
    store: ReasoningStore | None,
    cache_namespace: str,
    repair_reasoning: bool,
    keep_reasoning: bool,
) -> tuple[list[dict[str, Any]], int, list[int], list[dict[str, Any]]]:
    if not isinstance(messages, list):
        return [], 0, [], []
    normalized_messages: list[dict[str, Any]] = []
    patched_count = 0
    missing_indexes: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for message in messages:
        normalized, patched, missing, diagnostic = normalize_message(
            message,
            store,
            normalized_messages,
            cache_namespace,
            repair_reasoning,
            keep_reasoning,
        )
        normalized_messages.append(normalized)
        if patched:
            patched_count += 1
        if missing:
            missing_indexes.append(len(normalized_messages) - 1)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return normalized_messages, patched_count, missing_indexes, diagnostics


def has_recovery_notice(message: dict[str, Any]) -> bool:
    content = message.get("content")
    return (
        message.get("role") == "assistant"
        and isinstance(content, str)
        and content.startswith(RECOVERY_NOTICE_TEXT)
    )


def strip_recovery_notice_for_upstream(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Cursor echoes the proxy's recovery notice back to us in later turns.
    The notice serves as a boundary marker for the proxy, but DeepSeek must
    not see proxy-generated prose. Return a copy with assistant prefixes
    stripped; leave the input untouched so cache scopes/recording contexts
    keep matching the with-prefix history that Cursor will send next time."""
    stripped: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            stripped.append(message)
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.startswith(RECOVERY_NOTICE_TEXT):
            stripped.append(message)
            continue
        cleaned = dict(message)
        cleaned["content"] = content[len(RECOVERY_NOTICE_TEXT) :].lstrip("\r\n")
        stripped.append(cleaned)
    return stripped


def leading_system_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leading_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "system":
            leading_messages.append(message)
            continue
        break
    return leading_messages


def active_messages_from_recovery_boundary(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, dict[str, Any]] | None:
    recovery_boundary_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if has_recovery_notice(messages[index])
        ),
        -1,
    )
    if recovery_boundary_index == -1:
        return None

    context_user_index = next(
        (
            index
            for index in range(recovery_boundary_index - 1, -1, -1)
            if messages[index].get("role") == "user"
        ),
        -1,
    )
    leading_messages = leading_system_messages(messages)
    recovered_tail = []
    if context_user_index != -1:
        recovered_tail.append(messages[context_user_index])
    recovered_tail.extend(messages[recovery_boundary_index:])
    active_messages = [
        *leading_messages,
        {"role": "system", "content": RECOVERY_SYSTEM_CONTENT},
        *recovered_tail,
    ]
    kept_context_messages = 1 if context_user_index != -1 else 0
    retired_messages = (
        recovery_boundary_index - len(leading_messages) - kept_context_messages
    )
    retired_messages = max(retired_messages, 0)
    step = {
        "strategy": "continued_recovery_boundary",
        "recovery_boundary_index": recovery_boundary_index,
        "context_user_index": context_user_index,
        "retired_prefix_messages": retired_messages,
    }
    return active_messages, retired_messages, step


def recover_messages_from_missing_reasoning(
    messages: list[dict[str, Any]],
    missing_indexes: list[int],
) -> tuple[list[dict[str, Any]], int, str | None, dict[str, Any]]:
    recovery_boundary_index = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if has_recovery_notice(messages[index])
            and any(missing_index < index for missing_index in missing_indexes)
        ),
        -1,
    )
    if recovery_boundary_index != -1:
        context_user_index = next(
            (
                index
                for index in range(recovery_boundary_index - 1, -1, -1)
                if messages[index].get("role") == "user"
            ),
            -1,
        )
        leading_messages = leading_system_messages(messages)
        recovered_tail = []
        if context_user_index != -1:
            recovered_tail.append(messages[context_user_index])
        recovered_tail.extend(messages[recovery_boundary_index:])
        recovered = [
            *leading_messages,
            {"role": "system", "content": RECOVERY_SYSTEM_CONTENT},
            *recovered_tail,
        ]
        kept_context_messages = 1 if context_user_index != -1 else 0
        omitted_messages = (
            recovery_boundary_index - len(leading_messages) - kept_context_messages
        )
        return (
            recovered,
            omitted_messages,
            None,
            {
                "strategy": "recovery_boundary",
                "missing_indexes": missing_indexes,
                "recovery_boundary_index": recovery_boundary_index,
                "context_user_index": context_user_index,
                "dropped_messages": omitted_messages,
                "notice": None,
            },
        )

    # Instead of dropping the entire conversation, inject placeholder reasoning
    # for messages that need it. This preserves context while satisfying the API
    # requirement that tool-call assistant messages have reasoning_content.
    patched_messages = list(messages)
    for idx in missing_indexes:
        if idx < len(patched_messages):
            msg = dict(patched_messages[idx])
            msg["reasoning_content"] = "..."
            patched_messages[idx] = msg
    LOG.info(
        "injected placeholder reasoning for %d messages (preserving full context)",
        len(missing_indexes),
    )
    return (
        patched_messages,
        0,
        None,
        {
            "strategy": "placeholder_reasoning",
            "missing_indexes": missing_indexes,
            "dropped_messages": 0,
            "notice": None,
        },
    )


def assistant_needs_reasoning_for_tool_context(
    message: dict[str, Any],
    prior_messages: list[dict[str, Any]],
) -> bool:
    if message.get("tool_calls"):
        return True
    for prior_message in reversed(prior_messages):
        role = prior_message.get("role")
        if role == "tool":
            return True
        if role in {"user", "system"}:
            return False
    return False


def upstream_model_for(original_model: str, config: ProxyConfig) -> str:
    if original_model.startswith("deepseek-"):
        return original_model
    LOG.warning(
        "rewriting non-DeepSeek model %r to configured fallback %r",
        original_model,
        config.upstream_model,
    )
    return config.upstream_model


def reasoning_model_family(upstream_model: str) -> str:
    if upstream_model in {"deepseek-v4-pro", "deepseek-v4-flash"}:
        return "deepseek-v4"
    return upstream_model


def reasoning_cache_namespace(
    config: ProxyConfig,
    upstream_model: str,
    thinking: Any,
    reasoning_effort: Any,
    authorization: str | None = None,
) -> str:
    auth_hash = ""
    if authorization:
        auth_hash = hashlib.sha256(authorization.encode("utf-8")).hexdigest()
    payload = {
        "base_url": config.upstream_base_url,
        "model": reasoning_model_family(upstream_model),
        "thinking": thinking,
        "reasoning_effort": reasoning_effort,
        "authorization_hash": auth_hash,
    }
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def response_recording_contexts(
    *items: tuple[str, list[dict[str, Any]]] | None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    contexts: list[tuple[str, list[dict[str, Any]]]] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        scope, messages = item
        if scope in seen:
            continue
        seen.add(scope)
        contexts.append((scope, messages))
    return contexts


def prepare_upstream_request(
    payload: dict[str, Any],
    config: ProxyConfig,
    store: ReasoningStore | None,
    authorization: str | None = None,
) -> PreparedRequest:
    original_model = str(payload.get("model") or config.upstream_model)
    upstream_model = upstream_model_for(original_model, config)

    prepared = {
        key: value for key, value in payload.items() if key in SUPPORTED_REQUEST_FIELDS
    }
    dropped_fields = sorted(
        key
        for key in payload.keys()
        if key not in SUPPORTED_REQUEST_FIELDS
        and key not in {"max_completion_tokens", "functions", "function_call"}
    )
    if dropped_fields:
        LOG.warning(
            "dropping unsupported request field(s): %s", ", ".join(dropped_fields)
        )
    if "max_tokens" not in prepared and "max_completion_tokens" in payload:
        prepared["max_tokens"] = payload["max_completion_tokens"]

    prepared["model"] = upstream_model
    if prepared.get("stream"):
        stream_options = prepared.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        else:
            stream_options = dict(stream_options)
        stream_options["include_usage"] = True
        prepared["stream_options"] = stream_options

    if "tools" in prepared and isinstance(prepared["tools"], list):
        prepared["tools"] = [normalize_tool(tool) for tool in prepared["tools"]]
    elif isinstance(payload.get("functions"), list):
        prepared["tools"] = [
            legacy_function_to_tool(function) for function in payload["functions"]
        ]

    if "tool_choice" in prepared:
        tool_choice = normalize_tool_choice(prepared["tool_choice"])
        if tool_choice is None:
            prepared.pop("tool_choice", None)
        else:
            prepared["tool_choice"] = tool_choice
    elif "function_call" in payload:
        tool_choice = convert_function_call(payload.get("function_call"))
        if tool_choice is not None:
            prepared["tool_choice"] = tool_choice

    prepared["thinking"] = {"type": config.thinking}
    thinking_enabled = config.thinking == "enabled"
    thinking_disabled = config.thinking == "disabled"
    if thinking_enabled:
        prepared["reasoning_effort"] = normalize_reasoning_effort(
            config.reasoning_effort
        )

    cache_namespace = reasoning_cache_namespace(
        config,
        upstream_model,
        prepared.get("thinking"),
        prepared.get("reasoning_effort"),
        authorization,
    )
    pre_repair_messages, _, _, _ = normalize_messages(
        payload.get("messages"),
        None,
        cache_namespace,
        repair_reasoning=False,
        keep_reasoning=not thinking_disabled,
    )
    record_response_messages = pre_repair_messages
    record_response_scope = conversation_scope(
        record_response_messages, cache_namespace
    )
    messages_for_repair = pre_repair_messages
    continued_recovery_boundary = False
    retired_prefix_messages = 0
    recovered_count = 0
    recovery_dropped_messages = 0
    recovery_notice = None
    recovery_steps: list[dict[str, Any]] = []
    if thinking_enabled and config.missing_reasoning_strategy == "recover":
        boundary = active_messages_from_recovery_boundary(pre_repair_messages)
        if boundary is not None:
            messages_for_repair, retired_prefix_messages, boundary_step = boundary
            continued_recovery_boundary = True
            recovery_steps.append(boundary_step)

    messages, patched_count, missing_indexes, reasoning_diagnostics = (
        normalize_messages(
            messages_for_repair,
            store,
            cache_namespace,
            repair_reasoning=thinking_enabled,
            keep_reasoning=not thinking_disabled,
        )
    )
    if missing_indexes:
        # Log detailed diagnostics to help identify why reasoning was missing
        total_assistant = sum(1 for m in messages if m.get("role") == "assistant")
        assistant_with_tools = sum(
            1 for m in messages
            if m.get("role") == "assistant" and m.get("tool_calls")
        )
        roles_summary = {}
        for m in messages:
            r = m.get("role", "unknown")
            roles_summary[r] = roles_summary.get(r, 0) + 1
        LOG.warning(
            "reasoning recovery triggered: %d missing out of %d messages "
            "(assistant=%d, with_tool_calls=%d, roles=%s, "
            "has_recovery_boundary=%s, patched=%d)",
            len(missing_indexes),
            len(messages),
            total_assistant,
            assistant_with_tools,
            roles_summary,
            continued_recovery_boundary,
            patched_count,
        )
        # Log the first few missing message details
        for idx in missing_indexes[:5]:
            msg = messages[idx] if idx < len(messages) else {}
            tc_names = [
                tc.get("function", {}).get("name", "?")
                for tc in (msg.get("tool_calls") or [])
                if isinstance(tc, dict)
            ]
            LOG.warning(
                "  missing[%d]: role=%s tool_calls=%s content_len=%d",
                idx,
                msg.get("role"),
                tc_names or None,
                len(msg.get("content") or ""),
            )
    while missing_indexes and config.missing_reasoning_strategy == "recover":
        recovered_messages, dropped_messages, notice, recovery_step = (
            recover_messages_from_missing_reasoning(messages, missing_indexes)
        )
        recovery_steps.append(recovery_step)
        if not dropped_messages:
            # Placeholder reasoning was injected — apply patched messages and stop
            messages = recovered_messages
            missing_indexes = []
            break
        recovered_count += len(missing_indexes)
        recovery_dropped_messages += dropped_messages
        if notice:
            recovery_notice = notice
        (
            messages,
            patched_count,
            missing_indexes,
            latest_diagnostics,
        ) = normalize_messages(
            recovered_messages,
            store,
            cache_namespace,
            repair_reasoning=thinking_enabled,
            keep_reasoning=not thinking_disabled,
        )
        reasoning_diagnostics.extend(latest_diagnostics)
    active_record_response_scope = conversation_scope(messages, cache_namespace)
    record_response_contexts = response_recording_contexts(
        (record_response_scope, record_response_messages),
        (active_record_response_scope, messages),
    )
    final_messages = strip_recovery_notice_for_upstream(messages)

    # Compress individual long messages before checking overall context size
    if config.max_single_message_tokens > 0:
        final_messages = compress_messages(
            final_messages, config.max_single_message_tokens
        )

    # Truncate context if it exceeds the configured token limit
    context_truncated_messages = 0
    if config.max_context_tokens > 0:
        estimated_input = estimate_messages_tokens(final_messages)
        LOG.info(
            "context estimate: ~%d tokens (limit=%d)",
            estimated_input,
            config.max_context_tokens,
        )
        summary_model = config.summary_model or upstream_model
        final_messages, context_truncated_messages = truncate_messages_to_fit(
            final_messages,
            config.max_context_tokens,
            target_ratio=config.truncation_target_ratio,
            strategy=config.context_overflow_strategy,
            base_url=config.upstream_base_url,
            authorization=authorization or "",
            model=summary_model,
            store=store,
            scope=active_record_response_scope,
        )
        if context_truncated_messages > 0:
            LOG.warning(
                "context truncation dropped %d message(s) to fit within %d token limit "
                "(strategy=%s, estimated %d tokens before truncation)",
                context_truncated_messages,
                config.max_context_tokens,
                config.context_overflow_strategy,
                estimate_messages_tokens(messages),
            )

    prepared["messages"] = final_messages

    return PreparedRequest(
        payload=prepared,
        original_model=original_model,
        upstream_model=upstream_model,
        cache_namespace=cache_namespace,
        patched_reasoning_messages=patched_count,
        missing_reasoning_messages=len(missing_indexes),
        recovered_reasoning_messages=recovered_count,
        recovery_dropped_messages=recovery_dropped_messages,
        recovery_notice=recovery_notice,
        record_response_scope=record_response_scope,
        record_response_messages=record_response_messages,
        record_response_contexts=record_response_contexts,
        reasoning_diagnostics=reasoning_diagnostics,
        recovery_steps=recovery_steps,
        continued_recovery_boundary=continued_recovery_boundary,
        retired_prefix_messages=retired_prefix_messages,
    )


def record_response_reasoning(
    response_payload: dict[str, Any],
    store: ReasoningStore | None,
    request_messages: list[dict[str, Any]],
    cache_namespace: str = "",
    scope: str | None = None,
    prior_messages: list[dict[str, Any]] | None = None,
    recording_contexts: list[tuple[str, list[dict[str, Any]]]] | None = None,
) -> int:
    if store is None:
        return 0
    stored = 0
    choices = response_payload.get("choices")
    if not isinstance(choices, list):
        return stored
    if recording_contexts is None:
        response_scope = (
            scope
            if scope is not None
            else conversation_scope(request_messages, cache_namespace)
        )
        response_prior_messages = (
            prior_messages if prior_messages is not None else request_messages
        )
        recording_contexts = [(response_scope, response_prior_messages)]
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if isinstance(message, dict):
            for response_scope, response_prior_messages in recording_contexts:
                stored += store.store_assistant_message(
                    message,
                    response_scope,
                    cache_namespace,
                    response_prior_messages,
                )
    return stored


def rewrite_response_body(
    body: bytes,
    original_model: str,
    store: ReasoningStore | None,
    request_messages: list[dict[str, Any]],
    cache_namespace: str = "",
    content_prefix: str | None = None,
    scope: str | None = None,
    prior_messages: list[dict[str, Any]] | None = None,
    recording_contexts: list[tuple[str, list[dict[str, Any]]]] | None = None,
    display_reasoning: bool = False,
    collapsible_reasoning: bool = True,
) -> bytes:
    response_payload = json.loads(body.decode("utf-8"))
    if isinstance(response_payload, dict):
        if content_prefix:
            prefix_response_content(response_payload, content_prefix)
        record_response_reasoning(
            response_payload,
            store,
            request_messages,
            cache_namespace,
            scope=scope,
            prior_messages=prior_messages,
            recording_contexts=recording_contexts,
        )
        if display_reasoning:
            fold_reasoning_into_content(response_payload, collapsible_reasoning)
        if "model" in response_payload:
            response_payload["model"] = original_model
    return json.dumps(
        response_payload, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def prefix_response_content(response_payload: dict[str, Any], prefix: str) -> bool:
    choices = response_payload.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        message["content"] = prefix + (content if isinstance(content, str) else "")
        return True
    return False
