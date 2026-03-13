"""
Token utilities with character-based estimation.

Uses conservative character-based estimation for context window management.
"""

import json

# Conservative chars-per-token that errs on overestimating (safer than underestimating)
_CHARS_PER_TOKEN_FALLBACK = 3.5

# Keep this for backward compatibility — no longer used but referenced in some imports
_QWEN_TOKENIZER_AVAILABLE = False


def count_tokens(text: str) -> int:
    """Estimate token count from text using character-based heuristic."""
    if text:
        return max(1, int(len(text) / _CHARS_PER_TOKEN_FALLBACK))
    return 0


def count_tokens_from_message(msg: dict) -> int:
    """Count tokens for a message dict (content + tool_calls if present)."""
    total = 0
    content = msg.get("content", "") or ""
    total += count_tokens(content)

    if msg.get("tool_calls"):
        total += count_tokens(json.dumps(msg["tool_calls"]))

    return max(1, total)


def count_tools_tokens(tools: list) -> int:
    """Count token overhead of tool schemas."""
    return count_tokens(json.dumps(tools))
