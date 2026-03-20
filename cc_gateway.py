"""
Claude Code Gateway — Anthropic Messages API → LLM Sandbox Proxy

Exposes an Anthropic-compatible /v1/messages endpoint that translates
requests to the UCSB LLM Sandbox Bot API format, enabling Claude Code
to use Sandbox-hosted models.

Tool use is supported by injecting tool definitions into the system prompt
and parsing XML tool_call blocks from the model's text output, converting
them to Anthropic-format tool_use content blocks.

Usage:
    BEDROCK_API_URL=https://... BEDROCK_API_KEY=... uvicorn cc_gateway:app --port 8781

Then configure Claude Code:
    export ANTHROPIC_BASE_URL=http://localhost:8781
    export ANTHROPIC_API_KEY=dummy
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
import logging
from typing import List, Optional, Tuple, Union

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

_log_level = os.environ.get("CC_GATEWAY_LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, _log_level, logging.INFO))
log = logging.getLogger("cc-gateway")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get("BEDROCK_API_URL")
API_KEY = os.environ.get("BEDROCK_API_KEY")
POLL_TIMEOUT = int(os.environ.get("POLL_TIMEOUT", "300"))
POLL_INITIAL_INTERVAL = float(os.environ.get("POLL_INITIAL_INTERVAL", "0.3"))
POLL_BACKOFF_MULTIPLIER = float(os.environ.get("POLL_BACKOFF_MULTIPLIER", "1.5"))
POLL_MAX_INTERVAL = float(os.environ.get("POLL_MAX_INTERVAL", "5.0"))
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "claude-v4.5-sonnet")

if not API_URL or not API_KEY:
    raise RuntimeError("BEDROCK_API_URL and BEDROCK_API_KEY must be set")

BOT_HEADERS = {"x-api-key": API_KEY, "Content-Type": "application/json"}

# ---------------------------------------------------------------------------
# Model mapping: Anthropic API names → Sandbox names
# ---------------------------------------------------------------------------
MODEL_MAP = {
    # 4.6 models not yet on Sandbox — fall back to 4.5 equivalents
    "claude-opus-4-6": "claude-v4.5-opus",
    "claude-opus-4-6-20250612": "claude-v4.5-opus",
    "claude-sonnet-4-6": "claude-v4.5-sonnet",
    "claude-sonnet-4-6-20250514": "claude-v4.5-sonnet",
    # 4.5 models
    "claude-opus-4-5": "claude-v4.5-opus",
    "claude-sonnet-4-5": "claude-v4.5-sonnet",
    "claude-sonnet-4-5-20250514": "claude-v4.5-sonnet",
    "claude-haiku-4-5": "claude-v4.5-haiku",
    "claude-haiku-4-5-20251001": "claude-v4.5-haiku",
}


def resolve_model(model: str) -> str:
    return MODEL_MAP.get(model, model)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Claude Code Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Anthropic-compatible request model
# ---------------------------------------------------------------------------
class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]


class MessagesRequest(BaseModel):
    model: str = DEFAULT_MODEL
    max_tokens: int = 8192
    messages: List[AnthropicMessage]
    system: Optional[Union[str, list]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Optional[dict] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[dict] = None


# ---------------------------------------------------------------------------
# Translate Anthropic messages → Sandbox content blocks
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN = 4

# Regex to extract tool_call XML blocks from model text output
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def estimate_tokens(text: str) -> int:
    return max(len(text) // CHARS_PER_TOKEN, 1)


def format_tools_for_prompt(tools: List[dict]) -> str:
    """Convert Anthropic tool definitions to compact text for system prompt injection."""
    if not tools:
        return ""

    lines = ["\n\nTOOLS (use <tool_call> format below):"]
    for tool in tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        schema = tool.get("input_schema", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Compact one-line param signatures: name(param1: type, param2?: type)
        params = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "str")
            opt = "" if pname in required else "?"
            params.append(f"{pname}{opt}: {ptype}")
        sig = f"{name}({', '.join(params)})" if params else f"{name}()"

        # Truncate description to first sentence
        short_desc = desc.split(". ")[0].split(".\n")[0]
        if len(short_desc) > 120:
            short_desc = short_desc[:117] + "..."
        lines.append(f"- {sig} — {short_desc}")

    lines.append("""
FORMAT: <tool_call>
{"tool": "name", "args": {"param": "value"}}
</tool_call>
Multiple calls allowed per response. When done, reply with plain text only (no <tool_call>). Do NOT repeat successful calls.""")

    return "\n".join(lines)


def parse_tool_calls(text: str) -> List[dict]:
    """Extract tool_call XML blocks from model text and return as Anthropic tool_use dicts."""
    calls = []
    for match in _TOOL_CALL_RE.finditer(text):
        try:
            data = json.loads(match.group(1))
            name = data.get("tool") or data.get("name")
            args = data.get("args") or data.get("arguments") or data.get("input") or {}
            # Fallback: remaining keys as args
            if not args and name:
                args = {k: v for k, v in data.items() if k not in ("tool", "name", "arguments", "input")}
            if name:
                calls.append({
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": name,
                    "input": args,
                })
        except json.JSONDecodeError:
            continue
    return calls


# ---------------------------------------------------------------------------
# Tool call validation
# ---------------------------------------------------------------------------
def _build_tool_index(tools: List[dict]) -> dict:
    """Build name → input_schema map from the request's tool definitions."""
    index = {}
    for tool in tools:
        name = tool.get("name", "")
        if name:
            index[name] = tool.get("input_schema", {})
    return index


def _validate_args(args: dict, schema: dict) -> List[str]:
    """Validate tool call args against its input_schema. Returns list of errors."""
    errors = []
    props = schema.get("properties", {})
    required = schema.get("required", [])

    # Check required params are present
    for req_param in required:
        if req_param not in args:
            errors.append(f"missing required param '{req_param}'")

    # Check no unknown params (only if schema defines properties)
    if props:
        for key in args:
            if key not in props:
                errors.append(f"unknown param '{key}'")

    # Check basic type conformance for present params
    type_map = {"string": str, "integer": int, "number": (int, float),
                "boolean": bool, "array": list, "object": dict}
    for key, value in args.items():
        if key not in props:
            continue
        expected_type = props[key].get("type")
        if expected_type and expected_type in type_map:
            if not isinstance(value, type_map[expected_type]):
                # Allow int where number expected
                if expected_type == "number" and isinstance(value, (int, float)):
                    continue
                errors.append(f"param '{key}' expected {expected_type}, got {type(value).__name__}")

    return errors


def validate_tool_calls(tool_calls: List[dict], tools: List[dict]) -> List[dict]:
    """Validate and filter tool calls. Returns only valid calls, logs rejections."""
    if not tools:
        return tool_calls

    index = _build_tool_index(tools)
    valid = []

    for tc in tool_calls:
        name = tc["name"]

        # Allowlist check
        if name not in index:
            log.warning("REJECTED tool call: '%s' not in allowlist (%d tools)",
                        name, len(index))
            continue

        # Schema validation
        schema = index[name]
        errors = _validate_args(tc["input"], schema)
        if errors:
            log.warning("REJECTED tool call: %s — %s", name, "; ".join(errors))
            continue

        valid.append(tc)

    rejected = len(tool_calls) - len(valid)
    if rejected:
        log.info("Validated tool calls: %d passed, %d rejected", len(valid), rejected)

    return valid


def strip_tool_calls(text: str) -> str:
    """Remove tool_call XML blocks from text."""
    return _TOOL_CALL_RE.sub("", text).strip()


def _get_block_dict(block) -> dict:
    """Normalize a content block to a plain dict."""
    if isinstance(block, dict):
        return block
    # Pydantic model or similar
    return block.model_dump() if hasattr(block, "model_dump") else block.__dict__


def assemble_content(req: MessagesRequest) -> List[dict]:
    """Flatten Anthropic messages into Sandbox content blocks with role prefixes."""
    content_blocks = []

    # Build tool_use_id → tool_name map for resolving tool results
    tool_id_to_name: dict[str, str] = {}
    for msg in req.messages:
        if isinstance(msg.content, list):
            for block in msg.content:
                b = _get_block_dict(block)
                if b.get("type") == "tool_use":
                    tool_id_to_name[b.get("id", "")] = b.get("name", "unknown")

    # System prompt
    system_text = ""
    if req.system:
        if isinstance(req.system, str):
            system_text = req.system
        elif isinstance(req.system, list):
            parts = []
            for block in req.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            system_text = "\n".join(parts)

    # Inject tool definitions into system prompt
    if req.tools:
        system_text += format_tools_for_prompt(req.tools)

    if system_text:
        content_blocks.append({
            "contentType": "text",
            "body": f"System instructions: {system_text}",
        })

    # Messages
    for msg in req.messages:
        role_prefix = "User" if msg.role == "user" else "Assistant"
        content = msg.content

        if isinstance(content, str):
            content_blocks.append({
                "contentType": "text",
                "body": f"{role_prefix}: {content}",
            })
        elif isinstance(content, list):
            for block in content:
                b = _get_block_dict(block)
                block_type = b.get("type", "")

                if block_type == "text":
                    text = b.get("text", "")
                    if text:
                        content_blocks.append({
                            "contentType": "text",
                            "body": f"{role_prefix}: {text}",
                        })

                elif block_type == "image":
                    source = b.get("source", {})
                    if source.get("type") == "base64":
                        content_blocks.append({
                            "contentType": "image",
                            "mediaType": source.get("media_type", "image/png"),
                            "body": source.get("data", ""),
                        })

                elif block_type == "tool_use":
                    # Assistant's tool call — include as text so model sees what it did
                    name = b.get("name", "")
                    input_data = b.get("input", {})
                    content_blocks.append({
                        "contentType": "text",
                        "body": (
                            f"Assistant: I used the {name} tool:\n"
                            f"<tool_call>\n"
                            f'{json.dumps({"tool": name, "args": input_data})}\n'
                            f"</tool_call>"
                        ),
                    })

                elif block_type == "tool_result":
                    # User's tool result — format with tool name for clarity
                    tool_use_id = b.get("tool_use_id", "")
                    is_error = b.get("is_error", False)
                    result_content = b.get("content", "")
                    tool_name = tool_id_to_name.get(tool_use_id, "unknown")

                    # content can be string or list of blocks
                    if isinstance(result_content, list):
                        parts = []
                        for rb in result_content:
                            if isinstance(rb, dict) and rb.get("type") == "text":
                                parts.append(rb.get("text", ""))
                        result_content = "\n".join(parts)

                    if is_error:
                        label = f"Tool ERROR from {tool_name}"
                    else:
                        label = f"Tool result from {tool_name} (success)"
                    content_blocks.append({
                        "contentType": "text",
                        "body": f"User: [{label}]: {result_content}",
                    })

    log.debug("Assembled %d content blocks for Sandbox", len(content_blocks))
    for i, cb in enumerate(content_blocks):
        body_preview = cb.get("body", "")[:200]
        log.debug("  Block %d [%s]: %s", i, cb.get("contentType"), body_preview)

    return content_blocks


# ---------------------------------------------------------------------------
# Send + Poll
# ---------------------------------------------------------------------------
def _is_turn_complete(msg: dict) -> bool:
    """Check if an assistant message represents a complete turn."""
    stop_reason = msg.get("stop_reason") or msg.get("stopReason") or ""
    if stop_reason == "tool_use":
        return False
    if stop_reason in ("end_turn", "stop", "max_tokens"):
        return True

    content = msg.get("content", [])
    has_text = any(
        c.get("contentType") == "text" and c.get("body", "").strip()
        for c in content
    )
    has_tool_use = any(c.get("contentType") == "toolUse" for c in content)

    if has_tool_use and not has_text:
        return False
    return has_text


def poll_for_reply(conversation_id: str, message_id: str) -> Tuple[str, str]:
    """Poll until complete. Returns (reply_text, stop_reason)."""
    interval = POLL_INITIAL_INTERVAL
    deadline = time.monotonic() + POLL_TIMEOUT
    attempt = 0

    while time.monotonic() < deadline:
        time.sleep(interval)
        attempt += 1

        resp = requests.get(
            f"{API_URL}/conversation/{conversation_id}/{message_id}",
            headers=BOT_HEADERS,
        )

        if resp.status_code == 429:
            log.warning("Poll attempt %d — rate limited, backing off", attempt)
            interval = POLL_MAX_INTERVAL
            continue

        if resp.status_code == 404:
            log.debug("Poll attempt %d — not ready yet", attempt)
            interval = min(interval * POLL_BACKOFF_MULTIPLIER, POLL_MAX_INTERVAL)
            continue

        resp.raise_for_status()
        data = resp.json()
        msg = data.get("message", {})

        if msg.get("role") == "assistant":
            if not _is_turn_complete(msg):
                log.debug("Poll attempt %d — tool use in progress", attempt)
                interval = min(interval * POLL_BACKOFF_MULTIPLIER, POLL_MAX_INTERVAL)
                continue

            content = msg.get("content", [])
            text_parts = [
                c.get("body", "")
                for c in content
                if c.get("contentType") == "text" and c.get("body", "").strip()
            ]
            if text_parts:
                stop = msg.get("stop_reason") or msg.get("stopReason") or "end_turn"
                return "\n\n".join(text_parts), stop

        log.debug("Poll attempt %d — no reply yet (next in %.1fs)", attempt, interval)
        interval = min(interval * POLL_BACKOFF_MULTIPLIER, POLL_MAX_INTERVAL)

    raise HTTPException(status_code=504, detail="Timed out waiting for Sandbox response")


def call_sandbox(req: MessagesRequest) -> Tuple[str, str, str]:
    """Send to Sandbox and return (reply_text, sandbox_model, stop_reason)."""
    sandbox_model = resolve_model(req.model)
    content_blocks = assemble_content(req)

    log.info("Sending to Sandbox: model=%s, %d content blocks, tools=%s",
             sandbox_model, len(content_blocks),
             len(req.tools) if req.tools else 0)

    # Log incoming message structure for debugging
    for i, msg in enumerate(req.messages):
        if isinstance(msg.content, list):
            types = [_get_block_dict(b).get("type", "?") for b in msg.content]
            log.debug("  Message %d [%s]: %s", i, msg.role, types)
        else:
            log.debug("  Message %d [%s]: text(%d chars)", i, msg.role, len(str(msg.content)))

    payload = {
        "message": {
            "role": "user",
            "parent_message_id": None,
            "content": content_blocks,
            "model": sandbox_model,
        },
    }

    log.debug("Payload: %s", json.dumps(payload, default=str)[:2000])

    post_resp = requests.post(
        f"{API_URL}/conversation", headers=BOT_HEADERS, json=payload
    )
    if post_resp.status_code != 200:
        log.error("Sandbox returned %d: %s", post_resp.status_code, post_resp.text[:500])
    post_resp.raise_for_status()
    resp_data = post_resp.json()
    message_id = resp_data.get("messageId")
    conv_id = resp_data.get("conversationId")

    reply, stop_reason = poll_for_reply(conv_id, message_id)

    # Log reply and tool call detection
    log.debug("Raw reply (%d chars): %s", len(reply), reply[:500])
    if req.tools:
        tool_calls = parse_tool_calls(reply)
        if tool_calls:
            for tc in tool_calls:
                log.info("Detected tool call: %s(%s)",
                         tc["name"],
                         ", ".join(f"{k}={str(v)[:50]}" for k, v in tc["input"].items()))
        else:
            log.info("No tool calls detected in reply")

    return reply, sandbox_model, stop_reason


# ---------------------------------------------------------------------------
# Build Anthropic-format responses (with tool_use support)
# ---------------------------------------------------------------------------
def _estimate_input_tokens(req: MessagesRequest) -> int:
    input_text = ""
    if req.system:
        input_text += str(req.system)
    for m in req.messages:
        input_text += str(m.content)
    return estimate_tokens(input_text)


def _build_content_blocks(reply: str, tools: Optional[List[dict]]) -> Tuple[List[dict], str]:
    """Parse reply into Anthropic content blocks. Returns (content, stop_reason)."""
    if not tools:
        return [{"type": "text", "text": reply}], "end_turn"

    tool_calls = parse_tool_calls(reply)
    if not tool_calls:
        return [{"type": "text", "text": reply}], "end_turn"

    # Validate: allowlist + schema check
    tool_calls = validate_tool_calls(tool_calls, tools)
    if not tool_calls:
        # All calls rejected — return as plain text so model sees what it tried
        return [{"type": "text", "text": reply}], "end_turn"

    # Build content: text (if any) + tool_use blocks
    content = []
    clean_text = strip_tool_calls(reply)
    if clean_text:
        content.append({"type": "text", "text": clean_text})
    content.extend(tool_calls)

    return content, "tool_use"


def build_message_response(reply: str, model: str, stop_reason: str, req: MessagesRequest) -> dict:
    """Build an Anthropic Messages API response."""
    content, actual_stop = _build_content_blocks(reply, req.tools)

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": actual_stop,
        "stop_sequence": None,
        "usage": {
            "input_tokens": _estimate_input_tokens(req),
            "output_tokens": estimate_tokens(reply),
        },
    }


def build_streaming_response(reply: str, model: str, stop_reason: str, req: MessagesRequest):
    """Yield SSE events in Anthropic streaming format."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    content_blocks, actual_stop = _build_content_blocks(reply, req.tools)

    input_tokens = _estimate_input_tokens(req)
    output_tokens = estimate_tokens(reply)

    # message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}})}\n\n"

    # Emit each content block
    for idx, block in enumerate(content_blocks):
        if block["type"] == "text":
            # content_block_start (text)
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

            # Stream text in chunks
            text = block["text"]
            chunk_size = 20
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'text_delta', 'text': chunk}})}\n\n"

            # content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"

        elif block["type"] == "tool_use":
            # content_block_start (tool_use)
            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': {'type': 'tool_use', 'id': block['id'], 'name': block['name'], 'input': {}}})}\n\n"

            # Stream the input JSON as a single delta
            input_json = json.dumps(block["input"])
            yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': input_json}})}\n\n"

            # content_block_stop
            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"

    # message_delta
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': actual_stop, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"

    # message_stop
    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.post("/v1/messages")
def messages(req: MessagesRequest):
    try:
        reply, model_used, stop_reason = call_sandbox(req)
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=e.response.status_code if e.response else 502,
            detail=f"Sandbox API error: {e}",
        )

    if req.stream:
        return StreamingResponse(
            build_streaming_response(reply, model_used, stop_reason, req),
            media_type="text/event-stream",
        )

    return build_message_response(reply, model_used, stop_reason, req)


@app.post("/v1/messages/count_tokens")
def count_tokens(request: Request):
    """Stub for token counting — returns estimate."""
    return {"input_tokens": 0}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {
        "message": "Claude Code Gateway — Anthropic Messages API → LLM Sandbox",
        "docs": "/docs",
        "usage": "export ANTHROPIC_BASE_URL=http://localhost:8781",
    }
