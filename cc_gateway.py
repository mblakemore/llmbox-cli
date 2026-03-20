"""
Claude Code Gateway — Anthropic Messages API → LLM Sandbox Proxy

Exposes an Anthropic-compatible /v1/messages endpoint that translates
requests to the UCSB LLM Sandbox Bot API format, enabling Claude Code
to use Sandbox-hosted models.

Usage:
    BEDROCK_API_URL=https://... BEDROCK_API_KEY=... uvicorn cc_gateway:app --port 8781

Then configure Claude Code:
    export ANTHROPIC_BASE_URL=http://localhost:8781
    export ANTHROPIC_API_KEY=dummy
"""

from __future__ import annotations

import json
import os
import time
import uuid
import logging
from typing import List, Optional, Tuple, Union

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("cc-gateway")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_URL = os.environ.get("BEDROCK_API_URL")
API_KEY = os.environ.get("BEDROCK_API_KEY")
POLL_TIMEOUT = int(os.environ.get("POLL_TIMEOUT", "180"))
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
    "claude-opus-4-6": "claude-v4.6-opus",
    "claude-opus-4-5": "claude-v4.5-opus",
    "claude-sonnet-4-5": "claude-v4.5-sonnet",
    "claude-sonnet-4-5-20250514": "claude-v4.5-sonnet",
    "claude-haiku-4-5": "claude-v4.5-haiku",
    "claude-haiku-4-5-20251001": "claude-v4.5-haiku",
    "claude-opus-4-6-20250612": "claude-v4.6-opus",
    "claude-sonnet-4-6": "claude-v4.6-sonnet",
    "claude-sonnet-4-6-20250514": "claude-v4.6-sonnet",
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
class ContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    # image support
    source: Optional[dict] = None


class AnthropicMessage(BaseModel):
    role: str
    content: Union[str, List[ContentBlock], List[dict]]


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


# ---------------------------------------------------------------------------
# Translate Anthropic messages → Sandbox content blocks
# ---------------------------------------------------------------------------
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(len(text) // CHARS_PER_TOKEN, 1)


def assemble_content(req: MessagesRequest) -> List[dict]:
    """Flatten Anthropic messages into Sandbox content blocks with role prefixes."""
    content_blocks = []

    # System prompt
    if req.system:
        if isinstance(req.system, str):
            content_blocks.append({
                "contentType": "text",
                "body": f"System instructions: {req.system}",
            })
        elif isinstance(req.system, list):
            for block in req.system:
                if isinstance(block, dict) and block.get("type") == "text":
                    content_blocks.append({
                        "contentType": "text",
                        "body": f"System instructions: {block.get('text', '')}",
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
                if isinstance(block, dict):
                    block_type = block.get("type", "")
                else:
                    block_type = block.type

                if block_type == "text":
                    text = block.get("text", "") if isinstance(block, dict) else (block.text or "")
                    content_blocks.append({
                        "contentType": "text",
                        "body": f"{role_prefix}: {text}",
                    })
                elif block_type == "image":
                    source = block.get("source", {}) if isinstance(block, dict) else (block.source or {})
                    if source.get("type") == "base64":
                        content_blocks.append({
                            "contentType": "image",
                            "mediaType": source.get("media_type", "image/png"),
                            "body": source.get("data", ""),
                        })

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

    log.info("Sending to Sandbox: model=%s, %d content blocks", sandbox_model, len(content_blocks))

    payload = {
        "message": {
            "role": "user",
            "parent_message_id": None,
            "content": content_blocks,
            "model": sandbox_model,
        },
    }

    post_resp = requests.post(
        f"{API_URL}/conversation", headers=BOT_HEADERS, json=payload
    )
    post_resp.raise_for_status()
    resp_data = post_resp.json()
    message_id = resp_data.get("messageId")
    conv_id = resp_data.get("conversationId")

    reply, stop_reason = poll_for_reply(conv_id, message_id)
    return reply, sandbox_model, stop_reason


# ---------------------------------------------------------------------------
# Build Anthropic-format responses
# ---------------------------------------------------------------------------
def build_message_response(reply: str, model: str, stop_reason: str, req: MessagesRequest) -> dict:
    """Build an Anthropic Messages API response."""
    input_text = ""
    if req.system:
        input_text += str(req.system)
    for m in req.messages:
        input_text += str(m.content)

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": reply}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": estimate_tokens(input_text),
            "output_tokens": estimate_tokens(reply),
        },
    }


def build_streaming_response(reply: str, model: str, stop_reason: str, req: MessagesRequest):
    """Yield SSE events in Anthropic streaming format."""
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    input_text = ""
    if req.system:
        input_text += str(req.system)
    for m in req.messages:
        input_text += str(m.content)
    input_tokens = estimate_tokens(input_text)
    output_tokens = estimate_tokens(reply)

    # message_start
    yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 0}}})}\n\n"

    # content_block_start
    yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

    # Stream the reply in chunks for a more natural feel
    chunk_size = 20  # characters per chunk
    for i in range(0, len(reply), chunk_size):
        chunk = reply[i:i + chunk_size]
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': chunk}})}\n\n"

    # content_block_stop
    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

    # message_delta
    yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"

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
