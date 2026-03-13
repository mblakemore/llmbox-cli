"""Bedrock Chat Published API client.

Replaces the local llama-server OpenAI-compatible API with the
AWS Bedrock Chat gateway (aws-samples/bedrock-chat).
"""

import json
import logging
import os
import time

import requests

_DEFAULT_CONFIG = {
    "api_url": os.environ.get("BEDROCK_API_URL", ""),
    "api_key": os.environ.get("BEDROCK_API_KEY", ""),
    "origin": "http://localhost:8000",
    "model": "claude-v4.5-sonnet",
    "poll_interval": 2,
    "poll_timeout": 180,
}


class BedrockChatAPI:
    """Client for the Bedrock Chat Published API."""

    def __init__(self, config: dict | None = None):
        cfg = {**_DEFAULT_CONFIG, **(config or {})}
        self.api_url = cfg["api_url"]
        self.model = cfg["model"]
        self.poll_interval = cfg["poll_interval"]
        self.poll_timeout = cfg["poll_timeout"]
        self.session = requests.Session()
        self.session.headers.update({
            "x-api-key": cfg["api_key"],
            "Content-Type": "application/json",
            "Origin": cfg["origin"],
        })
        self.log = logging.getLogger("agent")

    def health(self) -> bool:
        try:
            resp = self.session.get(f"{self.api_url}/health", timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def send(self, prompt: str, enable_reasoning: bool = False) -> tuple[str, str]:
        """Send a message. Returns (conversation_id, message_id)."""
        payload = {
            "message": {
                "content": [{"contentType": "text", "body": prompt}],
                "model": self.model,
            },
        }
        if enable_reasoning:
            payload["enableReasoning"] = True

        resp = self.session.post(f"{self.api_url}/conversation", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["conversationId"], data["messageId"]

    def poll(self, conversation_id: str, cancel_check=None) -> dict:
        """Poll until assistant response is ready. Returns the assistant message dict."""
        deadline = time.time() + self.poll_timeout
        while time.time() < deadline:
            time.sleep(self.poll_interval)
            if cancel_check:
                cancel_check()
            resp = self.session.get(
                f"{self.api_url}/conversation/{conversation_id}", timeout=30)
            resp.raise_for_status()
            conv = resp.json()
            last_id = conv.get("lastMessageId")
            if last_id:
                last_msg = conv.get("messageMap", {}).get(last_id, {})
                if last_msg.get("role") == "assistant":
                    return last_msg
        raise TimeoutError(f"No response after {self.poll_timeout}s")

    def send_and_wait(self, prompt: str, enable_reasoning: bool = False,
                      cancel_check=None) -> dict:
        """Send a message and wait for the response. Returns the full assistant message."""
        conv_id, _ = self.send(prompt, enable_reasoning)
        return self.poll(conv_id, cancel_check=cancel_check)

    def extract_text(self, msg: dict) -> str:
        """Extract text content from an assistant message."""
        parts = []
        for content in msg.get("content", []):
            if content.get("contentType") == "text":
                parts.append(content.get("body", ""))
        return "\n".join(parts)

    def extract_reasoning(self, msg: dict) -> str | None:
        """Extract reasoning/thinking content if present."""
        for content in msg.get("content", []):
            if content.get("contentType") == "reasoning":
                return content.get("text", "")
        log = msg.get("thinkingLog")
        if log:
            return str(log)
        return None

    def list_models(self) -> list[str]:
        """Fetch available models from the OpenAPI spec."""
        try:
            resp = self.session.get(f"{self.api_url}/openapi.json", timeout=10)
            resp.raise_for_status()
            spec = resp.json()
            return spec["components"]["schemas"]["MessageInputWithoutMessageId"]["properties"]["model"]["enum"]
        except Exception as e:
            self.log.warning("Failed to fetch model list: %s", e)
            return []
