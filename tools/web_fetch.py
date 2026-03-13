"""Web fetch tool — retrieve and convert web pages to readable text.

Content is saved to a file and a short summary is returned to keep the
conversation context small.  The agent can then read the file with the
file tool if it needs the full content.
"""

import hashlib
import os
import requests
from markdownify import markdownify


_MAX_CHARS = 50000
_TIMEOUT = 30
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; tool-agent/1.0)"
}
# Max chars to include inline in the tool result (keeps context small)
_INLINE_PREVIEW = 2000


def fn(url: str) -> str:
    """Fetch a URL and save its content to a file.

    Returns a short preview + the file path.  Use the file tool to read
    the full content.

    Args:
        url: The URL to fetch.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"Error fetching URL: {e}"

    content_type = resp.headers.get("content-type", "")

    # Plain text or JSON — use as-is
    if "text/plain" in content_type or "application/json" in content_type:
        text = resp.text[:_MAX_CHARS]
    else:
        # HTML — convert to markdown
        try:
            md = markdownify(resp.text, strip=["img", "script", "style", "nav", "footer", "header"])
            # Clean up excessive blank lines
            lines = md.splitlines()
            cleaned = []
            blank_count = 0
            for line in lines:
                if not line.strip():
                    blank_count += 1
                    if blank_count <= 2:
                        cleaned.append("")
                else:
                    blank_count = 0
                    cleaned.append(line)
            text = "\n".join(cleaned).strip()
        except Exception:
            text = resp.text

        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS]

    # Save to file so the full content survives context compression
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    save_dir = os.path.join(os.getcwd(), "state", "fetched")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{url_hash}.md")
    with open(save_path, "w") as f:
        f.write(f"# Fetched: {url}\n\n{text}")

    total_chars = len(text)
    total_lines = text.count("\n") + 1
    preview = text[:_INLINE_PREVIEW]
    if len(text) > _INLINE_PREVIEW:
        preview += f"\n\n[... truncated — {total_chars} chars total, {total_lines} lines]"

    return (
        f"[Fetched: {url} — saved to {save_path} ({total_chars} chars, {total_lines} lines)]\n"
        f"[Use file tool to read full content if needed]\n\n"
        f"{preview}"
    )


definition = {
    "type": "function",
    "function": {
        "name": "web_fetch",
        "description": (
            "Fetch a web page, save it to state/fetched/<hash>.md, and return "
            "a short preview. The full content is in the saved file — use the "
            "file tool to read it. Do NOT re-fetch a URL that was already fetched; "
            "read the saved file instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch.",
                },
            },
            "required": ["url"],
        },
    },
}
