"""Think tool — opt-in deep reasoning via a separate API call with reasoning enabled."""

import json
import logging
import os
import sys

BLUE = "\033[34m"
RESET = "\033[0m"


def _get_api():
    """Get a BedrockChatAPI instance."""
    # Import here to avoid circular imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from bedrock_api import BedrockChatAPI

    config = {}
    config_path = os.path.join(os.getcwd(), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            config = cfg.get("llm", {})
        except Exception:
            pass
    return BedrockChatAPI(config)


def fn(prompt: str, depth: str = "brief", context: str = "") -> str:
    """Make a standalone reasoning call with thinking enabled.

    Args:
        prompt: The problem or question to reason through.
        depth: Reasoning depth — "brief", "normal", or "deep".
        context: Optional conversation context to include.
    """
    log = logging.getLogger("agent")
    api = _get_api()

    full_prompt = ""
    if context:
        full_prompt += f"Context:\n{context}\n\n"
    full_prompt += prompt

    log.info("THINK [depth=%s, context=%d chars]: %s", depth, len(context), prompt)

    print(f"  {BLUE}[Thinking ({depth})]", end="", flush=True)

    try:
        msg = api.send_and_wait(full_prompt, enable_reasoning=True)
    except Exception as e:
        print(f" error{RESET}")
        return f"Error calling API: {e}"

    reasoning = api.extract_reasoning(msg)
    answer = api.extract_text(msg)

    if reasoning:
        print(f"\n{reasoning}")
    if answer:
        print(f"\n  [Answer] {answer}{RESET}")
    else:
        print(f" done{RESET}")

    log.info("THINK REASONING: %s", reasoning or "(none)")
    log.info("THINK ANSWER: %s", answer or "(empty)")

    return answer if answer else "Error: empty response from model"


definition = {
    "type": "function",
    "function": {
        "name": "think",
        "description": (
            "Invoke a separate reasoning call with chain-of-thought enabled. "
            "Reasoning and answer are returned. "
            "Only the final conclusion is returned to the conversation. "
            "Depth: 'brief' for simple decisions; "
            "'normal' for multi-step problems; "
            "'deep' for complex analysis. "
            "Pass 'context' when the prompt references prior discussion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The question or problem to reason through.",
                },
                "depth": {
                    "type": "string",
                    "enum": ["brief", "normal", "deep"],
                    "description": "'brief' for quick tasks, 'normal' for moderate problems, 'deep' for complex analysis.",
                },
                "context": {
                    "type": "string",
                    "description": "Relevant background info the thinker needs.",
                },
            },
            "required": ["prompt", "depth"],
        },
    },
}
