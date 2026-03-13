"""
llmbox_lib — library interface for llmbox.

Wraps the agent loop into a class that can be used programmatically
without terminal I/O. All output goes through callbacks instead of print().

Usage:
    from llmbox_lib import Agent

    agent = Agent()
    result = agent.run("list the files in this directory")
    print(result.text)
    print(result.tool_calls)
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from bedrock_api import BedrockChatAPI
from tools import MAP_FN, tools, load_extra_tools

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_TAG_RE = re.compile(r'</?think>')

_UNICODE_MAP = str.maketrans({
    "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "*",
    "\u00a0": " ", "\u200b": "",
})


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """Result of a single tool execution."""
    name: str
    args: dict
    output: str
    is_error: bool = False


@dataclass
class TurnResult:
    """Result of a single agent turn (one LLM call + tool executions)."""
    text: str
    tool_results: list[ToolResult] = field(default_factory=list)
    reasoning: str | None = None


@dataclass
class RunResult:
    """Result of a complete agent run (all turns until done)."""
    text: str
    turns: list[TurnResult] = field(default_factory=list)
    total_turns: int = 0
    status: str = "done"  # "done", "max_turns", "error"


# ── Agent class ───────────────────────────────────────────────────────

class Agent:
    """Programmatic agent that runs the tool-use loop without terminal I/O.

    Args:
        config: Config dict with optional "llm", "context", "cycle" sections.
            Falls back to environment variables and defaults.
        system_prompt: Additional system instructions prepended to the tool prompt.
        tools_dir: Path to a directory of extra tool modules to load.
        on_turn: Callback called after each turn with (turn_number, TurnResult).
        on_tool: Callback called before each tool execution with (tool_name, tool_args).
        log: Logger instance. If None, a quiet logger is created.
    """

    _DEFAULT_CONFIG = {
        "llm": {
            "api_url": os.environ.get("BEDROCK_API_URL", ""),
            "api_key": os.environ.get("BEDROCK_API_KEY", ""),
            "origin": "http://localhost:8000",
            "model": "claude-v4.5-sonnet",
            "poll_interval": 2,
            "poll_timeout": 180,
        },
        "context": {
            "max_context_chars": 80000,
            "summary_threshold": 5,
        },
        "cycle": {
            "max_turns": 100,
            "wind_down_turns": 10,
        },
    }

    def __init__(self, config=None, system_prompt=None, tools_dir=None,
                 on_turn=None, on_tool=None, log=None):
        cfg = json.loads(json.dumps(self._DEFAULT_CONFIG))
        if config:
            for section, values in config.items():
                if section in cfg and isinstance(cfg[section], dict):
                    cfg[section].update(values)
                else:
                    cfg[section] = values

        self.api = BedrockChatAPI(cfg["llm"])
        self.max_context_chars = cfg["context"]["max_context_chars"]
        self.summary_threshold = cfg["context"]["summary_threshold"]
        self.max_turns = cfg["cycle"]["max_turns"]
        self.wind_down_turns = cfg["cycle"]["wind_down_turns"]
        self.extra_system_prompt = system_prompt or ""
        self.on_turn = on_turn
        self.on_tool = on_tool
        self.log = log or logging.getLogger("llmbox_lib")

        self.conversation_history = []
        self.summary_state = {"text": "", "up_to": 0}

        if tools_dir and os.path.isdir(tools_dir):
            load_extra_tools(tools_dir)

    @property
    def model(self):
        return self.api.model

    @model.setter
    def model(self, value):
        self.api.model = value

    def health(self) -> bool:
        """Check if the API gateway is reachable."""
        return self.api.health()

    def list_models(self) -> list[str]:
        """List available models from the gateway."""
        return self.api.list_models()

    def reset(self):
        """Clear conversation history and summary."""
        self.conversation_history.clear()
        self.summary_state = {"text": "", "up_to": 0}

    def run(self, prompt: str, max_turns: int | None = None) -> RunResult:
        """Run the agent loop until completion.

        Args:
            prompt: The user prompt to send.
            max_turns: Override max turns for this run.

        Returns:
            RunResult with the final text and all turn details.
        """
        self.conversation_history.append({"role": "user", "content": prompt})
        self.log.info("USER: %s", prompt[:200])

        limit = max_turns or self.max_turns
        turns = []
        final_text = ""

        for turn_num in range(1, limit + 1):
            turn_result = self._run_turn(turn_num, limit)
            turns.append(turn_result)
            final_text = turn_result.text

            if self.on_turn:
                self.on_turn(turn_num, turn_result)

            if not turn_result.tool_results:
                return RunResult(
                    text=final_text, turns=turns,
                    total_turns=turn_num, status="done",
                )

        return RunResult(
            text=final_text, turns=turns,
            total_turns=len(turns), status="max_turns",
        )

    def _run_turn(self, turn_num, max_turns) -> TurnResult:
        """Execute a single turn: build prompt, call LLM, execute tools."""
        # Build prompt
        prompt_str, oldest_idx = self._build_prompt()

        # Resummarize if needed
        unsummarized = oldest_idx - self.summary_state["up_to"]
        if unsummarized >= self.summary_threshold:
            new_msgs = self.conversation_history[self.summary_state["up_to"]:oldest_idx]
            self._generate_summary(new_msgs)
            prompt_str, oldest_idx = self._build_prompt()

        # Wind-down injection
        remaining = max_turns - turn_num
        if 0 < remaining <= self.wind_down_turns:
            prompt_str = self._inject_wind_down(
                prompt_str, f"[SYSTEM: {remaining} turns remaining. Begin wrapping up.]")
        elif remaining <= 0:
            prompt_str = self._inject_wind_down(
                prompt_str, f"[SYSTEM: You are past the turn limit. Finish immediately.]")

        self.log.info("Turn %d/%d | prompt %d chars", turn_num, max_turns, len(prompt_str))

        # Call LLM
        try:
            msg = self.api.send_and_wait(prompt_str)
        except Exception as e:
            self.log.error("API call failed: %s", e)
            return TurnResult(text=f"Error: {e}")

        full_content = self.api.extract_text(msg)
        reasoning = self.api.extract_reasoning(msg)

        # Parse tool calls
        tool_calls = self._parse_tool_calls(full_content)
        text_part = self._strip_tool_calls(full_content)
        text_part = self._sanitize(text_part)
        if text_part.startswith("Assistant:"):
            text_part = text_part[len("Assistant:"):].strip()

        # Record assistant message
        assistant_msg = {"role": "assistant", "content": full_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        self.conversation_history.append(assistant_msg)

        # Execute tools
        tool_results = []
        for tc in tool_calls:
            if self.on_tool:
                self.on_tool(tc["name"], tc["args"])

            if tc["name"] not in MAP_FN:
                output = f"Error: Unknown tool '{tc['name']}'"
                is_error = True
            else:
                try:
                    output = str(MAP_FN[tc["name"]](**tc["args"]))
                    is_error = output.startswith("Error")
                except Exception as e:
                    output = f"Error executing tool: {e}"
                    is_error = True

            tool_results.append(ToolResult(
                name=tc["name"], args=tc["args"],
                output=output, is_error=is_error,
            ))

            self.conversation_history.append({
                "role": "tool", "name": tc["name"], "content": output,
            })
            self.log.info("TOOL %s: %s", tc["name"], output[:200])

        return TurnResult(text=text_part, tool_results=tool_results, reasoning=reasoning)

    # ── Internal helpers ──────────────────────────────────────────────

    def _build_tool_system_prompt(self):
        tool_descriptions = []
        for tool_def in tools:
            fn_def = tool_def["function"]
            name = fn_def["name"]
            desc = fn_def.get("description", "")
            params = fn_def.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])

            param_lines = []
            for pname, pinfo in props.items():
                req = " (required)" if pname in required else ""
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                if pinfo.get("enum"):
                    pdesc += f" One of: {pinfo['enum']}"
                param_lines.append(f"    - {pname} ({ptype}{req}): {pdesc}")

            params_str = "\n".join(param_lines) if param_lines else "    (no parameters)"
            tool_descriptions.append(f"  {name}: {desc}\n  Parameters:\n{params_str}")

        tools_block = "\n\n".join(tool_descriptions)
        base = f"You are an autonomous agent with access to tools.\n\nAVAILABLE TOOLS:\n{tools_block}\n\n"
        base += ("TO USE A TOOL, include a tool call block in your response:\n\n"
                 "<tool_call>\n"
                 "{\"tool\": \"tool_name\", \"args\": {\"param1\": \"value1\"}}\n"
                 "</tool_call>\n\n"
                 "RULES:\n"
                 "- You may use multiple tool calls in a single response.\n"
                 "- When done, respond with plain text (no tool_call block).\n"
                 "- Read files before overwriting them.\n")
        if self.extra_system_prompt:
            base += f"\n{self.extra_system_prompt}\n"
        return base

    def _build_prompt(self):
        max_chars = self.max_context_chars
        parts = [f"[System]\n{self._build_tool_system_prompt()}\n[End System]\n"]

        if self.summary_state["text"]:
            parts.append(f"Progress summary:\n{self.summary_state['text']}")

        overhead = sum(len(p) for p in parts) + 500
        budget = max_chars - overhead
        selected = []
        oldest_idx = len(self.conversation_history)

        for i in range(len(self.conversation_history) - 1, -1, -1):
            msg = self.conversation_history[i]
            msg_chars = len(msg.get("content", "") or "")
            if msg.get("tool_calls"):
                msg_chars += len(json.dumps(msg["tool_calls"]))
            if sum(len(m.get("content", "") or "") for m in selected) + msg_chars > budget:
                break
            selected.append(msg)
            oldest_idx = i

        selected.reverse()

        for msg in selected:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        parts.append(f"[Tool call: {tc['name']}({json.dumps(tc.get('args', {}))})]")
            elif role == "tool":
                name = msg.get("name", "?")
                parts.append(f"[Tool result ({name}): {content}]")

        prompt = "\n\n".join(parts) + "\n\nAssistant:"
        return prompt, oldest_idx

    def _generate_summary(self, messages):
        transcript_parts = []
        for m in messages:
            role = m["role"].upper()
            content = (m.get("content", "") or "")[:600]
            if role == "TOOL":
                name = m.get("name", "?")
                transcript_parts.append(f"TOOL RESULT ({name}): {content}")
            else:
                transcript_parts.append(f"{role}: {content}")
        transcript = "\n".join(transcript_parts)

        old = self.summary_state["text"]
        if old:
            prompt = (f"Previous summary:\n{old}\n\nNew messages:\n{transcript}\n\n"
                      f"Write an updated summary under 500 words.")
        else:
            prompt = f"Conversation transcript:\n{transcript}\n\nWrite a concise summary under 500 words."

        try:
            msg = self.api.send_and_wait(prompt)
            self.summary_state["text"] = self.api.extract_text(msg).strip()
        except Exception as e:
            self.log.error("Summary generation failed: %s", e)

        self.summary_state["up_to"] = len(self.conversation_history) - 1

    def _inject_wind_down(self, prompt, message):
        if prompt.endswith("Assistant:"):
            prompt = prompt[:-len("Assistant:")]
        return prompt + f"\n\n{message}\n\nAssistant:"

    @staticmethod
    def _parse_tool_calls(text):
        calls = []
        for match in _TOOL_CALL_RE.finditer(text):
            try:
                data = json.loads(match.group(1))
                name = data.get("tool") or data.get("name")
                args = data.get("args") or data.get("arguments") or {}
                if not args and name:
                    args = {k: v for k, v in data.items() if k not in ("tool", "name")}
                if name:
                    calls.append({"name": name, "args": args})
            except json.JSONDecodeError:
                continue
        return calls

    @staticmethod
    def _strip_tool_calls(text):
        return _TOOL_CALL_RE.sub("", text).strip()

    @staticmethod
    def _sanitize(text):
        text = _THINK_TAG_RE.sub('', text)
        return text.translate(_UNICODE_MAP)
