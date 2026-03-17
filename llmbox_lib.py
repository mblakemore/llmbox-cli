"""
llmbox_lib — core agent implementation.

The Agent class is the single implementation of the agent loop (both dev and
long modes). It can be used programmatically via callbacks, or driven by the
CLI wrapper in llmbox.py.

Usage:
    from llmbox_lib import Agent

    agent = Agent()
    result = agent.run("list the files in this directory")
    print(result.text)
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field

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
    status: str = "done"  # "done", "max_turns", "error", "cancelled"


# ── Callbacks ────────────────────────────────────────────────────────

class NullCallbacks:
    """Default no-op callbacks. Subclass and override for UI integration."""

    def check_cancelled(self):
        """Raise an exception to abort. Called during API polling and tool execution."""
        pass

    def on_api_start(self, label: str):
        """Called before an API call (e.g. start spinner)."""
        pass

    def on_api_response(self):
        """Called when API response arrives (e.g. stop spinner)."""
        pass

    def on_api_done(self):
        """Called after response processing is complete."""
        pass

    def on_assistant_text(self, text: str, reasoning: str | None):
        """Called with the assistant's text output."""
        pass

    def on_tool_batch_start(self, count: int):
        """Called before executing a batch of tool calls."""
        pass

    def on_tool_start(self, name: str, args: dict):
        """Called before a single tool executes."""
        pass

    def on_tool_result(self, name: str, args: dict, result: str, is_error: bool):
        """Called after a tool executes."""
        pass

    def on_turn_end(self, turn: int, turn_result: TurnResult):
        """Called after each turn completes (e.g. save checkpoint)."""
        pass

    def on_summary_start(self, count: int):
        """Called when summarization begins."""
        pass

    def on_summary_done(self):
        """Called when summarization completes."""
        pass

    def on_forced_think(self, tool_name: str, count: int):
        """Called when forced-think is triggered for repeated errors."""
        pass

    def on_truncation_recovered(self, attempts: int):
        """Called when truncation is recovered."""
        pass

    def on_truncation_failed(self, attempts: int):
        """Called when truncation recovery fails."""
        pass

    def on_context_recovery(self, auto: bool) -> str:
        """Called when long mode hits context limit. Return "dev" or "long"."""
        return "dev"


# ── Agent class ───────────────────────────────────────────────────────

class Agent:
    """Core agent that runs the tool-use loop.

    Args:
        config: Config dict with optional "llm", "context", "cycle" sections.
        system_prompt: Additional system instructions prepended to the tool prompt.
        tools_dir: Path to a directory of extra tool modules to load.
        callbacks: Callbacks instance for UI integration. Defaults to NullCallbacks.
        log: Logger instance. If None, a quiet logger is created.
        mode: Conversation mode — "dev" (prompt stuffing) or "long" (server-side caching).

    Backward compatibility: on_turn and on_tool keyword args are still accepted
    and wrapped into callbacks if no callbacks instance is provided.
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
            "max_full_lines": 400,
            "preview_lines": 100,
            "max_context_chars": 80000,
            "summary_threshold": 5,
        },
        "cycle": {
            "max_turns": 100,
            "wind_down_turns": 10,
        },
    }

    _MODEL_CONTEXT_CHARS = {
        "claude-v4.5-opus": 700000,
        "claude-v4.5-sonnet": 700000,
        "claude-v4.5-haiku": 700000,
        "claude-v3.7-sonnet": 700000,
        "llama": 450000,
        "mistral": 112000,
        "deepseek": 224000,
        "amazon-nova": 450000,
        "qwen": 112000,
    }

    def __init__(self, config=None, system_prompt=None, tools_dir=None,
                 callbacks=None, log=None, mode="dev",
                 on_turn=None, on_tool=None):
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
        self.log = log or logging.getLogger("llmbox_lib")

        # Callbacks
        if callbacks:
            self.cb = callbacks
        elif on_turn or on_tool:
            # Backward compat: wrap simple callbacks
            self.cb = _LegacyCallbacks(on_turn, on_tool)
        else:
            self.cb = NullCallbacks()

        # Mode state
        self.mode = mode
        self.conversation_id = None
        self.approx_char_usage = 0

        # Conversation state
        self.conversation_history = []
        self.summary_state = {"text": "", "up_to": 0}
        self.initial_files = None

        # Tool error tracking for loop detection
        self._recent_tool_errors = []

        if tools_dir and os.path.isdir(tools_dir):
            load_extra_tools(tools_dir)

    @property
    def model(self):
        return self.api.model

    @model.setter
    def model(self, value):
        self.api.model = value

    def health(self) -> bool:
        return self.api.health()

    def list_models(self) -> list[str]:
        return self.api.list_models()

    def reset(self):
        """Clear conversation history and summary."""
        self.conversation_history.clear()
        self.summary_state = {"text": "", "up_to": 0}
        self.conversation_id = None
        self.approx_char_usage = 0
        self._recent_tool_errors.clear()

    def switch_mode(self, new_mode: str):
        """Switch conversation mode with summary carry-over."""
        if new_mode == self.mode:
            return
        if new_mode not in ("dev", "long"):
            raise ValueError(f"Unknown mode: {new_mode}")

        if self.conversation_history:
            recent = self.conversation_history[-20:]
            self._generate_summary(recent)

        self.mode = new_mode
        self.conversation_id = None
        self.approx_char_usage = 0

    def _get_context_limit_chars(self):
        model = self.api.model
        for prefix, chars in self._MODEL_CONTEXT_CHARS.items():
            if model.startswith(prefix) or prefix in model:
                return int(chars * 0.8)
        return int(450000 * 0.8)

    # ── Main run loop ────────────────────────────────────────────────

    def run(self, prompt: str, max_turns: int | None = None,
            auto: bool = False, start_turn: int = 0) -> RunResult:
        """Run the agent loop until completion.

        Args:
            prompt: The user prompt to send.
            max_turns: Override max turns for this run.
            auto: If True, context recovery auto-switches to dev mode.
            start_turn: Starting turn number (for checkpoint resume).
        """
        self.conversation_history.append({"role": "user", "content": prompt})
        self.log.info("USER: %s", prompt[:200])
        return self._run_loop(max_turns=max_turns, auto=auto, start_turn=start_turn)

    def run_continue(self, max_turns: int | None = None,
                     auto: bool = False, start_turn: int = 0) -> RunResult:
        """Continue the agent loop without adding a new user message.

        Used when conversation_history already has the latest user message
        (e.g. after checkpoint restore).
        """
        return self._run_loop(max_turns=max_turns, auto=auto, start_turn=start_turn)

    def _run_loop(self, max_turns=None, auto=False, start_turn=0) -> RunResult:
        """Core agent loop shared by run() and run_continue()."""
        limit = max_turns or self.max_turns
        turns = []
        final_text = ""
        turn = start_turn

        while True:
            turn += 1

            try:
                if self.mode == "long":
                    turn_result = self._run_turn_long(turn, limit, auto=auto)
                else:
                    turn_result = self._run_turn_dev(turn, limit)
            except _CancelledSignal:
                self.cb.on_turn_end(turn, TurnResult(text=""))
                return RunResult(
                    text=final_text, turns=turns,
                    total_turns=turn, status="cancelled",
                )

            turns.append(turn_result)
            final_text = turn_result.text

            self.cb.on_turn_end(turn, turn_result)

            if not turn_result.tool_results:
                return RunResult(
                    text=final_text, turns=turns,
                    total_turns=turn, status="done",
                )

            if turn >= limit:
                # Allow overtime but still track it
                pass

    # ── Dev mode turn ────────────────────────────────────────────────

    def _run_turn_dev(self, turn_num, max_turns) -> TurnResult:
        """Single turn in dev mode (prompt stuffing)."""
        prompt_str, oldest_idx = self._build_prompt()

        # Resummarize if needed
        unsummarized = oldest_idx - self.summary_state["up_to"]
        if unsummarized >= self.summary_threshold:
            new_msgs = self.conversation_history[self.summary_state["up_to"]:oldest_idx]
            self.cb.on_summary_start(len(new_msgs))
            self._generate_summary(new_msgs)
            self.cb.on_summary_done()
            prompt_str, oldest_idx = self._build_prompt()

        # Wind-down
        wind_down = self._get_wind_down(turn_num, max_turns)
        if wind_down:
            prompt_str = self._inject_wind_down(prompt_str, wind_down)

        self.log.info("--- Turn %d/%d | dev | prompt %d chars | %d msgs",
                      turn_num, max_turns, len(prompt_str),
                      len(self.conversation_history))

        # Call LLM
        self.cb.on_api_start("\nAssistant: ")
        try:
            msg = self.api.send_and_wait(prompt_str,
                                         cancel_check=self.cb.check_cancelled)
        except Exception as e:
            self.cb.on_api_done()
            if self._is_cancel(e):
                raise _CancelledSignal()
            self.log.error("API call failed: %s", e)
            return TurnResult(text=f"Error: {e}")

        self.cb.on_api_response()
        self.cb.on_api_done()

        return self._process_response(msg)

    # ── Long mode turn ───────────────────────────────────────────────

    def _run_turn_long(self, turn_num, max_turns, auto=False) -> TurnResult:
        """Single turn in long mode (server-side conversation caching)."""
        is_first_turn = self.conversation_id is None

        wind_down_prefix = ""
        wind_down = self._get_wind_down(turn_num, max_turns)
        if wind_down:
            wind_down_prefix = wind_down + "\n\n"

        # Build message to send
        if is_first_turn:
            parts = [self._build_tool_system_prompt()]
            if self.initial_files:
                parts.append(self.initial_files)
            if self.summary_state.get("text"):
                parts.append(f"Progress summary:\n{self.summary_state['text']}")
            user_msg = self._last_user_message()
            parts.append(wind_down_prefix + user_msg if wind_down_prefix else user_msg)
            send_text = "\n\n".join(parts)
        elif self._pending_tool_results():
            tool_count = self._count_trailing_tool_results()
            send_text = wind_down_prefix + self._format_tool_results(tool_count)
        else:
            send_text = wind_down_prefix + self._last_user_message()

        # Check context limit
        projected = self.approx_char_usage + len(send_text)
        context_limit = self._get_context_limit_chars()
        if projected > context_limit:
            self.log.warning("Projected %d > limit %d — recovery", projected, context_limit)
            new_mode = self._recover_context(auto=auto)
            self.mode = new_mode
            if new_mode == "dev":
                return self._run_turn_dev(turn_num, max_turns)
            else:
                # Retry as first turn of new conversation
                return self._run_turn_long(turn_num, max_turns, auto=auto)

        self.log.info("--- Turn %d/%d | long | conv=%s | ~%dk chars",
                      turn_num, max_turns, self.conversation_id or "(new)",
                      self.approx_char_usage // 1000)

        # Call LLM
        self.cb.on_api_start("\nAssistant: ")
        try:
            msg, conv_id = self.api.send_and_wait_conv(
                send_text, conversation_id=self.conversation_id,
                cancel_check=self.cb.check_cancelled)
            if not self.conversation_id:
                self.conversation_id = conv_id
                self.log.info("Long mode: new conversation %s", conv_id)
        except Exception as e:
            self.cb.on_api_done()
            if self._is_cancel(e):
                raise _CancelledSignal()
            self.log.error("API call failed: %s", e)
            # Server 500 might be context overflow
            if "500" in str(e) or "Server Error" in str(e):
                new_mode = self._recover_context(auto=auto)
                self.mode = new_mode
                if new_mode == "dev":
                    return self._run_turn_dev(turn_num, max_turns)
                else:
                    return self._run_turn_long(turn_num, max_turns, auto=auto)
            return TurnResult(text=f"Error: {e}")

        self.cb.on_api_response()
        self.cb.on_api_done()

        full_content = self.api.extract_text(msg)
        self.approx_char_usage += len(send_text) + len(full_content)

        # Handle truncation
        full_content = self._handle_truncation(full_content)

        reasoning = self.api.extract_reasoning(msg)
        return self._process_response_from_text(full_content, reasoning)

    # ── Response processing (shared) ─────────────────────────────────

    def _process_response(self, msg) -> TurnResult:
        """Process an API response message: extract text, parse tools, execute."""
        full_content = self.api.extract_text(msg)
        reasoning = self.api.extract_reasoning(msg)
        return self._process_response_from_text(full_content, reasoning)

    def _process_response_from_text(self, full_content, reasoning) -> TurnResult:
        """Process response text: parse tools, record history, execute tools."""
        tool_calls = self._parse_tool_calls(full_content)
        text_part = self._strip_tool_calls(full_content)
        text_part = self._sanitize(text_part)
        if text_part.startswith("Assistant:"):
            text_part = text_part[len("Assistant:"):].strip()

        self.cb.on_assistant_text(text_part, reasoning)
        self.log.info("ASSISTANT: %s", full_content[:500])

        # Record assistant message
        assistant_msg = {"role": "assistant", "content": full_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        self.conversation_history.append(assistant_msg)

        if not tool_calls:
            return TurnResult(text=text_part, reasoning=reasoning)

        # Execute tools
        tool_results = self._execute_tools(tool_calls)
        return TurnResult(text=text_part, tool_results=tool_results, reasoning=reasoning)

    # ── Tool execution ───────────────────────────────────────────────

    def _execute_tools(self, tool_calls) -> list[ToolResult]:
        """Execute tool calls with loop detection and callbacks."""
        results = []
        self.cb.on_tool_batch_start(len(tool_calls))

        for tc in tool_calls:
            self.cb.check_cancelled()
            name = tc["name"]
            args = tc["args"]

            self.log.info("TOOL CALL: %s(%s)", name, json.dumps(args))
            self.cb.on_tool_start(name, args)

            if name not in MAP_FN:
                output = f"Error: Unknown tool '{name}'"
                is_error = True
            else:
                try:
                    output = str(MAP_FN[name](**args))
                    is_error = output.startswith("Error")
                except Exception as e:
                    output = f"Error executing tool: {e}"
                    is_error = True

            self.cb.on_tool_result(name, args, output, is_error)

            results.append(ToolResult(name=name, args=args, output=output, is_error=is_error))
            self.conversation_history.append({
                "role": "tool", "name": name, "content": output,
            })
            self.log.info("TOOL RESULT [%s]: %s", name, output[:500])

            # Loop detection
            if is_error:
                error_sig = (name, output[:100])
                self._recent_tool_errors.append(error_sig)
                consecutive = sum(1 for e in self._recent_tool_errors if e == error_sig)
                if consecutive >= 3:
                    self._forced_think(name, args, output, consecutive)
            else:
                self._recent_tool_errors[:] = [
                    e for e in self._recent_tool_errors if e[0] != name
                ]

        return results

    def _forced_think(self, func_name, func_args, result_str, consecutive):
        """Force a think call when the same tool error repeats."""
        think_prompt = (
            f"MANDATORY REFLECTION: I have called {func_name} "
            f"{consecutive} times and gotten the same error each time.\n\n"
            f"The error is: {result_str[:300]}\n\n"
            f"My last arguments were: {json.dumps(func_args)}\n\n"
            f"I MUST answer these questions:\n"
            f"1. What exactly is the error telling me?\n"
            f"2. What parameter am I missing or getting wrong?\n"
            f"3. What is a DIFFERENT way to accomplish my goal?\n"
            f"4. Should I just skip this step and move on?"
        )
        self.log.warning("Loop detected: %s x%d — forcing think", func_name, consecutive)
        self.cb.on_forced_think(func_name, consecutive)
        if "think" in MAP_FN:
            think_result = MAP_FN["think"](prompt=think_prompt)
            self.conversation_history.append({
                "role": "assistant",
                "content": f"[Forced reflection]\n{think_result}",
            })
            self.log.info("FORCED THINK RESULT: %s", think_result)

    # ── Summarization ────────────────────────────────────────────────

    def _format_for_summary(self, messages):
        """Format messages into a readable transcript for the summarizer."""
        parts = []
        for m in messages:
            role = m["role"].upper()
            if role == "TOOL":
                name = m.get("name", "?")
                content = m.get("content", "")
                is_error = content.startswith("Error") or "Error:" in content[:50]
                max_len = 800 if is_error else 500
                if len(content) > max_len:
                    content = content[:max_len] + "..."
                parts.append(f"TOOL RESULT ({name}): {content}")
            elif role == "ASSISTANT":
                text = m.get("content", "")
                tool_calls = m.get("tool_calls", [])
                if text:
                    if len(text) > 600:
                        text = text[:600] + "..."
                    parts.append(f"ASSISTANT: {text}")
                for tc in tool_calls:
                    name = tc.get("name", "?")
                    tc_args = json.dumps(tc.get("args", {}))
                    if len(tc_args) > 200:
                        tc_args = tc_args[:200] + "..."
                    parts.append(f"ASSISTANT called {name}({tc_args})")
            else:
                content = m.get("content", "")
                if len(content) > 800:
                    content = content[:800] + "..."
                parts.append(f"{role}: {content}")
        return "\n".join(parts)

    def _generate_summary(self, new_messages):
        """Generate an updated conversation summary via LLM call."""
        transcript = self._format_for_summary(new_messages)

        structure = (
            "Structure the summary with these sections:\n"
            "1. GOAL: The user's current objective\n"
            "2. PROGRESS: What has been accomplished\n"
            "3. DECISIONS & OUTCOMES: Key decisions made and their results "
            "(include approaches that FAILED and why)\n"
            "4. COMPLETED ACTIONS: List actions that are DONE and must NOT be repeated\n"
            "5. CURRENT STATE: Where things stand right now, what files were modified\n"
            "6. NEXT: The single next action to take\n"
            "Keep it under 500 words. Be specific about file paths, error messages, and tool results."
        )

        old = self.summary_state["text"]
        if old:
            prompt = (
                f"Here is the previous summary of the conversation so far:\n\n"
                f"{old}\n\n"
                f"Here are the new messages since that summary:\n\n"
                f"{transcript}\n\n"
                f"Write an updated summary that combines the previous summary with the new messages.\n\n"
                f"{structure}"
            )
        else:
            prompt = (
                f"Here is a conversation transcript:\n\n"
                f"{transcript}\n\n"
                f"Write a concise summary.\n\n"
                f"{structure}"
            )

        self.log.info("Generating conversation summary...")
        try:
            msg = self.api.send_and_wait(prompt)
            self.summary_state["text"] = self.api.extract_text(msg).strip()
            self.log.info("SUMMARY: %s", self.summary_state["text"])
        except Exception as e:
            self.log.error("Summary generation failed: %s", e)

        self.summary_state["up_to"] = max(
            len(self.conversation_history) - 1, self.summary_state["up_to"])

    # ── Truncation handling (long mode) ──────────────────────────────

    def _handle_truncation(self, full_content):
        """Detect and recover from truncated tool call responses."""
        MAX_CONTINUATIONS = 3

        if "<tool_call>" not in full_content or "</tool_call>" in full_content:
            return full_content

        self.log.warning("Truncated response — attempting continuation")
        combined = full_content

        for attempt in range(MAX_CONTINUATIONS):
            tail = combined[-200:]
            continuation_prompt = (
                "Your response was truncated. Continue from exactly where you left off. "
                f"Last part ended with:\n...{tail}"
            )

            try:
                msg, _ = self.api.send_and_wait_conv(
                    continuation_prompt, conversation_id=self.conversation_id,
                    cancel_check=self.cb.check_cancelled)
            except Exception as e:
                self.log.error("Continuation attempt %d failed: %s", attempt + 1, e)
                break

            part = self.api.extract_text(msg)
            self.approx_char_usage += len(continuation_prompt) + len(part)
            combined += part
            self.log.info("Continuation %d: +%d chars (total %d)",
                          attempt + 1, len(part), len(combined))

            if "</tool_call>" in combined:
                self.cb.on_truncation_recovered(attempt + 1)
                return combined

        self.cb.on_truncation_failed(MAX_CONTINUATIONS)
        self.log.warning("Still truncated after %d continuations", MAX_CONTINUATIONS)
        return combined

    # ── Context recovery (long mode) ─────────────────────────────────

    def _recover_context(self, auto=False):
        """Recover from context limit in long mode. Returns new mode string."""
        self.log.warning("Long mode context limit — initiating recovery")

        # Try to get summary from server conversation
        summary_text = self.summary_state.get("text", "")
        if self.conversation_id and not summary_text:
            try:
                conv = self.api.get_conversation(self.conversation_id)
                msg_map = conv.get("messageMap", {})
                messages = []
                for mid, msg in msg_map.items():
                    if msg.get("role") in ("user", "assistant"):
                        content = ""
                        for c in msg.get("content", []):
                            if c.get("contentType") == "text":
                                content += c.get("body", "")
                        messages.append({"role": msg["role"], "content": content[:600]})
                if messages:
                    self.cb.on_summary_start(len(messages[-20:]))
                    self._generate_summary(messages[-20:])
                    self.cb.on_summary_done()
                    summary_text = self.summary_state["text"]
            except Exception as e:
                self.log.error("Failed to fetch conversation for recovery: %s", e)

        if not summary_text:
            recent = self.conversation_history[-20:] if self.conversation_history else []
            if recent:
                self.cb.on_summary_start(len(recent))
                self._generate_summary(recent)
                self.cb.on_summary_done()

        self.conversation_id = None
        self.approx_char_usage = 0

        return self.cb.on_context_recovery(auto)

    # ── Prompt building (dev mode) ───────────────────────────────────

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
        base = (
            f"You are an autonomous agent with access to tools for file operations, "
            f"command execution, web fetching, and more.\n\n"
            f"AVAILABLE TOOLS:\n{tools_block}\n\n"
            f"TO USE A TOOL, include a tool call block in your response:\n\n"
            f"<tool_call>\n"
            f'{{\"tool\": \"tool_name\", \"args\": {{\"param1\": \"value1\", \"param2\": \"value2\"}}}}\n'
            f"</tool_call>\n\n"
            f"RULES:\n"
            f"- You may use multiple tool calls in a single response.\n"
            f"- After tool execution, you will receive results and can make more calls or give a final answer.\n"
            f"- When done, respond with plain text (no tool_call block).\n"
            f"- Always explain what you're doing before tool calls.\n"
            f"- Be careful with destructive commands — ask before deleting files or modifying system config.\n"
            f"- Do not use interactive commands (vim, less, top).\n"
            f"- Read files before overwriting them.\n"
        )
        if self.extra_system_prompt:
            base += f"\n{self.extra_system_prompt}\n"
        return base

    def _build_prompt(self):
        """Build prompt for dev mode. Returns (prompt_str, oldest_included_idx)."""
        max_chars = self.max_context_chars
        parts = [f"[System]\n{self._build_tool_system_prompt()}\n[End System]\n"]

        if self.initial_files:
            parts.append(self.initial_files)

        if self.summary_state["text"]:
            parts.append(f"Progress summary of work done so far:\n{self.summary_state['text']}")
            parts.append(
                f"IMPORTANT: Your working directory is '{os.getcwd()}'. "
                "Use relative paths — do not cd elsewhere. "
                "Continue where you left off. Do not repeat already-completed steps."
            )

        overhead = sum(len(p) for p in parts) + 500
        budget = max_chars - overhead
        selected = []
        oldest_idx = len(self.conversation_history)

        for i in range(len(self.conversation_history) - 1, -1, -1):
            msg = self.conversation_history[i]
            content = msg.get("content", "") or ""
            msg_chars = len(content)
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
        self.log.debug("Prompt built: %d chars, %d msgs (oldest_idx=%d)",
                       len(prompt), len(selected), oldest_idx)
        return prompt, oldest_idx

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_wind_down(self, turn_num, max_turns):
        """Return wind-down message string or None."""
        remaining = max_turns - turn_num
        if 0 < remaining <= self.wind_down_turns:
            return (
                f"[SYSTEM: {remaining} turns remaining before overtime. "
                f"Begin wrapping up — save your progress (CONSOLIDATE), "
                f"commit your work (PERSIST), and stop. "
                f"Do not start new tasks.]"
            )
        elif remaining <= 0:
            overtime = -remaining
            return (
                f"[SYSTEM: You are {overtime} turns past the turn limit. "
                f"Finish what you are doing immediately — CONSOLIDATE and PERSIST now. "
                f"Do not start anything new.]"
            )
        return None

    @staticmethod
    def _inject_wind_down(prompt, message):
        if prompt.endswith("Assistant:"):
            prompt = prompt[:-len("Assistant:")]
        return prompt + f"\n\n{message}\n\nAssistant:"

    def _last_user_message(self):
        for msg in reversed(self.conversation_history):
            if msg["role"] == "user":
                return msg["content"]
        return ""

    def _pending_tool_results(self):
        """Check if the last messages in history are tool results."""
        if not self.conversation_history:
            return False
        return self.conversation_history[-1]["role"] == "tool"

    def _count_trailing_tool_results(self):
        count = 0
        for msg in reversed(self.conversation_history):
            if msg["role"] == "tool":
                count += 1
            else:
                break
        return count

    def _format_tool_results(self, tool_count):
        """Format trailing tool results as a single message for long mode."""
        parts = ["[Tool results]"]
        tool_msgs = []
        for msg in self.conversation_history[-(tool_count * 2):]:
            if msg["role"] == "tool":
                tool_msgs.append(msg)
        for msg in tool_msgs[-tool_count:]:
            name = msg.get("name", "?")
            content = msg.get("content", "")
            parts.append(f"{name}: {content}")
        return "\n\n".join(parts)

    @staticmethod
    def _is_cancel(exc):
        """Check if an exception is a cancellation."""
        return type(exc).__name__ == "CancelledError"

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


class _CancelledSignal(Exception):
    """Internal signal for cancellation within the run loop."""
    pass


class _LegacyCallbacks(NullCallbacks):
    """Wraps old-style on_turn/on_tool into the callbacks interface."""
    def __init__(self, on_turn=None, on_tool=None):
        self._on_turn = on_turn
        self._on_tool = on_tool

    def on_tool_start(self, name, args):
        if self._on_tool:
            self._on_tool(name, args)

    def on_turn_end(self, turn, turn_result):
        if self._on_turn:
            self._on_turn(turn, turn_result)
