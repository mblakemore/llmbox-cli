#!/usr/bin/env python3
"""
Agent script with file reading/writing tools.
Connects to Bedrock Chat Published API and executes tool calls in an agentic loop.
"""

import json
import logging
import logging.handlers
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from cancel import cancellable, check_cancelled, CancelledError
from spinner import StreamStatus
from bedrock_api import BedrockChatAPI
from tools import MAP_FN, tools, load_extra_tools
from tools.exec_command import cleanup_temp_sessions

DIM = "\033[90m"
BOLD = "\033[1m"
BLUE = "\033[34m"
RESET = "\033[0m"

_FILE_REF = re.compile(r"@(\S+)")

# Regex to extract tool calls from model output
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)

# ── Configuration ──────────────────────────────────────────────────────

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
        "summary_threshold": 5,
        "max_context_chars": 80000,  # ~20k tokens
    },
    "cycle": {
        "max_turns": 100,
        "wind_down_turns": 10,
    },
}


def _load_config():
    config = json.loads(json.dumps(_DEFAULT_CONFIG))
    config_path = os.path.join(os.getcwd(), "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                user_config = json.load(f)
            for section, values in user_config.items():
                if section in config and isinstance(config[section], dict):
                    config[section].update(values)
                else:
                    config[section] = values
        except Exception as e:
            print(f"Warning: failed to load config.json: {e}")
    return config


_config = _load_config()

_MAX_FULL_LINES = _config["context"]["max_full_lines"]
_PREVIEW_LINES = _config["context"]["preview_lines"]
_SUMMARY_THRESHOLD = _config["context"]["summary_threshold"]
_MAX_CONTEXT_CHARS = _config["context"]["max_context_chars"]
_MAX_TURNS = _config["cycle"]["max_turns"]
_WIND_DOWN_TURNS = _config["cycle"]["wind_down_turns"]

# Initialize API client
_api = BedrockChatAPI(_config["llm"])

# Load agent-specific tools from CWD/tools/ if it exists
_agent_tools_dir = os.path.join(os.getcwd(), "tools")
if os.path.isdir(_agent_tools_dir):
    load_extra_tools(_agent_tools_dir)


# ── Tool prompt generation ────────────────────────────────────────────

def _build_tool_system_prompt():
    """Generate a system prompt that describes all available tools."""
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

    return f"""\
You are an autonomous agent with access to tools for file operations, \
command execution, web fetching, and more.

AVAILABLE TOOLS:
{tools_block}

TO USE A TOOL, include a tool call block in your response:

<tool_call>
{{"tool": "tool_name", "args": {{"param1": "value1", "param2": "value2"}}}}
</tool_call>

RULES:
- You may use multiple tool calls in a single response.
- After tool execution, you will receive results and can make more calls or give a final answer.
- When done, respond with plain text (no tool_call block).
- Always explain what you're doing before tool calls.
- Be careful with destructive commands — ask before deleting files or modifying system config.
- Do not use interactive commands (vim, less, top).
- Read files before overwriting them.
"""


_TOOL_SYSTEM_PROMPT = _build_tool_system_prompt()


# ── Text utilities ─────────────────────────────────────────────────────

_UNICODE_MAP = str.maketrans({
    "\u2014": "--", "\u2013": "-", "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2022": "*",
    "\u00a0": " ", "\u200b": "",
})

_THINK_TAG_RE = re.compile(r'</?think>')


def _sanitize(text):
    """Replace common Unicode characters with ASCII equivalents and strip think tags."""
    text = _THINK_TAG_RE.sub('', text)
    return text.translate(_UNICODE_MAP)


# ── Token/char estimation ─────────────────────────────────────────────

def _estimate_chars(msg):
    """Estimate the character count of a conversation message."""
    content = msg.get("content", "") or ""
    total = len(content)
    if msg.get("tool_calls"):
        total += len(json.dumps(msg["tool_calls"]))
    return max(1, total)


# ── File reference expansion ──────────────────────────────────────────

def _expand_file_refs(text):
    """Expand @filepath references in user input to inline file contents."""
    refs = _FILE_REF.findall(text)
    if not refs:
        return text, None, None

    seen = set()
    attachments = []
    for ref in refs:
        if ref in seen:
            continue
        seen.add(ref)

        p = Path(ref)
        if not p.exists():
            return None, None, f"Error: file '{ref}' does not exist"
        if p.is_dir():
            return None, None, f"Error: '{ref}' is a directory, not a file"

        lines = p.read_text().splitlines(True)
        total = len(lines)
        if total <= _MAX_FULL_LINES or p.name == "agent.md":
            content = "".join(lines)
            header = f"[{ref}: {total} lines]"
        else:
            content = "".join(lines[:_PREVIEW_LINES])
            header = f"[{ref}: first {_PREVIEW_LINES} of {total} lines]"

        resolved = str(p.resolve())
        if p.name == "agent.md":
            header = (f"[AGENT IDENTITY FILE: {ref} (loaded from {resolved}). "
                      f"This is YOUR agent.md — do not search for it elsewhere. {total} lines]")

        attachments.append(f"{header}\n{content}")
        print(f"  {DIM}{header}{RESET}")

    files_content = "\n\n".join(attachments)
    cwd = os.getcwd()
    if any(Path(ref).name == "agent.md" for ref in seen):
        preamble = (
            f"[SYSTEM CONTEXT: Your working directory is {cwd}. "
            f"All relative paths resolve from here. "
            f"Do not cd to other repositories or search for files outside this tree.]\n\n"
        )
    else:
        preamble = ""
    expanded = text + "\n\n" + preamble + files_content
    return expanded, preamble + files_content if preamble else files_content, None


# ── Summarization ─────────────────────────────────────────────────────

def _format_for_summary(messages):
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
                args = json.dumps(tc.get("args", {}))
                if len(args) > 200:
                    args = args[:200] + "..."
                parts.append(f"ASSISTANT called {name}({args})")
        else:
            content = m.get("content", "")
            if len(content) > 800:
                content = content[:800] + "..."
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _generate_summary(old_summary, new_messages, log):
    """Call the LLM to produce an updated conversation summary."""
    transcript = _format_for_summary(new_messages)

    structure_instruction = (
        "Structure the summary with these sections:\n"
        "1. GOAL: The user's current objective\n"
        "2. PROGRESS: What has been accomplished\n"
        "3. DECISIONS & OUTCOMES: Key decisions made and their results "
        "(include approaches that FAILED and why — this prevents repeating mistakes)\n"
        "4. COMPLETED ACTIONS: List actions that are DONE and must NOT be repeated\n"
        "5. CURRENT STATE: Where things stand right now, what files were modified\n"
        "6. NEXT: The single next action to take\n"
        "Keep it under 500 words. Be specific about file paths, error messages, and tool results."
    )

    if old_summary:
        prompt = (
            f"Here is the previous summary of the conversation so far:\n\n"
            f"{old_summary}\n\n"
            f"Here are the new messages since that summary:\n\n"
            f"{transcript}\n\n"
            f"Write an updated summary that combines the previous summary with the new messages.\n\n"
            f"{structure_instruction}"
        )
    else:
        prompt = (
            f"Here is a conversation transcript:\n\n"
            f"{transcript}\n\n"
            f"Write a concise summary.\n\n"
            f"{structure_instruction}"
        )

    log.info("Generating conversation summary...")
    try:
        msg = _api.send_and_wait(prompt)
        summary = _api.extract_text(msg).strip()
        log.info("SUMMARY: %s", summary)
        return summary
    except Exception as e:
        log.error("Summary generation failed: %s", e)
        return old_summary or ""


# ── Context window management ─────────────────────────────────────────

def _build_prompt(conversation_history, summary_state, initial_files, log,
                  max_chars_override=None):
    """Build the full prompt from conversation history using prompt stuffing.

    Returns (prompt_str, oldest_included_idx).
    """
    max_chars = max_chars_override or _MAX_CONTEXT_CHARS
    parts = []

    # System prompt with tools
    parts.append(f"[System]\n{_TOOL_SYSTEM_PROMPT}\n[End System]\n")

    # Initial files (agent.md etc.) — always include if present
    if initial_files:
        parts.append(initial_files)

    # Summary context
    if summary_state["text"]:
        parts.append(f"Progress summary of work done so far:\n{summary_state['text']}")
        parts.append(
            f"IMPORTANT: Your working directory is '{os.getcwd()}'. "
            "Use relative paths — do not cd elsewhere. "
            "Continue where you left off. Do not repeat already-completed steps."
        )

    # Build conversation from history (most recent first, within budget)
    overhead = sum(len(p) for p in parts) + 500  # room for wrapper text
    budget = max_chars - overhead
    selected = []
    oldest_idx = len(conversation_history)

    for i in range(len(conversation_history) - 1, -1, -1):
        msg = conversation_history[i]
        msg_chars = _estimate_chars(msg)
        if sum(_estimate_chars(m) for m in selected) + msg_chars > budget:
            break
        selected.append(msg)
        oldest_idx = i

    selected.reverse()

    # Format messages
    for msg in selected:
        role = msg["role"]
        content = msg.get("content", "")
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
            # Include tool call info
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    parts.append(f"[Tool call: {tc['name']}({json.dumps(tc.get('args', {}))})]")
        elif role == "tool":
            name = msg.get("name", "?")
            parts.append(f"[Tool result ({name}): {content}]")

    prompt = "\n\n".join(parts) + "\n\nAssistant:"

    log.debug("Prompt built: %d chars, %d messages included (oldest_idx=%d)",
              len(prompt), len(selected), oldest_idx)
    return prompt, oldest_idx


def _maybe_resummarize(conversation_history, summary_state, oldest_idx, log, force=False):
    """Check if enough messages have fallen out of the window to warrant a new summary."""
    unsummarized = oldest_idx - summary_state["up_to"]

    if not force and unsummarized < _SUMMARY_THRESHOLD:
        return False

    new_messages = conversation_history[summary_state["up_to"]:oldest_idx]
    print(f"  {DIM}[summarizing {len(new_messages)} messages...]{RESET}", flush=True)
    summary = _generate_summary(summary_state["text"], new_messages, log)
    summary_state["text"] = summary
    summary_state["up_to"] = oldest_idx
    print(f"  {DIM}[summary updated]{RESET}")
    return True


# ── Tool call parsing ─────────────────────────────────────────────────

def _parse_tool_calls(response_text):
    """Extract tool calls from model response text.

    Returns list of dicts with 'name' and 'args' keys.
    """
    calls = []
    for match in _TOOL_CALL_RE.finditer(response_text):
        try:
            data = json.loads(match.group(1))
            name = data.get("tool") or data.get("name")
            args = data.get("args") or data.get("arguments") or {}
            # Handle flat format: {"tool": "exec_command", "command": "ls"}
            if not args and name:
                args = {k: v for k, v in data.items() if k not in ("tool", "name")}
            if name:
                calls.append({"name": name, "args": args})
        except json.JSONDecodeError:
            continue
    return calls


def _strip_tool_calls(response_text):
    """Remove tool call blocks from response text."""
    return _TOOL_CALL_RE.sub("", response_text).strip()


# ── Logger setup ──────────────────────────────────────────────────────

def _setup_logger():
    """Create a structured logger with levels, rotation, and console output."""
    history_dir = os.path.join(os.getcwd(), ".history")
    os.makedirs(history_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(history_dir, f"session_{timestamp}.log")
    error_log_path = os.path.join(history_dir, "errors.log")

    logger = logging.getLogger("agent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=10*1024*1024, backupCount=5)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(file_handler)

    error_handler = logging.handlers.RotatingFileHandler(
        error_log_path, maxBytes=5*1024*1024, backupCount=3)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(logging.Formatter('%(asctime)s ERROR %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(error_handler)

    return logger, log_path, error_log_path


# ── Conversation checkpoints (for -c continue) ──────────────────────

_CHECKPOINT_PATH = os.path.join(os.getcwd(), "state", "conversation_checkpoint.json")


def _save_checkpoint(conversation_history, summary_state, turn, initial_files):
    """Save conversation state so a crashed cycle can be resumed with -c."""
    try:
        checkpoint = {
            "conversation_history": conversation_history,
            "summary_state": summary_state,
            "turn": turn,
            "initial_files": initial_files,
        }
        os.makedirs(os.path.dirname(_CHECKPOINT_PATH), exist_ok=True)
        with open(_CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
    except Exception:
        pass


def _load_checkpoint():
    """Load a saved conversation checkpoint."""
    if not os.path.exists(_CHECKPOINT_PATH):
        return None
    try:
        with open(_CHECKPOINT_PATH) as f:
            cp = json.load(f)
        return (
            cp["conversation_history"],
            cp["summary_state"],
            cp.get("turn", 0),
            cp.get("initial_files"),
        )
    except Exception:
        return None


def _delete_checkpoint():
    """Remove checkpoint after a clean exit."""
    try:
        if os.path.exists(_CHECKPOINT_PATH):
            os.remove(_CHECKPOINT_PATH)
    except Exception:
        pass


# ── Cycle auto-increment ─────────────────────────────────────────────

def _auto_increment_cycle(log):
    """Check if the current cycle was already committed and bump if so."""
    state_path = os.path.join(os.getcwd(), "state", "current-state.json")
    if not os.path.exists(state_path):
        return

    try:
        with open(state_path) as f:
            state = json.load(f)
        cycle = int(state.get("cycle", 0))
        if cycle <= 0:
            return

        result = subprocess.run(
            ["git", "log", "--oneline", "-20"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return

        committed_cycles = set()
        for line in result.stdout.strip().split("\n"):
            m = re.search(r'\bC(\d+):', line)
            if m:
                committed_cycles.add(int(m.group(1)))

        if not committed_cycles:
            return

        highest_committed = max(committed_cycles)

        if cycle <= highest_committed:
            new_cycle = highest_committed + 1
            state["cycle"] = new_cycle
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2)
                f.write("\n")

            focus_path = os.path.join(os.getcwd(), "state", "focus.json")
            if os.path.exists(focus_path):
                try:
                    with open(focus_path) as f:
                        focus = json.load(f)
                    if int(focus.get("cycle", 0)) <= cycle:
                        focus["cycle"] = new_cycle
                        with open(focus_path, "w") as f:
                            json.dump(focus, f, indent=2)
                            f.write("\n")
                except Exception:
                    pass

            log.info("AUTO-INCREMENT: cycle %d already committed, bumped state to %d",
                     cycle, new_cycle)
            print(f"  [auto-increment: cycle {cycle} already committed → starting cycle {new_cycle}]")
    except Exception as e:
        log.warning("Auto-increment check failed: %s", e)


# ── Main agent loop ───────────────────────────────────────────────────

def run_agent_interactive(initial_prompt=None, auto=False, continue_mode=False):
    """Interactive agent that maintains conversation history."""

    log, log_path, error_log_path = _setup_logger()

    print("="*60)
    print("Agent with File Tools - Bedrock Chat API")
    print("="*60)
    print(f"Model: {_api.model} | Max turns: {_MAX_TURNS}")
    print(f"Session log: {log_path}")
    print(f"Error log: {error_log_path}")
    print("Press Escape twice to cancel")
    print("Type 'exit' or 'quit' to end conversation\n")

    # Health check
    status = StreamStatus()
    status.start("  Checking API health ")
    healthy = _api.health()
    status.first_token()
    status.finish()
    if healthy:
        print(f"  {DIM}[API healthy]{RESET}")
    else:
        print(f"  {BOLD}[WARNING: API health check failed]{RESET}")

    log.info("Session started | model=%s max_turns=%d", _api.model, _MAX_TURNS)
    log.info("Tools registered: %s", [t["function"]["name"] for t in tools])

    # ── Continue mode: resume from checkpoint ──
    start_turn = 0
    conversation_history = []
    summary_state = {"text": "", "up_to": 0}
    initial_files = None

    if continue_mode:
        cp = _load_checkpoint()
        if cp:
            conversation_history, summary_state, start_turn, initial_files = cp
            log.info("CONTINUE: resuming from checkpoint (turn %d, %d messages)",
                     start_turn, len(conversation_history))
            print(f"  [continuing from turn {start_turn} with {len(conversation_history)} messages]")
            conversation_history.append({"role": "user", "content":
                "Continue where you left off. The session was interrupted — "
                "pick up from your current phase and finish the cycle."})
            result = run_agent_single(conversation_history, summary_state, initial_files, log,
                                      start_turn=start_turn)
            if auto:
                cleanup_temp_sessions()
                _delete_checkpoint()
                log.info("Session ended (continue mode) | %d messages", len(conversation_history))
                return
        else:
            print("  [no checkpoint found — starting fresh]")
            log.info("CONTINUE: no checkpoint found, starting fresh")

    if not continue_mode:
        _auto_increment_cycle(log)

    if not (continue_mode and start_turn > 0):
        conversation_history = []
        summary_state = {"text": "", "up_to": 0}
        initial_files = None

        # Auto-load agent.md from cwd if present
        agent_md = Path(os.getcwd()) / "agent.md"
        if agent_md.exists():
            content = agent_md.read_text()
            total = len(content.splitlines())
            resolved = str(agent_md.resolve())
            header = (f"[AGENT IDENTITY FILE: agent.md (loaded from {resolved}). "
                      f"This is YOUR agent.md — do not search for it elsewhere. {total} lines]")
            cwd = os.getcwd()
            preamble = (
                f"[SYSTEM CONTEXT: Your working directory is {cwd}. "
                f"All relative paths resolve from here. "
                f"Do not cd to other repositories or search for files outside this tree.]\n\n"
            )
            initial_files = f"{preamble}{header}\n{content}"
            print(f"  {DIM}{header}{RESET}")
            log.info("Auto-loaded agent.md (%d lines)", total)

    if initial_prompt and not (continue_mode and start_turn > 0):
        print(f"You: {initial_prompt}")
        expanded, files, err = _expand_file_refs(initial_prompt)
        if err:
            print(err)
            return
        if files:
            initial_files = files
        conversation_history.append({"role": "user", "content": expanded})
        log.info("USER: %s", expanded)
        result = run_agent_single(conversation_history, summary_state, initial_files, log)

        if auto:
            if result == "cancelled":
                print(f"\n{BOLD}[Agent paused — enter guidance, or press Enter to resume]{RESET}")
                try:
                    guidance = input("\nOperator: ").strip()
                except (EOFError, KeyboardInterrupt):
                    log.info("Session ended (operator cancelled) | %d messages", len(conversation_history))
                    print()
                    return
                if guidance:
                    expanded_g, files_g, err_g = _expand_file_refs(guidance)
                    if err_g:
                        print(err_g)
                    else:
                        if files_g:
                            initial_files = files_g
                        conversation_history.append({"role": "user", "content": expanded_g})
                        log.info("OPERATOR: %s", expanded_g)
                else:
                    conversation_history.append({"role": "user", "content":
                        "Continue where you left off. Finish your current cycle."})
                    log.info("OPERATOR: [resume — no guidance]")
                run_agent_single(conversation_history, summary_state, initial_files, log)
            cleanup_temp_sessions()
            _delete_checkpoint()
            log.info("Session ended (auto mode) | %d messages in history", len(conversation_history))
            return

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        if user_input.strip() == "/clear":
            conversation_history.clear()
            summary_state["text"] = ""
            summary_state["up_to"] = 0
            initial_files = None
            log, log_path, error_log_path = _setup_logger()
            print(f"Conversation cleared. New session: {log_path}")
            continue

        expanded, files, err = _expand_file_refs(user_input)
        if err:
            print(err)
            continue
        if files:
            initial_files = files

        conversation_history.append({"role": "user", "content": expanded})
        log.info("USER: %s", expanded)

        run_agent_single(conversation_history, summary_state, initial_files, log)

    cleanup_temp_sessions()
    _delete_checkpoint()
    log.info("Session ended | %d messages in history", len(conversation_history))


def run_agent_single(conversation_history: list, summary_state: dict, initial_files,
                     log: logging.Logger, start_turn=0):
    """Run the agentic loop with turn limits and wind-down."""

    turn = start_turn

    # Track repeated tool failures to break infinite loops
    _recent_tool_errors = []
    _REPEAT_THRESHOLD = 3

    while True:
        turn += 1

        # ── Wind-down and overtime warnings ──
        remaining = _MAX_TURNS - turn
        wind_down_msg = None
        if 0 < remaining <= _WIND_DOWN_TURNS:
            wind_down_msg = (
                f"[SYSTEM: {remaining} turns remaining before overtime. "
                f"Begin wrapping up — save your progress (CONSOLIDATE), "
                f"commit your work (PERSIST), and stop. "
                f"Do not start new tasks.]"
            )
            log.info("Wind-down: %d turns remaining", remaining)
        elif remaining <= 0:
            overtime = -remaining
            wind_down_msg = (
                f"[SYSTEM: You are {overtime} turns past the turn limit. "
                f"Finish what you are doing immediately — CONSOLIDATE and PERSIST now. "
                f"Do not start anything new.]"
            )
            log.warning("Overtime: %d turns past limit (%d)", overtime, _MAX_TURNS)

        # Build the prompt
        prompt, oldest_idx = _build_prompt(
            conversation_history, summary_state, initial_files, log)

        # Resummarize if needed
        if _maybe_resummarize(conversation_history, summary_state, oldest_idx, log):
            prompt, oldest_idx = _build_prompt(
                conversation_history, summary_state, initial_files, log)

        # Inject wind-down
        if wind_down_msg:
            prompt = prompt.rstrip()
            if prompt.endswith("Assistant:"):
                prompt = prompt[:-len("Assistant:")]
            prompt += f"\n\n{wind_down_msg}\n\nAssistant:"

        log.info("--- Turn %d/%d | prompt %d chars (history has %d msgs)",
                 turn, _MAX_TURNS, len(prompt), len(conversation_history))

        # Call the model
        status = StreamStatus()
        status.start("\nAssistant: ")

        try:
            with cancellable():
                msg = _api.send_and_wait(prompt, cancel_check=check_cancelled)
        except CancelledError:
            status.finish()
            print(f"\n[cancelled]{RESET}")
            log.info("CANCELLED during API call")
            return "cancelled"
        except TimeoutError:
            status.finish()
            log.error("API call timed out")
            print(f"\n  {DIM}[timed out waiting for response]{RESET}")
            return "error"
        except Exception as e:
            status.finish()
            log.error("API call failed: %s", e)
            print(f"\nError calling API: {e}")
            return "error"

        status.first_token()
        status.finish()

        # Extract response
        full_content = _api.extract_text(msg)
        reasoning = _api.extract_reasoning(msg)

        if reasoning:
            print(f"{BLUE}[Reasoning]\n{reasoning}{RESET}\n")

        # Parse tool calls
        tool_calls = _parse_tool_calls(full_content)
        text_part = _strip_tool_calls(full_content)

        if text_part:
            text_part = _sanitize(text_part)
            # Strip leading "Assistant:" if echoed
            if text_part.startswith("Assistant:"):
                text_part = text_part[len("Assistant:"):].strip()
            print(text_part)

        log.info("ASSISTANT: %s", full_content[:500])

        # Record in history
        assistant_msg = {"role": "assistant", "content": full_content}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        conversation_history.append(assistant_msg)

        if not tool_calls:
            log.info("No tool calls — stopping")
            return "done"

        # Execute tool calls
        log.info("Executing %d tool call(s)", len(tool_calls))
        print(f"\n{DIM}Executing {len(tool_calls)} tool call(s)...")
        try:
            with cancellable():
                for tc in tool_calls:
                    check_cancelled()
                    func_name = tc["name"]
                    func_args = tc["args"]

                    log.info("TOOL CALL: %s(%s)", func_name, json.dumps(func_args))

                    _STREAMING_TOOLS = {"think"}
                    use_spinner = func_name not in _STREAMING_TOOLS

                    if use_spinner:
                        tool_status = StreamStatus()
                        tool_status.start(f"  -> {func_name} ")

                    if func_name not in MAP_FN:
                        result_str = f"Error: Unknown tool '{func_name}'"
                    else:
                        try:
                            result_str = str(MAP_FN[func_name](**func_args))
                        except Exception as e:
                            result_str = f"Error executing tool: {str(e)}"

                    if use_spinner:
                        tool_status.first_token()
                        tool_status.finish()
                    print(f"\r\033[K  -> {func_name}({', '.join(f'{k}={repr(v)[:50]}' for k, v in func_args.items())})")

                    conversation_history.append({
                        "role": "tool",
                        "name": func_name,
                        "content": result_str,
                    })

                    log.info("TOOL RESULT [%s]: %s", func_name, result_str[:500])

                    # Track repeated errors to detect infinite loops
                    if result_str.startswith("Error"):
                        error_sig = (func_name, result_str[:100])
                        _recent_tool_errors.append(error_sig)
                        consecutive = sum(1 for e in _recent_tool_errors if e == error_sig)
                        if consecutive >= _REPEAT_THRESHOLD:
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
                            log.warning("Loop detected: %s x%d — forcing think",
                                        func_name, consecutive)
                            print(f"  [loop detected — forcing think]")
                            if "think" in MAP_FN:
                                think_result = MAP_FN["think"](prompt=think_prompt)
                                conversation_history.append({
                                    "role": "assistant",
                                    "content": f"[Forced reflection]\n{think_result}",
                                })
                                log.info("FORCED THINK RESULT: %s", think_result)
                    else:
                        _recent_tool_errors[:] = [e for e in _recent_tool_errors if e[0] != func_name]
                    print(f"    Result: {result_str}{RESET}")
        except CancelledError:
            print(f"\n[cancelled]{RESET}")
            log.info("CANCELLED during tool execution")
            _save_checkpoint(conversation_history, summary_state, turn, initial_files)
            return "cancelled"

        # Save checkpoint after each turn
        _save_checkpoint(conversation_history, summary_state, turn, initial_files)


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Agent with file tools (Bedrock Chat API)")
    parser.add_argument("-a", "--auto", action="store_true",
                        help="Automation mode: run prompt and exit (no interactive loop)")
    parser.add_argument("-c", "--continue", dest="continue_mode", action="store_true",
                        help="Continue from last checkpoint")
    parser.add_argument("-r", "--repeat", type=int, nargs="?", const=0, default=None,
                        help="Repeat N times (fresh each run). 0 or omit = indefinite. Implies -a.")
    parser.add_argument("-m", "--model", default=None,
                        help="Override model (e.g. claude-v4.5-opus)")
    parser.add_argument("prompt", nargs="*", help="Initial prompt")
    args = parser.parse_args()

    if args.model:
        _api.model = args.model

    initial_prompt = " ".join(args.prompt).strip() or None

    if args.continue_mode:
        run_agent_interactive(initial_prompt=initial_prompt, auto=True, continue_mode=True)
    elif args.repeat is not None:
        n = args.repeat
        run = 0
        try:
            while n == 0 or run < n:
                run += 1
                label = f"run {run}/{n}" if n > 0 else f"run {run}"
                print(f"\n{'='*60}\n{label}\n{'='*60}")
                run_agent_interactive(initial_prompt=initial_prompt, auto=True)
        except KeyboardInterrupt:
            print(f"\n\nStopped after {run} run(s).")
    else:
        run_agent_interactive(initial_prompt=initial_prompt, auto=args.auto)


if __name__ == "__main__":
    main()
