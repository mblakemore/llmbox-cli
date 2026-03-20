#!/usr/bin/env python3
"""
Agent CLI — thin wrapper around llmbox_lib.Agent.

Provides interactive terminal features: spinners, cancellation, checkpoint
management, interactive commands (/help, /mode, /model, etc.).
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
from llmbox_lib import Agent, NullCallbacks, TurnResult
from tools import tools
from tools.exec_command import cleanup_temp_sessions

# Optional prompt_toolkit TUI
try:
    from tui import TuiSession, TuiCallbacks, show_context, show_tools
    _HAS_TUI = True
except ImportError:
    _HAS_TUI = False

DIM = "\033[90m"
BOLD = "\033[1m"
BLUE = "\033[34m"
RESET = "\033[0m"

_FILE_REF = re.compile(r"@(\S+)")


# ── Configuration ──────────────────────────────────────────────────────

_DEFAULT_CONFIG = {
    "llm": {
        "api_url": os.environ.get("BEDROCK_API_URL", ""),
        "api_key": os.environ.get("BEDROCK_API_KEY", ""),
        "origin": "http://localhost:8000",
        "model": "claude-v4.5-sonnet",
        "poll_interval": 0.3,
        "poll_backoff": 1.5,
        "poll_max_interval": 5.0,
        "poll_timeout": 180,
    },
    "context": {
        "max_full_lines": 400,
        "preview_lines": 100,
        "summary_threshold": 5,
        "max_context_chars": 80000,
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
                    for k, v in values.items():
                        if v or v == 0 or v is False:
                            config[section][k] = v
                else:
                    config[section] = values
        except Exception as e:
            print(f"Warning: failed to load config.json: {e}")
    return config


_config = _load_config()

_MAX_FULL_LINES = _config["context"]["max_full_lines"]
_PREVIEW_LINES = _config["context"]["preview_lines"]


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


# ── Logger setup ──────────────────────────────────────────────────────

def _setup_logger():
    history_dir = os.path.join(os.getcwd(), ".llmbox", "history")
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


# ── Conversation checkpoints ──────────────────────────────────────────

_CHECKPOINT_PATH = os.path.join(os.getcwd(), ".llmbox", "state", "conversation_checkpoint.json")


def _save_checkpoint(agent, turn=0):
    """Save conversation state so a crashed cycle can be resumed with -c."""
    try:
        checkpoint = {
            "mode": agent.mode,
            "turn": turn,
            "conversation_history": agent.conversation_history,
            "summary_state": agent.summary_state,
            "initial_files": agent.initial_files,
        }
        if agent.mode == "long" and agent.conversation_id:
            checkpoint["conversation_id"] = agent.conversation_id
            checkpoint["approx_chars"] = agent.approx_char_usage
        os.makedirs(os.path.dirname(_CHECKPOINT_PATH), exist_ok=True)
        with open(_CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f)
    except Exception:
        pass


def _load_checkpoint():
    if not os.path.exists(_CHECKPOINT_PATH):
        return None
    try:
        with open(_CHECKPOINT_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def _delete_checkpoint():
    try:
        if os.path.exists(_CHECKPOINT_PATH):
            os.remove(_CHECKPOINT_PATH)
    except Exception:
        pass


# ── Cycle auto-increment ─────────────────────────────────────────────

def _auto_increment_cycle(log):
    state_path = os.path.join(os.getcwd(), ".llmbox", "state", "current-state.json")
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

            focus_path = os.path.join(os.getcwd(), ".llmbox", "state", "focus.json")
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


# ── Terminal callbacks ────────────────────────────────────────────────

class TerminalCallbacks(NullCallbacks):
    """Callbacks that provide terminal UI: spinners, cancellation, output, checkpoints."""

    def __init__(self, agent, log, auto=False):
        self.agent = agent
        self.log = log
        self.auto = auto
        self._status = None
        self._tool_status = None
        self._cancellable = None

    def check_cancelled(self):
        check_cancelled()

    def on_api_start(self, label):
        self._cancellable = cancellable()
        self._cancellable.__enter__()
        self._status = StreamStatus()
        self._status.start(label)

    def on_api_response(self):
        if self._status:
            self._status.first_token()

    def on_api_done(self):
        if self._status:
            self._status.finish()
            self._status = None
        if self._cancellable:
            self._cancellable.__exit__(None, None, None)
            self._cancellable = None

    def on_assistant_text(self, text, reasoning):
        if reasoning:
            print(f"{BLUE}[Reasoning]\n{reasoning}{RESET}\n")
        if text:
            print(f"\nAssistant: {text}")

    def on_tool_batch_start(self, count):
        # Enter cancellable mode for tool execution
        self._cancellable = cancellable()
        self._cancellable.__enter__()
        print(f"\n{DIM}Executing {count} tool call(s)...")

    def on_tool_start(self, name, args):
        if name not in {"think"}:
            self._tool_status = StreamStatus()
            self._tool_status.start(f"  -> {name} ")
        else:
            self._tool_status = None

    def on_tool_result(self, name, args, result, is_error):
        if self._tool_status:
            self._tool_status.first_token()
            self._tool_status.finish()
            self._tool_status = None
        args_str = ', '.join(f'{k}={repr(v)[:50]}' for k, v in args.items())
        print(f"\r\033[K  -> {name}({args_str})")
        print(f"    Result: {result}{RESET}")

    def on_turn_end(self, turn, turn_result):
        # Exit cancellable mode if still active
        if self._cancellable:
            self._cancellable.__exit__(None, None, None)
            self._cancellable = None
        _save_checkpoint(self.agent, turn)

    def on_summary_start(self, count):
        print(f"  {DIM}[summarizing {count} messages...]{RESET}", flush=True)

    def on_summary_done(self):
        print(f"  {DIM}[summary updated]{RESET}")

    def on_forced_think(self, tool_name, count):
        print(f"  [loop detected — forcing think]")

    def on_truncation_recovered(self, attempts):
        print(f"  {DIM}[truncation recovered after {attempts} continuation(s)]{RESET}")

    def on_truncation_failed(self, attempts):
        print(f"  {BOLD}[WARNING: response still truncated after {attempts} continuations]{RESET}")

    def on_context_recovery(self, auto):
        print(f"\n  {BOLD}[Context limit approaching — recovering...]{RESET}")

        if auto:
            self.log.info("Auto recovery: switching to dev mode")
            print(f"  {DIM}[auto-switching to dev mode]{RESET}")
            return "dev"

        print(f"\n  {BOLD}Context limit reached. Choose how to continue:{RESET}")
        print(f"    1. Continue in long mode (new server conversation, summary carried over)")
        print(f"    2. Switch to dev mode (prompt stuffing, unlimited context)")
        try:
            choice = input("  Choice [1/2, default=2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "2"

        if choice == "1":
            self.log.info("Recovery: continuing in long mode")
            print(f"  {DIM}[starting new long mode conversation]{RESET}")
            return "long"
        else:
            self.log.info("Recovery: switching to dev mode")
            print(f"  {DIM}[switching to dev mode]{RESET}")
            return "dev"


# ── Main agent loop ───────────────────────────────────────────────────

def run_agent_interactive(initial_prompt=None, auto=False, continue_mode=False,
                          mode="dev", model_override=None):
    """Interactive agent that maintains conversation history."""

    log, log_path, error_log_path = _setup_logger()

    # Create agent
    agent = Agent(config=_config, log=log, mode=mode)
    if model_override:
        agent.model = model_override

    # Load agent-specific tools from CWD/tools/
    agent_tools_dir = os.path.join(os.getcwd(), "tools")
    if os.path.isdir(agent_tools_dir):
        from tools import load_extra_tools
        load_extra_tools(agent_tools_dir)

    # Set up TUI or plain terminal callbacks
    use_tui = _HAS_TUI and not auto and sys.stdin.isatty()
    tui_session = None

    if use_tui:
        tui_session = TuiSession(agent)
        cb = TuiCallbacks(agent=agent, tui_session=tui_session, log=log, auto=auto)
    else:
        cb = TerminalCallbacks(agent=agent, log=log, auto=auto)
    agent.cb = cb

    mode_label = f" | Mode: {agent.mode}" if agent.mode != "dev" else ""
    print("="*60)
    print("llmbox - UCSB LLM Sandbox AI Assistant")
    print("="*60)
    print(f"Model: {agent.model} | Max turns: {agent.max_turns}{mode_label}")
    print(f"Session log: {log_path}")
    print(f"Error log: {error_log_path}")
    if use_tui:
        print("Ctrl+C to cancel | Ctrl+N for newline")
    else:
        print("Press Escape twice to cancel")
    print("Type /help for commands\n")

    # Health check
    status = StreamStatus()
    status.start("  Checking API health ")
    healthy = agent.health()
    status.first_token()
    status.finish()
    if healthy:
        print(f"  {DIM}[API healthy]{RESET}\n")
    else:
        print(f"  {BOLD}[WARNING: API health check failed]{RESET}\n")

    log.info("Session started | model=%s max_turns=%d mode=%s",
             agent.model, agent.max_turns, agent.mode)
    log.info("Tools registered: %s", [t["function"]["name"] for t in tools])

    # ── Continue mode: resume from checkpoint ──
    start_turn = 0

    if continue_mode:
        cp = _load_checkpoint()
        if cp:
            agent.conversation_history = cp.get("conversation_history", [])
            agent.summary_state = cp.get("summary_state", {"text": "", "up_to": 0})
            start_turn = cp.get("turn", 0)
            agent.initial_files = cp.get("initial_files")
            cp_mode = cp.get("mode", "dev")

            if cp_mode == "long":
                cp_conv_id = cp.get("conversation_id")
                if cp_conv_id:
                    try:
                        agent.api.get_conversation(cp_conv_id)
                        agent.mode = "long"
                        agent.conversation_id = cp_conv_id
                        agent.approx_char_usage = cp.get("approx_chars", 0)
                        log.info("CONTINUE: restored long mode conversation %s", cp_conv_id)
                        print(f"  {DIM}[restored long mode conversation]{RESET}")
                    except Exception:
                        log.warning("CONTINUE: long mode conversation lost — falling back to dev")
                        print(f"  {BOLD}[long mode conversation lost — falling back to dev mode]{RESET}")
                        agent.mode = "dev"

            log.info("CONTINUE: resuming from checkpoint (turn %d, %d messages, mode=%s)",
                     start_turn, len(agent.conversation_history), agent.mode)
            print(f"  [continuing from turn {start_turn} with {len(agent.conversation_history)} messages]")

            agent.conversation_history.append({"role": "user", "content":
                "Continue where you left off. The session was interrupted — "
                "pick up from your current phase and finish the cycle."})
            result = agent.run_continue(auto=auto, start_turn=start_turn)

            if auto:
                cleanup_temp_sessions()
                _delete_checkpoint()
                log.info("Session ended (continue mode) | %d messages",
                         len(agent.conversation_history))
                return
        else:
            print("  [no checkpoint found — starting fresh]")
            log.info("CONTINUE: no checkpoint found, starting fresh")

    if not continue_mode:
        _auto_increment_cycle(log)

    if not (continue_mode and start_turn > 0):
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
            agent.initial_files = f"{preamble}{header}\n{content}"
            print(f"  {DIM}{header}{RESET}")
            log.info("Auto-loaded agent.md (%d lines)", total)

    if initial_prompt and not (continue_mode and start_turn > 0):
        print(f"You: {initial_prompt}")
        expanded, files, err = _expand_file_refs(initial_prompt)
        if err:
            print(err)
            return
        if files:
            agent.initial_files = files

        log.info("USER: %s", expanded)
        result = agent.run(expanded, auto=auto)

        if auto:
            if result.status == "cancelled":
                print(f"\n{BOLD}[Agent paused — enter guidance, or press Enter to resume]{RESET}")
                try:
                    guidance = input("\nOperator: ").strip()
                except (EOFError, KeyboardInterrupt):
                    log.info("Session ended (operator cancelled)")
                    print()
                    return
                if guidance:
                    expanded_g, files_g, err_g = _expand_file_refs(guidance)
                    if err_g:
                        print(err_g)
                    else:
                        if files_g:
                            agent.initial_files = files_g
                        agent.run(expanded_g, auto=auto)
                else:
                    agent.run("Continue where you left off. Finish your current cycle.",
                              auto=auto)

            cleanup_temp_sessions()
            _delete_checkpoint()
            log.info("Session ended (auto mode) | %d messages",
                     len(agent.conversation_history))
            return

    # ── Interactive loop ──
    _interactive_loop(agent, log, tui_session)

    cleanup_temp_sessions()
    _delete_checkpoint()
    log.info("Session ended | %d messages in history", len(agent.conversation_history))


# ── Interactive loop ─────────────────────────────────────────────────

def _interactive_loop(agent, log, tui_session=None):
    """Main interactive input loop. Uses TuiSession if available, else raw input()."""
    while True:
        try:
            if tui_session:
                user_input = tui_session.prompt()
            else:
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

        # ── Commands ──
        if user_input.strip() == "/help":
            print(f"  {BOLD}Commands:{RESET}")
            print(f"    /help              Show this help message")
            print(f"    /clear             Reset conversation history")
            print(f"    /mode [dev|long]   Show or switch conversation mode")
            print(f"    /models            List available models")
            print(f"    /model <name>      Set model (or pick from menu)")
            print(f"    /context           Show context usage")
            print(f"    /verbose           Toggle verbose tool output")
            print(f"    /tools             Show recent tool history")
            print(f"    @file              Attach file contents to prompt")
            print(f"    exit, quit         End session")
            if tui_session:
                print(f"    Ctrl+C             Cancel current operation")
                print(f"    Ctrl+N             Insert newline")
            else:
                print(f"    Escape x2          Cancel current operation")
            print(f"\n  {BOLD}Modes:{RESET}")
            print(f"    dev                Prompt stuffing with rolling summary (default)")
            print(f"    long               Server-side conversation caching")
            continue

        if user_input.strip() == "/mode" or user_input.strip().startswith("/mode "):
            parts = user_input.strip().split(None, 1)
            if len(parts) == 1:
                if tui_session:
                    show_context(agent)
                else:
                    print(f"  Mode: {BOLD}{agent.mode}{RESET}")
                    if agent.mode == "dev":
                        print(f"    Messages: {len(agent.conversation_history)}")
                        print(f"    Summary: {len(agent.summary_state.get('text', ''))} chars")
                        print(f"    Context budget: {agent.max_context_chars:,} chars")
                    else:
                        print(f"    Conversation: {agent.conversation_id or '(none)'}")
                        limit = agent._get_context_limit_chars()
                        print(f"    Approx usage: {agent.approx_char_usage:,} / {limit:,} chars "
                              f"({agent.approx_char_usage * 100 // max(limit, 1)}%)")
            else:
                new_mode = parts[1].strip().lower()
                if new_mode not in ("dev", "long"):
                    print(f"  Unknown mode '{new_mode}'. Use 'dev' or 'long'.")
                elif new_mode == agent.mode:
                    print(f"  Already in {agent.mode} mode.")
                else:
                    print(f"  {DIM}[switching to {new_mode} mode...]{RESET}")
                    agent.switch_mode(new_mode)
                    print(f"  Switched to {BOLD}{new_mode}{RESET} mode (summary carried over)")
            continue

        if user_input.strip() == "/clear":
            agent.reset()
            agent.initial_files = None
            log, _, _ = _setup_logger()
            print(f"Conversation cleared.")
            continue

        if user_input.strip() == "/context":
            if _HAS_TUI:
                show_context(agent)
            else:
                print(f"  Mode: {BOLD}{agent.mode}{RESET}")
                print(f"  Messages: {len(agent.conversation_history)}")
                print(f"  Summary: {len(agent.summary_state.get('text', ''))} chars")
            continue

        if user_input.strip() == "/verbose":
            if tui_session:
                tui_session.verbose = not tui_session.verbose
                state = "on" if tui_session.verbose else "off"
                print(f"  Verbose tool output: {BOLD}{state}{RESET}")
            else:
                print(f"  /verbose requires prompt_toolkit TUI")
            continue

        if user_input.strip() == "/tools":
            if tui_session:
                show_tools(tui_session)
            else:
                print(f"  /tools requires prompt_toolkit TUI")
            continue

        if user_input.strip() == "/models":
            models = agent.list_models()
            if models:
                print(f"  Current: {BOLD}{agent.model}{RESET}")
                print(f"  Available:")
                for m in models:
                    marker = " *" if m == agent.model else ""
                    print(f"    {m}{marker}")
            else:
                print("  Could not fetch model list")
            continue

        if user_input.strip().startswith("/model"):
            parts = user_input.strip().split(None, 1)
            if len(parts) == 2:
                agent.model = parts[1]
                print(f"  Model set to: {BOLD}{agent.model}{RESET}")
            else:
                models = agent.list_models()
                if not models:
                    print("  Could not fetch model list. Usage: /model <model-name>")
                else:
                    print(f"  Current: {BOLD}{agent.model}{RESET}")
                    for i, m in enumerate(models, 1):
                        marker = " (current)" if m == agent.model else ""
                        print(f"    {i}. {m}{marker}")
                    try:
                        choice = input("  Select model number (or Enter to cancel): ").strip()
                    except (EOFError, KeyboardInterrupt):
                        print()
                        continue
                    if choice.isdigit() and 1 <= int(choice) <= len(models):
                        agent.model = models[int(choice) - 1]
                        print(f"  Model set to: {BOLD}{agent.model}{RESET}")
                    elif choice:
                        print("  Invalid selection")
            continue

        # ── Regular prompt ──
        expanded, files, err = _expand_file_refs(user_input)
        if err:
            print(err)
            continue
        if files:
            agent.initial_files = files

        log.info("USER: %s", expanded)
        try:
            agent.run(expanded, auto=False)
        except KeyboardInterrupt:
            print(f"\n{BOLD}[Cancelled]{RESET}")


# ── Entry point ──────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="llmbox - UCSB LLM Sandbox AI Assistant")
    parser.add_argument("-a", "--auto", action="store_true",
                        help="Automation mode: run prompt and exit (no interactive loop)")
    parser.add_argument("-c", "--continue", dest="continue_mode", action="store_true",
                        help="Continue from last checkpoint")
    parser.add_argument("-r", "--repeat", type=int, nargs="?", const=0, default=None,
                        help="Repeat N times (fresh each run). 0 or omit = indefinite. Implies -a.")
    parser.add_argument("-m", "--model", default=None,
                        help="Override model (e.g. claude-v4.5-opus)")
    parser.add_argument("--mode", choices=["dev", "long"], default="dev",
                        help="Conversation mode: dev (prompt stuffing) or long (server-side caching)")
    parser.add_argument("prompt", nargs="*", help="Initial prompt")
    args = parser.parse_args()

    initial_prompt = " ".join(args.prompt).strip() or None

    if args.continue_mode:
        run_agent_interactive(initial_prompt=initial_prompt, auto=True,
                              continue_mode=True, mode=args.mode,
                              model_override=args.model)
    elif args.repeat is not None:
        n = args.repeat
        run = 0
        try:
            while n == 0 or run < n:
                run += 1
                label = f"run {run}/{n}" if n > 0 else f"run {run}"
                print(f"\n{'='*60}\n{label}\n{'='*60}")
                run_agent_interactive(initial_prompt=initial_prompt, auto=True,
                                      mode=args.mode, model_override=args.model)
        except KeyboardInterrupt:
            print(f"\n\nStopped after {run} run(s).")
    else:
        run_agent_interactive(initial_prompt=initial_prompt, auto=args.auto,
                              mode=args.mode, model_override=args.model)


if __name__ == "__main__":
    main()
