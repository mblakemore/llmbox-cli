"""
TUI — prompt_toolkit-based interactive interface for llmbox.

Provides TuiSession (rich input with multiline, autocomplete, toolbar) and
TuiCallbacks (compact tool output, /verbose toggle, /tools history, /context).
"""

import os
import sys

from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.formatted_text import FormattedText, ANSI, HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style

from cancel import set_tui_mode
from llmbox_lib import NullCallbacks
from spinner import StreamStatus

# ── Style ─────────────────────────────────────────────────────────────

_STYLE = Style.from_dict({
    "bottom-toolbar":      "bg:#005f87 #cccccc",
    "bottom-toolbar.text": "#aaaacc",
    "prompt":              "bold",
    "reasoning":           "#6688cc",
    "dim":                 "#888888",
    "error":               "#cc4444 bold",
    "success":             "#44aa44",
})

DIM = "\033[90m"
BOLD = "\033[1m"
BLUE = "\033[34m"
RED = "\033[31m"
GREEN = "\033[32m"
RESET = "\033[0m"


def _print(text=""):
    """Print text, same as print(). Kept as alias for llmbox.py import compatibility."""
    print(text)

# ── Commands list ─────────────────────────────────────────────────────

_COMMANDS = [
    ("/help", "Show help"),
    ("/clear", "Reset conversation"),
    ("/mode", "Show or switch mode"),
    ("/model", "Show or switch model"),
    ("/models", "List available models"),
    ("/context", "Show context usage"),
    ("/verbose", "Toggle verbose tool output"),
    ("/tools", "Show recent tool history"),
]


# ── Completer ─────────────────────────────────────────────────────────

class LlmboxCompleter(Completer):
    """Autocomplete / commands and @ file paths."""

    def __init__(self):
        self._path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor

        # Slash commands: complete when line starts with /
        if text.lstrip().startswith("/"):
            word = text.lstrip()
            for cmd, desc in _COMMANDS:
                if cmd.startswith(word):
                    yield Completion(
                        cmd, start_position=-len(word),
                        display_meta=desc,
                    )
            return

        # @ file paths: find the last @ and complete the path after it
        at_pos = text.rfind("@")
        if at_pos >= 0:
            # Only trigger if @ is at start or preceded by whitespace
            if at_pos == 0 or text[at_pos - 1] in (" ", "\t", "\n"):
                path_part = text[at_pos + 1:]
                # Create a sub-document for the path portion
                from prompt_toolkit.document import Document
                sub_doc = Document(path_part, len(path_part))
                for completion in self._path_completer.get_completions(sub_doc, complete_event):
                    yield completion


# ── Key bindings ──────────────────────────────────────────────────────

def _build_key_bindings():
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        """Enter submits the prompt."""
        event.current_buffer.validate_and_handle()

    @kb.add("c-n")
    def _newline_ctrl_n(event):
        """Ctrl+N inserts a newline."""
        event.current_buffer.insert_text("\n")

    return kb


# ── TuiSession ────────────────────────────────────────────────────────

class TuiSession:
    """Rich input session with multiline editing, completions, and toolbar."""

    def __init__(self, agent):
        self.agent = agent
        self._verbose = False
        self._tool_history = []

        set_tui_mode()

        self._session = PromptSession(
            message="\nYou: ",
            multiline=False,
            key_bindings=_build_key_bindings(),
            completer=LlmboxCompleter(),
            complete_while_typing=False,
            bottom_toolbar=self._toolbar,
            style=_STYLE,
            color_depth=ColorDepth.TRUE_COLOR,
            enable_history_search=True,
        )

    def _toolbar(self):
        agent = self.agent
        cwd = os.path.basename(os.getcwd()) or "/"
        mode = agent.mode
        model = agent.model
        msgs = len(agent.conversation_history)

        # Context usage
        if mode == "long" and agent.conversation_id:
            limit = agent._get_context_limit_chars()
            usage = agent.approx_char_usage
            pct = usage * 100 // max(limit, 1)
            ctx = f"~{pct}%"
        elif mode == "dev":
            limit = agent.max_context_chars
            # Rough estimate: sum of conversation history
            used = sum(len(m.get("content", "") or "") for m in agent.conversation_history)
            used += len(agent.summary_state.get("text", ""))
            pct = used * 100 // max(limit, 1)
            ctx = f"~{min(pct, 100)}%"
        else:
            ctx = "—"

        verbose = " | verbose" if self._verbose else ""
        text = f" [{cwd}] | {model} | {mode} | {msgs} msgs | ctx {ctx}{verbose} "
        # Pad to terminal width so background fills the entire row
        try:
            width = os.get_terminal_size().columns
        except OSError:
            width = 80
        text = text.ljust(width)
        return HTML(
            f'<style fg="#003660" bg="#FFD100">{text}</style>'
        )

    def prompt(self):
        """Read user input. Returns stripped string. Raises EOFError/KeyboardInterrupt."""
        return self._session.prompt().strip()

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        self._verbose = value

    @property
    def tool_history(self):
        return self._tool_history

    def patch_stdout(self):
        """No-op — prompt is blocking so spinner output can go to real stdout."""
        return None


# ── TuiCallbacks ──────────────────────────────────────────────────────

class TuiCallbacks(NullCallbacks):
    """Callbacks that render output via prompt_toolkit, with compact tool display."""

    def __init__(self, agent, tui_session, log, auto=False):
        self.agent = agent
        self.tui = tui_session
        self.log = log
        self.auto = auto
        self._status = None
        self._tool_status = None

    def check_cancelled(self):
        # In TUI mode, cancellation is via KeyboardInterrupt
        # which prompt_toolkit handles; during API waits, the
        # cancel_check callback in bedrock_api raises it
        pass

    def on_api_start(self, label):
        self._status = StreamStatus()
        self._status.start(label)

    def on_api_response(self):
        if self._status:
            self._status.first_token()

    def on_api_done(self):
        if self._status:
            self._status.finish()
            self._status = None

    def on_assistant_text(self, text, reasoning):
        if reasoning:
            _print(f"{BLUE}[Reasoning]\n{reasoning}{RESET}\n")
        if text:
            _print(text)

    def on_tool_batch_start(self, count):
        if self.tui.verbose:
            _print(f"\n{DIM}Executing {count} tool call(s)...")
        # No cancellable needed — Ctrl+C works naturally

    def on_tool_start(self, name, args):
        if self.tui.verbose and name not in {"think"}:
            self._tool_status = StreamStatus()
            self._tool_status.start(f"  -> {name} ")
        else:
            self._tool_status = None

    def on_tool_result(self, name, args, result, is_error):
        if self._tool_status:
            self._tool_status.first_token()
            self._tool_status.finish()
            self._tool_status = None

        # Store in history
        self.tui.tool_history.append({
            "name": name,
            "args": args,
            "result": result,
            "is_error": is_error,
        })
        # Keep last 50
        if len(self.tui.tool_history) > 50:
            self.tui.tool_history[:] = self.tui.tool_history[-50:]

        if self.tui.verbose:
            # Full output (like TerminalCallbacks)
            args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())
            _print(f"  -> {name}({args_str})")
            _print(f"    Result: {result}{RESET}")
        else:
            # Compact one-liner
            key_args = _compact_args(name, args)
            if is_error:
                err_msg = result.split("\n")[0][:80]
                _print(f"  {DIM}-> {name}({key_args}) {RED}✗{RESET} {DIM}{err_msg}{RESET}")
            else:
                _print(f"  {DIM}-> {name}({key_args}) {GREEN}✓{RESET}")

    def on_turn_end(self, turn, turn_result):
        from llmbox import _save_checkpoint
        _save_checkpoint(self.agent, turn)

    def on_summary_start(self, count):
        _print(f"  {DIM}[summarizing {count} messages...]{RESET}")

    def on_summary_done(self):
        _print(f"  {DIM}[summary updated]{RESET}")

    def on_forced_think(self, tool_name, count):
        _print(f"  [loop detected — forcing think]")

    def on_truncation_recovered(self, attempts):
        _print(f"  {DIM}[truncation recovered after {attempts} continuation(s)]{RESET}")

    def on_truncation_failed(self, attempts):
        _print(f"  {BOLD}[WARNING: response still truncated after {attempts} continuations]{RESET}")

    def on_context_recovery(self, auto):
        _print(f"\n  {BOLD}[Context limit approaching — recovering...]{RESET}")

        if auto:
            self.log.info("Auto recovery: switching to dev mode")
            _print(f"  {DIM}[auto-switching to dev mode]{RESET}")
            return "dev"

        _print(f"\n  {BOLD}Context limit reached. Choose how to continue:{RESET}")
        print(f"    1. Continue in long mode (new server conversation, summary carried over)")
        print(f"    2. Switch to dev mode (prompt stuffing, unlimited context)")
        try:
            choice = input("  Choice [1/2, default=2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            choice = "2"

        if choice == "1":
            self.log.info("Recovery: continuing in long mode")
            _print(f"  {DIM}[starting new long mode conversation]{RESET}")
            return "long"
        else:
            self.log.info("Recovery: switching to dev mode")
            _print(f"  {DIM}[switching to dev mode]{RESET}")
            return "dev"


# ── /context command ──────────────────────────────────────────────────

def show_context(agent):
    """Print context usage display."""
    mode = agent.mode
    model = agent.model

    summary_len = len(agent.summary_state.get("text", ""))
    msgs = len(agent.conversation_history)
    summary_part = f" | Summary: {summary_len:,} chars" if summary_len else ""

    _print(f"\n  {BOLD}Context Usage{RESET}")
    _print(f"  Mode: {mode} | Model: {model}")
    _print(f"  Messages: {msgs}{summary_part}")

    if mode == "long" and agent.conversation_id:
        limit = agent._get_context_limit_chars()
        usage = agent.approx_char_usage
        pct = min(usage * 100 // max(limit, 1), 100)
        _print(f"  Conversation: {agent.conversation_id}")
        _print(f"  Context: ~{usage:,} / {limit:,} chars ({pct}%)")
        _print_bar(pct)
    else:
        limit = agent.max_context_chars
        used = sum(len(m.get("content", "") or "") for m in agent.conversation_history)
        used += summary_len
        pct = min(used * 100 // max(limit, 1), 100)
        _print(f"  Context: ~{used:,} / {limit:,} chars ({pct}%)")
        _print_bar(pct)
    print()


def _print_bar(pct):
    """Print a progress bar."""
    width = 30
    filled = int(width * pct / 100)
    bar = "█" * filled + "░" * (width - filled)
    color = GREEN if pct < 60 else ("\033[33m" if pct < 80 else RED)
    _print(f"  {color}[{bar}] {pct}%{RESET}")


# ── /tools command ────────────────────────────────────────────────────

def show_tools(tui_session):
    """Show recent tool execution history."""
    history = tui_session.tool_history
    if not history:
        _print(f"  {DIM}No tool calls yet.{RESET}")
        return

    _print(f"\n  {BOLD}Recent Tool Calls ({len(history)}){RESET}\n")
    for i, entry in enumerate(history[-20:], 1):
        name = entry["name"]
        args = entry["args"]
        result = entry["result"]
        is_error = entry["is_error"]

        args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
        status = f"{RED}✗{RESET}" if is_error else f"{GREEN}✓{RESET}"
        _print(f"  {DIM}{i:2d}.{RESET} {name}({args_str}) {status}")

        # Show truncated result
        lines = result.strip().split("\n")
        if len(lines) <= 3:
            for line in lines:
                _print(f"      {DIM}{line}{RESET}")
        else:
            for line in lines[:2]:
                _print(f"      {DIM}{line}{RESET}")
            _print(f"      {DIM}... ({len(lines) - 2} more lines){RESET}")
    print()


# ── Helpers ───────────────────────────────────────────────────────────

def _compact_args(name, args):
    """Pick the most informative arg(s) for compact display."""
    if not args:
        return ""

    # Tool-specific key arg selection
    _KEY_ARGS = {
        "file": ["operation", "path"],
        "exec_command": ["command"],
        "search_files": ["pattern"],
        "web_fetch": ["url"],
        "think": [],
        "task_tracker": ["operation"],
        "read_pdf": ["path"],
        "sleep": ["seconds"],
    }

    keys = _KEY_ARGS.get(name)
    if keys is not None:
        parts = []
        for k in keys:
            if k in args:
                v = str(args[k])
                if len(v) > 60:
                    v = v[:57] + "..."
                parts.append(v)
        return ", ".join(parts) if parts else ""

    # Fallback: first arg, truncated
    k, v = next(iter(args.items()))
    v = str(v)
    if len(v) > 60:
        v = v[:57] + "..."
    return f"{k}={v}"
