"""Execute shell commands via subprocess.

Each command runs in a fresh bash shell rooted at the agent's working
directory (os.getcwd()).  Compound commands like 'cd ../e1 && git log'
work within a single call but do NOT affect future calls — every
invocation starts from the agent's home directory.

Sessions only matter for background processes — they track the Popen
handle and accumulated output so the agent can poll later.
"""

import atexit
import hashlib
import os
import re
import secrets
import subprocess
import threading

# Detect when the agent is trying to create/overwrite files via shell.
# Must avoid false positives on: 2>&1, pipes, tee for output capture, appending to logs.
def _is_file_write_command(command):
    """Check if a command is primarily trying to write files via shell."""
    # Heredoc file creation: cat > file << 'EOF' or similar
    if re.search(r'<<\s*[\'"]?\w+[\'"]?', command) and re.search(r'>\s*\S+\.(?:py|json|md|txt|sh|yaml|yml|toml|cfg)\b', command):
        return True
    # Command starts with cat/echo/printf redirected to a file (the primary action is writing)
    if re.search(r'^\s*(?:cat|echo|printf)\s+.*?[^2]>\s*\S', command):
        return True
    return False


# Max temporary sessions per agent
_MAX_TEMP_SESSIONS = 4

# Background sessions: {session_id: {"bg_proc": Popen|None, "bg_output": str}}
_sessions = {}
_main_session_id = None
_temp_session_ids = []


def _derive_main_session():
    """Derive a stable main session name from the agent's working directory."""
    cwd = os.getcwd()
    agent_name = os.path.basename(cwd)
    path_hash = hashlib.md5(cwd.encode()).hexdigest()[:6]
    return f"agent-{agent_name}-{path_hash}"


def _get_or_create_session(session_id=None, new_session=False):
    """Get an existing session or create a new one. Returns (session_id, error)."""
    global _main_session_id

    if session_id:
        if session_id not in _sessions:
            return None, f"Error: session '{session_id}' does not exist"
        return session_id, None

    if new_session:
        # Clean up finished temp sessions
        _temp_session_ids[:] = [s for s in _temp_session_ids if s in _sessions]
        if len(_temp_session_ids) >= _MAX_TEMP_SESSIONS:
            return None, (
                f"Error: temporary session limit reached ({_MAX_TEMP_SESSIONS}). "
                f"Active temp sessions: {', '.join(_temp_session_ids)}. "
                f"Use an existing session_id or wait for one to be cleaned up."
            )
        sid = f"agent-tmp-{secrets.token_hex(4)}"
        _sessions[sid] = {"bg_proc": None, "bg_output": ""}
        _temp_session_ids.append(sid)
        return sid, None

    # Main session
    if _main_session_id and _main_session_id in _sessions:
        return _main_session_id, None

    sid = _derive_main_session()
    _sessions[sid] = {"bg_proc": None, "bg_output": ""}
    _main_session_id = sid
    return sid, None


def _read_bg_output(proc, session):
    """Background thread: read process output incrementally."""
    parts = []
    try:
        for line in proc.stdout:
            parts.append(line)
            session["bg_output"] = "".join(parts)
        proc.wait()
    except Exception as e:
        parts.append(f"\nError reading output: {e}\n")
    session["bg_output"] = "".join(parts)


def cleanup_temp_sessions():
    """Kill all temporary sessions and their background processes."""
    for sid in _temp_session_ids[:]:
        session = _sessions.pop(sid, None)
        if session and session.get("bg_proc"):
            try:
                session["bg_proc"].kill()
            except Exception:
                pass
    _temp_session_ids.clear()


# Clean up if the process exits unexpectedly
atexit.register(cleanup_temp_sessions)


def fn(command: str = "", session_id: str = "", timeout: float = 120,
       background: bool = False, new_session: bool = False) -> str:
    """Execute a shell command in the agent's working directory.

    Args:
        command: Shell command to execute. If empty, checks on a background session.
        session_id: Existing session to reuse (only for background process polling).
        timeout: Max seconds to wait (default 120). LLM-calling scripts may need 300+.
        background: If true, start the command and return immediately.
        new_session: If true, create a new temporary session for parallel work.
    """
    if not command and not session_id:
        return "Error: at least one of 'command' or 'session_id' is required"

    sid, err = _get_or_create_session(session_id, new_session)
    if err:
        return err
    session = _sessions[sid]

    # ── Polling (no command) ──────────────────────────────────────────
    if not command:
        bg = session.get("bg_proc")
        if bg:
            output = session.get("bg_output", "")
            if bg.poll() is not None:
                rc = bg.returncode
                session["bg_proc"] = None
                return f"[session: {sid}] exit={rc} (background process finished)\n{output}"
            else:
                # Show tail of output so far
                tail = output[-4000:] if len(output) > 4000 else output
                return f"[session: {sid}] (still running)\n{tail}"
        return f"[session: {sid}] (idle)"

    # ── Guards ────────────────────────────────────────────────────────

    # Every command runs from the agent's home directory
    home_cwd = os.getcwd()

    # Block cd to paths outside the repo tree.
    # Relative cd (cd ../shared && ...) is fine — only block absolute paths and ~ expansion
    # that leave the repo.
    cd_match = re.match(r'^cd\s+(\S+)\s*&&\s*(.+)', command)
    if cd_match:
        target_dir = cd_match.group(1)
        # Expand ~ so we can check the resolved path
        expanded = os.path.expanduser(target_dir)
        # Only check absolute paths (relative ones are fine — they stay in the repo)
        if os.path.isabs(expanded):
            # Allow cd within the repo tree (parent of home_cwd holds all worktrees)
            repo_root = os.path.dirname(home_cwd)  # e.g. /droid/repos/agent-triad-ex1
            if not expanded.rstrip('/').startswith(repo_root.rstrip('/')):
                return (
                    f"Error: You are trying to cd to '{target_dir}' which is outside "
                    f"your repo tree ('{repo_root}'). Your working directory is "
                    f"'{home_cwd}'. Use relative paths — the session is already in "
                    f"the correct directory."
                )

    # Redirect file writes to the file tool for read-before-write safety
    if _is_file_write_command(command):
        return (
            "Error: Do not write files via shell commands (cat >, echo >, heredoc). "
            "Use the 'file' tool with action='write' instead — it tracks reads and "
            "prevents accidental overwrites of files you haven't reviewed."
        )

    # ── Background execution ──────────────────────────────────────────
    if background:
        try:
            proc = subprocess.Popen(
                ['bash', '-c', f'{command} 2>&1'],
                cwd=home_cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as e:
            return f"Error starting background command: {e}"
        session["bg_proc"] = proc
        session["bg_output"] = ""
        t = threading.Thread(target=_read_bg_output, args=(proc, session), daemon=True)
        t.start()
        return f"[session: {sid}]\nCommand started in background. Poll with session_id to check output."

    # ── Foreground execution ──────────────────────────────────────────
    try:
        result = subprocess.run(
            ['bash', '-c', f'{command} 2>&1'],
            cwd=home_cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (
            f"[session: {sid}] (timed out after {timeout}s)\n"
            f"The command is no longer running. Try a shorter operation or "
            f"use background=true for long-running commands."
        )
    except Exception as e:
        return f"Error running command: {e}"

    output = result.stdout.rstrip('\n')
    return f"[session: {sid}] exit={result.returncode}\n{output}"


definition = {
    "type": "function",
    "function": {
        "name": "exec_command",
        "description": (
            "Execute a shell command. "
            "Every command starts from the agent's home directory — "
            "use 'cd ../e1 && cmd' for one-off commands in other directories. "
            "Set new_session=true to create a temporary session for background work "
            "(e.g., running a server). "
            "Temp sessions are cleaned up at end of cycle. "
            "Commands that call the LLM or do heavy computation may take minutes — "
            "the default timeout is 120s. For long-running commands, use "
            "background=true and poll with session_id to check output. "
            "WARNING: Do NOT use this tool to write files (cat >, echo >, heredocs). "
            "Use the 'file' tool instead — it tracks reads and prevents accidental overwrites."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute. If empty, checks on a background session.",
                    "default": "",
                },
                "session_id": {
                    "type": "string",
                    "description": "Existing session ID for polling a background process.",
                    "default": "",
                },
                "timeout": {
                    "type": "number",
                    "description": "Max seconds to wait for the command to finish (default 120). LLM-calling scripts may need 300+.",
                    "default": 120,
                },
                "background": {
                    "type": "boolean",
                    "description": "If true, start the command and return immediately without waiting. Poll with session_id later.",
                    "default": False,
                },
                "new_session": {
                    "type": "boolean",
                    "description": "If true, create a new temporary session instead of using the main one. Use for parallel tasks like running a server.",
                    "default": False,
                },
            },
            "required": [],
        },
    },
}
