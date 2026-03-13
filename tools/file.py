"""Unified file operations tool — read, write, append, delete, list."""

import os
from pathlib import Path

# Paths that would create suspiciously deep nesting are probably mistakes
_MAX_NEW_DIRS = 3

# Track files that have been read this session — writes to existing unread files
# are blocked to prevent blind overwrites.
_read_files = set()


def _resolve_path(path):
    """Resolve path, stripping accidental cwd prefix duplications."""
    p = Path(path)
    if not p.is_absolute():
        cwd = Path.cwd()
        # Detect when the agent passes something like "droid/repos/project/e2/file"
        # which duplicates the cwd structure. Try to resolve it.
        try:
            resolved = (cwd / p).resolve()
            # Check if the relative path starts with parts of the cwd
            cwd_str = str(cwd) + "/"
            if path.startswith(str(cwd)[1:]):  # e.g. "droid/repos/..." matching "/droid/repos/..."
                # Strip the cwd prefix from the relative path
                stripped = "/" + path
                return Path(stripped)
        except (ValueError, OSError):
            pass
    return p


def fn(action: str, path: str = ".", content: str = "", start_line: int = 0, end_line: int = 0) -> str:
    """Perform file operations.

    Args:
        action: One of "read", "write", "append", "delete", "list".
        path: File or directory path.
        content: Content for write/append actions.
        start_line: For read: first line (1-indexed, default: all). For write: first line to replace.
        end_line: For read: last line (1-indexed, default: all). For write: last line to replace (0 = to end of file).
    """
    try:
        resolved = str(_resolve_path(path))
        if action == "read":
            return _read(resolved, start_line, end_line)
        elif action == "write":
            return _write(resolved, content, start_line, end_line)
        elif action == "append":
            return _append(resolved, content)
        elif action == "delete":
            return _delete(resolved)
        elif action == "list":
            return _list(resolved)
        else:
            return f"Error: unknown action '{action}'. Use: read, write, append, delete, list."
    except Exception as e:
        return f"Error ({action}): {e}"


def _read(path, start_line, end_line):
    p = Path(path)
    if not p.exists():
        return f"Error: '{path}' does not exist"
    if p.is_dir():
        return f"Error: '{path}' is a directory. Use action='list' instead."

    with open(p, 'r') as f:
        lines = f.readlines()

    total = len(lines)
    # Default: read entire file
    s = max(1, start_line) if start_line > 0 else 1
    e = min(total, end_line) if end_line > 0 else total

    selected = lines[s - 1:e]
    # Number lines for easy reference
    numbered = "".join(f"{i:4d}  {line}" for i, line in enumerate(selected, s))

    info = f"[{path}: lines {s}-{e} of {total}]\n"
    if e < total:
        info += f"[Use start_line={e + 1} to continue reading]\n"

    # Track that this file has been read
    _read_files.add(str(p.resolve()))

    return info + numbered


def _write(path, content, start_line, end_line):
    p = Path(path)

    # If file exists but hasn't been read this session, force a read first
    if p.exists() and str(p.resolve()) not in _read_files:
        return (f"Error: '{path}' exists but has not been read this session. "
                f"You must read the file first (action='read') before writing to it, "
                f"so you can verify your changes are accurate and won't overwrite important content.")

    # Line-range replacement
    if start_line > 0 or end_line > 0:
        if not p.exists():
            return f"Error: cannot replace lines — '{path}' does not exist"
        if start_line <= 0:
            start_line = 1

        with open(p, 'r') as f:
            lines = f.readlines()

        # end_line=0 means "to end of file"
        if end_line <= 0:
            end_line = len(lines)

        if start_line > end_line:
            return f"Error: start_line ({start_line}) > end_line ({end_line})"
        if start_line > len(lines):
            return f"Error: start_line ({start_line}) exceeds file length ({len(lines)} lines)"
        if end_line > len(lines):
            return f"Error: end_line ({end_line}) exceeds file length ({len(lines)} lines)"

        start_idx = start_line - 1
        end_idx = end_line

        new_lines = content.splitlines(True) if content else []
        if new_lines and not new_lines[-1].endswith("\n"):
            if end_idx < len(lines) or (lines and lines[-1].endswith("\n")):
                new_lines[-1] += "\n"

        lines[start_idx:end_idx] = new_lines

        with open(p, 'w') as f:
            f.writelines(lines)

        old_count = end_line - start_line + 1
        new_count = len(new_lines)
        delta = new_count - old_count
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        return (f"Replaced lines {start_line}-{end_line} in '{path}' "
                f"({old_count} → {new_count} lines, {delta_str}). "
                f"File now has {len(lines)} lines.")

    # Full file write
    # Guard against creating suspiciously deep directory trees
    dirs_to_create = []
    check = p.parent
    while check and not check.exists():
        dirs_to_create.append(check)
        check = check.parent
    if len(dirs_to_create) > _MAX_NEW_DIRS:
        return (f"Error: writing '{path}' would create {len(dirs_to_create)} nested directories. "
                f"This usually means the path is wrong. Use a relative path from your working directory "
                f"(e.g. 'state/file.json' not '/droid/repos/.../state/file.json').")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        f.write(content)
    return f"Wrote '{path}' ({len(content)} chars)"


def _append(path, content):
    p = Path(path)
    if p.suffix.lower() == '.json':
        return (f"Error: cannot append to JSON file '{path}' — breaks structure. "
                f"Use action='write' with full contents instead.")
    with open(p, 'a') as f:
        f.write(content)
    return f"Appended to '{path}' ({len(content)} chars)"


def _delete(path):
    p = Path(path)
    if not p.exists():
        return f"Error: '{path}' does not exist"
    if p.is_dir():
        if any(p.iterdir()):
            return f"Error: directory '{path}' is not empty"
        p.rmdir()
        return f"Deleted empty directory '{path}'"
    p.unlink()
    return f"Deleted '{path}'"


def _list(path):
    p = Path(path)
    if not p.exists():
        return f"Error: '{path}' does not exist"
    if not p.is_dir():
        return f"Error: '{path}' is not a directory"
    entries = sorted(p.iterdir())
    if not entries:
        return "(empty directory)"
    parts = []
    for e in entries:
        suffix = "/" if e.is_dir() else ""
        parts.append(f"{e.name}{suffix}")
    return "\n".join(parts)


definition = {
    "type": "function",
    "function": {
        "name": "file",
        "description": (
            "Unified file operations — the ONLY reliable way to read and write files.\n"
            "Always use this tool instead of shell commands (cat, echo, heredocs) for file I/O.\n"
            "Actions:\n"
            "- read: Read file contents (with optional line range). Returns numbered lines.\n"
            "- write: Create/overwrite a file, or replace a line range (set start_line, end_line defaults to end of file).\n"
            "- append: Append content to end of file (not for JSON files).\n"
            "- delete: Delete a file or empty directory.\n"
            "- list: List directory contents.\n"
            "IMPORTANT: You MUST read an existing file before writing to it. "
            "The tool will reject writes to files you haven't read this session."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "write", "append", "delete", "list"],
                    "description": "The operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path. For 'list', defaults to current directory.",
                },
                "content": {
                    "type": "string",
                    "description": "Content for write/append. For line-range write, replaces the specified lines.",
                },
                "start_line": {
                    "type": "integer",
                    "description": "For read: first line to return (1-indexed). For write: first line to replace. Omit or 0 for full write.",
                },
                "end_line": {
                    "type": "integer",
                    "description": "For read: last line to return (1-indexed). For write: last line to replace. Omit or 0 = to end of file.",
                },
            },
            "required": ["action", "path"],
        },
    },
}
