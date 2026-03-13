"""Task tracker tool — persistent task management via state/tasks.json."""

import json
from datetime import datetime
from pathlib import Path


_TASKS_FILE = "state/tasks.json"


def _load_tasks():
    p = Path(_TASKS_FILE)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, IOError):
        return []


def _save_tasks(tasks):
    p = Path(_TASKS_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(tasks, indent=2) + "\n")


def _next_id(tasks):
    return max((t.get("id", 0) for t in tasks), default=0) + 1


def fn(action: str, description: str = "", task_id: int = 0, status: str = "") -> str:
    """Manage persistent tasks.

    Args:
        action: One of "add", "done", "update", "drop", "list".
        description: Task description (for add) or note (for update).
        task_id: Task ID (for done, update, drop).
        status: New status string (for update). Common: "in_progress", "blocked", "deferred".
    """
    tasks = _load_tasks()

    # Treat update with completed/done status as the "done" action
    if action == "update" and status in ("completed", "done"):
        action = "done"

    # Auto-resolve task_id: if missing, try to find a unique open task
    if action in ("done", "update", "drop") and task_id <= 0:
        open_tasks = [t for t in tasks if t["status"] not in ("done", "completed")]
        if len(open_tasks) == 1:
            task_id = open_tasks[0]["id"]
        elif description:
            # Try fuzzy match by description substring
            desc_lower = description.lower()
            matches = [t for t in open_tasks if desc_lower in t["description"].lower()
                       or t["description"].lower() in desc_lower]
            if len(matches) == 1:
                task_id = matches[0]["id"]

    if action == "add":
        if not description:
            return "Error: description required for 'add'"
        task = {
            "id": _next_id(tasks),
            "description": description,
            "status": "open",
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        tasks.append(task)
        _save_tasks(tasks)
        return f"Added task #{task['id']}: {description}"

    elif action == "done":
        if task_id <= 0:
            available = [f"#{t['id']} ({t['status']}): {t['description']}" for t in tasks if t["status"] != "done"]
            return f"Error: task_id required for 'done'. Example: task_tracker(action=\"done\", task_id=1)\nOpen tasks:\n" + ("\n".join(available) if available else "(none)")
        for t in tasks:
            if t["id"] == task_id:
                t["status"] = "done"
                t["completed"] = datetime.now().isoformat(timespec="seconds")
                if description:
                    t["note"] = description
                _save_tasks(tasks)
                return f"Completed task #{task_id}: {t['description']}"
        return f"Error: task #{task_id} not found"

    elif action == "update":
        if task_id <= 0:
            available = [f"#{t['id']} ({t['status']}): {t['description']}" for t in tasks if t["status"] != "done"]
            return f"Error: task_id required for 'update'. Example: task_tracker(action=\"update\", task_id=1, status=\"in_progress\")\nOpen tasks:\n" + ("\n".join(available) if available else "(none)")
        for t in tasks:
            if t["id"] == task_id:
                if status:
                    t["status"] = status
                if description:
                    t["note"] = description
                _save_tasks(tasks)
                return f"Updated task #{task_id}: status={t['status']}"
        return f"Error: task #{task_id} not found"

    elif action == "drop":
        if task_id <= 0:
            return "Error: task_id required for 'drop'"
        for i, t in enumerate(tasks):
            if t["id"] == task_id:
                removed = tasks.pop(i)
                _save_tasks(tasks)
                return f"Dropped task #{task_id}: {removed['description']}"
        return f"Error: task #{task_id} not found"

    elif action == "list":
        if not tasks:
            return "No tasks."
        lines = []
        for t in tasks:
            marker = "x" if t["status"] == "done" else " "
            line = f"[{marker}] #{t['id']} ({t['status']}): {t['description']}"
            if t.get("note"):
                line += f" — {t['note']}"
            lines.append(line)
        open_count = sum(1 for t in tasks if t["status"] != "done")
        done_count = sum(1 for t in tasks if t["status"] == "done")
        lines.append(f"\n{open_count} open, {done_count} done")
        return "\n".join(lines)

    else:
        return f"Error: unknown action '{action}'. Use: add, done, update, drop, list."


definition = {
    "type": "function",
    "function": {
        "name": "task_tracker",
        "description": (
            "Manage persistent tasks stored in state/tasks.json. "
            "Use this to track work items across turns and cycles. "
            "Actions: add (new task), done (complete), update (change status/note), "
            "drop (remove), list (show all). "
            "Tasks persist across context window resets and conversation summaries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "done", "update", "drop", "list"],
                    "description": "The operation to perform.",
                },
                "description": {
                    "type": "string",
                    "description": "Task description (for add) or note (for update/done).",
                },
                "task_id": {
                    "type": "integer",
                    "description": "Task ID (for done, update, drop).",
                },
                "status": {
                    "type": "string",
                    "description": "New status (for update). Common: 'in_progress', 'blocked', 'deferred'.",
                },
            },
            "required": ["action"],
        },
    },
}
