# Library usage

`llmbox_lib` provides an `Agent` class for using the agent programmatically without terminal I/O.

```python
from llmbox_lib import Agent

agent = Agent()
result = agent.run("What files are in the current directory?", max_turns=5)
print(result.text)
```

The `Agent` class accepts optional configuration, a custom system prompt, callbacks, and a mode:

```python
agent = Agent(
    config={
        "llm": {"model": "claude-v4.5-opus"},
        "cycle": {"max_turns": 20},
    },
    system_prompt="You are a log analysis assistant. Be concise.",
    on_tool=lambda name, args: print(f"  [tool] {name}"),
    on_turn=lambda n, result: print(f"  [turn {n}] done"),
    mode="long",  # use server-side conversation caching
)

result = agent.run("Analyze the log files in /var/log/myapp")
print(result.text)
print(f"Completed in {result.total_turns} turns (status: {result.status})")
```

`run()` returns a `RunResult` with structured data:
- `result.text` — final assistant text
- `result.turns` — list of `TurnResult` objects, each with `.text`, `.tool_results`, `.reasoning`
- `result.total_turns` — number of turns taken
- `result.status` — `"done"`, `"max_turns"`, or `"error"`

Use `agent.reset()` to clear conversation history between runs, or omit it to continue the same conversation across multiple `run()` calls. Use `agent.switch_mode("long")` to switch modes with summary carry-over.

See `examples/process_automation.py` for a full example.
