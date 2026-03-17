# llmbox-cli

A Python CLI agent that connects to an [AWS Bedrock Chat](https://github.com/aws-samples/bedrock-chat) Published API gateway and runs an agentic tool-use loop. The agent generates `<tool_call>` XML blocks to invoke tools for file operations, shell commands, web fetching, and more.

## Setup

### Dependencies

```bash
pip install requests markdownify PyMuPDF
```

### Configuration

Set environment variables:

```bash
export BEDROCK_API_URL="https://your-bedrock-chat-gateway.example.com"
export BEDROCK_API_KEY="your-api-key"
```

Or create a `config.json` in the working directory:

```json
{
  "llm": {
    "api_url": "https://your-bedrock-chat-gateway.example.com",
    "api_key": "your-api-key",
    "model": "claude-v4.5-sonnet"
  }
}
```

### Shell wrapper

To use `llmbox` as a command from any directory, symlink the wrapper script to somewhere on your PATH:

```bash
mkdir -p ~/.local/bin
ln -s "$(pwd)/llmbox.sh" ~/.local/bin/llmbox
```

Make sure `~/.local/bin` is on your PATH. If it isn't, add this to your `~/.bashrc` or `~/.zshrc`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Usage

```bash
# Interactive mode
llmbox

# Single prompt (auto mode — runs and exits)
llmbox -a "analyze the codebase and suggest improvements"

# Continue from last checkpoint
llmbox -c

# Repeat N times (0 = indefinite), implies auto mode
llmbox -r 3 "run the next cycle"

# Override the model
llmbox -m claude-v4.5-opus "your prompt"

# Use long mode (server-side conversation caching)
llmbox --mode long
llmbox --mode long -a "research this topic in depth"
```

### Running with Python directly

```bash
python llmbox.py
python llmbox.py -a "your prompt"
python llmbox.py -c
python llmbox.py -r 3 "run the next cycle"
python llmbox.py -m claude-v4.5-opus "your prompt"
python llmbox.py --mode long "your prompt"
```

### Interactive commands

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/clear` | Reset conversation history |
| `/mode [dev\|long]` | Show or switch conversation mode |
| `/models` | List available models from the gateway |
| `/model <name>` | Set the model (or pick from a menu if no name given) |
| `@path/to/file` | Attach file contents to your prompt |
| `exit`, `quit` | End the session |
| **Escape x2** | Cancel the current operation |

### Agent identity

Place an `agent.md` file in the working directory to provide the agent with persistent identity and instructions. It is automatically loaded at the start of each session.

## Built-in tools

| Tool | Description |
|------|-------------|
| `file` | Read, write, append, delete, and list files. Enforces read-before-write. |
| `exec_command` | Run shell commands with timeout, background execution, and session management. |
| `search_files` | Regex search across files (like grep). |
| `web_fetch` | Fetch web pages, convert to markdown, save to `state/fetched/`. |
| `think` | Separate reasoning API call with chain-of-thought enabled. |
| `task_tracker` | Persistent task management stored in `state/tasks.json`. |
| `read_pdf` | Extract text from PDF files with page range support. |
| `sleep` | Pause execution for a specified duration. |

## Custom tools

Add tool modules to a `tools/` directory in the working directory. Each module should export:

- `fn` — the callable implementation
- `definition` — an OpenAI-compatible tool schema dict

Custom tools with the same name as built-in tools will override them.

## Conversation modes

The agent supports two conversation modes, selectable via `--mode` or `/mode`:

| | **dev** (default) | **long** |
|---|---|---|
| Context strategy | Client-side prompt stuffing with rolling summarization | Server-side conversation caching |
| Best for | Development workflows, agentic tool use, code generation | Extended Q&A, research, brainstorming |
| Max session length | Unlimited (summary compresses indefinitely) | Bounded by model context window |

**dev mode** rebuilds the full prompt each turn from recent history and a rolling summary. Old messages are automatically summarized and pruned. Best for tool-heavy workflows.

**long mode** uses the server's conversation memory. The server retains all messages exactly, giving perfect recall of earlier turns. When the context window fills up, it recovers by summarizing and starting a new conversation (or switching to dev mode).

Switch modes mid-session with `/mode dev` or `/mode long`. Use `/mode` to see current stats.

## Library usage

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

## How it works

In **dev mode** (default), the agent uses a prompt-stuffing approach: each turn, it builds a single text prompt containing the system instructions, agent identity file, a rolling conversation summary, and recent message history — all fitted within a configurable context budget (default ~20k tokens). The LLM responds with text that may contain `<tool_call>` XML blocks, which are parsed and executed locally. This loop continues until the model responds without tool calls or the turn limit is reached.

In **long mode**, the agent uses the server's conversation memory. The first message includes the system prompt and tools; subsequent messages send only tool results or new user input. The server maintains the full conversation tree, so no client-side prompt building is needed. When the estimated context usage approaches 80% of the model's window, the agent recovers by summarizing the conversation and starting fresh.

Conversation state is checkpointed to `state/conversation_checkpoint.json` after each turn, allowing you to continue from the last checkpoint with the `-c` flag. Long mode checkpoints include the server conversation ID, which is verified on resume.
