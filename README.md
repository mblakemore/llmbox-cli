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
```

### Running with Python directly

```bash
python llmbox.py
python llmbox.py -a "your prompt"
python llmbox.py -c
python llmbox.py -r 3 "run the next cycle"
python llmbox.py -m claude-v4.5-opus "your prompt"
```

### Interactive commands

- Type `exit` or `quit` to end the session
- Type `/clear` to reset conversation history
- Press **Escape twice** (within 400ms) to cancel the current operation
- Reference files with `@path/to/file` to include their contents in the prompt

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

## How it works

The agent uses a **prompt-stuffing** approach: each turn, it builds a single text prompt containing the system instructions, agent identity file, a rolling conversation summary, and recent message history — all fitted within a configurable context budget (default ~20k tokens). The LLM responds with text that may contain `<tool_call>` XML blocks, which are parsed and executed locally. This loop continues until the model responds without tool calls or the turn limit is reached.

Conversation state is checkpointed to `state/conversation_checkpoint.json` after each turn, allowing you to continue from the last checkpoint with the `-c` flag.
