# Claude Code Gateway

`cc_gateway.py` is a local proxy that lets [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Anthropic's CLI coding assistant) use models hosted on the UCSB LLM Sandbox.

It translates Anthropic's Messages API format into LLM Sandbox Bot API calls, handling the send-then-poll pattern transparently.

## How it works

```
Claude Code  ──POST /v1/messages──▶  cc_gateway.py  ──POST /conversation──▶  LLM Sandbox
             ◀──SSE stream──────── (poll until ready) ◀──GET /conversation──
```

1. Claude Code sends a standard Anthropic Messages API request
2. The gateway flattens the messages into Sandbox content blocks
3. POSTs to the Sandbox, then polls with exponential backoff until the response is ready
4. Returns the response as an Anthropic-format SSE stream (or non-streaming JSON)

## Prerequisites

```bash
pip install fastapi uvicorn requests
```

You also need your Sandbox credentials:
- `BEDROCK_API_URL` — your Sandbox Bot API URL
- `BEDROCK_API_KEY` — your API key

If you've already run `./setup.sh` for llmbox, these should be in your shell profile.

## Quick start

### 1. Start the gateway

```bash
uvicorn cc_gateway:app --port 8781
```

Or with explicit credentials:

```bash
BEDROCK_API_URL=https://your-sandbox-url BEDROCK_API_KEY=your-key uvicorn cc_gateway:app --port 8781
```

### 2. Configure Claude Code

Set these environment variables before launching Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8781
export ANTHROPIC_API_KEY=not-needed
```

The API key value doesn't matter (the gateway uses your Sandbox credentials), but Claude Code requires it to be set.

To make this permanent, add to your `~/.zshrc` or `~/.bashrc`:

```bash
# Claude Code via LLM Sandbox
export ANTHROPIC_BASE_URL=http://localhost:8781
export ANTHROPIC_API_KEY=not-needed
```

### 3. Use Claude Code normally

```bash
claude          # interactive mode
claude "prompt" # one-shot
```

Claude Code will route all requests through the gateway to the Sandbox.

## Model mapping

The gateway maps Anthropic model names to Sandbox equivalents:

| Claude Code sends | Sandbox receives |
|-------------------|------------------|
| `claude-opus-4-6` | `claude-v4.6-opus` |
| `claude-opus-4-5` | `claude-v4.5-opus` |
| `claude-sonnet-4-5` | `claude-v4.5-sonnet` |
| `claude-sonnet-4-6` | `claude-v4.6-sonnet` |
| `claude-haiku-4-5` | `claude-v4.5-haiku` |

Unrecognized model names are passed through as-is, so if the Sandbox adds new models they'll work without a gateway update.

## Configuration

All settings are via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `BEDROCK_API_URL` | (required) | Sandbox Bot API base URL |
| `BEDROCK_API_KEY` | (required) | Sandbox API key |
| `DEFAULT_MODEL` | `claude-v4.5-sonnet` | Model when none specified |
| `POLL_TIMEOUT` | `180` | Max seconds to wait for a response |
| `POLL_INITIAL_INTERVAL` | `0.5` | Initial poll interval (seconds) |
| `POLL_BACKOFF_MULTIPLIER` | `1.3` | Exponential backoff factor |
| `POLL_MAX_INTERVAL` | `5.0` | Max poll interval (seconds) |

## Running as a background service

To keep the gateway running persistently:

```bash
# Simple background process
nohup uvicorn cc_gateway:app --port 8781 &> ~/.llmbox-gateway.log &

# Or with systemd (create ~/.config/systemd/user/cc-gateway.service)
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/messages` | Anthropic Messages API (main endpoint) |
| `POST /v1/messages/count_tokens` | Token count stub |
| `GET /health` | Health check |
| `GET /` | Info |
| `GET /docs` | Auto-generated API docs (FastAPI) |

## Limitations

- **No real streaming**: The Sandbox API doesn't support streaming, so responses are polled and then chunked into SSE events to simulate streaming. You'll see a pause while the model generates, then the full response appears quickly.
- **No server-side conversation memory**: Each request is a fresh conversation on the Sandbox. Claude Code manages its own conversation context, so this is fine for normal usage.
- **Token counts are estimates**: The Sandbox doesn't report exact token usage, so counts are approximated at ~4 characters per token.

## Troubleshooting

**Claude Code hangs or times out**
- Check the gateway is running: `curl http://localhost:8781/health`
- Check Sandbox credentials: `curl -H "x-api-key: $BEDROCK_API_KEY" $BEDROCK_API_URL/health`
- Increase `POLL_TIMEOUT` for slow models (e.g. Opus)

**"Connection refused" errors**
- Make sure the gateway is running on the expected port
- Check `ANTHROPIC_BASE_URL` matches the gateway address

**Wrong model being used**
- Check model mapping in the gateway logs (`INFO` level)
- Use `/model` in Claude Code settings to verify
