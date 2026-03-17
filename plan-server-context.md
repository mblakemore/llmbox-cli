# Plan: Conversation Modes

## Goal

Add a mode system so users can choose the context management strategy that fits their workflow. Default remains the current prompt-stuffing approach (now called "dev" mode). A new "long" mode uses server-side conversation caching. The architecture supports additional modes later.

## Conversation Modes

### Mode comparison

| | **dev** (default) | **long** | future modes |
|---|---|---|---|
| Context strategy | Client-side prompt stuffing with rolling summarization | Server-side conversation caching | TBD |
| Best for | Development workflows, agentic tool use, code generation | Extended Q&A, research, brainstorming, long discussions | — |
| Context pruning | Automatic — old messages fall out, get summarized | None — server stores everything until context limit | — |
| Tool result handling | Large results naturally pruned as they age out | Accumulate forever, eat context budget fast | — |
| Token efficiency | Pays per-turn for recent context only (constant budget) | Pays once for cached prefix (if server caches), but grows linearly | — |
| Recovery on failure | Full history available client-side, resume seamlessly | Summarize + start new conversation | — |
| Fidelity | Lossy — older turns compressed to summary | Lossless — exact messages preserved until context limit | — |
| Max session length | Unlimited (summary compresses indefinitely) | Bounded by model context window | — |

### When to use each mode

**dev mode** (prompt stuffing) — recommended for:
- Agentic tool-use loops (file reads, commands, search results create lots of disposable context)
- Long-running development sessions that may span hundreds of turns
- Workflows where recent context matters more than exact recall of early turns
- Any session where tool results dominate the conversation

**long mode** (server-side caching) — recommended for:
- Extended text conversations (Q&A, brainstorming, research)
- Sessions where exact recall of earlier messages matters
- Shorter conversations that won't approach context limits
- When the server supports prompt caching (reduced token cost for repeated prefixes)

---

## Pre-Development Structure

### Prerequisites

- [ ] Server `continueGenerate` bug reported to AWS devs
- [ ] Server generation parameters (temperature, top_p, max_tokens) feature request filed
- [ ] Confirm server max token gen setting is restored to production value (was lowered to 3k for testing)

### Risk / Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Server loses conversation mid-session | User loses context, potential crash | Client-side token tracking triggers proactive recovery before limit; catch 500 errors and recover via summary |
| Truncation recovery adds meta-messages to server history | Pollutes context with "continue from..." messages | Cap continuation attempts (max 3); if still incomplete, warn user and treat as text response |
| Model context window sizes unknown/varying | Can't set accurate safety thresholds | Start with conservative estimates; add `/context` command to show usage; log actual failures to calibrate |
| @file references inflate server history permanently | Large files eat context budget in long mode and can't be pruned | Warn user when attaching large files in long mode; consider summarizing file contents instead of raw inclusion |
| Mode switch mid-conversation loses nuance | Summary is lossy; switching long→dev drops exact history | Inform user what will be carried over (summary only) before switching |
| Think tool in long mode | Think tool creates its own API instance — does it need conversation_id? | No change needed — think tool is intentionally a standalone call, stays independent of conversation mode |
| `-c` resume with stale/lost server conversation | Checkpoint has conversation_id but server may have purged it | On resume, verify conversation exists via GET; if 404, fall back to dev mode with empty state |

### Known API limitations (not changeable client-side)

- No per-request control of: temperature, top_p, top_k, max_tokens, stop sequences
- `enableReasoning` is the only generation toggle (off by default, used only by think tool)
- `continueGenerate` is non-functional
- Conversations are append-only (no delete, edit, branch, or prune)
- Cannot inject assistant-role messages (role field ignored)

---

## Verified API Behavior

Tested against `ylfhl5t02h.execute-api.us-east-1.amazonaws.com/api`:

### Endpoints (from OpenAPI spec)

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Health check |
| POST | /conversation | Send a message (new or follow-up) |
| GET | /conversation/{conversation_id} | Get full conversation with messageMap |
| GET | /conversation/{conversation_id}/{message_id} | Get assistant response to a specific user message |

**No DELETE, PATCH, or PUT** — conversations are append-only, no server-side cleanup mechanism.

### Conversation flow (verified)

- **New conversation**: `POST /conversation` with `{"message": {...}}` → returns `{conversationId, messageId}`
- **Continue conversation**: `POST /conversation` with `{"conversationId": "...", "message": {...}}` → returns `{conversationId, messageId}` (conversationId goes in the **body**, not the URL — `POST /conversation/{id}` returns 405)
- **Poll specific message**: `GET /conversation/{convId}/{userMsgId}` → returns 404 while processing, 200 with assistant response when ready
- **Full conversation**: `GET /conversation/{convId}` → returns full `messageMap` with all messages
- Server maintains full message tree; follow-up messages correctly reference prior context (tested: model recalled info from message 1 when asked in message 2)
- **Cannot inject assistant messages** — `role` field is ignored; all sent messages become `user` role. Cannot reconstruct or clone a conversation by replaying history.

### POST /conversation schema

```json
{
  "conversationId": "string | null",  // null = new conversation
  "message": {
    "content": [{"contentType": "text", "body": "..."}],
    "model": "claude-v4.5-sonnet"
  },
  "continueGenerate": false,  // BROKEN — do not use
  "enableReasoning": false    // enable chain-of-thought (off by default; only think tool uses it)
}
```

### GET response format (message-specific endpoint)

```json
{
  "conversationId": "...",
  "message": {
    "role": "assistant",
    "content": [{"contentType": "text", "body": "..."}],
    "model": "claude-v4.5-sonnet",
    "parent": "user-message-id",
    "children": [],
    "thinkingLog": null
  },
  "createTime": 1773708172.35
}
```

### Message-specific endpoint constraints

- Only accepts **user** message IDs (not assistant IDs — returns 404)
- Returns the **assistant's response** to that user message
- Fails if the user message has no children (no response yet — returns 404)

## Additional API Features Investigated

### Native tool use support — NOT usable for custom tools

The API schema includes `toolUse` and `toolResult` content types, but the gateway doesn't expose a `tools` parameter for registering custom tool definitions. The model returns tool calls as XML text (`<tool_call>` blocks), not native `toolUse` content.

**Verdict**: Continue using XML `<tool_call>` parsing. No change needed.

### `continueGenerate` flag — BROKEN, do not use

- API accepts `continueGenerate: true` and returns a messageId, but the message never generates (stays 404 forever, `lastMessageId` never updates)
- Tested multiple times — consistently non-functional
- **Verdict**: Do not implement. Use manual continuation instead.

### Truncation recovery — manual continuation via follow-up message

When the server's max output token limit truncates a response mid-tool-call:

1. Detect: response has `<tool_call>` but no `</tool_call>`
2. Send a follow-up user message: `"Your response was truncated. Continue from exactly where you left off. Last part ended with:\n...{tail}"`
3. The model continues generating from where it stopped
4. Repeat until `</tool_call>` appears, then concatenate all parts
5. Parse tool calls from the concatenated result

**Caveat for long mode**: Each continuation adds a user+assistant message pair to the server-side history. Cap at 3 continuation attempts to avoid pollution. If still incomplete after 3, treat as a text response and warn the user.

**Tested with server max tokens = 3k:**
- Initial response: 31,894 chars (truncated mid-code)
- After 2 continuation rounds: 44,874 chars total with properly closed `</tool_call>`
- Model correctly picks up mid-line and doesn't repeat content

### Conversation management — no server-side cleanup

- No way to delete, truncate, or branch conversations via the API
- Conversations are append-only
- No way to "redo" the last message — the message tree only grows
- Cannot inject assistant messages (role field ignored, always stored as user)
- **Known bug**: Server crashes if conversation exceeds model context limits (no graceful error)

---

## Context Limit Recovery (long mode)

When the server-side conversation approaches or exceeds the model's context window:

### Detection

- Track approximate token usage client-side (chars sent + received per turn, using existing `_estimate_chars()` logic)
- Check **before sending** each turn — if projected usage exceeds threshold, trigger recovery before the server crashes
- Safety threshold: 80% of model context window (conservative to account for estimation error)
- Also catch 500 errors reactively as a fallback

### Model context window sizes (initial estimates)

| Model | Context window | Safety threshold (80%) |
|-------|---------------|----------------------|
| claude-v4.5-opus | 200k tokens (~700k chars) | 560k chars |
| claude-v4.5-sonnet | 200k tokens (~700k chars) | 560k chars |
| claude-v4.5-haiku | 200k tokens (~700k chars) | 560k chars |
| claude-v3.7-sonnet | 200k tokens (~700k chars) | 560k chars |
| llama models | 128k tokens (~450k chars) | 360k chars |
| mistral models | 32k tokens (~112k chars) | 90k chars |
| deepseek-r1 | 64k tokens (~224k chars) | 180k chars |
| amazon-nova-* | 128k tokens (~450k chars) | 360k chars |
| qwen3-32b | 32k tokens (~112k chars) | 90k chars |

These are estimates. Log actual failures to calibrate over time.

### Recovery steps

1. **Summarize**: fetch the full conversation tree via `GET /conversation/{id}` (if still accessible), or use the last known assistant response as context. Generate a summary via a separate LLM call (new one-shot conversation, not the overflowed one).
2. **Start fresh**: create a new conversation with the summary injected in the first user message, preserving the agent's understanding of the session so far.
3. **User chooses next mode** (interactive) or auto-decides (auto/library mode):
   - **Continue in long mode** — new `conversation_id`, summary as seed, server-side caching resumes
   - **Switch to dev mode** — carry the summary into prompt-stuffing mode, which can run indefinitely without hitting context limits again

### What we DON'T need

- Full client-side history tracking in long mode — the server has it until it doesn't, and summary is sufficient for recovery
- Conversation cloning/replay — can't inject assistant messages, and regenerated responses would differ

---

## Detailed Changes

### Phase 1: `bedrock_api.py` — New methods

**`send()` — add `conversation_id` parameter**
- If `conversation_id` is provided, include it in the POST body
- Returns `(conversation_id, message_id)` as before
- No other changes to signature or behavior

**`poll_message()` — new method**
```python
def poll_message(self, conversation_id: str, message_id: str, cancel_check=None) -> dict:
```
- `GET /conversation/{conv_id}/{msg_id}`
- Polls until 200 (404 means not ready yet)
- Returns the assistant message dict (same shape as existing `poll()` return)
- Same timeout/interval behavior as `poll()`

**`get_conversation()` — new method**
```python
def get_conversation(self, conversation_id: str) -> dict:
```
- `GET /conversation/{conv_id}`
- Returns full conversation dict with messageMap
- Used for recovery: fetch tree before summarizing

**`send_and_wait()` — add `conversation_id` parameter**
- Pass through to `send()`
- Use `poll_message()` instead of `poll()` when conversation_id is provided

Keep existing `poll()` unchanged for backward compatibility.

### Phase 2: `llmbox.py` — Mode system

#### 2a: Mode flag and state

**New argparse flag**: `--mode dev|long` (default: `dev`)

**Mode state** (module-level or passed through):
```python
_mode = "dev"               # current mode
_conversation_id = None     # long mode: server conversation ID
_approx_token_usage = 0     # long mode: running token estimate
```

#### 2b: `/mode` interactive command

- `/mode` with no args: print current mode and stats
  - dev: show summary size, message count, context budget usage
  - long: show conversation_id, approx token usage, estimated remaining capacity
- `/mode dev`: switch to dev mode
  - If currently in long mode: generate summary from server conversation, seed into dev mode summary_state
  - Reset conversation_id
- `/mode long`: switch to long mode
  - If currently in dev mode: generate summary from conversation_history, create new server conversation seeded with summary + system prompt
  - Set conversation_id

#### 2c: Long mode flow in `run_agent_single()`

**First turn (no conversation_id yet):**
1. Build message: system prompt + tool descriptions + agent.md + user message (all in one text body)
2. `send_and_wait()` without conversation_id → creates new conversation
3. Store returned conversation_id
4. Parse response, execute tools as normal

**Subsequent turns:**
1. If tool results from previous turn: format as user message, `send_and_wait()` with conversation_id
2. If new user input: send as-is with conversation_id
3. If wind-down: prepend wind-down text to the message being sent
4. Update `_approx_token_usage` with chars sent + received
5. Check if approaching context limit → trigger recovery if needed
6. Parse response, execute tools as normal

**Truncation handling:**
1. After receiving response, check for `<tool_call>` without `</tool_call>`
2. If truncated: send continuation follow-up with conversation_id (up to 3 attempts)
3. Concatenate all parts before parsing tool calls
4. If still incomplete after 3 attempts: treat as text, warn user

#### 2d: `/help` update

Add mode info to `/help` output.

#### 2e: `-a` (auto) and `-r` (repeat) with long mode

- `-a` with long mode: works normally, single run with server-side context
- `-r` with long mode: each repeat starts a fresh conversation (reset conversation_id between runs)
- `-c` with long mode: checkpoint stores conversation_id; on resume, verify conversation still exists via GET; if gone, fall back to dev mode

### Phase 3: `llmbox_lib.py` — Add mode support

- `Agent.__init__()` accepts `mode="dev"|"long"`
- Store `self.mode`, `self.conversation_id`, `self.approx_token_usage`
- `Agent.run()` dispatches to `_run_turn_dev()` or `_run_turn_long()` based on mode
- `Agent.reset()` clears conversation_id and token tracking in long mode
- `Agent.switch_mode(new_mode)` handles state transition with summary

### Phase 4: Checkpoint format

**dev mode** (unchanged):
```json
{
  "mode": "dev",
  "conversation_history": [...],
  "summary_state": {...},
  "turn": 12,
  "initial_files": "..."
}
```

**long mode**:
```json
{
  "mode": "long",
  "conversation_id": "01KKWP0R2VHGM94R7DMFZMD94G",
  "turn": 5,
  "approx_tokens": 15000
}
```

On `-c` resume with long mode checkpoint:
1. Try `GET /conversation/{id}` to verify it exists
2. If exists: continue normally
3. If 404: warn user, fall back to dev mode with empty state

---

## Message format for tool results (long mode)

Since the server only has user/assistant roles, tool results go as a user message:

```
[Tool results]
exec_command: [session: agent-main-abc123] exit=0
ls output here...

file: [myfile.py: lines 1-50 of 50]
file contents here...
```

This is the same format used in dev mode's prompt stuffing, just sent as a standalone message rather than embedded in the prompt.

---

## Resolved questions

- **`/clear` in long mode**: starts a brand new conversation (drops conversation_id, resets token tracking)
- **Native tool use**: not available for custom tools; continue with XML parsing
- **`continueGenerate`**: broken on server; use manual follow-up continuation instead
- **Context limit in long mode**: summarize → start new conversation → user picks mode to continue in
- **Conversation reconstruction**: not possible (can't inject assistant messages); summary-based recovery only
- **Client-side history in long mode**: not needed; server has it, summary is sufficient for recovery
- **Mode switching**: supported mid-session via `/mode` command with summary carry-over
- **Think tool**: stays independent — uses its own one-shot API calls, unaffected by conversation mode
- **`enableReasoning`**: off by default, only used by think tool, no interaction with modes
- **Generation parameters**: not controllable per-request (temperature, top_p, max_tokens all server-side only)
- **Auto/repeat modes**: `-a` works with both modes; `-r` resets conversation_id between runs
- **@file in long mode**: works but inflates server history permanently (no pruning) — consider adding a warning for large files

## Open questions

- What are the actual context window sizes per model on this server? Initial estimates in the table above need validation.
- Should the token tracking be char-based estimation (like current) or should we try to get actual usage from the server? (Server doesn't expose this currently.)
- What other conversation modes might be useful? (e.g., "minimal" with aggressive pruning for very long automation runs)
- Should we add a `/context` command to show current usage stats (tokens used, estimated remaining, mode)?

---

## Testing Strategy

### Unit tests (if test framework added later)
- `bedrock_api.py`: `send()` with/without conversation_id, `poll_message()` 404→200 cycle, `get_conversation()` parse
- Mode switching state transitions: dev→long, long→dev, summary carry-over
- Token estimation accuracy vs actual for different message types
- Truncation detection and concatenation logic

### Manual integration tests
- [ ] Long mode: multi-turn conversation, verify model recalls earlier messages
- [ ] Long mode: tool use loop (file read → modify → verify), confirm tool results sent correctly
- [ ] Long mode: hit context limit (use small model or lowered server setting), verify recovery flow
- [ ] Long mode: `-c` resume with valid checkpoint
- [ ] Long mode: `-c` resume with lost/expired conversation_id → falls back to dev
- [ ] Mode switch: dev → long mid-session, verify summary seeds new conversation
- [ ] Mode switch: long → dev mid-session, verify summary carries into prompt-stuffing
- [ ] Truncation: force truncation with low server max tokens, verify continuation and concatenation
- [ ] `/mode` command: displays stats, switches correctly
- [ ] `-r` with long mode: each run gets fresh conversation
- [ ] Library: `Agent(mode="long")` basic multi-turn test

### Acceptance criteria
- [ ] `llmbox --mode long "prompt"` completes a multi-turn tool-use session
- [ ] Context limit recovery works without user intervention in auto mode
- [ ] `/mode` switches cleanly in both directions with summary carry-over
- [ ] `-c` gracefully handles missing server conversations
- [ ] No regressions in dev mode (default) behavior
- [ ] `/help` documents mode system
- [ ] README and CLAUDE.md updated

---

## Order of implementation

### Phase 1 — API layer (low risk, no behavior change)
1. `bedrock_api.py`: add `conversation_id` param to `send()` and `send_and_wait()`
2. `bedrock_api.py`: add `poll_message()` and `get_conversation()`
3. Test: verify new methods work against live API

### Phase 2 — Long mode core (new code path, dev mode untouched)
4. `llmbox.py`: add `--mode` argparse flag, mode state variables
5. `llmbox.py`: implement long mode flow in `run_agent_single()` (first turn, subsequent turns, tool results)
6. Test: basic long mode multi-turn conversation with tools

### Phase 3 — Resilience (recovery, truncation, checkpointing)
7. `llmbox.py`: token tracking and context limit detection
8. `llmbox.py`: context limit recovery (summarize → new conversation → mode choice)
9. `llmbox.py`: truncation detection and continuation loop
10. `llmbox.py`: long mode checkpointing and `-c` resume with verification
11. Test: force context overflow, truncation, stale checkpoint

### Phase 4 — Interactive commands and mode switching
12. `llmbox.py`: `/mode` command (display and switch)
13. `llmbox.py`: update `/help` with mode info
14. `llmbox.py`: update `/clear` for long mode
15. Test: mode switching in both directions, `/mode` display

### Phase 5 — Library and docs
16. `llmbox_lib.py`: add `mode` param, `switch_mode()`, long mode run loop
17. Update README.md and CLAUDE.md
18. Full integration test pass
