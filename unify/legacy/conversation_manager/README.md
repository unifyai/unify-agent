# ConversationManager

The `ConversationManager` is the assistant's persistent interaction loop. It
reads the chat, decides whether to speak or wait, dispatches work to the
`Actor` and steers that work while it runs. It is the "front office" that
faces the user; the `Actor` is the brain that carries out the work.

There is one user, one chat, and the loop is reactive: it only ever acts on a
message it received or on the work that message started.

## Running

The chat CLI boots the loop in-process:

```bash
.venv/bin/python -m unify
```

`unify/cli.py` publishes `UnifyMessageReceived` events on the in-memory event
broker and renders the `UnifyMessageSent` (and `DirectMessageEvent`) replies.
`/attach <path>` copies a file into the workspace's `Attachments/` folder and
puts its workspace path on the next message. Any client that speaks the same
events over the broker drives the same loop. Programmatic callers use
`run_conversation_manager()` from `main.py`.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────┐
│                          Chat front end (cli.py)                          │
│        UnifyMessageReceived  ──▶   broker   ──▶  UnifyMessageSent         │
└───────────────────────────────────────────────────────────────────────────┘
                                    │  In-memory event broker (app:comms:*)
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          ConversationManager                              │
│                                                                           │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────────────────┐  │
│  │   ChatHistory   │  │ NotificationBar │  │          Brain            │  │
│  │   (messages)    │  │    (pending)    │  │     (async tool loop)     │  │
│  └─────────────────┘  └─────────────────┘  └───────────────────────────┘  │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  EventHandler — routes broker events, requests brain turns          │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
                                    │  actions (steerable handles)
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                                 Actor                                     │
│     Runs code-first plans over the workspace and the stored skills        │
└───────────────────────────────────────────────────────────────────────────┘
```

## Key components

### `ConversationManager` (`conversation_manager.py`)

The central object. It holds the live conversation state (`ChatHistory`,
`NotificationBar`, the in-flight and completed actions) and runs the brain,
an async tool loop that reads that state and decides what to do next.

- `wait_for_events()` subscribes to the broker channels and dispatches each
  event through `EventHandler`.
- `request_llm_run()` asks for a brain turn through the `Debouncer`, which
  coalesces bursts of events into one turn and never cancels a running one.
- `run_llm()` executes one brain turn: render the state panes, call the LLM,
  run the tools it chose. The brain's own LLM messages live in
  `brain_messages`, in memory only.
- `get_recent_transcript()` returns the tail of the conversation as
  role/content turns; the handle's `ask` builds its context from it.

### `ChatHistory` (`domains/chat_history.py`)

The conversation itself: an in-memory list of `ChatMessage` (role, content,
attachment paths, timestamp) mirrored one row per message to the store's
`messages` table. `bind()` runs during manager init and writes through
anything that arrived earlier; `load()` prepends the previous sessions'
messages so the conversation survives a restart, and the slow brain's first
turn after a boot holds at the hydration gate until that load has landed.

Identity comes from `SESSION_DETAILS`: the user's name labels their lines
in the rendered conversation and fills the "User details" section of the
system prompt; the assistant's own lines render as `You`.

### The brain (`domains/brain.py`, `brain_tools.py`, `brain_action_tools.py`)

The brain's tools fall into three groups:

- **Speaking**: `send_unify_message` (with an optional workspace-file
  attachment) and the `wait` choice.
- **Actions**: `act` dispatches a request to the `Actor` and returns a
  steerable handle. The brain quotes attachment paths from the conversation
  into the query so the actor can open the files.
- **Steering**: six fixed, handle-addressed tools (`interject_action`,
  `ask_action`, `pause_action`, `resume_action`, `stop_action`,
  `answer_clarification_action`) steer any in-flight action, and
  `ask_action` also answers questions about a completed one from its
  preserved trajectory; `task_actions.py` renders the ready-to-use
  invocations shown in the state panes.

Prompt composition lives in `prompt_builders.py`; `domains/renderer.py`
renders the panes the brain reads each turn: notifications, in-flight and
completed actions, recent tool executions, and the one conversation.

### `ConversationManagerHandle` (`handle.py`)

The handle the `Actor` receives so a running plan can reach back into the
conversation:

- `ask(question, response_format)` starts a nested tool loop that either
  answers from the recent conversation (PATH 1) or posts the question to the
  chat and waits for the reply (PATH 2). While a PATH 2 question is open the
  handle is registered as `active_ask_handle`, and the next inbound chat
  message is delivered to it as the answer instead of waking the brain.
- `interject(message)` injects information into the conversation.

### `EventHandler` (`domains/event_handlers.py`)

Registry-based dispatcher: `@EventHandler.register(EventClass)` binds a
handler. Inbound and outbound chat messages are appended to the
`ChatHistory` and summarised on the `NotificationBar`; actor lifecycle events
update the action panes; `Error` events surface as notifications so the brain
can recover.

### Events (`events.py`)

Dataclass events carried over the broker:

- **Chat**: `UnifyMessageReceived`, `UnifyMessageSent` (content, attachment
  paths, timestamp), `DirectMessageEvent`.
- **Actor**: `ActorHandleStarted`, `ActorHandleResponse`, `ActorResult`,
  `ActorNotification`, `ActorSessionResponse`, `ActorClarificationRequest`,
  `ActionStopRequested`.
- **State**: `NotificationInjectedEvent`, `OpenSlowBrainTurn`,
  `InitializationComplete`, `Error`.

The broker itself (`event_broker.py`, `in_memory_event_broker.py`) is an
in-process pub/sub keyed by channel prefix (`app:comms:*`, `app:actor:*`).

### Initialization (`domains/managers_utils.py`)

`init_conv_manager` runs in a worker thread: it initializes the runtime,
binds the chat table and starts loading the stored conversation, builds the
`ConversationManagerHandle` and the `Actor`. The brain serves during that
window; the initialization-complete notification tells it whether any
history was restored.

## Tests

```bash
tests/parallel_run.sh tests/conversation_manager/
```

| Directory | Covers |
|---|---|
| `core/` | Initialisation, chat persistence, event handling, brain turns, resilience |
| `handle/` | The `ask` flow, PATH 1 and PATH 2, interjections |
| `actions/` | Action dispatch and steering; `actions/integration/` drives a real `CodeActActor` |
| `flows/` | End-to-end conversation flows |

The parent conftest pins the registry's actor to the simulated one so brain
tests stay cheap; the integration suite builds a `CodeActActor` directly.
