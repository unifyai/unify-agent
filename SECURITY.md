# Security Policy

This document describes unify-agent's trust model, names the load-bearing
boundaries, and defines what's in and out of scope for vulnerability
reports.

---

## 1. Reporting a Vulnerability

**Do not open public issues for security vulnerabilities.**

> TODO(lead): security contact and response policy after the wind-down.
> Name the private channel (for example GitHub private vulnerability
> reporting on this repository, once it is enabled) and what response, if
> any, a reporter can expect. Until this is filled in, no response time is
> promised.

unify-agent does not operate a bug bounty program.

A useful report includes:

- A concise description and severity assessment.
- The affected component, identified by file path and line range
  (e.g. `unify/sandbox.py:120-145`).
- Environment details (commit SHA, OS and kernel, bubblewrap version,
  Python version).
- A reproduction against `main`.
- A statement of which trust boundary in §2 is crossed.

Please read §2 and §3 before submitting. Findings that don't cross a
documented boundary are still welcome as regular issues or pull requests.

---

## 2. Trust Model

unify-agent is a single-user agent runtime that runs on the operator's
machine. It has no server component, no accounts and no network listener.
Its outbound connections are the model provider (completions and
embeddings), the package index (through the harness's install proxy) and,
with in-process embeddings, Hugging Face for the model weights.

### 2.1 Definitions

- **Operator.** The person who runs the `unify` command. The operator's
  user account is the trust envelope.
- **Harness.** The `unify` process: it holds the provider key, makes every
  model call and starts every sandbox. It is not itself confined.
- **Agents.** The model-driven parts: the actor, and with memory on, the
  capture fork, the memory writer and its helpers, and the librarian.
- **Model-written code.** Every program, shell command and subprocess an
  agent's output causes to run.
- **Inbound surface.** Anything that brings attacker-influenced content
  into an agent's context: files placed in the workspace, the request text,
  anything fetched over a network the operator enabled, and items the
  memory stored from earlier tasks.
- **Action surface.** Anything model-written code does that touches the
  world: files in the workspace, calls to the operator's own tools, and
  network traffic where the operator allows it.

### 2.2 The load-bearing fact: model-written code runs in bubblewrap

The actor writes a Python program per turn and the harness runs it. It runs
in a sandboxed worker process (`unify/actor/execution/worker.py`), and
every shell command and subprocess runs under the same policy
(`unify/sandbox.py`). The boundary is OS-level:

- The host root is mounted read-only, and only as an allow-list: the system
  directories, a few files of `/etc`, the interpreter and the package. The
  workspace and a private `/tmp` are the only writable places.
- `UNIFY_HOME` is hidden behind an empty tmpfs, except read-only views of
  the session transcripts and the package environment, the writable
  workspace and the request's export of the memory library. The store file
  is not mounted.
- Credential locations (`~/.ssh`, `~/.config`, `~/.aws`, `~/.gnupg` and
  others), `.env` files and key-shaped files are hidden. Environment
  variables whose names look like credentials are removed.
- The network namespace is private: no network at all, unless the operator
  sets `UNIFY_WORKSPACE_NETWORK=proxy`, in which case the only route out is
  the operator's proxy on the host.
- A seccomp filter limits socket families and refuses new user namespaces
  and io_uring; resource limits bound memory, files and processes.
- Without bubblewrap, nothing runs: the harness refuses rather than run
  unconfined.

Package installs run under the same policy, from prebuilt wheels only,
through an allow-listing proxy that reaches only the package index. Memory
agents run the code they write in boxes with no network and an empty
environment (`unify/memory/sandbox_run.py`).

What this does **not** confine:

- **The harness itself.** It runs with the operator's full rights.
- **The operator's own tools.** Functions registered with
  `UNIFY_ENV_NAMESPACES` run in the harness process, outside the sandbox.
  Model-written code can call them with any arguments.
- **What the workspace holds.** Model-written code can read and change
  every file in it.
- **The model's decisions.** If an attacker steers an agent's context, the
  sandbox bounds what the resulting code can reach, not what it tries.

### 2.3 Credential surfaces

- **`.env`** in the checkout, or the process environment: the provider key.
  Readable by the harness and by anything else the operator runs; hidden
  from model-written code.
- **`<UNIFY_HOME>`** (default `~/.unify`): the store, the memory library
  (a git repository), episode records, the shared run record, transcripts
  and logs. Ordinary files under the operator's home, protected only by file
  permissions. Model-written code sees only the parts listed in §2.2.

### 2.4 In-process heuristics (useful, not boundaries)

The following shape what agents do. They are not boundaries:

- Tool docstrings, prompts and argument validation steer the model toward
  safer choices.
- Effect labels on stored functions and on the operator's tools (`read`,
  `write`, `destructive`) describe; they do not prevent.
- The memory gate labels what it finds on each stored item. Apart from its
  safety and integrity checks, a label informs later agents; it does not
  stop an item from being served.

None of these survive a model that wants to do something they don't allow.
The sandbox in §2.2 is the boundary.

### 2.5 Inbound surfaces

Every byte that reaches a model is attacker-influenceable the moment it
came from outside the operator's head:

- **Workspace files**: whatever the operator places in the workspace.
- **Network content**, if the operator enables a proxy.
- **Memory.** With memory on, what one task leaves behind (captures, stored
  functions and notes) is served to later tasks. Content from an
  adversarial input can therefore steer a later, unrelated task.

The supported posture for adversarial inbound surfaces is to run unify-agent
inside a whole-machine boundary (a VM, or a container that still lets
bubblewrap start), with a scoped provider key, and with memory off or with
a memory home kept for that purpose only.

---

## 3. Scope

### 3.1 In Scope

- **Sandbox escapes**: model-written code reading a hidden path, writing
  outside the workspace and `/tmp`, reaching a network the policy does not
  allow, or obtaining the provider key.
- **Trust-boundary bypasses** that let content from an inbound surface
  run code in the harness process, exfiltrate credentials, or persist data
  without the operator's involvement.
- **Parsing-surface bugs**: path traversal, command injection, or a bypass
  of the read-only authorizer for model-written SQL (`unify/db.py`).
- **Hard-coded credentials or secrets** in the repository.
- **Supply-chain issues** affecting `uv.lock` or the pinned unillm commit.

### 3.2 Out of Scope

- **Prompt injection alone**, without a demonstrated boundary bypass.
  Prompts are influenceable by definition; mitigations are heuristics
  (§2.4).
- **Anything in the `unillm` repository**: report against that repository
  directly.
- **Operator-chosen exposures**: a world-readable `.env` or `UNIFY_HOME`,
  a permissive proxy behind `UNIFY_WORKSPACE_NETWORK=proxy`, or a tool
  registered with `UNIFY_ENV_NAMESPACES` that does more than intended.
- **Provider-side findings**: report bugs in model provider APIs to the
  provider.
- **Platforms the harness refuses to run on** (anything without
  bubblewrap, such as macOS).

---

## 4. Hardening

Recommendations for operators feeding unify-agent untrusted inbound
surfaces:

- **Use scoped provider keys** where the provider supports them (per-project
  keys, spend limits), and keep the memory writer's spend cap low.
- **Run unify-agent in a VM or container** if you intend to expose it to
  adversarial content.
- **Leave `UNIFY_WORKSPACE_NETWORK` unset** unless a task needs the network,
  and then let your proxy allow only what it needs.
- **Register only the tools a task needs**, and make them check their
  arguments: they run outside the sandbox.
- **Tighten permissions** on `.env` and `UNIFY_HOME` (`chmod 600` /
  `chmod 700`) and consider full-disk encryption on the host.
- **Read what memory stored** (the git repository under
  `~/.unify/memory`) before relying on it, and use a separate `UNIFY_HOME`
  for untrusted work.
- **Run with `unify act --debug`** during the first few sessions to see what
  the agents are doing.

---

## 5. Disclosure

> TODO(lead): disclosure and fix policy after the wind-down (whether fixes
> will be made, where they land, how reporters are credited).
