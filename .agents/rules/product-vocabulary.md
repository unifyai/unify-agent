# Product Vocabulary: One Noun Per Concept

The model reasons in the words we hand it. When two concepts share a word, the model has to guess which one a prompt
means, and the guess is invisible until it calls the wrong tool or reads the wrong thing.

## The canonical nouns

| Concept | Word | Where |
|---|---|---|
| One request and everything the actor did for it | **episode** | `unify/memory/` |
| One `execute_code` call: a program run in the persistent interpreter | **cell** | `unify/actor/` |
| A host-side operation a cell calls as `tools.<name>(...)` | **tool** | `unify/common/tool_registry.py` |
| The embedding application's own operations, `primitives.<name>` | **namespace** (its members: **methods**) | `unify/function_manager/primitives/` |
| The shared, append-only message log of a run | **run record** | `unify/agents/` |
| What the episode-end fork wrote down that an episode did | **capture** | `unify/memory/captures.py` |
| A tested function or note the writer landed | **library item** | `unify/memory/` |
| The agent that keeps the library | **writer** (its unit of work: a **pass**) | `unify/memory/writer_pass.py` |
| The agent that answers what the library holds | **librarian** | `unify/memory/librarian.py` |
| What the gate records about an item instead of refusing it | **label** | `unify/memory/gate.py` |
| A model-call route of its own for one memory agent | **lane** | `unify/lanes/` |

Do not write "workflow" to mean a multi-step anything: it is an episode, a pass or a procedure. Do not call a capture
a "skill", and do not call the library a "store".

## Before naming a new first-class type

1. **Grep first.** `rg -ic "<candidate>" unify/ tests/`. If the word appears in prompts, descriptions or examples
   meaning something else, you have a collision.
2. **Prefer an unused word** over renaming existing prose. An unused word is free; a rename costs a sweep through
   every prompt that uses it.
3. **If you keep the colliding word, do the cleanup in the same change.** A half-renamed codebase teaches the model
   both meanings at once.
4. **Watch for the near-synonym trap.** A new noun must not collide *conceptually* either.
5. **No world vocabulary.** The harness's own words name no benchmark, environment or task, and nothing keys on a
   task id.

## What gets renamed, and what does not

Rename **our** vocabulary: prompts, tool descriptions, example titles, section headers, internal identifiers, test
names and fixtures.

Leave alone:

- **Third-party product names.**
- **The shared `global-agent-rules` submodule.** Edit it in its own repo, never here.

After editing anything under `.agents/`, regenerate the aggregate:
`python3 .agents/global-rules/build_agents_md.py`. Never hand-edit `AGENTS.md`.
