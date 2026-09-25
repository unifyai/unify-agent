---
title: "Why we split skills into functions and guidance"
description: "Skill folders nest scripts inside prose. Unify keeps two libraries instead — executable functions and prose guidance, linked many-to-many — so shared rules live once and whole tasks distill into deterministic code."
status: draft
---

# Why we split skills into functions and guidance

Agent skills have converged on a folder convention. A skill is a directory: a `SKILL.md` with frontmatter and prose instructions, plus an optional `scripts/` subfolder for helper code. Claude's skills work this way, and so do the open-source agents built in that mould. The agent sees a catalogue of skill names and descriptions in its system prompt, loads the full body when a task matches, and reaches any scripts through the prose.

It's a reasonable design. Progressive disclosure keeps context small, and a folder is easy to package and share. We went a different way in Unify, and having lived with both shapes I think the difference compounds. We have no skill folders at all. We keep two libraries — executable **functions** and prose **guidance** — linked many-to-many, both searchable on their own.

## What the folder can't express

The skill folder makes prose the front door. The `SKILL.md` is the retrievable unit; a script is an implementation detail of a document. Two things fall out of that, and both bit us back when we worked this way ourselves.

First, shared prose has nowhere to live. Take a rule like "use this tone when writing professional documents". That applies to email drafting, to deck building, to social posts — to any function that writes for an audience. In a folder world you paste it into every skill that needs it, and the copies drift apart. Or you make a standalone "tone" skill and hope the model happens to load two skills and merge them. There's no way to say *this one piece of guidance belongs to those five capabilities*.

Second, code is second-class. A script can only be found through its skill's description. If the task in front of the agent matches the script but not the prose wrapped around it, the code may as well not exist. And sure — you could write the whole procedure into `scripts/` and leave a one-line `SKILL.md` that says "run the script". People do. But at that point the document exists only to make the program retrievable, which tells you the retrieval model is wrong. The unit you wanted was the program.

## Two libraries, linked both ways

In Unify, a stored function is a real catalogue entry: name, signature, docstring, and the implementation itself, indexed for semantic search. Guidance is a separate entry: a title and freeform prose. The link between them is explicit and many-to-many — guidance carries `function_ids`, functions carry the inverse `guidance_ids`, with foreign keys enforced both ways.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/.github/main/public_images/skills-vs-library-dark.png">
    <img src="https://raw.githubusercontent.com/unifyai/.github/main/public_images/skills-vs-library-light.png" alt="Left: two skill folders, each with a SKILL.md owning nested scripts, and the same tone rules pasted into both. Right: a functions library and a guidance library linked many-to-many, with one professional-tone guidance entry linked to three functions, searched and executed directly by the CodeActActor." width="820">
  </picture>
</p>

So the tone rule is one guidance entry, linked to `draft_email`, `build_deck`, and `post_to_linkedin`. Edit it once and every linked function feels the change. A deck-layout procedure links to just the one function it governs. And nothing forces a pairing: guidance with an empty `function_ids` is fine ("prefer the staging database for experiments" attaches to nothing executable), and most functions carry no guidance at all, because a good docstring already says what they do.

Retrieval treats the two libraries as peers. The actor's discovery step searches functions and guidance separately, and `execute_function` runs a function by name with no guidance gate anywhere in the path. Code doesn't need a document wrapped around it to be found or to run.

## Whole tasks become functions

The part I'd defend hardest is what happens after a task finishes. Every `act(...)` run can end with a librarian pass: a separate loop reviews the trajectory, searches the existing stores for duplicates, and decides what deserves to persist. Reusable code that ran successfully becomes a function. A non-obvious procedural insight becomes guidance, linked to the functions it composes via `function_ids`. Very often it stores nothing, which is the right call more than people expect.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/.github/main/public_images/storage-check-dark.png">
    <img src="https://raw.githubusercontent.com/unifyai/.github/main/public_images/storage-check-light.png" alt="The StorageCheck flow: a trajectory snapshot feeds a skill librarian LLM loop, which reviews the trajectory, dedups against existing stores, and fans out to add functions, add guidance, certify a task entrypoint, or store nothing." width="820">
  </picture>
</p>

Our storage prompt contains a line I'm fond of: when the only reusable artifact is one standalone function, store the function only — *do not manufacture a wrapper procedure*. The exact move the folder design forces on you is the one we explicitly ban. The distilled program is the first-class thing; prose exists only when there's genuine know-how the code can't carry.

## Deterministic code, focused LLM calls

There's a cost argument hiding under the design argument. The default agent loop — in Claude Code and in most of the open-source agents — is think, call a tool, think again. Every step is a fresh pass through a frontier model. For a task you run weekly, you pay for the same forty thinking steps every week, and any of them can wander.

A distilled function here is ordinary Python: loops, branches, calls to managed primitives. For the genuinely fuzzy substeps, the sandbox injects a `query_llm(...)` helper (backed by [unillm](https://github.com/unifyai/unillm), so it can hit any provider), and our prompting explicitly pushes stored functions toward this shape:

```python
async def triage_inbox(label: str) -> list[str]:
    emails = fetch_unread(label)          # deterministic
    to_reply = []
    for e in emails:                      # deterministic
        c = await query_llm(              # focused LLM call
            f"Classify for triage.\nSubject: {e.subject}\nBody: {e.body}",
            response_format=EmailClassification,
        )
        if c.needs_reply and c.confidence >= 0.8:
            to_reply.append(e.id)
    return to_reply
```

The control flow is code, so it does the same thing every time. The model is invoked only where judgment is needed, on a small prompt, with a typed response format. The second time the task comes around, the agent doesn't re-derive the procedure — it finds `triage_inbox` in the library and calls it. Two focused LLM calls instead of forty open-ended ones. That's most of our cost story, and it falls out of making the program the stored unit rather than the paragraph.

## What the folder design gets right

To be fair to it: progressive disclosure is the correct instinct, and we do the same thing — search first, load bodies on demand. Some of these agents also close the write loop, nudging the model to save a skill after a task goes well, and that instinct is right too. The disagreement isn't about whether agents should learn from their own work. It's about the shape of what's learned. A folder of prose with code tucked inside optimises for shipping packs of instructions to an agent. A graph of functions and guidance optimises for an agent building its own library as it goes.

Folders do share better — a directory zips and installs anywhere, and there are real registries built on that. Our stores are database-backed instead: rows in one local SQLite file, with nothing to zip or download. I'll take that trade, but it is one.

I wonder whether the folder designs converge here anyway. The moment you want one rule shared across five skills, or a script findable without its wrapper, you start building an index over the folders — and the index *is* the two libraries. The folder is just packaging.

## Where to look

All open at [github.com/unifyai/unify](https://github.com/unifyai/unify):

- The function model, with `guidance_ids`: [`unify/function_manager/types/function.py`](https://github.com/unifyai/unify/blob/main/unify/function_manager/types/function.py)
- The guidance model, with `function_ids`: [`unify/guidance_manager/types/guidance.py`](https://github.com/unifyai/unify/blob/main/unify/guidance_manager/types/guidance.py)
- Discovery and storage prompts, and the StorageCheck loop: [`unify/actor/code_act_actor.py`](https://github.com/unifyai/unify/blob/main/unify/actor/code_act_actor.py), [`unify/actor/prompt_builders.py`](https://github.com/unifyai/unify/blob/main/unify/actor/prompt_builders.py)
- The `query_llm` sandbox helper: [`unify/common/reasoning.py`](https://github.com/unifyai/unify/blob/main/unify/common/reasoning.py), [`unify/function_manager/execution_env.py`](https://github.com/unifyai/unify/blob/main/unify/function_manager/execution_env.py)
