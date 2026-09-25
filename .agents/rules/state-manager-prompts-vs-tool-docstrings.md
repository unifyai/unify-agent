---
description: State Manager Interface Design (Base APIs, Prompts, and Tool Docstrings)
---

# State Manager Interface Design

## Base Class Public APIs

The public API for all state managers (`FunctionManager`, `GuidanceManager`) is fully contained in the docstrings of the abstract methods defined on the base class `Base{SomeManager}` in `base.py`. All high level usage instructions should be fully encapsulated in these docstrings. These docstrings are then attached to the public methods of any derived class via `@functools.wraps(Base{StateManager}.{public_method}, updated=())`. These docstrings should not make **any** reference to **other managers** (we don't want to lock in any brittle cross-references, as other managers may change) and should also not make any reference to their **internal implementation**, including the private tools used for any particular instantiation of this abstract base class, with a consistent implementation agnostic public API.

## Prompts vs Tool Docstrings

The prompts in each prompt builder file should focus on the high level usage patterns, general guidance to the LLM, and specifically how to reason about the **composition** of tools, which tool to use in which scenario with contrastive explanations etc. However, in order to have a fully modular design and maximise our separation of concerns, it's very important that we do **not** bloat these prompts with any purely tool-specific information. This belongs exclusively in the tool's unique docstring (which the LLM gets access to). If the guidance is about deciding between two tools or using these tools together for complex composite behaviour, then it belongs in the prompt for the high-level public method in `prompt_builders.py`. If it's purely tool-specific, then it belongs in the tools own docstring.

## Tests That Guard the Prompts

`tests/actor/code_act/test_prompt_builders.py` and `tests/conversation_manager/core/test_prompt_builders.py` pin phrases of the rendered prompts with exact substring asserts. Change a pinned phrase and its assert together, and keep each pinned phrase on one line of the prompt source: the source's line breaks survive into the rendered prompt, so re-wrapping a paragraph can split a phrase and fail the assert. `tests/test_prompt_token_budgets.py` caps the tokens every call pays for its system prompt and tool schemas; tighten its budget in the same change as a cut.
