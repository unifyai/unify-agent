# Contributing to unify-agent

## Getting started

You need Linux or WSL2 with bubblewrap, util-linux and git, plus Python 3.12
and [uv](https://docs.astral.sh/uv/) (see the [README](README.md#quickstart)).
unify-agent depends on its model client, `unillm`, pinned to one commit. Until
that commit is on GitHub, uv installs unillm from a checkout beside this one,
which must be at that commit:

```bash
git clone https://github.com/unifyai/unify-agent.git
git clone https://github.com/unifyai/unillm.git
git -C unillm checkout 9d8542206047d8131cf23f5511f4d6afcc81fc75

cd unify-agent
uv sync --group dev
```

The `dev` group holds the test and lint tools, and the packages the tests of
the in-process embedder need. The `[tool.uv.sources]` entry in `pyproject.toml`
links the sibling checkout as an editable install; remove it once the pinned
commit is published.

## Running the tests

```bash
uv run pytest -n auto                               # everything
uv run pytest -n auto tests/function_manager        # one area
uv run pytest tests/test_cli_act.py::test_name      # one test
tests/parallel_run.sh tests/                        # one tmux session per test file
```

`tests/parallel_run.sh` needs tmux; [`tests/README.md`](tests/README.md)
describes it and the rest of the tooling. Every pytest process re-runs itself
inside bubblewrap, so model-written code under test cannot read the rest of the
machine.

### Tests never call a model

- Every model reply in a test is scripted (`tests/scripted_model.py`;
  `scripted(...)` in `tests/cache_discipline_helpers.py`; the `loop_model`
  fixture in `tests/async_tool_loop`). A test states what the model says and
  asserts what the harness does with it.
- No test needs a key. When no provider key is set, the root `conftest.py`
  sets `OPENROUTER_API_KEY=ci-placeholder-not-a-key`, a placeholder that is
  plainly not a secret, so settings load and nothing can authenticate anywhere.
  CI sets the same placeholder and holds no secret.
- The guard in `tests/conftest.py` holds in every session, key or not: the
  model client's provider calls fail with an error that says how to script the
  model, and a socket may connect only to this machine. A test marked
  `live_provider` or `requires_provider_key` fails.
- A check whose subject is a live model's answer belongs in
  [`model_evals/`](model_evals/README.md). Those checks need a key, spend money
  and are run by hand; the suite never collects them.

## Code style

`black` and `autoflake` (pinned in the `lint` group) run as pre-commit hooks.
Install the hooks once per checkout:

```bash
./scripts/install-git-hooks.sh   # or: pre-commit install
pre-commit run --all-files       # run them by hand
```

## Branches and pull requests

- Open pull requests against `main`. Keep each one to one concern, with the
  tests passing locally.
- Backward compatibility is not required. Remove what a change makes unused.

## Design principles

- **No knobs.** Memory has one switch, `UNIFY_MEMORY` (`off`, `on`).
  Beyond it, only essential configuration (the provider key,
  the model and effort, paths, the memory writer's spend cap) is allowed, and
  every environment variable is listed in [`CONFIG.md`](CONFIG.md). Do not add
  switches, flags or code kept for a path that no longer runs.
- **No regex routing.** If the system handles something wrong, fix the prompt
  or the tool's docstring, not a hard-coded rule.
- **No defensive coding.** Catch an exception only to handle a specific,
  recoverable error. Fail loudly otherwise.
- **General, not task-specific.** Prompts and memory never key on task ids,
  answers or one kind of task.
- **Scripted models in tests.** See above.

[`ARCHITECTURE.md`](ARCHITECTURE.md) describes how the actor and its tool loop
fit together, and [`docs/memory.md`](docs/memory.md) the memory.

## Maintainers

TODO(lead): who maintains and reviews after the wind-down.
`.github/CODEOWNERS` still names the former team and paths that no longer
exist.

## Questions

Open an issue on this repository. Report security problems privately, as
[`SECURITY.md`](SECURITY.md) describes, never in a public issue.
