# Contributing to Unify

## Getting started

Unify depends on one sibling repository, the LLM client `unillm`. Clone both as siblings:

```bash
git clone https://github.com/unifyai/unify-agent.git
git clone https://github.com/unifyai/unillm.git

cd unify-agent
pip install uv && uv sync --all-groups
cp .env.example .env      # add one LLM provider key
```

This installs Unify and `unillm` (linked via `[tool.uv.sources]` in `pyproject.toml`). Then chat with the assistant:

```bash
.venv/bin/python -m unify
```

## Running tests

Tests use real LLM calls with cached responses. After the first run, cached responses replay instantly. Every test process gets its own SQLite store, so there is nothing to start first.

```bash
# Run all tests
tests/parallel_run.sh tests/

# Run a specific module
tests/parallel_run.sh tests/function_manager/

# Run a specific test
tests/parallel_run.sh tests/function_manager/storage/test_venvs.py::test_name
```

See [tests/README.md](tests/README.md) for the full testing philosophy.

## Code style

We use `black` (pinned in the `lint` dependency group) for formatting and `autoflake` for unused import removal, enforced by pre-commit hooks. Install the hooks once per checkout so they run automatically on every commit:

```bash
./scripts/install-git-hooks.sh   # or: pre-commit install
```

Run them manually any time:

```bash
pre-commit run --all-files
```

## Pull requests

- Open PRs against `main`; there is one branch
- Keep PRs focused — one concern per PR
- Tests should pass locally before opening a PR
- We don't require backward compatibility (see the project's aggressive refactoring philosophy)

## Design principles

- **No regex routing.** If the system handles something wrong, fix the prompt or tool docstring, not a hardcoded rule.
- **No defensive coding.** Don't wrap things in try/except unless you're handling a specific, recoverable error. Fail loud.
- **English as API.** Managers communicate through natural-language interfaces. The Actor orchestrates through English-language primitives.
- **Real LLMs in tests.** We never mock the LLM client. Responses are cached for speed, not faked.

See [`VISION.md`](VISION.md) for the bets the project is making — including the things it deliberately *isn't* trying to be. Most "why isn't there a PR for X?" questions are explained by that document.

## Maintainers

Unify is maintained by [Unify](https://unify.ai). The current maintainer team (in commit-count order, deduplicated via [`.mailmap`](.mailmap)):

- **Daniel Lenton** ([@djl11](https://github.com/djl11)) — project lead
- **Yusha Arif** ([@YushaArif99](https://github.com/YushaArif99))
- **Ved Patwardhan** ([@vedpatwardhan](https://github.com/vedpatwardhan))
- **JG** ([@juliagsy](https://github.com/juliagsy))
- **Haris Mahmood** ([@hmahmood24](https://github.com/hmahmood24))
- **Mostafa Hany** ([@CatB1t](https://github.com/CatB1t))
- **Yasser** ([@Infrared1029](https://github.com/Infrared1029))
- **Nassim Berrada** ([@nassimberrada](https://github.com/nassimberrada))

### Area familiarity

The repository's [`.github/CODEOWNERS`](.github/CODEOWNERS) is the canonical routing file — anything not matched by a specific rule requires `@unifyai/Engineers` review.

For PRs that touch a specific subsystem, the table below is a rough guide to who has the deepest familiarity (derived from commit history; team members rotate and overlap). You don't need to tag a reviewer manually — opening a PR is enough, we'll route. The list is a hint for when a fast review matters.

| Area | Reviewers (rough) |
|---|---|
| `unify/actor/` (CodeAct Actor) | @YushaArif99, @djl11 |
| `unify/conversation_manager/` (the interaction loop) | @djl11, @vedpatwardhan, @juliagsy |
| `unify/db.py` (the local store) | @djl11 |
| `unify/function_manager/` | @djl11, @YushaArif99, @juliagsy |
| `unify/guidance_manager/` | @djl11 |
| `tests/conftest.py`, `tests/parallel_run.sh` | @djl11, @CatB1t |

## Questions?

- **Architectural questions** — [GitHub Discussions](https://github.com/unifyai/unify/discussions)
- **Quick questions / chat** — [Discord](https://discord.com/invite/sXyFF8tDtm)
- **Security** — see [`SECURITY.md`](SECURITY.md); do not open public issues for security vulnerabilities.
