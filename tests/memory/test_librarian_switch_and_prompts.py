"""``UNIFY_MEMORY_LIBRARIAN`` (design J(b).2), the actor's constant pointer (J(b).5), the in-code tool on the actor's
allow-list (J(b).4 step 2), the librarian brief (J(b).6) and the use record's helper counts (J(b).4, audit).

* Off by default; on with tools in code off is refused when the settings load, with a reason.
* On, the actor's memory section is a constant pointer whatever the library holds: the same bytes for an empty
  library, a large one and across commits, with no index text.
* The brief is general: no benchmark word, nothing for or against task ids or answers, nothing about checking
  examples.
* ``memory:ask`` is one of the package's helper calls: counted apart, never as a use of an item.

No model is called and no box is started.
"""

from __future__ import annotations

import re

import pytest

from tests.actor.code_act.test_prompt_generality import BENCHMARK_WORDS
from tests.memory.librarian_world import FILES, big_files
from tests.memory.test_episodes import _ep
from tests.memory.test_library_helper import _commit
from unify.memory import library_prompts
from unify.memory.gitio import Repo
from unify.memory.integration import prompt
from unify.memory.integration.checkout import export_actor_library_mode
from unify.memory.integration.request import (
    add_runtime_calls,
    library_helper_calls,
    memory_use,
    runtime_calls_empty,
)
from unify.memory.library_index import index_view
from unify.settings import ProductionSettings, SETTINGS

# ==== ASSUMED NAMES (the integrator adapts these to the code, here only) ======================================
SWITCH = "UNIFY_MEMORY_LIBRARIAN"
#: The in-code tool ``memory.ask`` reaches the host with (design J(b).4 step 2).
ASK_TOOL = "ask_library"
#: The helper's audit id (design J(b).4) and its name among ``library_helper_calls``.
ASK_ID = "memory:ask"
ASK_HELPER = "ask"
#: The use record's new fields (design J(b).4).
ASK_CALLS, ASK_ROWS = "ask_calls", "ask_rows"
#: The pointer, word for word as the design gives it (J(b).5); compared with whitespace folded.
POINTER = (
    "Your library of procedures and notes from earlier work is the `memory` package in your cells. "
    "`memory.ask(text)` asks the librarian, who has read every item, what each was built for and how each has "
    "been used, which items fit; it answers with each item that fits and a ready call, or says nothing fits and "
    "what differs."
)


def librarian_brief() -> str:
    """The librarian's brief (design J(b).6), as its system prompt carries it between the shared opening and the
    catalogue."""
    from unify.memory import librarian

    return librarian.BRIEF


def actor_allow() -> tuple[str, ...]:
    """The actor's in-code allow-list for a request, as ``tool_handlers.actor_binding`` builds it."""
    from unify.actor import tool_handlers as H
    from unify.common import tool_registry as R

    return tuple(R.allow_list("actor", H.actor_flags("actor", False)))


# ==============================================================================================================


def _set(monkeypatch, name: str, value: str) -> None:
    if hasattr(SETTINGS, name):  # while the setting does not exist yet, the test fails on what it checks
        monkeypatch.setattr(SETTINGS, name, value)


def _flat(text: str) -> str:
    return " ".join(text.split())


# --- the setting -----------------------------------------------------------------------------------------------


def test_the_setting_is_off_by_default_and_parses_on_and_off():
    assert getattr(ProductionSettings(), SWITCH) == "off"
    assert getattr(ProductionSettings(**{SWITCH: "on", "UNIFY_TOOLS_IN_CODE": "on"}), SWITCH) == "on"
    assert getattr(ProductionSettings(**{SWITCH: "off"}), SWITCH) == "off"
    with pytest.raises(ValueError, match=SWITCH):
        ProductionSettings(**{SWITCH: "maybe", "UNIFY_TOOLS_IN_CODE": "on"})


def test_on_with_tools_in_code_off_is_refused_at_settings_load_with_a_reason():
    with pytest.raises(ValueError) as caught:
        ProductionSettings(**{SWITCH: "on", "UNIFY_TOOLS_IN_CODE": "off"})
    text = str(caught.value)
    assert SWITCH in text and "UNIFY_TOOLS_IN_CODE" in text


# --- the actor's pointer ----------------------------------------------------------------------------------------


def _export(tmp_path, name: str, files: dict | None):
    mem = Repo.init_bare(tmp_path / name / "memory")
    sha = _commit(mem, files, "seed") if files else mem.head()
    co = tmp_path / name / "co"
    export_actor_library_mode(mem.git_dir, sha, co)
    return mem, co


@pytest.fixture
def librarian_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY", "on")
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY", "on")
    _set(monkeypatch, SWITCH, "on")


def test_the_pointer_has_the_same_bytes_for_an_empty_and_a_large_library(tmp_path, librarian_on):
    _, empty = _export(tmp_path, "empty", None)
    _, small = _export(tmp_path, "small", FILES)
    _, large = _export(tmp_path, "large", big_files(60))
    texts = [prompt.render_memory_library_mode(co)[0] for co in (empty, small, large)]
    assert texts[0] == texts[1] == texts[2], [len(t) for t in texts]
    assert _flat(POINTER) in _flat(texts[0])


def test_the_pointer_has_the_same_bytes_across_commits_and_carries_no_index(tmp_path, librarian_on):
    mem, co = _export(tmp_path, "lib", FILES)
    one, shown = prompt.render_memory_library_mode(co)
    index = (co / "INDEX.md").read_text()
    new = _commit(mem, {"memory/orders/more.py": 'def more(x):\n    """More."""\n    return x\n'}, "a pass lands")
    export_actor_library_mode(mem.git_dir, new, co)
    two = prompt.render_memory_library_mode(co)[0]
    assert one == two
    # no index text: no item, package or note of the library, and not the index view
    for name in ("memory.orders", "memory.billing", "total_amount", "read_orders", "notes/orders", "more("):
        assert name not in one, name
    assert index_view(index) not in one
    assert shown.get("items", []) == [] and shown.get("channels", []) == []
    # the rest of the guide stays
    for call in ("memory.show(", "memory.run("):
        assert call in one, call


def test_off_the_actor_section_is_the_inline_index_as_before(tmp_path, monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    _set(monkeypatch, SWITCH, "off")
    _, co = _export(tmp_path, "lib", FILES)
    text = prompt.render_memory_library_mode(co, actor_index="inline")[0]
    assert "memory.ask" not in text
    assert "total_amount" in text  # the index view is still there


# --- the in-code tool -------------------------------------------------------------------------------------------


def test_the_ask_tool_is_on_the_actors_allow_list_only_when_on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_TOOLS_IN_CODE", "on")
    _set(monkeypatch, SWITCH, "off")
    assert ASK_TOOL not in actor_allow()
    _set(monkeypatch, SWITCH, "on")
    assert ASK_TOOL in actor_allow()


# --- the brief -------------------------------------------------------------------------------------------------

#: Text for or against task ids or answers (the brief must say neither).
_IDS_OR_ANSWERS = re.compile(
    r"\b(task[\s_-]?ids?|identifiers?|ids? of (the )?tasks?|(correct|expected|right|true|final|gold) "
    r"(answers?|outputs?|results?|solutions?)|store (the |an |its )?answers?|answer keys?)\b",
    re.IGNORECASE,
)


def test_the_brief_says_role_inputs_and_answer_form():
    brief = librarian_brief()
    assert isinstance(brief, str) and brief.strip()
    for needed in ("memory.show", "tools.finish", "fits", "nearest", "differs", "why", "call"):
        assert needed in brief, needed
    assert re.search(r"\bevery (card|item)\b", brief, re.IGNORECASE)  # read every card, name every item


def test_the_brief_is_general_and_says_nothing_of_ids_answers_or_examples():
    brief = librarian_brief()
    assert library_prompts.benchmark_words(brief) == []
    assert BENCHMARK_WORDS.findall(brief) == []
    assert library_prompts.example_checks(brief) == []
    assert _IDS_OR_ANSWERS.findall(brief) == []
    assert librarian_brief() == brief  # constant bytes


# --- the helper's audit and the use record -----------------------------------------------------------------------


def test_ask_is_counted_among_the_helper_calls_and_never_as_a_use():
    acc = runtime_calls_empty()
    add_runtime_calls(acc, {"calls": {"helpers": {ASK_ID: {"ok": 2, "error": 1}}}})
    assert acc["malformed"] == 0  # a helper of the package
    assert ASK_ID not in acc["counts"]  # never a use of an item
    got = library_helper_calls(acc)
    assert got[ASK_HELPER] == {"ok": 2, "refused": 0, "error": 1}
    rec = memory_use(_ep(transcript=[], actions=[]), ["memory.orders.sums:total_amount"], runtime_calls=acc,
                     library_mode=True)
    assert "error" not in rec, rec
    assert rec["library_helper_calls"][ASK_HELPER] == {"ok": 2, "refused": 0, "error": 1}
    assert ASK_ID not in (rec.get("items") or {})


def test_the_library_mode_use_record_carries_the_ask_fields():
    acc = runtime_calls_empty()
    add_runtime_calls(acc, {"calls": {"helpers": {"memory:index": {"ok": 1}}}})
    rec = memory_use(_ep(transcript=[], actions=[]), [], runtime_calls=acc, library_mode=True)
    assert "error" not in rec, rec
    assert rec[ASK_CALLS] == 0 and rec[ASK_ROWS] == []
    off = memory_use(_ep(transcript=[], actions=[]), [], runtime_calls=acc)
    assert ASK_CALLS not in off and ASK_ROWS not in off
