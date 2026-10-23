"""A recorded procedure called with no arguments (or with a request's text) runs on the request in every form it can.

* ``value``: its ``read_input`` reads the request; ``state``: it is handed the state of a request that has just begun
  (``.request``, ``.messages``, ``.last``, ``.data``); ``state_env``: a call cannot pass it the tools it acts through,
  so every call of it refuses before anything runs, says where its code is and how its work is done from here, and
  records no use of it.
* Its card says what its record shows: accepted work only when an accepted episode is on record; that its episode
  failed when it did; the no-argument call only for the forms that take the request.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.memory.test_capture_examples import PRELUDE, _cell  # noqa: F401 - the cell runner
from tests.memory.test_layout import _tree
from unify.memory import captures
from unify.memory import library_export as lx
from unify.memory.catalogue import write_files

STATE_PROC = '''"""Count the rows of the table the request carries."""


def run(state):
    """The number of rows in the request's first JSON value."""
    return len(state.data[0])
'''
ENV_PROC = '''"""Write the request's rows to a file."""


def run(state, env):
    """Write the rows; nothing is handed back."""
    env.files.write("rows.csv")
    return None
'''
VALUE_PROC = '''"""Count the rows the request carries."""

import json


def read_input(text):
    return json.loads(text.split("ROWS\\n", 1)[1])


def run(value):
    """The number of rows."""
    return len(value)
'''


def _put(store: Path, source: str, form: str, outcome: str) -> str:
    cid = captures.capture_id(source)
    d = store / cid
    d.mkdir(parents=True, exist_ok=True)
    rec = {
        "episodes": ["ep-1"],
        "cases": [{"episode": "ep-1", "outcome": outcome}],
        "outcome": outcome,
        "entry_point": {"form": form, "name": "run"},
    }
    (d / captures.PROCEDURE_FILE).write_text(source)
    (d / captures.RECORD_FILE).write_text(json.dumps(rec))
    (d / captures.SAVED_FILE).write_text(json.dumps({"docstring": "", "asked": "", "approach": ""}))
    return cid


def _copy(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    store = captures.store_dir(tmp_path / "state")
    ids = {
        "state": _put(store, STATE_PROC, "state", "positive"),
        "env": _put(store, ENV_PROC, "state_env", "none"),
        "value": _put(store, VALUE_PROC, "value", "none"),
    }
    tree = _tree(tmp_path / "lib")
    write_files(tree, lx.generated_library_mode(tree))
    for rel, data in captures.export_files(store).items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_bytes(data)
    return tree, ids


def test_each_form_runs_on_a_request_or_refuses_cleanly(tmp_path):
    root, ids = _copy(tmp_path)
    request = 'ROWS\n[{"id": 1}, {"id": 2}, {"id": 3}]'
    cap = {k: f"capture:{v}" for k, v in ids.items()}
    got = _cell(
        root,
        "seen = []\n"
        "sys.addaudithook(lambda e, a: seen.append(a[0]) if e == 'memory.call' else None)\n"
        f"print(json.dumps({{'state': memory.get({cap['state']!r})({request!r}), "
        f"'value': memory.get({cap['value']!r})({request!r}), "
        f"'env': err(memory.get({cap['env']!r}), {request!r}), "
        f"'env_none': err(memory.get({cap['env']!r})), "
        f"'env_other': err(memory.get({cap['env']!r}), [1, 2], rows=3), "
        "'calls': seen}))\n",
    )
    assert got["state"] == 3 and got["value"] == 3
    # a procedure that acts through its episode's tools is run by no call: with a text, with no arguments or with
    # other arguments, the call refuses, names where its code is, and records no use
    for key in ("env", "env_none", "env_other"):
        name, message = got[key]
        assert name == "WrongCall" and "nothing was run: " in message, got[key]
        assert f"print(memory.get('{cap['env']}').text) shows its code" in message, message
        assert "no call can pass it tools" in message and "exec(" not in message, message
    assert cap["env"] not in got["calls"] and sorted(got["calls"]) == sorted([cap["state"], cap["value"]])


def test_a_card_says_only_what_the_record_shows(tmp_path):
    root, ids = _copy(tmp_path)
    cards = _cell(
        root,
        "print(json.dumps({k: str(memory.get('capture:' + v)).split('\\n\\n', 1)[0] for k, v in "
        + repr(ids)
        + ".items()}))\n",
    )
    assert "re-runs work an earlier request accepted" in cards["state"]
    assert "Calling it with no arguments runs it on the current request" in cards["state"]
    assert "accepted" not in cards["value"] and "no outcome of that request is on record" in cards["value"]
    assert "Calling it with no arguments" not in cards["env"]
    assert "no call can pass it tools, so a call does not run it here" in cards["env"]
    assert f"print(memory.get('capture:{ids['env']}').text) shows its code" in cards["env"]


def test_a_failed_attempts_card_never_says_accepted(tmp_path):
    store = captures.store_dir(tmp_path / "state")
    cid = _put(store, STATE_PROC, "state", "contrary")
    tree = _tree(tmp_path / "lib")
    write_files(tree, lx.generated_library_mode(tree))
    for rel, data in captures.export_files(store).items():
        (tree / rel).parent.mkdir(parents=True, exist_ok=True)
        (tree / rel).write_bytes(data)
    card = _cell(tree, f"print(json.dumps(str(memory.get('capture:{cid}'))))\n")
    assert "its episode failed" in card and "accepted" not in card.split("\n\n", 1)[0]


def test_the_catalogue_states_each_forms_real_reader(tmp_path):
    root, ids = _copy(tmp_path)
    store = captures.store_dir(tmp_path / "state")
    text = lx.catalogue_text(root, captures=captures.export_files(store))

    def card(k):
        return text.split(f"### capture:{ids[k]}\n", 1)[1].split("###", 1)[0]

    assert "reader: the request's state" in card("state")
    assert "acts through the episode's tools" in card("env") and "reader read_input" not in card("env")
    assert "reader read_input" in card("value")


def test_a_writer_item_and_the_captures_it_generalises_link_both_ways(tmp_path):
    from unify.memory.library_helper import ITEMS_FILE

    root, ids = _copy(tmp_path)
    data = json.loads((root / ITEMS_FILE).read_text())
    item = sorted(i for i, r in data["items"].items() if r.get("kind") == "function")[0]
    data["items"][item]["episodes"] = ["ep-1"]  # the episode every fixture capture comes from
    (root / ITEMS_FILE).write_text(json.dumps(data))
    got = _cell(
        root,
        f"print(json.dumps({{'item': str(memory.get({item!r})), "
        f"'cap': str(memory.get('capture:{ids['state']}'))}}))\n",
    )
    line = next(x for x in got["item"].splitlines() if x.startswith("Generalises: "))
    assert f"capture:{ids['state']}" in line and f"capture:{ids['value']}" in line
    assert f"Generalised by: {item}" in got["cap"]
