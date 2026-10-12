"""The writer's and the curator's ``write`` and ``edit`` file tools on /memory: path rules, links planted before a
call and swapped in during one, the file mode, non-regular targets, exact single matches, the switch, the pass loop
(tool events and the quota measure), and the prompts with the switch off and on."""

import asyncio
import json
import os
import stat
from pathlib import Path

import pytest

from unify.memory import agent_profiles, edit_tools, helpers, tool_spec
from unify.memory import library_prompts as lp
from unify.memory.index import estimate_tokens
from unify.memory.integration import switch
from unify.memory.trigger import PassRequest
from unify.memory.writer_pass import QUOTA_FILE_BYTES, PassConfig, writer_tools
from unify.settings import SETTINGS
from tests.memory.test_writer_library_tools import EPS
from tests.memory.test_writer_pass import Turns, _call, _writer


@pytest.fixture
def box(tmp_path):
    b = tmp_path / "memory"
    (b / "memory" / "pkg").mkdir(parents=True)
    (b / ".pass").mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("keep\n")
    return b


def _outside(box: Path) -> Path:
    return box.parent / "outside"


def _w(box, path, content="x = 1\n"):
    return edit_tools.call("write", {"path": path, "content": content}, box)


def _e(box, path, old, new):
    return edit_tools.call("edit", {"path": path, "old": old, "new": new}, box)


# --- path rules ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,form",
    [
        ("/memory/memory/pkg/a.py", "bad_path"),
        ("/etc/passwd", "bad_path"),
        ("../outside/secret.txt", "bad_path"),
        ("memory/pkg/../../x.py", "bad_path"),
        ("memory//pkg/a.py", "bad_path"),
        ("memory/pkg/", "bad_path"),
        ("./memory/pkg/a.py", "bad_path"),
        ("", "bad_path"),
        ("memory/pkg/a\x00.py", "bad_path"),
        ("memory/pkg/\udcff.py", "bad_path"),
        (".git/config", "reserved"),
        ("memory/pkg/.git/x", "reserved"),
        (".gitattributes", "reserved"),
        ("INDEX.md", "reserved"),
        ("links.json", "reserved"),
        ("memory/__init__.py", "reserved"),
        (".memory/data.json", "reserved"),
        (".pass/fixtures.json", "reserved"),
        (".pass/other/manifest.json", "reserved"),
    ],
)
def test_paths_outside_the_rules_are_refused_and_nothing_is_written(box, path, form):
    before = sorted(str(p) for p in box.parent.rglob("*"))
    for got in (_w(box, path), _e(box, path, "a", "b")):
        assert got.status == "refused" and got.form == form, (path, got.text)
        assert got.text.startswith("refused: ")
    assert sorted(str(p) for p in box.parent.rglob("*")) == before


def test_a_non_string_path_or_content_is_refused(box):
    assert edit_tools.call("write", {"path": 3, "content": "x"}, box).form == "bad_path"
    assert (
        edit_tools.call("write", {"path": "memory/pkg/a.py"}, box).form == "bad_content"
    )
    assert _w(box, "memory/pkg/a.py", "\udcff").form == "bad_content"
    assert not (box / "memory/pkg/a.py").exists()


def test_the_manifest_is_the_one_writable_file_under_pass(box):
    got = _w(box, ".pass/manifest.json", '{"items": []}\n')
    assert got.status == "ok", got.text
    assert (box / ".pass/manifest.json").read_text() == '{"items": []}\n'
    assert _e(box, ".pass/manifest.json", "[]", '["a"]').status == "ok"
    assert json.loads((box / ".pass/manifest.json").read_text()) == {"items": ["a"]}


# --- writes and edits read back ------------------------------------------------------------------------------


def test_a_write_makes_missing_directories_and_reads_back(box):
    got = _w(box, "memory/text/tests/data/in.json", '{"a": 1}\n')
    assert got.status == "ok" and got.text.startswith(
        "ok: created memory/text/tests/data/in.json",
    )
    assert (box / "memory/text/tests/data/in.json").read_text() == '{"a": 1}\n'
    assert got.paths == (
        {"path": "/memory/memory/text/tests/data/in.json", "mode": "write"},
    )
    again = _w(box, "memory/text/tests/data/in.json", "")
    assert (
        again.text.startswith("ok: replaced")
        and (box / "memory/text/tests/data/in.json").read_text() == ""
    )
    # no temporary file is left behind
    assert sorted(p.name for p in (box / "memory/text/tests/data").iterdir()) == [
        "in.json",
    ]


def test_an_edit_replaces_the_one_occurrence_and_reads_back(box):
    _w(box, "memory/pkg/a.py", "def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    got = _e(box, "memory/pkg/a.py", "    return 1\n", "    return 10\n")
    assert got.status == "ok", got.text
    assert (
        box / "memory/pkg/a.py"
    ).read_text() == "def f():\n    return 10\n\n\ndef g():\n    return 2\n"


def test_edits_with_zero_or_several_matches_are_refused_with_the_count(box):
    _w(box, "memory/pkg/a.py", "x = 1\nx = 1\ny = 2\n")
    none = _e(box, "memory/pkg/a.py", "z = 3", "z = 4")
    assert none.form == "match_count" and "occurs 0 times" in none.text
    two = _e(box, "memory/pkg/a.py", "x = 1", "x = 5")
    assert two.form == "match_count" and "occurs 2 times" in two.text
    # overlapping occurrences count too: "aa" occurs twice in "aaa"
    _w(box, "memory/pkg/b.py", "aaa")
    assert "occurs 2 times" in _e(box, "memory/pkg/b.py", "aa", "b").text
    # exact text, never a pattern
    assert _e(box, "memory/pkg/a.py", "x = .", "q").form == "match_count"
    assert (box / "memory/pkg/a.py").read_text() == "x = 1\nx = 1\ny = 2\n"


def test_edit_refuses_missing_files_bytes_that_are_not_text_and_empty_or_unchanged_text(
    box,
):
    assert _e(box, "memory/pkg/none.py", "a", "b").form == "not_found"
    assert _e(box, "memory/nope/none.py", "a", "b").form == "not_found"
    (box / "memory/pkg/bin.dat").write_bytes(b"\xff\xfe\x00a")
    assert _e(box, "memory/pkg/bin.dat", "a", "b").form == "not_text"
    _w(box, "memory/pkg/a.py", "x = 1\n")
    assert _e(box, "memory/pkg/a.py", "", "b").form == "bad_old"
    assert _e(box, "memory/pkg/a.py", "x", "x").form == "bad_old"
    assert (
        edit_tools.call(
            "edit",
            {"path": "memory/pkg/a.py", "old": 1, "new": "2"},
            box,
        ).form
        == "bad_old"
    )


# --- modes and non-regular targets ---------------------------------------------------------------------------


def test_every_written_file_is_0644_and_the_execute_bit_never_survives(box):
    target = box / "memory/pkg/run.py"
    target.write_text("print(1)\n")
    target.chmod(0o755)
    old = os.umask(0o000)
    try:
        assert _w(box, "memory/pkg/run.py", "print(2)\n").status == "ok"
        assert stat.S_IMODE(target.stat().st_mode) == 0o644
        target.chmod(0o777)
        assert _e(box, "memory/pkg/run.py", "2", "3").status == "ok"
        assert stat.S_IMODE(target.stat().st_mode) == 0o644
        assert _w(box, "memory/pkg/new.py").status == "ok"
        assert stat.S_IMODE((box / "memory/pkg/new.py").stat().st_mode) == 0o644
    finally:
        os.umask(old)


def test_a_target_that_is_not_a_regular_file_is_refused(box):
    (box / "memory/pkg/dir.py").mkdir()
    os.mkfifo(box / "memory/pkg/pipe.py")
    for path in ("memory/pkg/dir.py", "memory/pkg/pipe.py"):
        assert _w(box, path).form == "not_regular", path
        assert _e(box, path, "a", "b").form == "not_regular", path
    assert (box / "memory/pkg/dir.py").is_dir() and stat.S_ISFIFO(
        os.lstat(box / "memory/pkg/pipe.py").st_mode,
    )
    (box / "memory/file").write_text("x")
    assert _w(box, "memory/file/a.py").form == "not_directory"
    assert _e(box, "memory/file/a.py", "a", "b").form == "not_directory"


# --- links ---------------------------------------------------------------------------------------------------


def test_a_link_at_the_leaf_is_refused_and_its_target_untouched(box):
    secret = _outside(box) / "secret.txt"
    (box / "memory/pkg/a.py").symlink_to(secret)
    (box / "memory/pkg/dangling.py").symlink_to(_outside(box) / "new.txt")
    assert _w(box, "memory/pkg/a.py", "pwned").form == "link"
    assert _e(box, "memory/pkg/a.py", "keep", "pwned").form == "link"
    assert _w(box, "memory/pkg/dangling.py", "pwned").form == "link"
    assert secret.read_text() == "keep\n" and not (_outside(box) / "new.txt").exists()
    assert (box / "memory/pkg/a.py").is_symlink()


def test_a_link_at_a_parent_is_refused_and_nothing_is_written_through_it(box):
    (box / "memory/linked").symlink_to(_outside(box))
    for got in (
        _w(box, "memory/linked/secret.txt", "pwned"),
        _w(box, "memory/linked/new.txt", "pwned"),
    ):
        assert got.form == "link" and "memory/linked" in got.text
    assert _e(box, "memory/linked/secret.txt", "keep", "pwned").form == "link"
    assert sorted(p.name for p in _outside(box).iterdir()) == ["secret.txt"]
    assert (_outside(box) / "secret.txt").read_text() == "keep\n"


def test_a_parent_swapped_for_a_link_between_check_and_open_is_never_followed(
    box,
    monkeypatch,
):
    """A cell turns memory/pkg into a link at the moment the tool opens it: the open itself refuses it."""
    real = edit_tools._openat
    (box / "memory/pkg/a.py").write_text("keep = 1\n")

    def swap(name, flags, dir_fd, mode=0o777):
        if name == "pkg":
            os.rename("pkg", "pkg-moved", src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
            os.symlink(str(_outside(box)), "pkg", dir_fd=dir_fd)
        return real(name, flags, dir_fd, mode)

    monkeypatch.setattr(edit_tools, "_openat", swap)
    assert _w(box, "memory/pkg/secret.txt", "pwned").form == "link"
    monkeypatch.setattr(edit_tools, "_openat", real)
    (box / "memory/pkg").unlink()
    (box / "memory/pkg-moved").rename(box / "memory/pkg")
    monkeypatch.setattr(edit_tools, "_openat", swap)
    assert _e(box, "memory/pkg/secret.txt", "keep", "pwned").form == "link"
    assert (_outside(box) / "secret.txt").read_text() == "keep\n"
    assert sorted(p.name for p in _outside(box).iterdir()) == ["secret.txt"]


def test_a_leaf_swapped_for_a_link_after_the_check_is_replaced_never_written_through(
    box,
    monkeypatch,
):
    """write: the leaf becomes a link after its check; the rename replaces the link itself."""
    secret = _outside(box) / "secret.txt"
    real_kind = edit_tools._leaf_kind

    def swap_after_check(dir_fd, leaf):
        got = real_kind(dir_fd, leaf)
        os.symlink(str(secret), leaf, dir_fd=dir_fd)
        return got

    monkeypatch.setattr(edit_tools, "_leaf_kind", swap_after_check)
    got = _w(box, "memory/pkg/a.py", "mine\n")
    assert got.status == "ok", got.text
    assert secret.read_text() == "keep\n"
    leaf = box / "memory/pkg/a.py"
    assert not leaf.is_symlink() and leaf.read_text() == "mine\n"


def test_an_edit_leaf_swapped_for_a_link_before_it_is_opened_is_refused(
    box,
    monkeypatch,
):
    secret = _outside(box) / "secret.txt"
    (box / "memory/pkg/a.py").write_text("keep\n")
    real = edit_tools._openat

    def swap(name, flags, dir_fd, mode=0o777):
        if name == "a.py":
            os.unlink("a.py", dir_fd=dir_fd)
            os.symlink(str(secret), "a.py", dir_fd=dir_fd)
        return real(name, flags, dir_fd, mode)

    monkeypatch.setattr(edit_tools, "_openat", swap)
    assert _e(box, "memory/pkg/a.py", "keep", "pwned").form == "link"
    assert secret.read_text() == "keep\n"


def test_a_hard_linked_file_is_replaced_and_its_other_name_keeps_its_bytes(box):
    secret = _outside(box) / "secret.txt"
    try:
        os.link(secret, box / "memory/pkg/hard.txt")
    except OSError:
        pytest.skip("no hard links here")
    assert _w(box, "memory/pkg/hard.txt", "mine\n").status == "ok"
    assert _e(box, "memory/pkg/hard.txt", "mine", "ours").status == "ok"
    assert (
        secret.read_text() == "keep\n"
        and (box / "memory/pkg/hard.txt").read_text() == "ours\n"
    )


# --- the spec and the switch ---------------------------------------------------------------------------------


def test_the_tools_exist_only_with_the_switch_and_only_for_the_writer_and_the_curator():
    on = frozenset({"edit"})
    for role in ("write", "curate"):
        assert not {"write", "edit"} & {s.name for s in tool_spec.tools_for(role)}
        assert {"write", "edit"} <= {s.name for s in tool_spec.tools_for(role, on)}
    assert not {"write", "edit"} & {s.name for s in tool_spec.tools_for("helper", on)}
    for agent in ("writer", "curator"):
        assert {"write", "edit"} <= set(agent_profiles.tool_names(agent, on))
        assert not {"write", "edit"} & set(agent_profiles.tool_names(agent))
    for agent in ("actor", "fork", "gate", "harness", "helper"):
        assert not {"write", "edit"} & set(agent_profiles.tool_names(agent, on)), agent
    for role in ("write", "curate"):
        off = writer_tools(library_mode=True, role=role)
        got = writer_tools(library_mode=True, role=role, edit_tools=True)
        assert [t for t in got if t["function"]["name"] not in ("write", "edit")] == off
        assert [t["function"]["name"] for t in got][-2:] == ["write", "edit"]
    names = {t["function"]["name"] for t in helpers.helper_tools(True)}
    assert not {"write", "edit"} & names


def test_the_setting_is_off_by_default_and_parses_on_and_off():
    assert SETTINGS.UNIFY_MEMORY_LIBRARY_EDIT_TOOLS == "off"
    assert PassConfig().edit_tools is False
    assert switch.PARSERS[switch.LIBRARY_EDIT_TOOLS] is switch.parse_library_edit_tools
    assert switch.parse_library_edit_tools("") == "off"
    assert switch.parse_library_edit_tools(" ON ") == "on"
    with pytest.raises(ValueError, match="UNIFY_MEMORY_LIBRARY_EDIT_TOOLS"):
        switch.parse_library_edit_tools("yes")

    class S:
        UNIFY_MEMORY_LIBRARY_EDIT_TOOLS = "on"

    assert switch.library_edit_tools(S()) is True
    S.UNIFY_MEMORY_LIBRARY_EDIT_TOOLS = ""
    assert switch.library_edit_tools(S()) is False


def test_the_tool_texts_are_not_repeated_in_the_briefs():
    on = lp.tool_flags(edit=True)
    for role, brief in (
        ("write", lp.write_brief_now(edit_tools=True)),
        ("curate", lp.curate_brief_now(edit_tools=True)),
    ):
        section = tool_spec.tools_section(role, on)
        assert "- `write`: " + edit_tools.WRITE.summary in section
        assert "- `edit`: " + edit_tools.EDIT.summary in section
        for spec in edit_tools.SPECS:
            assert spec.description not in brief
            # the description's sentences appear nowhere in the brief; the Tools bullet is the brief's one mention
            for sentence in spec.description.split(". "):
                assert sentence.strip(". ") not in brief.replace(section, ""), sentence
        assert lp.benchmark_words(brief) == [] and lp.example_checks(brief) == []
        assert not any(w in brief.lower() for w in lp.BANNED_WORDS)


# --- the pass loop -------------------------------------------------------------------------------------------


def _library_pass(tmp_path, turns, **cfg):
    model = Turns(turns)
    _, _, writer = _writer(
        tmp_path,
        model,
        library_mode=True,
        max_calls=20,
        pass_record_root=str(tmp_path / "out"),
        **cfg,
    )
    writer.load = EPS.__getitem__
    out = asyncio.run(
        writer.run(PassRequest("incremental", "svc", ["e1"], False), "p1"),
    )
    return out, model, writer


def test_the_pass_runs_write_and_edit_and_records_each_as_a_tool_event(tmp_path):
    path = "memory/text/clean.py"
    turns = [
        _call(
            "w",
            "write",
            {"path": path, "content": "def clean(s):\n    return s.strip()\n"},
        ),
        _call(
            "e",
            "edit",
            {"path": path, "old": "s.strip()", "new": "s.strip().lower()"},
        ),
        _call("r", "read", {"path": f"/memory/{path}"}),
        _call("z", "edit", {"path": path, "old": "nothing here", "new": "x"}),
        _call("l", "write", {"path": "../escape.py", "content": "x"}),
    ]
    out, model, writer = _library_pass(tmp_path, turns, edit_tools=True)
    assert model.outputs["w"].startswith("ok: created memory/text/clean.py")
    assert model.outputs["e"].startswith("ok: edited memory/text/clean.py")
    assert "return s.strip().lower()" in model.outputs["r"]
    assert "occurs 0 times" in model.outputs["z"]
    assert model.outputs["l"].startswith("refused: ")
    rows = {
        r["tool"]: r
        for r in writer.tool_events
        if r["tool"] in ("write", "edit") and r["status"] == "ok"
    }
    assert rows["write"]["paths"] == [{"path": f"/memory/{path}", "mode": "write"}]
    assert rows["edit"]["agent"] == "writer"
    refused = [
        r
        for r in writer.tool_events
        if r["tool"] in ("write", "edit") and r["status"] == "refused"
    ]
    assert [r["refusal"]["form"] for r in refused] == ["match_count", "bad_path"]
    codes = json.loads((tmp_path / "out" / "p1" / "refusal-codes.json").read_text())
    assert {"match_count", "link", "bad_path"} <= set(codes["edit"])


def test_with_the_switch_off_the_tools_are_unknown(tmp_path):
    out, model, writer = _library_pass(
        tmp_path,
        [_call("w", "write", {"path": "memory/text/a.py", "content": "x"})],
    )
    assert "unknown tool 'write'" in model.outputs["w"]
    assert "write" not in {
        t["function"]["name"] for t in writer_tools(library_mode=True)
    }


def test_a_write_over_the_file_quota_refuses_the_pass_as_a_cell_breach_does(tmp_path):
    big = "x" * (QUOTA_FILE_BYTES + 1)
    turn = {
        "role": "assistant",
        "tool_calls": [
            _call("w", "write", {"path": "memory/text/big.txt", "content": big})[
                "tool_calls"
            ][0],
            _call("n", "write", {"path": "memory/text/a.py", "content": "x"})[
                "tool_calls"
            ][0],
        ],
    }
    out, _, writer = _library_pass(tmp_path, [turn], edit_tools=True)
    replies = {
        m["tool_call_id"]: m["content"]
        for m in writer.messages
        if m.get("role") == "tool"
    }
    assert "over quota: /memory/memory/text/big.txt" in replies["w"]
    assert "the pass is refused" in replies["w"]
    assert replies["n"].startswith("not run: over quota")
    assert not out.passed and "over_quota" in out.codes


# --- prompts with the switch off and on ----------------------------------------------------------------------


def _actor_sections(tmp_path):
    from unify.memory.integration.prompt import render_memory_library_mode

    checkout = tmp_path / "copy"
    checkout.mkdir(exist_ok=True)
    return [
        render_memory_library_mode(checkout, actor_index=form)[0]
        for form in ("pointer", "inline")
    ]


def test_the_actor_prompt_is_byte_identical_with_the_switch_off_and_on(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_EDIT_TOOLS", "off")
    off = (
        _actor_sections(tmp_path),
        lp.guide_library_mode(),
        agent_profiles.tool_names("actor"),
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_MEMORY_LIBRARY_EDIT_TOOLS", "on")
    on = (
        _actor_sections(tmp_path),
        lp.guide_library_mode(),
        agent_profiles.tool_names("actor", lp.tool_flags(edit=True)),
    )
    assert on == off
    assert agent_profiles.profile("actor")["tools"] == []


def _sizes(tmp_path, full: bool) -> dict:
    """The rendered brief, tool JSON and first message of each agent, switch off and on (chars, tokens)."""
    out = {}
    for on in (False, True):
        sub = tmp_path / f"{'full' if full else 'base'}-{'on' if on else 'off'}"
        sub.mkdir()
        cfg = dict(library_mode=True, edit_tools=on)
        if full:
            cfg.update(fork=True, pool=True, similar_index=str(sub / "similar"))
        _, _, writer = _writer(sub, Turns([]), **cfg)
        if full:
            writer.helper_turn = object()
        for agent, role, req in (
            ("writer", "write", PassRequest("incremental", "svc", ["e1", "e2"], False)),
            ("curator", "curate", PassRequest("curate", "svc", [], False)),
        ):
            writer._role = role
            texts = {
                "brief": writer._brief_library_mode(),
                "tools": json.dumps(
                    writer_tools(
                        library_mode=True,
                        role=role,
                        pool=full,
                        helper=full,
                        similar=full,
                        edit_tools=on,
                    ),
                ),
                "first": writer._first_message_library_mode(req, "p1"),
            }
            out[(agent, on)] = texts
        out[("helper", on)] = {
            "brief": helpers.helper_brief(full),
            "tools": json.dumps(helpers.helper_tools(full)),
            "first": (
                "Task from the writer:\nread e1 and draft a parser\n\nEpisodes to start from: e1\n"
                "Time you have: about 600 s; then you hand back what you have staged. "
                "Operational step guard: 500 model calls."
            ),
        }
        out[("actor", on)] = {
            # the same library copy both times: the section names the copy's location
            "brief": "\n".join(_actor_sections(tmp_path)),
            "tools": json.dumps(
                agent_profiles.tool_names("actor", lp.tool_flags(edit=on)),
            ),
            "first": "",
        }
    return out


@pytest.mark.parametrize("full", [False, True])
def test_prompt_sizes_with_the_switch_off_and_on(tmp_path, full):
    sizes = _sizes(tmp_path, full)
    lines = [
        f"prompt sizes ({'every optional input on' if full else 'optional inputs off'}): chars/tokens",
    ]
    for agent in ("writer", "curator", "helper", "actor"):
        off, on = sizes[(agent, False)], sizes[(agent, True)]
        cells = []
        for part in ("brief", "tools", "first"):
            a, b = off[part], on[part]
            cells.append(
                f"{part} {len(a)}/{estimate_tokens(a)} -> {len(b)}/{estimate_tokens(b)}",
            )
        lines.append(f"{agent}: " + "; ".join(cells))
        # the helper and the actor never change; the first messages never change
        assert off["first"] == on["first"], agent
        if agent in ("helper", "actor"):
            assert off == on, agent
        else:
            # the brief grows by its two Tools bullets only
            bullets = f"- `write`: {edit_tools.WRITE.summary}\n- `edit`: {edit_tools.EDIT.summary}"
            assert len(on["brief"]) - len(off["brief"]) == len(bullets) + 1, agent
            assert bullets in on["brief"] and bullets not in off["brief"]
    print("\n".join(lines))
