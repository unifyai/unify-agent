"""A pass's transcripts are stored whole: every message, never cut, redacted, in chunk files under the pass record
folder with a list of their sha256; the git note carries the lines while they fit, else a pointer row.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from unify.memory import transcripts
from unify.memory.redact import Redactor
from unify.memory.writer_pass import WriterPass, transcript_lines


def read_transcript(root, pass_id: str, agent: str = transcripts.WRITER) -> list[str]:
    """The lines of a transcript :func:`unify.memory.transcripts.write_chunks` stored, chunk by chunk, each checked
    against its sha256 in the list (a :class:`ValueError` names a chunk that does not match)."""
    folder = Path(root) / pass_id
    prefix = "transcript" if agent == transcripts.WRITER else f"transcript-{agent}"
    manifest = json.loads((folder / f"{prefix}.json").read_bytes())
    lines: list[str] = []
    for chunk in manifest["chunks"]:
        data = (folder / chunk["name"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != chunk["sha256"]:
            raise ValueError(f"transcript chunk {chunk['name']} does not match its sha256")
        lines.extend(data.decode("utf-8").splitlines())
    return lines

SECRET = "planted-fake-token-91c0e2aa"  # pragma: allowlist secret


def _messages(n: int) -> list[dict]:
    msgs = [
        {"role": "system", "content": "the brief " * 50},
        {"role": "user", "content": f"Pass p1 over episodes; token {SECRET}"},
    ]
    while len(msgs) < n:
        i = len(msgs)
        msgs.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"c{i}",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": json.dumps({"path": f"/inputs/{i}"}),
                        },
                    },
                ],
            },
        )
        # results far longer than the bounded view's per-string cut, some with the secret in them
        body = f"{i}:" + ("z" * 9_000) + (f" {SECRET}" if i % 7 == 0 else "")
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": body})
    return msgs[:n]


def _store() -> tuple[dict, callable]:
    blobs: dict[str, bytes] = {}

    def put(data: bytes) -> str:
        sha = hashlib.sha256(data).hexdigest()
        blobs[sha] = data
        return sha

    return blobs, put


def test_a_500_message_transcript_is_stored_whole_and_reads_back_identical(tmp_path):
    msgs = _messages(500)
    red = Redactor({"FAKE_TOKEN": SECRET})
    _, put = _store()
    manifest = transcripts.record_transcript(
        tmp_path,
        "p1",
        "writer",
        msgs,
        redactor=red,
        store=put,
    )
    expected = transcript_lines(msgs, "p1", Redactor({"FAKE_TOKEN": SECRET}), store=put)
    assert len(manifest["chunks"]) > 1  # several pages
    assert manifest["lines"] == len(expected) == 501  # the header and every message
    # chunk by chunk: each file's digest is the listed one, and the pages put together are the lines
    folder = tmp_path / "p1"
    listed = json.loads((folder / "transcript.json").read_text())
    assert listed == manifest
    paged: list[str] = []
    for chunk in manifest["chunks"]:
        data = (folder / chunk["name"]).read_bytes()
        assert (
            hashlib.sha256(data).hexdigest() == chunk["sha256"]
            and len(data) == chunk["bytes"]
        )
        page = data.decode().splitlines()
        assert len(page) == chunk["lines"]
        paged += page
    assert paged == expected == read_transcript(tmp_path, "p1")
    rows = [json.loads(ln) for ln in paged]
    assert [r["i"] for r in rows[1:]] == list(
        range(500),
    )  # no message dropped, none marked truncated
    assert not any("truncated" in r for r in rows)
    # nothing cut: each tool result is whole (redacted)
    assert rows[16]["message"]["content"].startswith(
        "14:" + "z" * 9_000,
    )  # message 15, a tool result
    text = "\n".join(paged)
    assert SECRET not in text and "<secret:FAKE_TOKEN>" in text


def test_a_changed_chunk_is_named_on_read(tmp_path):
    lines = [json.dumps({"i": i, "x": "y" * 100}) for i in range(50)]
    manifest = transcripts.write_chunks(
        tmp_path,
        "p2",
        "writer",
        lines,
        chunk_bytes=1_000,
    )
    assert (
        len(manifest["chunks"]) > 2
        and read_transcript(tmp_path, "p2") == lines
    )
    victim = tmp_path / "p2" / manifest["chunks"][1]["name"]
    victim.write_text(victim.read_text().replace("y", "Y", 1))
    with pytest.raises(ValueError, match=manifest["chunks"][1]["name"]):
        read_transcript(tmp_path, "p2")


def test_a_line_longer_than_a_page_is_one_whole_chunk(tmp_path):
    lines = ["a" * 50, "b" * 5_000, "c" * 50]
    manifest = transcripts.write_chunks(
        tmp_path,
        "p3",
        "writer",
        lines,
        chunk_bytes=1_000,
    )
    assert [c["lines"] for c in manifest["chunks"]] == [1, 1, 1]
    assert read_transcript(tmp_path, "p3") == lines


def test_a_helper_uses_the_same_mechanism_beside_the_writer(tmp_path):
    w = transcripts.write_chunks(tmp_path, "p4", "writer", ["{}"])
    h = transcripts.record_transcript(tmp_path, "p4", "helper-1", _messages(9))
    names = sorted(p.name for p in (tmp_path / "p4").iterdir())
    assert names == [
        "transcript-0.jsonl",
        "transcript-helper-1-0.jsonl",
        "transcript-helper-1.json",
        "transcript.json",
    ]
    assert (w["agent"], h["agent"]) == ("writer", "helper-1") and h["lines"] == 10
    assert (
        read_transcript(tmp_path, "p4", "helper-1")[1:]
        == transcript_lines(_messages(9), "p4")[1:]
    )


def test_a_recorded_transcript_is_never_written_over_and_names_are_checked(tmp_path):
    transcripts.write_chunks(tmp_path, "p5", "writer", ["{}"])
    with pytest.raises(OSError):
        transcripts.write_chunks(tmp_path, "p5", "writer", ['{"other": 1}'])
    assert read_transcript(tmp_path, "p5") == ["{}"]
    for agent in ("../up", "a/b", "", ".hidden"):
        with pytest.raises(ValueError):
            transcripts.write_chunks(tmp_path, "p6", agent, ["{}"])
    with pytest.raises(ValueError):
        transcripts.write_chunks(tmp_path, "../p7", "writer", ["{}"])


def test_the_note_keeps_the_lines_while_they_fit_else_a_pointer_row(tmp_path):
    small = ['{"i": 0}', '{"i": 1}']
    manifest = transcripts.write_chunks(tmp_path, "p8", "writer", small)
    assert transcripts.note_lines("p8", small, manifest) == small
    big = [json.dumps({"i": i, "x": "y" * 1_000}) for i in range(400)]
    manifest = transcripts.write_chunks(
        tmp_path,
        "p9",
        "writer",
        big,
        chunk_bytes=100_000,
    )
    (row,) = transcripts.note_lines("p9", big, manifest)
    pointer = json.loads(row)
    assert pointer["pass_id"] == "p9"
    t = pointer["transcript"]
    assert (
        t["folder"] == "p9" and t["manifest"] == "transcript.json" and t["lines"] == 400
    )
    assert [(c["name"], c["sha256"]) for c in t["chunks"]] == [
        (c["name"], c["sha256"]) for c in manifest["chunks"]
    ]
    # no chunk files: the note carries every line, whatever the size
    assert transcripts.note_lines("p9", big, None) == big


def test_the_writer_note_points_to_its_whole_transcript(tmp_path):
    msgs = _messages(500)
    _, put = _store()
    lines = transcript_lines(msgs, "p10", Redactor({"FAKE_TOKEN": SECRET}), store=put)

    def writer(root):
        return SimpleNamespace(
            cfg=SimpleNamespace(pass_record_root=root),
            transcript=lambda pid: list(lines),
        )

    (row,) = WriterPass.transcript_note(writer(str(tmp_path)), "p10")
    assert json.loads(row)["transcript"]["lines"] == 501
    assert read_transcript(tmp_path, "p10") == lines
    # no record folder: the note itself carries the whole transcript
    assert WriterPass.transcript_note(writer(None), "p10") == lines
