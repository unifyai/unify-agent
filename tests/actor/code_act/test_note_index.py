"""Symbolic: ``UNIFY_NOTE_INDEX``, the notes written for the closest earlier requests, with their functions attached.

Notes are ranked by the request each was written for, not by their own
text; a function no note links stands in as a note of its own, built at read
time; the functions the top notes link are listed and bound, never called.
The embedder and the libraries are stand-ins: nothing leaves the process.
"""

from __future__ import annotations

import numpy as np
import pytest

from unify.actor import note_index as ni
from unify.function_manager import task_origin
from unify.settings import ProductionSettings, SETTINGS

_CHECKER_OUTCOME = ni.checker_outcome


def _fn(fid, name, doc, origin=""):
    metadata = {task_origin.REQUESTS_FIELD: [origin]} if origin else {}
    return {
        "function_id": fid,
        "name": name,
        "argspec": "(x)",
        "docstring": doc,
        "implementation": f"def {name}(x):\n    raise RuntimeError('called')\n",
        "metadata": metadata,
        "usage_calls": 0,
    }


def _note(gid, title, content, function_ids=(), origin=""):
    metadata = {task_origin.REQUESTS_FIELD: [origin]} if origin else {}
    return {
        "guidance_id": gid,
        "title": title,
        "content": content,
        "function_ids": list(function_ids),
        "metadata": metadata,
    }


class _Manager:
    """A library that only reads: any other attribute (a write) fails the test."""

    def __init__(self, rows):
        self._rows_ = rows

    def _evidence_rows(self):
        return list(self._rows_)

    def __getattr__(self, name):
        raise AssertionError(f"the note index touched {name}")


CONCEPTS = ["payment", "total", "trip", "refund", "invoice", "weather"]


def _embed(texts):
    out = []
    for t in texts:
        low = t.lower()
        v = np.array([0.1] + [float(c in low) for c in CONCEPTS], dtype=np.float32)
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_NOTE_INDEX", True)


@pytest.fixture(autouse=True)
def outcomes(monkeypatch):
    """The checker's outcome of each writer's session, by its request (unknown unless set)."""
    kept: dict[str, bool] = {}
    by_key = lambda: {task_origin.text_key(t): v for t, v in kept.items()}
    monkeypatch.setattr(ni, "checker_outcome", lambda key: by_key().get(key))
    return kept


@pytest.fixture
def warnings(monkeypatch):
    """The note index's warnings (unify's loggers do not reach pytest's capture)."""
    seen: list[str] = []
    monkeypatch.setattr(
        ni.logger,
        "warning",
        lambda msg, *a, **k: seen.append(str(msg)),
    )
    return seen


class _Binder:
    """Binds each name to a callable that fails the test if anything calls it."""

    def __init__(self):
        self.names: list[list[str]] = []
        self.namespace: dict = {}
        self.calls: list[str] = []

    def __call__(self, names):
        self.names.append(list(names))
        for name in names:

            def boom(*a, _name=name, **k):
                self.calls.append(_name)
                raise AssertionError(f"{_name} was called")

            self.namespace[name] = boom
        return {name: False for name in names}


def _section(functions, notes, request, **kw):
    kw.setdefault("embed", _embed)
    return ni.section(_Manager(functions), _Manager(notes), request, **kw)


def _headings(text):
    return [ln for ln in text.splitlines() if ln.startswith("### ")]


# ── off ──────────────────────────────────────────────────────────────────


def test_the_switch_is_off_by_default():
    assert ProductionSettings.model_fields["UNIFY_NOTE_INDEX"].default is False
    assert ni.enabled() is False


def test_off_no_embedding_no_section_and_nothing_bound(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_NOTE_INDEX", False)

    def boom(*_a, **_k):
        raise AssertionError("used while off")

    notes = [_note(1, "Payments", "Sum them.", origin="Total my payments.")]
    assert (
        ni.section(
            _Manager([]),
            _Manager(notes),
            "Total my payments.",
            embed=boom,
            bind=boom,
        )
        == ""
    )


# ── ranking ──────────────────────────────────────────────────────────────


def test_a_note_written_for_a_matching_request_outranks_one_whose_text_matches(on):
    notes = [
        # Its text is about the request; it was written for a trip.
        _note(
            1,
            "Total card payments",
            "Total payment amounts.",
            origin="Plan my trip.",
        ),
        # Its text is not; it was written for this kind of request.
        _note(
            2,
            "Weather check",
            "Look up the weather.",
            origin="Total my card payments.",
        ),
    ]
    text = _section([], notes, "Total my card payments for May.")
    heads = _headings(text)
    assert heads[0] == "### Note 2: Weather check"
    assert heads[1] == "### Note 1: Total card payments"


def test_shared_words_never_outrank_a_close_embedding(on):
    """Ranking is the embedding cosine alone: no word or identifier overlap."""
    request = "Send invoice inv-2024q3 to the finance team by Friday."
    wordy = "Send invoice inv-2024q3 to the finance team by Friday, as a poem."
    close = "Email accounting the Q3 bill."
    table = {
        request: [1.0, 0.0, 0.0],
        wordy: [0.0, 1.0, 0.0],  # nearly every word and the identifier shared; far
        close: [0.9, 0.1, 0.0],  # hardly a word shared; close
    }

    def embed(texts):
        out = np.array([table[t] for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)

    notes = [
        _note(1, "Close", "x", origin=close),
        _note(2, "Wordy", "y", origin=wordy),  # newer, so it would win a tie
    ]
    text = _section([], notes, request, embed=embed)
    assert _headings(text) == ["### Note 1: Close", "### Note 2: Wordy"]


def test_ties_go_to_the_newest_entry(on):
    notes = [
        _note(1, "Older", "x", origin="Total my payment."),
        _note(2, "Newer", "y", origin="Total my payment."),
    ]
    text = _section([], notes, "Total my payment.")
    assert _headings(text) == ["### Note 2: Newer", "### Note 1: Older"]


def test_a_note_scores_its_best_origin_request(on):
    note = _note(1, "Refunds", "Refund rules.")
    note["metadata"] = {
        task_origin.REQUESTS_FIELD: ["Plan my trip.", "Refund my invoice."],
    }
    other = _note(2, "Payments", "x", origin="Total my payment.")
    selected = ni.rank(
        ni.build_index([], [note, other]),
        "Refund invoice 7.",
        embed=_embed,
    )
    assert [n.row["guidance_id"] for n, _ in selected] == [1, 2]
    assert selected[0][1] == pytest.approx(
        float(np.dot(*_embed(["Refund invoice 7.", "Refund my invoice."]))),
    )


def test_only_the_top_three_are_shown(on):
    requests = [
        "Total my payment and refund my invoice.",
        "Total my payment and refund.",
        "Total my payment.",
        "Plan a trip.",
        "Check the weather.",
    ]
    notes = [
        _note(i + 1, f"Note {i + 1}", "x", origin=r) for i, r in enumerate(requests)
    ]
    text = _section([], notes, "Total my payment and refund my invoice.")
    assert _headings(text) == [
        "### Note 1: Note 1",
        "### Note 2: Note 2",
        "### Note 3: Note 3",
    ]
    assert ni.K_NOTES == 3


def test_a_note_with_no_origin_request_is_not_ranked(on):
    notes = [
        _note(1, "Total payments", "Total payment amounts."),
        _note(2, "Trips", "Plan trips.", origin="Plan my trip."),
    ]
    text = _section([], notes, "Total my payments.")
    assert _headings(text) == ["### Note 2: Trips"]


# ── attachment ───────────────────────────────────────────────────────────


def test_the_linked_functions_are_listed_and_bound_and_never_called(on):
    functions = [
        _fn(1, "total_payments", "Total card payments.", origin="Total my payments."),
        _fn(2, "refund_payment", "Refund one payment."),
    ]
    notes = [
        _note(
            7,
            "Card payments",
            "Amounts are negative.",
            function_ids=[1, 2],
            origin="Total my card payments.",
        ),
    ]
    bind = _Binder()
    text = _section(functions, notes, "Total my card payments for May.", bind=bind)
    assert text.startswith(ni.HEADER)
    assert ni.CALL_FORM.strip() in text
    lines = text.splitlines()
    at = lines.index("### Note 7: Card payments")
    assert lines[at + 1 : at + 6] == [
        "Amounts are negative.",
        'Written for: "Total my card payments."',
        "Functions it links:",
        "- `total_payments(x)` (loaded): Total card payments.",
        "- `refund_payment(x)` (loaded): Refund one payment.",
    ]
    assert bind.names == [["total_payments", "refund_payment"]]
    assert set(bind.namespace) == {"total_payments", "refund_payment"}
    assert bind.calls == []


def test_without_a_binder_nothing_says_loaded(on):
    functions = [_fn(1, "total_payments", "Total card payments.")]
    notes = [_note(7, "Card payments", "x", function_ids=[1], origin="Total payments.")]
    text = _section(functions, notes, "Total payments.")
    assert "- `total_payments(x)`: Total card payments." in text
    assert "(loaded" not in text and ni.CALL_FORM.strip() not in text


def test_a_function_two_notes_link_is_described_once(on):
    functions = [_fn(1, "total_payments", "Total card payments.")]
    notes = [
        _note(1, "A", "x", function_ids=[1], origin="Total my payment."),
        _note(2, "B", "y", function_ids=[1], origin="Total my payment and refund."),
    ]
    bind = _Binder()
    text = _section(functions, notes, "Total my payment.", bind=bind)
    assert "- `total_payments(x)` (loaded): Total card payments." in text
    assert "- `total_payments(x)` (loaded): shown above" in text
    assert bind.names == [["total_payments"]]


# ── stand-in notes ───────────────────────────────────────────────────────


def test_an_unlinked_function_stands_in_by_its_own_request_and_nothing_is_written(on):
    functions = [
        _fn(
            1,
            "refund_invoice",
            "Refund an invoice in full.\n\nMore detail.",
            origin="Refund invoice 12.",
        ),
        # Linked to a note: no stand-in of its own.
        _fn(2, "plan_trip", "Plan a trip.", origin="Refund invoice 12."),
        # No request recorded: not ranked.
        _fn(3, "weather", "Check the weather."),
    ]
    notes = [_note(9, "Trips", "Plan trips.", function_ids=[2], origin="Plan my trip.")]
    bind = _Binder()
    # _Manager fails on any attribute but _evidence_rows: no write is made.
    text = _section(functions, notes, "Refund invoice 40.", bind=bind)
    heads = _headings(text)
    assert heads == [
        f"### {ni.STAND_IN}: Refund an invoice in full.",
        "### Note 9: Trips",
    ]
    lines = text.splitlines()
    at = lines.index(heads[0])
    assert lines[at + 1 : at + 3] == [
        'Written for: "Refund invoice 12."',
        "- `refund_invoice(x)` (loaded)",
    ]
    assert bind.names == [["refund_invoice", "plan_trip"]]
    assert bind.calls == []
    index = ni.build_index(functions, notes)
    stand_in = [n for n in index if n.stand_in]
    assert [(n.functions, n.origins) for n in stand_in] == [
        (["refund_invoice"], ["Refund invoice 12."]),
    ]


# ── failures ─────────────────────────────────────────────────────────────


def test_an_embedding_failure_leaves_no_section_and_binds_nothing(on, warnings):
    def broken(_texts):
        raise RuntimeError("embedding service down")

    functions = [_fn(1, "total_payments", "Total card payments.")]
    notes = [_note(1, "Payments", "x", function_ids=[1], origin="Total payments.")]
    bind = _Binder()
    text = _section(functions, notes, "Total payments.", embed=broken, bind=bind)
    assert text == ""
    assert bind.names == []
    assert any("embedding failed" in w for w in warnings)


def test_an_empty_index_embeds_nothing_and_warns(on, warnings):
    def boom(_texts):
        raise AssertionError("embedded an empty index")

    notes = [_note(1, "Payments", "x")]  # no request recorded
    assert _section([], notes, "Total payments.", embed=boom) == ""
    assert any("note index left out" in w for w in warnings)


def test_any_other_failure_never_reaches_the_task(on, warnings):
    class Broken:
        def _evidence_rows(self):
            raise OSError("store unreadable")

    bind = _Binder()
    text = ni.section(Broken(), Broken(), "Total payments.", embed=_embed, bind=bind)
    assert text == "" and bind.names == []
    assert any("store unreadable" in w for w in warnings)


def test_a_failing_binder_leaves_the_section_without_loaded_marks(on):
    def broken(_names):
        raise RuntimeError("no sandbox")

    functions = [_fn(1, "total_payments", "Total card payments.")]
    notes = [_note(1, "Payments", "x", function_ids=[1], origin="Total payments.")]
    text = _section(functions, notes, "Total payments.", bind=broken)
    assert "- `total_payments(x)`: Total card payments." in text
    assert "(loaded" not in text


# ── evidence already kept ────────────────────────────────────────────────


def test_only_recorded_cases_are_shown_and_no_lexical_evidence_is_used(
    on,
    monkeypatch,
):
    """The lead's rule: nothing shown is chosen or described by shared words."""
    from unify.function_manager import entry_record, store_cases

    monkeypatch.setattr(SETTINGS, "UNIFY_TASK_ORIGIN", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_ENTRY_RECORD", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_LISTING_PROVENANCE", True)
    monkeypatch.setattr(SETTINGS, "UNIFY_LISTING_USAGE", True)

    def lexical(*_a, **_k):
        raise AssertionError("a lexical evidence helper was called")

    for owner, name in (
        (task_origin, "listing_notes"),
        (task_origin, "shared_identifiers"),
        (task_origin, "similarity"),
        (task_origin, "token_weights"),
        (entry_record, "record_text"),
        (entry_record, "recurrence_phrase"),
        (task_origin.Marker, "origin_line"),
    ):
        monkeypatch.setattr(owner, name, lexical)
    monkeypatch.setattr(store_cases, "enabled", lambda: True)

    def summaries(rows):
        for row in rows:
            row["cases"] = f"{row['name']}(1) -> 2"
        return rows

    monkeypatch.setattr(store_cases, "with_summaries", summaries)
    functions = [_fn(1, "total_payments", "Total card payments.")]
    notes = [_note(4, "Payments", "x", function_ids=[1], origin="Total payments.")]
    lines = _section(functions, notes, "Total payments.").splitlines()
    at = lines.index("- `total_payments(x)`: Total card payments.")
    assert lines[at + 1 :] == ["  cases: total_payments(1) -> 2"]


def test_the_written_for_line_is_the_latest_request_clipped(on):
    long = "Refund invoice " + "x" * 400
    note = _note(1, "Refunds", "x")
    note["metadata"] = {task_origin.REQUESTS_FIELD: ["Plan my trip.", long]}
    lines = _section([], [note], "Plan my trip.").splitlines()
    (written,) = [ln for ln in lines if ln.startswith("Written for: ")]
    # The latest request, not the closest one; at most 200 characters.
    assert written.startswith('Written for: "Refund invoice xxx')
    assert len(written) == len('Written for: ""') + ni.WRITTEN_FOR_CHARS


# ── the trust label (ADR-16 (e)) ─────────────────────────────────────────


def test_a_note_whose_last_writer_was_not_accepted_is_labelled(on, outcomes):
    notes = [
        _note(1, "Failed", "x", origin="Total my payment."),
        _note(2, "Accepted", "y", origin="Total my payment and refund."),
        _note(3, "Unknown", "z", origin="Total my payment and refund my invoice."),
    ]
    outcomes["Total my payment."] = False
    outcomes["Total my payment and refund."] = True
    heads = _headings(_section([], notes, "Total my payment."))
    assert heads == [
        f"### Note 1: Failed {ni.FAILED_WRITER}",
        "### Note 2: Accepted",
        "### Note 3: Unknown",
    ]


def test_only_the_latest_writer_counts(on, outcomes):
    note = _note(1, "Payments", "x")
    note["metadata"] = {
        task_origin.REQUESTS_FIELD: ["Total my payment.", "Total my payment again."],
    }
    outcomes["Total my payment."] = False
    outcomes["Total my payment again."] = True
    assert ni.FAILED_WRITER not in _section([], [note], "Total my payment.")
    outcomes["Total my payment."] = True
    outcomes["Total my payment again."] = False
    assert ni.FAILED_WRITER in _section([], [note], "Total my payment.")


def test_the_session_that_wrote_the_current_content_is_the_writer(on, outcomes):
    from unify.function_manager import verified_guard

    note = _note(1, "Payments", "x", origin="Total my payment again.")
    outcomes["Total my payment again."] = True
    outcomes["Total my payment."] = False
    note["metadata"][verified_guard.FIELD] = task_origin.text_key("Total my payment.")
    assert ni.FAILED_WRITER in _section([], [note], "Total my payment.")


def test_a_stand_in_whose_function_writer_failed_is_labelled(on, outcomes):
    functions = [
        _fn(1, "refund_invoice", "Refund an invoice.", origin="Refund invoice 12."),
    ]
    outcomes["Refund invoice 12."] = False
    heads = _headings(_section(functions, [], "Refund invoice 40."))
    assert heads == [f"### {ni.STAND_IN}: Refund an invoice. {ni.FAILED_WRITER}"]


def _log_outcome(path, text, source, solved):
    from contextlib import closing

    with closing(task_origin._connect_outcomes(path)) as conn, conn:
        conn.execute(
            "INSERT INTO request_outcomes (text_key, source, solved) VALUES (?, ?, ?)",
            (task_origin.text_key(text), source, int(solved)),
        )


def test_only_the_checkers_verdict_labels_never_the_reviews(
    on,
    monkeypatch,
    tmp_path,
):
    """ADR-2: the storage review's judgement is not a verdict."""
    log = tmp_path / "request_log.sqlite"
    monkeypatch.setattr(task_origin, "request_log_path", lambda: log)
    monkeypatch.setattr(ni, "checker_outcome", _CHECKER_OUTCOME)
    notes = [
        _note(1, "Review only", "x", origin="Total my payment."),
        _note(2, "Checker failed", "y", origin="Total my payment and refund."),
        _note(3, "Checker accepted", "z", origin="Total my payment, refund, invoice."),
    ]
    _log_outcome(log, "Total my payment.", task_origin.REVIEW, False)
    _log_outcome(log, "Total my payment and refund.", task_origin.CHECKER, False)
    # The review said accepted; the checker's "not accepted" still labels.
    _log_outcome(log, "Total my payment and refund.", task_origin.REVIEW, True)
    _log_outcome(log, "Total my payment, refund, invoice.", task_origin.CHECKER, True)
    _log_outcome(log, "Total my payment, refund, invoice.", task_origin.REVIEW, False)
    heads = _headings(_section([], notes, "Total my payment."))
    assert heads == [
        "### Note 1: Review only",
        f"### Note 2: Checker failed {ni.FAILED_WRITER}",
        "### Note 3: Checker accepted",
    ]


def test_the_label_never_hides_or_reranks(on, outcomes):
    notes = [
        _note(i + 1, f"Note {i + 1}", "x", origin=r)
        for i, r in enumerate(
            [
                "Total my payment and refund my invoice.",
                "Total my payment and refund.",
                "Total my payment.",
                "Plan a trip.",
            ],
        )
    ]
    request = "Total my payment and refund my invoice."
    before = [h.split(" (")[0] for h in _headings(_section([], notes, request))]
    for note in notes:
        outcomes[note["metadata"][task_origin.REQUESTS_FIELD][0]] = False
    after = _headings(_section([], notes, request))
    assert all(h.endswith(ni.FAILED_WRITER) for h in after)
    assert [h[: -len(ni.FAILED_WRITER) - 1] for h in after] == before
    assert len(after) == ni.K_NOTES


# ── prerequisites and wording ────────────────────────────────────────────


@pytest.mark.parametrize(
    "switches",
    [
        {"UNIFY_TASK_ORIGIN": False, "UNIFY_GUIDANCE_ORIGIN": True},
        {"UNIFY_TASK_ORIGIN": True, "UNIFY_GUIDANCE_ORIGIN": False},
    ],
)
def test_the_index_refuses_to_start_without_the_requests_it_ranks_by(
    monkeypatch,
    switches,
):
    monkeypatch.setattr(SETTINGS, "UNIFY_NOTE_INDEX", True)
    for name in (
        "UNIFY_TRY_FIRST",
        "UNIFY_ENTRY_RECORD",
        "UNIFY_LISTING_PROVENANCE",
        "UNIFY_LESSON_STATUS",
    ):
        monkeypatch.setattr(SETTINGS, name, False)
    monkeypatch.setattr(SETTINGS, "UNIFY_PROTECT_VERIFIED", "")
    for name, value in switches.items():
        monkeypatch.setattr(SETTINGS, name, value)
    with pytest.raises(ValueError, match="UNIFY_NOTE_INDEX needs"):
        ni.require_prerequisites()


def test_the_texts_force_nothing_and_ask_for_no_example_check():
    text = (ni.HEADER + ni.INTRO + ni.CALL_FORM + ni.STAND_IN).lower()
    for word in ("must", "always", "example", "verify", "test"):
        assert word not in text
