"""The one identifier rule: what is stripped from signatures and compared content, and what is kept."""

from __future__ import annotations

import pytest

from unify.memory import signatures as sg

KEPT = ["sha256", "utf8", "float64", "base64", "int32", "x2", "step3", "float128"]
STRIPPED = [
    "123e4567-e89b-12d3-a456-426614174000",
    "0f3a9c7e1b",
    "ab12cd34ef",
    "run_000123",
    "case-2041x",
]


@pytest.mark.parametrize("token", KEPT)
def test_ordinary_words_with_numbers_are_kept(token):
    assert not sg.is_identifier(token)
    assert sg.strip_identifiers(f"use {token} here") == f"use {token} here"


@pytest.mark.parametrize("token", STRIPPED)
def test_identifier_forms_are_stripped(token):
    assert sg.is_identifier(token)
    assert sg.strip_identifiers(f"see {token}, then go") == f"see {sg.ID_MARK}, then go"


def test_values_the_request_labels_as_ids_are_stripped_whatever_their_form():
    request = 'task: lunar-otter\nrequest_id = "q-17x"\nFollow #abc-room for details.\n'
    assert sg.printed_ids(request) == {"lunar-otter", "q-17x", "abc-room"}
    text = "Solve lunar-otter (q-17x) as abc-room did; lunar-otters differ"
    assert sg.strip_identifiers(text, request=request) == (
        f"Solve {sg.ID_MARK} ({sg.ID_MARK}) as {sg.ID_MARK} did; lunar-otters differ"
    )
    # without the request, ordinary-looking aliases stay: only their labels make them ids
    assert "lunar-otter" in sg.strip_identifiers(text)


def test_short_labelled_values_and_unlabelled_fields_are_not_taken():
    assert sg.printed_ids("id: 7\ncolour: lunar-otter\n# Heading\nC# is a language") == set()
