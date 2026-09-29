"""Symbolic: ``UNIFY_STORE_VERIFY`` stores a function only after it passed on a held-out task.

A scripted verifier stands in for an environment: it describes a held-out task
(the session's request, sibling requests that differ in one detail, the
held-out request, the credential parameters and the session's credentials) and
runs a candidate against fake ``primitives`` that record every call. The cases
are the ones a real run produced: a function that hard-codes one request's
phone number, three versions of a file-renaming planner that each froze one
request's date format, a function that kept a login token or password, and one
that asks the caller for an access token. With the switch off, add_functions
stores as shipped and the review has no check tool. No model is called.
"""

from __future__ import annotations

import ast
import sys
import types
from types import SimpleNamespace

import pytest

from tests.helpers import _handle_project
from unify.function_manager import store_verify
from unify.function_manager.function_manager import FunctionManager
from unify.function_manager.primitives import (
    EnvironmentMethod,
    EnvironmentNamespace,
    EnvironmentSurface,
    register_environment,
)
from unify.function_manager.primitives.environment import clear_environment_namespaces
from unify.settings import SETTINGS

SESSION_REQUEST = (
    "All text messages and voice messages from 3654328626 are spam, delete them."
)
SIBLINGS = (
    "All text messages and voice messages from 9294880327 are spam, delete them.",
    "All text messages and voice messages from 5708520672 are spam, delete them.",  # pragma: allowlist secret
)  # pragma: allowlist secret
TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJwaG9uZSs1NTUifQ.c2lnbmF0dXJlLXZhbHVlLWhlcmU"  # pragma: allowlist secret
PASSWORD = "Qz7!pw9"  # pragma: allowlist secret

HARD_CODED = (
    "def delete_spam() -> int:\n"
    '    """Delete every text message from the spam number."""\n'
    "    token = primitives.phone.login(username='me', password=None)['access_token']\n"
    "    found = primitives.phone.search_text_messages(access_token=token, phone_number='3654328626')\n"
    "    for message in found:\n"
    "        primitives.phone.delete_text_message(access_token=token, text_message_id=message['id'])\n"
    "    return len(found)\n"
)
PARAMETERISED = (
    "def delete_spam_from(phone_number: str) -> int:\n"
    '    """Delete every text message from ``phone_number`` (as in 3654328626 last time)."""\n'
    "    print('deleting messages from', phone_number, 'e.g. 3654328626')\n"
    "    token = primitives.phone.login(username='me', password=None)['access_token']\n"
    "    found = primitives.phone.search_text_messages(access_token=token, phone_number=phone_number)\n"
    "    for message in found:\n"
    "        primitives.phone.delete_text_message(access_token=token, text_message_id=message['id'])\n"
    "    return len(found)\n"
)

# The three stored versions of the planner in the 28 September canary (docs/appworld-canary-20260928.md
# section 3), reduced to the line that names the file; each sibling asks for another prefix format.
FORMATS = ("YYYY-MM-DD_", "YYYY_MM_DD-", "YYYY_MM_DD_")


def _request(fmt: str) -> str:
    return (
        f'Rename every file in my downloads folder by adding the prefix "{fmt}" based on its creation date, '
        f"then move the files not from this year to the trash."
    )


PLANNERS = {
    "v1": (
        0,
        "def plan_prefix_moves(files: list) -> list:\n"
        '    """Plan the renames."""\n'
        "    plan = []\n"
        "    for created, name in files:\n"
        '        plan.append(f"{created:%Y-%m-%d}_{name}")\n'
        "    return plan\n",
        "YYYY-MM-DD_",
    ),
    "v2": (
        1,
        "def plan_prefix_moves(files: list) -> list:\n"
        '    """Plan the renames."""\n'
        '    return [f"{created:%Y_%m_%d}-{name}" for created, name in files]\n',
        "YYYY_MM_DD-",
    ),
    "v3": (
        2,
        "def plan_prefix_moves(files: list, date_separator: str = '-') -> list:\n"
        '    """Plan the renames; the separator after the date is an input."""\n'
        '    if date_separator not in ("-", "_"):\n'
        '        raise ValueError(f"unsupported separator {date_separator!r}")\n'
        '    return [f"{created:%Y_%m_%d}{date_separator}{name}" for created, name in files]\n',
        "YYYY_MM_DD",
    ),
}


class ScriptedVerifier:
    def __init__(self):
        self.held = {
            "available": True,
            "source_texts": [SESSION_REQUEST],
            "sibling_texts": list(SIBLINGS),
            "held_out_text": SIBLINGS[0],
            "credential_params": ["access_token", "password"],
            "secrets": [TOKEN, PASSWORD],
        }
        self.runs: list[dict] = []
        self.calls: list[tuple] = []
        self.outcome = True

    def held_out(self, name):
        return dict(self.held)

    def run(self, candidate, call_kwargs):
        verifier = self

        class Phone:
            def __getattr__(self, method):
                def call(**kwargs):
                    verifier.calls.append((method, kwargs))
                    if method == "login":
                        return {"access_token": "fresh"}
                    if method == "search_text_messages":
                        return (
                            [{"id": 1}, {"id": 2}]
                            if kwargs.get("phone_number") == "9294880327"
                            else []
                        )
                    return {"ok": True}

                return call

        fn = candidate.load(SimpleNamespace(phone=Phone()), {"marker": "held-out"})
        self.runs.append(
            {
                "name": candidate.name,
                "kwargs": dict(call_kwargs),
                "effects": candidate.effects,
                "depends_on": candidate.depends_on,
                "sha256": candidate.sha256,
            },
        )
        try:
            value = fn(**call_kwargs)
        except Exception as exc:
            return {"ok": False, "reason": f"{type(exc).__name__}"}
        ok = self.outcome and value == 2
        return {
            "ok": ok,
            "reason": "deleted" if ok else "wrong outcome",
            "details": {"api_calls": len(self.calls)},
        }


@pytest.fixture
def phone_env():
    clear_environment_namespaces()
    methods = tuple(
        EnvironmentMethod(
            name=n,
            call=lambda **kw: None,
            effect=e,
            signature="(**kwargs)",
        )
        for n, e in (
            ("login", "write"),
            ("search_text_messages", "read"),
            ("delete_text_message", "destructive"),
        )
    )
    register_environment(
        EnvironmentSurface(
            namespaces=(EnvironmentNamespace(name="phone", methods=methods),),
        ),
        source="tests:phone",
    )
    from unify.function_manager import function_manager as fm_module

    fm_module._PRIMITIVES_SEEDED_FOR.clear()
    yield
    clear_environment_namespaces()
    fm_module._PRIMITIVES_SEEDED_FOR.clear()


@pytest.fixture
def verify_on(monkeypatch, tmp_path, phone_env):
    scripted = ScriptedVerifier()
    module = types.ModuleType("scripted_store_verifier")
    module.make = lambda: scripted
    monkeypatch.setitem(sys.modules, "scripted_store_verifier", module)
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_STORE_ADMISSION",
        str(tmp_path / "verdict.json"),
        raising=False,
    )
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_STORE_VERIFY",
        "scripted_store_verifier:make",
        raising=False,
    )
    store_verify.reset()
    yield scripted
    store_verify.reset()


def _node(source: str):
    return ast.parse(source).body[0]


def _held(**overrides) -> store_verify.HeldOut:
    base = dict(available=True, source_texts=(SESSION_REQUEST,), sibling_texts=SIBLINGS)
    base.update(overrides)
    return store_verify.HeldOut(**base)


# -- static check (a): request details ---------------------------------------------------------------------------


def test_a_hard_coded_request_detail_is_named():
    found = store_verify.varying_literals(
        _node(HARD_CODED),
        [SESSION_REQUEST],
        SIBLINGS,
    )
    assert [lit.text for lit in found] == ["3654328626"]
    problems = store_verify.static_problems(_node(HARD_CODED), _held())
    assert (
        len(problems) == 1
        and "'3654328626'" in problems[0]
        and "make it a parameter" in problems[0]
    )


def test_a_detail_copied_from_the_held_out_request_is_refused_too():
    copied = HARD_CODED.replace("3654328626", "9294880327")
    found = store_verify.varying_literals(_node(copied), [SESSION_REQUEST], SIBLINGS)
    assert [lit.text for lit in found] == ["9294880327"]


def test_the_parameterised_version_passes_and_messages_are_skipped():
    # the docstring and the print call name the old number; neither is code that uses it
    assert store_verify.static_problems(_node(PARAMETERISED), _held()) == []


@pytest.mark.parametrize("version", sorted(PLANNERS))
def test_every_planner_version_of_the_canary_is_refused(version):
    index, source, shown = PLANNERS[version]
    request = _request(FORMATS[index])
    siblings = [_request(f) for i, f in enumerate(FORMATS) if i != index]
    found = store_verify.varying_literals(_node(source), [request], siblings)
    assert [lit.text for lit in found] == [shown]
    assert found[0].kind == "pattern"


def test_date_patterns_are_rebuilt_from_format_strings():
    assert store_verify.strftime_pattern("%Y-%m-%d") == "YYYY-MM-DD"
    source = (
        "def name_it(created, name):\n"
        "    a = created.strftime('%Y_%m_%d') + '-' + name\n"
        "    b = '{:%Y-%m-%d}_{}'.format(created, name)\n"
        "    return a, b\n"
    )
    texts = {lit.text for lit in store_verify.body_literals(_node(source))}
    assert {"YYYY_MM_DD", "YYYY-MM-DD_"} <= texts


def test_a_value_the_request_shares_with_every_sibling_is_kept():
    source = (
        "def delete_spam_from(phone_number: str, pages: int = 10) -> str:\n"
        "    if phone_number == '3654328626':\n"
        "        return 'the old number'\n"
        "    return primitives.phone.search_text_messages(phone_number=phone_number, kind='voice messages')\n"
    )
    # compared with a parameter: a check of the input, not a baked-in value; 'voice messages' is in every request
    assert (
        store_verify.varying_literals(_node(source), [SESSION_REQUEST], SIBLINGS) == []
    )


def test_request_tokens_keep_values_whole():  # pragma: allowlist secret
    assert store_verify.request_tokens('prefix "YYYY-MM-DD_" in ~/downloads/.') == [
        "prefix",
        "yyyy-mm-dd_",
        "in",
        "~/downloads/",
    ]
    spans = store_verify.differing_spans("Send $91 privately", "Send $100 publicly")
    assert spans == [(["$91", "privately"], ["$100", "publicly"])]
    amount = "def pay():\n    return primitives.venmo.pay(amount=91, private=True)\n"
    assert [
        lit.text
        for lit in store_verify.varying_literals(
            _node(amount),
            ["Send $91 privately"],
            ["Send $100 publicly"],
        )
    ] == ["91"]


# -- static check (b): credentials -------------------------------------------------------------------------------


def test_a_credential_literal_is_refused_without_being_shown():
    source = (
        "def delete_spam_from(phone_number: str) -> int:\n"
        f"    primitives.phone.login(username='me', password={PASSWORD!r})\n"
        f"    return primitives.phone.search_text_messages(access_token={TOKEN!r}, phone_number=phone_number)\n"
    )
    problems = store_verify.credential_problems(
        _node(source),  # pragma: allowlist secret
        secrets=[TOKEN, PASSWORD],
    )
    assert len(problems) == 2
    assert all("credentials" in p for p in problems)
    assert not any(PASSWORD in p or TOKEN in p for p in problems)


def test_token_shaped_literals_and_globals_are_refused():
    source = (
        "def cached(phone_number: str) -> int:\n"
        "    global TOKEN_CACHE\n"
        "    TOKEN_CACHE = 'a9F3kQ7zL2mX8vB4nR6tY1wE5uI0oP'\n"  # pragma: allowlist secret
        "    return 1\n"
    )
    problems = store_verify.credential_problems(_node(source))
    assert any("looks like a token" in p for p in problems)
    assert any("global TOKEN_CACHE" in p for p in problems)
    assert not store_verify.looks_like_token("text_message_id_and_voice_message_id")
    assert store_verify.looks_like_token(TOKEN)


def test_a_credential_parameter_must_default_to_none():
    required = (
        "def delete_with(access_token: str, phone_number: str) -> int:\n    return 1\n"
    )
    problems = store_verify.credential_problems(
        _node(required),
        credential_params=["access_token", "password"],
    )
    assert (
        len(problems) == 1
        and "`access_token`" in problems[0]
        and "default to None" in problems[0]
    )
    optional = "def delete_with(phone_number: str, *, access_token=None) -> int:\n    return 1\n"
    assert (
        store_verify.credential_problems(
            _node(optional),
            credential_params=["access_token"],
        )
        == []
    )


# -- the tool and the gate ---------------------------------------------------------------------------------------


@_handle_project
def test_hard_coded_function_is_refused_by_the_gate_naming_the_literal(verify_on):
    fm = FunctionManager()
    result = fm.add_functions(implementations=[HARD_CODED], raise_on_error=False)
    assert result["delete_spam"].startswith(
        "error: 'delete_spam' was not stored, because it has not passed",
    )
    assert "'3654328626'" in result["delete_spam"]
    assert "delete_spam" not in fm.list_functions()
    checked = fm.check_function(implementation=HARD_CODED)
    assert (
        checked["static_problems"] and "'3654328626'" in checked["static_problems"][0]
    )
    ran = fm.check_function(implementation=HARD_CODED, call_kwargs={})
    assert "passed" not in ran and verify_on.runs == []  # refused before any run


@_handle_project
def test_checked_function_is_stored_and_ran_against_the_verifier_world(verify_on):
    fm = FunctionManager()
    first = fm.check_function(implementation=PARAMETERISED)
    assert first["available"] is True and first["static_problems"] == []
    assert first["held_out_request"] == SIBLINGS[0]
    assert "passed" not in first and verify_on.runs == []
    refused = fm.add_functions(implementations=[PARAMETERISED], raise_on_error=False)
    assert "no passing run yet" in refused["delete_spam_from"]
    ran = fm.check_function(
        implementation=PARAMETERISED,
        call_kwargs={"phone_number": "9294880327"},
    )
    assert ran["passed"] is True, ran
    assert verify_on.runs[0]["kwargs"] == {"phone_number": "9294880327"}
    assert verify_on.runs[0]["effects"] == ("destructive", "read", "write")
    assert [c[0] for c in verify_on.calls] == [
        "login",
        "search_text_messages",
        "delete_text_message",
        "delete_text_message",
    ]
    assert fm.add_functions(implementations=[PARAMETERISED]) == {
        "delete_spam_from": "added",
    }
    assert "delete_spam_from" in fm.list_functions()
    # a pass is for that exact source: an edit needs its own check
    edited = PARAMETERISED.replace("return len(found)", "return len(found) + 0")
    result = fm.add_functions(
        implementations=[edited],
        overwrite=True,
        raise_on_error=False,
    )
    assert result["delete_spam_from"].startswith("error:")


@_handle_project
def test_a_failed_run_records_no_pass(verify_on):
    fm = FunctionManager()
    verify_on.outcome = False
    ran = fm.check_function(
        implementation=PARAMETERISED,
        call_kwargs={"phone_number": "9294880327"},
    )
    assert ran["passed"] is False and ran["reason"] == "wrong outcome"
    assert fm.add_functions(implementations=[PARAMETERISED], raise_on_error=False)[
        "delete_spam_from"
    ].startswith(
        "error:",
    )


@_handle_project
def test_run_checks_are_capped_per_review(verify_on):
    fm = FunctionManager()
    verify_on.outcome = False
    for i in range(store_verify.MAX_RUN_CHECKS):
        ran = fm.check_function(
            implementation=PARAMETERISED,
            call_kwargs={"phone_number": "9294880327"},
        )
        assert ran["run_checks_left"] == store_verify.MAX_RUN_CHECKS - i - 1
    over = fm.check_function(
        implementation=PARAMETERISED,
        call_kwargs={"phone_number": "9294880327"},
    )
    assert (
        "used up" in over["result"]
        and len(verify_on.runs) == store_verify.MAX_RUN_CHECKS
    )


@_handle_project
def test_credentials_in_call_kwargs_and_required_credential_parameters_are_refused(
    verify_on,
):
    fm = FunctionManager()
    ran = fm.check_function(
        implementation=PARAMETERISED,
        call_kwargs={"phone_number": "9294880327", "access_token": "x"},
    )
    assert "may not carry credentials" in ran["result"] and verify_on.runs == []
    required = (
        "def delete_with(access_token: str, phone_number: str) -> int:\n"
        "    return len(primitives.phone.search_text_messages(access_token=access_token, "
        "phone_number=phone_number))\n"
    )
    checked = fm.check_function(implementation=required)
    assert any("`access_token`" in p for p in checked["static_problems"])
    stolen = PARAMETERISED.replace("password=None", f"password={PASSWORD!r}")
    result = fm.add_functions(implementations=[stolen], raise_on_error=False)
    assert (
        "credentials" in result["delete_spam_from"]
        and PASSWORD not in result["delete_spam_from"]
    )


@_handle_project
def test_no_held_out_task_means_nothing_new_is_stored(verify_on):
    fm = FunctionManager()
    verify_on.held = {
        "available": False,
        "reason": "the last task was not an admitted training task",
    }
    checked = fm.check_function(
        implementation=PARAMETERISED,
        call_kwargs={"phone_number": "1"},
    )
    assert checked["available"] is False and verify_on.runs == []
    result = fm.add_functions(implementations=[PARAMETERISED], raise_on_error=False)
    assert (
        "no held-out task is available (the last task was not an admitted training task)"
        in result["delete_spam_from"]
    )


def test_the_switch_needs_admission_and_a_loadable_factory(monkeypatch):
    store_verify.reset()
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_STORE_VERIFY",
        "no_such_module_here:make",
        raising=False,
    )
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_ADMISSION", "", raising=False)
    with pytest.raises(
        store_verify.StoreVerifyError,
        match="needs UNIFY_STORE_ADMISSION",
    ):
        store_verify.verifier()
    monkeypatch.setattr(
        SETTINGS,
        "UNIFY_STORE_ADMISSION",
        "/tmp/verdict.json",
        raising=False,
    )
    with pytest.raises(
        store_verify.StoreVerifyError,
        match="failed: ModuleNotFoundError",
    ):
        store_verify.verifier()
    store_verify.reset()


def _storage_tool_names() -> set:
    import unify.actor.code_act_actor as caa
    from unify.guidance_manager.guidance_manager import GuidanceManager

    actor = SimpleNamespace(
        function_manager=FunctionManager(),
        guidance_manager=GuidanceManager(),
    )
    tools, _, _ = caa._build_storage_tools(actor=actor, ask_tools={})
    return set(tools)


@_handle_project
def test_the_review_gets_the_tool_and_one_sentence_only_while_set(
    verify_on,
    monkeypatch,
):
    import unify.actor.code_act_actor as caa

    assert "FunctionManager_check_function" in _storage_tool_names()
    assert store_verify.doctrine() in caa._storage_environment_note()
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "", raising=False)
    assert "FunctionManager_check_function" not in _storage_tool_names()
    assert store_verify.doctrine() not in caa._storage_environment_note()


@_handle_project
def test_off_stores_the_hard_coded_function_as_shipped(monkeypatch):
    monkeypatch.setattr(SETTINGS, "UNIFY_STORE_VERIFY", "", raising=False)
    store_verify.reset()
    fm = FunctionManager()
    assert fm.add_functions(implementations=[HARD_CODED]) == {"delete_spam": "added"}
    assert "error" in fm.check_function(implementation=HARD_CODED)
    assert not store_verify.enabled()
