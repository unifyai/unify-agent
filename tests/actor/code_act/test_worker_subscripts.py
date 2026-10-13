"""Symbolic: a cell can subscript a harness object (``apis["music"]``) across the worker boundary.

An environment global such as AppWorld's ``apis`` is a remote object in the
sandboxed worker; its cells index it by app name. ``proxy[key]`` now crosses
the boundary for a ``str`` or ``int`` key, as ``proxy.name`` does: the worker
adds ``{"item": key}`` to the target's path and the harness applies
``obj[key]`` at the same trust as ``getattr``. Other keys (slices, tuples,
objects, ``bool``) and keys starting with ``_`` are refused on both sides, a
module or the environment reached by a subscript is refused as by an
attribute, and iterating a proxy stays a ``TypeError``. The unit tests use a
recorded channel; the end-to-end test is in
``test_worker_environment_globals.py``. No model is called.
"""

from __future__ import annotations

import json
import os

import pytest

from unify.actor.execution import worker_child
from unify.actor.execution.worker import PythonWorker
from unify.actor.execution.worker_child import BoundaryRefusal


class _Channel:
    """Stands in for the worker's channel: records each describe request."""

    def __init__(self, desc: dict) -> None:
        self.desc = desc
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, op: str, **fields):
        self.requests.append((op, fields))
        return self.desc, None


def _proxy(desc: dict) -> tuple[worker_child.RemoteNamespace, _Channel]:
    w = object.__new__(worker_child.Worker)
    w._describe_cache = {}
    channel = _Channel(desc)
    w.request_sync = channel
    root = {"root": "apis", "path": []}
    return worker_child.RemoteNamespace(w, root, "apis", "<apis>"), channel


# ── the worker side ─────────────────────────────────────────────────────────


def test_a_subscript_adds_an_item_element_to_the_target():
    apis, channel = _proxy({"kind": "namespace", "repr": "<music app>"})
    music = apis["music"]
    assert isinstance(music, worker_child.RemoteNamespace)
    assert repr(music) == "<music app>"
    assert object.__getattribute__(music, "_label") == "apis['music']"
    assert channel.requests == [
        ("describe", {"target": {"root": "apis", "path": [{"item": "music"}]}}),
    ]
    # Attributes and further subscripts extend the same path.
    channel.desc = {"kind": "callable", "async": False}
    show = music.show
    assert isinstance(show, worker_child.RemoteCallable)
    assert object.__getattribute__(show, "_target")["path"] == [
        {"item": "music"},
        "show",
    ]
    channel.desc = {"kind": "value", "value": 7}
    assert apis[3] == 7
    assert channel.requests[-1][1]["target"]["path"] == [{"item": 3}]
    # The description is cached per target, as for attributes.
    count = len(channel.requests)
    apis["music"]
    assert len(channel.requests) == count


@pytest.mark.parametrize("key", [slice(0, 2), (1, 2), True, 1.5, b"music", None])
def test_only_str_and_int_subscripts_cross(key):
    apis, channel = _proxy({"kind": "namespace", "repr": "<x>"})
    with pytest.raises(BoundaryRefusal, match="only str and int subscripts"):
        apis[key]
    assert channel.requests == []


def test_a_key_starting_with_underscore_stays_in_the_harness():
    apis, channel = _proxy({"kind": "namespace", "repr": "<x>"})
    with pytest.raises(KeyError, match="do not cross the worker boundary"):
        apis["_client"]
    assert channel.requests == []


def test_iterating_a_proxy_is_still_a_type_error():
    apis, channel = _proxy({"kind": "value", "value": 1})
    with pytest.raises(TypeError):
        iter(apis)
    with pytest.raises(TypeError):
        "music" in apis  # noqa: B015
    assert channel.requests == []


# ── the harness side ────────────────────────────────────────────────────────


class _App:
    def show(self, item: str) -> str:
        return f"shown {item}"


PLANTED = "planted-value-for-a-test"  # pragma: allowlist secret


class _Apps:
    """A mapping and a sequence at once; each key is also an attribute, so a
    subscript and an attribute that reach the same object can be compared."""

    boundary_subscript = True

    def __init__(self) -> None:
        self.apps = {
            "music": _App(),
            "env": os.environ,
            "os": os,
            "api_key": PLANTED,
            "count": 3,
        }
        for name, value in self.apps.items():
            setattr(self, name, value)

    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.apps.values())[key]
        return self.apps[key]


def _harness() -> tuple[PythonWorker, dict]:
    worker = PythonWorker()
    worker._synced = {"apis": (("kind", json.dumps("remote")),)}
    return worker, {"apis": _Apps()}


def test_the_harness_applies_item_elements():
    worker, shadow = _harness()
    show = worker._resolve(
        shadow,
        {"root": "apis", "path": [{"item": "music"}, "show"]},
    )
    assert show("song") == "shown song"
    assert isinstance(
        worker._resolve(shadow, {"root": "apis", "path": [{"item": 0}]}),
        _App,
    )
    with pytest.raises(KeyError):
        worker._resolve(shadow, {"root": "apis", "path": [{"item": "absent"}]})


@pytest.mark.parametrize(
    "element",
    [
        {"item": "_apps"},
        {"item": True},
        {"item": 1.5},
        {"item": [0]},
        {"item": None},
        {"item": "music", "attr": "show"},
        {},
    ],
)
def test_the_harness_refuses_other_item_elements(element):
    worker, shadow = _harness()
    with pytest.raises(BoundaryRefusal):
        worker._resolve(shadow, {"root": "apis", "path": [element]})


def test_a_module_or_the_environment_behind_a_subscript_is_refused():
    worker, shadow = _harness()
    with pytest.raises(BoundaryRefusal, match="harness's environment"):
        worker._resolve(shadow, {"root": "apis", "path": [{"item": "env"}]})
    with pytest.raises(BoundaryRefusal, match="harness module"):
        worker._resolve(shadow, {"root": "apis", "path": [{"item": "os"}]})


# ── a subscript is checked exactly as an attribute is ───────────────────────


@pytest.mark.parametrize(
    "name, match",
    [("env", "harness's environment"), ("os", "harness module")],
)
def test_a_subscript_is_refused_wherever_the_attribute_is(name, match):
    worker, shadow = _harness()
    for element in (name, {"item": name}):
        with pytest.raises(BoundaryRefusal, match=match):
            worker._resolve(shadow, {"root": "apis", "path": [element]})


@pytest.mark.parametrize("element", ["music", {"item": "music"}, {"item": 0}])
def test_a_subscript_reaches_nothing_from_a_root_that_is_not_exposed(element):
    worker, shadow = _harness()
    shadow["hidden"] = _Apps()  # in the namespace, never synced as remote
    with pytest.raises(BoundaryRefusal, match="not a harness object exposed"):
        worker._resolve(shadow, {"root": "hidden", "path": [element]})
    with pytest.raises(BoundaryRefusal, match="unknown or belongs to a worker"):
        worker._resolve(shadow, {"ref": 12345, "path": [element]})


async def _describe(worker: PythonWorker, shadow: dict, element) -> dict:
    reply: dict = {}
    await worker._answer(
        "describe",
        {"target": {"root": "apis", "path": [element]}},
        shadow,
        reply,
    )
    return reply


@pytest.mark.asyncio
async def test_a_credential_named_key_stays_in_the_harness_as_the_attribute_does():
    worker, shadow = _harness()
    for element in ("api_key", {"item": "api_key"}):
        reply: dict = {}
        with pytest.raises(BoundaryRefusal, match="looks like a credential"):
            await worker._answer(
                "describe",
                {"target": {"root": "apis", "path": [element]}},
                shadow,
                reply,
            )
        assert PLANTED not in json.dumps(reply)
    # Other values are described alike either way.
    for name in ("count", "music"):
        by_attribute = await _describe(worker, shadow, name)
        by_subscript = await _describe(worker, shadow, {"item": name})
        assert by_attribute == by_subscript
    assert (await _describe(worker, shadow, {"item": "count"}))["value"] == {
        "kind": "value",
        "value": 3,
    }


# ── only a declared-safe object is subscripted ─────────────────────────────


class _Recorder:
    """An object whose ``__getitem__`` records each call; its class does not
    set ``boundary_subscript``."""

    def __init__(self) -> None:
        self.calls: list = []

    def __getitem__(self, key):
        self.calls.append(key)
        return _App()


class _Unmarked(_Recorder):
    boundary_subscript = "yes"  # only ``True`` itself opts in


@pytest.mark.parametrize("cls", [_Recorder, _Unmarked])
@pytest.mark.parametrize("key", ["music", 0])
def test_an_object_that_does_not_opt_in_is_refused_before_its_getitem_runs(
    cls,
    key,
):
    worker, shadow = _harness()
    recorder = cls()
    shadow["apis"] = recorder
    with pytest.raises(BoundaryRefusal, match="boundary_subscript = True"):
        worker._resolve(shadow, {"root": "apis", "path": [{"item": key}]})
    assert recorder.calls == []
    # Reached through an attribute of an object that does opt in, likewise.
    apps = _Apps()
    apps.recorder = recorder
    shadow["apis"] = apps
    with pytest.raises(BoundaryRefusal, match="boundary_subscript = True"):
        worker._resolve(
            shadow,
            {"root": "apis", "path": ["recorder", {"item": key}]},
        )
    assert recorder.calls == []


def test_a_plain_list_tuple_or_dict_is_subscripted():
    worker, shadow = _harness()
    apps = _Apps()
    apps.listed = [_App()]
    apps.paired = (1, _App())
    apps.mapped = {"music": _App()}
    shadow["apis"] = apps
    for path in (
        ["listed", {"item": 0}, "show"],
        ["paired", {"item": 1}, "show"],
        ["mapped", {"item": "music"}, "show"],
    ):
        assert worker._resolve(shadow, {"root": "apis", "path": path})("x") == (
            "shown x"
        )


# ── a container crosses by value only without a credential inside ───────────

KEY_SHAPED = "sk-or-v1-planted-readable-test-value"  # pragma: allowlist secret


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cfg, planted",
    [
        ({"OPENROUTER_API_KEY": "planted-low-entropy"}, "planted-low-entropy"),
        ({"x": KEY_SHAPED}, KEY_SHAPED),
        ({"outer": [{"nested": (1, KEY_SHAPED)}]}, KEY_SHAPED),
        ([{"db_password": "planted-word"}], "planted-word"),
        (
            {"url": "https://user:planted-word@example.invalid/x"},
            "planted-word",
        ),
        ({KEY_SHAPED: 1}, KEY_SHAPED),
    ],
)
async def test_a_container_holding_a_credential_stays_in_the_harness(cfg, planted):
    worker, shadow = _harness()
    shadow["apis"].apps["cfg"] = cfg
    shadow["apis"].cfg = cfg
    for element in ("cfg", {"item": "cfg"}):
        reply: dict = {}
        with pytest.raises(BoundaryRefusal, match="looks like a credential") as err:
            await worker._answer(
                "describe",
                {"target": {"root": "apis", "path": [element]}},
                shadow,
                reply,
            )
        assert planted not in str(err.value)
        assert planted not in json.dumps(reply)


@pytest.mark.asyncio
async def test_a_container_without_a_credential_crosses_by_value():
    worker, shadow = _harness()
    cfg = {"name": "music", "items": [1, "two", ("three",)], "author": "a"}
    shadow["apis"].apps["cfg"] = cfg
    shadow["apis"].cfg = cfg
    for element in ("cfg", {"item": "cfg"}):
        reply = await _describe(worker, shadow, element)
        assert reply["value"]["kind"] == "value"
    cyclic: list = []
    cyclic.append(cyclic)
    from unify.actor.execution.worker import _credential_inside

    assert _credential_inside(cyclic) is None
    assert _credential_inside({"a": [cyclic, {"b": KEY_SHAPED}]}) is not None


# ── an environment global wrapped for its observers ─────────────────────────


def _observed(value):
    from unify.function_manager.primitives.observers import observed_globals

    return observed_globals({"apis": value}, enabled=True)["apis"]


def test_a_wrapped_object_is_subscripted_as_the_object_it_wraps():
    worker, shadow = _harness()
    shadow["apis"] = _observed(_Apps())
    show = worker._resolve(
        shadow,
        {"root": "apis", "path": [{"item": "music"}, "show"]},
    )
    assert show("song") == "shown song"
    recorder = _Recorder()
    shadow["apis"] = _observed(recorder)
    with pytest.raises(BoundaryRefusal, match="boundary_subscript = True"):
        worker._resolve(shadow, {"root": "apis", "path": [{"item": "music"}]})
    assert recorder.calls == []


def test_a_module_or_the_environment_behind_a_wrapped_object_is_refused():
    worker, shadow = _harness()
    shadow["apis"] = _observed(_Apps())
    for element in ("env", {"item": "env"}):
        with pytest.raises(BoundaryRefusal, match="harness's environment"):
            worker._resolve(shadow, {"root": "apis", "path": [element]})
    for element in ("os", {"item": "os"}):
        with pytest.raises(BoundaryRefusal, match="harness module"):
            worker._resolve(shadow, {"root": "apis", "path": [element]})
