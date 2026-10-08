from unify.memory_v2.episodes import Action
from unify.memory_v2.fingerprint import Generations, error_signature, fingerprint, shape


def test_shape_ignores_values_and_list_length():
    assert shape({"a": [1, 2, 3], "b": "x"}) == shape({"b": "y", "a": [9]})
    assert shape({"a": 1}) != shape({"a": "1"})


def test_error_signature_normalises_digits():
    assert (
        error_signature("KeyError: 'ship_date' at row 12\nmore")
        == "KeyError: 'ship_date' at row #"
    )


def test_generation_bumps_on_new_shape_and_new_error_but_not_new_method():
    g = Generations()
    a = [Action(0, "venmo", "login", [], {}, {"token": "t"}, "ok")]
    assert g.observe(fingerprint(a)) == set()
    assert (
        g.observe(fingerprint([Action(0, "venmo", "friends", [], {}, [], "ok")]))
        == set()
    )  # new method: no bump
    assert g.observe(
        fingerprint(
            [Action(0, "venmo", "login", [], {}, {"token": "t", "exp": 3}, "ok")],
        ),
    ) == {"venmo"}
    assert g.generation("venmo") == 1
    assert g.observe(
        fingerprint(
            [Action(0, "venmo", "friends", [], {}, None, "error", error="HTTP 422")],
        ),
    ) == {"venmo"}


# --- per-kind fingerprints (spec J1) ---------------------------------------------------------------------


def _sh(tail, code=0, ch="shell:uv"):
    return Action(
        0,
        ch,
        "run",
        ["uv run pytest"],
        {},
        {"exit_code": code, "tail": tail},
        "ok" if code == 0 else "error",
        kind="shell",
    )


def _wt(path, shape_):
    return Action(
        0,
        "worktree:workspace",
        "read",
        [path],
        {},
        {"blob_before": None, "blob_after": None, "size": 1, "shape": shape_},
        "ok",
        "read",
        kind="worktree",
    )


def _dl(obs):
    return Action(-1, "dialogue:user", "reply", ["do"], {}, obs, "ok", kind="dialogue")


def test_tool_fingerprint_entries_keep_their_v0_form():
    fp = fingerprint([Action(0, "venmo", "me", [], {}, {"user_id": "u"}, "ok")])
    assert fp == {"venmo.me": {"shapes": ["{user_id:str}"], "errors": []}}


def test_shell_fingerprint_output_shapes_and_exit_codes():
    from unify.memory_v2.analysis import shellout

    fp = fingerprint(
        [_sh("=== 3 passed in 0.1s ==="), _sh("=== 1 failed in 0.2s ===", 1)],
    )
    [(key, entry)] = fp.items()
    assert key == "shell:uv.run" and entry["channel"] == "shell:uv"
    assert entry["shapes"] == [shellout.signature({"format": "lines"})]
    assert entry["errors"] == ["exit:1"]
    g = Generations()
    g.observe(fingerprint([_sh("=== 3 passed in 0.1s ===")]))
    assert (
        g.observe(fingerprint([_sh("=== 4 passed in 0.3s ===")])) == set()
    )  # same shape
    assert g.observe(fingerprint([_sh("=== 1 failed ===", 2)])) == {
        "shell:uv",
    }  # first error
    assert g.observe(fingerprint([_sh('{"ok": true}')])) == {
        "shell:uv",
    }  # new output shape
    assert g.generation("shell:uv") == 2


def test_worktree_fingerprint_per_path_family_from_recorded_shapes_or_blobs(tmp_path):
    from unify.memory_v2.analysis import shapes
    from unify.memory_v2.blobs import BlobStore
    from unify.memory_v2.fingerprint import path_family

    assert path_family("finance/ap/2026-10/invoices.csv") == "finance/ap/#-#/*.csv"
    assert path_family("README") == "*"
    comma = b"vendor_id,amount\nV-1,2.5\n"
    longer = b"vendor_id,amount\nV-1,2.5\nV-2,7\n"
    tab = b"vendor_id\tamount\nV-1\t2.5\n"
    g = Generations()
    a = _wt("finance/ap/2026-10/inv.csv", shapes.shape("inv.csv", comma))
    assert g.observe(fingerprint([a])) == set()
    b = _wt("finance/ap/2026-11/inv.csv", shapes.shape("inv.csv", longer))
    fp = fingerprint([a, b])
    assert (
        list(fp) == ["worktree:workspace.finance/ap/#-#/*.csv"]
        and len(fp[list(fp)[0]]["shapes"]) == 1
    )
    assert g.observe(fp) == set()  # more rows: same shape
    other = _wt("ops/stock.tsv", shapes.shape("stock.tsv", tab))
    assert g.observe(fingerprint([other])) == set()  # a new family is not drift
    store = BlobStore(tmp_path / "b")
    sha = store.put(tab)
    c = Action(
        0,
        "worktree:workspace",
        "read",
        ["finance/ap/2026-12/inv.csv"],
        {},
        {"blob_before": sha, "blob_after": sha, "size": len(tab)},
        "ok",
        kind="worktree",
    )
    assert fingerprint([c]) == {
        "worktree:workspace.finance/ap/#-#/*.csv": {
            "shapes": [],
            "errors": [],
            "channel": "worktree:workspace",
        },
    }  # no recorded shape and no blob store: nothing to compare
    assert g.observe(fingerprint([c], blobs=store)) == {
        "worktree:workspace",
    }  # tab-delimited now
    listing = Action(
        0,
        "worktree:workspace",
        "list",
        ["ap"],
        {},
        {"entries": []},
        "ok",
        kind="worktree",
    )
    assert fingerprint([listing]) == {}


def test_dialogue_fingerprint_observation_shapes():
    g = Generations()
    g.observe(fingerprint([_dl("Status: health 9\nInventory: wood 1")]))
    assert (
        g.observe(fingerprint([_dl("Status: health 3\nInventory: stone 2")])) == set()
    )
    assert g.observe(
        fingerprint([_dl("Status: health 3\nInventory: x\nNearby: table")]),
    ) == {
        "dialogue:user",
    }
