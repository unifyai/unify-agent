# tests/memory_v2/test_memory_repo.py
from unify.memory_v2.gitio import Repo
from unify.memory_v2.memory_repo import (
    MemoryRepo,
    items,
    remove_function,
    unlist_function,
)

MOD = '''"""Venmo."""
__all__ = ["login", "list_friends"]

def _token(c):
    return c

def login(apis, username: str, password: str) -> str:
    """Log in once and return the access token.

    Effect: read
    Input: env
    """
    return apis.venmo.login(username=username, password=password)["access_token"]

def list_friends(apis, token: str, page_limit: int = 20) -> list:
    """All friends, across pages.

    Effect: read
    """
    return []
'''


def _checkout(tmp_path):
    root = tmp_path / "co"
    (root / "env" / "venmo").mkdir(parents=True)
    (root / "env" / "venmo" / "__init__.py").write_text(MOD)
    (root / "env" / "venmo" / "NOTES.md").write_text(
        "# venmo\n\n## Pagination\nPages hold 20.\n",
    )
    (root / "workflows").mkdir()
    (root / "workflows" / "pay.md").write_text(
        "---\ntitle: Pay back\n---\nUse login.\n",
    )
    return root


def test_items_lists_public_functions_notes_and_workflows(tmp_path):
    rep = items(_checkout(tmp_path))
    ids = {i.item_id: i for i in rep.items}
    assert "env/venmo:login" in ids and "env/venmo:_token" not in ids
    assert ids["env/venmo:login"].effect == "read" and ids["env/venmo:login"].listed
    assert ids["env/venmo:login"].input == "env"
    assert ids["env/venmo:list_friends"].input == ""  # no Input: line
    assert (
        "env/venmo/NOTES.md#pagination" in ids
        and "workflows/pay.md" in ids
        and rep.errors == []
    )


def test_items_reports_syntax_error(tmp_path):
    root = _checkout(tmp_path)
    (root / "env" / "venmo" / "__init__.py").write_text("def broken(:\n")
    rep = items(root)
    assert rep.errors and rep.errors[0].startswith("env/venmo/__init__.py")


def test_unlist_keeps_code_importable(tmp_path):
    src = unlist_function(MOD, "list_friends")
    assert "def list_friends" in src and '"list_friends"' not in src.split("def ")[0]


def test_remove_function_cuts_def_and_all_entry():
    src = remove_function(MOD, "login")
    assert "def login" not in src and '"login"' not in src and "def list_friends" in src
    compile(src, "x", "exec")


def test_hide_commits_on_main_with_trailers(tmp_path):
    repo = Repo.init_bare(tmp_path / "mem.git")
    with repo.temp_checkout() as wt:
        (wt / "env" / "venmo").mkdir(parents=True)
        (wt / "env" / "venmo" / "__init__.py").write_text(MOD)
        s = repo.commit_all(wt, "seed", {})
    repo.fast_forward("main", s, expected_old=repo.head())
    sha = MemoryRepo(repo).hide("env/venmo:login", "input check failed", ["e7"])
    body = repo.run("log", "-1", "--format=%B", sha)
    assert "Hide: input check failed" in body and "Evidence: e7" in body
    assert "def login" not in repo.show("main", "env/venmo/__init__.py").decode()


NO_ALL = '''"""Mod."""
from __future__ import annotations


def _h():
    return 1


def a(apis):
    """A.

    Effect: read
    """


def b(apis):
    """B.

    Effect: read
    """
'''


def test_unlist_without_all_synthesizes_it(tmp_path):
    from unify.memory_v2.index import build_index

    src = unlist_function(NO_ALL, "b")
    compile(src, "x", "exec")
    assert "def b" in src and '__all__ = ["a"]' in src
    assert src.index("__future__") < src.index("__all__")
    root = tmp_path / "co"
    (root / "env" / "m").mkdir(parents=True)
    (root / "env" / "m" / "__init__.py").write_text(src)
    listed = {i.name for i in items(root).items if i.listed}
    assert listed == {"a"}
    idx = build_index(root)
    assert "a(apis)" in idx and "b(apis)" not in idx


def test_annassign_all_is_recognised(tmp_path):
    src = MOD.replace("__all__ =", "__all__: list[str] =")
    root = tmp_path / "co"
    (root / "env" / "m").mkdir(parents=True)
    (root / "env" / "m" / "__init__.py").write_text(
        unlist_function(src, "list_friends"),
    )
    listed = {i.name: i.listed for i in items(root).items}
    assert listed == {"login": True, "list_friends": False}


def test_items_reports_undecodable_module(tmp_path):
    root = _checkout(tmp_path)
    (root / "env" / "venmo" / "__init__.py").write_bytes(b"\xff\xfe\x00bad")
    assert items(root).errors
