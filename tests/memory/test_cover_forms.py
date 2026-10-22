"""What a recorded cover can be given as (:func:`unify.memory.cover_forms.unfit_reason`), and the blob a work-tree
action shows (:func:`unify.memory.cover_forms.blob_sha`). Records only: nothing is run.

The held-out check that ran an item on unseen values is gone (rev 2.1: the gate runs no code); the structural check
that a declared ``input`` form is one the covers can give stays, with the same reasons.
"""

from unify.memory.cover_forms import blob_sha, unfit_reason
from unify.memory.episodes import Action

TOOL = Action(0, "shop", "search", [], {"colour": "red"}, {"items": []}, "ok", "read")
TOOL_TEXT = Action(1, "shop", "search", [], {"colour": "red"}, "3 results", "ok", "read")
TOOL_JSON_TEXT = Action(2, "shop", "search", [], {}, '{"items": [1]}', "ok", "read")
WORKTREE = Action(
    3, "worktree:workspace", "write", ["a.txt"], {}, {"blob_before": "b0", "blob_after": "b1"}, "ok", kind="worktree"
)
WORKTREE_READ = Action(
    4, "worktree:workspace", "read", ["a.txt"], {}, {"blob_before": "b0", "blob_after": "b0"}, "ok", kind="worktree"
)
DIALOGUE_DICT = Action(5, "dialogue:user", "reply", ["look"], {}, {"screen": "x"}, "ok", kind="dialogue")
DIALOGUE_TEXT = Action(6, "dialogue:user", "reply", ["look"], {}, "Inventory: wood 1", "ok", kind="dialogue")


def test_a_declared_form_a_cover_cannot_give_has_a_reason():
    assert unfit_reason(DIALOGUE_DICT, "path") == "declares input path, which a dialogue cover cannot give"
    assert unfit_reason(WORKTREE, "env") == "declares input env, which a worktree cover cannot give"
    assert unfit_reason(TOOL, "bytes") == "declares input bytes, which a tool cover cannot give"
    assert unfit_reason(DIALOGUE_DICT, "text") == (
        "declares input text, which a dialogue cover with a non-text observation cannot give"
    )
    assert unfit_reason(TOOL_TEXT, "observation") == (
        "declares input observation, which a tool cover without a JSON response cannot give"
    )


def test_forms_the_covers_can_give_and_an_undeclared_form_have_none():
    assert unfit_reason(WORKTREE, "text") is None
    assert unfit_reason(TOOL, "env") is None
    assert unfit_reason(DIALOGUE_DICT, "observation") is None
    assert unfit_reason(DIALOGUE_TEXT, "text") is None
    assert unfit_reason(TOOL_JSON_TEXT, "observation") is None  # a string holding JSON is a JSON observation
    assert unfit_reason(TOOL_TEXT, None) is None


def test_a_work_tree_actions_blob_is_a_writes_after_blob_else_its_before_blob():
    assert blob_sha(WORKTREE) == "b1"
    assert blob_sha(WORKTREE_READ) == "b0"
    assert blob_sha(TOOL) is None
