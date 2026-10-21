import pytest

from unify.setting_aliases import FORMER_NAMES, apply_former_names


def test_a_former_name_is_read_as_its_current_name():
    old, new = next(iter(FORMER_NAMES.items()))
    env = {old: "on"}
    assert apply_former_names(env) == [old] and env == {new: "on"}


def test_a_former_token_name_leaves_the_environment():
    env = {"unify_memory_v2_sol_token": "t", "OTHER": "x"}  # pragma: allowlist secret
    assert apply_former_names(env) == ["unify_memory_v2_sol_token"]
    assert env == {
        "UNIFY_MEMORY_WRITER_TOKEN": "t",
        "OTHER": "x",
    }  # pragma: allowlist secret


def test_both_names_with_different_values_are_refused():
    old, new = next(iter(FORMER_NAMES.items()))
    with pytest.raises(ValueError):
        apply_former_names({old: "on", new: "off"})


def test_every_current_name_is_a_setting():
    from unify.settings import ProductionSettings

    fields = set(ProductionSettings.model_fields)
    assert set(FORMER_NAMES.values()) <= fields
