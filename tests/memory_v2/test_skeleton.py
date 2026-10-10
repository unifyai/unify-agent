def test_package_imports():
    import unify.memory_v2 as m

    assert "spec v0" in (m.__doc__ or "")
