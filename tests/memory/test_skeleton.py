def test_package_imports():
    import unify.memory as m

    assert "spec v0" in (m.__doc__ or "")
