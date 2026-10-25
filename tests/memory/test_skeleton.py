def test_package_imports():
    import unify.memory as m

    assert "docs/memory.md" in (m.__doc__ or "")
