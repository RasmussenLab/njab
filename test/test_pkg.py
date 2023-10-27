def test_package():
    """Test that the package is importable."""
    import njab
    assert njab.__version__
    import njab.plotting.metrics
