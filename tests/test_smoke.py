"""Smoke test to verify the package is importable."""


def test_import() -> None:
    """Verify ngboost_lightning can be imported."""
    import ngboost_lightning

    assert hasattr(ngboost_lightning, "__version__")
    assert ngboost_lightning.__version__ == "0.1.0"
