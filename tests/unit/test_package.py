"""Basic package tests."""

import pharia_telemetry


def test_package_import():
    """Test that the package can be imported."""
    assert pharia_telemetry.__version__ is not None


def test_package_version():
    """Test that the package has a version."""
    assert isinstance(pharia_telemetry.__version__, str)
    assert len(pharia_telemetry.__version__.split(".")) >= 2
