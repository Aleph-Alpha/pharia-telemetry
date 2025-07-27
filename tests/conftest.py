"""Pytest configuration and shared fixtures."""

import pytest


@pytest.fixture
def sample_data():
    """Sample data for testing."""
    return {"test": "data"}


# Add more fixtures as needed for telemetry testing
