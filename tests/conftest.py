"""Common test fixtures and helpers."""

import os
import sys
from pathlib import Path

import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def data_dir():
    """Path to the data directory."""
    path = Path(__file__).parent.parent / "data"
    if not path.exists():
        pytest.skip("data/ directory not found")
    return str(path)