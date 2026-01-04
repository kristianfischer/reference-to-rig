"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Add engine to path
engine_path = Path(__file__).parent.parent
sys.path.insert(0, str(engine_path))

# Set test environment
os.environ["RTR_DEBUG"] = "true"
os.environ["RTR_DATA_DIR"] = str(Path(__file__).parent / "test_data")


@pytest.fixture(autouse=True)
def setup_test_env(tmp_path):
    """Setup test environment for each test."""
    import os
    os.environ["RTR_DATA_DIR"] = str(tmp_path / "data")
    os.environ["RTR_PROJECTS_DIR"] = str(tmp_path / "projects")
    os.environ["RTR_CAPTURE_LIBRARY_DIR"] = str(tmp_path / "captures")

    # Create directories
    (tmp_path / "data").mkdir()
    (tmp_path / "projects").mkdir()
    (tmp_path / "captures").mkdir()
    (tmp_path / "captures" / "nam_models").mkdir()
    (tmp_path / "captures" / "cab_irs").mkdir()


