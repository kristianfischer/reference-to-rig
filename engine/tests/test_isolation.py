"""Tests for guitar isolation module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.isolation.adapter import MockSAMBackend, get_isolation_backend


@pytest.fixture
def sample_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Generate test audio (440Hz sine + noise)
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t) * 0.5
        audio += np.random.randn(len(audio)) * 0.1
        sf.write(f.name, audio, sr)
        yield Path(f.name)


class TestMockSAMBackend:
    """Tests for mock SAM isolation backend."""

    def test_backend_name(self):
        backend = MockSAMBackend()
        assert backend.name == "mock_sam"

    def test_isolate_produces_output(self, sample_audio_file):
        backend = MockSAMBackend()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            output_path = Path(out.name)

        result = backend.isolate(sample_audio_file, output_path)

        assert output_path.exists()
        assert "confidence" in result
        assert "duration" in result
        assert 0 <= result["confidence"] <= 1
        assert result["duration"] > 0

    def test_isolate_preserves_duration(self, sample_audio_file):
        backend = MockSAMBackend()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            output_path = Path(out.name)

        # Get input duration
        input_info = sf.info(sample_audio_file)

        result = backend.isolate(sample_audio_file, output_path)

        # Check duration is approximately preserved
        assert abs(result["duration"] - input_info.duration) < 0.1

    def test_progress_callback(self, sample_audio_file):
        backend = MockSAMBackend()
        progress_values = []

        def callback(p):
            progress_values.append(p)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            output_path = Path(out.name)

        backend.isolate(sample_audio_file, output_path, progress_callback=callback)

        # Check progress was reported
        assert len(progress_values) > 0
        assert progress_values[-1] == 1.0


class TestGetIsolationBackend:
    """Tests for backend factory."""

    def test_get_mock_backend(self):
        backend = get_isolation_backend("mock")
        assert isinstance(backend, MockSAMBackend)

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            get_isolation_backend("nonexistent")


