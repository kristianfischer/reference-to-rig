"""Tests for SAM Audio isolation backend.

Note: These tests require SAM Audio to be installed and configured.
Tests are skipped if SAM Audio is not available.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Check if SAM Audio is available
try:
    from sam_audio import SAMAudio, SAMAudioProcessor
    SAM_AUDIO_AVAILABLE = True
except ImportError:
    SAM_AUDIO_AVAILABLE = False

from app.isolation.adapter import SAMAudioBackend, MockSAMBackend, get_isolation_backend


@pytest.fixture
def sample_audio_file():
    """Create a temporary audio file with guitar-like content."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sr = 48000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))

        # Create guitar-like signal (E chord harmonics)
        freqs = [82.41, 110.0, 146.83, 164.81, 196.0, 246.94]  # E chord
        audio = np.zeros_like(t)
        for freq in freqs:
            # Fundamental + harmonics
            audio += np.sin(2 * np.pi * freq * t) * 0.3
            audio += np.sin(2 * np.pi * freq * 2 * t) * 0.15
            audio += np.sin(2 * np.pi * freq * 3 * t) * 0.05

        # Add some "strumming" envelope
        envelope = np.exp(-t / 2) * (1 - np.exp(-t * 10))
        audio = audio * envelope

        # Add a bit of noise (like amp hiss)
        audio += np.random.randn(len(audio)) * 0.02

        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.9

        sf.write(f.name, audio, sr)
        yield Path(f.name)


class TestSAMAudioBackend:
    """Tests for real SAM Audio backend."""

    @pytest.mark.skipif(not SAM_AUDIO_AVAILABLE, reason="SAM Audio not installed")
    def test_backend_initialization(self):
        """Test that backend initializes without loading model."""
        backend = SAMAudioBackend(
            model_name="facebook/sam-audio-large",
            device="cpu",
        )
        assert backend.name == "sam_audio"
        assert backend._model is None  # Model not loaded yet

    @pytest.mark.skipif(not SAM_AUDIO_AVAILABLE, reason="SAM Audio not installed")
    @pytest.mark.slow
    def test_isolate_basic(self, sample_audio_file):
        """Test basic isolation (requires GPU for reasonable speed)."""
        backend = SAMAudioBackend(prompt="Electric guitar")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            output_path = Path(out.name)

        result = backend.isolate(sample_audio_file, output_path)

        assert output_path.exists()
        assert "confidence" in result
        assert "duration" in result
        assert 0 <= result["confidence"] <= 1
        assert result["duration"] > 0

    @pytest.mark.skipif(not SAM_AUDIO_AVAILABLE, reason="SAM Audio not installed")
    @pytest.mark.slow
    def test_isolate_with_reranking(self, sample_audio_file):
        """Test isolation with candidate re-ranking."""
        backend = SAMAudioBackend(
            prompt="Electric guitar playing",
            use_reranking=True,
            reranking_candidates=2,
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            output_path = Path(out.name)

        result = backend.isolate(sample_audio_file, output_path)

        assert output_path.exists()
        assert result["prompt_used"] == "Electric guitar playing"


class TestBackendFactory:
    """Tests for backend factory function."""

    def test_get_mock_backend(self):
        """Test getting mock backend."""
        backend = get_isolation_backend("mock")
        assert isinstance(backend, MockSAMBackend)

    @pytest.mark.skipif(not SAM_AUDIO_AVAILABLE, reason="SAM Audio not installed")
    def test_get_sam_audio_backend(self):
        """Test getting SAM Audio backend."""
        backend = get_isolation_backend("sam_audio")
        assert isinstance(backend, SAMAudioBackend)

    def test_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError):
            get_isolation_backend("invalid_backend")


class TestMockBackendFallback:
    """Test mock backend as fallback when SAM Audio unavailable."""

    def test_mock_produces_output(self, sample_audio_file):
        """Verify mock backend works as fallback."""
        backend = MockSAMBackend()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            output_path = Path(out.name)

        result = backend.isolate(sample_audio_file, output_path)

        assert output_path.exists()
        assert result["confidence"] > 0
        assert result["duration"] > 0

        # Verify output audio is valid
        audio, sr = sf.read(output_path)
        assert len(audio) > 0
        assert sr == 48000


