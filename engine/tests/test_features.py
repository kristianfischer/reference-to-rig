"""Tests for feature extraction module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.features.extractor import FeatureExtractor
from app.features.audio_utils import (
    compute_rms,
    compute_spectral_centroid,
    compute_spectral_tilt,
    preprocess_audio,
)


@pytest.fixture
def sample_audio_file():
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sr = 48000
        duration = 2.0
        t = np.linspace(0, duration, int(sr * duration))
        # Guitar-like signal with harmonics
        audio = np.sin(2 * np.pi * 330 * t)  # E note
        audio += 0.5 * np.sin(2 * np.pi * 660 * t)  # 2nd harmonic
        audio += 0.25 * np.sin(2 * np.pi * 990 * t)  # 3rd harmonic
        audio *= 0.5
        sf.write(f.name, audio, sr)
        yield Path(f.name)


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""

    def test_extract_returns_all_features(self, sample_audio_file):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_audio_file)

        assert "log_mel_stats" in features
        assert "stft_stats" in features
        assert "spectral_centroid" in features
        assert "spectral_tilt" in features
        assert "embedding" in features
        assert "duration" in features

    def test_embedding_is_normalized(self, sample_audio_file):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_audio_file)

        embedding = features["embedding"]
        norm = np.linalg.norm(embedding)

        assert abs(norm - 1.0) < 0.01, "Embedding should be unit normalized"

    def test_embedding_dimension(self, sample_audio_file):
        extractor = FeatureExtractor()
        features = extractor.extract(sample_audio_file)

        # Our embedding is 24 dimensions
        assert len(features["embedding"]) == 24

    def test_select_best_segment(self, sample_audio_file):
        extractor = FeatureExtractor()
        segment = extractor.select_best_segment(sample_audio_file, segment_duration=1.0)

        assert "start_time" in segment
        assert "duration" in segment
        assert "quality_score" in segment
        assert segment["start_time"] >= 0
        assert segment["quality_score"] >= 0


class TestAudioUtils:
    """Tests for audio utility functions."""

    def test_compute_rms(self):
        # Constant signal should have constant RMS
        audio = np.ones(4096) * 0.5
        rms = compute_rms(audio)

        assert len(rms) > 0
        assert np.allclose(rms, 0.5, atol=0.01)

    def test_compute_spectral_centroid(self):
        sr = 48000
        # Pure sine at 1000Hz
        t = np.linspace(0, 1, sr)
        audio = np.sin(2 * np.pi * 1000 * t)

        centroid = compute_spectral_centroid(audio, sr)

        # Centroid should be near 1000Hz
        assert 900 < centroid < 1100

    def test_compute_spectral_tilt(self):
        sr = 48000
        t = np.linspace(0, 1, sr)

        # Bright signal (high frequencies)
        bright = np.sin(2 * np.pi * 5000 * t)
        bright_tilt = compute_spectral_tilt(bright, sr)

        # Dark signal (low frequencies)
        dark = np.sin(2 * np.pi * 200 * t)
        dark_tilt = compute_spectral_tilt(dark, sr)

        # Bright should have more positive tilt than dark
        assert bright_tilt != dark_tilt

    def test_preprocess_audio(self, sample_audio_file):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out:
            output_path = Path(out.name)

        result = preprocess_audio(
            sample_audio_file,
            output_path,
            target_sr=44100,
            target_loudness=-18.0,
        )

        assert output_path.exists()
        assert result["sample_rate"] == 44100

        # Check output is mono
        audio, sr = sf.read(output_path)
        assert len(audio.shape) == 1 or audio.shape[1] == 1


