"""
Acoustic Feature Extraction for NAM Matching.

This module extracts a 155-dimensional feature vector from audio that captures:
- Spectral characteristics (brightness, EQ curve, tonal character)
- Dynamics (compression, touch sensitivity, feel)
- Harmonic content (distortion character, gain amount)
- Texture (multi-resolution spectral patterns)

The same features are extracted from:
1. Reference audio (isolated guitar from user)
2. NAM probe outputs (DI signals rendered through NAM models)

This enables direct acoustic comparison for tone matching.
"""

import json
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import structlog
from scipy import signal

logger = structlog.get_logger()

# Feature configuration
N_MELS = 64
HOP_LENGTH = 512
N_FFT = 2048
SAMPLE_RATE = 48000

# Total feature dimensions: 64 + 64 + 10 + 10 + 6 + 1 = 155
FEATURE_DIM = 155


class AcousticFeatureExtractor:
    """
    Extract 155-dimensional acoustic feature vectors for tone matching.
    
    Feature breakdown:
    - Mel mean (64 dims): Average energy per mel band
    - Mel std (64 dims): Energy variance per mel band  
    - Spectral stats (10 dims): Centroid, rolloff, flatness, etc.
    - Dynamics stats (10 dims): RMS, crest factor, loudness, etc.
    - Band ratios (6 dims): Energy distribution across frequency bands
    - Level-dependent (1 dim): How brightness changes with level
    
    Total: 155 dimensions
    """
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        normalization_stats_path: Optional[Path] = None,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Loudness meter (ITU BS.1770)
        self.meter = pyln.Meter(sample_rate)
        
        # Load normalization stats if available
        self.norm_mean = None
        self.norm_std = None
        if normalization_stats_path and normalization_stats_path.exists():
            self._load_normalization_stats(normalization_stats_path)
    
    def _load_normalization_stats(self, path: Path) -> None:
        """Load pre-computed normalization statistics."""
        with open(path) as f:
            stats = json.load(f)
        self.norm_mean = np.array(stats["mean"], dtype=np.float32)
        self.norm_std = np.array(stats["std"], dtype=np.float32)
        logger.info("Loaded normalization stats", path=str(path))
    
    def save_normalization_stats(self, path: Path) -> None:
        """Save normalization statistics."""
        if self.norm_mean is None or self.norm_std is None:
            raise ValueError("No normalization stats to save")
        
        stats = {
            "mean": self.norm_mean.tolist(),
            "std": self.norm_std.tolist(),
            "feature_dim": FEATURE_DIM,
        }
        with open(path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info("Saved normalization stats", path=str(path))
    
    def compute_normalization_stats(self, feature_vectors: np.ndarray) -> None:
        """
        Compute normalization statistics from a collection of feature vectors.
        
        Args:
            feature_vectors: Array of shape (n_samples, FEATURE_DIM)
        """
        self.norm_mean = np.mean(feature_vectors, axis=0).astype(np.float32)
        self.norm_std = np.std(feature_vectors, axis=0).astype(np.float32)
        # Prevent division by zero
        self.norm_std = np.maximum(self.norm_std, 1e-6)
        logger.info(
            "Computed normalization stats",
            n_samples=len(feature_vectors),
            mean_range=(float(self.norm_mean.min()), float(self.norm_mean.max())),
        )
    
    def extract_from_file(self, audio_path: Path) -> np.ndarray:
        """
        Extract features from an audio file.
        
        Args:
            audio_path: Path to audio file (WAV, FLAC, etc.)
            
        Returns:
            155-dimensional feature vector (L2 normalized)
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return self.extract(audio)
    
    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features from audio array.
        
        Args:
            audio: Mono audio array, assumed to be at self.sample_rate
            
        Returns:
            155-dimensional feature vector (L2 normalized)
        """
        # Ensure minimum length
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        
        features = []
        
        # 1. MEL SPECTROGRAM FEATURES (128 dims)
        mel_features = self._extract_mel_features(audio)
        features.extend(mel_features)
        
        # 2. SPECTRAL SHAPE FEATURES (10 dims)
        spectral_features = self._extract_spectral_features(audio)
        features.extend(spectral_features)
        
        # 3. DYNAMICS FEATURES (10 dims)
        dynamics_features = self._extract_dynamics_features(audio)
        features.extend(dynamics_features)
        
        # 4. BAND RATIO FEATURES (6 dims)
        band_features = self._extract_band_features(audio)
        features.extend(band_features)
        
        # 5. LEVEL-DEPENDENT FEATURE (1 dim)
        level_features = self._extract_level_dependent_features(audio)
        features.extend(level_features)
        
        # Convert to numpy array
        feature_vector = np.array(features, dtype=np.float32)
        
        # Apply z-score normalization if stats are available
        if self.norm_mean is not None and self.norm_std is not None:
            feature_vector = (feature_vector - self.norm_mean) / self.norm_std
        
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(feature_vector)
        if norm > 0:
            feature_vector = feature_vector / norm
        
        return feature_vector
    
    def _extract_mel_features(self, audio: np.ndarray) -> list:
        """Extract mel spectrogram statistics (128 dims)."""
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        
        # Convert to log scale (dB)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # Statistics per mel band
        mel_mean = np.mean(log_mel, axis=1)  # 64 dims
        mel_std = np.std(log_mel, axis=1)    # 64 dims
        
        features = []
        features.extend(mel_mean.tolist())
        features.extend(mel_std.tolist())
        
        return features
    
    def _extract_spectral_features(self, audio: np.ndarray) -> list:
        """Extract spectral shape features (10 dims)."""
        features = []
        
        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features.append(float(np.mean(centroid)))
        features.append(float(np.std(centroid)))
        
        # Spectral rolloff (where HF energy drops)
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features.append(float(np.mean(rolloff)))
        features.append(float(np.std(rolloff)))
        
        # Spectral flatness (tonal vs noisy)
        flatness = librosa.feature.spectral_flatness(
            y=audio, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features.append(float(np.mean(flatness)))
        features.append(float(np.std(flatness)))
        
        # Spectral bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        features.append(float(np.mean(bandwidth)))
        features.append(float(np.std(bandwidth)))
        
        # Zero crossing rate (rough texture proxy)
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))
        
        return features
    
    def _extract_dynamics_features(self, audio: np.ndarray) -> list:
        """Extract dynamics/loudness features (10 dims)."""
        features = []
        
        # RMS energy
        rms = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))
        features.append(float(np.max(rms)))
        features.append(float(np.percentile(rms, 90)))
        
        # Crest factor (peak / RMS) - indicates compression
        peak = np.max(np.abs(audio))
        mean_rms = np.mean(rms)
        crest_factor = peak / (mean_rms + 1e-10)
        features.append(float(crest_factor))
        
        # Integrated loudness (LUFS)
        try:
            loudness = self.meter.integrated_loudness(audio)
            if np.isnan(loudness) or np.isinf(loudness):
                loudness = -60.0
        except Exception:
            loudness = -60.0
        features.append(float(loudness))
        
        # Onset strength (transient intensity)
        onset_env = librosa.onset.onset_strength(
            y=audio, sr=self.sample_rate, hop_length=self.hop_length
        )
        features.append(float(np.mean(onset_env)))
        features.append(float(np.std(onset_env)))
        
        # Dynamic range (difference between loud and quiet parts)
        rms_db = librosa.amplitude_to_db(rms)
        dynamic_range = float(np.percentile(rms_db, 95) - np.percentile(rms_db, 5))
        features.append(dynamic_range)
        
        # Attack slope (how fast transients rise)
        # Use onset envelope derivative
        onset_diff = np.diff(onset_env)
        attack_slope = float(np.percentile(onset_diff[onset_diff > 0], 90)) if np.any(onset_diff > 0) else 0.0
        features.append(attack_slope)
        
        return features
    
    def _extract_band_features(self, audio: np.ndarray) -> list:
        """Extract frequency band energy ratios (6 dims)."""
        # Compute power spectrum
        n_fft = 4096  # Higher resolution for band analysis
        spectrum = np.abs(librosa.stft(audio, n_fft=n_fft)) ** 2
        freqs = librosa.fft_frequencies(sr=self.sample_rate, n_fft=n_fft)
        
        # Sum power across time
        total_power = np.sum(spectrum)
        if total_power == 0:
            return [0.0] * 6
        
        # Define bands (Hz) - tuned for guitar
        bands = [
            (20, 150),      # Sub/bass
            (150, 400),     # Low-mid (body)
            (400, 1000),    # Mid (presence)
            (1000, 2500),   # Upper-mid (cut)
            (2500, 6000),   # Presence/bite
            (6000, 20000),  # Air/sizzle
        ]
        
        features = []
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            band_power = np.sum(spectrum[mask, :])
            ratio = band_power / total_power
            features.append(float(ratio))
        
        return features
    
    def _extract_level_dependent_features(self, audio: np.ndarray) -> list:
        """
        Extract level-dependent coloration feature (1 dim).
        
        Measures how spectral centroid changes with amplitude.
        High values = brighter when louder (typical of tube amps).
        """
        # Get frame-by-frame RMS and centroid
        rms = librosa.feature.rms(
            y=audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        # Ensure same length
        min_len = min(len(rms), len(centroid))
        rms = rms[:min_len]
        centroid = centroid[:min_len]
        
        # Filter out silent frames
        mask = rms > np.percentile(rms, 10)
        if np.sum(mask) < 10:
            return [0.0]
        
        rms_filtered = rms[mask]
        centroid_filtered = centroid[mask]
        
        # Linear regression: centroid vs RMS
        # slope indicates level-dependent brightness
        try:
            slope, _ = np.polyfit(rms_filtered, centroid_filtered, 1)
            # Normalize by mean centroid
            mean_centroid = np.mean(centroid_filtered)
            if mean_centroid > 0:
                slope = slope / mean_centroid
            return [float(slope)]
        except Exception:
            return [0.0]
    
    def extract_raw(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract features WITHOUT normalization.
        Used for computing normalization stats across the library.
        """
        if len(audio) < self.n_fft:
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        
        features = []
        features.extend(self._extract_mel_features(audio))
        features.extend(self._extract_spectral_features(audio))
        features.extend(self._extract_dynamics_features(audio))
        features.extend(self._extract_band_features(audio))
        features.extend(self._extract_level_dependent_features(audio))
        
        return np.array(features, dtype=np.float32)


def extract_features(audio_path: Path, extractor: Optional[AcousticFeatureExtractor] = None) -> np.ndarray:
    """
    Convenience function to extract features from a file.
    
    Args:
        audio_path: Path to audio file
        extractor: Optional pre-configured extractor
        
    Returns:
        155-dimensional feature vector
    """
    if extractor is None:
        extractor = AcousticFeatureExtractor()
    return extractor.extract_from_file(audio_path)

