"""Feature extraction for audio analysis and matching."""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import structlog

from app.features.audio_utils import (
    compute_rms,
    compute_spectral_centroid,
    compute_spectral_tilt,
    find_transients,
)

logger = structlog.get_logger()


class FeatureExtractor:
    """Extract features for tone matching.

    Features extracted:
    - Log-mel spectrogram statistics
    - Multi-resolution STFT statistics
    - Spectral centroid (brightness)
    - Spectral tilt (balance)
    - Embedding vector (for ANN search)
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio_path: Path) -> dict:
        """
        Extract all features from audio file.

        Returns:
            dict with feature keys:
                - log_mel_stats: dict of mel spectrogram statistics
                - stft_stats: dict of multi-resolution STFT stats
                - spectral_centroid: float
                - spectral_tilt: float
                - embedding: numpy array for ANN search
        """
        logger.info("Extracting features", path=str(audio_path))

        # Load audio
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Compute features
        log_mel_stats = self._compute_log_mel_stats(audio, sr)
        stft_stats = self._compute_multi_res_stft_stats(audio, sr)
        centroid = compute_spectral_centroid(audio, sr, self.n_fft)
        tilt = compute_spectral_tilt(audio, sr, self.n_fft)
        embedding = self._compute_embedding(audio, sr, log_mel_stats, centroid, tilt)

        return {
            "log_mel_stats": log_mel_stats,
            "stft_stats": stft_stats,
            "spectral_centroid": centroid,
            "spectral_tilt": tilt,
            "embedding": embedding,
            "duration": len(audio) / sr,
        }

    def _compute_log_mel_stats(self, audio: np.ndarray, sr: int) -> dict:
        """Compute log-mel spectrogram statistics."""
        # Simple mel filterbank implementation
        n_fft = self.n_fft
        hop = self.hop_length
        n_mels = self.n_mels

        # Compute STFT
        n_frames = 1 + (len(audio) - n_fft) // hop
        mel_spec = np.zeros((n_mels, n_frames))

        # Create mel filterbank
        mel_filters = self._create_mel_filterbank(sr, n_fft, n_mels)

        for i in range(n_frames):
            start = i * hop
            end = start + n_fft
            frame = audio[start:end] * np.hanning(n_fft)
            spectrum = np.abs(np.fft.rfft(frame)) ** 2
            mel_spec[:, i] = np.dot(mel_filters, spectrum)

        # Log scale
        log_mel = np.log10(mel_spec + 1e-10)

        return {
            "mean": float(np.mean(log_mel)),
            "std": float(np.std(log_mel)),
            "max": float(np.max(log_mel)),
            "min": float(np.min(log_mel)),
            "band_means": [float(np.mean(log_mel[i::8, :])) for i in range(8)],
        }

    def _create_mel_filterbank(
        self, sr: int, n_fft: int, n_mels: int
    ) -> np.ndarray:
        """Create mel filterbank matrix."""
        fmin = 0
        fmax = sr / 2

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * np.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)

        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        # Convert to FFT bins
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        # Create filterbank
        n_bins = n_fft // 2 + 1
        filterbank = np.zeros((n_mels, n_bins))

        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]

            # Left slope
            for j in range(left, center):
                if center > left:
                    filterbank[i, j] = (j - left) / (center - left)

            # Right slope
            for j in range(center, right):
                if right > center:
                    filterbank[i, j] = (right - j) / (right - center)

        return filterbank

    def _compute_multi_res_stft_stats(self, audio: np.ndarray, sr: int) -> dict:
        """Compute STFT statistics at multiple resolutions."""
        resolutions = [512, 1024, 2048, 4096]
        stats = {}

        for n_fft in resolutions:
            hop = n_fft // 4
            n_frames = 1 + (len(audio) - n_fft) // hop

            magnitudes = []
            for i in range(n_frames):
                start = i * hop
                end = start + n_fft
                if end > len(audio):
                    break
                frame = audio[start:end] * np.hanning(n_fft)
                mag = np.abs(np.fft.rfft(frame))
                magnitudes.append(mag)

            if magnitudes:
                all_mags = np.array(magnitudes)
                stats[f"res_{n_fft}"] = {
                    "mean": float(np.mean(all_mags)),
                    "std": float(np.std(all_mags)),
                    "energy_low": float(np.mean(all_mags[:, :n_fft//8])),
                    "energy_mid": float(np.mean(all_mags[:, n_fft//8:n_fft//2])),
                    "energy_high": float(np.mean(all_mags[:, n_fft//2:])),
                }

        return stats

    def _compute_embedding(
        self,
        audio: np.ndarray,
        sr: int,
        mel_stats: dict,
        centroid: float,
        tilt: float,
    ) -> np.ndarray:
        """
        Compute embedding vector for ANN search.

        This embedding maps audio characteristics to the same semantic space
        as our NAM index: style(4), gain_range(4), brightness(3), tags(13) = 24 dims
        
        We infer these from audio features:
        - Style/gain: detected from distortion/harmonic content
        - Brightness: from spectral centroid
        - Tags: inferred from overall character
        """
        # Index structure matches build_index.py:
        # [0-3]:   style (clean, crunch, overdrive, high_gain)
        # [4-7]:   gain_range (clean, crunch, high_gain, extreme)
        # [8-10]:  brightness (dark, neutral, bright)
        # [11-23]: tags (fender, marshall, mesa, vox, dumble, evh, american, british, boutique, di, sm57, ribbon, blend)
        
        embedding = np.zeros(24, dtype=np.float32)
        
        # Analyze distortion/gain level from harmonic content and dynamics
        rms = compute_rms(audio)
        mean_rms = float(np.mean(rms))
        rms_std = float(np.std(rms))
        peak_to_rms = float(np.max(np.abs(audio))) / (mean_rms + 1e-10)
        
        # Compute THD-like metric (harmonic distortion indicator)
        n_fft = 4096
        spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, 1/sr)
        
        # Find fundamental region (80-400Hz for guitar)
        fund_mask = (freqs >= 80) & (freqs < 400)
        fund_energy = np.sum(spectrum[fund_mask] ** 2)
        
        # Harmonic region (400-4000Hz)
        harm_mask = (freqs >= 400) & (freqs < 4000)
        harm_energy = np.sum(spectrum[harm_mask] ** 2)
        
        # High frequency content (4000Hz+)
        high_mask = freqs >= 4000
        high_energy = np.sum(spectrum[high_mask] ** 2)
        
        total_energy = fund_energy + harm_energy + high_energy + 1e-10
        harm_ratio = harm_energy / total_energy
        high_ratio = high_energy / total_energy
        
        # Estimate gain level
        # Clean: low harmonic ratio, high peak-to-rms (dynamics preserved)
        # Crunch: moderate harmonics, moderate compression
        # High gain: high harmonics, low peak-to-rms (compressed)
        
        gain_score = harm_ratio * 2 + (1 - min(1, peak_to_rms / 10)) * 0.5
        
        # Style encoding (with spread)
        if gain_score < 0.3:
            # Clean
            embedding[0] = 1.0  # clean
            embedding[1] = 0.3  # slight crunch
            embedding[4] = 1.0  # clean gain
            embedding[5] = 0.2
        elif gain_score < 0.5:
            # Crunch
            embedding[0] = 0.3
            embedding[1] = 1.0  # crunch
            embedding[2] = 0.4  # overdrive
            embedding[4] = 0.3
            embedding[5] = 1.0  # crunch gain
            embedding[6] = 0.3
        elif gain_score < 0.7:
            # Overdrive
            embedding[1] = 0.4
            embedding[2] = 1.0  # overdrive
            embedding[3] = 0.4  # high_gain
            embedding[5] = 0.4
            embedding[6] = 1.0  # high_gain
            embedding[7] = 0.2
        else:
            # High gain
            embedding[2] = 0.4
            embedding[3] = 1.0  # high_gain
            embedding[6] = 0.5
            embedding[7] = 1.0  # extreme
        
        # Brightness encoding from spectral centroid
        # Typical centroid ranges: dark <1500, neutral 1500-3000, bright >3000
        if centroid < 1500:
            embedding[8] = 1.0   # dark
            embedding[9] = 0.3   # neutral
        elif centroid < 3000:
            embedding[8] = 0.2
            embedding[9] = 1.0   # neutral
            embedding[10] = 0.2
        else:
            embedding[9] = 0.3
            embedding[10] = 1.0  # bright
        
        # Tag inference (less certain, use lower weights)
        # American vs British character
        # American (Fender): scooped mids, clean headroom
        # British (Marshall): mid-focused, earlier breakup
        mid_mask = (freqs >= 500) & (freqs < 2000)
        mid_energy = np.sum(spectrum[mid_mask] ** 2) / total_energy
        
        if mid_energy < 0.3 and gain_score < 0.4:
            # Scooped, clean = American/Fender character
            embedding[11] = 0.3  # fender
            embedding[17] = 0.3  # american
        elif mid_energy > 0.4:
            # Mid-focused = British character
            embedding[12] = 0.3  # marshall
            embedding[18] = 0.3  # british
        
        # High gain specific tags
        if gain_score > 0.6:
            embedding[13] = 0.2  # mesa
            embedding[16] = 0.2  # evh
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding

    def select_best_segment(
        self,
        audio_path: Path,
        segment_duration: float = 5.0,
        min_rms_threshold: float = 0.01,
    ) -> dict:
        """
        Select the best segment for matching.

        Criteria:
        - Sufficient energy (not silence)
        - Low bleed/noise (based on spectral characteristics)
        - Stable tone (consistent energy)

        Returns:
            dict with segment info:
                - start_time: float
                - duration: float
                - quality_score: float
        """
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        segment_samples = int(segment_duration * sr)
        hop_samples = segment_samples // 4

        best_score = -1
        best_start = 0

        n_segments = max(1, (len(audio) - segment_samples) // hop_samples)

        for i in range(n_segments):
            start = i * hop_samples
            end = start + segment_samples
            if end > len(audio):
                break

            segment = audio[start:end]

            # Compute segment quality score
            rms = np.sqrt(np.mean(segment ** 2))
            if rms < min_rms_threshold:
                continue

            # Energy stability (lower variance = more stable)
            rms_frames = compute_rms(segment)
            stability = 1.0 / (1.0 + np.std(rms_frames))

            # Spectral clarity (higher centroid variance = more musical content)
            centroid = compute_spectral_centroid(segment, sr)
            clarity = min(1.0, centroid / 3000)

            score = rms * stability * clarity

            if score > best_score:
                best_score = score
                best_start = start

        return {
            "start_time": best_start / sr,
            "duration": segment_duration,
            "quality_score": best_score,
        }


