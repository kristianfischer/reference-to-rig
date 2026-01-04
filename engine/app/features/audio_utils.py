"""Audio utility functions for preprocessing and analysis."""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import structlog
from scipy import signal

logger = structlog.get_logger()


def get_audio_info(path: Path) -> dict:
    """Get basic audio file information."""
    info = sf.info(path)
    return {
        "duration": info.duration,
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "format": info.format,
        "subtype": info.subtype,
    }


def preprocess_audio(
    input_path: Path,
    output_path: Path,
    target_sr: int = 48000,
    target_loudness: float = -18.0,
) -> dict:
    """
    Preprocess audio for analysis.

    1. Resample to target sample rate
    2. Convert to mono
    3. Loudness normalize to target LUFS

    Args:
        input_path: Source audio file
        output_path: Output path for preprocessed audio
        target_sr: Target sample rate (default 48kHz)
        target_loudness: Target loudness in LUFS (default -18)

    Returns:
        dict with preprocessing info
    """
    import pyloudnorm as pyln
    from scipy.signal import resample_poly
    from math import gcd

    logger.info("Preprocessing audio", input=str(input_path), target_sr=target_sr)

    # Load audio
    audio, sr = sf.read(input_path)

    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != target_sr:
        # Use rational resampling
        g = gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up, down)
        sr = target_sr

    # Loudness normalize
    meter = pyln.Meter(sr)
    current_loudness = meter.integrated_loudness(audio)

    if not np.isinf(current_loudness):
        gain_db = target_loudness - current_loudness
        gain_linear = 10 ** (gain_db / 20)
        audio = audio * gain_linear

        # Prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.99:
            audio = audio / max_val * 0.99
    else:
        logger.warning("Could not measure loudness, skipping normalization")

    # Save preprocessed audio
    sf.write(output_path, audio, sr)

    duration = len(audio) / sr
    logger.info(
        "Preprocessing complete",
        output=str(output_path),
        duration=duration,
        sample_rate=sr,
    )

    return {
        "duration": duration,
        "sample_rate": sr,
        "loudness": target_loudness,
    }


def compute_rms(audio: np.ndarray, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """Compute RMS energy over frames."""
    n_frames = 1 + (len(audio) - frame_length) // hop_length
    rms = np.zeros(n_frames)

    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        rms[i] = np.sqrt(np.mean(audio[start:end] ** 2))

    return rms


def compute_spectral_centroid(audio: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    """Compute spectral centroid (brightness indicator)."""
    # Compute magnitude spectrum
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)

    # Weighted mean frequency
    centroid = np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)
    return float(centroid)


def compute_spectral_tilt(audio: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    """Compute spectral tilt (slope of spectrum)."""
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, 1/sr)

    # Log-scale for perceptual relevance
    log_freqs = np.log10(freqs[1:] + 1)  # Skip DC
    log_spectrum = np.log10(spectrum[1:] + 1e-10)

    # Linear regression slope
    slope, _ = np.polyfit(log_freqs, log_spectrum, 1)
    return float(slope)


def find_transients(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.3,
) -> np.ndarray:
    """Find transient locations in audio."""
    # Compute onset strength
    hop_length = 512
    n_fft = 2048

    # Simple onset detection via spectral flux
    n_frames = 1 + (len(audio) - n_fft) // hop_length
    onset_strength = np.zeros(n_frames)

    prev_spectrum = None
    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft
        frame = audio[start:end] * np.hanning(n_fft)
        spectrum = np.abs(np.fft.rfft(frame))

        if prev_spectrum is not None:
            # Spectral flux (only positive changes)
            flux = np.sum(np.maximum(spectrum - prev_spectrum, 0))
            onset_strength[i] = flux

        prev_spectrum = spectrum

    # Normalize
    if np.max(onset_strength) > 0:
        onset_strength = onset_strength / np.max(onset_strength)

    # Find peaks above threshold
    peaks = []
    for i in range(1, len(onset_strength) - 1):
        if (onset_strength[i] > onset_strength[i-1] and
            onset_strength[i] > onset_strength[i+1] and
            onset_strength[i] > threshold):
            peaks.append(i * hop_length)

    return np.array(peaks)


