"""Audio rendering with NAM models, IRs, and EQ.

This module renders synthesized audio by:
1. Applying NAM model simulation (or mock)
2. Convolving with cabinet IR
3. Applying parametric EQ
4. Loudness matching to reference
"""

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import structlog
from scipy import signal

from app.config import settings
from app.models import EQSettings
from app.rendering.nam_adapter import get_nam_backend

logger = structlog.get_logger()


class AudioRenderer:
    """Render synthesized audio from NAM + IR + EQ chain."""

    def __init__(self):
        self.nam_backend = get_nam_backend(
            settings.nam_backend,
            device=settings.nam_device,
        )
        self.sample_rate = settings.sample_rate

    def render(
        self,
        source_path: Path,
        output_path: Path,
        nam_model_id: str,
        ir_id: str,
        input_gain_db: float,
        eq_settings: EQSettings,
        target_loudness: Optional[float] = None,
    ) -> dict:
        """
        Render synthesized audio through the signal chain.

        Args:
            source_path: Input audio (typically isolated guitar)
            output_path: Output path for rendered audio
            nam_model_id: NAM model identifier
            ir_id: Cabinet IR identifier
            input_gain_db: Input gain adjustment
            eq_settings: EQ parameters to apply
            target_loudness: Target loudness in LUFS (default from settings)

        Returns:
            dict with render info
        """
        logger.info(
            "Rendering audio",
            source=str(source_path),
            nam=nam_model_id,
            ir=ir_id,
        )

        # Load source audio
        audio, sr = sf.read(source_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Resample if needed
        if sr != self.sample_rate:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, self.sample_rate)
            audio = resample_poly(audio, self.sample_rate // g, sr // g)
            sr = self.sample_rate

        # Step 1: Apply input gain
        gain_linear = 10 ** (input_gain_db / 20)
        audio = audio * gain_linear

        # Step 2: Process through NAM model
        audio = self.nam_backend.process(audio, sr, nam_model_id)

        # Step 3: Convolve with cabinet IR
        audio = self._apply_ir(audio, sr, ir_id)

        # Step 4: Apply EQ
        audio = self._apply_eq(audio, sr, eq_settings)

        # Step 5: Loudness match
        target = target_loudness or settings.target_loudness_lufs
        audio = self._loudness_normalize(audio, sr, target)

        # Step 6: Prevent clipping
        max_val = np.max(np.abs(audio))
        if max_val > 0.99:
            audio = audio / max_val * 0.99

        # Save output
        sf.write(output_path, audio, sr)

        duration = len(audio) / sr
        logger.info("Render complete", output=str(output_path), duration=duration)

        return {
            "output_path": str(output_path),
            "duration": duration,
            "sample_rate": sr,
        }

    def _apply_ir(self, audio: np.ndarray, sr: int, ir_id: str) -> np.ndarray:
        """Convolve audio with cabinet IR."""
        # Load IR (or use mock)
        ir = self._load_ir(ir_id, sr)

        # Convolve
        output = signal.fftconvolve(audio, ir, mode='same')

        return output

    def _load_ir(self, ir_id: str, target_sr: int) -> np.ndarray:
        """Load cabinet IR from file or generate mock."""
        # Try to load real IR
        ir_path = settings.capture_library_dir / "cab_irs"

        # Search for IR file
        for ext in [".wav", ".aif", ".aiff"]:
            candidates = list(ir_path.glob(f"*{ir_id}*{ext}"))
            if candidates:
                ir, ir_sr = sf.read(candidates[0])
                if len(ir.shape) > 1:
                    ir = np.mean(ir, axis=1)

                # Resample if needed
                if ir_sr != target_sr:
                    from scipy.signal import resample_poly
                    from math import gcd
                    g = gcd(ir_sr, target_sr)
                    ir = resample_poly(ir, target_sr // g, ir_sr // g)

                return ir

        # Generate mock IR
        logger.debug("Using mock IR", ir_id=ir_id)
        return self._generate_mock_ir(ir_id, target_sr)

    def _generate_mock_ir(self, ir_id: str, sr: int) -> np.ndarray:
        """Generate a mock cabinet IR for testing."""
        # Create a simple impulse response that simulates cabinet characteristics
        ir_length = int(0.1 * sr)  # 100ms IR
        ir = np.zeros(ir_length)

        # Initial impulse
        ir[0] = 1.0

        # Early reflections
        for i in range(1, 10):
            pos = int(i * sr * 0.001)  # Every 1ms
            if pos < ir_length:
                ir[pos] = 0.5 ** i

        # Exponential decay
        decay = np.exp(-np.arange(ir_length) / (sr * 0.03))
        noise = np.random.randn(ir_length) * 0.1
        ir = ir + noise * decay

        # Apply cabinet-like filtering based on ir_id
        nyquist = sr / 2

        # Low-pass to simulate speaker response
        if "bright" in ir_id.lower() or "jensen" in ir_id.lower():
            cutoff = 8000 / nyquist
        elif "dark" in ir_id.lower():
            cutoff = 4000 / nyquist
        else:
            cutoff = 6000 / nyquist

        b, a = signal.butter(4, cutoff, btype='low')
        ir = signal.filtfilt(b, a, ir)

        # High-pass to remove sub frequencies
        hp_cutoff = 60 / nyquist
        b, a = signal.butter(2, hp_cutoff, btype='high')
        ir = signal.filtfilt(b, a, ir)

        # Normalize
        ir = ir / np.max(np.abs(ir))

        return ir

    def _apply_eq(self, audio: np.ndarray, sr: int, eq_settings: EQSettings) -> np.ndarray:
        """Apply parametric EQ to audio."""
        nyquist = sr / 2

        # Apply highpass if set
        if eq_settings.highpass_freq and eq_settings.highpass_freq > 20:
            hp_freq = min(eq_settings.highpass_freq, nyquist * 0.9)
            b, a = signal.butter(2, hp_freq / nyquist, btype='high')
            audio = signal.filtfilt(b, a, audio)

        # Apply lowpass if set
        if eq_settings.lowpass_freq and eq_settings.lowpass_freq < nyquist:
            lp_freq = min(eq_settings.lowpass_freq, nyquist * 0.9)
            b, a = signal.butter(2, lp_freq / nyquist, btype='low')
            audio = signal.filtfilt(b, a, audio)

        # Apply parametric bands
        for band in eq_settings.bands:
            if abs(band.gain_db) < 0.1:
                continue  # Skip negligible adjustments

            freq = min(band.frequency, nyquist * 0.9)
            gain_linear = 10 ** (band.gain_db / 20)
            q = band.q

            if band.band_type == "peak":
                audio = self._apply_peak_eq(audio, sr, freq, gain_linear, q)
            elif band.band_type == "lowshelf":
                audio = self._apply_shelf_eq(audio, sr, freq, gain_linear, "low")
            elif band.band_type == "highshelf":
                audio = self._apply_shelf_eq(audio, sr, freq, gain_linear, "high")

        return audio

    def _apply_peak_eq(
        self,
        audio: np.ndarray,
        sr: int,
        freq: float,
        gain: float,
        q: float,
    ) -> np.ndarray:
        """Apply peak EQ filter."""
        # Design biquad peak filter
        w0 = 2 * np.pi * freq / sr
        A = np.sqrt(gain)
        alpha = np.sin(w0) / (2 * q)

        b0 = 1 + alpha * A
        b1 = -2 * np.cos(w0)
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * np.cos(w0)
        a2 = 1 - alpha / A

        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1, a1/a0, a2/a0])

        return signal.filtfilt(b, a, audio)

    def _apply_shelf_eq(
        self,
        audio: np.ndarray,
        sr: int,
        freq: float,
        gain: float,
        shelf_type: str,
    ) -> np.ndarray:
        """Apply shelf EQ filter."""
        w0 = 2 * np.pi * freq / sr
        A = np.sqrt(gain)
        S = 1.0  # Shelf slope
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/S - 1) + 2)

        if shelf_type == "low":
            b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
            a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
        else:  # high shelf
            b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
            b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
            b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
            a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
            a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
            a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        b = np.array([b0/a0, b1/a0, b2/a0])
        a = np.array([1, a1/a0, a2/a0])

        return signal.filtfilt(b, a, audio)

    def _loudness_normalize(
        self,
        audio: np.ndarray,
        sr: int,
        target_lufs: float,
    ) -> np.ndarray:
        """Normalize audio to target loudness."""
        import pyloudnorm as pyln

        meter = pyln.Meter(sr)
        current_loudness = meter.integrated_loudness(audio)

        if np.isinf(current_loudness):
            return audio

        gain_db = target_lufs - current_loudness
        gain_linear = 10 ** (gain_db / 20)

        return audio * gain_linear

