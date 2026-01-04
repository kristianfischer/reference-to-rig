"""NAM (Neural Amp Modeler) backend adapter.

This module provides an interface for NAM model processing.
Includes a mock backend for testing without real NAM models.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import structlog
from scipy import signal

from app.config import settings

logger = structlog.get_logger()


class NAMBackend(ABC):
    """Abstract interface for NAM processing backends."""

    @abstractmethod
    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model_id: str,
    ) -> np.ndarray:
        """
        Process audio through NAM model.

        Args:
            audio: Input audio (mono, float32)
            sample_rate: Audio sample rate
            model_id: NAM model identifier

        Returns:
            Processed audio
        """
        pass

    @abstractmethod
    def load_model(self, model_id: str) -> bool:
        """Load a NAM model by ID. Returns success status."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass


class MockNAMBackend(NAMBackend):
    """Mock NAM backend for testing.

    Simulates amp modeling by applying waveshaping and filtering
    that mimics tube amp characteristics.
    """

    def __init__(self):
        self._loaded_models: set[str] = set()

    @property
    def name(self) -> str:
        return "mock_nam"

    def load_model(self, model_id: str) -> bool:
        """Mock model loading."""
        self._loaded_models.add(model_id)
        logger.debug("Mock NAM model loaded", model_id=model_id)
        return True

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model_id: str,
    ) -> np.ndarray:
        """Process audio through mock amp simulation."""
        logger.debug("Processing through mock NAM", model_id=model_id)

        # Determine amp characteristics based on model_id
        if "clean" in model_id.lower() or "fender" in model_id.lower():
            return self._process_clean(audio, sample_rate)
        elif "high" in model_id.lower() or "mesa" in model_id.lower() or "rectifier" in model_id.lower():
            return self._process_high_gain(audio, sample_rate)
        elif "crunch" in model_id.lower() or "marshall" in model_id.lower():
            return self._process_crunch(audio, sample_rate)
        else:
            return self._process_crunch(audio, sample_rate)  # Default

    def _process_clean(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Clean amp simulation."""
        # Slight compression
        threshold = 0.6
        ratio = 2.0
        output = np.where(
            np.abs(audio) > threshold,
            np.sign(audio) * (threshold + (np.abs(audio) - threshold) / ratio),
            audio
        )

        # Gentle tube warmth (subtle even harmonics)
        output = output + 0.05 * output ** 2

        # Clean tone EQ: slight presence boost
        nyquist = sr / 2
        b, a = signal.butter(2, [3000 / nyquist, 5000 / nyquist], btype='band')
        presence = signal.filtfilt(b, a, output) * 0.2
        output = output + presence

        return output

    def _process_crunch(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Crunch/overdrive amp simulation."""
        # Input boost
        audio = audio * 2.0

        # Soft clipping (tube-like)
        output = np.tanh(audio * 1.5)

        # Add harmonics
        output = output + 0.1 * np.tanh(audio * 3) ** 2

        # Marshall-style mid boost
        nyquist = sr / 2
        b, a = signal.butter(2, [600 / nyquist, 2500 / nyquist], btype='band')
        mids = signal.filtfilt(b, a, output) * 0.3
        output = output + mids

        # Presence
        b, a = signal.butter(2, 3500 / nyquist, btype='high')
        presence = signal.filtfilt(b, a, output) * 0.15
        output = output + presence

        return output * 0.7  # Output level

    def _process_high_gain(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """High gain amp simulation."""
        # High input gain
        audio = audio * 4.0

        # Multiple stages of clipping
        stage1 = np.tanh(audio * 2)
        stage2 = np.tanh(stage1 * 1.5)
        output = np.tanh(stage2 * 1.2)

        # Tight low end
        nyquist = sr / 2
        b, a = signal.butter(4, 100 / nyquist, btype='high')
        output = signal.filtfilt(b, a, output)

        # Scoop mids slightly
        b, a = signal.butter(2, [400 / nyquist, 800 / nyquist], btype='band')
        mids = signal.filtfilt(b, a, output) * -0.2
        output = output + mids

        # Aggressive presence
        b, a = signal.butter(2, 2500 / nyquist, btype='high')
        presence = signal.filtfilt(b, a, output) * 0.25
        output = output + presence

        return output * 0.6


class RealNAMBackend(NAMBackend):
    """Real NAM backend using neural-amp-modeler library.

    Loads .nam model files and processes audio through neural amp models.

    Reference:
        https://github.com/sdatkinson/neural-amp-modeler
        pip install neural-amp-modeler
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize NAM backend.

        Args:
            device: Device to run on (None for auto-detect, 'cpu' for CPU-only)
        """
        import torch

        self._models: dict[str, object] = {}
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("RealNAMBackend initialized", device=self._device)

    @property
    def name(self) -> str:
        return "nam"

    def _find_model_path(self, model_id: str) -> Optional[Path]:
        """Find model file path from ID using metadata.json."""
        import json
        
        metadata_path = settings.capture_library_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
            
            for model in data.get("nam_models", []):
                if model.get("id") == model_id:
                    return settings.capture_library_dir / model.get("file_path", "")
        
        # Fallback: try direct path
        direct_path = settings.capture_library_dir / "nam_models" / f"{model_id}.nam"
        if direct_path.exists():
            return direct_path
        
        # Fallback: search for partial match
        nam_dir = settings.capture_library_dir / "nam_models"
        for nam_file in nam_dir.glob("*.nam"):
            if model_id in nam_file.stem.lower().replace(" ", "_").replace("-", "_"):
                return nam_file
        
        return None

    def load_model(self, model_id: str) -> bool:
        """Load a NAM model from .nam file.

        Args:
            model_id: Model identifier (from metadata.json)

        Returns:
            True if loaded successfully
        """
        import json
        import torch
        from nam.models import init_from_nam

        if model_id in self._models:
            return True

        # Look up file path from metadata
        model_path = self._find_model_path(model_id)
        if model_path is None or not model_path.exists():
            logger.error("NAM model not found", model_id=model_id, path=str(model_path))
            return False

        try:
            with open(model_path, "r") as f:
                config = json.load(f)

            model = init_from_nam(config)
            model = model.to(self._device).eval()
            self._models[model_id] = model

            # Some models don't specify sample_rate - default to 48kHz
            sr = getattr(model, 'sample_rate', None) or 48000
            rf = getattr(model, 'receptive_field', None) or 0
            
            logger.info(
                "NAM model loaded",
                model_id=model_id,
                sample_rate=sr,
                receptive_field=rf,
            )
            return True

        except Exception as e:
            logger.error("Failed to load NAM model", model_id=model_id, error=str(e))
            return False

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        model_id: str,
    ) -> np.ndarray:
        """Process audio through NAM model.

        Args:
            audio: Input audio (mono, float32, -1 to 1 range)
            sample_rate: Audio sample rate
            model_id: NAM model identifier

        Returns:
            Processed audio
        """
        import torch
        from scipy import signal as scipy_signal

        # Ensure model is loaded
        if model_id not in self._models:
            if not self.load_model(model_id):
                logger.warning("Using mock processing, model not found", model_id=model_id)
                return audio

        model = self._models[model_id]

        # Resample if needed (NAM models typically expect 48kHz)
        # Default to 48000 if model doesn't specify sample_rate
        target_sr = getattr(model, 'sample_rate', None) or 48000
        if target_sr and sample_rate != target_sr:
            logger.debug("Resampling for NAM", from_sr=sample_rate, to_sr=target_sr)
            num_samples = int(len(audio) * target_sr / sample_rate)
            audio = scipy_signal.resample(audio, num_samples)

        # Convert to torch tensor
        # NAM expects shape (batch, samples)
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self._device)

        # Process through model
        with torch.no_grad():
            output = model(audio_tensor)

        # Convert back to numpy
        output_audio = output.squeeze(0).cpu().numpy()

        # Resample back if we changed sample rate
        if target_sr and sample_rate != target_sr:
            num_samples = int(len(output_audio) * sample_rate / target_sr)
            output_audio = scipy_signal.resample(output_audio, num_samples)

        return output_audio

    def get_model_info(self, model_id: str) -> Optional[dict]:
        """Get information about a loaded model.

        Returns:
            dict with sample_rate, receptive_field, etc. or None if not loaded
        """
        if model_id not in self._models:
            if not self.load_model(model_id):
                return None

        model = self._models[model_id]
        return {
            "sample_rate": getattr(model, 'sample_rate', None) or 48000,
            "receptive_field": getattr(model, 'receptive_field', None) or 0,
        }


def get_nam_backend(
    backend_type: Literal["nam", "mock"] = "mock",
    device: Optional[str] = None,
) -> NAMBackend:
    """Factory function to get NAM backend.

    Args:
        backend_type: Which backend to use
        device: Device for real NAM backend (None=auto, 'cpu', 'cuda')

    Returns:
        NAMBackend instance
    """
    logger.info("Creating NAM backend", type=backend_type, device=device)

    if backend_type == "mock":
        return MockNAMBackend()
    elif backend_type == "nam":
        return RealNAMBackend(device=device)
    else:
        raise ValueError(f"Unknown backend: {backend_type}. Available: mock, nam")

