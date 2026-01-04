"""Isolation backend adapter pattern.

This module provides a clean interface for guitar isolation backends.
The current implementation includes:
- MockSAMBackend: Deterministic mock for testing
- SAMAudioBackend: Real SAM Audio integration using facebook/sam-audio-large

SAM-Audio Reference:
    https://huggingface.co/facebook/sam-audio-large
    Paper: https://arxiv.org/abs/2512.18099

To add a new backend:
1. Implement the IsolationBackend protocol
2. Add to get_isolation_backend() factory
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Literal, Optional
import tempfile
import os

import structlog

logger = structlog.get_logger()

# Guitar-related text prompts for SAM-Audio
GUITAR_PROMPTS = [
    "Electric guitar playing",
    "Distorted electric guitar",
    "Clean electric guitar",
    "Guitar",
    "Electric guitar",
]

# Chunking settings for long audio
DEFAULT_CHUNK_DURATION = 30  # seconds per chunk
OVERLAP_DURATION = 2  # seconds of overlap between chunks
MAX_DURATION_WITHOUT_CHUNKING = 60  # auto-chunk if longer than this


class IsolationBackend(ABC):
    """Abstract interface for audio isolation backends."""

    @abstractmethod
    def isolate(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> dict:
        """
        Isolate guitar from audio.

        Args:
            input_path: Path to input audio file
            output_path: Path to save isolated guitar
            progress_callback: Optional callback for progress updates (0.0-1.0)

        Returns:
            dict with keys:
                - confidence: float (0-1) indicating isolation quality
                - duration: float, duration of isolated audio in seconds
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass


class MockSAMBackend(IsolationBackend):
    """Mock isolation backend for testing.

    Produces deterministic output by applying a simple filter
    to simulate guitar isolation. This allows end-to-end testing
    without the real SAM Audio model.
    """

    @property
    def name(self) -> str:
        return "mock_sam"

    def isolate(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> dict:
        """Simulate guitar isolation with deterministic processing."""
        import time
        import numpy as np
        import soundfile as sf
        from scipy import signal

        logger.info("Mock isolation starting", input=str(input_path))

        # Load audio
        if progress_callback:
            progress_callback(0.1)

        audio, sr = sf.read(input_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)  # Convert to mono

        # Simulate processing time
        time.sleep(0.5)
        if progress_callback:
            progress_callback(0.3)

        # Apply "guitar-like" filtering (mock isolation)
        # This simulates extracting guitar frequencies
        nyquist = sr / 2

        # Bandpass filter to emphasize guitar range (80Hz - 5kHz)
        low = 80 / nyquist
        high = 5000 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        isolated = signal.filtfilt(b, a, audio)

        if progress_callback:
            progress_callback(0.6)

        # Add slight emphasis on midrange (presence)
        mid_low = 800 / nyquist
        mid_high = 3000 / nyquist
        b_mid, a_mid = signal.butter(2, [mid_low, mid_high], btype='band')
        midrange = signal.filtfilt(b_mid, a_mid, audio) * 0.3
        isolated = isolated + midrange

        if progress_callback:
            progress_callback(0.8)

        # Normalize
        max_val = np.max(np.abs(isolated))
        if max_val > 0:
            isolated = isolated / max_val * 0.9

        # Save output
        sf.write(output_path, isolated, sr)

        if progress_callback:
            progress_callback(1.0)

        duration = len(isolated) / sr

        # Mock confidence based on signal characteristics
        # In a real system, this would be model confidence
        rms = np.sqrt(np.mean(isolated ** 2))
        confidence = min(0.95, 0.7 + rms * 0.5)

        logger.info(
            "Mock isolation complete",
            output=str(output_path),
            duration=duration,
            confidence=confidence,
        )

        return {
            "confidence": confidence,
            "duration": duration,
        }


class SAMAudioBackend(IsolationBackend):
    """Real SAM Audio backend using facebook/sam-audio-large.

    SAM-Audio is a foundation model for isolating any sound in audio
    using text, visual, or temporal prompts. For guitar isolation,
    we use text prompting with guitar-related descriptions.

    Features:
    - Automatic chunking for long audio (>60s)
    - Crossfade merging for seamless chunk boundaries
    - Configurable re-ranking for quality vs speed tradeoff

    Requirements:
        pip install git+https://github.com/facebookresearch/sam-audio.git
        pip install torch torchaudio
        huggingface-cli login  # Must have access to facebook/sam-audio-large

    Reference:
        https://huggingface.co/facebook/sam-audio-large
        Paper: https://arxiv.org/abs/2512.18099
    """

    def __init__(
        self,
        model_name: str = "facebook/sam-audio-large",
        device: Optional[str] = None,
        prompt: str = "Electric guitar playing",
        use_reranking: bool = True,
        reranking_candidates: int = 1,
        chunk_duration: int = DEFAULT_CHUNK_DURATION,
    ):
        """
        Initialize SAM Audio backend.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (None for auto-detect)
            prompt: Text description for guitar isolation
            use_reranking: Whether to use candidate re-ranking for better quality
            reranking_candidates: Number of candidates for re-ranking (1 = fast, 4 = quality)
            chunk_duration: Duration of each chunk in seconds for long audio
        """
        self._model = None
        self._processor = None
        self._model_name = model_name
        self._prompt = prompt
        self._use_reranking = use_reranking
        self._reranking_candidates = reranking_candidates
        self._chunk_duration = chunk_duration

        # Auto-detect device
        if device is None:
            import torch
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            import torch
            self._device = torch.device(device)

        logger.info(
            "SAMAudioBackend initialized",
            model=model_name,
            device=str(self._device),
            prompt=prompt,
            chunk_duration=chunk_duration,
        )

    @property
    def name(self) -> str:
        return "sam_audio"

    def _ensure_model(self) -> None:
        """Load model and processor if not already loaded."""
        if self._model is not None:
            return

        try:
            import torch
            from sam_audio import SAMAudio, SAMAudioProcessor

            logger.info("Loading SAM Audio model...", model=self._model_name)

            self._processor = SAMAudioProcessor.from_pretrained(self._model_name)
            self._model = SAMAudio.from_pretrained(self._model_name)
            self._model = self._model.to(self._device).eval()

            logger.info(
                "SAM Audio model loaded",
                model=self._model_name,
                device=str(self._device),
            )

        except ImportError as e:
            raise ImportError(
                "SAM Audio package not installed. Install with:\n"
                "  pip install git+https://github.com/facebookresearch/sam-audio.git\n"
                "  pip install torch torchaudio\n"
                "And authenticate with HuggingFace:\n"
                "  huggingface-cli login\n"
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SAM Audio model '{self._model_name}'. "
                f"Ensure you have access to the model on HuggingFace.\n"
                f"Original error: {e}"
            ) from e

    def _load_audio(self, file_path: Path):
        """Load audio from file."""
        import torchaudio
        
        waveform, sample_rate = torchaudio.load(str(file_path))
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform, sample_rate

    def _split_audio_into_chunks(self, waveform, sample_rate: int):
        """Split audio waveform into overlapping chunks."""
        import torch
        
        chunk_samples = int(self._chunk_duration * sample_rate)
        overlap_samples = int(OVERLAP_DURATION * sample_rate)
        stride = chunk_samples - overlap_samples

        chunks = []
        total_samples = waveform.shape[1]

        start = 0
        while start < total_samples:
            end = min(start + chunk_samples, total_samples)
            chunk = waveform[:, start:end]
            chunks.append(chunk)

            if end >= total_samples:
                break
            start += stride

        return chunks

    def _merge_chunks_with_crossfade(self, chunks: list, sample_rate: int):
        """Merge audio chunks with crossfade on overlapping regions."""
        import torch
        
        if len(chunks) == 1:
            chunk = chunks[0]
            # Ensure 2D tensor
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)
            return chunk

        overlap_samples = int(OVERLAP_DURATION * sample_rate)

        # Ensure all chunks are 2D [channels, samples]
        processed_chunks = []
        for chunk in chunks:
            if chunk.dim() == 1:
                chunk = chunk.unsqueeze(0)
            processed_chunks.append(chunk)

        result = processed_chunks[0]

        for i in range(1, len(processed_chunks)):
            prev_chunk = result
            next_chunk = processed_chunks[i]

            # Handle case where chunks are shorter than overlap
            actual_overlap = min(overlap_samples, prev_chunk.shape[1], next_chunk.shape[1])

            if actual_overlap <= 0:
                # No overlap possible, just concatenate
                result = torch.cat([prev_chunk, next_chunk], dim=1)
                continue

            # Create fade curves
            fade_out = torch.linspace(1.0, 0.0, actual_overlap).to(prev_chunk.device)
            fade_in = torch.linspace(0.0, 1.0, actual_overlap).to(next_chunk.device)

            # Get overlapping regions
            prev_overlap = prev_chunk[:, -actual_overlap:]
            next_overlap = next_chunk[:, :actual_overlap]

            # Crossfade mix
            crossfaded = prev_overlap * fade_out + next_overlap * fade_in

            # Concatenate: non-overlap of prev + crossfaded + non-overlap of next
            result = torch.cat([
                prev_chunk[:, :-actual_overlap],
                crossfaded,
                next_chunk[:, actual_overlap:]
            ], dim=1)

        return result

    def _process_single(self, file_path: str, prompt: str):
        """Process a single audio file (or chunk) through SAM Audio."""
        import torch
        
        inputs = self._processor(
            audios=[file_path],
            descriptions=[prompt],
        ).to(self._device)

        with torch.inference_mode():
            result = self._model.separate(
                inputs,
                predict_spans=False,  # Don't predict time spans
                reranking_candidates=self._reranking_candidates if self._use_reranking else 1,
            )

        return result

    def isolate(
        self,
        input_path: Path,
        output_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> dict:
        """
        Isolate guitar from audio using SAM Audio text prompting.

        Automatically handles long audio files by chunking and crossfade merging.
        """
        import torch
        import torchaudio
        import numpy as np
        import soundfile as sf

        logger.info(
            "SAM Audio isolation starting",
            input=str(input_path),
            prompt=self._prompt,
        )

        if progress_callback:
            progress_callback(0.05)

        # Load model
        self._ensure_model()

        if progress_callback:
            progress_callback(0.1)

        # Load audio to check duration
        waveform, sample_rate = self._load_audio(input_path)
        duration = waveform.shape[1] / sample_rate

        logger.info(
            "Audio loaded",
            duration=f"{duration:.1f}s",
            sample_rate=sample_rate,
        )

        # Decide whether to use chunking
        use_chunking = duration > MAX_DURATION_WITHOUT_CHUNKING

        if use_chunking:
            # Process with chunking
            result_data = self._isolate_chunked(
                waveform, sample_rate, input_path, progress_callback
            )
        else:
            # Process without chunking
            result_data = self._isolate_single(
                input_path, progress_callback
            )

        target_audio = result_data["target"]
        residual_audio = result_data["residual"]
        output_sample_rate = result_data["sample_rate"]

        # Normalize output
        max_val = np.max(np.abs(target_audio))
        if max_val > 0:
            target_audio = target_audio / max_val * 0.95

        # Save isolated audio
        sf.write(output_path, target_audio, output_sample_rate)

        if progress_callback:
            progress_callback(0.95)

        # Calculate confidence based on separation quality
        target_energy = np.sqrt(np.mean(target_audio ** 2))
        residual_energy = np.sqrt(np.mean(residual_audio ** 2))

        if residual_energy > 0:
            separation_ratio = target_energy / (target_energy + residual_energy)
        else:
            separation_ratio = 1.0

        confidence = min(0.98, 0.5 + separation_ratio * 0.48)
        final_duration = len(target_audio) / output_sample_rate

        if progress_callback:
            progress_callback(1.0)

        logger.info(
            "SAM Audio isolation complete",
            output=str(output_path),
            duration=final_duration,
            confidence=confidence,
            sample_rate=output_sample_rate,
            chunked=use_chunking,
        )

        return {
            "confidence": confidence,
            "duration": final_duration,
            "sample_rate": output_sample_rate,
            "prompt_used": self._prompt,
            "chunked": use_chunking,
        }

    def _isolate_single(
        self,
        input_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> dict:
        """Process audio without chunking (for short files)."""
        import numpy as np

        if progress_callback:
            progress_callback(0.3)

        logger.debug("Processing without chunking")
        result = self._process_single(str(input_path), self._prompt)

        if progress_callback:
            progress_callback(0.8)

        # Extract audio - handle both 1D and 2D tensors
        target = result.target[0].cpu()
        residual = result.residual[0].cpu()
        
        if target.dim() == 1:
            target_audio = target.numpy()
        else:
            target_audio = target.squeeze(0).numpy()
            
        if residual.dim() == 1:
            residual_audio = residual.numpy()
        else:
            residual_audio = residual.squeeze(0).numpy()

        sample_rate = self._processor.audio_sampling_rate

        return {
            "target": target_audio,
            "residual": residual_audio,
            "sample_rate": sample_rate,
        }

    def _isolate_chunked(
        self,
        waveform,
        sample_rate: int,
        input_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> dict:
        """Process long audio with chunking and crossfade merging."""
        import torch
        import torchaudio
        import numpy as np

        duration = waveform.shape[1] / sample_rate
        logger.info(
            f"Audio is {duration:.1f}s, splitting into chunks",
            chunk_duration=self._chunk_duration,
        )

        chunks = self._split_audio_into_chunks(waveform, sample_rate)
        num_chunks = len(chunks)

        logger.info(f"Processing {num_chunks} chunks")

        target_chunks = []
        residual_chunks = []

        for i, chunk in enumerate(chunks):
            chunk_progress = 0.15 + (i / num_chunks) * 0.65
            if progress_callback:
                progress_callback(chunk_progress)

            logger.debug(f"Processing chunk {i+1}/{num_chunks}")

            # Save chunk to temp file for processor
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                torchaudio.save(tmp.name, chunk, sample_rate)
                chunk_path = tmp.name

            try:
                result = self._process_single(chunk_path, self._prompt)
                
                target = result.target[0].cpu()
                residual = result.residual[0].cpu()
                
                # Ensure proper shape
                if target.dim() == 1:
                    target = target.unsqueeze(0)
                if residual.dim() == 1:
                    residual = residual.unsqueeze(0)
                    
                target_chunks.append(target)
                residual_chunks.append(residual)
            finally:
                os.unlink(chunk_path)

        if progress_callback:
            progress_callback(0.85)

        logger.debug("Merging chunks with crossfade")
        
        # Use the processor's sample rate for merging
        output_sample_rate = self._processor.audio_sampling_rate
        
        target_merged = self._merge_chunks_with_crossfade(target_chunks, output_sample_rate)
        residual_merged = self._merge_chunks_with_crossfade(residual_chunks, output_sample_rate)

        # Convert to numpy
        target_audio = target_merged.squeeze(0).numpy()
        residual_audio = residual_merged.squeeze(0).numpy()

        return {
            "target": target_audio,
            "residual": residual_audio,
            "sample_rate": output_sample_rate,
        }

    def isolate_with_multiple_prompts(
        self,
        input_path: Path,
        output_path: Path,
        prompts: Optional[list[str]] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> dict:
        """
        Try multiple prompts and select the best result.

        This can improve isolation quality by finding the prompt
        that best matches the guitar sound in the recording.

        Args:
            input_path: Path to input audio
            output_path: Path for output
            prompts: List of prompts to try (defaults to GUITAR_PROMPTS)
            progress_callback: Progress callback

        Returns:
            dict with isolation results including which prompt worked best
        """
        import torch
        import numpy as np
        import soundfile as sf

        prompts = prompts or GUITAR_PROMPTS
        self._ensure_model()

        if progress_callback:
            progress_callback(0.1)

        best_result = None
        best_energy = 0
        best_prompt = prompts[0]

        progress_per_prompt = 0.7 / len(prompts)

        for i, prompt in enumerate(prompts):
            logger.debug("Trying prompt", prompt=prompt)

            result = self._process_single(str(input_path), prompt)
            
            target = result.target[0].cpu()
            if target.dim() == 1:
                target_np = target.numpy()
            else:
                target_np = target.squeeze(0).numpy()
                
            energy = np.sqrt(np.mean(target_np ** 2))

            if energy > best_energy:
                best_energy = energy
                best_result = result
                best_prompt = prompt

            if progress_callback:
                progress_callback(0.1 + (i + 1) * progress_per_prompt)

        # Use best result
        target = best_result.target[0].cpu()
        residual = best_result.residual[0].cpu()
        
        if target.dim() == 1:
            target_audio = target.numpy()
        else:
            target_audio = target.squeeze(0).numpy()
            
        if residual.dim() == 1:
            residual_audio = residual.numpy()
        else:
            residual_audio = residual.squeeze(0).numpy()
            
        sample_rate = self._processor.audio_sampling_rate

        # Normalize and save
        max_val = np.max(np.abs(target_audio))
        if max_val > 0:
            target_audio = target_audio / max_val * 0.95

        sf.write(output_path, target_audio, sample_rate)

        if progress_callback:
            progress_callback(0.95)

        # Calculate confidence
        target_energy = np.sqrt(np.mean(target_audio ** 2))
        residual_energy = np.sqrt(np.mean(residual_audio ** 2))
        separation_ratio = target_energy / (target_energy + residual_energy + 1e-10)
        confidence = min(0.98, 0.5 + separation_ratio * 0.48)

        duration = len(target_audio) / sample_rate

        if progress_callback:
            progress_callback(1.0)

        logger.info(
            "SAM Audio multi-prompt isolation complete",
            output=str(output_path),
            best_prompt=best_prompt,
            confidence=confidence,
        )

        return {
            "confidence": confidence,
            "duration": duration,
            "sample_rate": sample_rate,
            "prompt_used": best_prompt,
            "prompts_tried": prompts,
        }


def get_isolation_backend(
    backend_type: Literal["sam_audio", "mock"] = "mock"
) -> IsolationBackend:
    """Factory function to get isolation backend.

    Args:
        backend_type: Which backend to use

    Returns:
        IsolationBackend instance
    """
    from app.config import settings

    logger.info("Creating isolation backend", type=backend_type)

    if backend_type == "mock":
        return MockSAMBackend()

    elif backend_type == "sam_audio":
        return SAMAudioBackend(
            model_name=settings.sam_audio_model,
            device=settings.sam_audio_device,
            prompt=settings.sam_audio_prompt,
            use_reranking=settings.sam_audio_use_reranking,
            reranking_candidates=settings.sam_audio_reranking_candidates,
        )

    else:
        raise ValueError(f"Unknown backend: {backend_type}. Available: mock, sam_audio")
