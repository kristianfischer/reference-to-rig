"""Guitar isolation background task."""

import asyncio
from pathlib import Path
from typing import Optional
from uuid import UUID

import numpy as np
import soundfile as sf
import structlog
from scipy import signal

from app.config import settings
from app.isolation.adapter import get_isolation_backend, SAMAudioBackend
from app.features.audio_utils import preprocess_audio
from app.models import ProjectStatus
from app.storage.project_manager import ProjectManager
from app.tasks.queue import Task

logger = structlog.get_logger()


def apply_trim(audio: np.ndarray, sample_rate: int, start: float, end: float) -> np.ndarray:
    """Trim audio to specified start/end times."""
    start_sample = int(start * sample_rate)
    end_sample = int(end * sample_rate)
    
    # Clamp to valid range
    start_sample = max(0, start_sample)
    end_sample = min(len(audio), end_sample)
    
    return audio[start_sample:end_sample]


def apply_pan(audio: np.ndarray, pan: float) -> np.ndarray:
    """
    Apply panning to extract a specific stereo position.
    
    pan: -100 (full left) to +100 (full right), 0 = center
    
    For isolating a panned instrument:
    - If guitar is panned right (pan > 0), we emphasize the right channel
    - If guitar is panned left (pan < 0), we emphasize the left channel
    """
    if len(audio.shape) == 1:
        # Mono audio, no panning to apply
        return audio
    
    if audio.shape[1] != 2:
        # Not stereo, return as-is
        return audio[:, 0] if audio.shape[1] > 1 else audio
    
    # Normalize pan to -1 to +1
    pan_norm = pan / 100.0
    
    left = audio[:, 0]
    right = audio[:, 1]
    
    if pan_norm == 0:
        # Center - sum to mono
        return (left + right) / 2
    elif pan_norm > 0:
        # Panned right - emphasize right channel
        # Use mid-side processing: extract side (L-R) and mix
        right_weight = 0.5 + pan_norm * 0.5
        left_weight = 0.5 - pan_norm * 0.5
        return left * left_weight + right * right_weight
    else:
        # Panned left - emphasize left channel
        pan_abs = abs(pan_norm)
        left_weight = 0.5 + pan_abs * 0.5
        right_weight = 0.5 - pan_abs * 0.5
        return left * left_weight + right * right_weight


class IsolationTask(Task):
    """Background task for guitar isolation."""

    def __init__(
        self,
        project_id: UUID,
        source_path: Path,
        output_dir: Path,
        trim_start: Optional[float] = None,
        trim_end: Optional[float] = None,
        pan: Optional[float] = None,
        prompt: Optional[str] = None,
    ):
        super().__init__()
        self.project_id = project_id
        self.source_path = source_path
        self.output_dir = output_dir
        self.trim_start = trim_start
        self.trim_end = trim_end
        self.pan = pan or 0
        self.prompt = prompt or "electric guitar"

    def run(self) -> dict:
        """Execute guitar isolation."""
        logger.info(
            "Running isolation",
            project_id=str(self.project_id),
            source=str(self.source_path),
            prompt=self.prompt,
            trim_start=self.trim_start,
            trim_end=self.trim_end,
            pan=self.pan,
        )

        try:
            # Step 1: Load and preprocess audio
            self.update_progress(0.05, "Loading audio...")
            audio, sr = sf.read(self.source_path)
            
            # Step 2: Apply trim if specified
            if self.trim_start is not None or self.trim_end is not None:
                self.update_progress(0.1, "Trimming audio...")
                start = self.trim_start or 0
                end = self.trim_end or (len(audio) / sr)
                audio = apply_trim(audio, sr, start, end)
                logger.info(
                    "Audio trimmed",
                    start=start,
                    end=end,
                    duration=len(audio) / sr,
                )
            
            # Step 3: Apply pan extraction if specified
            if self.pan != 0 and len(audio.shape) > 1:
                self.update_progress(0.15, "Extracting panned audio...")
                audio = apply_pan(audio, self.pan)
                logger.info("Pan extraction applied", pan=self.pan)
            elif len(audio.shape) > 1:
                # Convert to mono
                audio = np.mean(audio, axis=1)
            
            # Step 4: Save preprocessed audio
            self.update_progress(0.2, "Preprocessing audio...")
            preprocessed_path = self.output_dir / "preprocessed.wav"
            sf.write(preprocessed_path, audio, sr)
            
            # Apply loudness normalization
            preprocess_audio(
                preprocessed_path,
                preprocessed_path,
                target_sr=settings.sample_rate,
                target_loudness=settings.target_loudness_lufs,
            )

            # Step 5: Run isolation with custom prompt
            self.update_progress(0.3, f"Isolating: {self.prompt}...")
            
            # Get backend and override prompt if using SAM Audio
            backend = get_isolation_backend(settings.isolation_backend)
            
            # If using SAM Audio backend, update the prompt
            if isinstance(backend, SAMAudioBackend):
                backend._prompt = self.prompt
                logger.info("Using custom prompt for SAM Audio", prompt=self.prompt)
            
            isolated_path = self.output_dir / "isolated_guitar.wav"

            result = backend.isolate(
                preprocessed_path,
                isolated_path,
                progress_callback=lambda p: self.update_progress(0.3 + p * 0.6, f"Isolating: {self.prompt}..."),
            )

            # Step 6: Update project
            self.update_progress(0.95, "Saving results...")

            # Run async update in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                project_manager = ProjectManager(settings.projects_dir)
                loop.run_until_complete(
                    project_manager.update_project(
                        self.project_id,
                        status=ProjectStatus.ISOLATED,
                        isolated_audio_path=str(isolated_path),
                        isolation_confidence=result["confidence"],
                    )
                )
            finally:
                loop.close()

            self.update_progress(1.0, "Isolation complete")

            return {
                "isolated_path": str(isolated_path),
                "confidence": result["confidence"],
                "duration": result.get("duration", 0),
                "prompt_used": self.prompt,
            }

        except Exception as e:
            logger.error("Isolation failed", error=str(e))
            # Update project status to error
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                project_manager = ProjectManager(settings.projects_dir)
                loop.run_until_complete(
                    project_manager.update_project(
                        self.project_id,
                        status=ProjectStatus.ERROR,
                    )
                )
            finally:
                loop.close()
            raise
