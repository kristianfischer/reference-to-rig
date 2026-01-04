"""Pydantic models for API request/response."""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    """Project processing status."""

    CREATED = "created"
    IMPORTED = "imported"
    ISOLATING = "isolating"
    ISOLATED = "isolated"
    MATCHING = "matching"
    MATCHED = "matched"
    ERROR = "error"


class TaskStatus(str, Enum):
    """Background task status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class EQBand(BaseModel):
    """Single EQ band parameters."""

    frequency: float = Field(..., description="Center frequency in Hz")
    gain_db: float = Field(..., description="Gain in dB")
    q: float = Field(default=1.0, description="Q factor (bandwidth)")
    band_type: str = Field(default="peak", description="Band type: peak, lowshelf, highshelf, highpass, lowpass")


class EQSettings(BaseModel):
    """Complete EQ configuration."""

    bands: list[EQBand] = Field(default_factory=list)
    highpass_freq: Optional[float] = Field(default=None, description="Highpass filter frequency")
    lowpass_freq: Optional[float] = Field(default=None, description="Lowpass filter frequency")


class MatchCandidate(BaseModel):
    """A single tone match candidate."""

    flavor: str = Field(..., description="Match flavor: balanced, brighter, thicker")
    nam_model_id: str = Field(..., description="NAM model identifier")
    nam_model_name: str = Field(..., description="NAM model display name")
    ir_id: str = Field(..., description="Cabinet IR identifier")
    ir_name: str = Field(..., description="Cabinet IR display name")
    input_gain_db: float = Field(..., description="Recommended input gain in dB")
    eq_settings: EQSettings = Field(..., description="Recommended EQ settings")
    similarity_score: float = Field(..., description="Similarity score (0-1, higher is better)")
    rendered_audio_path: Optional[str] = Field(default=None, description="Path to rendered audio file")


class ProjectCreate(BaseModel):
    """Request to create a new project."""

    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)


class ProjectResponse(BaseModel):
    """Project information response."""

    id: UUID
    name: str
    description: Optional[str]
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime
    source_audio_path: Optional[str] = None
    isolated_audio_path: Optional[str] = None
    isolation_confidence: Optional[float] = None


class ImportResponse(BaseModel):
    """Response after audio import."""

    project_id: UUID
    source_audio_path: str
    duration_seconds: float
    sample_rate: int
    channels: int


class IsolationRequest(BaseModel):
    """Request to start guitar isolation."""
    
    trim_start: Optional[float] = Field(default=None, description="Start time in seconds")
    trim_end: Optional[float] = Field(default=None, description="End time in seconds")
    pan: Optional[float] = Field(default=0, ge=-100, le=100, description="Pan position: -100 (L) to +100 (R)")
    prompt: Optional[str] = Field(default="electric guitar", description="Isolation prompt for SAM Audio")


class IsolationResponse(BaseModel):
    """Response after guitar isolation."""

    project_id: UUID
    task_id: str
    status: TaskStatus


class MatchResponse(BaseModel):
    """Response after tone matching."""

    project_id: UUID
    task_id: str
    status: TaskStatus


class TaskStatusResponse(BaseModel):
    """Response for task status query."""

    task_id: str
    status: TaskStatus
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class MatchResultsResponse(BaseModel):
    """Complete matching results."""

    project_id: UUID
    reference_audio_path: str
    isolated_audio_path: str
    isolation_confidence: float
    candidates: list[MatchCandidate]
    created_at: datetime


class RigRecipe(BaseModel):
    """Exportable rig recipe."""

    version: str = "1.0"
    project_name: str
    created_at: datetime
    reference_description: Optional[str] = None
    candidates: list[MatchCandidate]
    notes: Optional[str] = None


class CaptureMetadata(BaseModel):
    """Metadata for a NAM capture or IR."""

    id: str
    name: str
    file_path: str
    capture_type: str = Field(..., description="nam_model or cab_ir")
    style: Optional[str] = None
    gain_range: Optional[str] = None  # clean, crunch, high_gain
    brightness: Optional[str] = None  # dark, neutral, bright
    tags: list[str] = Field(default_factory=list)
    embedding: Optional[list[float]] = None


