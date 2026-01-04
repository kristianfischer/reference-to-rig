"""Application configuration via Pydantic settings."""

from pathlib import Path
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="RTR_",
    )

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False

    # Paths
    data_dir: Path = Path("./data")
    capture_library_dir: Path = Path("./capture_library")
    projects_dir: Path = Path("./data/projects")

    # Audio
    sample_rate: int = 48000
    target_loudness_lufs: float = -18.0

    # Matching
    top_k_nam_candidates: int = 10
    top_m_ir_candidates: int = 10
    optimization_iterations: int = 50

    # Backend mode
    isolation_backend: Literal["sam_audio", "mock"] = "mock"
    nam_backend: Literal["nam", "mock"] = "mock"

    # SAM Audio settings (when isolation_backend=sam_audio)
    # See: https://huggingface.co/facebook/sam-audio-large
    # Models: sam-audio-small (~1GB VRAM), sam-audio-base (~2GB), sam-audio-large (~4GB+)
    sam_audio_model: str = "facebook/sam-audio-small"  # Use small for low VRAM GPUs
    sam_audio_device: Optional[str] = "cpu"  # 'cpu' for safety, 'cuda' if you have 4GB+ VRAM
    sam_audio_prompt: str = "electric guitar"  # Use lowercase noun-phrase format
    sam_audio_use_reranking: bool = True  # Better quality, higher latency
    sam_audio_reranking_candidates: int = 1  # 1=fast, 4=quality (more VRAM)

    # NAM settings (when nam_backend=nam)
    # Download models from: https://tonehunt.org
    nam_device: Optional[str] = None  # None for auto-detect, 'cpu' for CPU-only

    # Database
    db_path: Path = Path("./data/reference_to_rig.db")
    faiss_index_path: Path = Path("./data/faiss_index.bin")

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self.capture_library_dir.mkdir(parents=True, exist_ok=True)
        (self.capture_library_dir / "nam_models").mkdir(exist_ok=True)
        (self.capture_library_dir / "cab_irs").mkdir(exist_ok=True)
        (self.capture_library_dir / "probes").mkdir(exist_ok=True)


settings = Settings()

