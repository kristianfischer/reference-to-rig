"""Database initialization and models."""

from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, Enum as SQLEnum, JSON
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

from app.config import settings
from app.models import ProjectStatus


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""

    pass


class ProjectRecord(Base):
    """Project database record."""

    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), default=ProjectStatus.CREATED.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    source_audio_path = Column(String(500), nullable=True)
    isolated_audio_path = Column(String(500), nullable=True)
    isolation_confidence = Column(Float, nullable=True)


class MatchResultRecord(Base):
    """Match results database record."""

    __tablename__ = "match_results"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()))
    project_id = Column(String(36), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    reference_audio_path = Column(String(500), nullable=False)
    isolated_audio_path = Column(String(500), nullable=False)
    isolation_confidence = Column(Float, default=0.0)
    candidates_json = Column(JSON, nullable=False)


class CaptureRecord(Base):
    """Capture library record (NAM models and IRs)."""

    __tablename__ = "captures"

    id = Column(String(36), primary_key=True)
    name = Column(String(200), nullable=False)
    file_path = Column(String(500), nullable=False)
    capture_type = Column(String(20), nullable=False)  # nam_model or cab_ir
    style = Column(String(50), nullable=True)
    gain_range = Column(String(20), nullable=True)
    brightness = Column(String(20), nullable=True)
    tags = Column(JSON, default=list)
    embedding_index = Column(Integer, nullable=True)  # FAISS index position


# Async engine and session
engine = create_async_engine(
    f"sqlite+aiosqlite:///{settings.db_path}",
    echo=settings.debug,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db() -> None:
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncSession:
    """Get database session."""
    async with async_session() as session:
        yield session

