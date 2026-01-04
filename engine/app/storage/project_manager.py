"""Project management and persistence."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID, uuid4

import aiofiles
import structlog
from sqlalchemy import select, delete

from app.models import (
    EQBand,
    EQSettings,
    MatchCandidate,
    MatchResultsResponse,
    ProjectResponse,
    ProjectStatus,
)
from app.storage.database import (
    async_session,
    ProjectRecord,
    MatchResultRecord,
)

logger = structlog.get_logger()


class ProjectManager:
    """Manages project lifecycle and persistence."""

    def __init__(self, projects_dir: Path):
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    async def create_project(
        self, name: str, description: Optional[str] = None
    ) -> ProjectResponse:
        """Create a new project."""
        project_id = uuid4()
        now = datetime.utcnow()

        # Create project directory
        project_dir = self.projects_dir / str(project_id)
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create database record
        async with async_session() as session:
            record = ProjectRecord(
                id=str(project_id),
                name=name,
                description=description,
                status=ProjectStatus.CREATED.value,
                created_at=now,
                updated_at=now,
            )
            session.add(record)
            await session.commit()

        logger.info("Project created", project_id=str(project_id), name=name)

        return ProjectResponse(
            id=project_id,
            name=name,
            description=description,
            status=ProjectStatus.CREATED,
            created_at=now,
            updated_at=now,
        )

    async def get_project(self, project_id: UUID) -> Optional[ProjectResponse]:
        """Get project by ID."""
        async with async_session() as session:
            result = await session.execute(
                select(ProjectRecord).where(ProjectRecord.id == str(project_id))
            )
            record = result.scalar_one_or_none()

            if not record:
                return None

            return ProjectResponse(
                id=UUID(record.id),
                name=record.name,
                description=record.description,
                status=ProjectStatus(record.status),
                created_at=record.created_at,
                updated_at=record.updated_at,
                source_audio_path=record.source_audio_path,
                isolated_audio_path=record.isolated_audio_path,
                isolation_confidence=record.isolation_confidence,
            )

    async def list_projects(self) -> list[ProjectResponse]:
        """List all projects."""
        async with async_session() as session:
            result = await session.execute(
                select(ProjectRecord).order_by(ProjectRecord.updated_at.desc())
            )
            records = result.scalars().all()

            return [
                ProjectResponse(
                    id=UUID(r.id),
                    name=r.name,
                    description=r.description,
                    status=ProjectStatus(r.status),
                    created_at=r.created_at,
                    updated_at=r.updated_at,
                    source_audio_path=r.source_audio_path,
                    isolated_audio_path=r.isolated_audio_path,
                    isolation_confidence=r.isolation_confidence,
                )
                for r in records
            ]

    async def update_project(
        self,
        project_id: UUID,
        status: Optional[ProjectStatus] = None,
        source_audio_path: Optional[str] = None,
        isolated_audio_path: Optional[str] = None,
        isolation_confidence: Optional[float] = None,
    ) -> bool:
        """Update project fields."""
        async with async_session() as session:
            result = await session.execute(
                select(ProjectRecord).where(ProjectRecord.id == str(project_id))
            )
            record = result.scalar_one_or_none()

            if not record:
                return False

            if status is not None:
                record.status = status.value
            if source_audio_path is not None:
                record.source_audio_path = source_audio_path
            if isolated_audio_path is not None:
                record.isolated_audio_path = isolated_audio_path
            if isolation_confidence is not None:
                record.isolation_confidence = isolation_confidence

            record.updated_at = datetime.utcnow()
            await session.commit()

            logger.info("Project updated", project_id=str(project_id), status=status)
            return True

    async def delete_project(self, project_id: UUID) -> bool:
        """Delete a project and its files."""
        async with async_session() as session:
            # Delete match results
            await session.execute(
                delete(MatchResultRecord).where(
                    MatchResultRecord.project_id == str(project_id)
                )
            )

            # Delete project record
            result = await session.execute(
                delete(ProjectRecord).where(ProjectRecord.id == str(project_id))
            )
            await session.commit()

            if result.rowcount == 0:
                return False

        # Delete project directory
        project_dir = self.projects_dir / str(project_id)
        if project_dir.exists():
            import shutil
            shutil.rmtree(project_dir)

        logger.info("Project deleted", project_id=str(project_id))
        return True

    async def save_match_results(
        self,
        project_id: UUID,
        reference_path: str,
        isolated_path: str,
        isolation_confidence: float,
        candidates: list[MatchCandidate],
    ) -> None:
        """Save match results to database."""
        async with async_session() as session:
            # Remove old results
            await session.execute(
                delete(MatchResultRecord).where(
                    MatchResultRecord.project_id == str(project_id)
                )
            )

            # Add new results
            record = MatchResultRecord(
                project_id=str(project_id),
                reference_audio_path=reference_path,
                isolated_audio_path=isolated_path,
                isolation_confidence=isolation_confidence,
                candidates_json=[c.model_dump() for c in candidates],
            )
            session.add(record)
            await session.commit()

        # Update project status
        await self.update_project(project_id, status=ProjectStatus.MATCHED)

    async def get_match_results(
        self, project_id: UUID
    ) -> Optional[MatchResultsResponse]:
        """Get match results for a project."""
        async with async_session() as session:
            result = await session.execute(
                select(MatchResultRecord)
                .where(MatchResultRecord.project_id == str(project_id))
                .order_by(MatchResultRecord.created_at.desc())
            )
            record = result.scalar_one_or_none()

            if not record:
                return None

            candidates = [
                MatchCandidate(**c) for c in record.candidates_json
            ]

            return MatchResultsResponse(
                project_id=project_id,
                reference_audio_path=record.reference_audio_path,
                isolated_audio_path=record.isolated_audio_path,
                isolation_confidence=record.isolation_confidence,
                candidates=candidates,
                created_at=record.created_at,
            )

    async def generate_recipe_markdown(
        self,
        project: ProjectResponse,
        results: MatchResultsResponse,
    ) -> str:
        """Generate human-readable rig recipe markdown."""
        lines = [
            f"# Rig Recipe: {project.name}",
            "",
            f"**Generated:** {results.created_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Isolation Confidence:** {results.isolation_confidence:.1%}",
            "",
        ]

        if project.description:
            lines.extend([
                "## Reference Description",
                project.description,
                "",
            ])

        lines.extend([
            "## Matched Tones",
            "",
        ])

        for i, candidate in enumerate(results.candidates, 1):
            lines.extend([
                f"### {i}. {candidate.flavor.title()} Match",
                "",
                f"**Similarity Score:** {candidate.similarity_score:.1%}",
                "",
                "#### Signal Chain",
                "",
                f"- **NAM Model:** {candidate.nam_model_name}",
                f"- **Cabinet IR:** {candidate.ir_name}",
                f"- **Input Gain:** {candidate.input_gain_db:+.1f} dB",
                "",
                "#### EQ Settings",
                "",
                "| Band | Frequency | Gain | Q |",
                "|------|-----------|------|---|",
            ])

            for band in candidate.eq_settings.bands:
                lines.append(
                    f"| {band.band_type.title()} | {band.frequency:.0f} Hz | "
                    f"{band.gain_db:+.1f} dB | {band.q:.1f} |"
                )

            if candidate.eq_settings.highpass_freq:
                lines.append(
                    f"| Highpass | {candidate.eq_settings.highpass_freq:.0f} Hz | - | - |"
                )
            if candidate.eq_settings.lowpass_freq:
                lines.append(
                    f"| Lowpass | {candidate.eq_settings.lowpass_freq:.0f} Hz | - | - |"
                )

            lines.extend(["", "---", ""])

        lines.extend([
            "## Notes",
            "",
            "- These settings are approximations based on spectral analysis",
            "- Fine-tune by ear for your specific guitar and playing style",
            "- Input gain may need adjustment based on your pickups",
            "",
            "---",
            "*Generated by Reference-to-Rig*",
        ])

        return "\n".join(lines)


