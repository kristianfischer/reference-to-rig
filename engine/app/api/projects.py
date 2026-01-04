"""Projects API endpoints."""

import shutil
from pathlib import Path
from typing import Optional
from uuid import UUID

import aiofiles
import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from app.config import settings
from app.models import (
    ImportResponse,
    IsolationRequest,
    IsolationResponse,
    MatchCandidate,
    MatchResponse,
    MatchResultsResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectStatus,
    RigRecipe,
    TaskStatus,
    TaskStatusResponse,
)
from app.storage.project_manager import ProjectManager
from app.tasks.queue import task_queue
from app.tasks.isolation_task import IsolationTask
from app.tasks.matching_task import MatchingTask

logger = structlog.get_logger()
router = APIRouter()

project_manager = ProjectManager(settings.projects_dir)


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(data: ProjectCreate) -> ProjectResponse:
    """Create a new project."""
    logger.info("Creating project", name=data.name)
    project = await project_manager.create_project(data.name, data.description)
    return project


@router.get("", response_model=list[ProjectResponse])
async def list_projects() -> list[ProjectResponse]:
    """List all projects."""
    return await project_manager.list_projects()


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(project_id: UUID) -> ProjectResponse:
    """Get project details."""
    project = await project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(project_id: UUID) -> None:
    """Delete a project."""
    success = await project_manager.delete_project(project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found")


@router.post("/{project_id}/import", response_model=ImportResponse)
async def import_audio(
    project_id: UUID,
    file: UploadFile = File(...),
) -> ImportResponse:
    """Import audio file into project."""
    project = await project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext}. Supported: wav, mp3, flac, ogg, m4a",
        )

    logger.info("Importing audio", project_id=str(project_id), filename=file.filename)

    # Save uploaded file
    project_dir = settings.projects_dir / str(project_id)
    source_path = project_dir / f"source{ext}"

    async with aiofiles.open(source_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Get audio info
    from app.features.audio_utils import get_audio_info

    audio_info = get_audio_info(source_path)

    # Update project
    await project_manager.update_project(
        project_id,
        status=ProjectStatus.IMPORTED,
        source_audio_path=str(source_path),
    )

    return ImportResponse(
        project_id=project_id,
        source_audio_path=str(source_path),
        duration_seconds=audio_info["duration"],
        sample_rate=audio_info["sample_rate"],
        channels=audio_info["channels"],
    )


@router.post("/{project_id}/record")
async def record_audio(project_id: UUID) -> dict:
    """
    Record audio from interface/mic.

    NOTE: This is a stub endpoint. Recording from audio interface requires:
    1. UI-side Web Audio API or native audio capture
    2. Stream the recorded audio to this endpoint
    3. Or use a separate recording service

    For MVP, use the import endpoint with pre-recorded files.
    """
    raise HTTPException(
        status_code=501,
        detail="Recording not implemented in MVP. Use /import with recorded files.",
    )


@router.post("/{project_id}/isolate", response_model=IsolationResponse)
async def isolate_guitar(
    project_id: UUID,
    request: IsolationRequest = IsolationRequest(),
) -> IsolationResponse:
    """Run guitar isolation on imported audio."""
    project = await project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status not in [ProjectStatus.IMPORTED, ProjectStatus.ISOLATED, ProjectStatus.MATCHED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot isolate from status: {project.status}. Import audio first.",
        )

    if not project.source_audio_path:
        raise HTTPException(status_code=400, detail="No source audio imported")

    logger.info(
        "Starting isolation",
        project_id=str(project_id),
        prompt=request.prompt,
        trim_start=request.trim_start,
        trim_end=request.trim_end,
        pan=request.pan,
    )

    # Update status
    await project_manager.update_project(project_id, status=ProjectStatus.ISOLATING)

    # Queue isolation task with options
    task = IsolationTask(
        project_id=project_id,
        source_path=Path(project.source_audio_path),
        output_dir=settings.projects_dir / str(project_id),
        trim_start=request.trim_start,
        trim_end=request.trim_end,
        pan=request.pan,
        prompt=request.prompt,
    )
    task_id = task_queue.submit(task)

    return IsolationResponse(
        project_id=project_id,
        task_id=task_id,
        status=TaskStatus.PENDING,
    )


@router.post("/{project_id}/match", response_model=MatchResponse)
async def match_tone(project_id: UUID) -> MatchResponse:
    """Run tone matching on isolated guitar."""
    project = await project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if project.status not in [ProjectStatus.ISOLATED, ProjectStatus.MATCHED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot match from status: {project.status}. Run isolation first.",
        )

    if not project.isolated_audio_path:
        raise HTTPException(status_code=400, detail="No isolated audio available")

    logger.info("Starting matching", project_id=str(project_id))

    # Update status
    await project_manager.update_project(project_id, status=ProjectStatus.MATCHING)

    # Queue matching task
    task = MatchingTask(
        project_id=project_id,
        isolated_path=Path(project.isolated_audio_path),
        output_dir=settings.projects_dir / str(project_id),
    )
    task_id = task_queue.submit(task)

    return MatchResponse(
        project_id=project_id,
        task_id=task_id,
        status=TaskStatus.PENDING,
    )


@router.get("/{project_id}/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(project_id: UUID, task_id: str) -> TaskStatusResponse:
    """Get status of a background task."""
    task_status = task_queue.get_status(task_id)
    if not task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    return task_status


@router.get("/{project_id}/results", response_model=Optional[MatchResultsResponse])
async def get_results(project_id: UUID) -> Optional[MatchResultsResponse]:
    """Get matching results for a project."""
    project = await project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    results = await project_manager.get_match_results(project_id)
    if not results:
        return None

    return results


@router.get("/{project_id}/audio/{audio_type}")
async def get_audio_file(project_id: UUID, audio_type: str) -> FileResponse:
    """
    Get audio file from project.

    audio_type: source, isolated, or match_{flavor} (e.g., match_balanced)
    """
    project = await project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    project_dir = settings.projects_dir / str(project_id)

    if audio_type == "source":
        # Find source file
        for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]:
            path = project_dir / f"source{ext}"
            if path.exists():
                return FileResponse(path, media_type="audio/wav")
        raise HTTPException(status_code=404, detail="Source audio not found")

    elif audio_type == "isolated":
        path = project_dir / "isolated_guitar.wav"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Isolated audio not found")
        return FileResponse(path, media_type="audio/wav")

    elif audio_type.startswith("match_"):
        flavor = audio_type.replace("match_", "")
        path = project_dir / f"rendered_{flavor}.wav"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Rendered audio for {flavor} not found")
        return FileResponse(path, media_type="audio/wav")

    else:
        raise HTTPException(status_code=400, detail=f"Unknown audio type: {audio_type}")


@router.get("/{project_id}/export/json")
async def export_json(project_id: UUID) -> RigRecipe:
    """Export rig recipe as JSON."""
    project = await project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    results = await project_manager.get_match_results(project_id)
    if not results:
        raise HTTPException(status_code=400, detail="No match results available")

    return RigRecipe(
        project_name=project.name,
        created_at=results.created_at,
        reference_description=project.description,
        candidates=results.candidates,
    )


@router.get("/{project_id}/export/markdown")
async def export_markdown(project_id: UUID) -> FileResponse:
    """Export rig recipe as Markdown file."""
    project = await project_manager.get_project(project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    results = await project_manager.get_match_results(project_id)
    if not results:
        raise HTTPException(status_code=400, detail="No match results available")

    # Generate markdown
    markdown = await project_manager.generate_recipe_markdown(project, results)

    # Save to file
    project_dir = settings.projects_dir / str(project_id)
    md_path = project_dir / "rig_recipe.md"

    async with aiofiles.open(md_path, "w") as f:
        await f.write(markdown)

    return FileResponse(
        md_path,
        media_type="text/markdown",
        filename=f"{project.name.replace(' ', '_')}_rig_recipe.md",
    )


