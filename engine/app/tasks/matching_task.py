"""Tone matching background task."""

import asyncio
from pathlib import Path
from uuid import UUID

import structlog

from app.config import settings
from app.features.acoustic_features import AcousticFeatureExtractor
from app.retrieval.acoustic_search import get_acoustic_search
from app.optimization.eq_optimizer import EQOptimizer
from app.rendering.renderer import AudioRenderer
from app.models import EQBand, EQSettings, MatchCandidate, ProjectStatus
from app.storage.project_manager import ProjectManager
from app.tasks.queue import Task

logger = structlog.get_logger()


class MatchingTask(Task):
    """Background task for tone matching.
    
    Uses acoustic feature extraction and probe-based NAM signatures
    for accurate tone matching based on how amps actually SOUND.
    """

    def __init__(
        self,
        project_id: UUID,
        isolated_path: Path,
        output_dir: Path,
    ):
        super().__init__()
        self.project_id = project_id
        self.isolated_path = isolated_path
        self.output_dir = output_dir

    def run(self) -> dict:
        """Execute tone matching pipeline."""
        logger.info(
            "Running matching",
            project_id=str(self.project_id),
            isolated=str(self.isolated_path),
        )

        try:
            # Get acoustic search (with pre-loaded indices)
            search = get_acoustic_search()
            
            # Step 1: Extract acoustic features from reference
            self.update_progress(0.1, "Extracting acoustic features...")
            
            # Use the search's extractor (has normalization stats)
            extractor = search.extractor
            reference_features = extractor.extract_from_file(self.isolated_path)
            
            logger.info(
                "Reference features extracted",
                shape=reference_features.shape,
                norm=float(sum(reference_features**2)**0.5),
            )

            # Step 2: Retrieve candidates using acoustic similarity
            self.update_progress(0.2, "Searching for similar NAM captures...")
            
            nam_results = search.find_similar_nams(
                reference_features,
                k=settings.top_k_nam_candidates,
            )
            
            ir_results = search.find_similar_irs(
                reference_features,
                k=settings.top_m_ir_candidates,
            )
            
            logger.info(
                "Candidates found",
                nam_count=len(nam_results),
                ir_count=len(ir_results),
                top_nam=nam_results[0][0].name if nam_results else "none",
                top_nam_score=nam_results[0][1] if nam_results else 0,
            )
            
            # Convert to format expected by optimizer
            nam_candidates = [metadata for metadata, score in nam_results]
            ir_candidates = [metadata for metadata, score in ir_results]

            # Step 3: Optimize for each flavor
            candidates: list[MatchCandidate] = []
            flavors = ["balanced", "brighter", "thicker"]

            optimizer = EQOptimizer()
            renderer = AudioRenderer()

            for i, flavor in enumerate(flavors):
                progress_base = 0.3 + i * 0.2
                self.update_progress(progress_base, f"Optimizing {flavor} match...")

                # Get best candidate for this flavor
                best_result = optimizer.optimize(
                    reference_path=self.isolated_path,
                    nam_candidates=nam_candidates,
                    ir_candidates=ir_candidates,
                    flavor=flavor,
                    reference_features={"embedding": reference_features},
                    progress_callback=lambda p: self.update_progress(
                        progress_base + p * 0.15, f"Optimizing {flavor}..."
                    ),
                )

                # Render synthesized audio
                self.update_progress(progress_base + 0.18, f"Rendering {flavor}...")
                rendered_path = self.output_dir / f"rendered_{flavor}.wav"
                renderer.render(
                    source_path=self.isolated_path,
                    output_path=rendered_path,
                    nam_model_id=best_result["nam_model_id"],
                    ir_id=best_result["ir_id"],
                    input_gain_db=best_result["input_gain_db"],
                    eq_settings=best_result["eq_settings"],
                )

                candidates.append(
                    MatchCandidate(
                        flavor=flavor,
                        nam_model_id=best_result["nam_model_id"],
                        nam_model_name=best_result["nam_model_name"],
                        ir_id=best_result["ir_id"],
                        ir_name=best_result["ir_name"],
                        input_gain_db=best_result["input_gain_db"],
                        eq_settings=best_result["eq_settings"],
                        similarity_score=best_result["similarity_score"],
                        rendered_audio_path=str(rendered_path),
                    )
                )

            # Step 4: Save results
            self.update_progress(0.95, "Saving results...")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                project_manager = ProjectManager(settings.projects_dir)

                # Get isolation confidence from project
                project = loop.run_until_complete(
                    project_manager.get_project(self.project_id)
                )
                isolation_confidence = project.isolation_confidence or 0.8

                loop.run_until_complete(
                    project_manager.save_match_results(
                        project_id=self.project_id,
                        reference_path=str(self.isolated_path),
                        isolated_path=str(self.isolated_path),
                        isolation_confidence=isolation_confidence,
                        candidates=candidates,
                    )
                )
            finally:
                loop.close()

            self.update_progress(1.0, "Matching complete")

            return {
                "candidates": [c.model_dump() for c in candidates],
            }

        except Exception as e:
            logger.error("Matching failed", error=str(e))
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
