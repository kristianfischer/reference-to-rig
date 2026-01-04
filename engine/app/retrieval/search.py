"""FAISS-based capture library search."""

from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from app.config import settings
from app.models import CaptureMetadata

logger = structlog.get_logger()


class CaptureSearch:
    """Search for NAM captures and IRs using FAISS.

    Uses approximate nearest neighbor search over precomputed
    embeddings to find similar captures quickly.
    """

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or settings.faiss_index_path
        self._nam_index = None
        self._ir_index = None
        self._nam_metadata: list[CaptureMetadata] = []
        self._ir_metadata: list[CaptureMetadata] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load indices if not already loaded."""
        if self._loaded:
            return

        try:
            self._load_indices()
        except Exception as e:
            logger.warning("Could not load FAISS indices, using mock data", error=str(e))
            self._load_mock_data()

        self._loaded = True

    def _load_indices(self) -> None:
        """Load FAISS indices from disk."""
        import faiss

        nam_index_path = self.index_path.parent / "nam_index.bin"
        ir_index_path = self.index_path.parent / "ir_index.bin"

        if nam_index_path.exists():
            self._nam_index = faiss.read_index(str(nam_index_path))
            logger.info("Loaded NAM index", size=self._nam_index.ntotal)

        if ir_index_path.exists():
            self._ir_index = faiss.read_index(str(ir_index_path))
            logger.info("Loaded IR index", size=self._ir_index.ntotal)

        # Load metadata
        import json
        metadata_path = settings.capture_library_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                data = json.load(f)
                self._nam_metadata = [
                    CaptureMetadata(**m) for m in data.get("nam_models", [])
                ]
                self._ir_metadata = [
                    CaptureMetadata(**m) for m in data.get("cab_irs", [])
                ]

    def _load_mock_data(self) -> None:
        """Load mock capture data for testing."""
        import faiss

        # Create mock NAM captures
        self._nam_metadata = [
            CaptureMetadata(
                id="nam_001",
                name="Clean Fender Twin",
                file_path="nam_models/fender_twin_clean.nam",
                capture_type="nam_model",
                style="clean",
                gain_range="clean",
                brightness="bright",
                tags=["fender", "clean", "sparkle"],
            ),
            CaptureMetadata(
                id="nam_002",
                name="Crunch Marshall JCM800",
                file_path="nam_models/marshall_jcm800_crunch.nam",
                capture_type="nam_model",
                style="crunch",
                gain_range="crunch",
                brightness="neutral",
                tags=["marshall", "crunch", "british"],
            ),
            CaptureMetadata(
                id="nam_003",
                name="High Gain Mesa Rectifier",
                file_path="nam_models/mesa_rectifier_high.nam",
                capture_type="nam_model",
                style="high_gain",
                gain_range="high_gain",
                brightness="dark",
                tags=["mesa", "high_gain", "metal"],
            ),
            CaptureMetadata(
                id="nam_004",
                name="Vox AC30 Chime",
                file_path="nam_models/vox_ac30_chime.nam",
                capture_type="nam_model",
                style="clean",
                gain_range="clean",
                brightness="bright",
                tags=["vox", "clean", "chime", "jangle"],
            ),
            CaptureMetadata(
                id="nam_005",
                name="Dumble Overdrive",
                file_path="nam_models/dumble_od.nam",
                capture_type="nam_model",
                style="overdrive",
                gain_range="crunch",
                brightness="neutral",
                tags=["dumble", "overdrive", "smooth"],
            ),
        ]

        # Create mock IRs
        self._ir_metadata = [
            CaptureMetadata(
                id="ir_001",
                name="Celestion Greenback 4x12",
                file_path="cab_irs/greenback_4x12.wav",
                capture_type="cab_ir",
                style="vintage",
                brightness="neutral",
                tags=["celestion", "greenback", "4x12"],
            ),
            CaptureMetadata(
                id="ir_002",
                name="Jensen P12R 1x12",
                file_path="cab_irs/jensen_p12r.wav",
                capture_type="cab_ir",
                style="vintage",
                brightness="bright",
                tags=["jensen", "vintage", "clean"],
            ),
            CaptureMetadata(
                id="ir_003",
                name="V30 4x12 Modern",
                file_path="cab_irs/v30_4x12.wav",
                capture_type="cab_ir",
                style="modern",
                brightness="neutral",
                tags=["v30", "modern", "tight"],
            ),
            CaptureMetadata(
                id="ir_004",
                name="Blue Alnico 2x12",
                file_path="cab_irs/blue_alnico_2x12.wav",
                capture_type="cab_ir",
                style="vintage",
                brightness="bright",
                tags=["celestion", "blue", "chime"],
            ),
        ]

        # Create mock FAISS indices
        embedding_dim = 24  # Match our feature extractor output
        n_nam = len(self._nam_metadata)
        n_ir = len(self._ir_metadata)

        # Generate deterministic mock embeddings
        np.random.seed(42)
        nam_embeddings = np.random.randn(n_nam, embedding_dim).astype(np.float32)
        ir_embeddings = np.random.randn(n_ir, embedding_dim).astype(np.float32)

        # Normalize
        nam_embeddings /= np.linalg.norm(nam_embeddings, axis=1, keepdims=True)
        ir_embeddings /= np.linalg.norm(ir_embeddings, axis=1, keepdims=True)

        # Create FAISS indices
        self._nam_index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine sim)
        self._nam_index.add(nam_embeddings)

        self._ir_index = faiss.IndexFlatIP(embedding_dim)
        self._ir_index.add(ir_embeddings)

        logger.info("Loaded mock capture data", nam_count=n_nam, ir_count=n_ir)

    def find_nam_models(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        style_filter: Optional[str] = None,
    ) -> list[CaptureMetadata]:
        """
        Find top-k similar NAM models.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            style_filter: Optional style filter (clean, crunch, high_gain)

        Returns:
            List of CaptureMetadata for matching NAM models
        """
        self._ensure_loaded()

        if self._nam_index is None or len(self._nam_metadata) == 0:
            logger.warning("No NAM index available")
            return []

        # Ensure query is 2D and correct dtype
        query = query_embedding.reshape(1, -1).astype(np.float32)

        # Adjust k if we have fewer items
        k = min(k, self._nam_index.ntotal)

        # Search
        distances, indices = self._nam_index.search(query, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue

            metadata = self._nam_metadata[idx]

            # Apply filter if specified
            if style_filter and metadata.style != style_filter:
                continue

            # Add score to metadata (via embedding field for now)
            metadata_copy = metadata.model_copy()
            metadata_copy.embedding = [float(score)]
            results.append(metadata_copy)

        return results

    def find_cabinet_irs(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        brightness_filter: Optional[str] = None,
    ) -> list[CaptureMetadata]:
        """
        Find top-k similar cabinet IRs.

        Args:
            query_embedding: Query embedding vector
            k: Number of results to return
            brightness_filter: Optional brightness filter (dark, neutral, bright)

        Returns:
            List of CaptureMetadata for matching IRs
        """
        self._ensure_loaded()

        if self._ir_index is None or len(self._ir_metadata) == 0:
            logger.warning("No IR index available")
            return []

        query = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(k, self._ir_index.ntotal)

        distances, indices = self._ir_index.search(query, k)

        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0:
                continue

            metadata = self._ir_metadata[idx]

            if brightness_filter and metadata.brightness != brightness_filter:
                continue

            metadata_copy = metadata.model_copy()
            metadata_copy.embedding = [float(score)]
            results.append(metadata_copy)

        return results

    def get_nam_by_id(self, nam_id: str) -> Optional[CaptureMetadata]:
        """Get NAM model by ID."""
        self._ensure_loaded()
        for m in self._nam_metadata:
            if m.id == nam_id:
                return m
        return None

    def get_ir_by_id(self, ir_id: str) -> Optional[CaptureMetadata]:
        """Get IR by ID."""
        self._ensure_loaded()
        for m in self._ir_metadata:
            if m.id == ir_id:
                return m
        return None


