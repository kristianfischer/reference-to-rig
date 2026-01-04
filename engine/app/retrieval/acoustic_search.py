"""
Acoustic-based NAM retrieval using probe-rendered signatures.

This module provides FAISS-based search over NAM models using
acoustic feature vectors extracted from probe-rendered audio.

The acoustic signatures capture actual tonal characteristics:
- Spectral shape (brightness, EQ curve)
- Dynamics (compression, feel)
- Harmonic content (distortion character)
- Texture patterns

This enables accurate tone matching based on how NAMs actually SOUND,
not just metadata labels from filenames.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import structlog

from app.config import settings
from app.models import CaptureMetadata
from app.features.acoustic_features import (
    AcousticFeatureExtractor,
    FEATURE_DIM,
)

logger = structlog.get_logger()


class AcousticCaptureSearch:
    """
    Search NAM captures and IRs using acoustic similarity.
    
    Uses FAISS inner-product search over L2-normalized feature vectors,
    which is equivalent to cosine similarity.
    
    Higher scores = more similar acoustic characteristics.
    """
    
    def __init__(
        self,
        index_dir: Optional[Path] = None,
        capture_library_dir: Optional[Path] = None,
    ):
        """
        Initialize acoustic search.
        
        Args:
            index_dir: Directory containing FAISS indices and metadata
            capture_library_dir: Directory containing NAM/IR files and metadata
        """
        self.index_dir = index_dir or settings.data_dir
        self.capture_library_dir = capture_library_dir or settings.capture_library_dir
        
        # Indices
        self._nam_index = None
        self._ir_index = None
        
        # Metadata mappings (index position -> model info)
        self._nam_index_map: list[dict] = []
        self._ir_index_map: list[dict] = []
        
        # Full metadata from metadata.json
        self._nam_metadata: dict[str, CaptureMetadata] = {}
        self._ir_metadata: dict[str, CaptureMetadata] = {}
        
        # Feature extractor with normalization stats
        self._extractor: Optional[AcousticFeatureExtractor] = None
        
        self._loaded = False
    
    def _ensure_loaded(self) -> None:
        """Load indices and metadata if not already loaded."""
        if self._loaded:
            return
        
        try:
            self._load_indices()
            self._loaded = True
        except Exception as e:
            logger.warning(
                "Could not load acoustic indices. "
                "Run 'python scripts/build_probe_index.py' to build them.",
                error=str(e)
            )
            # Fall back to metadata-based search
            self._load_fallback()
            self._loaded = True
    
    def _load_indices(self) -> None:
        """Load FAISS indices and metadata."""
        import faiss
        
        # Load NAM index
        nam_index_path = self.index_dir / "nam_acoustic_index.bin"
        if nam_index_path.exists():
            self._nam_index = faiss.read_index(str(nam_index_path))
            logger.info("Loaded NAM acoustic index", size=self._nam_index.ntotal)
        
        # Load IR index
        ir_index_path = self.index_dir / "ir_acoustic_index.bin"
        if ir_index_path.exists():
            self._ir_index = faiss.read_index(str(ir_index_path))
            logger.info("Loaded IR acoustic index", size=self._ir_index.ntotal)
        
        # Load index metadata (maps index position to model ID)
        index_metadata_path = self.index_dir / "index_metadata.json"
        if index_metadata_path.exists():
            with open(index_metadata_path) as f:
                index_metadata = json.load(f)
            self._nam_index_map = index_metadata.get("nam_models", [])
            self._ir_index_map = index_metadata.get("cab_irs", [])
        
        # Load normalization stats for feature extractor
        norm_stats_path = self.index_dir / "normalization_stats.json"
        self._extractor = AcousticFeatureExtractor(
            normalization_stats_path=norm_stats_path if norm_stats_path.exists() else None
        )
        
        # Load full metadata
        self._load_capture_metadata()
    
    def _load_fallback(self) -> None:
        """Load fallback when acoustic indices don't exist."""
        logger.info("Using metadata-based fallback (acoustic index not found)")
        
        # Just load metadata
        self._load_capture_metadata()
        
        # Create extractor without normalization
        self._extractor = AcousticFeatureExtractor()
    
    def _load_capture_metadata(self) -> None:
        """Load capture metadata from metadata.json."""
        metadata_path = self.capture_library_dir / "metadata.json"
        
        if not metadata_path.exists():
            logger.warning("metadata.json not found")
            return
        
        with open(metadata_path) as f:
            data = json.load(f)
        
        # Index by ID for fast lookup
        for m in data.get("nam_models", []):
            try:
                metadata = CaptureMetadata(**m)
                self._nam_metadata[m["id"]] = metadata
            except Exception as e:
                logger.debug(f"Failed to parse NAM metadata: {e}")
        
        for m in data.get("cab_irs", []):
            try:
                metadata = CaptureMetadata(**m)
                self._ir_metadata[m["id"]] = metadata
            except Exception as e:
                logger.debug(f"Failed to parse IR metadata: {e}")
        
        logger.info(
            "Loaded capture metadata",
            nam_count=len(self._nam_metadata),
            ir_count=len(self._ir_metadata),
        )
    
    @property
    def extractor(self) -> AcousticFeatureExtractor:
        """Get the feature extractor with normalization loaded."""
        self._ensure_loaded()
        return self._extractor
    
    def find_similar_nams(
        self,
        query_features: np.ndarray,
        k: int = 10,
    ) -> list[tuple[CaptureMetadata, float]]:
        """
        Find NAM models acoustically similar to query features.
        
        Args:
            query_features: 155-dim feature vector (should be L2 normalized)
            k: Number of results to return
            
        Returns:
            List of (CaptureMetadata, similarity_score) tuples,
            sorted by similarity (highest first)
        """
        self._ensure_loaded()
        
        if self._nam_index is None:
            logger.warning("NAM acoustic index not available, using fallback")
            return self._fallback_nam_search(k)
        
        # Ensure query is correct shape and type
        query = query_features.reshape(1, -1).astype(np.float32)
        
        # Search
        k = min(k, self._nam_index.ntotal)
        distances, indices = self._nam_index.search(query, k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._nam_index_map):
                continue
            
            # Get model ID from index map
            model_info = self._nam_index_map[idx]
            model_id = model_info["id"]
            
            # Get full metadata
            metadata = self._nam_metadata.get(model_id)
            if metadata is None:
                # Create minimal metadata
                metadata = CaptureMetadata(
                    id=model_id,
                    name=model_info.get("name", model_id),
                    file_path=f"nam_models/{model_id}.nam",
                    capture_type="nam_model",
                )
            
            results.append((metadata, float(score)))
        
        return results
    
    def find_similar_irs(
        self,
        query_features: np.ndarray,
        k: int = 10,
    ) -> list[tuple[CaptureMetadata, float]]:
        """
        Find cabinet IRs acoustically similar to query features.
        
        Args:
            query_features: 155-dim feature vector
            k: Number of results
            
        Returns:
            List of (CaptureMetadata, similarity_score) tuples
        """
        self._ensure_loaded()
        
        if self._ir_index is None:
            logger.warning("IR acoustic index not available, using fallback")
            return self._fallback_ir_search(k)
        
        query = query_features.reshape(1, -1).astype(np.float32)
        
        k = min(k, self._ir_index.ntotal)
        distances, indices = self._ir_index.search(query, k)
        
        results = []
        for idx, score in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self._ir_index_map):
                continue
            
            ir_info = self._ir_index_map[idx]
            ir_id = ir_info["id"]
            
            metadata = self._ir_metadata.get(ir_id)
            if metadata is None:
                metadata = CaptureMetadata(
                    id=ir_id,
                    name=ir_info.get("name", ir_id),
                    file_path=f"cab_irs/{ir_id}.wav",
                    capture_type="cab_ir",
                )
            
            results.append((metadata, float(score)))
        
        return results
    
    def _fallback_nam_search(self, k: int) -> list[tuple[CaptureMetadata, float]]:
        """Fallback search when acoustic index isn't available."""
        # Return first k NAM models with dummy scores
        results = []
        for i, (model_id, metadata) in enumerate(self._nam_metadata.items()):
            if i >= k:
                break
            # Dummy score based on position
            score = 0.5 - (i * 0.01)
            results.append((metadata, score))
        return results
    
    def _fallback_ir_search(self, k: int) -> list[tuple[CaptureMetadata, float]]:
        """Fallback search when acoustic index isn't available."""
        results = []
        for i, (ir_id, metadata) in enumerate(self._ir_metadata.items()):
            if i >= k:
                break
            score = 0.5 - (i * 0.01)
            results.append((metadata, score))
        return results
    
    def get_nam_by_id(self, nam_id: str) -> Optional[CaptureMetadata]:
        """Get NAM metadata by ID."""
        self._ensure_loaded()
        return self._nam_metadata.get(nam_id)
    
    def get_ir_by_id(self, ir_id: str) -> Optional[CaptureMetadata]:
        """Get IR metadata by ID."""
        self._ensure_loaded()
        return self._ir_metadata.get(ir_id)
    
    @property
    def nam_count(self) -> int:
        """Number of NAM models in index."""
        self._ensure_loaded()
        return self._nam_index.ntotal if self._nam_index else len(self._nam_metadata)
    
    @property
    def ir_count(self) -> int:
        """Number of IRs in index."""
        self._ensure_loaded()
        return self._ir_index.ntotal if self._ir_index else len(self._ir_metadata)


# Global instance (lazy-loaded)
_search_instance: Optional[AcousticCaptureSearch] = None


def get_acoustic_search() -> AcousticCaptureSearch:
    """Get or create the global acoustic search instance."""
    global _search_instance
    if _search_instance is None:
        _search_instance = AcousticCaptureSearch()
    return _search_instance

