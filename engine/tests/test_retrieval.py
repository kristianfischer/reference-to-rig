"""Tests for capture retrieval module."""

import numpy as np
import pytest

from app.retrieval.search import CaptureSearch


class TestCaptureSearch:
    """Tests for FAISS-based capture search."""

    def test_mock_data_loaded(self):
        search = CaptureSearch()
        search._ensure_loaded()

        assert len(search._nam_metadata) > 0
        assert len(search._ir_metadata) > 0

    def test_find_nam_models(self):
        search = CaptureSearch()

        # Random query embedding
        query = np.random.randn(24).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = search.find_nam_models(query, k=3)

        assert len(results) <= 3
        assert all(r.capture_type == "nam_model" for r in results)

    def test_find_cabinet_irs(self):
        search = CaptureSearch()

        query = np.random.randn(24).astype(np.float32)
        query = query / np.linalg.norm(query)

        results = search.find_cabinet_irs(query, k=2)

        assert len(results) <= 2
        assert all(r.capture_type == "cab_ir" for r in results)

    def test_get_nam_by_id(self):
        search = CaptureSearch()
        search._ensure_loaded()

        # Get first NAM model ID
        nam_id = search._nam_metadata[0].id

        result = search.get_nam_by_id(nam_id)
        assert result is not None
        assert result.id == nam_id

    def test_get_ir_by_id(self):
        search = CaptureSearch()
        search._ensure_loaded()

        ir_id = search._ir_metadata[0].id

        result = search.get_ir_by_id(ir_id)
        assert result is not None
        assert result.id == ir_id

    def test_get_nonexistent_returns_none(self):
        search = CaptureSearch()

        assert search.get_nam_by_id("nonexistent") is None
        assert search.get_ir_by_id("nonexistent") is None


