"""Feature extraction module."""

from app.features.acoustic_features import (
    AcousticFeatureExtractor,
    FEATURE_DIM,
    extract_features,
)

__all__ = [
    "AcousticFeatureExtractor",
    "FEATURE_DIM",
    "extract_features",
]
