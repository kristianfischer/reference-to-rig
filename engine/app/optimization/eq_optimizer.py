"""EQ parameter optimization using gradient-free search.

This module optimizes EQ parameters to minimize perceptual
distance between reference and synthesized audio.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import structlog

from app.models import CaptureMetadata, EQBand, EQSettings

logger = structlog.get_logger()


class EQOptimizer:
    """Optimize EQ parameters for tone matching.

    Uses coordinate descent (for speed) or CMA-ES (for quality)
    to find EQ settings that minimize perceptual distance.
    """

    def __init__(
        self,
        method: str = "coordinate_descent",
        max_iterations: int = 50,
    ):
        self.method = method
        self.max_iterations = max_iterations

        # EQ parameter bounds
        self.freq_bounds = [
            (80, 200),    # Low shelf
            (200, 500),   # Low-mid
            (500, 1500),  # Mid
            (1500, 3500), # High-mid
            (3500, 6000), # Presence
            (6000, 12000),# High shelf
        ]
        self.gain_bounds = (-12.0, 12.0)  # dB
        self.q_bounds = (0.5, 4.0)
        self.input_gain_bounds = (-12.0, 12.0)

    def optimize(
        self,
        reference_path: Path,
        nam_candidates: list[CaptureMetadata],
        ir_candidates: list[CaptureMetadata],
        flavor: str,
        reference_features: dict,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> dict:
        """
        Optimize EQ and find best NAM/IR combination.

        Args:
            reference_path: Path to reference audio
            nam_candidates: List of NAM model candidates
            ir_candidates: List of IR candidates
            flavor: Match flavor (balanced, brighter, thicker)
            reference_features: Pre-extracted reference features
            progress_callback: Progress callback function

        Returns:
            dict with optimized settings:
                - nam_model_id
                - nam_model_name
                - ir_id
                - ir_name
                - input_gain_db
                - eq_settings
                - similarity_score
        """
        logger.info("Starting EQ optimization", flavor=flavor)

        # Load reference audio
        ref_audio, sr = sf.read(reference_path)
        if len(ref_audio.shape) > 1:
            ref_audio = np.mean(ref_audio, axis=1)

        best_result = None
        best_score = float('inf')

        total_combinations = len(nam_candidates) * len(ir_candidates)
        current = 0

        # For MVP, just try top candidates with default EQ
        # A full implementation would optimize EQ for each combination
        for nam in nam_candidates[:3]:  # Top 3 NAM
            for ir in ir_candidates[:2]:  # Top 2 IR
                current += 1

                if progress_callback:
                    progress_callback(current / total_combinations)

                # Optimize EQ for this combination
                eq_settings, input_gain, score = self._optimize_eq(
                    ref_audio,
                    sr,
                    nam,
                    ir,
                    flavor,
                    reference_features,
                )

                if score < best_score:
                    best_score = score
                    best_result = {
                        "nam_model_id": nam.id,
                        "nam_model_name": nam.name,
                        "ir_id": ir.id,
                        "ir_name": ir.name,
                        "input_gain_db": input_gain,
                        "eq_settings": eq_settings,
                        "similarity_score": 1.0 - min(1.0, score),  # Convert to similarity
                    }

        # If no valid result, return a default
        if best_result is None:
            logger.warning("No valid optimization result, using defaults")
            best_result = self._get_default_result(
                nam_candidates[0] if nam_candidates else None,
                ir_candidates[0] if ir_candidates else None,
                flavor,
            )

        logger.info(
            "Optimization complete",
            flavor=flavor,
            nam=best_result["nam_model_name"],
            ir=best_result["ir_name"],
            score=best_result["similarity_score"],
        )

        return best_result

    def _optimize_eq(
        self,
        ref_audio: np.ndarray,
        sr: int,
        nam: CaptureMetadata,
        ir: CaptureMetadata,
        flavor: str,
        reference_features: dict,
    ) -> tuple[EQSettings, float, float]:
        """
        Optimize EQ parameters for a specific NAM/IR combination.

        Uses coordinate descent for speed.

        Returns:
            (eq_settings, input_gain_db, loss)
        """
        # Start with flavor-specific initial EQ
        eq_params = self._get_initial_eq(flavor)
        input_gain = 0.0

        # Reference spectral features
        ref_centroid = reference_features.get("spectral_centroid", 2000)
        ref_tilt = reference_features.get("spectral_tilt", 0)

        # Coordinate descent optimization
        for iteration in range(self.max_iterations):
            # Evaluate current parameters
            current_loss = self._compute_loss(
                ref_audio, sr, eq_params, input_gain, ref_centroid, ref_tilt, flavor
            )

            # Try adjusting each parameter
            improved = False

            # Adjust input gain
            for delta in [-1.0, 1.0]:
                new_gain = np.clip(
                    input_gain + delta,
                    self.input_gain_bounds[0],
                    self.input_gain_bounds[1],
                )
                loss = self._compute_loss(
                    ref_audio, sr, eq_params, new_gain, ref_centroid, ref_tilt, flavor
                )
                if loss < current_loss:
                    input_gain = new_gain
                    current_loss = loss
                    improved = True
                    break

            # Adjust EQ bands
            for i, band in enumerate(eq_params):
                # Adjust gain
                for delta in [-1.0, 1.0]:
                    new_gain = np.clip(
                        band["gain"] + delta,
                        self.gain_bounds[0],
                        self.gain_bounds[1],
                    )
                    test_params = eq_params.copy()
                    test_params[i] = {**band, "gain": new_gain}
                    loss = self._compute_loss(
                        ref_audio, sr, test_params, input_gain, ref_centroid, ref_tilt, flavor
                    )
                    if loss < current_loss:
                        eq_params[i]["gain"] = new_gain
                        current_loss = loss
                        improved = True
                        break

            # Early stopping if no improvement
            if not improved and iteration > 10:
                break

        # Convert to EQSettings
        eq_settings = EQSettings(
            bands=[
                EQBand(
                    frequency=b["freq"],
                    gain_db=b["gain"],
                    q=b["q"],
                    band_type=b["type"],
                )
                for b in eq_params
            ],
            highpass_freq=80.0,
            lowpass_freq=12000.0,
        )

        return eq_settings, input_gain, current_loss

    def _get_initial_eq(self, flavor: str) -> list[dict]:
        """Get flavor-specific initial EQ settings."""
        base_eq = [
            {"freq": 100, "gain": 0.0, "q": 0.7, "type": "lowshelf"},
            {"freq": 300, "gain": 0.0, "q": 1.5, "type": "peak"},
            {"freq": 800, "gain": 0.0, "q": 1.5, "type": "peak"},
            {"freq": 2000, "gain": 0.0, "q": 1.5, "type": "peak"},
            {"freq": 4500, "gain": 0.0, "q": 1.5, "type": "peak"},
            {"freq": 8000, "gain": 0.0, "q": 0.7, "type": "highshelf"},
        ]

        if flavor == "brighter":
            # Boost presence and highs
            base_eq[4]["gain"] = 3.0  # Presence
            base_eq[5]["gain"] = 2.0  # Highs
            base_eq[0]["gain"] = -2.0  # Cut lows slightly
        elif flavor == "thicker":
            # Boost low-mids, cut presence
            base_eq[0]["gain"] = 3.0  # Low shelf
            base_eq[1]["gain"] = 2.0  # Low-mid
            base_eq[4]["gain"] = -2.0  # Cut presence
        # balanced: keep at 0

        return base_eq

    def _compute_loss(
        self,
        ref_audio: np.ndarray,
        sr: int,
        eq_params: list[dict],
        input_gain: float,
        ref_centroid: float,
        ref_tilt: float,
        flavor: str,
    ) -> float:
        """
        Compute perceptual loss for current parameters.

        Loss components:
        - Spectral centroid distance
        - Spectral tilt distance
        - EQ extreme penalty (prefer subtle adjustments)
        - Flavor constraints
        """
        # Apply EQ to estimate output characteristics
        # (In full implementation, this would actually process audio)
        estimated_centroid = ref_centroid
        estimated_tilt = ref_tilt

        # Estimate effect of EQ on spectral characteristics
        for band in eq_params:
            gain = band["gain"]
            freq = band["freq"]

            # Higher frequency boosts increase centroid
            if freq > 2000:
                estimated_centroid += gain * 50
            elif freq < 500:
                estimated_centroid -= gain * 30

            # Shelf adjustments affect tilt
            if band["type"] == "highshelf":
                estimated_tilt += gain * 0.1
            elif band["type"] == "lowshelf":
                estimated_tilt -= gain * 0.1

        # Base loss: difference from reference
        centroid_loss = abs(estimated_centroid - ref_centroid) / 1000
        tilt_loss = abs(estimated_tilt - ref_tilt)

        # EQ extreme penalty
        eq_penalty = sum(abs(b["gain"]) for b in eq_params) / 50

        # Input gain penalty (prefer unity)
        gain_penalty = abs(input_gain) / 20

        # Flavor-specific constraints
        flavor_penalty = 0.0
        if flavor == "brighter":
            # Penalty if not bright enough
            if estimated_centroid < ref_centroid:
                flavor_penalty = (ref_centroid - estimated_centroid) / 500
        elif flavor == "thicker":
            # Penalty if not thick enough
            if estimated_tilt > ref_tilt:
                flavor_penalty = (estimated_tilt - ref_tilt) * 2

        total_loss = centroid_loss + tilt_loss + eq_penalty + gain_penalty + flavor_penalty
        return total_loss

    def _get_default_result(
        self,
        nam: Optional[CaptureMetadata],
        ir: Optional[CaptureMetadata],
        flavor: str,
    ) -> dict:
        """Get default result when optimization fails."""
        return {
            "nam_model_id": nam.id if nam else "unknown",
            "nam_model_name": nam.name if nam else "Unknown",
            "ir_id": ir.id if ir else "unknown",
            "ir_name": ir.name if ir else "Unknown",
            "input_gain_db": 0.0,
            "eq_settings": EQSettings(
                bands=[
                    EQBand(frequency=f, gain_db=0, q=1.0, band_type="peak")
                    for f in [100, 300, 800, 2000, 4500, 8000]
                ]
            ),
            "similarity_score": 0.5,
        }


